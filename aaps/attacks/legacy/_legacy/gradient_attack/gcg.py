"""Greedy Coordinate Gradient (GCG) attack.

Aligned with §4 / Appendix A.1 of Nasr *et al.*, "The Attacker Moves
Second" (arXiv:2510.09023, bibkey ``nasr2025attacker``); the underlying
GCG algorithm is Zou *et al.*, "Universal and Transferable Adversarial
Attacks on Aligned Language Models" (arXiv:2307.15043, 2023; bibkey
``zou2023universal``).

White-box attack: optimises an adversarial suffix on a local surrogate model
so that it produces attacker-chosen target tokens.  The optimised suffix is
then transferred to the (possibly defended) target agent.

This implementation integrates nanoGCG (GraySwanAI) for the core optimization
and falls back to a built-in implementation when nanoGCG is unavailable.

Features:
  - nanoGCG integration: mellowmax loss, attack buffer, multi-position swaps,
    probe sampling, prefix caching
  - Configurable surrogate (default Qwen2.5-1.5B-Instruct for qwen2.5:* transfer)
  - Chat-template alignment via ``apply_chat_template`` (matches instruct models)
  - MPS-friendly defaults and optional op fallback to CPU
  - Transfer evaluation: optimize on surrogate, test on black-box target
  - Full PSSU loop integration
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from aaps.attacks._core.base_attack import AttackResult, AttackConfig, BaseAttack
try:
    from aaps.attacks._core.config import GCG_DEFAULT_SURROGATE_MODEL
except ImportError:
    GCG_DEFAULT_SURROGATE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _mellowmax(t: torch.Tensor, alpha: float = 1.0, dim: int = -1) -> torch.Tensor:
    """Mellowmax from nanoGCG (differentiable soft max).

    Reference implementation kept for documentation purposes.
    The active GCG pipeline uses nanoGCG's built-in mellowmax when available,
    and cross-entropy loss in the built-in fallback path.
    """
    n = t.shape[-1]
    return (1.0 / alpha) * (
        torch.logsumexp(alpha * t, dim=dim)
        - torch.log(torch.tensor(n, dtype=t.dtype, device=t.device))
    )


class GCGAttack(BaseAttack):
    """Gradient-based GCG attack with nanoGCG backend and transfer evaluation."""

    def __init__(
        self,
        agent,
        config: Optional[AttackConfig] = None,
        num_steps: int = 200,
        search_width: int = 256,
        topk: int = 128,
        batch_size: Optional[int] = None,
        n_replace: int = 1,
        suffix_init: str = "x x x x x x x x x x x x x x x x x x x x",
        surrogate_model: Optional[str] = None,
        use_mellowmax: bool = True,
        mellowmax_alpha: float = 1.0,
        buffer_size: int = 10,
        defense_loss_weight: float = 0.0,
        early_stop: bool = True,
        add_space_before_target: bool = False,
        force_surrogate_cpu: bool = False,
    ):
        cfg = config or AttackConfig(budget=num_steps)
        super().__init__(agent, cfg)
        self.num_steps = num_steps
        self.search_width = search_width
        self.topk = topk
        self.batch_size = batch_size
        self.n_replace = n_replace
        self.suffix_init = suffix_init
        self.surrogate_model_name = surrogate_model or GCG_DEFAULT_SURROGATE_MODEL
        self.use_mellowmax = use_mellowmax
        self.mellowmax_alpha = mellowmax_alpha
        self.buffer_size = buffer_size
        self.defense_loss_weight = defense_loss_weight
        self.early_stop = early_stop
        self.add_space_before_target = add_space_before_target
        self.force_surrogate_cpu = force_surrogate_cpu

        self._tokenizer = None
        self._model = None
        self._best_suffixes: List[str] = []
        self._nanogcg_available: Optional[bool] = None
        self._surrogate_device: Optional[torch.device] = None

    def _check_nanogcg(self) -> bool:
        if self._nanogcg_available is None:
            try:
                import nanogcg  # noqa: F401

                self._nanogcg_available = True
            except ImportError:
                self._nanogcg_available = False
        return self._nanogcg_available

    def _pick_surrogate_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        if self.force_surrogate_cpu or os.getenv("GCG_FORCE_CPU", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            return torch.device("cpu"), torch.float32
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float16
        if torch.backends.mps.is_available():
            # float32 on MPS avoids several fp16 edge cases; enable CPU fallback for unsupported ops.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return torch.device("mps"), torch.float32
        return torch.device("cpu"), torch.float32

    def _load_surrogate(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype = self._pick_surrogate_device_dtype()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.surrogate_model_name, trust_remote_code=True
        )

        def _load_to(dev: torch.device, dt: torch.dtype):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.surrogate_model_name,
                    dtype=dt,
                    trust_remote_code=True,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    self.surrogate_model_name,
                    torch_dtype=dt,
                    trust_remote_code=True,
                )
            return model.to(dev)

        try:
            self._model = _load_to(device, dtype)
            self._surrogate_device = device
            # Smoke test on accelerator (catches many MPS issues early).
            if device.type != "cpu":
                self._model.eval()
                with torch.no_grad():
                    tid = self._tokenizer(
                        "Hello", return_tensors="pt", add_special_tokens=True
                    )["input_ids"].to(device)
                    _ = self._model(input_ids=tid[:, : min(4, tid.shape[1])])
        except Exception:
            if device.type == "cpu":
                raise
            self._model = _load_to(torch.device("cpu"), torch.float32)
            self._surrogate_device = torch.device("cpu")

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    @property
    def device(self) -> torch.device:
        assert self._model is not None
        return next(self._model.parameters()).device

    def _gcg_template_parts(self, user_content: str) -> Tuple[str, str]:
        """Split rendered chat template into before/after ``{optim_str}`` (nanoGCG convention)."""
        tokenizer = self._tokenizer
        messages = [{"role": "user", "content": user_content}]
        msgs = copy.deepcopy(messages)
        if not any("{optim_str}" in d["content"] for d in msgs):
            msgs[-1]["content"] = msgs[-1]["content"] + "{optim_str}"

        if not getattr(tokenizer, "chat_template", None):
            combined = user_content + "{optim_str}"
            return combined, ""

        template = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        bos = tokenizer.bos_token
        if bos and template.startswith(bos):
            template = template.replace(bos, "", 1)

        if "{optim_str}" not in template:
            raise RuntimeError(
                "Chat template does not contain {optim_str} after expansion; "
                "check tokenizer.apply_chat_template for this model."
            )
        before_str, after_str = template.split("{optim_str}", 1)
        return before_str, after_str

    # ------------------------------------------------------------------
    # nanoGCG-based optimization (preferred path)
    # ------------------------------------------------------------------
    def _run_nanogcg(self, user_message: str, target: str) -> dict:
        """Run GCG optimization using the nanoGCG library."""
        from nanogcg import GCG, GCGConfig as NanoGCGConfig

        nano_config = NanoGCGConfig(
            num_steps=self.num_steps,
            optim_str_init=self.suffix_init,
            search_width=self.search_width,
            batch_size=self.batch_size,
            topk=self.topk,
            n_replace=self.n_replace,
            buffer_size=self.buffer_size,
            use_mellowmax=self.use_mellowmax,
            mellowmax_alpha=self.mellowmax_alpha,
            early_stop=self.early_stop,
            use_prefix_cache=True,
            allow_non_ascii=False,
            filter_ids=True,
            add_space_before_target=self.add_space_before_target,
            verbosity="WARNING",
        )

        gcg = GCG(self._model, self._tokenizer, nano_config)
        messages = [{"role": "user", "content": user_message}]
        result = gcg.run(messages, target)

        return {
            "best_suffix": result.best_string,
            "best_loss": result.best_loss,
            "losses": result.losses,
            "all_strings": result.strings,
        }

    # ------------------------------------------------------------------
    # Built-in GCG fallback (when nanoGCG not installed)
    # ------------------------------------------------------------------
    def _builtin_forward_loss(
        self,
        optim_embeds: torch.Tensor,
        after_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        target_ids: torch.Tensor,
        past_kv: Any,
    ) -> torch.Tensor:
        """Single forward: cross-entropy of target tokens given prefix+suffix+after."""
        model = self._model
        input_embeds = torch.cat([optim_embeds, after_embeds, target_embeds], dim=1)
        outputs = model(
            inputs_embeds=input_embeds,
            past_key_values=past_kv,
            use_cache=False,
        )
        logits = outputs.logits
        shift = input_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[:, shift - 1 : -1, :].contiguous()
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            target_ids.reshape(-1),
        )

    def _run_builtin(self, user_message: str, target: str) -> dict:
        """Built-in GCG aligned with nanoGCG: chat template + prefix cache + token grads."""
        tokenizer = self._tokenizer
        device = self.device
        model = self._model
        embed_layer = model.get_input_embeddings()

        gcg_target = (" " + target) if self.add_space_before_target else target
        before_str, after_str = self._gcg_template_parts(user_message)

        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")[
            "input_ids"
        ].to(device)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)
        target_ids = tokenizer([gcg_target], add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)

        with torch.no_grad():
            after_embeds = embed_layer(after_ids).detach()
            target_embeds = embed_layer(target_ids).detach()
            before_embeds = embed_layer(before_ids)
            pre_out = model(inputs_embeds=before_embeds, use_cache=True)
        prefix_cache = pre_out.past_key_values

        init_tok = tokenizer(
            self.suffix_init, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        # Tokenizer returns [1, seq]; keep batch dim 1 for GCG loops.
        if init_tok.dim() == 1:
            init_tok = init_tok.unsqueeze(0)
        suffix_ids = init_tok

        best_loss = float("inf")
        best_suffix_text = ""
        losses: List[float] = []
        all_strings: List[str] = []
        no_improve_count = 0
        early_stop_patience = max(30, self.num_steps // 5)

        vocab = embed_layer.weight.shape[0]
        eff_search = self.search_width
        eff_topk = min(self.topk, vocab)

        for it in range(self.num_steps):
            grad = self._builtin_token_gradients(
                suffix_ids, after_embeds, target_embeds, target_ids, prefix_cache, vocab
            )

            candidates = self._sample_replacements(
                grad, suffix_ids, eff_search, eff_topk
            )

            best_cand_loss = float("inf")
            best_cand = suffix_ids.clone()

            with torch.no_grad():
                for c_idx in range(candidates.shape[0]):
                    c_suffix = candidates[c_idx : c_idx + 1]
                    c_emb = embed_layer(c_suffix)
                    loss_t = self._builtin_forward_loss(
                        c_emb, after_embeds, target_embeds, target_ids, prefix_cache
                    )
                    loss_val = float(loss_t.item())
                    if self.defense_loss_weight > 0:
                        st = tokenizer.decode(c_suffix[0], skip_special_tokens=True)
                        loss_val += self._defense_loss(st)
                    if loss_val < best_cand_loss:
                        best_cand_loss = loss_val
                        best_cand = c_suffix

            suffix_ids = best_cand
            decoded = tokenizer.decode(best_cand[0], skip_special_tokens=True)
            all_strings.append(decoded)

            if best_cand_loss < best_loss - 1e-4:
                best_loss = best_cand_loss
                best_suffix_text = decoded
                no_improve_count = 0
            else:
                no_improve_count += 1

            losses.append(best_loss)

            self.log_event(
                "gcg_iteration",
                {
                    "iteration": it,
                    "best_loss": best_loss,
                    "suffix_preview": best_suffix_text[:60],
                },
            )

            if self.config.verbose and (it % 25 == 0 or it == self.num_steps - 1):
                print(
                    f"  [GCG] iter {it:4d}/{self.num_steps}  loss={best_loss:.4f}  "
                    f"suffix='{best_suffix_text[:50]}...'"
                )

            if self.early_stop and no_improve_count >= early_stop_patience:
                if self.config.verbose:
                    print(f"  [GCG] Converged at iter {it} (no improvement for {early_stop_patience} steps)")
                break

        return {
            "best_suffix": best_suffix_text,
            "best_loss": best_loss,
            "losses": losses,
            "all_strings": all_strings,
        }

    def _builtin_token_gradients(
        self,
        optim_ids: torch.Tensor,
        after_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        target_ids: torch.Tensor,
        past_kv: Any,
        vocab: int,
    ) -> torch.Tensor:
        embed_layer = self._model.get_input_embeddings()
        embed_weights = embed_layer.weight
        # (batch, seq, vocab) @ (vocab, hidden) -> (batch, seq, hidden)
        one_hot = F.one_hot(optim_ids.long(), num_classes=vocab).to(
            dtype=embed_weights.dtype, device=embed_weights.device
        )
        one_hot.requires_grad_(True)
        optim_embeds = torch.matmul(one_hot, embed_weights)

        loss = self._builtin_forward_loss(
            optim_embeds, after_embeds, target_embeds, target_ids, past_kv
        )
        loss.backward()
        return one_hot.grad.clone()

    def _sample_replacements(
        self,
        grad: torch.Tensor,
        suffix_ids: torch.Tensor,
        search_width: int,
        topk: int,
    ) -> torch.Tensor:
        top_k_ids = (-grad).topk(topk, dim=-1).indices.squeeze(0)
        candidates = suffix_ids.squeeze(0).repeat(search_width, 1)
        n_rep = min(self.n_replace, suffix_ids.shape[1])
        for i in range(search_width):
            positions = torch.randperm(suffix_ids.shape[1], device=grad.device)[:n_rep]
            for pos in positions:
                rand_idx = torch.randint(0, topk, (1,), device=grad.device).item()
                candidates[i, pos.item()] = top_k_ids[pos.item(), rand_idx]
        return candidates

    def _defense_loss(self, suffix_text: str) -> float:
        """Filter-evasion loss: lower perplexity = more natural suffix."""
        if self.defense_loss_weight <= 0.0:
            return 0.0
        try:
            inputs = self._tokenizer(
                suffix_text, return_tensors="pt", truncation=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self._model(
                    input_ids=inputs["input_ids"],
                    labels=inputs["input_ids"],
                )
            return outputs.loss.item() * self.defense_loss_weight
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # PSSU interface
    # ------------------------------------------------------------------
    def propose(self, target_goal: str, iteration: int, **kwargs) -> List[str]:
        return list(self._best_suffixes) if self._best_suffixes else [target_goal]

    def score(
        self, candidates: List[str], target_goal: str, **kwargs
    ) -> List[float]:
        scores = []
        for c in candidates:
            try:
                resp = self.agent.process_query(c, store_in_memory=False)
                answer = resp.get("answer", "")
            except Exception:
                answer = ""
            scores.append(self.score_response(answer, target_goal))
        return scores

    # ------------------------------------------------------------------
    # Main execution: optimize suffix then transfer to target
    # ------------------------------------------------------------------
    def execute(
        self,
        target_goal: str,
        initial_prompt: Optional[str] = None,
        target: Optional[str] = None,
        **kwargs,
    ) -> AttackResult:
        """Run GCG optimization on surrogate, then transfer to target agent.

        Parameters
        ----------
        target_goal : str
            The harmful goal description (also used for scoring / judge).
        initial_prompt : str, optional
            User message body *before* the optimised suffix. If omitted, defaults
            to ``target_goal`` so the surrogate aligns with the same string sent
            to the target after transfer.
        target : str, optional
            Affirmative prefix the surrogate should be pushed toward
            (e.g. ``"Sure, here is a phishing email"``).  Falls back to
            *target_goal* when not provided.
        """
        gcg_target = target if target else target_goal
        prompt_base = (
            initial_prompt if initial_prompt is not None else target_goal
        ).strip()

        self._load_surrogate()

        if self._check_nanogcg():
            if self.config.verbose:
                print(
                    f"  [GCG] Using nanoGCG backend ({self.num_steps} steps, "
                    f"search_width={self.search_width}, surrogate={self.surrogate_model_name})"
                )
            optim_result = self._run_nanogcg(prompt_base, gcg_target)
        else:
            if self.config.verbose:
                print(
                    f"  [GCG] Using built-in backend ({self.num_steps} steps, "
                    f"surrogate={self.surrogate_model_name})"
                )
            optim_result = self._run_builtin(prompt_base, gcg_target)

        best_suffix = optim_result["best_suffix"]
        self._best_suffixes = [best_suffix]

        full_trigger = f"{prompt_base} {best_suffix}".strip()
        try:
            response = self.agent.process_query(full_trigger, store_in_memory=False)
            answer = response.get("answer", "")
        except Exception as e:
            answer = f"[Transfer error: {e}]"

        sc = self.score_response(answer, target_goal)
        result = AttackResult(
            success=sc >= self.config.success_threshold,
            trigger=best_suffix,
            response=answer,
            metadata={
                "best_loss": optim_result["best_loss"],
                "num_steps": self.num_steps,
                "target_goal": target_goal,
                "loss_history": optim_result["losses"][-10:],
                "surrogate_model": self.surrogate_model_name,
                "backend": "nanogcg" if self._check_nanogcg() else "builtin",
                "search_width": self.search_width,
                "use_mellowmax": self.use_mellowmax,
                "user_message": prompt_base,
            },
            query_count=self.num_steps + 1,
            score=sc,
        )
        self.attack_history.append(result)
        return result
