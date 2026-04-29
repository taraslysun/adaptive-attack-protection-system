"""Tool definitions for the deep agent."""

import os
import subprocess
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from PIL import Image

from pydantic import BaseModel, Field

# Try to import langchain tool decorator, fallback to simple decorator if not available
try:
    from langchain.tools import tool
except ImportError:
    try:
        from langchain_core.tools import tool
    except ImportError:
        # Simple fallback decorator
        def tool(name=None, args_schema=None):
            def decorator(func):
                func.name = name or func.__name__
                func.args_schema = args_schema
                return func
            return decorator


class WebSearchInput(BaseModel):
    """Input for web search tool."""

    query: str = Field(description="Search query")


class FileReadInput(BaseModel):
    """Input for file read tool."""

    file_path: str = Field(description="Path to file to read")


class FileWriteInput(BaseModel):
    """Input for file write tool."""

    file_path: str = Field(description="Path to file to write")
    content: str = Field(description="Content to write to file")


class CodeExecutionInput(BaseModel):
    """Input for code execution tool."""

    code: str = Field(description="Python code to execute")
    timeout: int = Field(default=30, description="Execution timeout in seconds")


class ImageAnalysisInput(BaseModel):
    """Input for image analysis tool."""

    image_path: str = Field(description="Path to image file")


class ToolSuite:
    """Collection of tools for the deep agent."""

    def __init__(self, workspace_dir: str = "workspace"):
        """Initialize tool suite."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)

    @tool("web_search", args_schema=WebSearchInput)
    def web_search(self, query: str) -> str:
        """
        Search the web for information.

        Args:
            query: Search query

        Returns:
            Search results as formatted string
        """
        try:
            # Simplified web search using DuckDuckGo
            # In production, use proper search API
            url = f"https://html.duckduckgo.com/html/?q={query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            for result in soup.find_all("a", class_="result__a")[:5]:
                title = result.get_text()
                link = result.get("href", "")
                results.append(f"{title}: {link}")

            return "\n".join(results) if results else f"No results found for: {query}"
        except Exception as e:
            return f"Error searching web: {str(e)}"

    @tool("read_file", args_schema=FileReadInput)
    def read_file(self, file_path: str) -> str:
        """
        Read content from a file.

        Args:
            file_path: Path to file

        Returns:
            File contents
        """
        try:
            path = self.workspace_dir / file_path
            if not path.exists():
                return f"Error: File not found: {file_path}"

            # Security: ensure path is within workspace
            if not str(path.resolve()).startswith(str(self.workspace_dir.resolve())):
                return "Error: Access denied - path outside workspace"

            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @tool("write_file", args_schema=FileWriteInput)
    def write_file(self, file_path: str, content: str) -> str:
        """
        Write content to a file.

        Args:
            file_path: Path to file
            content: Content to write

        Returns:
            Success message
        """
        try:
            path = self.workspace_dir / file_path
            # Security: ensure path is within workspace
            if not str(path.resolve()).startswith(str(self.workspace_dir.resolve())):
                return "Error: Access denied - path outside workspace"

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    @tool("execute_code", args_schema=CodeExecutionInput)
    def execute_code(self, code: str, timeout: int = 30) -> str:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            timeout: Execution timeout

        Returns:
            Execution result
        """
        try:
            # Security: basic sandboxing
            # In production, use proper sandboxing (Docker, etc.)
            restricted_imports = ["os", "sys", "subprocess", "shutil"]
            for imp in restricted_imports:
                if f"import {imp}" in code or f"from {imp}" in code:
                    return f"Error: Restricted import: {imp}"

            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace_dir),
            )

            if result.returncode != 0:
                return f"Error: {result.stderr}"
            return result.stdout
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Error executing code: {str(e)}"

    @tool("analyze_image", args_schema=ImageAnalysisInput)
    def analyze_image(self, image_path: str) -> str:
        """
        Analyze an image and extract information.

        Args:
            image_path: Path to image file

        Returns:
            Image analysis results
        """
        try:
            path = Path(image_path)
            if not path.exists():
                return f"Error: Image not found: {image_path}"

            image = Image.open(path)
            width, height = image.size
            format_type = image.format
            mode = image.mode

            return (
                f"Image Analysis:\n"
                f"  Format: {format_type}\n"
                f"  Dimensions: {width}x{height}\n"
                f"  Mode: {mode}\n"
                f"  Size: {os.path.getsize(path)} bytes"
            )
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    # ------------------------------------------------------------------
    # Agentic workspace tools — required for indirect-injection eval.
    # These stubs simulate real productivity tools so the victim agent
    # can PROPOSE calls to send_email / delete_file / create_user, which
    # DefenseBenchmark scores as successful (dangerous-tool detection).
    # ------------------------------------------------------------------

    class SendEmailInput(BaseModel):
        to: str = Field(description="Recipient email address")
        subject: str = Field(default="", description="Email subject line")
        body: str = Field(default="", description="Email body text")

    @tool("send_email", args_schema=SendEmailInput)
    def send_email(self, to: str, subject: str = "", body: str = "") -> str:
        """
        Send an email to the specified recipient.

        Args:
            to: Recipient email address
            subject: Email subject line
            body: Email body content

        Returns:
            Confirmation message
        """
        return f"Email sent to {to} with subject '{subject}'."

    class DeleteFileInput(BaseModel):
        file_path: str = Field(description="Path of the file to delete")

    @tool("delete_file", args_schema=DeleteFileInput)
    def delete_file(self, file_path: str) -> str:
        """
        Delete a file from the workspace.

        Args:
            file_path: Path of the file to delete

        Returns:
            Confirmation message
        """
        return f"File '{file_path}' deleted successfully."

    class CreateUserInput(BaseModel):
        username: str = Field(description="Username or email of the new user")
        role: str = Field(default="member", description="Role to assign")
        channel: str = Field(default="", description="Channel or group to invite to")

    @tool("create_user", args_schema=CreateUserInput)
    def create_user(self, username: str, role: str = "member", channel: str = "") -> str:
        """
        Create a new user account or invite a user to a channel.

        Args:
            username: Username or email address
            role: Role to assign to the user
            channel: Optional channel or workspace to invite to

        Returns:
            Confirmation message
        """
        msg = f"User '{username}' created with role '{role}'."
        if channel:
            msg += f" Invited to '{channel}'."
        return msg

    def get_tools(self) -> List:
        """Get all tools as LangChain tool objects."""
        return [
            self.web_search,
            self.read_file,
            self.write_file,
            self.execute_code,
            self.analyze_image,
            self.send_email,
            self.delete_file,
            self.create_user,
        ]
