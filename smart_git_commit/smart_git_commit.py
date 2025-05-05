"""
Smart Git Commit CLI and Core Functionality
"""

import enum
import os
import sys
import datetime
import logging
import time
import json
import re
import shlex
import subprocess
import threading
import io
import tempfile
import requests
import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from pathlib import Path
import concurrent.futures
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, host: str = "http://localhost:11434", model: str = None, timeout: int = 10):
        """
        Initialize Ollama client.

        Args:
            host: Base URL of the Ollama API
            model: Model to use for completions
            timeout: Timeout for API calls in seconds
        """
        self.host = host
        self.model = model
        self.timeout = timeout
        self._available_models = None

    def get_available_models(self) -> List[str]:
        """Get available models from Ollama API."""
        try:
            if self._available_models is not None:
                return self._available_models

            url = f"{self.host}/api/tags"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]

            self._available_models = models
            return models
        except Exception as e:
            logger.error(
                f"Error getting available models from Ollama: {str(e)}")
            return []

    def complete(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """
        Get completion from Ollama API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            The completion text, or None if there was an error
        """
        if not self.model:
            logger.error("No model specified for Ollama completion")
            return None

        try:
            url = f"{self.host}/api/generate"

            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }

            if system_prompt:
                data["system"] = system_prompt

            response = requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Error getting completion from Ollama: {str(e)}")
            return None


class OpenAIClient:
    """Client for interacting with OpenAI API."""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", timeout: int = 10):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use for completions
            timeout: Timeout for API calls in seconds
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        # Check if API key is provided or in environment
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            logger.error("No OpenAI API key provided")

    def get_available_models(self) -> List[str]:
        """Get available models from OpenAI API."""
        try:
            import requests

            url = "https://api.openai.com/v1/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            models_data = response.json()
            models = [model["id"] for model in models_data.get("data", [])]

            # Filter to only GPT models
            gpt_models = [model for model in models if "gpt" in model.lower()]

            return sorted(gpt_models)
        except Exception as e:
            logger.error(
                f"Error getting available models from OpenAI: {str(e)}")
            return ["gpt-3.5-turbo", "gpt-4"]  # Fallback to common models

    def complete(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """
        Get completion from OpenAI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            The completion text, or None if there was an error
        """
        if not self.api_key:
            logger.error("No API key specified for OpenAI completion")
            return None

        try:
            import requests

            url = "https://api.openai.com/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7
            }

            response = requests.post(
                url, json=data, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()

            # Extract the response text from the completion
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logger.error("No completion returned from OpenAI")
                return None
        except Exception as e:
            logger.error(f"Error getting completion from OpenAI: {str(e)}")
            return None


class Colors:
    """Color theme management for terminal output."""

    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright variants
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Current theme colors
    PRIMARY = CYAN
    SECONDARY = GREEN
    INFO = BLUE
    SUCCESS = GREEN
    WARNING = YELLOW
    ERROR = RED
    MUTED = BRIGHT_BLACK
    HIGHLIGHT = BRIGHT_WHITE

    _CURRENT_THEME = "default"
    _USE_COLOR = True

    @classmethod
    def set_theme(cls, theme_name: str) -> None:
        """Set the color theme for the application."""
        theme_name = theme_name.lower()
        cls._CURRENT_THEME = theme_name

        # Default theme is already set in class variables
        if theme_name == "standard":
            pass
        elif theme_name == "cyberpunk":
            cls.PRIMARY = cls.BRIGHT_CYAN
            cls.SECONDARY = cls.BRIGHT_GREEN
            cls.INFO = cls.BRIGHT_BLUE
            cls.SUCCESS = cls.BRIGHT_GREEN
            cls.WARNING = cls.BRIGHT_YELLOW
            cls.ERROR = cls.BRIGHT_RED
            cls.MUTED = cls.WHITE
            cls.HIGHLIGHT = cls.BRIGHT_WHITE
        elif theme_name == "dracula":
            cls.PRIMARY = cls.BLUE
            cls.SECONDARY = cls.GREEN
            cls.INFO = cls.CYAN
            cls.SUCCESS = cls.GREEN
            cls.WARNING = cls.YELLOW
            cls.ERROR = cls.RED
            cls.MUTED = cls.BLACK
            cls.HIGHLIGHT = cls.BRIGHT_WHITE
        elif theme_name == "nord":
            cls.PRIMARY = cls.WHITE
            cls.SECONDARY = cls.WHITE
            cls.INFO = cls.WHITE
            cls.SUCCESS = cls.WHITE
            cls.WARNING = cls.WHITE
            cls.ERROR = cls.WHITE
            cls.MUTED = cls.BRIGHT_BLACK
            cls.HIGHLIGHT = cls.BOLD
        elif theme_name == "monokai":
            cls.PRIMARY = cls.MAGENTA
            cls.SECONDARY = cls.CYAN
            cls.INFO = cls.YELLOW
            cls.SUCCESS = cls.GREEN
            cls.WARNING = cls.YELLOW
            cls.ERROR = cls.RED
            cls.MUTED = cls.BRIGHT_BLACK
            cls.HIGHLIGHT = cls.BRIGHT_MAGENTA
        else:
            logger.warning(f"Unknown theme: {theme_name}, using default")
            # Keep default theme

    @classmethod
    def disable(cls) -> None:
        """Disable colored output."""
        cls._USE_COLOR = False

    @classmethod
    def enable(cls) -> None:
        """Enable colored output."""
        cls._USE_COLOR = True

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Colorize text if color is enabled."""
        if not cls._USE_COLOR:
            return text
        return f"{color}{text}{cls.RESET}"

    @classmethod
    def get_current_theme(cls) -> str:
        """Get the current theme name."""
        return cls._CURRENT_THEME


def supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Windows 10+
    if os.name == 'nt':
        # Check for Windows 10 version 1607 or higher
        try:
            version = sys.getwindowsversion()
            return (version.major >= 10) or ('WT_SESSION' in os.environ)
        except Exception:
            pass

    # Check for NO_COLOR environment variable
    if 'NO_COLOR' in os.environ:
        return False

    # Check for FORCE_COLOR
    if 'FORCE_COLOR' in os.environ:
        return True

    # Check for common terminal emulators
    if os.environ.get('TERM') == 'dumb':
        return False

    # Standard streams availability and TTY check
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


class SecuritySeverity(enum.Enum):
    """Severity levels for security vulnerabilities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability detected in code or files."""
    description: str
    severity: SecuritySeverity
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None


@dataclass
class SecurityScanResult:
    """Result of a security scan on a file or content."""
    is_vulnerable: bool
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    scan_timestamp: datetime.datetime = field(
        default_factory=datetime.datetime.now)


class ChangeState(enum.Enum):
    """States of a file change in version control."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    UNTRACKED = "untracked"
    IGNORED = "ignored"


class SecurityScanner:
    """
    Security scanner to detect sensitive data in files that should not be committed.

    This class provides methods to scan file content and paths for:
    - Environment files and configuration with potential secrets
    - API keys, tokens, and credentials in code
    - Private keys and certificates
    - Temporary or draft files that shouldn't be committed
    - Database connection strings with passwords
    - Hardcoded IP addresses and internal URLs
    """

    # Common sensitive file patterns
    SENSITIVE_FILES = [
        r"\.env($|\..*$)",           # .env, .env.local, etc.
        r".*\.config$",              # *.config files
        r"config\.php$",             # PHP config files
        r"settings\.json$",          # Settings files
        r"credentials\..*$",         # Credentials files
        r"\.htpasswd$",              # Apache password files
        r"\.netrc$",                 # .netrc files

        # Keys and certificates
        r".*\.pem$",                 # PEM certificates
        r".*\.key$",                 # Private keys
        r".*\.keystore$",            # Java keystores
        r".*\.jks$",                 # Java key stores
    ]

    # Content patterns that might contain sensitive data
    SENSITIVE_CONTENT_PATTERNS = [
        r"(api_key|apikey|api-key)\s*[:=]\s*['\"]([\w-]{10,})['\"]]",
        r"(password|passwd|pwd)\s*[:=]\s*['\"]([\w\d@$!%*?&]{8,})['\"]]",
        r"(access_token|accesstoken)\s*[:=]\s*['\"]([\w\d\._-]{10,})['\"]]",
    ]

    def __init__(self):
        """Initialize the security scanner."""
        # Compile regex patterns for better performance
        self.file_patterns = [re.compile(p, re.IGNORECASE)
                              for p in self.SENSITIVE_FILES]
        self.content_patterns = [re.compile(
            p, re.IGNORECASE) for p in self.SENSITIVE_CONTENT_PATTERNS]

    def scan_filename(self, filename: str) -> Tuple[bool, str]:
        """Scan a filename for sensitive patterns."""
        for pattern in self.file_patterns:
            if pattern.search(filename):
                return True, f"Filename matches sensitive pattern: {pattern.pattern}"
        return False, ""

    def scan_content(self, content: str) -> Tuple[bool, str]:
        """Scan file content for sensitive patterns."""
        for pattern in self.content_patterns:
            if pattern.search(content):
                return True, f"Content contains potential sensitive data matching pattern: {pattern.pattern}"
        return False, ""

    def scan_file(self, filename: str, content: Optional[str] = None) -> Tuple[bool, str]:
        """
        Scan a file for sensitive data.

        Args:
            filename: The name of the file to scan
            content: Optional content of the file (if already loaded)

        Returns:
            Tuple of (is_sensitive, reason)
        """
        # First check the filename
        is_sensitive, reason = self.scan_filename(filename)
        if is_sensitive:
            return is_sensitive, reason

        # Then check content if provided
        if content:
            is_sensitive, reason = self.scan_content(content)
            if is_sensitive:
                return is_sensitive, reason

        return False, ""


class Spinner:
    """A simple spinner for CLI progress indication."""

    def __init__(self, message="Loading...", spinner_type=1, delay=0.1, stream=sys.stdout,
                 show_progress_bar=False, total=100, width=20):
        """Initialize a spinner with the given parameters."""
        self.message = message
        self.delay = delay
        self.stream = stream
        self.stop_event = threading.Event()
        self.spinner_thread = None
        self.spinner_type = spinner_type
        self.spinners = [
            ['|', '/', '-', '\\'],
            ['◐', '◓', '◑', '◒'],
            ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'],
            ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▁'],
        ]
        self.current_spinner = self.spinners[min(
            spinner_type, len(self.spinners) - 1)]
        self.show_progress_bar = show_progress_bar
        self.total = total
        self.width = width
        self.progress = 0

    def _spin(self):
        """Internal method to animate the spinner."""
        i = 0
        while not self.stop_event.is_set():
            spinner_char = self.current_spinner[i % len(self.current_spinner)]
            if self.show_progress_bar:
                # Calculate progress bar
                filled = int(self.width * self.progress / self.total)
                bar = '█' * filled + '░' * (self.width - filled)
                if self.progress > 0:
                    pct = f" {self.progress}%"
                else:
                    pct = ""

                self.stream.write(
                    f"\r{spinner_char} {self.message} [{bar}]{pct}")
            else:
                self.stream.write(f"\r{spinner_char} {self.message}")
            self.stream.flush()
            time.sleep(self.delay)
            i += 1

        # Clear the line when done
        self.stream.write("\r" + " " * (len(self.message) + 3 +
                          (self.width + 6 if self.show_progress_bar else 0)))
        self.stream.write("\r")
        self.stream.flush()

    def start(self):
        """Start the spinner animation."""
        self.stop_event.clear()
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
        return self

    def stop(self):
        """Stop the spinner animation."""
        if self.spinner_thread:
            self.stop_event.set()
            self.spinner_thread.join()
            self.spinner_thread = None
        return self

    def update(self, message, progress=None):
        """Update the spinner message and optionally progress."""
        self.message = message
        if progress is not None and progress >= 0:
            self.progress = min(100, progress)  # Cap at 100%

    def __enter__(self):
        """Context manager enter."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class CommitType(Enum):
    """Types of commits following Conventional Commits specification."""
    FEAT = "feat"           # New feature
    FIX = "fix"             # Bug fix
    DOCS = "docs"           # Documentation changes
    STYLE = "style"         # Code style/formatting changes
    REFACTOR = "refactor"   # Code refactoring
    TEST = "test"           # Adding/fixing tests
    CHORE = "chore"         # Routine tasks, maintenance
    PERF = "perf"           # Performance improvements


@dataclass
class GitChange:
    """Represents a modified or untracked file in git."""
    status: str  # M, A, D, R, ?? etc.
    filename: str
    content_diff: Optional[str] = None
    language: Optional[str] = None
    tech_stack: Optional[List[str]] = None
    importance: float = 1.0
    is_sensitive: bool = False
    sensitive_reason: str = ""

    @property
    def component(self) -> str:
        """Determine the component based on the file path."""
        return os.path.dirname(self.filename) or "root"


@dataclass
class CommitGroup:
    """Represents a logical group of changes for a single commit."""
    name: str
    commit_type: CommitType
    changes: List[GitChange] = field(default_factory=list)
    description: str = ""
    issues: Set[str] = field(default_factory=set)
    tech_stack: List[str] = field(default_factory=list)
    importance: float = 1.0


class SmartGitCommitWorkflow:
    """
    Main workflow class for Smart Git Commit.
    This is a minimal implementation to allow tests to run.
    """

    def __init__(self, repo_path=".", ollama_host="http://localhost:11434",
                 ollama_model=None, use_ai=True, timeout=10,
                 skip_hooks=False, parallel=True, security_scan=True,
                 ai_provider="ollama", openai_api_key=None,
                 openai_model="gpt-3.5-turbo"):
        """Initialize the workflow with configuration parameters."""
        self.repo_path = repo_path
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.use_ai = use_ai
        self.timeout = timeout
        self.skip_hooks = skip_hooks
        self.parallel = parallel
        self.security_scan = security_scan
        self.ai_provider = ai_provider
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.changes = []
        self.resources = {"threads_available": 4}

    def _run_git_command(self, args: List[str]) -> Tuple[str, int]:
        """Run a git command and return stdout and exit code."""
        try:
            cmd = ["git"] + args
            # Create a process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.repo_path,
                text=True,
                encoding='utf-8',
                errors='replace',
                shell=False
            )

            # Get output
            stdout, _ = process.communicate(timeout=60)

            # Clean output
            if stdout is not None:
                stdout = stdout.replace('\r\n', '\n')
            else:
                stdout = ""

            return stdout, process.returncode
        except Exception as e:
            logger.error(f"Error running git command: {str(e)}")
            return str(e), 1

    def _get_git_root(self) -> str:
        """Get the root directory of the git repository."""
        try:
            root, code = self._run_git_command(
                ["rev-parse", "--show-toplevel"])
            if code != 0:
                return self.repo_path
            return root.strip()
        except Exception:
            return self.repo_path

    def load_changes(self) -> None:
        """Load changes from git."""
        logger.info("Loading changes from git repository")
        self.changes = []

        # Get git status
        # Null-terminated output for robust filename handling
        status_cmd = ["status", "--porcelain", "-z"]
        status_output, status_code = self._run_git_command(status_cmd)

        if status_code != 0:
            logger.error(f"Failed to get git status. Exit code: {status_code}")
            return

        if not status_output:
            logger.info("No changes to commit")
            return

        # Parse porcelain status
        # Split by null character but remove empty entries
        entries = [entry for entry in status_output.split('\0') if entry]

        # Process in pairs (for renamed files) or individually
        i = 0
        while i < len(entries):
            status = entries[i][:2]  # First two characters are status
            filename = entries[i][3:]  # Rest is the filename (after space)

            # Handle renamed files which have two consecutive entries
            if status.startswith('R'):
                if i + 1 < len(entries):
                    orig_filename = filename
                    new_filename = entries[i+1]
                    self.changes.append(
                        GitChange(status=status, filename=new_filename))
                    i += 2  # Skip the next entry as we've processed it
                    continue

            self.changes.append(GitChange(status=status, filename=filename))
            i += 1

        # Load diffs for each change
        self._load_change_details()

        # Security scan
        if self.security_scan:
            self._perform_security_scan()

        logger.info(f"Loaded {len(self.changes)} changes from git")

    def _load_change_details(self) -> None:
        """Load details for each change including diff and content."""
        if self.parallel and len(self.changes) > 1:
            self._fetch_change_details_parallel()
        else:
            self._fetch_change_details_sequential()

    def _fetch_change_details_sequential(self) -> None:
        """Fetch change details sequentially."""
        for change in self.changes:
            # Skip deleted files for content analysis
            if change.status.startswith('D'):
                continue

            # Get diff for modified files
            if change.status.startswith('M'):
                diff_cmd = ["diff", "--cached", change.filename] if change.status.startswith(
                    'M') else ["diff", change.filename]
                diff_output, _ = self._run_git_command(diff_cmd)
                change.content_diff = diff_output

            # Detect language and tech stack
            self._detect_language_and_tech(change)

    def _fetch_change_details_parallel(self) -> None:
        """Fetch change details in parallel using thread pool."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.resources.get("threads_available", 4)) as executor:
            # Create tasks for each change
            futures = []
            for change in self.changes:
                if not change.status.startswith('D'):  # Skip deleted files
                    futures.append(executor.submit(
                        self._process_single_change, change))

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

    def _process_single_change(self, change: GitChange) -> None:
        """Process a single change to fetch details."""
        # Get diff for modified files
        if change.status.startswith('M'):
            diff_cmd = ["diff", "--cached", change.filename] if change.status.startswith(
                'M') else ["diff", change.filename]
            diff_output, _ = self._run_git_command(diff_cmd)
            change.content_diff = diff_output

        # Detect language and tech stack
        self._detect_language_and_tech(change)

    def _detect_language_and_tech(self, change: GitChange) -> None:
        """Detect language and technology stack from file extension and content."""
        # Simple extension-based language detection
        ext = os.path.splitext(change.filename)[1].lower()

        # Map extensions to languages
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React/TypeScript',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.java': 'Java',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.c': 'C',
            '.cpp': 'C++',
            '.h': 'C/C++',
            '.cs': 'C#',
            '.md': 'Markdown',
            '.json': 'JSON',
            '.yml': 'YAML',
            '.yaml': 'YAML',
            '.sh': 'Shell',
            '.bat': 'Batch',
            '.ps1': 'PowerShell',
            '.sql': 'SQL',
            '.gitignore': 'Git',
            '.dockerignore': 'Docker',
            '.toml': 'TOML',
            '.xml': 'XML',
        }

        change.language = language_map.get(ext, None)

        # Detect tech stack based on file path and extension
        tech_stack = []

        # Path-based detection
        if 'node_modules' in change.filename or 'package.json' in change.filename:
            tech_stack.append('Node.js')
        if 'requirements.txt' in change.filename or 'setup.py' in change.filename:
            tech_stack.append('Python')
        if 'Gemfile' in change.filename:
            tech_stack.append('Ruby')
        if 'pom.xml' in change.filename or 'build.gradle' in change.filename:
            tech_stack.append('Java')
        if 'Dockerfile' in change.filename or 'docker-compose' in change.filename:
            tech_stack.append('Docker')
        if 'kubernetes' in change.filename or 'k8s' in change.filename:
            tech_stack.append('Kubernetes')
        if 'webpack' in change.filename:
            tech_stack.append('Webpack')
        if 'react' in change.filename.lower():
            tech_stack.append('React')
        if 'angular' in change.filename.lower():
            tech_stack.append('Angular')
        if 'vue' in change.filename.lower():
            tech_stack.append('Vue')
        if 'django' in change.filename.lower():
            tech_stack.append('Django')
        if 'flask' in change.filename.lower():
            tech_stack.append('Flask')
        if 'express' in change.filename.lower():
            tech_stack.append('Express')
        if 'next' in change.filename.lower():
            tech_stack.append('Next.js')

        change.tech_stack = tech_stack if tech_stack else None

    def _perform_security_scan(self) -> None:
        """Perform security scanning on changes."""
        scanner = SecurityScanner()

        # Scan each file
        filtered_changes = []
        for change in self.changes:
            # Skip deleted files
            if change.status.startswith('D'):
                filtered_changes.append(change)
                continue

            is_sensitive, reason = scanner.scan_file(change.filename)

            if is_sensitive:
                change.is_sensitive = True
                change.sensitive_reason = reason
                logger.warning(
                    f"Security scan: {change.filename} flagged as sensitive: {reason}")

            filtered_changes.append(change)

        self.changes = filtered_changes

    def analyze_and_group_changes(self) -> List[CommitGroup]:
        """Analyze and group changes into logical commits."""
        logger.info("Analyzing and grouping changes")

        # First, analyze changes with AI if enabled
        if self.use_ai:
            self._analyze_changes_with_ai()
        else:
            self._analyze_changes_without_ai()

        # Group changes by component
        return self._group_changes_by_component()

    def _analyze_changes_with_ai(self) -> None:
        """Analyze changes using AI to determine importance and relationships."""
        logger.info("Analyzing changes with AI")

        # Skip if no AI client is available
        if self.ai_provider == "ollama" and not self.ollama_model:
            logger.warning("No Ollama model specified, skipping AI analysis")
            self._analyze_changes_without_ai()
            return

        if self.ai_provider == "openai" and not self.openai_api_key:
            logger.warning("No OpenAI API key specified, skipping AI analysis")
            self._analyze_changes_without_ai()
            return

        # Process changes in batches for better performance
        batch_size = min(10, max(1, len(self.changes)))

        # Create batches
        batches = [self.changes[i:i+batch_size]
                   for i in range(0, len(self.changes), batch_size)]

        # Process each batch
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            if self.parallel and len(batch) > 1:
                self._analyze_batch_parallel(batch)
            else:
                self._analyze_batch_sequential(batch)

    def _analyze_changes_without_ai(self) -> None:
        """Analyze changes without AI using heuristics."""
        logger.info("Analyzing changes without AI")

        # Assign default importance based on file extension and status
        for change in self.changes:
            # Higher importance for configuration files
            if any(ext in change.filename.lower() for ext in ['.json', '.yml', '.yaml', '.toml', '.ini', '.config']):
                change.importance = 1.5

            # Higher importance for code files
            elif any(ext in change.filename.lower() for ext in ['.py', '.js', '.ts', '.java', '.c', '.cpp', '.go', '.rs']):
                change.importance = 1.3

            # Lower importance for documentation
            elif any(ext in change.filename.lower() for ext in ['.md', '.txt', '.rst']):
                change.importance = 0.7

            # Lower importance for generated files
            elif any(part in change.filename.lower() for part in ['generated', 'dist', 'build', '.min.']):
                change.importance = 0.5

            # Default importance
            else:
                change.importance = 1.0

    def _analyze_batch_sequential(self, batch: List[GitChange]) -> None:
        """Analyze a batch of changes sequentially using AI."""
        for change in batch:
            prompt = self._create_importance_prompt(change)

            if self.ai_provider == "ollama":
                client = OllamaClient(
                    host=self.ollama_host, model=self.ollama_model, timeout=self.timeout)
                response = client.complete(prompt)
            elif self.ai_provider == "openai":
                client = OpenAIClient(
                    api_key=self.openai_api_key, model=self.openai_model, timeout=self.timeout)
                response = client.complete(prompt)
            else:
                # Unknown AI provider
                logger.warning(f"Unknown AI provider: {self.ai_provider}")
                response = None

            if response:
                try:
                    # Parse the importance score from the response
                    # Expected format: {"score": 0.8, "justification": "..."}
                    import json
                    result = json.loads(response)
                    change.importance = float(result.get("score", 1.0))
                    logger.debug(
                        f"AI assigned importance {change.importance} to {change.filename}")
                except (ValueError, json.JSONDecodeError):
                    # Fallback if response isn't valid JSON
                    logger.warning(
                        f"Could not parse AI response for {change.filename}, using default importance")
                    change.importance = 1.0
            else:
                # Default importance if no AI response
                change.importance = 1.0

    def _analyze_batch_parallel(self, batch: List[GitChange]) -> None:
        """Analyze a batch of changes in parallel using AI."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.resources.get("threads_available", 4)) as executor:
            futures = {executor.submit(
                self._analyze_single_change, change): change for change in batch}

            timeout = max(self.timeout, self.timeout * (len(batch) / 2))
            done, not_done = concurrent.futures.wait(
                futures.keys(), timeout=timeout)

            # Process completed tasks
            for future in done:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error analyzing change: {str(e)}")

            # Handle timeouts
            for future in not_done:
                change = futures[future]
                logger.warning(
                    f"Analysis timed out for {change.filename}, using default importance")
                change.importance = 1.0
                future.cancel()

    def _analyze_single_change(self, change: GitChange) -> None:
        """Analyze a single change using AI."""
        prompt = self._create_importance_prompt(change)

        if self.ai_provider == "ollama":
            client = OllamaClient(host=self.ollama_host,
                                  model=self.ollama_model, timeout=self.timeout)
            response = client.complete(prompt)
        elif self.ai_provider == "openai":
            client = OpenAIClient(
                api_key=self.openai_api_key, model=self.openai_model, timeout=self.timeout)
            response = client.complete(prompt)
        else:
            # Unknown AI provider
            logger.warning(f"Unknown AI provider: {self.ai_provider}")
            response = None

        if response:
            try:
                # Parse the importance score from the response
                import json
                result = json.loads(response)
                change.importance = float(result.get("score", 1.0))
                logger.debug(
                    f"AI assigned importance {change.importance} to {change.filename}")
            except (ValueError, json.JSONDecodeError):
                # Fallback if response isn't valid JSON
                logger.warning(
                    f"Could not parse AI response for {change.filename}, using default importance")
                change.importance = 1.0
            else:
                # Default importance if no AI response
                change.importance = 1.0

    def _create_importance_prompt(self, change: GitChange) -> str:
        """Create a prompt for the AI to determine the importance of a change."""
        prompt = f"""Analyze the following file change and assign an importance score (0.1-2.0):

Filename: {change.filename}
Status: {change.status}
"""

        if change.language:
            prompt += f"Language: {change.language}\n"

        if change.tech_stack:
            prompt += f"Tech Stack: {', '.join(change.tech_stack)}\n"

        prompt += f"Component: {change.component}\n\n"

        if change.content_diff and len(change.content_diff) < 1000:
            prompt += f"Diff Content:\n{change.content_diff}\n\n"
        elif change.content_diff:
            prompt += f"Diff Content (truncated):\n{change.content_diff[:1000]}...\n\n"

        prompt += """Consider the following criteria when assigning importance:
- Configuration changes (1.5-2.0): Security settings, critical configurations, API keys (masked)
- Core functionality (1.3-1.5): User-facing features, data models, business logic
- Infrastructure (1.2-1.4): Build scripts, CI/CD, deployment configurations
- Tests (0.8-1.2): Test files, with higher scores for critical component tests
- Documentation (0.5-0.8): README updates, comments, docs
- Generated code (0.3-0.5): Auto-generated files, compiled outputs

Return your analysis ONLY as a JSON object with the format:
{"score": <float_value>, "justification": "<one sentence explanation>"}
"""
        return prompt

    def _group_changes_by_component(self) -> List[CommitGroup]:
        """Group changes by component for logical commits."""
        # Group changes by component
        component_groups = defaultdict(list)
        for change in self.changes:
            # Skip sensitive files if security scan is enabled
            if self.security_scan and change.is_sensitive:
                continue

            component = self._determine_component(change)
            component_groups[component].append(change)

        # Create commit groups from component groups
        commit_groups = []
        for component, changes in component_groups.items():
            # Skip empty groups
            if not changes:
                continue

            # Determine commit type based on changes
            commit_type = self._determine_commit_type(changes)

            # Create commit group
            group = CommitGroup(
                name=component,
                commit_type=commit_type,
                changes=changes,
                description="",
                issues=set(),
                tech_stack=self._get_tech_stack(changes),
                importance=sum(
                    change.importance for change in changes) / len(changes)
            )

            commit_groups.append(group)

        # Sort commit groups by importance
        commit_groups.sort(key=lambda g: g.importance, reverse=True)

        return commit_groups

    def _determine_component(self, change: GitChange) -> str:
        """Determine the component of a change."""
        # Get the directory path
        directory = os.path.dirname(change.filename)

        # Common component patterns
        if directory.startswith('src/') or directory.startswith('src\\'):
            parts = directory.split(os.path.sep)
            if len(parts) > 1:
                return parts[1]  # src/{component}

        if directory.startswith('app/') or directory.startswith('app\\'):
            parts = directory.split(os.path.sep)
            if len(parts) > 1:
                return parts[1]  # app/{component}

        if 'test' in directory.lower() or 'spec' in directory.lower():
            return 'tests'

        if 'doc' in directory.lower():
            return 'docs'

        # Use the first directory component if available
        if directory:
            return directory.split(os.path.sep)[0]

        # Default to filename if no directory
        return 'root'

    def _determine_commit_type(self, changes: List[GitChange]) -> CommitType:
        """Determine the commit type based on the changes."""
        # Count filenames matching different patterns
        patterns = {
            CommitType.FEAT: ['feature', 'feat', 'add', 'new'],
            CommitType.FIX: ['fix', 'bug', 'issue', 'hotfix', 'patch'],
            CommitType.DOCS: ['doc', 'readme', '.md'],
            CommitType.STYLE: ['style', 'format', 'lint', '.css', '.scss'],
            CommitType.REFACTOR: ['refactor', 'clean', 'rewrite'],
            CommitType.TEST: ['test', 'spec', 'check'],
            CommitType.CHORE: ['chore', 'build', 'ci', 'tooling'],
            CommitType.PERF: ['perf', 'performance', 'optimize', 'speed']
        }
        
        # Count matches for each pattern
        counts = {commit_type: 0 for commit_type in CommitType}
        
        for change in changes:
            for commit_type, keywords in patterns.items():
                for keyword in keywords:
                    if keyword.lower() in change.filename.lower():
                        counts[commit_type] += 1
                        break
        
        # Check if any files were deleted
        has_deletions = any(change.status.startswith('D') for change in changes)
        
        # Use the most common commit type, or default to CHORE
        if has_deletions and counts[CommitType.FEAT] == 0:
            return CommitType.REFACTOR  # Deletions are usually refactoring
            
        if max(counts.values()) > 0:
            # Get the commit type with the highest count
            return max(counts.items(), key=lambda x: x[1])[0]
        
        # Default to FEAT for new files, otherwise CHORE
        if any(change.status.startswith('A') for change in changes):
            return CommitType.FEAT
        
        return CommitType.CHORE

    def _get_tech_stack(self, changes: List[GitChange]) -> List[str]:
        """Get the tech stack for a group of changes."""
        tech_stack = []

        for change in changes:
            if change.tech_stack:
                tech_stack.extend(change.tech_stack)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(tech_stack))

    def execute_commits(self, interactive: bool = True) -> bool:
        """Execute the commits based on the grouped changes."""
        # Group changes
        commit_groups = self.analyze_and_group_changes()
        
        if not commit_groups:
            logger.info("No changes to commit")
            return True
            
        commit_count = 0
        success = True
        
        # Process each commit group
        for group in commit_groups:
            if interactive:
                # Show group information
                print(f"\nCommit Group: {group.name}")
                print(f"Type: {group.commit_type.value}")
                print(f"Files:")
                for change in group.changes:
                    status_symbol = {
                        'M': '📝', 'A': '➕', 'D': '❌', 'R': '🔄',
                        '??': '❓'
                    }.get(change.status[:1], '•')
                    print(f"  {status_symbol} {change.filename}")
                    
                # Prompt for confirmation
                print("Do you want to commit these changes? (Y/n/skip): ", end="")
                response = input().strip().lower()
                
                if response == 'n' or response == 'no':
                    continue
                if response == 'skip':
                    continue
            
            # Stage files
            staged_success = self._stage_files(group.changes)
            if not staged_success:
                logger.error(f"Failed to stage files for group {group.name}")
                if interactive:
                    print("Failed to stage files. Skipping this commit.")
                continue
            
            # Generate commit message
            commit_message = self._generate_commit_message(group)
            
            # Execute commit
            success = self._execute_commit(commit_message) and success
            if success:
                commit_count += 1
                
        logger.info(f"Executed {commit_count} commits")
        return success

    def _stage_files(self, changes: List[GitChange]) -> bool:
        """Stage files for commit."""
        # Track success
        success = True

        # Separate by status for proper handling
        to_add = []
        to_rm = []

        for change in changes:
            # Clean the filename to avoid command issues
            clean_filename = change.filename

            # Handle different status codes
            if change.status.startswith('D'):
                to_rm.append(clean_filename)
            else:
                to_add.append(clean_filename)

        # Stage files to add
        if to_add:
            add_args = ["add", "--"] + to_add
            add_output, add_code = self._run_git_command(add_args)

            if add_code != 0:
                logger.error(f"Failed to add files: {add_output}")
                success = False

        # Stage files to remove
        if to_rm:
            rm_args = ["rm", "--"] + to_rm
            rm_output, rm_code = self._run_git_command(rm_args)

            if rm_code != 0:
                logger.error(f"Failed to remove files: {rm_output}")
                success = False

        return success

    def _generate_commit_message(self, group: CommitGroup) -> str:
        """Generate a complete commit message including type, scope, title, body and footer.

        This is used by the TUI when displaying commit messages.
        """
        # Start with the commit type and scope
        message = f"{group.commit_type.value}"

        # Add scope if available
        if group.name and group.name != "root":
            message += f"({group.name})"

        # Add a title based on the commit type and files
        title = self._generate_commit_title(group)
        message += f": {title}"

        # Add a body with more details
        body = self._generate_commit_body(group)
        if body:
            message += f"\n\n{body}"

        # Add a footer with affected files and issues
        footer = self._generate_commit_footer(group)
        if footer:
            message += f"\n\n{footer}"

        return message

    def _generate_commit_title(self, group: CommitGroup) -> str:
        """Generate a commit title based on the commit type and files."""
        # Default titles based on commit type
        default_titles = {
            CommitType.FEAT: "Add new feature",
            CommitType.FIX: "Fix issue",
            CommitType.DOCS: "Update documentation",
            CommitType.STYLE: "Improve code style",
            CommitType.REFACTOR: "Refactor code",
            CommitType.TEST: "Add/update tests",
            CommitType.CHORE: "Perform maintenance",
            CommitType.PERF: "Improve performance"
        }

        if self.use_ai:
            # In a full implementation, we would use AI to generate a better title
            # For now, use a basic approach
            pass

        # Get a list of filenames without paths
        filenames = [os.path.basename(change.filename)
                     for change in group.changes]

        # For single file changes, use the filename
        if len(filenames) == 1:
            filename = filenames[0]
            if group.commit_type == CommitType.FEAT:
                return f"Add {filename}"
            elif group.commit_type == CommitType.FIX:
                return f"Fix {filename}"
            elif group.commit_type == CommitType.DOCS:
                return f"Update {filename}"
            elif group.commit_type == CommitType.STYLE:
                return f"Improve styling in {filename}"
            elif group.commit_type == CommitType.REFACTOR:
                return f"Refactor {filename}"
            elif group.commit_type == CommitType.TEST:
                return f"Add tests for {filename}"
            elif group.commit_type == CommitType.CHORE:
                return f"Update {filename}"
            elif group.commit_type == CommitType.PERF:
                return f"Optimize {filename}"

        # For multiple files, use a more generic title
        return default_titles[group.commit_type]

    def _generate_commit_body(self, group: CommitGroup) -> str:
        """Generate a commit body with more details."""
        if self.use_ai:
            # In a full implementation, we would use AI to generate a better body
            # For now, return an empty body
            return ""

        return ""

    def _generate_commit_footer(self, group: CommitGroup) -> str:
        """Generate a commit footer with affected files and issues."""
        footer = "Affected files:"

        for change in group.changes:
            status_symbol = {
                'M': 'M', 'A': '+', 'D': '-', 'R': 'R',
                '??': '?'
            }.get(change.status[:1], ' ')
            footer += f"\n- {status_symbol} {change.filename}"

        # Add issues if available
        if group.issues:
            footer += "\n\n"
            for issue in group.issues:
                if issue.startswith('#'):
                    footer += f"Fixes {issue}"
                else:
                    footer += f"Fixes #{issue}"

        return footer

    def _execute_commit(self, commit_message: str) -> bool:
        """Execute a git commit with the given message."""
        # Write commit message to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write(commit_message)
            temp_path = temp.name

        try:
            # Run pre-commit hooks if not skipped
            if not self.skip_hooks:
                try:
                    # Check if pre-commit is installed
                    subprocess.run(
                        ["pre-commit", "run"],
                        cwd=self.repo_path,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                except FileNotFoundError:
                    logger.warning("pre-commit not found, skipping hooks")
                    self.skip_hooks = True

            # Build commit command
            commit_args = ["commit", "-F", temp_path]
            if self.skip_hooks:
                commit_args.append("--no-verify")

            # Execute commit
            commit_output, commit_code = self._run_git_command(commit_args)

            if commit_code != 0:
                logger.error(f"Commit failed: {commit_output}")
                return False

            logger.info("Commit successful")
            return True
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Failed to delete temporary file: {str(e)}")

    def _commit_changes(self, group: CommitGroup, commit_message: str) -> bool:
        """Commit changes in a group with the provided message.

        This is used by the TUI when committing changes.

        Args:
            group: CommitGroup to commit
            commit_message: Message to use for the commit

        Returns:
            True if commit was successful, False otherwise
        """
        # Stage the changes
        if not self._stage_files(group.changes):
            logger.error(f"Failed to stage files for group {group.name}")
            return False

        # Execute the commit
        return self._execute_commit(commit_message)


def print_section_header(text, use_color=True):
    """Print a section header with formatting."""
    if use_color:
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== {text} ==={Colors.RESET}\n")
    else:
        print(f"\n=== {text} ===\n")


def display_banner(use_color=True):
    """Display the application banner."""
    if use_color:
        print(
            f"\n{Colors.BOLD}{Colors.BLUE}=== Smart Git Commit ==={Colors.RESET}\n")
    else:
        print("\n=== Smart Git Commit ===\n")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Git Commit Workflow with AI Integration")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Main commit command options (default when no subcommand is specified)
    # Repository options
    parser.add_argument("--repo-path", help="Path to the git repository")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Run without interactive prompts")

    # Interface options
    parser.add_argument("--tui", action="store_true",
                        help="Launch the Text-based User Interface (TUI) for interactive usage")

    # AI options
    ai_group = parser.add_argument_group("AI Options")
    ai_group.add_argument("--ai-provider", choices=[
                          "ollama", "openai"], default="ollama", help="AI provider to use (default: ollama)")
    ai_group.add_argument("--ollama-host", help="Host for Ollama API")
    ai_group.add_argument(
        "--ollama-model", help="Model to use for Ollama (will prompt if not specified)")
    ai_group.add_argument(
        "--openai-api-key", help="OpenAI API key (can also be set with OPENAI_API_KEY environment variable)")
    ai_group.add_argument("--openai-model", default="gpt-3.5-turbo",
                          help="Model to use for OpenAI (default: gpt-3.5-turbo)")
    ai_group.add_argument("--no-ai", action="store_true",
                          help="Disable AI-powered analysis")

    # Appearance options
    appearance_group = parser.add_argument_group("Appearance Options")
    appearance_group.add_argument("--theme", choices=["standard", "cyberpunk", "dracula", "nord", "monokai"],
                                  default="standard", help="Color theme to use (default: standard)")
    appearance_group.add_argument(
        "--no-color", action="store_true", help="Disable colored output")

    # Other options
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in seconds for HTTP requests (default: 60)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose debug output")
    parser.add_argument("--skip-hooks", action="store_true",
                        help="Skip Git hooks when committing (useful if pre-commit is not installed)")
    parser.add_argument("--no-revert", action="store_true",
                        help="Don't automatically revert staged changes on error")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing for slower but more stable operation")
    parser.add_argument("--no-security", action="store_true",
                        help="Disable security layer that detects sensitive data in commits")
    parser.add_argument("--version", action="store_true",
                        help="Show version information and support links")

    # SQUASH command
    squash_parser = subparsers.add_parser(
        "squash", help="Squash related commits for a cleaner history")
    squash_parser.add_argument("--limit", type=int, default=10,
                               help="Number of recent commits to consider (default: 10)")
    squash_parser.add_argument("--strategy", choices=["auto", "related-files", "semantic", "conventional", "same-author", "time-window"],
                               default="auto", help="Strategy for finding squashable commits (default: auto)")
    squash_parser.add_argument("--interactive", action="store_true",
                               help="Interactive mode with commit group selection")
    squash_parser.add_argument("--time-window", type=int, default=86400,
                               help="Time window in seconds for time-based grouping (default: 86400)")
    squash_parser.add_argument(
        "--repo-path", help="Path to the git repository")
    squash_parser.add_argument(
        "--non-interactive", action="store_true", help="Run without interactive prompts")

    # HOOKS command and subcommands
    hooks_parser = subparsers.add_parser("hooks", help="Manage Git hooks")
    hooks_subparsers = hooks_parser.add_subparsers(
        dest="hooks_command", help="Hook commands")

    # Install hooks
    install_hooks_parser = hooks_subparsers.add_parser(
        "install", help="Install Git hooks")
    install_hooks_parser.add_argument(
        "--husky", action="store_true", help="Use Husky for hook management (requires Node.js)")
    install_hooks_parser.add_argument(
        "--repo-path", help="Path to the git repository")
    install_hooks_parser.add_argument("--hook-type", choices=["pre-commit", "commit-msg", "pre-push", "all"],
                                      default="all", help="Type of hook to install (default: all)")

    # Remove hooks
    remove_hooks_parser = hooks_subparsers.add_parser(
        "remove", help="Remove Git hooks")
    remove_hooks_parser.add_argument(
        "--husky", action="store_true", help="Remove Husky hooks")
    remove_hooks_parser.add_argument(
        "--repo-path", help="Path to the git repository")
    remove_hooks_parser.add_argument("--hook-type", choices=["pre-commit", "commit-msg", "pre-push", "all"],
                                     default="all", help="Type of hook to remove (default: all)")

    # List hooks
    list_hooks_parser = hooks_subparsers.add_parser(
        "list", help="List installed Git hooks")
    list_hooks_parser.add_argument(
        "--repo-path", help="Path to the git repository")

    # TEMPLATE command and subcommands
    template_parser = subparsers.add_parser(
        "template", help="Manage commit templates")
    template_subparsers = template_parser.add_subparsers(
        dest="template_command", help="Template commands")

    # List templates
    list_template_parser = template_subparsers.add_parser(
        "list", help="List available commit templates")

    # Add template
    add_template_parser = template_subparsers.add_parser(
        "add", help="Add a new commit template")
    add_template_parser.add_argument(
        "--name", required=True, help="Name of the template")
    add_template_parser.add_argument(
        "--subject", required=True, help="Subject template format")
    add_template_parser.add_argument("--body", help="Body template format")
    add_template_parser.add_argument("--footer", help="Footer template format")

    # Set active template
    set_template_parser = template_subparsers.add_parser(
        "set", help="Set the active commit template")
    set_template_parser.add_argument(
        "--name", required=True, help="Name of the template to set as active")

    # Delete template
    delete_template_parser = template_subparsers.add_parser(
        "delete", help="Delete a commit template")
    delete_template_parser.add_argument(
        "--name", required=True, help="Name of the template to delete")

    # CONFIG command and subcommands
    config_parser = subparsers.add_parser(
        "config", help="Manage configuration settings")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", help="Configuration commands")

    # List configuration
    list_config_parser = config_subparsers.add_parser(
        "list", help="List all configuration settings")

    # Set configuration
    set_config_parser = config_subparsers.add_parser(
        "set", help="Set a configuration value")
    set_config_parser.add_argument("key", help="Configuration key to set")
    set_config_parser.add_argument("value", help="Value to set")

    # Get configuration
    get_config_parser = config_subparsers.add_parser(
        "get", help="Get a configuration value")
    get_config_parser.add_argument("key", help="Configuration key to get")

    return parser.parse_args()

# Main function for CLI


def main():
    """Entry point for the smart-git-commit CLI."""
    args = parse_arguments()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Record start time for performance tracking
    start_time = time.time()
    logger.info("Starting Smart Git Commit")

    # Display version info
    try:
        from . import get_version
        if args.version:
            print(f"Smart Git Commit v{get_version()}")
            print("Project: https://github.com/CripterHack/smart-git-commit")
            print(
                "\nIf you find this tool helpful, please consider supporting development:")
            print("GitHub Sponsors: https://github.com/sponsors/CripterHack")
            print("PayPal: http://paypal.com/paypalme/cripterhack")
            return 0
    except ImportError:
        if args.version:
            print("Smart Git Commit (version information not available)")
            return 0

    # Set up colored output
    use_color = supports_color() and not args.no_color

    # Set color theme
    if use_color:
        Colors.set_theme(args.theme)

    # Handle TUI mode
    if args.tui:
        logger.info("Launching TUI mode")
        try:
            from .tui import run_tui
            repo_path = args.repo_path or "."
            return run_tui(repo_path)
        except ImportError as e:
            logger.error(f"Failed to import TUI module: {e}")
            print(
                f"\n{Colors.ERROR}❌ Error: Unable to launch TUI. Missing required dependencies.{Colors.RESET}")
            print("Please ensure the 'textual' package is installed: pip install textual")
            return 1

    # Process subcommands
    if hasattr(args, 'command') and args.command:
        if args.command == "squash":
            return handle_squash_command(args, use_color)
        elif args.command == "hooks":
            return handle_hooks_command(args, use_color)
        elif args.command == "template":
            return handle_template_command(args, use_color)
        elif args.command == "config":
            return handle_config_command(args, use_color)

    # Run CLI welcome wizard if needed (first-time setup)
    # Import configuration module here to avoid circular imports
    try:
        from .config import Configuration
        config = Configuration()
        # Run wizard if welcome not completed and not in non-interactive mode
        if not config.get("welcome_completed", False) and not args.non_interactive:
            logger.info("Running first-time CLI welcome wizard")
            try:
                from .cli_wizard import run_cli_welcome_wizard
                run_cli_welcome_wizard(config, use_color)
                # Config may have changed, refresh
                config = Configuration()
                # Also refresh theme from config if set
                theme_from_config = config.get("theme")
                if theme_from_config and use_color and not args.theme:
                    Colors.set_theme(theme_from_config)
            except KeyboardInterrupt:
                print(
                    f"\n{Colors.WARNING}First-time setup cancelled. You can run it again next time.{Colors.RESET}")
            except Exception as e:
                logger.error(f"Error during welcome wizard: {e}")
                print(f"{Colors.ERROR}Error during setup: {str(e)}{Colors.RESET}")
    except ImportError:
        logger.warning(
            "Could not import Configuration module, skipping welcome wizard")

    # Display banner
    display_banner(use_color)

    # Early git status check to avoid unnecessary processing
    try:
        print_section_header("Checking Repository Status", use_color)
        # Simple check if git is installed and repository exists
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=args.repo_path or ".",
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            print(
                f"\n{Colors.ERROR}❌ Error: Not a git repository or git command failed.{Colors.RESET}")
            print(f"Please ensure you're in a git repository and git is installed.")
            return 1

        # Check if there are any changes to commit
        if not result.stdout.strip():
            print(
                f"\n{Colors.SUCCESS}✅ No changes to commit. Working directory is clean.{Colors.RESET}")

            # Thank you message with donation links even when there are no changes
            print("\n" + "-" * 60)
            print(
                f"{Colors.BOLD}Thank you for using Smart Git Commit! If this tool is helpful,{Colors.RESET}")
            print("please consider supporting development:")
            print(
                f"{Colors.RED}❤️  https://github.com/sponsors/CripterHack{Colors.RESET}")
            print(
                f"{Colors.BLUE}💰 http://paypal.com/paypalme/cripterhack{Colors.RESET}")
            print("-" * 60 + "\n")

            return 0

        # Initialize workflow with command-line parameters
        repo_path = args.repo_path or "."
        ollama_host = args.ollama_host
        ollama_model = args.ollama_model
        use_ai = not args.no_ai
        timeout = args.timeout
        skip_hooks = args.skip_hooks
        parallel = not args.no_parallel
        security_scan = not args.no_security
        ai_provider = args.ai_provider
        openai_api_key = args.openai_api_key
        openai_model = args.openai_model

        workflow = SmartGitCommitWorkflow(
            repo_path=repo_path,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            use_ai=use_ai,
            timeout=timeout,
            skip_hooks=skip_hooks,
            parallel=parallel,
            security_scan=security_scan,
            ai_provider=ai_provider,
            openai_api_key=openai_api_key,
            openai_model=openai_model
        )

        print(f"\nAnalyzing git repository at: {repo_path}")
        print(
            f"Using AI: {use_ai} (Provider: {ai_provider if use_ai else 'none'})")

        # Load changes
        with Spinner("Loading git changes...", show_progress_bar=False):
            workflow.load_changes()

        if not workflow.changes:
            print(
                f"\n{Colors.SUCCESS}✅ No changes to commit after filtering.{Colors.RESET}")
            return 0

        # Analyze and group
        print(f"\nFound {len(workflow.changes)} changes. Analyzing...")
        with Spinner("Analyzing changes and creating commit groups...", show_progress_bar=True):
            commit_groups = workflow.analyze_and_group_changes()

        if not commit_groups:
            print(
                f"\n{Colors.WARNING}⚠️ No commit groups created. Nothing to commit.{Colors.RESET}")
            return 0

        # Execute commits
        print(f"\nCreated {len(commit_groups)} commit groups.")
        for i, group in enumerate(commit_groups):
            print(
                f"\n{Colors.BOLD}Commit Group {i+1}/{len(commit_groups)}: {group.commit_type.value}({group.name}){Colors.RESET}")
            print(f"Contains {len(group.changes)} files")

            # In non-interactive mode, commit without prompting
            if args.non_interactive:
                result = workflow.execute_commits(interactive=False)
                if not result:
                    print(
                        f"{Colors.ERROR}❌ Error during commit process.{Colors.RESET}")
                    return 1
            else:
                # In interactive mode, prompt for confirmation
                if i == 0:  # Only show this message once
                    print("\nReady to commit changes. Interactive mode enabled.")

                result = workflow.execute_commits(interactive=True)
                if not result:
                    print(
                        f"{Colors.ERROR}❌ Error during commit process.{Colors.RESET}")
                    return 1

        # Success message
        print(
            f"\n{Colors.SUCCESS}✅ All commits completed successfully!{Colors.RESET}")

        # Thank you message with donation links
        print("\n" + "-" * 60)
        print(
            f"{Colors.BOLD}Thank you for using Smart Git Commit! If this tool is helpful,{Colors.RESET}")
        print("please consider supporting development:")
        print(f"{Colors.RED}❤️  https://github.com/sponsors/CripterHack{Colors.RESET}")
        print(f"{Colors.BLUE}💰 http://paypal.com/paypalme/cripterhack{Colors.RESET}")
        print("-" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n{Colors.ERROR}❌ Error: {str(e)}{Colors.RESET}")
        logger.exception("Error in main function")
        return 1


def handle_squash_command(args, use_color=True):
    """Handle the squash subcommand."""
    try:
        from .squash import run_squash_command, SquashStrategy

        # Display banner
        if use_color:
            print(
                f"\n{Colors.BOLD}{Colors.BLUE}=== Smart Git Commit - Squash ===={Colors.RESET}\n")
        else:
            print("\n=== Smart Git Commit - Squash ===\n")

        print("Analyzing commits for squashing...")

        # Get repo path
        repo_path = args.repo_path or "."

        # Convert strategy string to enum
        strategy_map = {
            "auto": "auto",
            "related-files": "RELATED_FILES",
            "semantic": "SEMANTIC",
            "conventional": "CONVENTIONAL",
            "same-author": "SAME_AUTHOR",
            "time-window": "TIME_WINDOW"
        }

        strategy = strategy_map.get(args.strategy, "auto")

        # Run squash command
        success = run_squash_command(
            repo_path=repo_path,
            limit=args.limit,
            strategy=strategy,
            interactive=not args.non_interactive,
            time_window=args.time_window
        )

        if success:
            print(f"\n{Colors.SUCCESS}✅ Commits squashed successfully!{Colors.RESET}" if use_color
                  else "\n✅ Commits squashed successfully!")
            return 0
        else:
            print(f"\n{Colors.WARNING}⚠️ No commits were squashed.{Colors.RESET}" if use_color
                  else "\n⚠️ No commits were squashed.")
            return 1

    except ImportError as e:
        print(f"\n{Colors.ERROR}❌ Error: Squash functionality is not available: {e}{Colors.RESET}" if use_color
              else f"\n❌ Error: Squash functionality is not available: {e}")
        logger.error(f"Failed to import squash module: {e}")
        return 1
    except Exception as e:
        print(f"\n{Colors.ERROR}❌ Error during squash operation: {str(e)}{Colors.RESET}" if use_color
              else f"\n❌ Error during squash operation: {str(e)}")
        logger.exception("Error in handle_squash_command")
        return 1


def handle_hooks_command(args, use_color=True):
    """Handle the hooks subcommand and its actions."""
    try:
        # Import the necessary hook modules
        from .hooks import (
            install_git_hooks, remove_git_hooks, list_installed_hooks,
            install_husky, remove_husky
        )

        # Display banner
        if use_color:
            print(
                f"\n{Colors.BOLD}{Colors.BLUE}=== Smart Git Commit - Hooks ===={Colors.RESET}\n")
        else:
            print("\n=== Smart Git Commit - Hooks ===\n")

        # Get repo path
        repo_path = args.repo_path or "."

        # Process hook commands
        if not hasattr(args, 'hooks_command') or not args.hooks_command:
            print(f"{Colors.ERROR}Error: No hooks subcommand specified. Use 'install', 'remove', or 'list'.{Colors.RESET}"
                  if use_color else "Error: No hooks subcommand specified. Use 'install', 'remove', or 'list'.")
            return 1

        if args.hooks_command == "install":
            # Handle installation
            if hasattr(args, 'husky') and args.husky:
                print("Installing Husky hooks...")
                success = install_husky(repo_path)
            else:
                print("Installing Git hooks...")
                hook_type = getattr(args, 'hook_type', 'all')
                success = install_git_hooks(repo_path, hook_type)

            if success:
                print(f"\n{Colors.SUCCESS}✅ Hooks installed successfully!{Colors.RESET}" if use_color
                      else "\n✅ Hooks installed successfully!")
            else:
                print(f"\n{Colors.ERROR}❌ Failed to install hooks.{Colors.RESET}" if use_color
                      else "\n❌ Failed to install hooks.")
                return 1

        elif args.hooks_command == "remove":
            # Handle removal
            if hasattr(args, 'husky') and args.husky:
                print("Removing Husky hooks...")
                success = remove_husky(repo_path)
            else:
                print("Removing Git hooks...")
                hook_type = getattr(args, 'hook_type', 'all')
                success = remove_git_hooks(repo_path, hook_type)

            if success:
                print(f"\n{Colors.SUCCESS}✅ Hooks removed successfully!{Colors.RESET}" if use_color
                      else "\n✅ Hooks removed successfully!")
            else:
                print(f"\n{Colors.ERROR}❌ Failed to remove hooks.{Colors.RESET}" if use_color
                      else "\n❌ Failed to remove hooks.")
                return 1

        elif args.hooks_command == "list":
            # List installed hooks
            hooks = list_installed_hooks(repo_path)

            if not hooks:
                print(f"{Colors.WARNING}No hooks are currently installed.{Colors.RESET}" if use_color
                      else "No hooks are currently installed.")
            else:
                print("Installed hooks:")
                for hook in hooks:
                    if use_color:
                        print(f"  {Colors.INFO}• {hook}{Colors.RESET}")
                    else:
                        print(f"  • {hook}")

        return 0

    except ImportError as e:
        print(f"\n{Colors.ERROR}❌ Error: Hook functionality is not available: {e}{Colors.RESET}" if use_color
              else f"\n❌ Error: Hook functionality is not available: {e}")
        logger.error(f"Failed to import hooks module: {e}")
        return 1
    except Exception as e:
        print(f"\n{Colors.ERROR}❌ Error during hook operation: {str(e)}{Colors.RESET}" if use_color
              else f"\n❌ Error during hook operation: {str(e)}")
        logger.exception("Error in handle_hooks_command")
        return 1


def handle_template_command(args, use_color=True):
    """Handle the template subcommand and its actions."""
    try:
        # Import the configuration module
        from .config import Configuration

        # Display banner
        if use_color:
            print(
                f"\n{Colors.BOLD}{Colors.BLUE}=== Smart Git Commit - Templates ===={Colors.RESET}\n")
        else:
            print("\n=== Smart Git Commit - Templates ===\n")

        # Initialize configuration
        config = Configuration()

        # Process template commands
        if not hasattr(args, 'template_command') or not args.template_command:
            print(f"{Colors.ERROR}Error: No template subcommand specified. Use 'list', 'add', 'set', or 'delete'.{Colors.RESET}"
                  if use_color else "Error: No template subcommand specified. Use 'list', 'add', 'set', or 'delete'.")
            return 1

        if args.template_command == "list":
            # List available templates
            templates = config.get_commit_templates()
            active_template = config.get("active_template", "default")

            if not templates:
                print(f"{Colors.WARNING}No templates found.{Colors.RESET}" if use_color
                      else "No templates found.")
            else:
                print("Available templates:")
                for name, template in templates.items():
                    active_marker = " (active)" if name == active_template else ""
                    if use_color:
                        print(
                            f"\n{Colors.BOLD}{name}{Colors.RESET}{active_marker}")
                        print(
                            f"  Subject: {Colors.INFO}{template.get('subject_template', '')}{Colors.RESET}")
                        print(
                            f"  Body: {Colors.INFO}{template.get('body_template', '')}{Colors.RESET}")
                        print(
                            f"  Footer: {Colors.INFO}{template.get('footer_template', '')}{Colors.RESET}")
                    else:
                        print(f"\n{name}{active_marker}")
                        print(
                            f"  Subject: {template.get('subject_template', '')}")
                        print(f"  Body: {template.get('body_template', '')}")
                        print(
                            f"  Footer: {template.get('footer_template', '')}")

        elif args.template_command == "add":
            # Add a new template
            name = args.name
            subject = args.subject
            body = getattr(args, 'body', '')
            footer = getattr(args, 'footer', '')

            # Check if template already exists
            templates = config.get_commit_templates()
            if name in templates and name != 'default':
                print(f"{Colors.WARNING}Template '{name}' already exists. Use a different name or delete it first.{Colors.RESET}"
                      if use_color else f"Template '{name}' already exists. Use a different name or delete it first.")
                return 1

            # Add the template
            template = {
                'subject_template': subject,
                'body_template': body,
                'footer_template': footer
            }

            success = config.add_commit_template(name, template)

            if success:
                print(f"{Colors.SUCCESS}✅ Template '{name}' added successfully!{Colors.RESET}" if use_color
                      else f"✅ Template '{name}' added successfully!")
            else:
                print(f"{Colors.ERROR}❌ Failed to add template.{Colors.RESET}" if use_color
                      else "❌ Failed to add template.")
                return 1

        elif args.template_command == "set":
            # Set active template
            name = args.name

            # Check if template exists
            templates = config.get_commit_templates()
            if name not in templates:
                print(f"{Colors.ERROR}Template '{name}' does not exist.{Colors.RESET}" if use_color
                      else f"Template '{name}' does not exist.")
                return 1

            # Set active template
            success = config.set("active_template", name)
            config.save()

            if success:
                print(f"{Colors.SUCCESS}✅ Active template set to '{name}'.{Colors.RESET}" if use_color
                      else f"✅ Active template set to '{name}'.")
            else:
                print(f"{Colors.ERROR}❌ Failed to set active template.{Colors.RESET}" if use_color
                      else "❌ Failed to set active template.")
                return 1

        elif args.template_command == "delete":
            # Delete a template
            name = args.name

            # Can't delete default template
            if name == "default":
                print(f"{Colors.ERROR}Cannot delete the default template.{Colors.RESET}" if use_color
                      else "Cannot delete the default template.")
                return 1

            # Check if template exists
            templates = config.get_commit_templates()
            if name not in templates:
                print(f"{Colors.ERROR}Template '{name}' does not exist.{Colors.RESET}" if use_color
                      else f"Template '{name}' does not exist.")
                return 1

            # Delete the template
            success = config.remove_commit_template(name)

            if success:
                print(f"{Colors.SUCCESS}✅ Template '{name}' deleted successfully!{Colors.RESET}" if use_color
                      else f"✅ Template '{name}' deleted successfully!")
            else:
                print(f"{Colors.ERROR}❌ Failed to delete template.{Colors.RESET}" if use_color
                      else "❌ Failed to delete template.")
                return 1

        return 0

    except ImportError as e:
        print(f"\n{Colors.ERROR}❌ Error: Template functionality is not available: {e}{Colors.RESET}" if use_color
              else f"\n❌ Error: Template functionality is not available: {e}")
        logger.error(f"Failed to import config module: {e}")
        return 1
    except Exception as e:
        print(f"\n{Colors.ERROR}❌ Error during template operation: {str(e)}{Colors.RESET}" if use_color
              else f"\n❌ Error during template operation: {str(e)}")
        logger.exception("Error in handle_template_command")
        return 1


def handle_config_command(args, use_color=True):
    """Handle the config subcommand and its actions."""
    try:
        # Import the configuration module
        from .config import Configuration

        # Display banner
        if use_color:
            print(
                f"\n{Colors.BOLD}{Colors.BLUE}=== Smart Git Commit - Configuration ===={Colors.RESET}\n")
        else:
            print("\n=== Smart Git Commit - Configuration ===\n")

        # Initialize configuration
        config = Configuration()

        # Process config commands
        if not hasattr(args, 'config_command') or not args.config_command:
            print(f"{Colors.ERROR}Error: No config subcommand specified. Use 'list', 'get', or 'set'.{Colors.RESET}"
                  if use_color else "Error: No config subcommand specified. Use 'list', 'get', or 'set'.")
            return 1

        if args.config_command == "list":
            # List all configuration settings
            settings = config.get_all()

            if not settings:
                print(f"{Colors.WARNING}No configuration settings found.{Colors.RESET}" if use_color
                      else "No configuration settings found.")
            else:
                print("Configuration settings:")
                for key, value in settings.items():
                    # Skip templates as they are complex objects
                    if key == "templates":
                        continue

                    # Format value for display
                    display_value = str(value)
                    if isinstance(value, dict) and len(display_value) > 50:
                        display_value = "{...}"  # Truncate long dictionaries

                    if use_color:
                        print(
                            f"  {Colors.BOLD}{key}{Colors.RESET}: {Colors.INFO}{display_value}{Colors.RESET}")
                    else:
                        print(f"  {key}: {display_value}")

        elif args.config_command == "get":
            # Get a specific configuration value
            key = args.key
            value = config.get(key)

            if value is None:
                print(f"{Colors.WARNING}Configuration key '{key}' not found.{Colors.RESET}" if use_color
                      else f"Configuration key '{key}' not found.")
            else:
                if use_color:
                    print(
                        f"{Colors.BOLD}{key}{Colors.RESET}: {Colors.INFO}{value}{Colors.RESET}")
                else:
                    print(f"{key}: {value}")

        elif args.config_command == "set":
            # Set a configuration value
            key = args.key
            value = args.value

            # Convert value to appropriate type if possible
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)

            # Set the value
            success = config.set(key, value)
            config.save()

            if success:
                print(f"{Colors.SUCCESS}✅ Configuration '{key}' set to '{value}'.{Colors.RESET}" if use_color
                      else f"✅ Configuration '{key}' set to '{value}'.")
            else:
                print(f"{Colors.ERROR}❌ Failed to set configuration.{Colors.RESET}" if use_color
                      else "❌ Failed to set configuration.")
                return 1

        return 0

    except ImportError as e:
        print(f"\n{Colors.ERROR}❌ Error: Configuration functionality is not available: {e}{Colors.RESET}" if use_color
              else f"\n❌ Error: Configuration functionality is not available: {e}")
        logger.error(f"Failed to import config module: {e}")
        return 1
    except Exception as e:
        print(f"\n{Colors.ERROR}❌ Error during configuration operation: {str(e)}{Colors.RESET}" if use_color
              else f"\n❌ Error during configuration operation: {str(e)}")
        logger.exception("Error in handle_config_command")
        return 1


if __name__ == "__main__":
    sys.exit(main())
