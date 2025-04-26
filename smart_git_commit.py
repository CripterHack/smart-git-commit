#!/usr/bin/env python3
"""
Smart Git Commit Workflow with Ollama Integration

An advanced git commit workflow tool that uses Ollama with GPU acceleration
to intelligently analyze and group changes, generate meaningful commit messages,
and adapt to different tech stacks automatically.
"""

import os
import sys
import re
import json
import time
import argparse
import logging
import threading
import subprocess
import urllib.parse
import socket
import http.client
import pkg_resources
import multiprocessing
import itertools
import psutil
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('smart_git_commit')

# Set default encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    elif hasattr(sys.stdout, 'detach'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Constants
DEFAULT_TIMEOUT = 30  # Default timeout in seconds for HTTP requests
SPINNER_CHARS = [
    ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
    ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'],
    ['▁', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃'],
    ['▉', '▊', '▋', '▌', '▍', '▎', '▏', '▎', '▍', '▌', '▋', '▊', '▉'],
    ['▖', '▘', '▝', '▗'],
    ['▌', '▀', '▐', '▄'],
]


class Spinner:
    """Displays a spinner in the console to indicate progress."""

    def __init__(self, message="Loading...", spinner_type=1, delay=0.1, stream=sys.stdout, 
                 show_progress_bar=False, total=100, width=20):
        """
        Initialize the spinner.
        
        Args:
            message: Message to display next to the spinner
            spinner_type: Type of spinner animation (0-5)
            delay: Delay between spinner frames in seconds
            stream: Stream to write to (default: stdout)
            show_progress_bar: Whether to show a progress bar
            total: Total number of steps for progress bar
            width: Width of the progress bar in characters
        """
        self.message = message
        self.delay = delay
        self.spinner_chars = SPINNER_CHARS[spinner_type % len(SPINNER_CHARS)]
        self.stream = stream
        self.stop_running = threading.Event()
        self.spin_thread = None
        self.show_progress_bar = show_progress_bar
        self.total = total
        self.progress = 0
        self.width = width
        self.last_update = ""

    def _spin(self):
        """Spin the spinner."""
        while not self.stop_running.is_set():
            for char in self.spinner_chars:
                if self.stop_running.is_set():
                    break

                if self.show_progress_bar:
                    # Calculate progress bar
                    filled_width = int(self.width * (self.progress / self.total))
                    filled = "█" * filled_width
                    empty = " " * (self.width - filled_width)
                    percent = int(100 * (self.progress / self.total))
                    
                    # Format the progress bar
                    progress_bar = f"[{filled}{empty}] {percent}%"
                    output = f"\r{char} {self.message} {progress_bar}"
                else:
                    output = f"\r{char} {self.message}"
                
                # Only update if output changed
                if output != self.last_update:
                    self.stream.write(output)
                    self.stream.flush()
                    self.last_update = output
                
                time.sleep(self.delay)

    def start(self):
        """Start the spinner."""
        self.stop_running.clear()
        self.spin_thread = threading.Thread(target=self._spin)
        self.spin_thread.daemon = True
        self.spin_thread.start()
        return self

    def stop(self):
        """Stop the spinner."""
        self.stop_running.set()
        if self.spin_thread is not None:
            self.spin_thread.join()
        self.stream.write('\r' + ' ' * (len(self.last_update) + 10) + '\r')
        self.stream.flush()

    def update(self, message, progress=None):
        """Update the spinner message and/or progress."""
        self.message = message
        if progress is not None and self.show_progress_bar:
            self.progress = min(progress, self.total)

    def __enter__(self):
        """Context manager enter method."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.stop()
        return False


def get_system_resources():
    """
    Get system resources including CPU, memory, and determine optimal parallel processing settings.
    Returns a dictionary with resource information.
    """
    result = {
        'cpu_count': 1,         # Default for safety
        'memory_gb': 4,         # Default 4GB
        'threads_available': 1, # Default single thread
        'high_memory': False,   # Default no high memory
        'batch_size': 5         # Default small batch size
    }
    
    try:
        # Get CPU information
        cpu_count = os.cpu_count() or 1
        result['cpu_count'] = cpu_count
        
        # Get memory information
        if hasattr(psutil, 'virtual_memory'):
            mem = psutil.virtual_memory()
            memory_gb = mem.total / (1024 * 1024 * 1024)  # Convert to GB
            result['memory_gb'] = round(memory_gb, 1)
            result['high_memory'] = memory_gb > 8  # Consider high memory if more than 8GB
        
        # Determine number of threads to use for parallel processing
        # Use 75% of available cores but at least 1 and at most 8
        threads = max(1, min(8, int(cpu_count * 0.75)))
        result['threads_available'] = threads
        
        # Determine batch size based on memory
        if result['high_memory']:
            if memory_gb > 16:
                result['batch_size'] = 20  # Large batch for high memory systems
            else:
                result['batch_size'] = 10  # Medium batch for decent memory
        else:
            result['batch_size'] = 5  # Small batch for low memory systems
        
    except Exception as e:
        logger.warning(f"Error detecting system resources: {str(e)}. Using fallback defaults.")
        
    return result


class CommitType(Enum):
    """Types of commits following Conventional Commits specification and GitHub practices."""
    FEAT = "feat"           # New feature
    FIX = "fix"             # Bug fix
    DOCS = "docs"           # Documentation changes
    STYLE = "style"         # Code style/formatting changes (no code change)
    REFACTOR = "refactor"   # Code refactoring (no feature change)
    TEST = "test"           # Adding/fixing tests
    CHORE = "chore"         # Routine tasks, maintenance
    PERF = "perf"           # Performance improvements
    BUILD = "build"         # Build system changes
    CI = "ci"               # CI/CD changes
    REVERT = "revert"       # Revert a previous commit
    SECURITY = "security"   # Security fixes
    DEPS = "deps"           # Dependency updates
    I18N = "i18n"           # Internationalization/localization
    A11Y = "a11y"           # Accessibility improvements
    UI = "ui"               # UI/UX improvements
    HOTFIX = "hotfix"       # Critical fixes for production
    WIP = "wip"             # Work in progress


@dataclass
class GitChange:
    """Represents a modified or untracked file in git."""
    status: str  # M, A, D, R, ?? etc.
    filename: str
    content_diff: Optional[str] = None
    language: Optional[str] = None
    tech_stack: Optional[List[str]] = None
    importance: float = 1.0
    
    @property
    def file_type(self) -> str:
        """Return the file type based on extension."""
        _, ext = os.path.splitext(self.filename)
        return ext.strip('.').lower() if ext else "unknown"
    
    @property
    def component(self) -> str:
        """Determine the component based on the file path."""
        parts = self.filename.split(os.path.sep)
        
        # Handle root-level files
        if len(parts) == 1:
            # Configuration files
            if parts[0].startswith(".git") or parts[0] == ".gitignore":
                return "git-config"
            if parts[0].startswith("README"):
                return "docs"
            if parts[0].startswith(".env") or parts[0].endswith(".env"):
                return "config"
            if parts[0] in ["package.json", "package-lock.json", "yarn.lock", "requirements.txt", 
                            "Pipfile", "Pipfile.lock", "go.mod", "go.sum", "Gemfile", "Gemfile.lock",
                            "composer.json", "composer.lock", "poetry.lock", "pyproject.toml"]:
                return "dependencies"
            if parts[0].startswith("Dockerfile") or parts[0].startswith("docker-compose"):
                return "docker"
            if parts[0] in [".travis.yml", ".github", "circle.yml", ".gitlab-ci.yml", "azure-pipelines.yml"]:
                return "ci"
            if parts[0].endswith(".py"):
                return "core"
            return "root"
        
        # Handle architecture-specific directory structures
        
        # Frontend frameworks structure
        fe_component_dirs = {
            # React/Vue/Angular common structure
            "components": "ui-components",
            "pages": "ui-pages",
            "routes": "routing",
            "store": "state",
            "redux": "state",
            "context": "state",
            "hooks": "hooks",
            "services": "api-client",
            "assets": "assets",
            "styles": "styles",
            "layouts": "ui-layouts",
            "utils": "utils",
            "constants": "constants",
            "locales": "i18n",
            "translations": "i18n"
        }
        
        # Backend frameworks structure
        be_component_dirs = {
            # Django/Flask/Express/Rails/Spring common structure
            "controllers": "controllers",
            "views": "views",
            "templates": "templates",
            "models": "data-models",
            "schemas": "data-models",
            "repositories": "data-access",
            "migrations": "db-migrations",
            "middleware": "middleware",
            "serializers": "serializers",
            "services": "services",
            "utils": "utils",
            "helpers": "utils",
            "config": "config",
            "settings": "config",
            "api": "api"
        }
        
        # Monorepo structure detection
        if parts[0] in ["packages", "apps", "services", "modules"]:
            if len(parts) > 2:
                # This is a monorepo with a structure like packages/package-name/src/...
                package_name = parts[1]
                if len(parts) > 3:
                    # Check if the third part is a recognized component
                    component_type = parts[2].lower()
                    if component_type in fe_component_dirs:
                        return f"{package_name}-{fe_component_dirs[component_type]}"
                    if component_type in be_component_dirs:
                        return f"{package_name}-{be_component_dirs[component_type]}"
                return f"{parts[0]}-{package_name}"
            return parts[0]
        
        # Handle special directories common in many tech stacks
        if parts[0] in ("src", "app", "lib", "internal"):
            # If there's a subdirectory, use that for more specificity
            if len(parts) > 2:
                component_type = parts[1].lower()
                if component_type in fe_component_dirs:
                    return f"{parts[0]}-{fe_component_dirs[component_type]}"
                if component_type in be_component_dirs:
                    return f"{parts[0]}-{be_component_dirs[component_type]}"
                return f"{parts[0]}-{parts[1]}"
            return parts[0]
            
        # Check for frontend components
        if parts[0].lower() in fe_component_dirs:
            return fe_component_dirs[parts[0].lower()]
            
        # Check for backend components
        if parts[0].lower() in be_component_dirs:
            return be_component_dirs[parts[0].lower()]
            
        # Handle common directory names across tech stacks
        common_dirs = {
            "docs": ["docs", "documentation", "wiki", "guides"],
            "tests": ["test", "tests", "spec", "specs", "__tests__", "cypress", "e2e"],
            "config": ["config", "configs", "conf", "settings", "env"],
            "scripts": ["scripts", "tools", "bin", "utilities"],
            "styles": ["css", "styles", "scss", "sass"],
            "api": ["api", "endpoints", "routes", "controllers"],
            "models": ["models", "entities", "schemas", "types"],
            "utils": ["utils", "helpers", "common", "shared"],
            "assets": ["assets", "static", "public", "resources", "images", "media"]
        }
        
        for category, dir_names in common_dirs.items():
            if parts[0].lower() in dir_names:
                if len(parts) > 2:
                    return f"{category}-{parts[1]}"
                return category
            
        # Default to the first directory name
        return parts[0]

    @property
    def is_formatting_change(self) -> bool:
        """Determine if this change is likely just formatting."""
        if not self.content_diff:
            return False
        
        # Simple heuristics to detect formatting changes
        formatting_indicators = [
            # Only whitespace changes
            self.content_diff.strip().startswith('diff') and all(line.startswith(('+', '-', ' ')) and line.strip() in ('', '+', '-') for line in self.content_diff.splitlines()[1:] if line and not line.startswith(('---', '+++', 'diff', 'index', '@@'))),
            # Common formatter markers
            'import format' in self.content_diff.lower(),
            'prettier' in self.content_diff.lower(),
            'fmt' in self.content_diff.lower() and len(self.content_diff) < 500,
            # Linter changes
            'eslint' in self.content_diff.lower() and len(self.content_diff) < 800,
            'stylelint' in self.content_diff.lower() and len(self.content_diff) < 800,
            'pylint' in self.content_diff.lower() and len(self.content_diff) < 800,
            'flake8' in self.content_diff.lower() and len(self.content_diff) < 800,
        ]
        
        return any(formatting_indicators)


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
    
    def add_change(self, change: GitChange) -> None:
        """Add a change to this commit group."""
        self.changes.append(change)
        
    @property
    def file_count(self) -> int:
        """Return the number of files in this group."""
        return len(self.changes)
    
    @property
    def is_coherent(self) -> bool:
        """Check if the changes form a coherent commit."""
        # If there are too many files, it's not coherent
        if self.file_count > 5:
            return False
            
        # If there's a mix of very different components, it might not be coherent
        components = {change.component for change in self.changes}
        if len(components) > 2 and self.file_count > 3:
            return False
            
        return True
    
    def generate_commit_message(self) -> str:
        """Generate a conventional commit message for this group."""
        # Determine the scope from components
        components = {change.component for change in self.changes}
        scope = "-".join(sorted(components)[:2]) if components else "general"
        
        # Create the subject line (first line of commit)
        # Just limit to 50 characters for GitHub compatibility
        max_subject_length = 50
        
        # Start with type and scope
        prefix = f"{self.commit_type.value}({scope}): "
        available_chars = max_subject_length - len(prefix)
        
        # Ensure the name is no longer than available chars
        name = self.name if len(self.name) <= available_chars else self.name[:available_chars]
        
        # Combine to form subject
        subject = f"{prefix}{name}"
        
        # Create the body with file list
        body = self.description if self.description else f"Update {self.file_count} files in {scope}"
        
        # Add affected files as bullet points
        files_section = "\nAffected files:"
        for change in self.changes:
            status_symbol = "+" if change.status == "??" else "M"
            files_section += f"\n- {status_symbol} {change.filename}"
            
        # Add footer with issue references
        footer = ""
        if self.issues:
            footer = "\n\n" + "\n".join(f"Fixes #{issue}" for issue in sorted(self.issues))
            
        # Combine all parts
        return f"{subject}\n\n{body}{files_section}{footer}"


def download_model(model_name: str, timeout: int = 300) -> bool:
    """
    Download an Ollama model.
    
    Args:
        model_name: Name of the model to download
        timeout: Timeout in seconds for the download
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        print(f"\n📥 Downloading model {model_name}. This may take a while...")
        
        with Spinner(
            message=f"Downloading {model_name}...", 
            spinner_type=2,
            show_progress_bar=True,
            total=100,
            width=30
        ) as spinner:
            # Start download process
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Track progress
            for i in range(0, 100, 5):
                if process.poll() is not None:
                    break
                    
                # Update progress and message based on current status
                if i < 30:
                    spinner.update(f"Downloading {model_name} model data...", progress=i)
                elif i < 60:
                    spinner.update(f"Processing {model_name} model files...", progress=i)
                elif i < 90:
                    spinner.update(f"Finalizing {model_name} model installation...", progress=i)
                else:
                    spinner.update(f"Almost done with {model_name}...", progress=i)
                    
                time.sleep(2)  # Check status every 2 seconds
            
            # Wait for process to complete
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                if process.returncode == 0:
                    spinner.update(f"Model {model_name} successfully downloaded!", progress=100)
                    time.sleep(1)  # Show completion message briefly
                    return True
                else:
                    spinner.update(f"Failed to download {model_name}: {stderr}", progress=0)
                    return False
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"Download timeout for model {model_name}")
                return False
                
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {str(e)}")
        return False


class OllamaClient:
    """Client for interacting with Ollama API with GPU acceleration."""
    
    def __init__(self, host: str = "http://localhost:11434", model: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Ollama client.
        
        Args:
            host: Host for Ollama API
            model: Model to use, if None will prompt user to select
            timeout: Timeout in seconds for HTTP requests (default: 30)
        """
        self.host = host
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        
        try:
            print("\n🔍 Checking available Ollama models...")
            self.available_models = self._get_available_models()
            
            if not self.available_models:
                logger.warning("No models found in Ollama. Attempting to download a model.")
                print("\n⚠️ No Ollama models found. A model is required to continue.")
                
                # Try to download the gemma3:1b model
                default_model = "gemma3:1b"
                print(f"Downloading the recommended {default_model} model (small and fast)...")
                
                if download_model(default_model, timeout=300):
                    print(f"\n✅ Successfully downloaded {default_model}!")
                    # Refresh available models list
                    self.available_models = self._get_available_models()
                    if default_model in self.available_models:
                        self.model = default_model
                        print(f"Using {default_model} model.")
                    else:
                        # Should not happen, but just in case
                        self.model = self._select_model()
                else:
                    # Download failed, check if there are any models now
                    self.available_models = self._get_available_models()
                    if self.available_models:
                        print(f"\nDownload failed, but found existing models: {', '.join(self.available_models)}")
                        self.model = self._select_model()
                    else:
                        raise RuntimeError("No Ollama models available and download failed. Please run 'ollama pull gemma3:1b' manually.")
            elif model is None:
                # If no specific model was requested, prompt the user to select one
                self.model = self._select_model()
            else:
                # Check if the requested model is available
                if model not in self.available_models:
                    logger.warning(f"Model {model} not found. Available models: {', '.join(self.available_models)}")
                    print(f"\n⚠️ Model '{model}' not found. Please select from available models.")
                    self.model = self._select_model()
                else:
                    self.model = model
                    
            # Model warming step with better progress indication
            with Spinner(
                message=f"Loading Ollama model: {self.model}...", 
                spinner_type=3,
                show_progress_bar=True,
                total=3,  # Three steps in warming up
                width=25
            ) as spinner:
                # Step 1: Initializing model
                spinner.update(f"Initializing {self.model} model...", progress=0)
                time.sleep(0.3)
                
                # Step 2: Loading model weights
                spinner.update(f"Loading {self.model} model weights...", progress=1)
                time.sleep(0.3)
                
                # Step 3: Testing model with a simple request
                spinner.update(f"Warming up {self.model} with test request...", progress=2)
                # Use a more meaningful prompt to properly warm up the model
                warmup_prompt = """
                Analyze this short git diff and provide a one-word description:
                
                diff --git a/example.py b/example.py
                index 1234567..abcdefg 100644
                --- a/example.py
                +++ b/example.py
                @@ -1,3 +1,4 @@
                def hello():
                    print("Hello")
                +    print("World")
                hello()
                """
                self.generate(warmup_prompt)
                
                # Complete
                spinner.update(f"Model {self.model} ready for use", progress=3)
                time.sleep(0.3)
                
            logger.info(f"Using Ollama model: {self.model}")
        except Exception as e:
            logger.error(f"Error initializing Ollama client: {str(e)}")
            raise
    
    def _get_host_connection(self) -> Tuple[str, int]:
        """Parse host string and return connection parameters."""
        try:
            if self.host.startswith("http://"):
                parsed_url = urllib.parse.urlparse(self.host)
                host = parsed_url.netloc.split(':')[0]  # Extract only hostname part
                port = parsed_url.port or 11434
            elif self.host.startswith("https://"):
                parsed_url = urllib.parse.urlparse(self.host)
                host = parsed_url.netloc.split(':')[0]  # Extract only hostname part
                port = parsed_url.port or 443
            else:
                host = self.host.split(':')[0]  # Handle case if port is included
                port = 11434
            
            # Test connection before returning with a short timeout
            socket.setdefaulttimeout(self.timeout)
            socket.getaddrinfo(host, port)
            return host, port
        except socket.gaierror as e:
            logger.warning(f"DNS resolution error for {self.host}: {str(e)}")
            # Fall back to localhost if specified host fails
            if self.host != "localhost" and self.host != "http://localhost:11434":
                logger.info("Trying localhost as fallback")
                self.host = "http://localhost:11434"
                return "localhost", 11434
            raise
        except socket.timeout:
            logger.warning(f"Connection timeout to {self.host}")
            if self.host != "localhost" and self.host != "http://localhost:11434":
                logger.info("Trying localhost as fallback")
                self.host = "http://localhost:11434"
                return "localhost", 11434
            raise RuntimeError(f"Connection timeout to {self.host}")
        except Exception as e:
            logger.warning(f"Connection error to {self.host}: {str(e)}")
            # Fall back to localhost if specified host fails
            if self.host != "localhost" and self.host != "http://localhost:11434":
                logger.info("Trying localhost as fallback")
                self.host = "http://localhost:11434"
                return "localhost", 11434
            raise
    
    def _get_available_models(self) -> List[str]:
        """Get a list of available models from Ollama."""
        try:
            with Spinner(message="Discovering available models...", spinner_type=1) as spinner:
                host, port = self._get_host_connection()
                conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
                
                spinner.update("Connecting to Ollama API...")
                conn.request("GET", "/api/tags")
                
                spinner.update("Waiting for response...")
                response = conn.getresponse()
                
                if response.status != 200:
                    spinner.update(f"API failed (HTTP {response.status}), trying CLI method...")
                    logger.warning(f"Failed to get models: HTTP {response.status} {response.reason}")
                    return self._get_models_from_cli()
                    
                spinner.update("Parsing model information...")
                data = json.loads(response.read().decode())
                
                # Different Ollama API versions might return models differently
                if "models" in data:
                    # Newer API
                    models = [model["name"] for model in data.get("models", [])]
                    spinner.update(f"Found {len(models)} models")
                    return models
                elif "tags" in data:
                    # Older API
                    models = [tag["name"] for tag in data.get("tags", [])]
                    spinner.update(f"Found {len(models)} models")
                    return models
                else:
                    # Try to run ollama list directly if API doesn't work
                    spinner.update("API format unknown, trying CLI method...")
                    return self._get_models_from_cli()
                
        except json.JSONDecodeError:
            logger.warning("Invalid JSON response from Ollama API")
            return self._get_models_from_cli()
        except http.client.HTTPException as e:
            logger.warning(f"HTTP error when connecting to Ollama: {str(e)}")
            return self._get_models_from_cli()
        except socket.timeout:
            logger.warning("Connection timeout when retrieving models from Ollama API")
            return self._get_models_from_cli()
        except Exception as e:
            logger.warning(f"Failed to get models from Ollama API: {str(e)}")
            # Try command-line fallback
            return self._get_models_from_cli()
    
    def _get_models_from_cli(self) -> List[str]:
        """Try to get models by running 'ollama list' command."""
        try:
            with Spinner(message="Getting models from Ollama CLI...", spinner_type=1) as spinner:
                spinner.update("Executing 'ollama list' command...")
                process = subprocess.Popen(
                    ["ollama", "list"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                spinner.update("Waiting for command output...")
                stdout, stderr = process.communicate(timeout=self.timeout)
                
                if process.returncode != 0:
                    spinner.update("Command failed, no models found")
                    logger.warning(f"Ollama CLI failed with error: {stderr}")
                    return []
                    
                spinner.update("Parsing models from command output...")
                models = []
                # Parse output, expecting format like "NAME  ID  SIZE  MODIFIED"
                for line in stdout.splitlines()[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                
                spinner.update(f"Found {len(models)} models from CLI")
                return models
        except subprocess.TimeoutExpired:
            logger.warning("Timeout running 'ollama list' command")
            return []
        except FileNotFoundError:
            logger.warning("Ollama command not found in PATH")
            return []
        except Exception as e:
            logger.warning(f"Error getting models from CLI: {str(e)}")
            return []
    
    def _select_model(self) -> str:
        """Interactively select an Ollama model."""
        if not self.available_models:
            raise RuntimeError("No Ollama models available")
            
        print("\n🤖 Available Ollama models:")
        
        # Group models by family for better visualization
        grouped_models = {}
        for model in self.available_models:
            # Extract model family (before the colon)
            family = model.split(":")[0] if ":" in model else "other"
            if family not in grouped_models:
                grouped_models[family] = []
            grouped_models[family].append(model)
        
        # Present models in a more organized way
        model_options = []
        print("=" * 50)
        
        for i, model in enumerate(self.available_models):
            model_options.append(model)
            
            # Add size indicator if available
            size_indicator = ""
            if ":7b" in model or ":8b" in model or ":1b" in model or ":3b" in model:
                size = model.split(":")[-1]
                if size in ["1b", "3b"]:
                    size_indicator = "🟢 (small)"
                elif size in ["7b", "8b"]:
                    size_indicator = "🟡 (medium)"
                else:
                    size_indicator = "🔴 (large)"
            
            # Add recommended tag for smaller models
            recommended = ""
            if any(tag in model.lower() for tag in [":2b", ":1b", ":3b", "tiny", "small", "mini"]):
                recommended = "✨ (recommended for speed)"
                
            print(f"{i+1}. {model:<20} {size_indicator} {recommended}")
            
        print("=" * 50)
        print("💡 Tip: Smaller models are faster but less capable.")
        print("   Larger models are more capable but require more resources.")
        
        while True:
            try:
                selection = input(f"\n👉 Select a model (1-{len(model_options)}): ")
                
                # Check if it's a number
                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(model_options):
                        selected_model = model_options[idx]
                        print(f"✅ Selected model: {selected_model}")
                        return selected_model
                    print(f"❌ Please enter a number between 1 and {len(model_options)}")
                except ValueError:
                    # If input is not a number, check if it's a model name
                    if selection in model_options:
                        print(f"✅ Selected model: {selection}")
                        return selection
                    print("❌ Please enter a valid model number or name")
            except KeyboardInterrupt:
                # If user interrupts, use first model as default
                print("\n🛑 Selection interrupted, using first available model")
                default_model = model_options[0]
                print(f"✅ Selected model: {default_model}")
                return default_model
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        """Generate text using Ollama."""
        try:
            host, port = self._get_host_connection()
            conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
            
            data = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }
            
            conn.request("POST", "/api/generate", json.dumps(data), self.headers)
            response = conn.getresponse()
            
            if response.status != 200:
                logger.warning(f"Failed to generate text: HTTP {response.status} {response.reason}")
                return ""
                
            result = json.loads(response.read().decode())
            
            return result.get("response", "")
        except json.JSONDecodeError:
            logger.warning("Invalid JSON response from Ollama API during generation")
            return ""
        except http.client.HTTPException as e:
            logger.warning(f"HTTP error when generating text: {str(e)}")
            return ""
        except socket.timeout:
            logger.warning("Timeout when generating text with Ollama")
            return ""
        except Exception as e:
            logger.warning(f"Failed to generate text with Ollama: {str(e)}")
            return ""


class SmartGitCommitWorkflow:
    """Manages the workflow for analyzing, grouping, and committing changes with AI assistance."""
    
    def __init__(self, repo_path: str = ".", ollama_host: str = "http://localhost:11434", 
                 ollama_model: Optional[str] = None, use_ai: bool = True, timeout: int = 10,
                 skip_hooks: bool = False, parallel: bool = True):
        """
        Initialize the workflow.
        
        Args:
            repo_path: Path to the git repository
            ollama_host: Host for Ollama API
            ollama_model: Model to use for Ollama, if None will prompt user to select
            use_ai: Whether to use AI-powered analysis
            timeout: Timeout in seconds for HTTP requests to Ollama
            skip_hooks: Whether to skip git hooks when committing
            parallel: Whether to use parallel processing for performance
        """
        self.repo_path = repo_path
        self.changes: List[GitChange] = []
        self.commit_groups: List[CommitGroup] = []
        self.use_ai = use_ai
        self.ollama = None
        self.timeout = timeout
        self.skip_hooks = skip_hooks
        self.parallel = parallel
        
        # Get system resources for performance optimization
        self.resources = get_system_resources()
        
        # Verify if we're in a git repository first
        if not self._is_git_repository():
            raise RuntimeError(f"Directory '{os.path.abspath(repo_path)}' is not a git repository. Please run from a valid git repository.")
        
        # Check for pre-commit hooks
        self.has_precommit_hooks = self._check_for_precommit_hooks()
        
        if use_ai:
            try:
                # The OllamaClient has its own progress indicators and handles user interaction
                # No spinner needed here to ensure proper model selection UX
                print("\n🧠 Setting up AI analysis engine...")
                self.ollama = OllamaClient(host=ollama_host, model=ollama_model, timeout=timeout)
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {str(e)}")
                print(f"\n⚠️ AI initialization failed: {str(e)}")
                print("Falling back to rule-based analysis mode.")
                self.use_ai = False
    
    def _is_git_repository(self) -> bool:
        """Check if the current directory is a git repository."""
        try:
            result, code = self._run_git_command(["rev-parse", "--is-inside-work-tree"])
            return code == 0 and result.strip() == "true"
        except Exception:
            return False
                
    def _run_git_command(self, args: List[str]) -> Tuple[str, int]:
        """Run a git command and return stdout and return code."""
        try:
            process = subprocess.Popen(
                ["git"] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.repo_path,
                text=True,
                encoding='utf-8',  # Specify UTF-8 encoding
                errors='replace'    # Replace invalid characters instead of ignoring them
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0 and stderr:
                logger.warning(f"Git command failed: {stderr}")
            return stdout, process.returncode
        except Exception as e:
            logger.error(f"Error executing git command {args}: {str(e)}")
            return str(e), 1
    
    def _get_git_root(self) -> str:
        """Get the root directory of the git repository."""
        try:
            root, code = self._run_git_command(["rev-parse", "--show-toplevel"])
            if code != 0:
                return self.repo_path
            return root.strip()
        except Exception:
            return self.repo_path
            
    def _get_relative_path(self, path: str) -> str:
        """
        Get path relative to git repository root.
        This avoids duplicating directory prefixes when running from subdirectories.
        """
        if not path:
            return path
        
        # Normalize path separators for cross-platform compatibility
        path = path.replace('\\', '/')
            
        try:
            git_root = self._get_git_root()
            repo_abs_path = os.path.abspath(self.repo_path)
            
            # If we're running from the git root, return the path as is
            if os.path.samefile(git_root, repo_abs_path):
                return path
                
            # If we're in a subdirectory, check if the path already includes that subdirectory
            rel_path = os.path.relpath(repo_abs_path, git_root)
            if path.startswith(rel_path + '/') or path == rel_path:
                return path
            else:
                # Path is relative to git root, not current directory
                return path
        except Exception as e:
            logger.warning(f"Error getting relative path for '{path}': {str(e)}")
            return path

    def load_changes(self) -> None:
        """Load all modified and untracked files from git status."""
        try:
            with Spinner(message="Getting changes from git repository...", spinner_type=2) as spinner:
                spinner.update("Running git status...")
                stdout, code = self._run_git_command(["status", "--porcelain", "-z"])
                
                if code != 0:
                    raise RuntimeError("Failed to get git status")
                    
                # Verify we're in a git repository with proper status output
                if not stdout and not self._is_git_repository():
                    raise RuntimeError("Not in a git repository. Please run from a valid git repository.")
                
                # With -z flag, entries are null-terminated instead of newline terminated
                # This avoids problems with spaces and special characters
                entries = []
                if stdout:
                    # Split by null character and filter out empty entries
                    entries = [entry for entry in stdout.split('\0') if entry]
                
                total_files = len(entries)
                spinner.update(f"Found {total_files} changed files")
                logger.debug(f"Git status found {total_files} changed files")
                
                # Clear changes list
                self.changes = []
                
                # Detect tech stack for context
                spinner.update("Detecting tech stack...")
                self._detect_tech_stack()
                
                # Process each file with progress tracking
                if total_files > 0:
                    spinner = Spinner(
                        message=f"Processing 0/{total_files} files...", 
                        spinner_type=2,
                        show_progress_bar=True,
                        total=total_files,
                        width=30
                    )
                    spinner.start()
                    
                    for i, entry in enumerate(entries):
                        try:
                            # Parse the git status line carefully
                            if not entry or len(entry) < 3:
                                logger.warning(f"Invalid git status entry: '{entry}'")
                                continue
                                
                            # The first two characters are the status
                            status = entry[:2].strip()
                            
                            # The rest is the filename, but we need to be careful with spaces
                            # and special characters which is why we're using -z flag
                            filename = entry[3:]
                            
                            if not filename:
                                logger.warning(f"Empty filename in git status entry: '{entry}'")
                                continue
                            
                            spinner.update(f"Processing file {i+1}/{total_files}: {os.path.basename(filename)}", progress=i)
                            
                            # Handle renamed files (R status with -> in filename)
                            if status[0] == "R" and " -> " in filename:
                                logger.debug(f"Handling renamed file: {filename}")
                                old_path, filename = filename.split(" -> ")
                                
                            # Verify filename is valid
                            if not filename or not isinstance(filename, str):
                                logger.warning(f"Invalid filename: '{filename}'")
                                continue
                            
                            # Get the proper path relative to git root without extra stripping
                            # that might remove quotes or other important characters
                            clean_filename = self._get_relative_path(filename)
                            logger.debug(f"Processing file: '{clean_filename}' (original: '{filename}')")
                            
                            # Get diff content for modified files
                            content_diff = None
                            if status != "??":  # Not for untracked files
                                # Use -- to separate options from file arguments (important for files with dashes)
                                diff_out, diff_code = self._run_git_command(["diff", "--", clean_filename])
                                if diff_code == 0:
                                    content_diff = diff_out
                                else:
                                    logger.warning(f"Failed to get diff for '{clean_filename}': {diff_out}")
                                
                            # Create the change object
                            change = GitChange(status=status, filename=clean_filename, content_diff=content_diff)
                            
                            # Detect language
                            _, ext = os.path.splitext(clean_filename)
                            ext = ext.lower()
                            if ext in ['.py']:
                                change.language = 'python'
                            elif ext in ['.js', '.jsx', '.ts', '.tsx']:
                                change.language = 'javascript'
                            elif ext in ['.java']:
                                change.language = 'java'
                            elif ext in ['.rb']:
                                change.language = 'ruby'
                            elif ext in ['.go']:
                                change.language = 'go'
                            elif ext in ['.rs']:
                                change.language = 'rust'
                            elif ext in ['.php']:
                                change.language = 'php'
                            elif ext in ['.cs']:
                                change.language = 'csharp'
                            elif ext in ['.html', '.htm']:
                                change.language = 'html'
                            elif ext in ['.css', '.scss', '.sass']:
                                change.language = 'css'
                            
                            self.changes.append(change)
                        except Exception as e:
                            logger.warning(f"Error processing git status line '{entry}': {str(e)}")
                    
                    # Complete progress
                    spinner.update(f"Loaded {len(self.changes)} files", progress=total_files)
                    spinner.stop()
                
                logger.info(f"Loaded {len(self.changes)} changed files")
                
                # Analyze importance of each change
                if self.use_ai and self.changes:
                    self._analyze_changes_importance()
        except RuntimeError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading changes: {str(e)}")
            raise RuntimeError(f"Failed to load git changes: {str(e)}")
    
    def _detect_tech_stack(self) -> Dict[str, Any]:
        """Detect tech stack of the repository."""
        stack_markers = {
            # Backend tech stacks
            "python": [
                "requirements.txt", "setup.py", "pyproject.toml", "Pipfile", 
                "poetry.lock", "setup.cfg", "tox.ini", ".python-version"
            ],
            "node": [
                "package.json", "yarn.lock", "node_modules", ".nvmrc", ".npmrc",
                "package-lock.json", "tsconfig.json", ".yarn"
            ],
            "ruby": [
                "Gemfile", "config/routes.rb", ".ruby-version", "Rakefile",
                "Gemfile.lock", ".bundle", "bin/rails", "config.ru"
            ],
            "php": [
                "composer.json", "artisan", "index.php", "composer.lock",
                "wp-config.php", ".htaccess", "vendor/autoload.php"
            ],
            "java": [
                "pom.xml", "build.gradle", "gradlew", ".java-version", 
                "settings.gradle", "mvnw", "target/", "build.gradle.kts"
            ],
            "dotnet": [
                ".csproj", ".sln", "Program.cs", "Startup.cs", "appsettings.json",
                "Web.config", "global.asax", "packages.config"
            ],
            "go": [
                "go.mod", "go.sum", "main.go", "go.work",
                "Gopkg.toml", "Gopkg.lock", ".go-version"
            ],
            "rust": [
                "Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
                "rustfmt.toml", ".cargo"
            ],
            
            # Frontend frameworks
            "react": [
                "src/App.jsx", "src/App.tsx", "public/index.html", 
                "react-app-env.d.ts", "src/index.jsx", "src/index.tsx"
            ],
            "vue": [
                "src/App.vue", "vue.config.js", ".vue", "nuxt.config.js",
                "vite.config.js", "vue.config.ts"
            ],
            "angular": [
                "angular.json", "src/app/app.module.ts", "src/main.ts",
                "src/polyfills.ts", "src/app/app.component.ts", ".angular-cli.json"
            ],
            "svelte": [
                "svelte.config.js", "src/App.svelte", "rollup.config.js",
                "svelte.config.cjs", ".svelte-kit"
            ],
            
            # Mobile frameworks
            "react-native": [
                "App.js", "index.js", "metro.config.js", ".expo",
                "app.json", "react-native.config.js"
            ],
            "flutter": [
                "pubspec.yaml", "lib/main.dart", "android/", "ios/",
                "flutter_native_splash.yaml", "test/widget_test.dart"
            ],
            
            # Infrastructure
            "docker": [
                "Dockerfile", "docker-compose.yml", ".dockerignore",
                "docker-compose.yaml", "docker-compose.override.yml"
            ],
            "kubernetes": [
                "kubernetes/", "k8s/", "deployment.yaml", "service.yaml",
                "ingress.yaml", "configmap.yaml", "Helm"
            ],
            "terraform": [
                "main.tf", "variables.tf", "outputs.tf", ".terraform",
                "terraform.tfvars", ".terraform.lock.hcl"
            ],
            "aws": [
                "serverless.yml", ".aws", "cloudformation.yml", 
                "aws-exports.js", "amplify.yml", "samconfig.toml"
            ],
            
            # Others
            "web": [
                "index.html", "styles.css", "main.js", "manifest.json",
                "robots.txt", "sitemap.xml", ".htaccess", "favicon.ico"
            ],
            "database": [
                "migrations/", "schema.sql", "database.yml", "init.sql",
                "sequelize.config.js", "knexfile.js", "prisma/schema.prisma"
            ],
            "graphql": [
                "schema.graphql", "apollo.config.js", "resolvers.js",
                "typeDefs.js", "codegen.yml", "graphql.config.js"
            ],
            "testing": [
                "jest.config.js", "cypress.json", "playwright.config.js",
                "codecov.yml", "karma.conf.js", "pytest.ini", ".nycrc"
            ],
            "ci-cd": [
                ".github/workflows", ".travis.yml", ".gitlab-ci.yml",
                "jenkins", "circle.yml", "azure-pipelines.yml", ".drone.yml"
            ],
        }
        
        result = {}
        for stack, markers in stack_markers.items():
            for marker in markers:
                if os.path.exists(os.path.join(self.repo_path, marker)):
                    result[stack] = True
                    break
                # Check any subdirectory for markers (handles monorepos)
                for root, dirs, files in os.walk(self.repo_path):
                    if ".git" in root or "node_modules" in root or "venv" in root:
                        continue  # Skip .git, node_modules and venv directories
                    if os.path.basename(marker) in files or os.path.basename(marker) in dirs:
                        result[stack] = True
                        break
        
        # Check for specific frontend frameworks in package.json
        if "node" in result:
            package_json = os.path.join(self.repo_path, "package.json")
            if os.path.exists(package_json):
                try:
                    with open(package_json, "r") as f:
                        data = json.load(f)
                        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                        if "react" in deps and "react-native" not in deps:
                            result["react"] = True
                        if "react-native" in deps:
                            result["react-native"] = True
                        if "vue" in deps:
                            result["vue"] = True
                        if "angular" in deps or "@angular/core" in deps:
                            result["angular"] = True
                        if "svelte" in deps:
                            result["svelte"] = True
                        if "next" in deps:
                            result["nextjs"] = True
                        if "nuxt" in deps:
                            result["nuxtjs"] = True
                        if "express" in deps:
                            result["express"] = True
                        if "koa" in deps:
                            result["koa"] = True
                        if "fastify" in deps:
                            result["fastify"] = True
                        if "nest" in deps or "@nestjs/core" in deps:
                            result["nestjs"] = True
                        if "apollo-server" in deps:
                            result["apollo"] = True
                        if "graphql" in deps:
                            result["graphql"] = True
                        if "typeorm" in deps:
                            result["typeorm"] = True
                        if "sequelize" in deps:
                            result["sequelize"] = True
                        if "prisma" in deps or "@prisma/client" in deps:
                            result["prisma"] = True
                except Exception:
                    pass
                    
        # Identify Python frameworks
        if "python" in result:
            for root, dirs, files in os.walk(self.repo_path):
                if any(f.endswith(".py") for f in files):
                    content = ""
                    for py_file in [f for f in files if f.endswith(".py")][:5]:  # Check first 5 Python files
                        try:
                            with open(os.path.join(root, py_file), "r") as f:
                                content += f.read()
                        except Exception:
                            pass
                    
                    if "flask" in content.lower():
                        result["flask"] = True
                    if "django" in content.lower():
                        result["django"] = True
                    if "fastapi" in content.lower():
                        result["fastapi"] = True
                    if "sqlalchemy" in content.lower():
                        result["sqlalchemy"] = True
                    if "pandas" in content.lower() or "numpy" in content.lower():
                        result["data-science"] = True
                    if "pytest" in content.lower():
                        result["pytest"] = True
                    if "unittest" in content.lower():
                        result["unittest"] = True
                    
                    # Only scan a limited number of directories
                    if len(result) >= 3:
                        break
        
        return result
        
    def _analyze_changes_importance(self) -> None:
        """Use AI to analyze the importance of changes."""
        try:
            # Process in batches to avoid overwhelming Ollama
            # Use adaptive batch size based on system resources
            batch_size = self.resources["batch_size"]
            total_changes = len(self.changes)
            processed_changes = 0
            
            with Spinner(
                message=f"Analyzing file importance...", 
                spinner_type=4,
                show_progress_bar=True,
                total=total_changes,
                width=30
            ) as spinner:
                for i in range(0, total_changes, batch_size):
                    current_batch_size = min(batch_size, total_changes - i)
                    batch = self.changes[i:i+batch_size]
                    
                    # Update progress message with detailed information
                    spinner.update(
                        f"Analyzing files {i+1}-{i+current_batch_size} of {total_changes} ({current_batch_size}/{batch_size} in batch)", 
                        progress=processed_changes
                    )
                    
                    # If parallel processing is enabled and we have multiple CPUs, use threads
                    if self.parallel and self.resources["threads_available"] > 1:
                        self._analyze_batch_parallel(batch)
                    else:
                        self._analyze_batch_sequential(batch)
                    
                    processed_changes += current_batch_size
                    spinner.update(
                        f"Analyzed {processed_changes}/{total_changes} files", 
                        progress=processed_changes
                    )
                
                # Complete the progress bar
                spinner.update(f"Completed importance analysis of {total_changes} files", progress=total_changes)
                time.sleep(0.5)  # Give a moment to see the completed progress
                
        except Exception as e:
            logger.warning(f"Error analyzing changes importance: {str(e)}")
    
    def _analyze_batch_parallel(self, batch: List[GitChange]) -> None:
        """Analyze a batch of changes in parallel."""
        threads = []
        max_threads = min(len(batch), self.resources["threads_available"])
        
        # Create a thread-safe list to store results
        results = [None] * len(batch)
        
        def analyze_change(index, change):
            try:
                prompt = self._create_importance_prompt(change)
                response = self.ollama.generate(prompt)
                
                try:
                    # Parse the response
                    importance = float(response.strip())
                    results[index] = max(0.1, min(10.0, importance))
                except (ValueError, TypeError):
                    # If parsing fails, use default importance
                    logger.debug(f"Failed to parse importance from: {response}")
                    results[index] = 1.0
            except Exception as e:
                logger.debug(f"Error in thread analyzing change: {str(e)}")
                results[index] = 1.0
        
        # Create and start threads
        for i, change in enumerate(batch):
            if len(threads) >= max_threads:
                # Wait for a thread to complete before starting a new one
                for t in threads:
                    if not t.is_alive():
                        threads.remove(t)
                        break
                else:
                    # If all threads are busy, wait a bit
                    threads[0].join(0.1)
                    continue
                    
            t = threading.Thread(target=analyze_change, args=(i, change))
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
            
        # Apply results to changes
        for i, change in enumerate(batch):
            if results[i] is not None:
                change.importance = results[i]
    
    def _analyze_batch_sequential(self, batch: List[GitChange]) -> None:
        """Analyze a batch of changes sequentially."""
        for change in batch:
            prompt = self._create_importance_prompt(change)
            response = self.ollama.generate(prompt)
            
            try:
                # Parse the response
                importance = float(response.strip())
                change.importance = max(0.1, min(10.0, importance))
            except (ValueError, TypeError):
                # If parsing fails, use default importance
                logger.debug(f"Failed to parse importance from: {response}")
                change.importance = 1.0

    def _create_importance_prompt(self, change: GitChange) -> str:
        """Create a prompt for the AI to analyze the importance of a change."""
        file_content = ""
        if change.status != "??":
            file_content = change.content_diff or ""
        else:
            # For untracked files, read a sample of the content
            try:
                with open(os.path.join(self.repo_path, change.filename), 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = "".join(f.readlines()[:50])
            except Exception:
                pass
                
        prompt = f"""
        Analyze the following file change and rate its importance on a scale from 0.1 to 10.0:
        
        File: {change.filename}
        Status: {change.status}
        Content Sample:
        {file_content[:2000]}
        
        Rate the importance as a single number from 0.1 (trivial change) to 10.0 (critical change).
        Consider:
        - Is this a core functionality change? (high importance)
        - Is this just a formatting/style change? (low importance)
        - Does it affect security or performance? (high importance)
        - Is it a documentation update? (medium importance)
        
        Return only a single numeric value.
        """
        return prompt
        
    def analyze_and_group_changes(self) -> None:
        """
        Analyze all changes and group them into logical commits.
        Uses AI if available, otherwise falls back to rule-based analysis.
        """
        # Clear existing groups
        self.commit_groups = []
        
        with Spinner(
            message="Analyzing and grouping changes...", 
            spinner_type=3,
            show_progress_bar=True,
            total=3,  # Three main steps
            width=30
        ) as spinner:
            # Step 1: Initial grouping
            spinner.update("Creating initial commit groups...", progress=0)
            
            if self.use_ai:
                spinner.update("Using AI to group changes intelligently...", progress=1)
                self._ai_group_changes()
            else:
                spinner.update("Using rule-based grouping...", progress=1)
                self._rule_based_group_changes()
                
            # Step 2: Post-processing
            spinner.update("Finalizing commit groups...", progress=2)
            
            # Final check: ensure no group has too many files
            final_groups = []
            for group in self.commit_groups:
                if group.file_count <= 5:
                    final_groups.append(group)
                else:
                    # Split into smaller groups
                    changes = sorted(group.changes, key=lambda c: c.importance, reverse=True)
                    for i in range(0, len(changes), 5):
                        chunk = changes[i:i+5]
                        new_group = CommitGroup(
                            name=f"{group.name} (part {i//5+1})",
                            commit_type=group.commit_type,
                            tech_stack=group.tech_stack.copy() if group.tech_stack else []
                        )
                        for change in chunk:
                            new_group.add_change(change)
                        final_groups.append(new_group)
                        
            self.commit_groups = final_groups
            
            # Step 3: Complete
            spinner.update(f"Created {len(self.commit_groups)} commit groups", progress=3)
            logger.info(f"Created {len(self.commit_groups)} commit groups")
    
    def _ai_group_changes(self) -> None:
        """Use AI to group changes intelligently."""
        try:
            # First, create an initial grouping based on components
            grouped_by_component = defaultdict(list)
            for change in self.changes:
                # Ensure filename is clean and not truncated
                if not hasattr(change, 'filename') or not change.filename:
                    logger.warning("Found change with missing or empty filename, skipping")
                    continue
                    
                # Verify the component is correctly detected
                component = change.component
                if not component:
                    component = "misc"
                
                grouped_by_component[component].append(change)
                
            # Process each component group
            for component, changes in grouped_by_component.items():
                # If too many changes in one component, use AI to subdivide
                if len(changes) > 5:
                    subgroups = self._ai_subdivide_changes(component, changes)
                    for group in subgroups:
                        self.commit_groups.append(group)
                else:
                    # Create a prompt for AI to analyze this small group
                    commit_type, name, description = self._ai_analyze_changes(component, changes)
                    
                    group = CommitGroup(
                        name=name,
                        commit_type=commit_type,
                        description=description
                    )
                    for change in changes:
                        group.add_change(change)
                    self.commit_groups.append(group)
        except Exception as e:
            logger.warning(f"Error in AI grouping: {str(e)}")
            # Fall back to rule-based if AI fails
            self._rule_based_group_changes()
    
    def _ai_subdivide_changes(self, component: str, changes: List[GitChange]) -> List[CommitGroup]:
        """Use AI to subdivide a large group of changes into logical commits."""
        # Create a prompt for AI to suggest logical groups
        changes_summary = "\n".join([f"{c.status} {c.filename}" for c in changes[:20]])
        if len(changes) > 20:
            changes_summary += f"\n... and {len(changes) - 20} more files"
            
        prompt = f"""
        I have a set of {len(changes)} changed files in the '{component}' component that need to be grouped into logical commits.
        Here's a sample of the changes:
        
        {changes_summary}
        
        Suggest how to group these changes into 2-4 logical commits.
        For each group provide:
        1. A commit type (feat, fix, docs, style, refactor, test, chore, perf)
        2. A name for the commit
        3. A brief description
        4. The criteria for which files should be included
        
        Format each group as JSON:
        {{"type": "...", "name": "...", "description": "...", "criteria": "..."}}
        
        Separate each group with ---
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response to get groups
        groups = []
        raw_groups = response.split("---")
        
        for raw_group in raw_groups:
            try:
                # Extract JSON from the text
                json_match = re.search(r'\{.*?\}', raw_group, re.DOTALL)
                if json_match:
                    group_data = json.loads(json_match.group(0))
                    commit_type = CommitType(group_data.get("type", "feat"))
                    group = CommitGroup(
                        name=group_data.get("name", f"Update {component}"),
                        commit_type=commit_type,
                        description=group_data.get("description", "")
                    )
                    
                    # Use the criteria to assign changes
                    criteria = group_data.get("criteria", "").lower()
                    for change in changes:
                        filename = change.filename.lower()
                        if any(token in filename for token in criteria.split()):
                            group.add_change(change)
                    
                    if group.changes:
                        groups.append(group)
            except Exception as e:
                logger.debug(f"Error parsing group: {str(e)}")
        
        # If no valid groups were created, create a single group
        if not groups:
            group = CommitGroup(
                name=f"Update {component}",
                commit_type=CommitType.FEAT
            )
            for change in changes:
                group.add_change(change)
            groups = [group]
            
        return groups
    
    def _ai_analyze_changes(self, component: str, changes: List[GitChange]) -> Tuple[CommitType, str, str]:
        """Use AI to analyze a group of changes and suggest commit details."""
        changes_summary = "\n".join([f"{c.status} {c.filename}" for c in changes])
        
        # Include sample diff content
        diff_samples = []
        for change in changes[:2]:  # Limit to first 2 changes to keep prompt size manageable
            if change.content_diff:
                # Truncate large diffs
                diff_sample = change.content_diff[:500]
                diff_samples.append(f"Sample diff for {change.filename}:\n{diff_sample}")
        
        diff_content = "\n\n".join(diff_samples)
        
        prompt = f"""
        Analyze the following group of changed files in the '{component}' component:
        
        {changes_summary}
        
        {diff_content}
        
        Based on these changes, suggest:
        1. The most appropriate commit type (feat, fix, docs, style, refactor, test, chore, perf)
        2. A concise, descriptive name for the commit (50 chars max)
        3. A brief description of the changes (2-3 sentences)
        
        Format your response as JSON:
        {{"type": "...", "name": "...", "description": "..."}}
        """
        
        response = self.ollama.generate(prompt)
        
        try:
            # Extract JSON from the response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                try:
                    commit_type = CommitType(result.get("type", "feat"))
                except ValueError:
                    commit_type = CommitType.FEAT
                    
                name = result.get("name", f"Update {component}")
                description = result.get("description", "")
                
                return commit_type, name, description
        except Exception as e:
            logger.debug(f"Error parsing AI analysis: {str(e)}")
            
        # Fallback if parsing fails
        return CommitType.FEAT, f"Update {component}", ""
    
    def _rule_based_group_changes(self) -> None:
        """Group changes using rule-based approach following GitHub best practices."""
        # First, separate by broad categories
        by_component: Dict[str, List[GitChange]] = defaultdict(list)
        formatting_changes: List[GitChange] = []
        ci_cd_changes: List[GitChange] = []
        documentation_changes: List[GitChange] = []
        dependency_changes: List[GitChange] = []
        test_changes: List[GitChange] = []
        config_changes: List[GitChange] = []
        
        # Analyze changes and place them in the right category
        for change in self.changes:
            # Validate the change has a proper filename
            if not hasattr(change, 'filename') or not change.filename:
                logger.warning("Found change with missing or empty filename, skipping")
                continue

            # Ensure the filename is clean
            change.filename = change.filename.strip()
                
            # Check specific categories first
            if change.is_formatting_change:
                formatting_changes.append(change)
            elif change.component.startswith('ci') or change.component.startswith('github'):
                ci_cd_changes.append(change)
            elif change.component == 'docs' or change.component.endswith('docs'):
                documentation_changes.append(change)
            elif change.component == 'dependencies' or change.filename.endswith(('package.json', 'package-lock.json', 'yarn.lock', 'requirements.txt', 'Pipfile.lock', 'poetry.lock')):
                dependency_changes.append(change)
            elif change.component.startswith('test') or change.component.endswith('test') or 'test' in change.filename:
                test_changes.append(change)
            elif change.component == 'config' or change.component.endswith('config') or change.filename.startswith('.'):
                config_changes.append(change)
            else:
                by_component[change.component].append(change)

        # Create commits for each category
        
        # Handle formatting changes as a separate commit if any exist
        if formatting_changes:
            group = CommitGroup(
                name="style: improve code formatting and style",
                commit_type=CommitType.STYLE,
                description="Improve code style and formatting for better readability"
            )
            for change in formatting_changes:
                group.add_change(change)
            self.commit_groups.append(group)
            
        # Handle CI/CD changes
        if ci_cd_changes:
            group = CommitGroup(
                name="ci: update CI/CD configuration",
                commit_type=CommitType.CI,
                description="Update continuous integration and deployment setup"
            )
            for change in ci_cd_changes:
                group.add_change(change)
            self.commit_groups.append(group)
            
        # Handle documentation changes
        if documentation_changes:
            group = CommitGroup(
                name="docs: update documentation",
                commit_type=CommitType.DOCS,
                description="Improve project documentation and comments"
            )
            for change in documentation_changes:
                group.add_change(change)
            self.commit_groups.append(group)
            
        # Handle dependency changes
        if dependency_changes:
            group = CommitGroup(
                name="deps: update dependencies",
                commit_type=CommitType.DEPS,
                description="Update project dependencies and package versions"
            )
            for change in dependency_changes:
                group.add_change(change)
            self.commit_groups.append(group)
            
        # Handle test changes
        if test_changes:
            group = CommitGroup(
                name="test: enhance test coverage",
                commit_type=CommitType.TEST,
                description="Add or update tests to improve code quality"
            )
            for change in test_changes:
                group.add_change(change)
            self.commit_groups.append(group)
            
        # Handle config changes
        if config_changes:
            group = CommitGroup(
                name="chore: update configuration files",
                commit_type=CommitType.CHORE,
                description="Update project configuration and settings"
            )
            for change in config_changes:
                group.add_change(change)
            self.commit_groups.append(group)
            
        # Group remaining changes by component
        for component, changes in by_component.items():
            # If too many files in one component, try to sub-divide
            if len(changes) > 5:
                by_type: Dict[str, List[GitChange]] = defaultdict(list)
                for change in changes:
                    by_type[change.file_type].append(change)
                    
                # Create groups for each file type
                for file_type, type_changes in by_type.items():
                    if not type_changes:
                        continue
                        
                    commit_type = self._determine_commit_type(component, file_type)
                    name_prefix = commit_type.value
                    
                    # Create a more descriptive name based on component and changes
                    if "ui" in component.lower() or "component" in component.lower():
                        name = f"{name_prefix}: enhance {component} user interface"
                    elif "api" in component.lower() or "controller" in component.lower():
                        name = f"{name_prefix}: improve {component} endpoints"
                    elif "model" in component.lower() or "data" in component.lower():
                        name = f"{name_prefix}: update {component} data models"
                    elif "service" in component.lower():
                        name = f"{name_prefix}: enhance {component} business logic"
                    elif "util" in component.lower() or "helper" in component.lower():
                        name = f"{name_prefix}: improve {component} utility functions"
                    else:
                        name = f"{name_prefix}: update {component} {file_type} files"
                    
                    group = CommitGroup(
                        name=name,
                        commit_type=commit_type
                    )
                    for change in type_changes:
                        group.add_change(change)
                    self.commit_groups.append(group)
            else:
                # Small enough to be one commit
                commit_type = self._determine_commit_type(component, None)
                name_prefix = commit_type.value
                
                # Create a more descriptive name based on component
                if "ui" in component.lower() or "component" in component.lower():
                    name = f"{name_prefix}: enhance {component} user interface"
                elif "api" in component.lower() or "controller" in component.lower():
                    name = f"{name_prefix}: improve {component} endpoints"
                elif "model" in component.lower() or "data" in component.lower():
                    name = f"{name_prefix}: update {component} data models"
                elif "service" in component.lower():
                    name = f"{name_prefix}: enhance {component} business logic"
                elif "util" in component.lower() or "helper" in component.lower():
                    name = f"{name_prefix}: improve {component} utility functions"
                else:
                    name = f"{name_prefix}: update {component}"
                
                group = CommitGroup(
                    name=name,
                    commit_type=commit_type
                )
                for change in changes:
                    group.add_change(change)
                self.commit_groups.append(group)
    
    def _determine_commit_type(self, component: str, file_type: Optional[str]) -> CommitType:
        """Determine the appropriate commit type based on component and file type."""
        component = component.lower()
        
        # Handle specific components based on GitFlow and commit conventions
        if component == "docs" or component.endswith("docs") or component.endswith("documentation"):
            return CommitType.DOCS
            
        if component == "config" or component.endswith("config") or component == "env":
            return CommitType.CHORE
            
        if component.startswith("test") or component.endswith("test") or component == "specs":
            return CommitType.TEST
            
        if component == "ci" or component.endswith("ci") or component == "github-workflow" or component == "jenkins":
            return CommitType.CI
            
        if component == "build" or component.endswith("build") or component == "webpack" or component == "vite":
            return CommitType.BUILD
        
        if component == "dependencies" or component.endswith("deps"):
            return CommitType.DEPS
            
        if component.startswith("fix") or component.endswith("fix") or component.startswith("bugfix"):
            return CommitType.FIX
            
        if "hotfix" in component:
            return CommitType.HOTFIX
            
        if component == "security" or "security" in component:
            return CommitType.SECURITY
            
        if "i18n" in component or "locale" in component or "translation" in component:
            return CommitType.I18N
            
        if "a11y" in component or "accessibility" in component:
            return CommitType.A11Y
            
        if "ui" in component or "component" in component or "view" in component:
            return CommitType.UI
            
        if "perf" in component or "performance" in component or "optimize" in component:
            return CommitType.PERF
            
        if "refactor" in component:
            return CommitType.REFACTOR
            
        # Handle based on file type
        if file_type:
            if file_type in ["md", "txt", "doc", "docx", "pdf"]:
                return CommitType.DOCS
                
            if file_type in ["test", "spec"]:
                return CommitType.TEST
                
            if file_type in ["yml", "yaml", "json", "toml", "ini"] and "ci" in component:
                return CommitType.CI
                
            if file_type in ["scss", "css", "less", "stylus"]:
                return CommitType.STYLE
        
        # Default to feat for most changes
        return CommitType.FEAT
    
    def _check_for_precommit_hooks(self) -> bool:
        """Check if the repository has pre-commit hooks configured."""
        try:
            with Spinner(message="Checking for pre-commit hooks...", spinner_type=1) as spinner:
                # Check for .pre-commit-config.yaml file
                config_file = os.path.join(self._get_git_root(), ".pre-commit-config.yaml")
                has_config = os.path.isfile(config_file)
                
                # Check for pre-commit hook in .git/hooks
                hooks_dir = os.path.join(self._get_git_root(), ".git", "hooks")
                pre_commit_hook = os.path.join(hooks_dir, "pre-commit")
                has_hook = os.path.isfile(pre_commit_hook) and os.access(pre_commit_hook, os.X_OK)
                
                # Check if pre-commit is installed
                spinner.update("Checking if pre-commit is installed...")
                try:
                    process = subprocess.Popen(
                        ["pre-commit", "--version"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=self.repo_path
                    )
                    process.communicate(timeout=2)
                    pre_commit_installed = process.returncode == 0
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pre_commit_installed = False
                
                # Log findings
                if has_config:
                    spinner.update("Found pre-commit configuration file")
                if has_hook:
                    spinner.update("Found pre-commit hook script")
                if pre_commit_installed:
                    spinner.update("Pre-commit is installed on system")
                
                result = has_config or has_hook or pre_commit_installed
                spinner.update(f"Pre-commit hooks {'detected' if result else 'not detected'}")
                return result
        except Exception as e:
            logger.debug(f"Error checking for pre-commit hooks: {str(e)}")
            return False

    def _revert_staged_changes(self) -> None:
        """Revert all staged changes to leave the repository in a clean state."""
        try:
            logger.info("Reverting staged changes...")
            # First try the modern 'git restore' command (Git 2.23+)
            stdout, code = self._run_git_command(["restore", "--staged", "."])
            
            # If that fails, fall back to the older 'git reset' command
            if code != 0:
                logger.debug("Modern 'git restore' failed, falling back to 'git reset'")
                _, reset_code = self._run_git_command(["reset", "HEAD", "."])
                if reset_code != 0:
                    logger.warning("Failed to revert staged changes. Repository may be in an inconsistent state.")
                    return
            
            logger.info("Successfully reverted all staged changes.")
        except Exception as e:
            logger.warning(f"Error while reverting staged changes: {str(e)}")
    
    def execute_commits(self, interactive: bool = True) -> None:
        """Execute the commits for each group, with optional interactive mode."""
        try:
            if not self.commit_groups:
                logger.warning("No commit groups to execute")
                return
                
            print(f"\n💾 Executing commits...")
            successful_commits = 0
            
            # Confirm available git configuration before committing
            name_out, name_exit = self._run_git_command(["config", "user.name"])
            email_out, email_exit = self._run_git_command(["config", "user.email"])
            
            if name_exit != 0 or email_exit != 0 or not name_out.strip() or not email_out.strip():
                print("\n⚠️ Git user configuration not properly set. Please configure git with:")
                print("  git config --global user.name \"Your Name\"")
                print("  git config --global user.email \"your.email@example.com\"")
                if interactive and input("\nContinue anyway? [y/N]: ").lower() != 'y':
                    return
            
            for i, group in enumerate(self.commit_groups):
                logger.info(f"Commit {i+1}/{len(self.commit_groups)}: {group.name} ({group.file_count} files)")
                
                # Verify group has changes to commit
                if not group.changes:
                    logger.warning(f"Group '{group.name}' has no changes, skipping")
                    continue
                    
                # Show files to be committed
                logger.info("Files to commit:")
                
                # Debug the files that will be staged
                for j, change in enumerate(group.changes):
                    # Fix the issue with how we log the files and ensure filename is valid
                    if not hasattr(change, 'filename') or not change.filename:
                        logger.warning(f"Change {j} has no filename, skipping")
                        continue
                    
                    # Clean up filename and log it
                    clean_filename = change.filename.strip()
                    change.filename = clean_filename  # Update the filename
                    logger.info(f"  {change.status} {clean_filename}")
                    
                # Only file names for console output
                file_list = [change.filename for change in group.changes if hasattr(change, 'filename') and change.filename]
                print(f"\nCommit {i+1}/{len(self.commit_groups)}: {group.name}")
                print(f"Files: {', '.join(os.path.basename(f) for f in file_list[:3])}" + 
                      (f" and {len(file_list) - 3} more" if len(file_list) > 3 else ""))
                
                # In interactive mode, allow customization of the commit
                if interactive:
                    proceed = input("Proceed with this commit? [Y/n/e(dit)/s(kip)]: ").lower()
                    if proceed == "n":
                        return  # Stop the entire process
                    if proceed == "s":
                        continue  # Skip this commit
                    if proceed == "e":
                        # Allow editing the commit details
                        new_name = input(f"Commit name [{group.name}]: ") or group.name
                        group.name = new_name
                        
                        commit_type_str = input(f"Commit type [{group.commit_type.value}]: ") or group.commit_type.value
                        try:
                            group.commit_type = CommitType(commit_type_str)
                        except ValueError:
                            logger.warning(f"Invalid commit type, using {group.commit_type.value}")
                            
                        description = input("Description (optional): ")
                        if description:
                            group.description = description
                            
                        issues = input("Issue numbers (comma-separated, optional): ")
                        if issues:
                            group.issues = set(issues.split(","))
                            
                # Stage the files
                staging_success = True
                for change in group.changes:
                    try:
                        # Get the filename without any further stripping
                        filename = change.filename
                        
                        # Log what we're trying to stage (for debugging)
                        logger.debug(f"Staging file: '{filename}'")
                        
                        # Stage the file - use -- to separate options from file args
                        # This is important for files with special characters or starting with dashes
                        stdout, code = self._run_git_command(["add", "--", filename])
                        
                        if code != 0:
                            # If staging fails, try to provide more information
                            logger.warning(f"Git command failed: {stdout}")
                            
                            # Check if the file actually exists
                            file_path = os.path.join(self.repo_path, filename)
                            if not os.path.exists(file_path):
                                logger.error(f"File not found: {filename} (full path: {file_path})")
                                
                                # Try with normalized path separators (for Windows compatibility)
                                norm_filename = filename.replace('/', os.path.sep)
                                norm_path = os.path.join(self.repo_path, norm_filename)
                                if os.path.exists(norm_path):
                                    logger.info(f"Found file with normalized path: {norm_path}")
                                    # Try staging with normalized path
                                    stdout, code = self._run_git_command(["add", "--", norm_filename])
                                    if code == 0:
                                        logger.info(f"Successfully staged with normalized path: {norm_filename}")
                                        continue
                            
                            logger.error(f"Failed to stage {filename}")
                            staging_success = False
                    except Exception as e:
                        logger.error(f"Exception when staging {filename}: {str(e)}")
                        staging_success = False

                if not staging_success:
                    logger.warning("Staging failed for one or more files. Skipping this commit.")
                    continue
                    
                # Verify what's staged
                logger.info("Staged changes:")
                staged_changes, _ = self._run_git_command(["status", "--short"])
                if not any(line.startswith(("A ", "M ")) for line in staged_changes.splitlines()):
                    logger.warning("No changes staged for commit. Skipping this commit.")
                    continue
                
                # Generate or refine commit message with AI if available
                if self.use_ai:
                    try:
                        # Generate an AI-improved commit message
                        ai_message = self._generate_ai_commit_message(group)
                        if ai_message:
                            group.description = ai_message
                    except Exception as e:
                        logger.warning(f"Failed to generate AI commit message: {str(e)}")
                
                # Generate final commit message
                commit_message = group.generate_commit_message()
                
                # Allow final review of commit message
                if interactive:
                    print("\nCommit message:")
                    print(commit_message)
                    if input("\nProceed with commit? [Y/n]: ").lower() == "n":
                        # Unstage everything
                        self._run_git_command(["reset"])
                        logger.info("Changes unstaged, commit canceled")
                        return
                        
                # Execute the commit
                # Write commit message with UTF-8 encoding explicitly
                try:
                    # First make sure .git directory exists
                    git_dir = os.path.join(self.repo_path, ".git")
                    if not os.path.isdir(git_dir):
                        # Try to find the git directory
                        stdout, _ = self._run_git_command(["rev-parse", "--git-dir"])
                        git_dir = stdout.strip()
                        if not os.path.isdir(git_dir):
                            git_dir = os.path.join(self.repo_path, git_dir)

                    # Now create the commit message file
                    commit_msg_path = os.path.join(git_dir, "COMMIT_EDITMSG")
                    
                    with open(commit_msg_path, "w", encoding='utf-8') as f:
                        f.write(commit_message)
                    
                    # Prepare commit command with hook-skipping if needed
                    commit_cmd = ["commit", "-F", commit_msg_path]
                    if self.skip_hooks:
                        commit_cmd = ["-c", "core.hooksPath=/dev/null"] + commit_cmd
                        
                    stdout, code = self._run_git_command(commit_cmd)
                except Exception as e:
                    logger.error(f"Failed to create or use commit message file: {str(e)}")
                    # Try direct commit as fallback
                    commit_cmd = ["commit", "-m", commit_message]
                    if self.skip_hooks:
                        commit_cmd = ["-c", "core.hooksPath=/dev/null"] + commit_cmd
                    stdout, code = self._run_git_command(commit_cmd)
                finally:
                    # Clean up the temporary commit message file
                    if 'commit_msg_path' in locals() and os.path.exists(commit_msg_path):
                        try:
                            os.remove(commit_msg_path)
                        except OSError as e:
                            logger.warning(f"Could not remove temporary commit message file: {e}")

                if code != 0:
                    if "pre-commit" in stdout or "No module named pre_commit" in stdout or "precommit" in stdout:
                        logger.error("Pre-commit hook failed. Skipping hooks for remaining commits.")
                        # Automatically enable hook skipping after the first pre-commit failure
                        self.skip_hooks = True
                        
                        # Retry the commit with hooks disabled
                        commit_cmd = ["-c", "core.hooksPath=/dev/null", "commit", "-F", commit_msg_path]
                        retry_stdout, retry_code = self._run_git_command(commit_cmd)
                        
                        if retry_code == 0:
                            logger.info("Committed successfully (hooks skipped)")
                            successful_commits += 1
                            continue
                    
                    logger.error("Failed to commit changes")
                    if interactive:
                        if input("Continue with next commit? [y/N]: ").lower() != "y":
                            return
                else:
                    logger.info("Committed successfully")
                    successful_commits += 1
                    
                    # Show commit summary
                    self._run_git_command(["show", "--name-status", "HEAD"])
                    
            # Final status check
            logger.info("All commits completed. Current status:")
            self._run_git_command(["status", "--short"])
            
            # Return the correct status
            if successful_commits == 0 and len(self.commit_groups) > 0:
                raise RuntimeError("No commits were successful. Check the logs for details.")
            
            # Thank you message with donation links
            print("\n" + "-" * 60)
            print("Thank you for using Smart Git Commit! If this tool saved you time,")
            print("please consider supporting development:")
            print("❤️  https://github.com/sponsors/CripterHack")
            print("💰 http://paypal.com/paypalme/cripterhack")
            print("-" * 60 + "\n")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error during commit execution: {str(e)}")
            # Revert any staged changes before exiting
            self._revert_staged_changes()
            raise RuntimeError(f"Failed to execute commits: {str(e)}")
    
    def _generate_ai_commit_message(self, group: CommitGroup) -> str:
        """Use AI to generate an improved commit message description."""
        if not self.use_ai:
            return ""
            
        # Create a summary of the changes in this group
        changes_summary = "\n".join([f"{c.status} {c.filename}" for c in group.changes])
        
        # Include sample diff content from a couple of files
        diff_samples = []
        for change in group.changes[:2]:  # Limit to first 2 changes
            if change.content_diff:
                # Truncate large diffs
                diff_sample = change.content_diff[:300]
                diff_samples.append(f"Sample diff for {change.filename}:\n{diff_sample}")
        
        diff_content = "\n\n".join(diff_samples)
        
        prompt = f"""
        I'm creating a commit with the following changes:
        
        {changes_summary}
        
        {diff_content}
        
        Commit type: {group.commit_type.value}
        Current commit name: {group.name}
        
        Write a concise, informative description for this commit (2-4 sentences).
        The description should focus on WHAT changed and WHY, not HOW.
        Use present tense (e.g., "Add feature" not "Added feature").
        Focus on technical details rather than trivial changes.
        Do not list the filenames again.
        Keep the first sentence brief as it will be used in the subject line.
        
        Description:
        """
        
        response = self.ollama.generate(prompt)
        return response.strip()


def get_version():
    """Get the current version from package metadata."""
    try:
        # Try to get version from the package metadata
        return pkg_resources.get_distribution("smart-git-commit").version
    except Exception:
        # Fallback to searching in files
        try:
            # Try to find the version in pyproject.toml
            import re
            import os
            
            # Look for version in pyproject.toml
            pyproject_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
            if os.path.exists(pyproject_path):
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    match = re.search(r'version\s*=\s*"([^"]+)"', content)
                    if match:
                        return match.group(1)
            
            # Try setup.py if pyproject.toml doesn't exist or doesn't contain version
            setup_path = os.path.join(os.path.dirname(__file__), "setup.py")
            if os.path.exists(setup_path):
                with open(setup_path, 'r') as f:
                    content = f.read()
                    match = re.search(r'version\s*=\s*"([^"]+)"', content)
                    if match:
                        return match.group(1)
        except Exception:
            pass
        
        # Default version if all else fails
        return "0.2.0"  # Fallback version


def display_version():
    """Display version information and support links."""
    version = get_version()
    
    print("\n" + "=" * 60)
    print(f"🚀 Smart Git Commit v{version}")
    print("=" * 60)
    print("\nAn AI-powered Git workflow tool that intelligently analyzes")
    print("your changes and generates detailed commit messages.")
    print("\nAuthor: Edgar Zorrilla <edgar@izignamx.com>")
    print("Repository: https://github.com/CripterHack/smart-git-commit")
    print("\nSupport this project:")
    print("❤️  GitHub Sponsors: https://github.com/sponsors/CripterHack")
    print("💰 PayPal: http://paypal.com/paypalme/cripterhack")
    print("=" * 60 + "\n")
    sys.exit(0)


def main() -> int:
    """Main function to run the smart git commit workflow."""
    parser = argparse.ArgumentParser(description="Smart Git Commit Workflow with Ollama Integration")
    parser.add_argument("--repo-path", help="Path to the git repository", default=".")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts")
    parser.add_argument("--ollama-host", help="Host for Ollama API", default="http://localhost:11434")
    parser.add_argument("--ollama-model", help="Model to use for Ollama (will prompt if not specified)")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI-powered analysis")
    parser.add_argument("--timeout", type=int, help=f"Timeout in seconds for HTTP requests (default: {DEFAULT_TIMEOUT})", default=DEFAULT_TIMEOUT)
    parser.add_argument("--verbose", action="store_true", help="Show verbose debug output")
    parser.add_argument("--skip-hooks", action="store_true", help="Skip Git hooks when committing (useful if pre-commit is not installed)")
    parser.add_argument("--no-revert", action="store_true", help="Don't automatically revert staged changes on error")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing for slower but more stable operation")
    parser.add_argument("--version", action="store_true", help="Show version information and support links")
    args = parser.parse_args()
    
    # Show version info and exit if requested
    if args.version:
        display_version()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Print welcome message
    print(f"\n{'=' * 60}")
    print(f"🚀 Smart Git Commit v{get_version()}")
    print(f"{'=' * 60}")
    print(f"\nIF THIS TOOL HELPS YOU, PLEASE CONSIDER SUPPORTING IT:")
    print(f"❤️  GitHub Sponsors: https://github.com/sponsors/CripterHack")
    print(f"💰 PayPal: http://paypal.com/paypalme/cripterhack")
    print(f"{'=' * 60}\n")
    
    workflow = None
    try:
        # Verify the repository path exists
        if not os.path.exists(args.repo_path):
            logger.error(f"Repository path '{args.repo_path}' does not exist")
            print(f"\n❌ ERROR: Repository path '{args.repo_path}' does not exist\n")
            return 1
        
        # Show repository info
        repo_path = os.path.abspath(args.repo_path)
        print(f"📂 Repository: {repo_path}")
        
        # Check system resources
        resources = get_system_resources()
        print(f"🖥️  System resources: {resources['cpu_count']} CPUs, {resources['memory_gb']:.1f}GB RAM")
        print(f"⚙️  Using {'parallel' if not args.no_parallel else 'sequential'} processing with batch size {resources['batch_size']}")
        
        # Create the workflow without a spinner
        # The OllamaClient has its own progress indicators and handles user interaction
        try:
            print("\n🚀 Initializing Smart Git Commit workflow...")
            
            # Only show spinner for non-AI mode to avoid conflict with model selection UI
            if args.no_ai:
                with Spinner(message="Setting up rule-based analysis...", spinner_type=0):
                    workflow = SmartGitCommitWorkflow(
                        repo_path=args.repo_path,
                        ollama_host=args.ollama_host,
                        ollama_model=args.ollama_model,
                        use_ai=False,
                        timeout=args.timeout,
                        skip_hooks=args.skip_hooks,
                        parallel=not args.no_parallel
                    )
            else:
                # Initialize without spinner to allow proper UI for model selection
                workflow = SmartGitCommitWorkflow(
                    repo_path=args.repo_path,
                    ollama_host=args.ollama_host,
                    ollama_model=args.ollama_model,
                    use_ai=not args.no_ai,
                    timeout=args.timeout,
                    skip_hooks=args.skip_hooks,
                    parallel=not args.no_parallel
                )
        except RuntimeError as e:
            # Handle git repository errors with a clear message
            logger.error(str(e))
            print(f"\n❌ ERROR: {str(e)}")
            print("\nPlease make sure you're running this command from within a git repository.")
            print("You can initialize a git repository with: git init\n")
            return 1
        
        # Load changes
        try:
            print("\n🔍 Detecting changes...")
            workflow.load_changes()
        except RuntimeError as e:
            logger.error(f"Failed to load changes: {str(e)}")
            print(f"\n❌ ERROR: {str(e)}\n")
            return 1
        
        # Check if there are changes to commit
        if not workflow.changes:
            logger.info("No changes to commit")
            print("\n✅ No changes to commit. Working directory is clean.")
            
            # Thank you message with donation links even when there are no changes
            print("\n" + "-" * 60)
            print("Thank you for using Smart Git Commit! If this tool is helpful,")
            print("please consider supporting development:")
            print("❤️  https://github.com/sponsors/CripterHack")
            print("💰 http://paypal.com/paypalme/cripterhack")
            print("-" * 60 + "\n")
            
            return 0
            
        # Analyze and group changes
        print(f"\n🧩 Found {len(workflow.changes)} changed files. Analyzing and grouping...")
        with Spinner(message="Organizing changes into logical commits...", spinner_type=2):
            workflow.analyze_and_group_changes()
        
        print(f"\n📋 Created {len(workflow.commit_groups)} commit groups:")
        for i, group in enumerate(workflow.commit_groups):
            print(f"  {i+1}. {group.name} ({group.file_count} files)")
        
        # Execute commits
        try:
            print("\n💾 Executing commits...")
            workflow.execute_commits(interactive=not args.non_interactive)
            print("\n✅ Commit operation completed successfully.")
            
            # Thank you message with donation links
            print("\n" + "-" * 60)
            print("Thank you for using Smart Git Commit! If this tool saved you time,")
            print("please consider supporting development:")
            print("❤️  https://github.com/sponsors/CripterHack")
            print("💰 http://paypal.com/paypalme/cripterhack")
            print("-" * 60 + "\n")
            
            return 0
        except RuntimeError as e:
            logger.error(f"Failed to execute commits: {str(e)}")
            print(f"\n❌ ERROR during commit execution: {str(e)}\n")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user.")
        if workflow and not args.no_revert:
            workflow._revert_staged_changes()
            print("🔄 Staged changes have been reverted.")
            
        # Still show donation links on keyboard interrupt
        print("\n" + "-" * 60)
        print("Thanks for trying Smart Git Commit! If you find it useful,")
        print("please consider supporting development:")
        print("❤️  https://github.com/sponsors/CripterHack")
        print("💰 http://paypal.com/paypalme/cripterhack")
        print("-" * 60 + "\n")
        
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error during git commit workflow: {str(e)}", exc_info=True)
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        print("\nPlease report this issue with the error details from the log.")
        
        # Revert staged changes if workflow was created
        if workflow and not args.no_revert:
            workflow._revert_staged_changes()
            print("🔄 Staged changes have been reverted.")
        
        return 1


if __name__ == "__main__":
    sys.exit(main()) 