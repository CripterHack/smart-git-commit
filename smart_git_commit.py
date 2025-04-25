#!/usr/bin/env python3
"""
Smart Git Commit Workflow with Ollama Integration

An advanced git commit workflow tool that uses Ollama with GPU acceleration
to intelligently analyze and group changes, generate meaningful commit messages,
and adapt to different tech stacks automatically.
"""

import os
import sys
import subprocess
import re
import json
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import argparse
import logging
from collections import defaultdict
import http.client
import urllib.request
import urllib.parse
import socket

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("smart_git_commit")


class CommitType(Enum):
    """Types of commits following Conventional Commits specification."""
    FEAT = "feat"
    FIX = "fix"
    DOCS = "docs"
    STYLE = "style"
    REFACTOR = "refactor"
    TEST = "test"
    CHORE = "chore"
    PERF = "perf"
    BUILD = "build"
    CI = "ci"


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
            if parts[0].startswith("README"):
                return "docs"
            if parts[0].startswith(".env"):
                return "config"
            if parts[0].startswith("package.json") or parts[0].startswith("requirements.txt"):
                return "dependencies"
            if parts[0].startswith("Dockerfile") or parts[0].startswith("docker-compose"):
                return "docker"
            if parts[0].endswith(".py"):
                return "core"
            return "root"
        
        # Handle special directories common in many tech stacks
        if parts[0] in ("src", "app", "lib", "internal"):
            # If there's a subdirectory, use that for more specificity
            if len(parts) > 2:
                return f"{parts[0]}-{parts[1]}"
            return parts[0]
            
        # Handle common directory names across tech stacks
        common_dirs = {
            "docs": ["docs", "documentation", "wiki"],
            "tests": ["test", "tests", "spec", "specs", "__tests__"],
            "config": ["config", "configs", "conf", "settings"],
            "scripts": ["scripts", "tools", "bin", "utilities"],
            "styles": ["css", "styles", "scss", "sass"],
            "api": ["api", "endpoints", "routes", "controllers"],
            "models": ["models", "entities", "schemas", "types"],
            "utils": ["utils", "helpers", "common"],
            "assets": ["assets", "static", "public", "resources"]
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
            'fmt' in self.content_diff.lower() and len(self.content_diff) < 500
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
        subject = f"{self.commit_type.value}({scope}): {self.name}"
        if len(subject) > 50:
            # Truncate if too long
            subject = subject[:47] + "..."
            
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


class OllamaClient:
    """Client for interacting with Ollama API with GPU acceleration."""
    
    def __init__(self, host: str = "http://localhost:11434", model: Optional[str] = None, timeout: int = 10):
        """
        Initialize the Ollama client.
        
        Args:
            host: Host for Ollama API
            model: Model to use for Ollama, if None will prompt user to select one
            timeout: Timeout in seconds for HTTP requests
        """
        self.host = host
        self.headers = {"Content-Type": "application/json"}
        self.timeout = timeout
        
        try:
            self.available_models = self._get_available_models()
            
            if not self.available_models:
                logger.warning("No models found in Ollama. Make sure Ollama is running.")
                raise RuntimeError("No Ollama models available")
                
            if model is None:
                self.model = self._select_model()
            else:
                if model not in self.available_models:
                    logger.warning(f"Model {model} not found. Available models: {', '.join(self.available_models)}")
                    self.model = self._select_model()
                else:
                    self.model = model
                    
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
            host, port = self._get_host_connection()
            conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
            conn.request("GET", "/api/tags")
            response = conn.getresponse()
            
            if response.status != 200:
                logger.warning(f"Failed to get models: HTTP {response.status} {response.reason}")
                return self._get_models_from_cli()
                
            data = json.loads(response.read().decode())
            
            # Different Ollama API versions might return models differently
            if "models" in data:
                # Newer API
                return [model["name"] for model in data.get("models", [])]
            elif "tags" in data:
                # Older API
                return [tag["name"] for tag in data.get("tags", [])]
            else:
                # Try to run ollama list directly if API doesn't work
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
            process = subprocess.Popen(
                ["ollama", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=self.timeout)
            if process.returncode != 0:
                logger.warning(f"Ollama CLI failed with error: {stderr}")
                return []
                
            models = []
            # Parse output, expecting format like "NAME  ID  SIZE  MODIFIED"
            for line in stdout.splitlines()[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
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
            
        print("\nAvailable Ollama models:")
        for i, model in enumerate(self.available_models):
            print(f"{i+1}. {model}")
            
        while True:
            try:
                selection = input(f"\nSelect a model (1-{len(self.available_models)}): ")
                idx = int(selection) - 1
                if 0 <= idx < len(self.available_models):
                    return self.available_models[idx]
                print(f"Please enter a number between 1 and {len(self.available_models)}")
            except ValueError:
                # If input is not a number, check if it's a model name
                if selection in self.available_models:
                    return selection
                print("Please enter a valid model number or name")
            except KeyboardInterrupt:
                # If user interrupts, use first model as default
                print("\nInterrupted, using first available model")
                return self.available_models[0]
    
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
                 skip_hooks: bool = False):
        """
        Initialize the workflow.
        
        Args:
            repo_path: Path to the git repository
            ollama_host: Host for Ollama API
            ollama_model: Model to use for Ollama, if None will prompt user to select
            use_ai: Whether to use AI-powered analysis
            timeout: Timeout in seconds for HTTP requests to Ollama
            skip_hooks: Whether to skip git hooks when committing
        """
        self.repo_path = repo_path
        self.changes: List[GitChange] = []
        self.commit_groups: List[CommitGroup] = []
        self.use_ai = use_ai
        self.ollama = None
        self.timeout = timeout
        self.skip_hooks = skip_hooks
        
        # Verify if we're in a git repository first
        if not self._is_git_repository():
            raise RuntimeError(f"Directory '{os.path.abspath(repo_path)}' is not a git repository. Please run from a valid git repository.")
        
        # Check for pre-commit hooks
        self.has_precommit_hooks = self._check_for_precommit_hooks()
        
        if use_ai:
            try:
                self.ollama = OllamaClient(host=ollama_host, model=ollama_model, timeout=timeout)
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {str(e)}")
                logger.info("Falling back to rule-based analysis")
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
        process = subprocess.Popen(
            ["git"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.repo_path,
            text=True,
            encoding='utf-8',  # Specify UTF-8 encoding
            errors='ignore'    # Ignore decoding errors
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0 and stderr:
            logger.warning(f"Git command failed: {stderr}")
        return stdout, process.returncode
    
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
        git_root = self._get_git_root()
        repo_abs_path = os.path.abspath(self.repo_path)
        
        # If we're running from the git root, return the path as is
        if os.path.samefile(git_root, repo_abs_path):
            return path
            
        # If we're in a subdirectory, check if the path already includes that subdirectory
        rel_path = os.path.relpath(repo_abs_path, git_root)
        if path.startswith(rel_path + os.path.sep):
            return path
        else:
            # Path is relative to git root, not current directory
            return path

    def load_changes(self) -> None:
        """Load all modified and untracked files from git status."""
        try:
            stdout, code = self._run_git_command(["status", "--porcelain"])
            if code != 0:
                raise RuntimeError("Failed to get git status")
                
            # Verify we're in a git repository with proper status output
            if not stdout and not self._is_git_repository():
                raise RuntimeError("Not in a git repository. Please run from a valid git repository.")
                
            self.changes = []
            # Detect tech stack but don't assign to unused variable
            self._detect_tech_stack()
            
            for line in stdout.splitlines():
                if not line.strip():
                    continue
                    
                status = line[:2].strip()
                filename = line[3:].strip()
                
                # Remove any leading "backend/" or similar prefix that might come from running in a subdirectory
                if " -> " in filename:  # Handle renamed files
                    old_path, filename = filename.split(" -> ")
                
                # Get the proper path relative to git root
                filename = self._get_relative_path(filename)
                
                # Get diff content for modified files
                content_diff = None
                if status != "??":  # Not for untracked files
                    diff_out, _ = self._run_git_command(["diff", "--", filename])
                    content_diff = diff_out
                    
                # Create the change object
                change = GitChange(status=status, filename=filename, content_diff=content_diff)
                
                # Detect language
                _, ext = os.path.splitext(filename)
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
                
            logger.info(f"Loaded {len(self.changes)} changed files")
            
            # Analyze importance of each change
            if self.use_ai:
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
            "python": ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
            "node": ["package.json", "yarn.lock", "node_modules"],
            "ruby": ["Gemfile", "config/routes.rb", ".ruby-version"],
            "php": ["composer.json", "artisan", "index.php"],
            "java": ["pom.xml", "build.gradle", "gradlew"],
            "dotnet": [".csproj", ".sln", "Program.cs"],
            "go": ["go.mod", "go.sum", "main.go"],
            "rust": ["Cargo.toml", "Cargo.lock"],
            "docker": ["Dockerfile", "docker-compose.yml"],
            "web": ["index.html", "styles.css", "main.js"],
        }
        
        result = {}
        for stack, markers in stack_markers.items():
            for marker in markers:
                if os.path.exists(os.path.join(self.repo_path, marker)):
                    result[stack] = True
                    break
        
        # Check for specific frontend frameworks
        if "node" in result:
            package_json = os.path.join(self.repo_path, "package.json")
            if os.path.exists(package_json):
                try:
                    with open(package_json, "r") as f:
                        data = json.load(f)
                        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                        if "react" in deps:
                            result["react"] = True
                        if "vue" in deps:
                            result["vue"] = True
                        if "angular" in deps or "@angular/core" in deps:
                            result["angular"] = True
                except Exception:
                    pass
                    
        return result
        
    def _analyze_changes_importance(self) -> None:
        """Use AI to analyze the importance of changes."""
        try:
            # Process in batches to avoid overwhelming Ollama
            batch_size = 5
            for i in range(0, len(self.changes), batch_size):
                batch = self.changes[i:i+batch_size]
                
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
        except Exception as e:
            logger.warning(f"Error analyzing changes importance: {str(e)}")
            
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
        
        if self.use_ai:
            self._ai_group_changes()
        else:
            self._rule_based_group_changes()
            
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
        logger.info(f"Created {len(self.commit_groups)} commit groups")
    
    def _ai_group_changes(self) -> None:
        """Use AI to group changes intelligently."""
        try:
            # First, create an initial grouping based on components
            grouped_by_component = defaultdict(list)
            for change in self.changes:
                grouped_by_component[change.component].append(change)
                
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
        """Group changes using rule-based approach."""
        # First, separate by broad categories
        by_component: Dict[str, List[GitChange]] = defaultdict(list)
        formatting_changes: List[GitChange] = []
        
        for change in self.changes:
            if change.is_formatting_change:
                formatting_changes.append(change)
            else:
                by_component[change.component].append(change)
                
        # Handle formatting changes as a separate commit if any exist
        if formatting_changes:
            group = CommitGroup(
                name="Improve code formatting and style",
                commit_type=CommitType.STYLE
            )
            for change in formatting_changes:
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
                    group = CommitGroup(
                        name=f"Update {component} {file_type} files",
                        commit_type=commit_type
                    )
                    for change in type_changes:
                        group.add_change(change)
                    self.commit_groups.append(group)
            else:
                # Small enough to be one commit
                commit_type = self._determine_commit_type(component, None)
                group = CommitGroup(
                    name=f"Update {component}",
                    commit_type=commit_type
                )
                for change in changes:
                    group.add_change(change)
                self.commit_groups.append(group)
    
    def _determine_commit_type(self, component: str, file_type: Optional[str]) -> CommitType:
        """Determine the appropriate commit type based on component and file type."""
        if component == "docs" or component.endswith("README"):
            return CommitType.DOCS
            
        if component == "config" or component.endswith("config"):
            return CommitType.CHORE
            
        if component.startswith("test") or file_type == "test":
            return CommitType.TEST
            
        if component == "ci" or component.endswith("ci"):
            return CommitType.CI
            
        if component == "build" or component.endswith("build"):
            return CommitType.BUILD
            
        # Default to feat for most changes
        return CommitType.FEAT
    
    def _check_for_precommit_hooks(self) -> bool:
        """Check if the repository has pre-commit hooks configured."""
        # Check for .pre-commit-config.yaml file
        precommit_config = os.path.join(self._get_git_root(), ".pre-commit-config.yaml")
        if os.path.exists(precommit_config):
            return True
            
        # Check for pre-commit hook file in the git hooks directory
        hooks_dir = os.path.join(self._get_git_root(), ".git", "hooks")
        pre_commit_hook = os.path.join(hooks_dir, "pre-commit")
        if os.path.exists(pre_commit_hook):
            # Check if it contains pre-commit references
            try:
                with open(pre_commit_hook, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    if 'pre-commit' in content or 'precommit' in content:
                        return True
            except Exception:
                pass
                
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
        if not self.commit_groups:
            logger.warning("No commit groups to execute")
            return

        try:
            # Check if git is properly configured for commits
            user_name, code1 = self._run_git_command(["config", "user.name"])
            user_email, code2 = self._run_git_command(["config", "user.email"])
            
            if code1 != 0 or code2 != 0 or not user_name.strip() or not user_email.strip():
                logger.error("Git is not properly configured. Please set user.name and user.email:")
                logger.error("  git config --global user.name \"Your Name\"")
                logger.error("  git config --global user.email \"your.email@example.com\"")
                return
                
            # Check if pre-commit hooks might cause issues
            if self.has_precommit_hooks and not self.skip_hooks:
                # Try to detect if pre-commit module is available
                _, pre_commit_check = self._run_git_command(["-c", "core.hooksPath=/dev/null", "status"])
                if pre_commit_check != 0:
                    # This indicates a potential issue with hooks
                    logger.warning("Pre-commit hooks detected that might cause issues.")
                    if interactive:
                        skip_hooks = input("Skip git hooks for these commits? [Y/n]: ").lower() != "n"
                        self.skip_hooks = skip_hooks
                    else:
                        logger.warning("Using --skip-hooks to bypass pre-commit hooks. Install pre-commit if needed.")
                        self.skip_hooks = True
                
            for i, group in enumerate(self.commit_groups):
                logger.info(f"Commit {i+1}/{len(self.commit_groups)}: {group.name} ({group.file_count} files)")
                
                # Show files to be committed
                logger.info("Files to commit:")
                for change in group.changes:
                    logger.info(f"  {change.status} {change.filename}")
                    
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
                    _, code = self._run_git_command(["add", change.filename])
                    if code != 0:
                        logger.error(f"Failed to stage {change.filename}")
                        staging_success = False
                        if interactive:
                            if input("Continue anyway? [y/N]: ").lower() != "y":
                                return
                        
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
                    if "pre-commit" in stdout or "precommit" in stdout:
                        logger.error("Pre-commit hook failed. Run with --skip-hooks to bypass, or install pre-commit module.")
                        if not self.skip_hooks and interactive:
                            skip_hooks = input("Skip git hooks for remaining commits? [Y/n]: ").lower() != "n"
                            if skip_hooks:
                                self.skip_hooks = True
                                # Retry the commit with hooks disabled
                                commit_cmd = ["-c", "core.hooksPath=/dev/null", "commit", "-F", commit_msg_path]
                                stdout, code = self._run_git_command(commit_cmd)
                                if code == 0:
                                    logger.info("Committed successfully (hooks skipped)")
                                    continue
                    
                    logger.error("Failed to commit changes")
                    if interactive:
                        if input("Continue with next commit? [y/N]: ").lower() != "y":
                            return
                else:
                    logger.info("Committed successfully")
                    
                    # Show commit summary
                    self._run_git_command(["show", "--name-status", "HEAD"])
                    
            # Final status check
            logger.info("All commits completed. Current status:")
            self._run_git_command(["status", "--short"])
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
        
        Description:
        """
        
        response = self.ollama.generate(prompt)
        return response.strip()


def main() -> int:
    """Main function to run the smart git commit workflow."""
    parser = argparse.ArgumentParser(description="Smart Git Commit Workflow with Ollama Integration")
    parser.add_argument("--repo-path", help="Path to the git repository", default=".")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts")
    parser.add_argument("--ollama-host", help="Host for Ollama API", default="http://localhost:11434")
    parser.add_argument("--ollama-model", help="Model to use for Ollama (will prompt if not specified)")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI-powered analysis")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds for HTTP requests", default=10)
    parser.add_argument("--verbose", action="store_true", help="Show verbose debug output")
    parser.add_argument("--skip-hooks", action="store_true", help="Skip Git hooks when committing (useful if pre-commit is not installed)")
    parser.add_argument("--no-revert", action="store_true", help="Don't automatically revert staged changes on error")
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    workflow = None
    try:
        # Verify the repository path exists
        if not os.path.exists(args.repo_path):
            logger.error(f"Repository path '{args.repo_path}' does not exist")
            print(f"\n❌ ERROR: Repository path '{args.repo_path}' does not exist\n")
            return 1
        
        # Create the workflow
        try:
            workflow = SmartGitCommitWorkflow(
                repo_path=args.repo_path,
                ollama_host=args.ollama_host,
                ollama_model=args.ollama_model,
                use_ai=not args.no_ai,
                timeout=args.timeout,
                skip_hooks=args.skip_hooks
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
            workflow.load_changes()
        except RuntimeError as e:
            logger.error(f"Failed to load changes: {str(e)}")
            print(f"\n❌ ERROR: {str(e)}\n")
            return 1
        
        # Check if there are changes to commit
        if not workflow.changes:
            logger.info("No changes to commit")
            print("\n✓ No changes to commit. Working directory is clean.")
            return 0
            
        # Analyze and group changes
        workflow.analyze_and_group_changes()
        
        # Execute commits
        try:
            workflow.execute_commits(interactive=not args.non_interactive)
            print("\n✓ Commit operation completed successfully.")
            return 0
        except RuntimeError as e:
            logger.error(f"Failed to execute commits: {str(e)}")
            print(f"\n❌ ERROR during commit execution: {str(e)}\n")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        if workflow and not args.no_revert:
            workflow._revert_staged_changes()
            print("Staged changes have been reverted.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error during git commit workflow: {str(e)}", exc_info=True)
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        print("\nPlease report this issue with the error details from the log.")
        
        # Revert staged changes if workflow was created
        if workflow and not args.no_revert:
            workflow._revert_staged_changes()
            print("Staged changes have been reverted.")
        
        return 1


if __name__ == "__main__":
    sys.exit(main()) 