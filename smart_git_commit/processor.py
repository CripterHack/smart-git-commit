from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import re
import os
import json
from enum import Enum
import difflib
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
import abc
import requests

from .git_utils import get_repository_details, parse_status_line
from .commit_group import CommitGroup, CommitType
from .smart_git_commit import GitChange

logger = logging.getLogger(__name__)

class Configuration:
    """Configuration management for smart-git-commit."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional config path."""
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path:
            return {}
            
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {str(e)}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.config[key] = value
        if self.config_path:
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save config to {self.config_path}: {str(e)}")

def get_config_file_path(repo_path: str = ".") -> Optional[str]:
    """Find the path to the config file."""
    # Check repo path first
    repo_config = os.path.join(repo_path, ".smart-commit.json")
    if os.path.exists(repo_config):
        return repo_config
    
    # Check home directory
    home_config = os.path.join(Path.home(), ".smart-commit.json")
    if os.path.exists(home_config):
        return home_config
        
    return None

def load_config() -> Configuration:
    """Load configuration."""
    config_path = get_config_file_path()
    return Configuration(config_path)


# --- AI Client Definitions ---

class AIClient(abc.ABC):
    """Abstract base class for AI clients."""
    
    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text based on a prompt."""
        raise NotImplementedError

class OllamaClient(AIClient):
    """Client for interacting with Ollama API."""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = None):
        """Initialize Ollama client."""
        self.host = host
        self.model = model or "llama3"  # Default model
        self._validate_setup()
    
    def _validate_setup(self) -> None:
        """Validate connection and model availability."""
        try:
            response = requests.get(f"{self.host}/api/tags")
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            
            if self.model not in models:
                logger.warning(f"Model '{self.model}' not found. Available: {models}")
                if models:
                    self.model = models[0]
                    logger.warning(f"Falling back to model: {self.model}")
        except Exception as e:
            logger.warning(f"Failed to validate Ollama setup: {str(e)}")
            # Continue with default model, generate will likely fail

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 500  # Limit response length
                }
            }
            if system_prompt:
                payload["system"] = system_prompt
                
            response = requests.post(
                f"{self.host}/api/generate", 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Failed to generate text with Ollama: {str(e)}")
            return ""

class OpenAIClient(AIClient):
    """Client for interacting with OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI client."""
        self.api_key = api_key
        self.model = model
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            logger.error("OpenAI library not found. Please install with: pip install openai")
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using OpenAI."""
        if not self.client:
            return ""
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate text with OpenAI: {str(e)}")
            return ""


def get_processor() -> 'CommitProcessor':
    """Get a commit processor based on configuration.
    
    Returns:
        CommitProcessor: Configured processor instance
        
    Raises:
        ValueError: If configuration is invalid or required settings are missing
    """
    config = load_config()
    processor_type = config.get('processor', 'ollama').lower()
    
    # Validate processor type
    valid_processors = {'openai', 'ollama'}
    if processor_type not in valid_processors:
        raise ValueError(f"Invalid processor type: {processor_type}. Must be one of: {', '.join(valid_processors)}")
    
    if processor_type == 'openai':
        api_key = config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required but not provided. Set in config file or OPENAI_API_KEY environment variable.")
        
        model = config.get('openai_model', 'gpt-3.5-turbo')
        return CommitProcessor(use_ai=True, ai_provider='openai')
        
    elif processor_type == 'ollama':
        # Check if custom Ollama settings are provided
        host = config.get('ollama_host', 'http://localhost:11434')
        model = config.get('ollama_model', 'llama3')
        return CommitProcessor(use_ai=True, ai_provider='ollama')
    
    # This should never be reached due to validation above
    raise ValueError(f"Unsupported processor type: {processor_type}")



# --- Commit Processor ---

class CommitProcessor:
    """Process git changes into commit groups and generate messages."""
    
    def __init__(self, use_ai: bool = True, ai_provider: str = 'ollama'):
        """Initialize processor with optional AI support.
        
        Args:
            use_ai (bool): Whether to use AI for message generation
            ai_provider (str): AI provider to use ('ollama' or 'openai')
            
        Raises:
            ValueError: If AI configuration is invalid
        """
        self.use_ai = use_ai
        self.ai_provider = ai_provider.lower()
        self.config = load_config()
        
        if use_ai:
            if self.ai_provider == 'ollama':
                host = self.config.get('ollama_host', 'http://localhost:11434')
                model = self.config.get('ollama_model', 'llama3')
                self.ai_client = OllamaClient(host=host, model=model)
            elif self.ai_provider == 'openai':
                api_key = self.config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key is required but not provided")
                model = self.config.get('openai_model', 'gpt-3.5-turbo')
                self.ai_client = OpenAIClient(api_key=api_key, model=model)
            else:
                raise ValueError(f"Unsupported AI provider: {self.ai_provider}")
        else:
            self.ai_client = None
            
        # Load component mapping rules
        self.component_rules = self.config.get('component_rules', {})
        
        # Get repository details
        self.repo_path = os.path.abspath(".")
        self.repo_details = get_repository_details(self.repo_path)
        logger.info(f"Initialized CommitProcessor for {self.repo_details['name']}")

    def get_changes(self) -> List[GitChange]:
        """Get all changed files from git status."""
        changes = []
        # Run git status to get changed files
        process = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            logger.error(f"Failed to get git status: {process.stderr}")
            return []
            
        status_lines = process.stdout.strip().split('\n')
        if not status_lines or status_lines[0] == '':
            logger.info("No changes detected in repository")
            return []
            
        for line in status_lines:
            if not line.strip():
                continue
                
            # Parse the status line
            status, filename = parse_status_line(line)
            
            # Map status to our ChangeState enum or simple strings
            if status == "??": status = "untracked"
            elif status == "A": status = "added"
            elif status == "M": status = "modified"
            elif status == "D": status = "deleted"
            elif status.startswith("R"): status = "renamed"
            else: status = "unknown"
            
            component = self._determine_component(filename)
            diff = self._get_diff(filename, status)
            
            change = GitChange(filename=filename, status=status, component=component, diff=diff)
            changes.append(change)
            
        logger.info(f"Found {len(changes)} changes in repository")
        return changes
        
    def _determine_component(self, filename: str) -> str:
        """Determine the component for a file based on rules."""
        for pattern, component in self.component_rules.items():
            match = re.search(pattern, filename)
            if match:
                # If the component has a capture group, use it
                if r'\1' in component and match.groups():
                    # Extract the first directory after src/
                    return match.group(1).split('/')[0].split('.')[0]
                return component
                
        # Default component based on file extension
        extension = os.path.splitext(filename)[1].lstrip('.').lower()
        if extension in ['js', 'ts', 'jsx', 'tsx']:
            return 'frontend'
        elif extension in ['py', 'rb', 'java', 'go', 'rs', 'c', 'cpp', 'h']:
            return 'backend'
        elif extension in ['css', 'scss', 'html', 'svg']:
            return 'ui'
        elif extension in ['md', 'rst', 'txt']:
            return 'docs'
        elif extension in ['json', 'yml', 'yaml', 'toml', 'ini']:
            return 'config'
        
        # Default to root if no match
        return "root"

    def _get_diff(self, filename: str, status: str) -> Optional[str]:
        """Get the diff content for a file."""
        if status == '??' or status == 'D':
            return ""
            
        try:
            # Get the diff
            process = subprocess.run(
                ['git', 'diff', '--cached', '--', filename] if status == 'A' else ['git', 'diff', '--', filename],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if process.returncode == 0:
                return process.stdout
                
        except Exception as e:
            logger.error(f"Failed to get diff for {filename}: {str(e)}")
            
        return ""

    def group_changes(self, changes: List[GitChange]) -> List[CommitGroup]:
        """Group changes into logical commits based on component."""
        groups = defaultdict(list)
        for change in changes:
            groups[change.component].append(change)
            
        commit_groups = []
        for component, component_changes in groups.items():
            # Determine commit type based on files
            commit_type = self._determine_commit_type(component_changes)
            
            # Create the group
            group = CommitGroup(
                changes=component_changes,
                commit_type=commit_type,
                component=component
            )
            group.issues = self._extract_issues(component_changes)
            commit_groups.append(group)
            
        return commit_groups

    def _determine_commit_type(self, changes: List[GitChange]) -> CommitType:
        """Determine the primary commit type based on changes.
        
        Rules:
        - If any change is a feature (new file, significant code), use FEAT
        - If only fixes (small modifications), use FIX
        - If only docs changes, use DOCS
        - If only tests, use TEST
        - Otherwise, default to CHORE
        """
        has_feature = False
        has_fix = False
        has_docs = False
        has_test = False
        
        for change in changes:
            # Simple heuristic for feature vs fix
            if change.status == 'added' or (change.diff and len(change.diff.splitlines()) > 20):
                has_feature = True
            elif change.status == 'modified':
                has_fix = True
                
            # Check for docs/tests based on component/path
            if change.component == 'docs':
                has_docs = True
            if change.component == 'tests':
                has_test = True
                
        # Determine type based on priority
        if has_feature: return CommitType.FEAT
        if has_fix: return CommitType.FIX
        if has_docs: return CommitType.DOCS
        if has_test: return CommitType.TEST
        return CommitType.CHORE

    def _extract_issues(self, changes: List[GitChange]) -> Set[str]:
        """Extract issue references (e.g., #123) from change diffs or messages."""
        issues = set()
        issue_pattern = r'#\d+'
        
        for change in changes:
            # Look for issues in diff content
            if change.diff:
                matches = re.findall(issue_pattern, change.diff)
                issues.update(matches)
            
            # Add logic to look in commit messages if needed (not implemented here)
        
        return issues

    def _generate_ai_commit_message(self, group: CommitGroup) -> str:
        """Generate a commit message using AI or rule-based approaches."""
        if not self.ai_client:
            return str(group)
            
        # Prepare the prompt
        prompt = f"Generate a commit message for the following changes:\n\n"
        for change in group.changes:
            prompt += f"- {change.filename} ({change.status})\n"
            if change.diff:
                prompt += f"Diff:\n{change.diff}\n\n"
                
        # Add context
        prompt += f"\nCommit type: {group.commit_type}\n"
        prompt += f"Component: {group.component}\n"
        if group.issues:
            prompt += f"Related issues: {', '.join(sorted(group.issues))}\n"
            
        # Add instructions
        system_prompt = """Generate a concise and descriptive commit message following the Conventional Commits format.
The first line should be a summary in the format: <type>[(scope)]: <description>
Add a blank line followed by more details if needed.
Focus on the WHAT and WHY, not the HOW.
Keep the first line under 72 characters.
Use imperative mood ("Add feature" not "Added feature").
"""
        
        # Generate the message
        message = self.ai_client.generate(prompt, system_prompt)
        if not message:
            return str(group)
            
        return message

    def generate_commit_message(self, group: CommitGroup) -> str:
        """Generate a commit message for a group of changes."""
        if self.use_ai and self.ai_client:
            return self._generate_ai_commit_message(group)
        else:
            return str(group)

    def commit_changes(self, group: CommitGroup, message: str) -> bool:
        """Commit a group of changes with the given message."""
        try:
            # Stage the files
            for change in group.changes:
                subprocess.run(
                    ['git', 'add', change.filename],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )
                
            # Create the commit
            subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Failed to commit changes: {str(e)}")
            return False 