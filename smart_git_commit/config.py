"""
Configuration management for Smart Git Commit.

This module provides functionality to read, write, and manage configuration settings
for Smart Git Commit, supporting persistent storage of user preferences.
"""

import os
import json
import logging
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set

logger = logging.getLogger(__name__)

# Default configuration directory
DEFAULT_CONFIG_DIR = os.path.expanduser("~/.config/smart-git-commit")
DEFAULT_CONFIG_FILE = "config.json"

# Default configuration values
DEFAULT_CONFIG = {
    "theme": "standard",
    "security_scan": True,
    "timeout": 60,
    "use_ai": True,
    "parallel": True,
    "skip_hooks": False,
    "ollama_host": "http://localhost:11434",
    "ollama_model": None,
    "welcome_completed": False,
    "commit_templates": {
        "default": {
            "subject_template": "{type}({scope}): {description}",
            "body_template": "{body}\n\nAffected files:\n{files}",
            "footer_template": "{issues}"
        },
        "conventional": {
            "subject_template": "{type}({scope}): {description}",
            "body_template": "{body}",
            "footer_template": "BREAKING CHANGE: {breaking_change}\n\n{issues}"
        },
        "detailed": {
            "subject_template": "{type}: {description}",
            "body_template": "Component: {scope}\n\n{body}\n\nFiles changed:\n{files}",
            "footer_template": "Resolves: {issues}"
        }
    },
    "active_template": "default",
    "ai_provider": "ollama",  # or "openai"
    "templates": {
        "conventional": {
            "subject_template": "{type}({scope}): {description}",
            "body_template": "{body}\n\nAffected files:\n{files}",
            "footer_template": "{issues}\n{breaking_change}"
        },
        "detailed": {
            "subject_template": "{type}({scope}): {description}",
            "body_template": "{body}\n\nChanges:\n{files}",
            "footer_template": "Issues: {issues}\n{breaking_change}"
        },
        "simple": {
            "subject_template": "{description}",
            "body_template": "{body}",
            "footer_template": "{issues}"
        }
    },
    "default_model": "llama3"
}

class Configuration:
    """
    Handles the configuration settings for Smart Git Commit.
    
    This class provides methods to:
    1. Load configuration from file
    2. Save configuration to file
    3. Access and modify configuration settings
    4. Reset configuration to defaults
    """
    
    def __init__(self, config_dir: Optional[str] = None, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
            config_file: Name of the configuration file
        """
        self.config_dir = config_dir or DEFAULT_CONFIG_DIR
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self.config_path = os.path.join(self.config_dir, self.config_file)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.load()
    
    def load(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if configuration was loaded successfully, False otherwise
        """
        if not os.path.exists(self.config_path):
            logger.info(f"Configuration file not found at {self.config_path}, using defaults")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                
            # Merge with defaults to ensure new options are included
            self._merge_config(loaded_config)
            logger.info(f"Successfully loaded configuration from {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def _merge_config(self, loaded_config: Dict[str, Any]) -> None:
        """
        Merge loaded configuration with defaults.
        
        Args:
            loaded_config: Configuration loaded from file
        """
        def deep_merge(default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge loaded config into default config."""
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        self.config = deep_merge(DEFAULT_CONFIG, loaded_config)
    
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if configuration was saved successfully, False otherwise
        """
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Successfully saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        # Support nested keys with dot notation
        if "." in key:
            parts = key.split(".")
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        # Support nested keys with dot notation
        if "." in key:
            parts = key.split(".")
            config = self.config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                elif not isinstance(config[part], dict):
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            self.config[key] = value
    
    def reset(self) -> None:
        """Reset configuration to defaults using a deep copy."""
        self.config = copy.deepcopy(DEFAULT_CONFIG)
    
    def add_commit_template(self, name: str, 
                           subject_template: str, 
                           body_template: str, 
                           footer_template: str) -> None:
        """
        Add a new commit template.
        
        Args:
            name: Template name
            subject_template: Subject line template
            body_template: Commit body template
            footer_template: Commit footer template
        """
        if "commit_templates" not in self.config:
            self.config["commit_templates"] = {}
        
        self.config["commit_templates"][name] = {
            "subject_template": subject_template,
            "body_template": body_template,
            "footer_template": footer_template
        }
    
    def remove_commit_template(self, name: str) -> bool:
        """
        Remove a commit template.
        
        Args:
            name: Template name
            
        Returns:
            True if template was removed, False otherwise
        """
        if (name in self.config.get("commit_templates", {}) and 
            name != "default"):  # Prevent removal of default template
            del self.config["commit_templates"][name]
            
            # If the active template was removed, revert to default
            if self.config.get("active_template") == name:
                self.config["active_template"] = "default"
                
            return True
        return False
    
    def get_commit_template(self, name: Optional[str] = None) -> Dict[str, str]:
        """
        Get a commit template.
        
        Args:
            name: Template name, defaults to active template
            
        Returns:
            Commit template as a dictionary
        """
        if name is None:
            name = self.config.get("active_template", "default")
        
        templates = self.config.get("commit_templates", {})
        if name in templates:
            return templates[name]
        return templates.get("default", DEFAULT_CONFIG["commit_templates"]["default"])
    
    def get_commit_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Get all commit templates.
        
        Returns:
            Dictionary of commit templates
        """
        # Use copy.deepcopy to avoid returning a reference to DEFAULT_CONFIG
        default_templates = copy.deepcopy(DEFAULT_CONFIG["commit_templates"])
        return self.config.get("commit_templates", default_templates)
    
    def set_active_template(self, name: str) -> bool:
        """
        Set the active commit template.
        
        Args:
            name: Template name
            
        Returns:
            True if template was set as active, False otherwise
        """
        if name in self.config.get("commit_templates", {}):
            self.config["active_template"] = name
            return True
        return False


# Global configuration instance
_config_instance = None

def get_config(reload: bool = False) -> Configuration:
    """
    Get the configuration instance.
    
    Args:
        reload: Whether to reload the configuration from file
        
    Returns:
        Configuration instance
    """
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = Configuration()
    return _config_instance 