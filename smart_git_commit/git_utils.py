"""Git utility functions for smart-git-commit."""

import os
import subprocess
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

def get_repository_details(repo_path: str = ".") -> Dict[str, str]:
    """Get details about the git repository."""
    try:
        # Check if this is a git repository
        try:
            subprocess.check_output(["git", "rev-parse", "--git-dir"], cwd=repo_path, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return {
                "name": "unknown",
                "branch": "unknown",
                "path": os.path.abspath(repo_path)
            }
            
        # Get repository name from remote URL or directory name
        try:
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_path,
                text=True
            ).strip()
            repo_name = remote_url.split("/")[-1].replace(".git", "")
        except subprocess.CalledProcessError:
            repo_name = os.path.basename(os.path.abspath(repo_path))
        
        # Get current branch
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                text=True
            ).strip()
        except subprocess.CalledProcessError:
            branch = "unknown"
        
        return {
            "name": repo_name,
            "branch": branch,
            "path": os.path.abspath(repo_path)
        }
    except Exception as e:
        logger.error(f"Failed to get repository details: {str(e)}")
        return {
            "name": "unknown",
            "branch": "unknown",
            "path": os.path.abspath(repo_path)
        }

def parse_status_line(line: str) -> Tuple[str, str]:
    """Parse a git status --porcelain line into status and filename."""
    if not line:
        return "", ""
        
    # Extract status and filename
    status = line[:2].strip()
    filename = line[3:].strip()
    
    # Handle renamed files
    if status.startswith("R"):
        # For renamed files, the format is "R old_name -> new_name"
        filename = filename.split(" -> ")[1]
    
    # Handle quoted filenames
    if filename.startswith('"') and filename.endswith('"'):
        filename = filename[1:-1]
    
    return status, filename

def get_git_root(path: str = ".") -> Optional[str]:
    """Get the root directory of the git repository."""
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            text=True
        ).strip()
        return os.path.abspath(git_root)
    except subprocess.CalledProcessError:
        return None

def get_git_hooks_dir(repo_path: str = ".") -> Optional[str]:
    """Get the hooks directory of the git repository."""
    git_root = get_git_root(repo_path)
    if not git_root:
        return None
    
    hooks_dir = os.path.join(git_root, ".git", "hooks")
    return hooks_dir if os.path.isdir(hooks_dir) else None

def get_staged_files(repo_path: str = ".") -> Dict[str, str]:
    """Get a dictionary of staged files and their status."""
    try:
        # Use --porcelain format for more reliable parsing
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            text=True,
            encoding='utf-8'
        ).strip()
        
        staged_files = {}
        for line in output.split("\n"):
            if not line:
                continue
                
            # Status is the first 2 characters
            status = line[0:2]
            
            # Only process staged files (status code in first column)
            if status[0] in ('M', 'A', 'D', 'R', 'C'):
                filename = line[3:]
                
                # Handle renamed files
                if status[0] == 'R':
                    # Split "old -> new" format
                    if " -> " in filename:
                        old_name, new_name = filename.split(" -> ", 1)
                        filename = new_name
                
                # Remove quotes from filename if present
                if filename.startswith('"') and filename.endswith('"'):
                    filename = filename[1:-1]
                
                staged_files[filename] = status[0]
        
        return staged_files
    except subprocess.CalledProcessError:
        return {} 