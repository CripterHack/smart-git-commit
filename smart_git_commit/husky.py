"""
Husky integration for Smart Git Commit.

This module provides functionality to integrate Husky for Git hooks,
supporting cross-platform pre-commit hooks.
"""

import os
import sys
import json
import logging
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger("smart_git_commit.husky")

class HuskyIntegration:
    """
    Class to handle Husky integration for Git hooks.
    
    This class provides functionality to:
    1. Set up Husky in a Git repository
    2. Create pre-commit hooks
    3. Configure hooks to run smart-git-commit
    """
    
    def __init__(self, repo_path: str = "."):
        """
        Initialize the Husky integration.
        
        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = os.path.abspath(repo_path)
        self.git_root = self._get_git_root(self.repo_path)
        
        if not self.git_root:
            logger.error(f"No Git repository found at {self.repo_path}")
            raise ValueError(f"No Git repository found at {self.repo_path}")
        
        logger.info(f"Initialized HuskyIntegration for Git repository at {self.git_root}")
        
    def _get_git_root(self, path: str) -> Optional[str]:
        """
        Find the root directory of the Git repository.
        
        Args:
            path: Path to start searching from
            
        Returns:
            Path to the Git root directory or None if not found
        """
        try:
            process = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if process.returncode == 0:
                git_root = process.stdout.strip()
                logger.info(f"Found Git root at {git_root}")
                return git_root
            else:
                logger.warning(f"Failed to find Git root: {process.stderr.strip()}")
                return None
        except Exception as e:
            logger.error(f"Error finding Git root: {str(e)}")
            return None
    
    def _check_node_installed(self) -> bool:
        """
        Check if Node.js is installed.
        
        Returns:
            True if Node.js is installed, False otherwise
        """
        try:
            process = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if process.returncode == 0:
                version = process.stdout.strip()
                logger.info(f"Node.js is installed: {version}")
                return True
            else:
                logger.warning("Node.js is not installed")
                return False
        except Exception:
            logger.warning("Failed to check if Node.js is installed")
            return False
    
    def _check_npm_installed(self) -> bool:
        """
        Check if npm is installed.
        
        Returns:
            True if npm is installed, False otherwise
        """
        try:
            process = subprocess.run(
                ["npm", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if process.returncode == 0:
                version = process.stdout.strip()
                logger.info(f"npm is installed: {version}")
                return True
            else:
                logger.warning("npm is not installed")
                return False
        except Exception:
            logger.warning("Failed to check if npm is installed")
            return False
    
    def _update_package_json(self) -> bool:
        """
        Update or create package.json file with Husky configuration.
        
        Returns:
            True if successful, False otherwise
        """
        package_json_path = os.path.join(self.git_root, "package.json")
        
        try:
            # Create default package.json content if it doesn't exist
            if not os.path.exists(package_json_path):
                logger.info("Creating new package.json file")
                default_package = {
                    "name": os.path.basename(self.git_root),
                    "version": "1.0.0",
                    "description": "Git repository with Husky hooks",
                    "private": True,
                    "scripts": {
                        "prepare": "husky install"
                    },
                    "devDependencies": {
                        "husky": "^8.0.0"
                    }
                }
                
                with open(package_json_path, 'w', encoding='utf-8') as f:
                    json.dump(default_package, f, indent=2)
                
                logger.info("Created package.json file")
                return True
            
            # Update existing package.json
            with open(package_json_path, 'r', encoding='utf-8') as f:
                try:
                    package_data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in {package_json_path}")
                    return False
            
            # Add husky to devDependencies if it doesn't exist
            if "devDependencies" not in package_data:
                package_data["devDependencies"] = {}
            
            package_data["devDependencies"]["husky"] = "^8.0.0"
            
            # Add prepare script if it doesn't exist
            if "scripts" not in package_data:
                package_data["scripts"] = {}
            
            package_data["scripts"]["prepare"] = "husky install"
            
            # Write updated package.json
            with open(package_json_path, 'w', encoding='utf-8') as f:
                json.dump(package_data, f, indent=2)
            
            logger.info(f"Updated {package_json_path} with Husky configuration")
            return True
            
        except Exception as e:
            logger.error(f"Error updating package.json: {str(e)}")
            return False
    
    def _create_husky_dir(self) -> bool:
        """
        Create the .husky directory and necessary files.
        
        Returns:
            True if successful, False otherwise
        """
        husky_dir = os.path.join(self.git_root, ".husky")
        
        try:
            # Create .husky directory if it doesn't exist
            if not os.path.exists(husky_dir):
                os.makedirs(husky_dir, exist_ok=True)
                logger.info(f"Created .husky directory at {husky_dir}")
            
            # Create .gitignore to prevent hooks from being tracked
            gitignore_path = os.path.join(husky_dir, ".gitignore")
            if not os.path.exists(gitignore_path):
                with open(gitignore_path, 'w', encoding='utf-8') as f:
                    f.write("_\n")  # Ignore internal husky directory
                logger.info(f"Created {gitignore_path}")
            
            # Create husky.sh script
            husky_sh_path = os.path.join(husky_dir, "husky.sh")
            if not os.path.exists(husky_sh_path):
                with open(husky_sh_path, 'w', encoding='utf-8') as f:
                    f.write("""#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"
""")
                # Make script executable
                os.chmod(husky_sh_path, 0o755)
                logger.info(f"Created {husky_sh_path}")
            
            # Create _husky directory for internal use
            husky_internal_dir = os.path.join(husky_dir, "_")
            if not os.path.exists(husky_internal_dir):
                os.makedirs(husky_internal_dir, exist_ok=True)
                
                # Create husky-sh helper script
                husky_helper_path = os.path.join(husky_internal_dir, "husky.sh")
                with open(husky_helper_path, 'w', encoding='utf-8') as f:
                    f.write("""#!/usr/bin/env sh
if [ -z "$husky_skip_init" ]; then
  debug () {
    if [ "$HUSKY_DEBUG" = "1" ]; then
      echo "husky (debug) - $1"
    fi
  }

  readonly hook_name="$(basename -- "$0")"
  debug "starting $hook_name..."

  if [ "$HUSKY" = "0" ]; then
    debug "HUSKY env variable is set to 0, skipping hook"
    exit 0
  fi

  if [ -f ~/.huskyrc ]; then
    debug "sourcing ~/.huskyrc"
    . ~/.huskyrc
  fi

  readonly husky_skip_init=1
  export husky_skip_init
  sh -e "$0" "$@"
  exitCode="$?"

  if [ $exitCode != 0 ]; then
    echo "husky - $hook_name hook exited with code $exitCode (error)"
  fi

  exit $exitCode
fi
""")
                os.chmod(husky_helper_path, 0o755)
                logger.info(f"Created {husky_helper_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating Husky directory: {str(e)}")
            return False
    
    def create_hook(self, hook_name: str, command: str) -> bool:
        """
        Create a Git hook using Husky.
        
        Args:
            hook_name: Name of the hook (e.g., pre-commit, pre-push)
            command: Command to run in the hook
            
        Returns:
            True if successful, False otherwise
        """
        hook_path = os.path.join(self.git_root, ".husky", hook_name)
        
        try:
            with open(hook_path, 'w', encoding='utf-8') as f:
                f.write(f"""#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

{command}
""")
            
            # Make hook executable
            os.chmod(hook_path, 0o755)
            logger.info(f"Created {hook_name} hook at {hook_path}")
            
            # Make sure .git/hooks is not corrupted
            git_hooks_dir = os.path.join(self.git_root, ".git", "hooks")
            husky_hook_path = os.path.join(git_hooks_dir, hook_name)
            
            # If the hook already exists but is not a husky hook, back it up
            if os.path.exists(husky_hook_path) and os.path.isfile(husky_hook_path):
                with open(husky_hook_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "husky.sh" not in content:
                        backup_path = f"{husky_hook_path}.backup.{int(os.path.getmtime(husky_hook_path))}"
                        shutil.copy2(husky_hook_path, backup_path)
                        logger.info(f"Backed up existing {hook_name} hook to {backup_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating {hook_name} hook: {str(e)}")
            return False
    
    def install_husky_npm(self) -> bool:
        """
        Install Husky using npm.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._check_node_installed() or not self._check_npm_installed():
            logger.warning("Node.js or npm is not installed, cannot install Husky via npm")
            return False
        
        try:
            # Update package.json
            if not self._update_package_json():
                return False
            
            # Run npm install to install Husky
            logger.info("Installing Husky via npm")
            process = subprocess.run(
                ["npm", "install"],
                cwd=self.git_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"Failed to install Husky: {process.stderr}")
                return False
            
            # Initialize Husky
            process = subprocess.run(
                ["npx", "husky", "install"],
                cwd=self.git_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"Failed to initialize Husky: {process.stderr}")
                return False
            
            logger.info("Successfully installed and initialized Husky")
            return True
            
        except Exception as e:
            logger.error(f"Error installing Husky: {str(e)}")
            return False
    
    def install_husky_manual(self) -> bool:
        """
        Install Husky manually (without npm).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create .husky directory and files
            if not self._create_husky_dir():
                return False
            
            logger.info("Successfully set up Husky manually")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Husky manually: {str(e)}")
            return False
    
    def setup_pre_commit(self, command: str) -> bool:
        """
        Set up a pre-commit hook with the specified command.
        
        Args:
            command: Command to run in the pre-commit hook
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure Husky is installed
        if not os.path.exists(os.path.join(self.git_root, ".husky")):
            if self._check_node_installed() and self._check_npm_installed():
                if not self.install_husky_npm():
                    if not self.install_husky_manual():
                        logger.error("Failed to install Husky")
                        return False
            else:
                if not self.install_husky_manual():
                    logger.error("Failed to install Husky manually")
                    return False
        
        # Create pre-commit hook
        return self.create_hook("pre-commit", command)
    
    def remove_hook(self, hook_name: str) -> bool:
        """
        Remove a specific Git hook.
        
        Args:
            hook_name: Name of the hook to remove
            
        Returns:
            True if successful, False otherwise
        """
        hook_path = os.path.join(self.git_root, ".husky", hook_name)
        
        try:
            if os.path.exists(hook_path):
                os.remove(hook_path)
                logger.info(f"Removed {hook_name} hook at {hook_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing {hook_name} hook: {str(e)}")
            return False
    
    def uninstall_husky(self) -> bool:
        """
        Uninstall Husky completely.
        
        Returns:
            True if successful, False otherwise
        """
        husky_dir = os.path.join(self.git_root, ".husky")
        
        try:
            if os.path.exists(husky_dir):
                shutil.rmtree(husky_dir)
                logger.info(f"Removed Husky directory at {husky_dir}")
            
            # Remove husky from package.json if it exists
            package_json_path = os.path.join(self.git_root, "package.json")
            if os.path.exists(package_json_path):
                try:
                    with open(package_json_path, 'r', encoding='utf-8') as f:
                        package_data = json.load(f)
                    
                    # Remove husky from devDependencies
                    if "devDependencies" in package_data and "husky" in package_data["devDependencies"]:
                        del package_data["devDependencies"]["husky"]
                        logger.info("Removed Husky from devDependencies in package.json")
                    
                    # Remove prepare script if it's for husky
                    if "scripts" in package_data and "prepare" in package_data["scripts"]:
                        if "husky install" in package_data["scripts"]["prepare"]:
                            del package_data["scripts"]["prepare"]
                            logger.info("Removed Husky prepare script from package.json")
                    
                    # Write updated package.json
                    with open(package_json_path, 'w', encoding='utf-8') as f:
                        json.dump(package_data, f, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error updating package.json during uninstall: {str(e)}")
            
            logger.info("Successfully uninstalled Husky")
            return True
            
        except Exception as e:
            logger.error(f"Error uninstalling Husky: {str(e)}")
            return False
            
    def get_hook_path(self, hook_name: str) -> str:
        """
        Get the path to a specific hook file.
        
        Args:
            hook_name: Name of the hook
            
        Returns:
            Path to the hook file
        """
        return os.path.join(self.git_root, ".husky", hook_name)
    
    def list_hooks(self) -> List[str]:
        """
        List all installed hooks.
        
        Returns:
            List of hook names
        """
        husky_dir = os.path.join(self.git_root, ".husky")
        
        if not os.path.exists(husky_dir):
            return []
        
        hooks = []
        for item in os.listdir(husky_dir):
            item_path = os.path.join(husky_dir, item)
            # Ignore directories and special files
            if not os.path.isfile(item_path) or item.startswith(".") or item == "husky.sh":
                continue
            hooks.append(item)
        
        return hooks
    
    def is_hook_installed(self, hook_name: str) -> bool:
        """
        Check if a specific hook is installed.
        
        Args:
            hook_name: Name of the hook
            
        Returns:
            True if the hook is installed, False otherwise
        """
        hook_path = self.get_hook_path(hook_name)
        return os.path.exists(hook_path) and os.path.isfile(hook_path)
    
    def is_husky_installed(self) -> bool:
        """
        Check if Husky is installed.
        
        Returns:
            True if Husky is installed, False otherwise
        """
        husky_dir = os.path.join(self.git_root, ".husky")
        husky_internal_dir = os.path.join(husky_dir, "_")
        
        return (
            os.path.exists(husky_dir) and 
            os.path.isdir(husky_dir) and 
            os.path.exists(husky_internal_dir) and
            os.path.isdir(husky_internal_dir)
        )


def setup_smart_git_commit_hook(repo_path: str = ".", 
                                command: Optional[str] = None,
                                python_path: Optional[str] = None) -> bool:
    """
    Set up Smart Git Commit as a pre-commit hook using Husky.
    
    Args:
        repo_path: Path to the Git repository
        command: Custom command to run in the pre-commit hook (optional)
        python_path: Path to the Python executable (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        husky = HuskyIntegration(repo_path)
        
        # Determine the command to run
        if command is None:
            # Find the path to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            
            # Determine the Python executable to use
            python_exec = python_path or sys.executable
            
            if platform.system() == "Windows":
                # On Windows, use cmd.exe syntax
                command = f'{python_exec} "{os.path.join(parent_dir, "smart_git_commit.py")}" pre-commit'
            else:
                # On Unix-like systems, use bash syntax
                command = f'{python_exec} "{os.path.join(parent_dir, "smart_git_commit.py")}" pre-commit'
        
        # Set up the pre-commit hook
        success = husky.setup_pre_commit(command)
        
        if success:
            logger.info("Successfully set up Smart Git Commit as a pre-commit hook")
        else:
            logger.error("Failed to set up Smart Git Commit as a pre-commit hook")
        
        return success
        
    except Exception as e:
        logger.error(f"Error setting up pre-commit hook: {str(e)}")
        return False


if __name__ == "__main__":
    # Configure logging for direct script execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Husky integration for Smart Git Commit")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up Husky and pre-commit hook")
    setup_parser.add_argument("--repo", default=".", help="Path to Git repository")
    setup_parser.add_argument("--command", help="Custom command for pre-commit hook")
    setup_parser.add_argument("--python", help="Path to Python executable")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove pre-commit hook")
    remove_parser.add_argument("--repo", default=".", help="Path to Git repository")
    
    # Uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall Husky completely")
    uninstall_parser.add_argument("--repo", default=".", help="Path to Git repository")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List installed hooks")
    list_parser.add_argument("--repo", default=".", help="Path to Git repository")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_smart_git_commit_hook(args.repo, args.command, args.python)
    elif args.command == "remove":
        husky = HuskyIntegration(args.repo)
        husky.remove_hook("pre-commit")
    elif args.command == "uninstall":
        husky = HuskyIntegration(args.repo)
        husky.uninstall_husky()
    elif args.command == "list":
        husky = HuskyIntegration(args.repo)
        hooks = husky.list_hooks()
        print(f"Installed hooks: {', '.join(hooks) if hooks else 'None'}")
    else:
        parser.print_help() 