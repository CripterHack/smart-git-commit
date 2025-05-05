"""
Git hook execution script for Smart Git Commit.

This script is typically invoked by Git hooks (e.g., commit-msg).
"""

import os
import re
import sys
import logging
import platform
import argparse
import subprocess
import stat
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple

# Configure logging
# Use a basic config if running as a standalone script might not have app logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("smart_git_commit.hook_script")

# Helper function to find git root (copied/adapted from hooks.py/utils for standalone use)
def get_git_root(path: str = ".") -> Optional[str]:
    """
    Find the Git root directory.
    """
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        logger.error("Git command not found. Ensure Git is installed and in PATH.")
        return None
    except Exception as e:
        logger.error(f"Error finding git root: {e}")
        return None

def find_git_dir(path: str = ".") -> Optional[str]:
    """Find the .git directory for a repository."""
    git_root = get_git_root(path)
    if not git_root:
        return None
    return os.path.join(git_root, ".git")

def get_commit_msg_path(git_dir: str = None) -> Optional[str]:
    """Get the path to the commit message file from args or default."""
    # Try getting path from command line arguments first (common for commit-msg hook)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return os.path.abspath(sys.argv[1])

    # Fallback to default path if not provided via args
    if not git_dir:
        git_dir = find_git_dir()

    if not git_dir:
        logger.warning("Could not find .git directory.")
        return None

    commit_msg_file = os.path.join(git_dir, "COMMIT_EDITMSG")
    if not os.path.exists(commit_msg_file):
         logger.warning(f"Default commit message file not found: {commit_msg_file}")
         return None

    return commit_msg_file

def get_commit_msg_from_file(file_path: str) -> Optional[str]:
    """Read commit message from a file."""
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Commit message file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        logger.error(f"Error reading commit message file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading {file_path}: {e}")
        return None


def write_commit_msg_to_file(file_path: str, message: str) -> bool:
    """Write commit message to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(message)
        return True
    except IOError as e:
        logger.error(f"Error writing commit message file {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error writing {file_path}: {e}")
        return False

def get_staged_diff(repo_path: str = ".") -> str:
    """Get staged Git diff for the repository."""
    git_root = get_git_root(repo_path)
    if not git_root:
        logger.error("Not inside a git repository.")
        return ""
    try:
        # Use -C <git_root> to ensure running from the root
        result = subprocess.run(
            ["git", "-C", git_root, "diff", "--staged"],
            capture_output=True,
            text=True, # Ensure text mode
            encoding='utf-8', # Specify encoding
            check=False # Don't check=True, empty diff is not an error
        )
        if result.returncode != 0:
             # Log stderr if there was an error
             logger.error(f"Failed to get staged diff: {result.stderr}")
             return ""
        return result.stdout
    except FileNotFoundError:
        logger.error("Git command not found.")
        return ""
    except Exception as e:
        logger.error(f"Error getting staged diff: {e}")
        return ""

def extract_commit_message(content: str) -> str:
    """Extract commit message from content, removing comments."""
    if not content:
        return ""
    lines = content.split('\n')
    result = []
    for line in lines:
        if not line.strip().startswith('#'):
            result.append(line)
    # Ensure trailing newline is preserved if present, then strip whitespace
    processed_content = '\n'.join(result)
    if content.endswith('\n') and not processed_content.endswith('\n'):
         processed_content += '\n'
    return processed_content.rstrip() # Use rstrip to only remove trailing whitespace


def run_formatter(diff: str, message: str) -> str:
     """Run commit message formatter based on the diff."""
     # Placeholder: In a real scenario, this might call the AI processor
     # or apply rule-based formatting. Currently relies on processor call
     # within process_hook.
     logger.debug("run_formatter called (currently a placeholder)")
     return message # Return original message, formatting happens in process_hook

def process_hook(diff_or_file_path: Union[str, Path], message: str = None) -> Union[str, bool]:
    """
    Process a commit with diff and message, or from a file path.
    This is the core logic called by the commit-msg hook typically.
    Args:
        diff_or_file_path: Either the Git diff to analyze or a file path to the commit message
        message: Original commit message (optional, only used when diff_or_file_path is a diff)

    Returns:
        Processed commit message as string (if diff/message passed directly)
        Boolean success flag (if file_path passed)
    """
    # Handle case where a file path is provided (typical for commit-msg hook)
    if isinstance(diff_or_file_path, (str, Path)) and message is None and os.path.exists(diff_or_file_path):
        file_path = str(diff_or_file_path)
        logger.info(f"Processing commit message file: {file_path}")
        # Get the diff
        diff = get_staged_diff()
        # Get the message from file
        original_message = get_commit_msg_from_file(file_path)

        if original_message is None:
             logger.error("Could not read original commit message.")
             return False # Indicate failure

        # Skip processing if no diff (unless message is empty, maybe generate?)
        # Let processor decide if empty message + diff is processable
        if not diff and original_message.strip():
            logger.info("No staged changes detected, skipping processing.")
            return True # Indicate success (no action needed)

        try:
            # Import processor dynamically to avoid making this script dependent
            # on the full app structure if possible, and avoid circular imports.
            # This assumes the package `smart_git_commit` is installed.
            try:
                 from smart_git_commit.processor import get_processor
            except ImportError:
                 logger.error("Failed to import smart_git_commit.processor. Is the package installed?")
                 return False

            processor = get_processor() # This might raise ValueError if config is bad

            # Process the message using the processor directly
            logger.info("Invoking processor...")
            processed_message = processor.process(diff, original_message or '') # Pass empty string if None

            if processed_message is None:
                 logger.error("Processor failed to return a message.")
                 return False

            # Write back to file if changed
            # Use extract_commit_message to compare meaningfully (ignore comments)
            cleaned_original = extract_commit_message(original_message)
            cleaned_processed = extract_commit_message(processed_message)

            if cleaned_processed != cleaned_original:
                logger.info("Commit message updated by processor.")
                success = write_commit_msg_to_file(file_path, processed_message)
                return success
            else:
                logger.info("No changes made to commit message by processor.")
                return True # Indicate success (no action needed)

        except Exception as e:
            logger.error(f"Error processing hook: {e}", exc_info=True)
            # Optionally, write the error to the commit msg file?
            # write_commit_msg_to_file(file_path, f"# Error processing commit: {e}\n{original_message}")
            return False # Indicate failure

    # Handle case where diff and message are provided explicitly (less common for hooks)
    elif isinstance(diff_or_file_path, str) and isinstance(message, str):
        diff = diff_or_file_path
        logger.debug("Processing hook with explicit diff and message.")

        # Skip processing if no diff or message? (Processor might handle empty message)
        if not diff:
             logger.debug("No diff provided, returning original message.")
             return message

        try:
            # This path might not need the full processor, maybe just formatting?
            # Currently calls run_formatter placeholder.
            processed_message = extract_commit_message(message or '')
            processed_message = run_formatter(diff, processed_message)
            return processed_message
        except Exception as e:
            logger.error(f"Error processing hook with explicit args: {e}", exc_info=True)
            return message # Return original on error
    else:
         logger.error(f"Invalid arguments passed to process_hook: {type(diff_or_file_path)}, {type(message)}")
         # Depending on context, return False or raise error
         if message is None: return False # Assume file path context
         else: return message # Assume explicit context

def main():
    """
    Main entry point for the hook script.
    Determines the hook type from the script name or args
    and executes the corresponding logic.
    """
    # Determine hook type (e.g., from script name 'commit-msg')
    script_name = os.path.basename(sys.argv[0])
    hook_type = script_name

    # Simple argument parsing for the hook script itself
    parser = argparse.ArgumentParser(
         description=f"Smart Git Commit Git Hook ({hook_type})."
    )
    # The primary argument for commit-msg hook is the file path
    parser.add_argument(
        'commit_msg_file',
        nargs='?', # Make it optional in case path is passed directly
        help='Path to the commit message file (provided by Git for commit-msg hook).'
    )

    args, remaining_args = parser.parse_known_args()

    commit_msg_file_path = args.commit_msg_file or (sys.argv[1] if len(sys.argv) > 1 else None)

    logger.info(f"Running hook: {hook_type}")
    if commit_msg_file_path:
        logger.info(f"Commit message file: {commit_msg_file_path}")
    else:
        logger.warning("No commit message file path provided.")

    exit_code = 1 # Default to failure

    # --- Hook Execution Logic ---
    if hook_type == "commit-msg":
        if not commit_msg_file_path:
            logger.error("Commit message file path is required for commit-msg hook.")
        elif not os.path.exists(commit_msg_file_path):
             logger.error(f"Commit message file not found: {commit_msg_file_path}")
        else:
            # Core processing logic
            success = process_hook(commit_msg_file_path)
            exit_code = 0 if success else 1

    elif hook_type == "pre-commit":
        # Add pre-commit logic here if needed (e.g., run linters based on staged files)
        logger.info("Pre-commit hook execution (placeholder).")
        exit_code = 0 # Success by default for placeholder

    # Add other hook types (pre-push, etc.) here if needed
    # elif hook_type == "pre-push":
    #     logger.info("Pre-push hook execution (placeholder).")
    #     exit_code = 0

    else:
        logger.warning(f"Hook type '{hook_type}' not explicitly handled by this script.")
        exit_code = 0 # Don't fail for unhandled hooks by default

    logger.info(f"Hook '{hook_type}' finished with exit code {exit_code}.")
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 