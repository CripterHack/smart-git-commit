"""
Smart Git Commit - An AI-powered Git commit workflow tool.
"""

# Define version function here to avoid circular imports
def get_version():
    """Get the version number of the package."""
    # Version should be read from setup.py or a single source of truth
    # For now, keeping it hardcoded as per original, but ideally fetch dynamically
    return "0.3.6"

# Import core functionality for compatibility & public API
# Group imports by module for clarity

# Main workflow and core data structures
from .smart_git_commit import (
    main,  # Main entry point
    SmartGitCommitWorkflow,
    GitChange,
    SecurityScanner,
    Spinner,
    SecurityVulnerability,
    SecuritySeverity,
    SecurityScanResult,
    ChangeState,
)

# CLI Wizard
from .cli_wizard import run_cli_welcome_wizard

# Squash related functionality
from .squash import (
    SquashStrategy,
    find_squashable_commits,
    run_squash_command,
    analyze_commits_for_squashing
)

# Hooks related functionality
from .hooks import (
    run_hook,
    install_git_hooks,
    remove_git_hooks,
    install_husky,
    remove_husky,
    list_installed_hooks,
    check_husky_compatibility,
    GitHook,
    get_hook_path,
    get_hook_types,
    get_hook_script,
    validate_hook_type,
    install_hook,
    remove_hook,
    format_hook_script
)

# Other core components
# from .change import Change # Removed unused import
from .colors import Colors, supports_color
# from .command import main as command_main # Removed redundant import
from .commit_group import CommitGroup, CommitType
from .config import Configuration, get_config
from .git_utils import (
    get_repository_details,
    get_staged_files,
    get_git_root,
    get_git_hooks_dir,
    parse_status_line,
)
from .processor import (
    AIClient, OllamaClient, OpenAIClient, CommitProcessor,
    get_processor
)
from .tui import run_tui, SmartGitCommitApp


__version__ = get_version()

# Define public API for the package
__all__ = [
    # Main entry point
    'main',

    # Classes
    # 'Change', # Removed
    'Colors',
    'CommitGroup',
    'CommitProcessor',
    'CommitType',
    'Configuration',
    'GitChange', # Added from smart_git_commit
    'GitHook',
    'SecurityScanner',
    'SmartGitCommitWorkflow',
    'SquashStrategy',
    'AIClient',
    'OllamaClient', # Keep the one from processor
    'OpenAIClient',
    'Spinner', # Added from smart_git_commit
    'SmartGitCommitApp', # Added from tui
    'SecurityVulnerability', # Added
    'SecuritySeverity', # Added
    'SecurityScanResult', # Added
    'ChangeState', # Added

    # Functions - Config
    'get_config',

    # Functions - Git Utils
    'get_repository_details',
    'parse_status_line',
    'get_staged_files',
    'get_git_root',
    'get_git_hooks_dir',

    # Functions - Hooks
    'run_hook', # Added
    'install_git_hooks',
    'remove_git_hooks',
    'list_installed_hooks',
    'get_hook_types',
    'validate_hook_type',
    'install_husky',
    'remove_husky',
    'check_husky_compatibility',
    'get_hook_path', # Added
    'get_hook_script', # Added
    'install_hook', # Added
    'remove_hook', # Added
    'format_hook_script', # Added

    # Functions - Processor
    'get_processor',

    # Functions - Squash
    'find_squashable_commits',
    'analyze_commits_for_squashing',
    'run_squash_command',

    # Functions - TUI
    'run_tui',
    
    # Functions - CLI Wizard
    'run_cli_welcome_wizard',

    # Functions - Misc
    'get_version',
    '__version__',
    'supports_color',
] 