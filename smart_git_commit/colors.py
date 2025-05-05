"""
Color definitions and theme management for terminal output.
"""

import os
import sys
import platform


class Colors:
    """Custom colors for different themes using ANSI color codes."""
    
    # ANSI color codes
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    BRIGHT_BLACK = "\033[0;90m"
    BRIGHT_RED = "\033[0;91m"
    BRIGHT_GREEN = "\033[0;92m"
    BRIGHT_YELLOW = "\033[0;93m"
    BRIGHT_BLUE = "\033[0;94m"
    BRIGHT_MAGENTA = "\033[0;95m"
    BRIGHT_CYAN = "\033[0;96m"
    BRIGHT_WHITE = "\033[0;97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    # Theme-specific color constants for backward compatibility
    # These might be redundant now with the getter methods
    CYBER_BLUE = BRIGHT_MAGENTA
    CYBER_PINK = BRIGHT_CYAN
    DRACULA_PURPLE = MAGENTA
    DRACULA_PINK = BRIGHT_BLUE
    DRACULA_RED = RED
    NORD_BLUE = BLUE
    NORD_CYAN = CYAN
    NORD_RED = RED
    
    # Current theme
    THEME = "standard"
    
    @classmethod
    def set_theme(cls, theme_name):
        """Set the current theme."""
        cls.THEME = theme_name.lower()
    
    @classmethod
    def get_primary(cls):
        """Get the primary theme color."""
        try:
            if cls.THEME == "cyberpunk":
                return cls.BRIGHT_MAGENTA
            elif cls.THEME == "dracula":
                return cls.MAGENTA
            elif cls.THEME == "nord":
                return cls.BLUE
            elif cls.THEME == "monokai":
                return cls.YELLOW
            else:
                return cls.BLUE
        except (AttributeError, TypeError):
            return ""
    
    @classmethod
    def get_secondary(cls):
        """Get the secondary theme color."""
        try:
            if cls.THEME == "cyberpunk":
                return cls.BRIGHT_CYAN
            elif cls.THEME == "dracula":
                return cls.BRIGHT_BLUE
            elif cls.THEME == "nord":
                return cls.CYAN
            elif cls.THEME == "monokai":
                return cls.MAGENTA
            else:
                return cls.GREEN
        except (AttributeError, TypeError):
            return ""
    
    @classmethod
    def get_success(cls):
        """Get the success theme color."""
        try:
            if cls.THEME == "cyberpunk":
                return cls.BRIGHT_GREEN
            elif cls.THEME == "dracula":
                return cls.GREEN
            elif cls.THEME == "nord":
                return cls.GREEN
            elif cls.THEME == "monokai":
                return cls.GREEN
            else:
                return cls.GREEN
        except (AttributeError, TypeError):
            return ""
    
    @classmethod
    def get_warning(cls):
        """Get the warning theme color."""
        try:
            if cls.THEME == "cyberpunk":
                return cls.BRIGHT_YELLOW
            elif cls.THEME == "dracula":
                return cls.YELLOW
            elif cls.THEME == "nord":
                return cls.YELLOW
            elif cls.THEME == "monokai":
                return cls.BRIGHT_YELLOW
            else:
                return cls.YELLOW
        except (AttributeError, TypeError):
            return ""
    
    @classmethod
    def get_error(cls):
        """Get the error theme color."""
        try:
            if cls.THEME == "cyberpunk":
                return cls.BRIGHT_RED
            elif cls.THEME == "dracula":
                return cls.RED
            elif cls.THEME == "nord":
                return cls.RED
            elif cls.THEME == "monokai":
                return cls.RED
            else:
                return cls.RED
        except (AttributeError, TypeError):
            return ""
    
    @classmethod
    def get_info(cls):
        """Get the info theme color."""
        try:
            if cls.THEME == "cyberpunk":
                return cls.BRIGHT_BLUE
            elif cls.THEME == "dracula":
                return cls.CYAN
            elif cls.THEME == "nord":
                return cls.BRIGHT_BLUE
            elif cls.THEME == "monokai":
                return cls.CYAN
            else:
                return cls.BRIGHT_WHITE
        except (AttributeError, TypeError):
            return ""
    
    @classmethod
    def get_highlight(cls):
        """Get the highlight theme color."""
        try:
            if cls.THEME == "cyberpunk":
                return cls.BRIGHT_WHITE
            elif cls.THEME == "dracula":
                return cls.BRIGHT_MAGENTA
            elif cls.THEME == "nord":
                return cls.BRIGHT_WHITE
            elif cls.THEME == "monokai":
                return cls.BRIGHT_GREEN
            else:
                return cls.BRIGHT_GREEN
        except (AttributeError, TypeError):
            return ""
    
    @classmethod
    def get_accent(cls):
        """Get the accent theme color."""
        try:
            if cls.THEME == "cyberpunk":
                return cls.BRIGHT_YELLOW
            elif cls.THEME == "dracula":
                return cls.BRIGHT_RED
            elif cls.THEME == "nord":
                return cls.BRIGHT_MAGENTA
            elif cls.THEME == "monokai":
                return cls.BRIGHT_BLUE
            else:
                return cls.YELLOW
        except (AttributeError, TypeError):
            return ""

# Check if terminal supports colors
def supports_color() -> bool:
    """
    Check if the current terminal supports color output.
    
    Returns:
        bool: True if the terminal supports color, False otherwise
    """
    # Check if the ANSI colors are disabled via environment variable
    if os.environ.get('ANSI_COLORS_DISABLED') is not None:
        return False
    
    # Check if running in a terminal that supports colors
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        return platform.system() != 'Windows' or ('ANSICON' in os.environ) or ('WT_SESSION' in os.environ) or os.environ.get('TERM_PROGRAM') == 'vscode'
    
    return False

# Set up colors based on terminal support
if not supports_color():
    # Disable colors if not supported
    for attr in dir(Colors):
        # Check if the attribute is a direct attribute (not a method or dunder)
        # and if its value is a string (potential ANSI code)
        if not attr.startswith('__') and not callable(getattr(Colors, attr)) and isinstance(getattr(Colors, attr), str):
            setattr(Colors, attr, "") 