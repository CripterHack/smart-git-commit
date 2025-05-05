"""
Unit tests for the Colors class.
"""
import os
import sys
import unittest
from unittest.mock import patch
import io

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_git_commit.colors import Colors, supports_color


class TestColors(unittest.TestCase):
    """Test case for Colors class and theme support."""
    
    def setUp(self):
        """Set up the test case."""
        # Store original theme to restore after tests
        self.original_theme = Colors.THEME
    
    def tearDown(self):
        """Clean up after the test case."""
        # Restore original theme
        Colors.THEME = self.original_theme
    
    def test_set_theme(self):
        """Test setting different themes."""
        # Test each valid theme
        for theme in ["standard", "cyberpunk", "dracula", "nord", "monokai"]:
            Colors.set_theme(theme)
            self.assertEqual(Colors.THEME, theme)
        
        # Test case insensitivity
        Colors.set_theme("CYBERPUNK")
        self.assertEqual(Colors.THEME, "cyberpunk")
    
    def test_invalid_theme_handling(self):
        """Test how the system handles invalid themes."""
        # Setting a non-existent theme should still work (theme name is stored)
        Colors.set_theme("nonexistent_theme")
        self.assertEqual(Colors.THEME, "nonexistent_theme")
        
        # Color getters should default to standard theme colors for unknown themes
        self.assertEqual(Colors.get_primary(), Colors.BLUE)
        self.assertEqual(Colors.get_secondary(), Colors.GREEN)
        self.assertEqual(Colors.get_success(), Colors.GREEN)
        self.assertEqual(Colors.get_warning(), Colors.YELLOW)
        self.assertEqual(Colors.get_error(), Colors.RED)
        self.assertEqual(Colors.get_info(), Colors.BRIGHT_WHITE)
        self.assertEqual(Colors.get_highlight(), Colors.BRIGHT_GREEN)
        self.assertEqual(Colors.get_accent(), Colors.YELLOW)
    
    def test_edge_case_themes(self):
        """Test edge cases for theme names."""
        # Empty theme name
        Colors.set_theme("")
        self.assertEqual(Colors.THEME, "")
        # Should still get valid colors (default to standard)
        self.assertEqual(Colors.get_primary(), Colors.BLUE)
        
        # Very long theme name
        long_theme = "a" * 100
        Colors.set_theme(long_theme)
        self.assertEqual(Colors.THEME, long_theme.lower())
        
        # Theme with special characters
        special_theme = "test!@#$%^&*()"
        Colors.set_theme(special_theme)
        self.assertEqual(Colors.THEME, special_theme.lower())
    
    def test_get_primary_color(self):
        """Test getting primary color for different themes."""
        # Standard theme
        Colors.set_theme("standard")
        self.assertEqual(Colors.get_primary(), Colors.BLUE)
        
        # Cyberpunk theme
        Colors.set_theme("cyberpunk")
        self.assertEqual(Colors.get_primary(), Colors.CYBER_BLUE)
        
        # Dracula theme
        Colors.set_theme("dracula")
        self.assertEqual(Colors.get_primary(), Colors.DRACULA_PURPLE)
        
        # Nord theme
        Colors.set_theme("nord")
        self.assertEqual(Colors.get_primary(), Colors.NORD_BLUE)
        
        # Monokai theme
        Colors.set_theme("monokai")
        self.assertEqual(Colors.get_primary(), Colors.YELLOW)
    
    def test_get_secondary_color(self):
        """Test getting secondary color for different themes."""
        # Standard theme
        Colors.set_theme("standard")
        self.assertEqual(Colors.get_secondary(), Colors.GREEN)
        
        # Cyberpunk theme
        Colors.set_theme("cyberpunk")
        self.assertEqual(Colors.get_secondary(), Colors.CYBER_PINK)
        
        # Dracula theme
        Colors.set_theme("dracula")
        self.assertEqual(Colors.get_secondary(), Colors.DRACULA_PINK)
        
        # Nord theme
        Colors.set_theme("nord")
        self.assertEqual(Colors.get_secondary(), Colors.NORD_CYAN)
        
        # Monokai theme
        Colors.set_theme("monokai")
        self.assertEqual(Colors.get_secondary(), Colors.MAGENTA)
    
    def test_get_error_color(self):
        """Test getting error color for different themes."""
        # Standard theme
        Colors.set_theme("standard")
        self.assertEqual(Colors.get_error(), Colors.RED)
        
        # Cyberpunk theme
        Colors.set_theme("cyberpunk")
        self.assertEqual(Colors.get_error(), Colors.BRIGHT_RED)
        
        # Dracula theme
        Colors.set_theme("dracula")
        self.assertEqual(Colors.get_error(), Colors.DRACULA_RED)
        
        # Nord theme
        Colors.set_theme("nord")
        self.assertEqual(Colors.get_error(), Colors.NORD_RED)
        
        # Monokai theme
        Colors.set_theme("monokai")
        self.assertEqual(Colors.get_error(), Colors.RED)
    
    def test_get_success_color(self):
        """Test getting success color for different themes."""
        # Standard theme
        Colors.set_theme("standard")
        self.assertEqual(Colors.get_success(), Colors.GREEN)
        
        # Cyberpunk theme
        Colors.set_theme("cyberpunk")
        self.assertEqual(Colors.get_success(), Colors.BRIGHT_GREEN)
        
        # Dracula theme
        Colors.set_theme("dracula")
        self.assertEqual(Colors.get_success(), Colors.GREEN)
        
        # Nord theme
        Colors.set_theme("nord")
        self.assertEqual(Colors.get_success(), Colors.GREEN)
        
        # Monokai theme
        Colors.set_theme("monokai")
        self.assertEqual(Colors.get_success(), Colors.GREEN)
    
    def test_get_warning_color(self):
        """Test getting warning color for different themes."""
        # Standard theme
        Colors.set_theme("standard")
        self.assertEqual(Colors.get_warning(), Colors.YELLOW)
        
        # Cyberpunk theme
        Colors.set_theme("cyberpunk")
        self.assertEqual(Colors.get_warning(), Colors.BRIGHT_YELLOW)
        
        # Dracula theme
        Colors.set_theme("dracula")
        self.assertEqual(Colors.get_warning(), Colors.YELLOW)
        
        # Nord theme
        Colors.set_theme("nord")
        self.assertEqual(Colors.get_warning(), Colors.YELLOW)
        
        # Monokai theme
        Colors.set_theme("monokai")
        self.assertEqual(Colors.get_warning(), Colors.BRIGHT_YELLOW)
    
    def test_get_info_color(self):
        """Test getting info color for different themes."""
        # Standard theme
        Colors.set_theme("standard")
        self.assertEqual(Colors.get_info(), Colors.BRIGHT_WHITE)
        
        # Cyberpunk theme
        Colors.set_theme("cyberpunk")
        self.assertEqual(Colors.get_info(), Colors.BRIGHT_BLUE)
        
        # Dracula theme
        Colors.set_theme("dracula")
        self.assertEqual(Colors.get_info(), Colors.CYAN)
        
        # Nord theme
        Colors.set_theme("nord")
        self.assertEqual(Colors.get_info(), Colors.BRIGHT_BLUE)
        
        # Monokai theme
        Colors.set_theme("monokai")
        self.assertEqual(Colors.get_info(), Colors.CYAN)
    
    def test_get_highlight_color(self):
        """Test getting highlight color for different themes."""
        # Standard theme
        Colors.set_theme("standard")
        self.assertEqual(Colors.get_highlight(), Colors.BRIGHT_GREEN)
        
        # Cyberpunk theme
        Colors.set_theme("cyberpunk")
        self.assertEqual(Colors.get_highlight(), Colors.BRIGHT_WHITE)
        
        # Dracula theme
        Colors.set_theme("dracula")
        self.assertEqual(Colors.get_highlight(), Colors.BRIGHT_MAGENTA)
        
        # Nord theme
        Colors.set_theme("nord")
        self.assertEqual(Colors.get_highlight(), Colors.BRIGHT_WHITE)
        
        # Monokai theme
        Colors.set_theme("monokai")
        self.assertEqual(Colors.get_highlight(), Colors.BRIGHT_GREEN)
    
    def test_get_accent_color(self):
        """Test getting accent color for different themes."""
        # Standard theme
        Colors.set_theme("standard")
        self.assertEqual(Colors.get_accent(), Colors.YELLOW)
        
        # Cyberpunk theme
        Colors.set_theme("cyberpunk")
        self.assertEqual(Colors.get_accent(), Colors.BRIGHT_YELLOW)
        
        # Dracula theme
        Colors.set_theme("dracula")
        self.assertEqual(Colors.get_accent(), Colors.BRIGHT_RED)
        
        # Nord theme
        Colors.set_theme("nord")
        self.assertEqual(Colors.get_accent(), Colors.BRIGHT_MAGENTA)
        
        # Monokai theme
        Colors.set_theme("monokai")
        self.assertEqual(Colors.get_accent(), Colors.BRIGHT_BLUE)
    
    def test_theme_color_constants_consistency(self):
        """Test that theme-specific color constants are consistent with their usage."""
        # Test cyberpunk theme constants match their usage
        self.assertEqual(Colors.CYBER_BLUE, Colors.BRIGHT_MAGENTA)
        self.assertEqual(Colors.CYBER_PINK, Colors.BRIGHT_CYAN)
        
        # Test dracula theme constants match their usage
        self.assertEqual(Colors.DRACULA_PURPLE, Colors.MAGENTA)
        self.assertEqual(Colors.DRACULA_PINK, Colors.BRIGHT_BLUE)
        self.assertEqual(Colors.DRACULA_RED, Colors.RED)
        
        # Test nord theme constants match their usage
        self.assertEqual(Colors.NORD_BLUE, Colors.BLUE)
        self.assertEqual(Colors.NORD_CYAN, Colors.CYAN)
        self.assertEqual(Colors.NORD_RED, Colors.RED)
    
    @patch('smart_git_commit.smart_git_commit.Colors.THEME')
    def test_color_getters_with_attribute_error(self, mock_theme):
        """Test that color getters handle AttributeError gracefully."""
        # Trigger AttributeError by deleting the theme attribute
        mock_theme.__get__ = unittest.mock.Mock(side_effect=AttributeError("Test error"))
        
        # Test each getter method returns empty string on error
        self.assertEqual(Colors.get_primary(), "")
        self.assertEqual(Colors.get_secondary(), "")
        self.assertEqual(Colors.get_success(), "")
        self.assertEqual(Colors.get_warning(), "")
        self.assertEqual(Colors.get_error(), "")
        self.assertEqual(Colors.get_info(), "")
        self.assertEqual(Colors.get_highlight(), "")
        self.assertEqual(Colors.get_accent(), "")
    
    @patch('smart_git_commit.smart_git_commit.Colors.THEME')
    def test_color_getters_with_type_error(self, mock_theme):
        """Test that color getters handle TypeError gracefully."""
        # Trigger TypeError by setting theme to a number
        mock_theme.__get__ = unittest.mock.Mock(side_effect=TypeError("Test error"))
        
        # Test each getter method returns empty string on error
        self.assertEqual(Colors.get_primary(), "")
        self.assertEqual(Colors.get_secondary(), "")
        self.assertEqual(Colors.get_success(), "")
        self.assertEqual(Colors.get_warning(), "")
        self.assertEqual(Colors.get_error(), "")
        self.assertEqual(Colors.get_info(), "")
        self.assertEqual(Colors.get_highlight(), "")
        self.assertEqual(Colors.get_accent(), "")


class TestColorSupportFunction(unittest.TestCase):
    """Tests for the supports_color utility function."""

    @patch('platform.system', return_value='Linux')
    @patch('sys.stdout.isatty', return_value=True)
    @patch.dict(os.environ, {}, clear=True)
    def test_supports_color_linux_tty(self, mock_isatty, mock_system):
        """Test color support on Linux TTY."""
        self.assertTrue(supports_color())

    @patch('platform.system', return_value='Windows')
    @patch('sys.stdout.isatty', return_value=True)
    @patch.dict(os.environ, {}, clear=True)
    def test_supports_color_windows_tty_no_env(self, mock_isatty, mock_system):
        """Test color support on Windows TTY without specific env vars."""
        self.assertFalse(supports_color())

    @patch('platform.system', return_value='Windows')
    @patch('sys.stdout.isatty', return_value=True)
    @patch.dict(os.environ, {'ANSICON': '1'}, clear=True)
    def test_supports_color_windows_tty_ansicon(self, mock_isatty, mock_system):
        """Test color support on Windows TTY with ANSICON."""
        self.assertTrue(supports_color())

    @patch('platform.system', return_value='Windows')
    @patch('sys.stdout.isatty', return_value=True)
    @patch.dict(os.environ, {'WT_SESSION': '1'}, clear=True)
    def test_supports_color_windows_tty_wt_session(self, mock_isatty, mock_system):
        """Test color support on Windows TTY with WT_SESSION."""
        self.assertTrue(supports_color())

    @patch('platform.system', return_value='Windows')
    @patch('sys.stdout.isatty', return_value=True)
    @patch.dict(os.environ, {'TERM_PROGRAM': 'vscode'}, clear=True)
    def test_supports_color_windows_tty_vscode(self, mock_isatty, mock_system):
        """Test color support on Windows TTY with TERM_PROGRAM=vscode."""
        self.assertTrue(supports_color())

    @patch('sys.stdout.isatty', return_value=False)
    def test_supports_color_not_tty(self, mock_isatty):
        """Test color support when stdout is not a TTY."""
        self.assertFalse(supports_color())

    @patch('sys.stdout.isatty', return_value=True)
    @patch.dict(os.environ, {'ANSI_COLORS_DISABLED': '1'}, clear=True)
    def test_supports_color_disabled_by_env(self, mock_isatty):
        """Test color support disabled by ANSI_COLORS_DISABLED env var."""
        self.assertFalse(supports_color())

    @patch('sys.stdout', new_callable=io.StringIO) # Simulate no isatty attribute
    def test_supports_color_no_isatty(self, mock_stdout):
        """Test color support when stdout lacks isatty attribute."""
        # StringIO doesn't have isatty, so this should directly test the hasattr check
        self.assertFalse(supports_color())


if __name__ == '__main__':
    unittest.main() 