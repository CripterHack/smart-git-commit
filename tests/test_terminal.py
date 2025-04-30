"""
Unit tests for terminal-related functions.
"""
import os
import sys
import platform
import unittest
from unittest.mock import patch, MagicMock
from smart_git_commit.colors import supports_color, Colors

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestTerminalFunctions(unittest.TestCase):
    """Test case for terminal-related functions."""
    
    def test_supports_color_with_env_variable(self):
        """Test that supports_color respects ANSI_COLORS_DISABLED environment variable."""
        with patch.dict(os.environ, {"ANSI_COLORS_DISABLED": "1"}):
            self.assertFalse(supports_color())
    
    @patch('sys.stdout')
    @patch('platform.system')
    def test_supports_color_windows_without_ansicon(self, mock_platform, mock_stdout):
        """Test color support detection on Windows without ANSICON."""
        # Setup mocks
        mock_platform.return_value = 'Windows'
        mock_stdout.isatty.return_value = True
        
        # Test with no environment variables
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(supports_color())
    
    @patch('sys.stdout')
    @patch('platform.system')
    def test_supports_color_windows_with_ansicon(self, mock_platform, mock_stdout):
        """Test color support detection on Windows with ANSICON."""
        # Setup mocks
        mock_platform.return_value = 'Windows'
        mock_stdout.isatty.return_value = True
        
        # Test with ANSICON environment variable
        with patch.dict(os.environ, {"ANSICON": "1"}):
            self.assertTrue(supports_color())
    
    @patch('sys.stdout')
    @patch('platform.system')
    def test_supports_color_windows_with_wt_session(self, mock_platform, mock_stdout):
        """Test color support detection on Windows with WT_SESSION (Windows Terminal)."""
        # Setup mocks
        mock_platform.return_value = 'Windows'
        mock_stdout.isatty.return_value = True
        
        # Test with WT_SESSION environment variable
        with patch.dict(os.environ, {"WT_SESSION": "1"}):
            self.assertTrue(supports_color())
    
    @patch('sys.stdout')
    @patch('platform.system')
    def test_supports_color_windows_with_vscode(self, mock_platform, mock_stdout):
        """Test color support detection on Windows with VS Code terminal."""
        # Setup mocks
        mock_platform.return_value = 'Windows'
        mock_stdout.isatty.return_value = True
        
        # Test with VS Code environment variable
        with patch.dict(os.environ, {"TERM_PROGRAM": "vscode"}):
            self.assertTrue(supports_color())
    
    @patch('sys.stdout')
    @patch('platform.system')
    def test_supports_color_unix(self, mock_platform, mock_stdout):
        """Test color support detection on Unix-like systems."""
        # Setup mocks
        mock_platform.return_value = 'Linux'
        mock_stdout.isatty.return_value = True
        
        # Test with no environment variables
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(supports_color())
    
    @patch('sys.stdout')
    def test_supports_color_not_tty(self, mock_stdout):
        """Test color support detection when output is not a TTY."""
        # Setup mock
        mock_stdout.isatty.return_value = False
        
        # Test with no environment variables
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(supports_color())
    
    @patch('sys.stdout')
    def test_supports_color_no_isatty(self, mock_stdout):
        """Test color support detection when stdout has no isatty method."""
        # Remove isatty attribute
        del mock_stdout.isatty
        
        # Test with no environment variables
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(supports_color())


if __name__ == '__main__':
    unittest.main() 