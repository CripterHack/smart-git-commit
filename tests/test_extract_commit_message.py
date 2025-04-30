#!/usr/bin/env python3
"""
Specific tests for extract_commit_message function in command.py module.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Direct import of the function we want to test
from smart_git_commit.command import extract_commit_message


class TestExtractCommitMessage(unittest.TestCase):
    """Test cases specifically for extract_commit_message function."""

    def test_extract_commit_message_success(self):
        """Test extract_commit_message with a valid file."""
        # Create a temporary file with test content
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write("Test commit message")
            temp_path = temp_file.name
        
        try:
            # Test the function
            message = extract_commit_message(temp_path)
            self.assertEqual(message, "Test commit message")
        finally:
            # Clean up
            os.remove(temp_path)

    def test_extract_commit_message_error(self):
        """Test extract_commit_message with a non-existent file."""
        # Use a file path that definitely doesn't exist
        non_existent_path = "/this/file/definitely/does/not/exist/12345.txt"
        
        # Mock the logger to verify it's called
        with mock.patch('smart_git_commit.command.logger') as mock_logger:
            # Call the function
            result = extract_commit_message(non_existent_path)
            
            # Verify empty string is returned
            self.assertEqual(result, "")
            
            # Verify logger.error was called
            mock_logger.error.assert_called_once()

    def test_extract_commit_message_exception(self):
        """Test extract_commit_message with an exception during file read."""
        # Mock open to raise an exception
        with mock.patch('builtins.open', side_effect=Exception("Test error")):
            # Mock the logger
            with mock.patch('smart_git_commit.command.logger') as mock_logger:
                # Call the function
                result = extract_commit_message("any_file.txt")
                
                # Verify empty string is returned
                self.assertEqual(result, "")
                
                # Verify logger.error was called with correct message
                mock_logger.error.assert_called_once_with("Failed to read commit message: Test error")

    def test_direct_extract_commit_message_exception_line92(self):
        """Direct test for line 92 (return "") in extract_commit_message."""
        # First, grab a direct reference to the module-level logger
        import smart_git_commit.command
        original_logger = smart_git_commit.command.logger
        
        try:
            # Replace logger with a mock to prevent actual logging during the test
            mock_logger = mock.MagicMock()
            smart_git_commit.command.logger = mock_logger
            
            # Now directly mock the built-in open function to raise an exception
            original_open = open
            
            def mock_open(*args, **kwargs):
                raise IOError("Mocked file error")
            
            builtins_name = "builtins.open"
            old_open = __builtins__[builtins_name] if builtins_name in __builtins__ else __builtins__['open']
            __builtins__['open'] = mock_open
            
            # Now call the function - this should hit line 92
            import smart_git_commit.command as cmd_module
            result = cmd_module.extract_commit_message("nonexistent.txt")
            
            # Verify empty string is returned (line 92)
            self.assertEqual(result, "")
            
            # Restore original open
            __builtins__['open'] = old_open
            
        finally:
            # Restore original logger
            smart_git_commit.command.logger = original_logger


if __name__ == "__main__":
    unittest.main() 