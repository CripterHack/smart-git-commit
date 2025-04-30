#!/usr/bin/env python3
"""
Test for a single line coverage (line 92) in extract_commit_message.
"""

import sys
import os
import unittest
from unittest import mock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module containing the function
import smart_git_commit.command


class TestSingleLine(unittest.TestCase):
    """Test specifically for line 92 in extract_commit_message."""
    
    def test_line_92_coverage(self):
        """Test that line 92 (return "") is covered in extract_commit_message."""
        # Replace the original open function with one that raises an exception
        original_open = __builtins__['open']
        
        try:
            # Define a custom open function that raises an exception
            def mock_open(*args, **kwargs):
                raise FileNotFoundError("Mock file not found")
            
            # Replace the built-in open function
            __builtins__['open'] = mock_open
            
            # Mock the logger
            with mock.patch('smart_git_commit.command.logger') as mock_logger:
                # Call the function - this should hit line 92
                result = smart_git_commit.command.extract_commit_message("any_file.txt")
                
                # Verify the result
                self.assertEqual(result, "")
                
                # Verify logger was called
                mock_logger.error.assert_called_once()
                
        finally:
            # Restore the original open function
            __builtins__['open'] = original_open


if __name__ == '__main__':
    unittest.main() 