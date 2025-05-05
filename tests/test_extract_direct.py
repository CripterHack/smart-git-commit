#!/usr/bin/env python3
"""
Simple test file to verify coverage of extract_commit_message function.
"""

import os
import sys
import unittest
from unittest import mock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the specific function we want to test
from smart_git_commit.command import extract_commit_message


class TestExtractDirect(unittest.TestCase):
    """Direct tests for extract_commit_message."""
    
    def test_extract_commit_message_file_not_found(self):
        """Test extract_commit_message when file is not found."""
        # Mock the logger to avoid actual logging
        with mock.patch('smart_git_commit.command.logger') as mock_logger:
            # Call function with a file that doesn't exist
            result = extract_commit_message('/path/that/does/not/exist.txt')
            
            # Verify that an empty string is returned
            self.assertEqual(result, "")
            
            # Verify that the logger was called
            mock_logger.error.assert_called()


if __name__ == '__main__':
    unittest.main() 