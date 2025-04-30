#!/usr/bin/env python3
"""
Dedicated test for extract_commit_message in command.py
"""

import os
import sys
import unittest
from unittest import mock
import tempfile
import coverage

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestExtractCommitMessage(unittest.TestCase):
    """Test for extract_commit_message function."""
    
    def setUp(self):
        """Set up test environment."""
        # Start coverage for just this test
        self.cov = coverage.Coverage(
            source=['smart_git_commit.command'],
            data_file='.coverage.extract',
            include=['**/command.py']
        )
        self.cov.start()
        
        # Now import the function after coverage is started
        from smart_git_commit.command import extract_commit_message
        self.extract_commit_message = extract_commit_message
        
        # Import module for mocking
        import smart_git_commit.command
        self.cmd_module = smart_git_commit.command
        
        # Store original logger
        self.original_logger = self.cmd_module.logger
    
    def tearDown(self):
        """Clean up after test."""
        # Restore original logger
        self.cmd_module.logger = self.original_logger
        
        # Stop and save coverage
        self.cov.stop()
        self.cov.save()
    
    def test_successful_read(self):
        """Test extract_commit_message with a valid file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write("Test message")
            temp_path = temp_file.name
        
        try:
            # Test with the file
            result = self.extract_commit_message(temp_path)
            self.assertEqual(result, "Test message")
        finally:
            # Clean up
            os.remove(temp_path)
    
    def test_file_not_found(self):
        """Test extract_commit_message with a non-existent file."""
        # Mock the logger
        mock_logger = mock.MagicMock()
        self.cmd_module.logger = mock_logger
        
        # Test with non-existent file
        result = self.extract_commit_message("/non/existent/file.txt")
        
        # Verify empty string returned
        self.assertEqual(result, "")
        
        # Verify logger called
        mock_logger.error.assert_called_once()


if __name__ == "__main__":
    # Run the test
    unittest.main(exit=False)
    
    # Print coverage report
    print("\nCoverage report:")
    cov = coverage.Coverage(data_file='.coverage.extract')
    cov.load()
    cov.report(include=['**/command.py']) 