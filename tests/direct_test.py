#!/usr/bin/env python3
"""
Direct test for extract_commit_message function to ensure coverage of line 92.
This test must be run with coverage directly:
python -m coverage run tests/direct_test.py; python -m coverage report -m smart_git_commit/command.py
"""

import os
import sys
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import coverage directly
import coverage
cov = coverage.Coverage(source=['smart_git_commit.command'])
cov.start()

# Import the functions to test
from smart_git_commit.command import extract_commit_message

# Create a mock logger to prevent actual logging
import logging
from unittest import mock

# Get line info for extract_commit_message function
function_source = inspect.getsource(extract_commit_message)
print("Source code of extract_commit_message:")
for i, line in enumerate(function_source.splitlines()):
    print(f"{i+1}: {line}")

# Replace the logger in the command module
import smart_git_commit.command
smart_git_commit.command.logger = mock.MagicMock()

# Define a test function
def test_extract_commit_message_line92():
    """Test that specifically exercises line 92."""
    # Create a non-existent file path
    non_existent_file = "/completely/invalid/path/that/does/not/exist.txt"
    
    # Call the function - should trigger the exception and return ""
    result = extract_commit_message(non_existent_file)
    
    # Print the result to verify
    print(f"Result from extract_commit_message: '{result}'")
    print(f"Expected result: ''")
    print(f"Test passed: {result == ''}")
    
    # Verify logger was called
    print(f"Logger error called: {smart_git_commit.command.logger.error.called}")
    
    # Return the result for verification
    return result == ""

# Run the test directly
if __name__ == "__main__":
    try:
        success = test_extract_commit_message_line92()
        # Stop coverage before exiting
        cov.stop()
        cov.save()
        # Print coverage report
        cov.report()
        sys.exit(0 if success else 1)  # Exit with status based on test result
    finally:
        # Always stop coverage to ensure data is saved
        if cov:
            cov.stop() 