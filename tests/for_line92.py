#!/usr/bin/env python3
"""
Focused test to address coverage of line 92 in command.py.
"""

import os
import sys
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import coverage directly to control it
from coverage import Coverage

# Create a coverage object with detailed options
cov = Coverage(
    source=['smart_git_commit.command'],
    data_file='.coverage.line92',
    include=['*/command.py'],
    check_preimported=True,
    debug=['dataio', 'pid']
)

# Start tracking coverage
cov.start()

# Import module with function to test
from smart_git_commit.command import extract_commit_message
import smart_git_commit.command

# Replace the logger to prevent actual logging during test
from unittest import mock
smart_git_commit.command.logger = mock.MagicMock()

def test_line92():
    """Test specifically targeting line 92 in command.py."""
    # Show the location of the extracted function
    cmd_file = inspect.getfile(extract_commit_message)
    cmd_line = inspect.getsourcelines(extract_commit_message)[1]
    print(f"Testing function in {cmd_file} starting at line {cmd_line}")
    
    # Show the source code for the function
    source_lines = inspect.getsourcelines(extract_commit_message)[0]
    for i, line in enumerate(source_lines, cmd_line):
        print(f"{i}: {line.rstrip()}")
    
    # Here we get a file path that definitely doesn't exist
    file_path = "____nonexistent____file____path____.txt" 
    
    print(f"\nTesting with nonexistent file: {file_path}")
    
    # Explicitly call the function to hit line 92
    result = extract_commit_message(file_path)
    
    # Check the result is empty string as expected
    print(f"Result: '{result}'")
    assert result == "", f"Expected empty string, got: '{result}'"
    print("Test successful: got empty string as expected")
    
    # Verify logger was called
    print(f"Logger.error called: {smart_git_commit.command.logger.error.called}")
    
    return True

if __name__ == "__main__":
    try:
        # Run the test
        test_passed = test_line92()
        
        # Stop coverage
        cov.stop()
        cov.save()
        
        # Report coverage
        print("\nCoverage report for command.py:")
        cov.report(include=["*/command.py"])
        
        # Generate HTML report for detailed inspection
        cov.html_report(directory='htmlcov')
        print("HTML coverage report generated in htmlcov directory")
        
        # Exit with success or failure
        sys.exit(0 if test_passed else 1)
    except Exception as e:
        print(f"Error during test: {e}")
        # Stop coverage on exception
        if 'cov' in locals():
            cov.stop()
        sys.exit(1)
    finally:
        # Ensure coverage is stopped
        if 'cov' in locals():
            cov.stop() 