#!/usr/bin/env python3
"""
Final attempt to directly test extract_commit_message.
"""

import os
import sys
import coverage

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Starting coverage measurement")
cov = coverage.Coverage(source=['smart_git_commit.command'])
cov.start()

# Import the module after starting coverage
import smart_git_commit.command as cmd

# Replace logger
class MockLogger:
    def error(self, msg):
        print(f"Logger.error: {msg}")

cmd.logger = MockLogger()

# Call the function with a non-existent file
result = cmd.extract_commit_message("/non/existent/file.txt")
print(f"Result from extract_commit_message: '{result}'")

# Stop coverage and report
cov.stop()
cov.save()
print("Coverage report:")
cov.report(include=["**/*command.py"])

# Exit with 0 if test passed
sys.exit(0 if result == "" else 1) 