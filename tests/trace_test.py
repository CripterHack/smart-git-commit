#!/usr/bin/env python3
"""
Direct line-by-line tracing of extract_commit_message.
"""

import os
import sys
import inspect
import coverage
import trace

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import specifically for test 
from smart_git_commit.command import extract_commit_message

# Set up tracing
tracer = trace.Trace(
    count=True,
    trace=True,  # Enable line tracing 
    countfuncs=True,
    countcallers=True
)

# Start tracing
tracer.runfunc(extract_commit_message, "/this/file/does/not/exist.txt")

# Get the source file of the function
source_file = inspect.getfile(extract_commit_message)
print(f"Source file: {source_file}")

# Print results
results = tracer.results()
print("\nLine counts:")
results.write_results(show_missing=True, summary=True, coverdir=".") 