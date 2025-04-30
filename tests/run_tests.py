#!/usr/bin/env python
"""
Test runner for Smart Git Commit.

This script discovers and runs tests in the tests directory.
Use -v for verbose output and -c for coverage reporting.
"""

import argparse
import os
import sys
import unittest
import subprocess
from pathlib import Path

def _run_unittest_discovery(test_loader, test_dir, verbose):
    """Run the unittest discovery with specified options."""
    test_suite = test_loader.discover(test_dir)
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

def run_tests(verbose=False, coverage=False):
    """Run the test suite."""
    # Configure test discovery
    test_loader = unittest.TestLoader()
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    if not os.path.exists(test_dir):
        print(f"Creating tests directory: {test_dir}")
        os.makedirs(test_dir)

    if coverage:
        try:
            # Check if coverage is installed
            import coverage as cov_module
            
            # Configure coverage
            cov = cov_module.Coverage(
                source=['smart_git_commit'],
                omit=['*/__pycache__/*', '*/tests/*']
            )
            
            # Start measuring coverage
            cov.start()
            
            # Run tests
            success = _run_unittest_discovery(test_loader, test_dir, verbose)
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            print("\nCoverage Report:")
            cov.report()
            
            try:
                # Generate HTML report
                html_dir = os.path.join(os.path.dirname(__file__), 'htmlcov')
                cov.html_report(directory=html_dir)
                print(f"\nHTML coverage report generated in: {html_dir}")
            except Exception as e:
                print(f"Warning: Failed to generate HTML coverage report: {e}")
            
            return success
            
        except ImportError:
            print("Coverage module not found. Install with: pip install coverage")
            print("Running tests without coverage...")
            return _run_unittest_discovery(test_loader, test_dir, verbose)
    else:
        # Run tests without coverage
        return _run_unittest_discovery(test_loader, test_dir, verbose)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Smart Git Commit tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-c', '--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('-s', '--skip-git-tests', action='store_true', help='Skip tests that require git operations')
    return parser.parse_args()

def main():
    """Entry point for the script."""
    args = parse_args()
    
    # Set environment variable to skip git-dependent tests if requested
    if args.skip_git_tests:
        os.environ['SKIP_GIT_TESTS'] = '1'
        print("Skipping git-dependent tests")
    
    print(f"Running tests {'with coverage' if args.coverage else 'without coverage'} {'in verbose mode' if args.verbose else ''}...")
    success = run_tests(verbose=args.verbose, coverage=args.coverage)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 