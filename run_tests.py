#!/usr/bin/env python3
"""
Helper script to run tests with coverage reporting for the Smart Git Commit tool.
"""

import os
import sys
import subprocess
import argparse


def run_tests(verbose=False, coverage=False):
    """Run the test suite with optional coverage reporting."""
    print("Running Smart Git Commit tests...")
    
    cmd = ["python", "-m", "unittest", "discover", "-s", "tests"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        try:
            # Check if coverage is installed
            import coverage
            print("Running tests with coverage reporting...")
            
            # Create a coverage object
            cov = coverage.Coverage()
            cov.start()
            
            # Run the tests
            subprocess.run(cmd, check=True)
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            print("\nCoverage report:")
            cov.report()
            
            # Generate HTML report
            report_dir = os.path.join(os.path.dirname(__file__), "coverage_html")
            cov.html_report(directory=report_dir)
            print(f"\nDetailed HTML coverage report generated in: {report_dir}")
            
            return True
        
        except ImportError:
            print("Warning: coverage package not installed. Running tests without coverage.")
            coverage = False
    
    if not coverage:
        # Run tests without coverage
        result = subprocess.run(cmd)
        return result.returncode == 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for Smart Git Commit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Run tests in verbose mode")
    parser.add_argument("-c", "--coverage", action="store_true", help="Generate coverage report")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_tests(verbose=args.verbose, coverage=args.coverage)
    sys.exit(0 if success else 1) 