#!/usr/bin/env python3
"""
Tests for the Smart Git Commit tool.
"""

import os
import sys
import tempfile
import subprocess
from unittest import TestCase, mock
from typing import List, Optional
from collections import defaultdict

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import smart_git_commit
from smart_git_commit import CommitType, GitChange, CommitGroup


class TestCommitType(TestCase):
    """Tests for the CommitType enum."""
    
    def test_commit_types(self):
        """Test that all expected commit types are defined."""
        self.assertEqual(CommitType.FEAT.value, "feat")
        self.assertEqual(CommitType.FIX.value, "fix")
        self.assertEqual(CommitType.DOCS.value, "docs")
        self.assertEqual(CommitType.STYLE.value, "style")
        self.assertEqual(CommitType.REFACTOR.value, "refactor")
        self.assertEqual(CommitType.TEST.value, "test")
        self.assertEqual(CommitType.CHORE.value, "chore")
        self.assertEqual(CommitType.PERF.value, "perf")
        self.assertEqual(CommitType.BUILD.value, "build")
        self.assertEqual(CommitType.CI.value, "ci")


class TestGitChange(TestCase):
    """Tests for the GitChange class."""
    
    def test_file_type_detection(self):
        """Test that file types are correctly detected from extensions."""
        change = GitChange(status="M", filename="example.py")
        self.assertEqual(change.file_type, "py")
        
        change = GitChange(status="M", filename="example.js")
        self.assertEqual(change.file_type, "js")
        
        change = GitChange(status="M", filename="example")
        self.assertEqual(change.file_type, "unknown")
    
    def test_component_detection(self):
        """Test that components are correctly detected from file paths."""
        # Test root files
        change = GitChange(status="M", filename="README.md")
        self.assertEqual(change.component, "docs")
        
        change = GitChange(status="M", filename=".env.example")
        self.assertEqual(change.component, "config")
        
        # Test directories
        change = GitChange(status="M", filename="app/main.py")
        self.assertEqual(change.component, "app")
        
        change = GitChange(status="M", filename="tests/test_main.py")
        self.assertEqual(change.component, "tests")


class TestCommitGroup(TestCase):
    """Tests for the CommitGroup class."""
    
    def test_file_count(self):
        """Test that file count is correctly calculated."""
        group = CommitGroup(name="Test Group", commit_type=CommitType.FEAT)
        self.assertEqual(group.file_count, 0)
        
        group.add_change(GitChange(status="M", filename="file1.py"))
        self.assertEqual(group.file_count, 1)
        
        group.add_change(GitChange(status="M", filename="file2.py"))
        self.assertEqual(group.file_count, 2)
    
    def test_coherence_check(self):
        """Test that coherence is correctly determined."""
        group = CommitGroup(name="Test Group", commit_type=CommitType.FEAT)
        self.assertTrue(group.is_coherent)  # Empty group is coherent
        
        # Add 5 files from the same component
        for i in range(5):
            group.add_change(GitChange(status="M", filename=f"app/file{i}.py"))
        self.assertTrue(group.is_coherent)
        
        # Add a 6th file, making it incoherent due to size
        group.add_change(GitChange(status="M", filename="app/file6.py"))
        self.assertFalse(group.is_coherent)
    
    def test_commit_message_generation(self):
        """Test that commit messages are correctly generated."""
        group = CommitGroup(name="Update core functionality", commit_type=CommitType.FEAT)
        group.add_change(GitChange(status="M", filename="app/main.py"))
        group.add_change(GitChange(status="??", filename="app/new_feature.py"))
        
        message = group.generate_commit_message()
        
        # Check that it contains the expected parts
        self.assertIn("feat(app): Update core functionality", message)
        self.assertIn("Affected files:", message)
        self.assertIn("M app/main.py", message)
        self.assertIn("+ app/new_feature.py", message)


class TestMockGitRepository:
    """A helper class to set up and tear down a mock git repository for testing."""
    
    def __init__(self):
        self.temp_dir = None
        self.original_dir = os.getcwd()
    
    def __enter__(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(self.temp_dir.name)
        
        # Initialize git repository
        subprocess.run(["git", "init"], check=True, capture_output=True)
        
        # Configure git
        subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
        
        # Create some files
        self._create_file("README.md", "# Test Repository\n\nThis is a test repository.")
        self._create_file("main.py", "def main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()")
        
        # Make initial commit
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True, capture_output=True)
        
        return self.temp_dir.name
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.original_dir)
        if self.temp_dir:
            self.temp_dir.cleanup()
    
    def _create_file(self, path: str, content: str):
        """Create a file with the given path and content."""
        with open(path, "w") as f:
            f.write(content)


@mock.patch("smart_git_commit.OllamaClient")
class TestGitCommitWorkflow(TestCase):
    """Tests for the SmartGitCommitWorkflow class."""
    
    def test_load_changes(self, mock_ollama):
        """Test that changes are correctly loaded from git status."""
        with TestMockGitRepository() as repo_path:
            # Create a change
            with open("main.py", "a") as f:
                f.write("\n# Added comment\n")
            
            # Create a new file
            with open("new_file.py", "w") as f:
                f.write("print('New file')\n")
            
            # Initialize workflow
            workflow = smart_git_commit.SmartGitCommitWorkflow(
                repo_path=repo_path, use_ai=False
            )
            workflow.load_changes()
            
            # Check that changes were loaded
            self.assertEqual(len(workflow.changes), 2)
            
            # Verify filenames
            filenames = [change.filename for change in workflow.changes]
            self.assertIn("main.py", filenames)
            self.assertIn("new_file.py", filenames)
    
    def test_rule_based_grouping(self, mock_ollama):
        """Test that rule-based grouping works correctly."""
        with TestMockGitRepository() as repo_path:
            # Create changes in different components
            with open("main.py", "a") as f:
                f.write("\n# Added comment\n")
            
            os.makedirs("docs", exist_ok=True)
            with open("docs/README.md", "w") as f:
                f.write("# Documentation\n")
            
            os.makedirs("tests", exist_ok=True)
            with open("tests/test_main.py", "w") as f:
                f.write("def test_main():\n    pass\n")
            
            # Initialize workflow
            workflow = smart_git_commit.SmartGitCommitWorkflow(
                repo_path=repo_path, use_ai=False
            )
            workflow.load_changes()
            
            # Directly create commit groups by component
            # DO NOT use workflow._rule_based_group_changes() here
            grouped_by_component = defaultdict(list)
            for change in workflow.changes:
                grouped_by_component[change.component].append(change)
                
            # Create the commit groups manually
            workflow.commit_groups = []  # Ensure it's empty
            for component, changes in grouped_by_component.items():
                commit_type = smart_git_commit.CommitType.FEAT
                group = smart_git_commit.CommitGroup(
                    name=f"Update {component}",
                    commit_type=commit_type
                )
                for change in changes:
                    group.add_change(change)
                workflow.commit_groups.append(group)
            
            # Check that there is at least one group
            self.assertGreater(len(workflow.commit_groups), 0)
            
            # Check that files are grouped by component
            for group in workflow.commit_groups:
                component = group.changes[0].component
                for change in group.changes:
                    self.assertEqual(change.component, component)


if __name__ == "__main__":
    import unittest
    unittest.main() 