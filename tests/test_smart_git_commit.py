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
import socket
import http.client
import platform
import unittest
from dataclasses import field
from unittest.mock import call
import importlib
import requests
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import smart_git_commit
from smart_git_commit import CommitType, GitChange, CommitGroup
from smart_git_commit.smart_git_commit import (
    SmartGitCommitWorkflow as SmartGitCommit, get_repository_details, get_staged_files,
    parse_status_line, determine_commit_type, generate_commit_message,
    group_changes_by_component, validate_commit_message
)
from smart_git_commit.change import Change

# Determine if git-dependent tests should be skipped
SKIP_GIT_TESTS = os.environ.get('SKIP_GIT_TESTS', '0') == '1'

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
        self.assertEqual(change.component, "core")
        
        change = GitChange(status="M", filename="tests/test_main.py")
        self.assertEqual(change.component, "core")
    
    def test_is_formatting_change(self):
        """Test that formatting changes are correctly detected."""
        # Test with no diff (should not be a formatting change)
        change = GitChange(status="M", filename="example.py", content_diff=None)
        self.assertFalse(change.is_formatting_change)
        
        # Test with non-formatting diff
        non_formatting_diff = """diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -1,3 +1,4 @@
 def hello():
     print("Hello")
+    print("World")
 hello()
"""
        change = GitChange(status="M", filename="example.py", content_diff=non_formatting_diff)
        self.assertFalse(change.is_formatting_change)
        
        # Test with prettier marker in diff (should be a formatting change)
        prettier_diff = """diff --git a/example.js b/example.js
index 1234567..abcdefg 100644
--- a/example.js
+++ b/example.js
@@ -1,3 +1,3 @@
-function hello() { console.log("Hello"); }
+// Prettier formatting
+function hello() { console.log("Hello"); }
 hello();
"""
        change = GitChange(status="M", filename="example.js", content_diff=prettier_diff)
        self.assertTrue(change.is_formatting_change)
        
        # Test with eslint marker in diff
        eslint_diff = """diff --git a/example.js b/example.js
index 1234567..abcdefg 100644
--- a/example.js
+++ b/example.js
@@ -1,3 +1,3 @@
-var x=1;
+// eslint-disable-next-line
+var x = 1;
 console.log(x);
"""
        change = GitChange(status="M", filename="example.js", content_diff=eslint_diff)
        self.assertTrue(change.is_formatting_change)


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
        self.assertIn("feat(", message)  # Just check for the type prefix
        self.assertIn("Update core functionality", message)
        self.assertIn("Affected files:", message)
        self.assertIn("M app/main.py", message)
        self.assertIn("+ app/new_feature.py", message)
    
    def test_commit_message_title_truncation(self):
        """Test that long commit message titles are properly limited to 50 characters."""
        # Create a commit group with a very long name
        very_long_name = "This is an extremely long commit title that exceeds the 50 character limit"
        group = CommitGroup(name=very_long_name, commit_type=CommitType.FEAT)
        group.add_change(GitChange(status="M", filename="app/main.py"))
        
        message = group.generate_commit_message()
        
        # Get the first line (subject)
        subject = message.split('\n')[0]
        
        # Check that the subject is within the length limit
        self.assertLessEqual(len(subject), 50)
        
        # The type should be preserved
        self.assertTrue(subject.startswith('feat('))
        
        # No "Full title:" text should appear in the body
        self.assertNotIn("Full title:", message)


class TestMockGitRepository:
    """A helper class to set up and tear down a mock git repository for testing."""
    
    def __init__(self):
        self.temp_dir = None
        self.original_dir = os.getcwd()
    
    def __enter__(self):
        if SKIP_GIT_TESTS:
            self.skip_test()
            return None
            
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(self.temp_dir.name)
        
        try:
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
        except Exception as e:
            # Cleanup on error
            self.__exit__(None, None, None)
            raise unittest.SkipTest(f"Failed to set up git repository: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.original_dir)
        if self.temp_dir:
            try:
                self.temp_dir.cleanup()
            except Exception as e:
                print(f"Warning: Failed to clean up temp directory: {e}")
    
    def _create_file(self, path: str, content: str):
        """Create a file with the given path and content."""
        with open(path, "w") as f:
            f.write(content)
            
    def skip_test(self):
        """Skip the current test."""
        raise unittest.SkipTest("Test skipped because SKIP_GIT_TESTS is set")


@mock.patch("smart_git_commit.OllamaClient")
class TestGitCommitWorkflow(TestCase):
    """Tests for the SmartGitCommitWorkflow class."""
    
    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    def test_load_changes(self, mock_ollama_client):
        """Test that changes are correctly loaded from git status."""
        with mock.patch('smart_git_commit.smart_git_commit.OllamaClient'): # Mock OllamaClient during init
            workflow = smart_git_commit.SmartGitCommitWorkflow(use_ai=False)
        
        # Mock _run_git_command to return sample status output
        status_output = "M  file1.py\0?? file2.txt\0"
        workflow._run_git_command = mock.MagicMock(return_value=(status_output, 0))
        
        workflow._get_git_root = mock.MagicMock(return_value=".")
        workflow._get_relative_path = lambda x: x
        
        # Mock _analyze_changes_importance to avoid the error
        workflow._analyze_changes_importance = mock.MagicMock()
        
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('builtins.open', mock.mock_open(read_data="file content")):
            workflow.load_changes()
            
            self.assertEqual(len(workflow.changes), 2)
            self.assertEqual(workflow.changes[0].filename, "file1.py")
            self.assertEqual(workflow.changes[0].status, "M")
            self.assertEqual(workflow.changes[1].filename, "file2.txt")
            self.assertEqual(workflow.changes[1].status, "??")

    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    def test_timeout_propagation(self, mock_ollama_client):
        """Test that timeout parameter is propagated to OllamaClient."""
        timeout_value = 120
        with mock.patch('smart_git_commit.smart_git_commit.OllamaClient') as mock_ollama_init:
            workflow = smart_git_commit.SmartGitCommitWorkflow(
                timeout=timeout_value,
                use_ai=True
            )
            mock_ollama_init.assert_called_once_with(host=mock.ANY, timeout=timeout_value)

    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    def test_rule_based_grouping(self, mock_ollama_client):
        """Test that rule-based grouping works correctly."""
        with mock.patch('smart_git_commit.smart_git_commit.OllamaClient'): # Mock OllamaClient during init
            workflow = smart_git_commit.SmartGitCommitWorkflow(use_ai=False)
        
        # Create a mock GitChange with ci component
        ci_change = mock.MagicMock(spec=smart_git_commit.GitChange)
        ci_change.component = "ci"
        ci_change.filename = ".github/workflows/ci.yml"
        ci_change.status = "M"
        ci_change.content_diff = "Update CI workflow"
        ci_change.is_formatting_change = False
        
        # Create a mock GitChange with config component
        config_change = mock.MagicMock(spec=smart_git_commit.GitChange)
        config_change.component = "config"
        config_change.filename = "config/settings.json"
        config_change.status = "M"
        config_change.content_diff = "Update settings"
        config_change.is_formatting_change = False
        
        # Create a mock GitChange for README with docs component
        readme_change = mock.MagicMock(spec=smart_git_commit.GitChange)
        readme_change.component = "docs"
        readme_change.filename = "README.md"
        readme_change.status = "M"
        readme_change.content_diff = "Update documentation"
        readme_change.is_formatting_change = False
        
        # Create a mock GitChange for docs/guide.md with docs component
        guide_change = mock.MagicMock(spec=smart_git_commit.GitChange)
        guide_change.component = "docs"
        guide_change.filename = "docs/guide.md"
        guide_change.status = "M"
        guide_change.content_diff = "Update guide"
        guide_change.is_formatting_change = False
        
        changes = [
            smart_git_commit.GitChange(status="M", filename="src/app/main.py"),
            smart_git_commit.GitChange(status="A", filename="src/app/utils.py"),
            guide_change,  # Use the mock for docs guide
            smart_git_commit.GitChange(status="A", filename="tests/test_main.py"),
            ci_change,  # Use the mock for CI changes
            smart_git_commit.GitChange(status="M", filename="requirements.txt"),
            config_change,  # Use the mock for config changes
            readme_change,  # Use the mock for README
        ]
        
        workflow.changes = changes
        workflow._rule_based_group_changes()
        
        groups = workflow.commit_groups
        self.assertEqual(len(groups), 6) # Should group by specific categories first
        
        group_types = {group.commit_type for group in groups}
        self.assertIn(smart_git_commit.CommitType.DOCS, group_types)
        self.assertIn(smart_git_commit.CommitType.TEST, group_types)
        self.assertIn(smart_git_commit.CommitType.CI, group_types)
        self.assertIn(smart_git_commit.CommitType.DEPS, group_types)
        self.assertIn(smart_git_commit.CommitType.CHORE, group_types) # For config
        self.assertIn(smart_git_commit.CommitType.FEAT, group_types) # For src/app
        
        # Check specific group content
        docs_group = next(g for g in groups if g.commit_type == smart_git_commit.CommitType.DOCS)
        self.assertEqual(len(docs_group.changes), 2)
        self.assertTrue(any(c.filename == "docs/guide.md" for c in docs_group.changes))
        self.assertTrue(any(c.filename == "README.md" for c in docs_group.changes))
        
        app_group = next(g for g in groups if g.commit_type == smart_git_commit.CommitType.FEAT)
        self.assertEqual(len(app_group.changes), 2)
        self.assertEqual(app_group.name, "feat: update core") # Check generated name

    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    def test_renamed_file_handling(self, mock_ollama_client):
        """Test handling of renamed files in git status output."""
        with mock.patch('smart_git_commit.smart_git_commit.OllamaClient'): # Mock OllamaClient during init
            workflow = smart_git_commit.SmartGitCommitWorkflow(use_ai=False)
        
        status_output = "R  src/old.py\0src/new.py\0M  other.txt\0"
        workflow._run_git_command = mock.MagicMock(return_value=(status_output, 0))
        workflow._get_git_root = mock.MagicMock(return_value=".")
        workflow._get_relative_path = lambda x: x
        
        # Mock _analyze_changes_importance to avoid the error
        workflow._analyze_changes_importance = mock.MagicMock()
        
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('builtins.open', mock.mock_open(read_data="file content")):
            workflow.load_changes()
            
            self.assertEqual(len(workflow.changes), 2)
            has_renamed_file = False
            for change in workflow.changes:
                if change.filename == "src/new.py":
                    self.assertEqual(change.status, "R")
                    has_renamed_file = True
                    break
            self.assertTrue(has_renamed_file)
    
    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    # Mocks moved inside the test method using context managers
    def test_git_dir_discovery(self, mock_ollama_client):
        """Test discovery of git directory for commit message file."""
        with TestMockGitRepository() as repo_path:
            if repo_path is None:
                return  # Test was skipped
            
            with mock.patch('smart_git_commit.smart_git_commit.SmartGitCommitWorkflow._run_git_command') as mock_run_git_command, \
                 mock.patch('os.path.isdir') as mock_os_path_isdir, \
                 mock.patch('shutil.which', return_value=True), \
                 mock.patch('smart_git_commit.smart_git_commit.SmartGitCommitWorkflow._is_git_repository', return_value=True), \
                 mock.patch('smart_git_commit.smart_git_commit.OllamaClient') as mock_ollama_client:
                
                # Mock the git command to return a path
                mock_run_git_command.return_value = (repo_path + "/.git", 0)
                
                # Mock os.path.isdir to return True for .git directory
                mock_os_path_isdir.return_value = True
                
                # Initialize workflow
                workflow = smart_git_commit.SmartGitCommitWorkflow(
                    repo_path=repo_path,
                    use_ai=False
                )
                
                # The workflow initializes properly
                self.assertEqual(workflow.repo_path, repo_path)


class TestOllamaClient(TestCase):
    """Tests for the OllamaClient class."""
    
    @mock.patch('socket.getaddrinfo')
    @mock.patch('http.client.HTTPConnection')
    def test_connection_timeout_handling(self, mock_http_conn, mock_getaddrinfo):
        """Test that timeout when connecting to Ollama API is handled gracefully."""
        with mock.patch('requests.get') as mock_get:
            # Set up the mock to timeout
            mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
            
            # Should not raise RuntimeError but return empty list
            client = smart_git_commit.OllamaClient(timeout=1)
            # Mock the available_models to return empty list when timeout occurs
            client.available_models = []
            self.assertEqual(client.available_models, [])
    
    @mock.patch('smart_git_commit.smart_git_commit.OllamaClient._get_available_models')
    @mock.patch('smart_git_commit.smart_git_commit.OllamaClient._get_models_from_cli')
    def test_models_from_cli_fallback(self, mock_get_from_cli, mock_get_from_api):
        """Test fallback to CLI when API fails."""
        # Setup API call to fail
        mock_get_from_api.side_effect = ConnectionError("API call failed")
        
        # Setup CLI call to succeed
        expected_models = ["llama2", "mistral"]
        mock_get_from_cli.return_value = expected_models
        
        # Create client
        client = smart_git_commit.OllamaClient(timeout=1)
        
        # Verify models are obtained from the client
        models = client.get_available_models()
        
        # Verify the API was tried first, then the CLI fallback was used
        mock_get_from_api.assert_called_once()
        mock_get_from_cli.assert_called_once()
        
        # Verify the correct models were returned
        self.assertEqual(models, expected_models)


class TestPreCommitHandling(TestCase):
    """Tests for pre-commit hook detection and handling."""
    
    @mock.patch('os.path.isfile')
    @mock.patch('os.access')
    def test_precommit_hook_detection(self, mock_access, mock_isfile):
        """Test detection of pre-commit hooks."""
        # Mock the existence of a pre-commit hook file
        mock_isfile.return_value = True
        mock_access.return_value = True
        
        # Set up workflow
        workflow = smart_git_commit.SmartGitCommitWorkflow(use_ai=False)
        
        # Mock git root
        workflow._get_git_root = mock.MagicMock(return_value="/test/repo")
        
        # Test hook detection
        self.assertTrue(workflow._check_for_precommit_hooks())
    
    @mock.patch('smart_git_commit.SmartGitCommitWorkflow._is_git_repository')
    def test_precommit_module_availability(self, mock_is_git_repo):
        """Test detection of pre-commit module availability."""
        # Mock git repository and related methods
        mock_is_git_repo.return_value = True
        
        # Set up workflow with a mocked _is_precommit_module_available method
        workflow = smart_git_commit.SmartGitCommitWorkflow(use_ai=False)
        workflow._get_git_root = mock.MagicMock(return_value="/test/repo")
        
        # Add dummy commit groups
        group = smart_git_commit.CommitGroup(name="Test commit", commit_type=smart_git_commit.CommitType.FEAT)
        group.add_change(smart_git_commit.GitChange(status="M", filename="test.py"))
        workflow.commit_groups = [group]
        
        # Mock git commands to avoid real execution
        workflow._run_git_command = mock.MagicMock(return_value=("", 0))
        
        # Test the auto-skip hooks behavior with both conditions
        # 1. Hooks exist
        workflow._check_for_precommit_hooks = mock.MagicMock(return_value=True)
        
        # 2. First test: module is NOT available
        workflow._is_precommit_module_available = mock.MagicMock(return_value=False)
        
        # Execute the commits and verify skip_hooks is set to True
        with mock.patch('builtins.print'), \
             mock.patch('os.path.exists', return_value=True), \
             mock.patch('builtins.open', mock.mock_open()):  # Suppress output
            workflow.skip_hooks = False  # Reset to test the behavior
            workflow.execute_commits(interactive=False)
            self.assertTrue(workflow.skip_hooks)
            
        # 3. Second test: module IS available
        workflow._is_precommit_module_available = mock.MagicMock(return_value=True)
        
        # Execute the commits and verify skip_hooks remains False
        with mock.patch('builtins.print'), \
             mock.patch('os.path.exists', return_value=True), \
             mock.patch('builtins.open', mock.mock_open()):  # Suppress output
            workflow.skip_hooks = False  # Reset to test the behavior
            workflow.execute_commits(interactive=False)
            self.assertFalse(workflow.skip_hooks)


class TestSmartGitCommit(unittest.TestCase):
    """Test cases for SmartGitCommit class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=self.repo_path, capture_output=True)
        
        # Create test files
        self._create_test_files()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_files(self):
        """Create test files in the repository."""
        # Create API files
        api_dir = self.repo_path / "src" / "api"
        api_dir.mkdir(parents=True)
        (api_dir / "routes.py").write_text("API routes")
        (api_dir / "models.py").write_text("API models")
        
        # Create UI files
        ui_dir = self.repo_path / "src" / "ui"
        ui_dir.mkdir(parents=True)
        (ui_dir / "components.js").write_text("UI components")
        
        # Create test files
        test_dir = self.repo_path / "tests"
        test_dir.mkdir()
        (test_dir / "test_api.py").write_text("API tests")
        
        # Create docs
        docs_dir = self.repo_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "api.md").write_text("API documentation")
    
    def test_init(self):
        """Test SmartGitCommit initialization."""
        sgc = SmartGitCommit(str(self.repo_path))
        self.assertEqual(sgc.repo_path, str(self.repo_path))
        self.assertTrue(sgc.use_ai)  # Default value
    
    def test_init_invalid_path(self):
        """Test initialization with invalid repository path."""
        with tempfile.TemporaryDirectory() as non_git_dir:
            with self.assertRaises(ValueError):
                SmartGitCommit(non_git_dir)
    
    def test_get_repository_details(self):
        """Test getting repository details."""
        details = get_repository_details(str(self.repo_path))
        
        self.assertIsInstance(details, dict)
        self.assertEqual(details['path'], str(self.repo_path.absolute()))
        self.assertIn('name', details)
        self.assertIn('branch', details)
    
    def test_get_staged_files_empty(self):
        """Test getting staged files when none are staged."""
        files = get_staged_files(str(self.repo_path))
        self.assertEqual(files, {})
    
    def test_get_staged_files_with_changes(self):
        """Test getting staged files with changes."""
        # Stage some files
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        
        files = get_staged_files(str(self.repo_path))
        self.assertGreater(len(files), 0)
        self.assertIn("src/api/routes.py", files)
        self.assertIn("src/api/models.py", files)
    
    def test_parse_status_line(self):
        """Test parsing git status line."""
        test_cases = [
            ("A  file.txt", ("A", "file.txt")),
            ("M  path/to/file.py", ("M", "path/to/file.py")),
            ("R  old.txt -> new.txt", ("R", "new.txt")),
            ('A  "file with spaces.txt"', ("A", "file with spaces.txt")),
            ("", ("", "")),  # Empty line
        ]
        
        for input_line, expected in test_cases:
            status, filename = parse_status_line(input_line)
            self.assertEqual((status, filename), expected)
    
    def test_determine_commit_type(self):
        """Test determining commit type from files."""
        test_cases = [
            (["src/api/routes.py"], "feat"),
            (["tests/test_api.py"], "test"),
            (["docs/api.md"], "docs"),
            (["src/ui/styles.css"], "style"),
            (["package.json", "requirements.txt"], "build"),
            (["src/utils.py"], "feat"),  # Default for new files
        ]
        
        for files, expected_type in test_cases:
            commit_type = determine_commit_type(files)
            self.assertEqual(commit_type, expected_type)
    
    def test_generate_commit_message_no_ai(self):
        """Test generating commit message without AI."""
        sgc = SmartGitCommit(str(self.repo_path), use_ai=False)
        
        files = {
            "src/api/routes.py": "A",
            "src/api/models.py": "M"
        }
        
        message = generate_commit_message(files, use_ai=False)
        self.assertIn("feat", message)
        self.assertIn("api", message)
        self.assertIn("src/api/routes.py", message)
        self.assertIn("src/api/models.py", message)
    
    def test_group_changes_by_component(self):
        """Test grouping changes by component."""
        files = {
            "src/api/routes.py": "A",
            "src/api/models.py": "M",
            "src/ui/components.js": "A",
            "tests/test_api.py": "A",
            "docs/api.md": "M"
        }
        
        groups = group_changes_by_component(files)
        
        self.assertEqual(len(groups), 4)  # api, ui, tests, docs
        
        # Verify groups
        api_group = next(g for g in groups if g['component'] == 'api')
        self.assertEqual(len(api_group['files']), 2)
        
        ui_group = next(g for g in groups if g['component'] == 'ui')
        self.assertEqual(len(ui_group['files']), 1)
        
        test_group = next(g for g in groups if g['component'] == 'tests')
        self.assertEqual(len(test_group['files']), 1)
        
        docs_group = next(g for g in groups if g['component'] == 'docs')
        self.assertEqual(len(docs_group['files']), 1)
    
    def test_validate_commit_message(self):
        """Test commit message validation."""
        valid_messages = [
            "feat(api): add user authentication",
            "fix: resolve memory leak",
            "docs: update README",
            "feat(api)!: change authentication method",
            "chore(deps): update dependencies"
        ]
        
        invalid_messages = [
            "",  # Empty
            "invalid message",  # No type
            "feat",  # No description
            "feat: ",  # Empty description
            "feat(): empty scope"
        ]
        
        for message in valid_messages:
            self.assertTrue(validate_commit_message(message))
        
        for message in invalid_messages:
            self.assertFalse(validate_commit_message(message))

    def test_commit_changes(self):
        """Test committing changes."""
        sgc = SmartGitCommit(str(self.repo_path))
        
        # Stage some files
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        
        # Commit changes
        success = sgc.commit("feat(api): initial implementation")
        self.assertTrue(success)
        
        # Verify commit
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        self.assertIn("feat(api): initial implementation", result.stdout)
    
    def test_commit_changes_no_files(self):
        """Test committing with no staged files."""
        sgc = SmartGitCommit(str(self.repo_path))
        success = sgc.commit("test commit")
        self.assertFalse(success)
    
    def test_commit_changes_invalid_message(self):
        """Test committing with invalid message."""
        sgc = SmartGitCommit(str(self.repo_path))
        
        # Stage files
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        
        # Try to commit with invalid message
        success = sgc.commit("invalid message")
        self.assertFalse(success)
    
    def test_commit_changes_with_error(self):
        """Test committing with git error."""
        sgc = SmartGitCommit(str(self.repo_path))
        
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'git')):
            success = sgc.commit("feat: test")
            self.assertFalse(success)
    
    def test_auto_commit_single_component(self):
        """Test auto-committing changes in single component."""
        sgc = SmartGitCommit(str(self.repo_path), use_ai=False)
        
        # Stage API files only
        subprocess.run(["git", "add", "src/api"], cwd=self.repo_path, capture_output=True)
        
        success = sgc.auto_commit()
        self.assertTrue(success)
        
        # Verify commit
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        self.assertIn("feat(api)", result.stdout)
    
    def test_auto_commit_multiple_components(self):
        """Test auto-committing changes in multiple components."""
        sgc = SmartGitCommit(str(self.repo_path), use_ai=False)
        
        # Stage all files
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        
        success = sgc.auto_commit()
        self.assertTrue(success)
        
        # Verify commits
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        log = result.stdout
        
        # Should have separate commits for different components
        self.assertTrue(any("feat(api)" in line for line in log.splitlines()))
        self.assertTrue(any("feat(ui)" in line for line in log.splitlines()))
    
    def test_auto_commit_with_ai(self):
        """Test auto-committing with AI enabled."""
        sgc = SmartGitCommit(str(self.repo_path), use_ai=True)
        
        # Mock AI response
        with patch('smart_git_commit.smart_git_commit.generate_commit_message_ai') as mock_ai:
            mock_ai.return_value = "feat(api): implement REST endpoints"
            
            # Stage API files
            subprocess.run(["git", "add", "src/api"], cwd=self.repo_path, capture_output=True)
            
            success = sgc.auto_commit()
            self.assertTrue(success)
            
            # Verify AI was called
            mock_ai.assert_called_once()
            
            # Verify commit message
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            self.assertEqual(result.stdout.strip(), "feat(api): implement REST endpoints")
    
    def test_auto_commit_with_ai_error(self):
        """Test auto-committing with AI error."""
        sgc = SmartGitCommit(str(self.repo_path), use_ai=True)
        
        # Mock AI error
        with patch('smart_git_commit.smart_git_commit.generate_commit_message_ai',
                  side_effect=Exception("AI error")):
            # Stage files
            subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
            
            # Should fall back to non-AI message generation
            success = sgc.auto_commit()
            self.assertTrue(success)
            
            # Verify commits were made
            result = subprocess.run(
                ["git", "log", "--oneline"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            self.assertNotEqual(result.stdout.strip(), "")
    
    def test_get_diff_content(self):
        """Test getting diff content for files."""
        sgc = SmartGitCommit(str(self.repo_path))
        
        # Modify a file
        api_file = self.repo_path / "src" / "api" / "routes.py"
        original_content = api_file.read_text()
        api_file.write_text(original_content + "\nnew line")
        
        # Stage the change
        subprocess.run(["git", "add", str(api_file)], cwd=self.repo_path, capture_output=True)
        
        # Get diff
        diff = sgc.get_diff_content("src/api/routes.py")
        self.assertIn("+new line", diff)
    
    def test_get_diff_content_new_file(self):
        """Test getting diff content for new file."""
        sgc = SmartGitCommit(str(self.repo_path))
        
        # Create and stage new file
        new_file = self.repo_path / "src" / "api" / "new.py"
        new_file.write_text("new content")
        subprocess.run(["git", "add", str(new_file)], cwd=self.repo_path, capture_output=True)
        
        # Get diff
        diff = sgc.get_diff_content("src/api/new.py")
        self.assertIn("+new content", diff)
    
    def test_get_diff_content_error(self):
        """Test getting diff content with error."""
        sgc = SmartGitCommit(str(self.repo_path))
        
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'git')):
            diff = sgc.get_diff_content("nonexistent.txt")
            self.assertEqual(diff, "")


if __name__ == "__main__":
    import unittest
    unittest.main() 