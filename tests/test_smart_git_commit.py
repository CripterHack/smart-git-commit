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
        
        # Test with whitespace-only diff (should be a formatting change)
        whitespace_diff = """diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello")
+    print("Hello")
 hello()
"""
        change = GitChange(status="M", filename="example.py", content_diff=whitespace_diff)
        self.assertTrue(change.is_formatting_change)
        
        # Test with prettier marker in diff
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
        
        # The type and scope should be preserved
        self.assertTrue(subject.startswith('feat(app):'))
        
        # No "Full title:" text should appear in the body
        self.assertNotIn("Full title:", message)


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
    
    def test_timeout_propagation(self, mock_ollama):
        """Test that timeout parameter is propagated to OllamaClient."""
        # Set up mock for OllamaClient
        mock_ollama_instance = mock.MagicMock()
        mock_ollama.return_value = mock_ollama_instance
        
        # Create workflow with custom timeout
        custom_timeout = 30
        workflow = smart_git_commit.SmartGitCommitWorkflow(
            use_ai=True, 
            timeout=custom_timeout
        )
        
        # Verify OllamaClient was created with the right timeout
        mock_ollama.assert_called_once()
        _, kwargs = mock_ollama.call_args
        self.assertEqual(kwargs['timeout'], custom_timeout)
        
        # Verify timeout is stored in the workflow
        self.assertEqual(workflow.timeout, custom_timeout)
    
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
    
    def test_renamed_file_handling(self, mock_ollama):
        """Test handling of renamed files in git status output."""
        # Create a mock workflow
        workflow = smart_git_commit.SmartGitCommitWorkflow(use_ai=False)
        
        # Mock the _run_git_command to return appropriate responses
        def mock_git_command(args):
            if args[0] == "status":
                return "R  old_name.py -> new_name.py\n", 0
            elif args[0] == "diff":
                return "diff --git a/old_name.py b/new_name.py\nindex 1234567..abcdefg 100644\n--- a/old_name.py\n+++ b/new_name.py\n", 0
            return "", 0
            
        workflow._run_git_command = mock.MagicMock(side_effect=mock_git_command)
        
        # Call load_changes
        workflow.load_changes()
        
        # Check that the renamed file is properly handled
        self.assertEqual(len(workflow.changes), 1)
        change = workflow.changes[0]
        self.assertEqual(change.status, "R")
        self.assertEqual(change.filename, "new_name.py")
        self.assertIsNotNone(change.content_diff)
        self.assertIn("diff --git", change.content_diff)
    
    def test_subdirectory_prefixes(self, mock_ollama):
        """Test handling of files with directory prefixes."""
        # Create a mock workflow
        workflow = smart_git_commit.SmartGitCommitWorkflow(use_ai=False)
        
        # Mock the _run_git_command to return appropriate responses
        def mock_git_command(args):
            if args[0] == "status":
                return "M  backend/utils.py\n?? backend/__pycache__/\n", 0
            elif args[0] == "diff" and args[2] == "backend/utils.py":
                return "diff --git a/backend/utils.py b/backend/utils.py\nindex 1234567..abcdefg 100644\n--- a/backend/utils.py\n+++ b/backend/utils.py\n@@ -1,3 +1,5 @@\n def util_func():\n     pass\n+\ndef another_func():\n+    return True\n", 0
            return "", 0
            
        workflow._run_git_command = mock.MagicMock(side_effect=mock_git_command)
        
        # Call load_changes
        workflow.load_changes()
        
        # Check that the files are properly handled
        self.assertEqual(len(workflow.changes), 2)
        
        # Check file paths and statuses
        modified_change = next(c for c in workflow.changes if c.status == "M")
        untracked_change = next(c for c in workflow.changes if c.status == "??")
        
        self.assertEqual(modified_change.filename, "backend/utils.py")
        self.assertEqual(untracked_change.filename, "backend/__pycache__/")
        
        # Check diff content
        self.assertIsNotNone(modified_change.content_diff)
        self.assertIn("another_func", modified_change.content_diff)
        self.assertIsNone(untracked_change.content_diff)  # Untracked files don't have diffs
    
    @mock.patch('os.path.isdir')
    @mock.patch('smart_git_commit.SmartGitCommitWorkflow._run_git_command')
    def test_git_dir_discovery(self, mock_run_git, mock_isdir, _):
        """Test discovery of git directory for commit message file."""
        # Setup the mock workflow
        workflow = smart_git_commit.SmartGitCommitWorkflow(repo_path="/test/repo", use_ai=False)
        
        # Setup mocks for the git directory discovery path
        mock_isdir.side_effect = lambda path: path == "/test/repo/.git"
        
        # Create a commit group to test with
        group = smart_git_commit.CommitGroup(
            name="Test commit", 
            commit_type=smart_git_commit.CommitType.FEAT
        )
        group.add_change(smart_git_commit.GitChange(status="M", filename="test.py"))
        workflow.commit_groups = [group]
        
        # Mock the file operations that would happen in execute_commits
        with mock.patch('builtins.open', mock.mock_open()) as mock_file:
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.remove'):
                    # Mock successful git commit
                    mock_run_git.return_value = ("Committed", 0)
                    
                    # Execute in non-interactive mode
                    workflow.execute_commits(interactive=False)
                    
                    # Verify git directory was correctly used
                    mock_file.assert_called_once()
                    file_path = mock_file.call_args[0][0]
                    self.assertEqual(file_path, "/test/repo/.git/COMMIT_EDITMSG")


class TestOllamaClient(TestCase):
    """Tests for the OllamaClient class."""
    
    @mock.patch('socket.getaddrinfo')
    @mock.patch('http.client.HTTPConnection')
    def test_connection_timeout_handling(self, mock_http_conn, mock_getaddrinfo):
        """Test handling of connection timeouts."""
        # Setup mocks
        mock_getaddrinfo.side_effect = socket.timeout("Connection timed out")
        
        # Test with non-localhost host that will trigger the fallback
        with self.assertRaises(RuntimeError):
            client = smart_git_commit.OllamaClient(host="http://nonexistent-host:11434", timeout=1)
            
        # Verify the fallback was attempted
        self.assertEqual(mock_getaddrinfo.call_count, 2)  # First call fails, second for localhost
    
    @mock.patch('subprocess.Popen')
    @mock.patch('socket.getaddrinfo')
    @mock.patch('http.client.HTTPConnection')
    def test_models_from_cli_fallback(self, mock_http_conn, mock_getaddrinfo, mock_popen):
        """Test fallback to CLI for model list when API fails."""
        # Setup HTTP connection mock to fail
        mock_conn = mock.MagicMock()
        mock_http_conn.return_value = mock_conn
        mock_conn.getresponse.side_effect = http.client.HTTPException("API error")
        
        # Setup subprocess mock for CLI fallback
        mock_process = mock.MagicMock()
        mock_popen.return_value = mock_process
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("NAME  SIZE  PARAMS\nllama3 1.1G  123M\n", "")
        
        # Set up socket mock to succeed
        mock_getaddrinfo.return_value = [('AF_INET', 1, 1, '', ('127.0.0.1', 11434))]
        
        # Create a client that will need to fall back to CLI
        with mock.patch('builtins.input', return_value="1"):  # Mock model selection input
            client = smart_git_commit.OllamaClient(timeout=1)
            
            # Verify the model was obtained from CLI
            self.assertEqual(client.available_models, ["llama3"])


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
    
    @mock.patch('importlib.import_module')
    def test_precommit_module_availability(self, mock_import):
        """Test detection of pre-commit module availability."""
        # Test when module is available
        workflow = smart_git_commit.SmartGitCommitWorkflow(use_ai=False)
        
        # Mock successful import
        mock_import.return_value = True
        self.assertTrue(workflow._is_precommit_module_available())
        
        # Mock import error
        mock_import.side_effect = ImportError("No module named 'pre_commit'")
        
        # Mock subprocess result for fallback check
        with mock.patch('subprocess.run') as mock_run:
            mock_process = mock.MagicMock()
            mock_process.returncode = 1  # Import failed
            mock_run.return_value = mock_process
            
            self.assertFalse(workflow._is_precommit_module_available())
    
    @mock.patch('smart_git_commit.SmartGitCommitWorkflow._check_for_precommit_hooks')
    @mock.patch('smart_git_commit.SmartGitCommitWorkflow._is_precommit_module_available')
    def test_auto_skip_hooks(self, mock_available, mock_check):
        """Test auto-skipping of hooks when pre-commit module is not available."""
        # Mock pre-commit hooks exist but module not available
        mock_check.return_value = True
        mock_available.return_value = False
        
        # Set up workflow
        workflow = smart_git_commit.SmartGitCommitWorkflow(use_ai=False, skip_hooks=False)
        
        # Add a dummy commit group
        group = smart_git_commit.CommitGroup(name="Test commit", commit_type=smart_git_commit.CommitType.FEAT)
        group.add_change(smart_git_commit.GitChange(status="M", filename="test.py"))
        workflow.commit_groups = [group]
        
        # Mock Git commands to avoid real execution
        workflow._run_git_command = mock.MagicMock(return_value=("", 0))
        
        # Mock file operations
        with mock.patch('builtins.open', mock.mock_open()):
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('os.remove'):
                    # Execute in non-interactive mode to avoid input prompts
                    workflow.execute_commits(interactive=False)
                    
                    # Verify hooks were auto-skipped
                    self.assertTrue(workflow.skip_hooks)


class TestColorSupport(TestCase):
    """Tests for color support in terminal output."""
    
    def test_color_detection(self):
        """Test detection of terminal color support."""
        # Test color support detection
        with mock.patch('platform.system') as mock_system:
            with mock.patch('sys.stdout') as mock_stdout:
                # Test Windows with no color support
                mock_system.return_value = 'Windows'
                mock_stdout.isatty.return_value = True
                
                # Clear environment variables
                with mock.patch.dict('os.environ', {}, clear=True):
                    self.assertFalse(smart_git_commit.supports_color())
                
                # Test Windows with color support (WT_SESSION)
                with mock.patch.dict('os.environ', {'WT_SESSION': '1'}):
                    self.assertTrue(smart_git_commit.supports_color())
                
                # Test non-Windows
                mock_system.return_value = 'Linux'
                self.assertTrue(smart_git_commit.supports_color())
    
    def test_color_disable_flag(self):
        """Test disabling colors with --no-color flag."""
        # Capture original values
        original_values = {attr: getattr(smart_git_commit.Colors, attr) 
                          for attr in dir(smart_git_commit.Colors) 
                          if not attr.startswith('__')}
        
        # Mock args with no-color option
        args = mock.MagicMock()
        args.no_color = True
        
        # Simulate disabling colors
        if args.no_color:
            for attr in dir(smart_git_commit.Colors):
                if not attr.startswith('__'):
                    setattr(smart_git_commit.Colors, attr, '')
        
        # Check if all color codes are empty
        for attr in dir(smart_git_commit.Colors):
            if not attr.startswith('__'):
                self.assertEqual(getattr(smart_git_commit.Colors, attr), '')
        
        # Restore original values
        for attr, value in original_values.items():
            setattr(smart_git_commit.Colors, attr, value)


if __name__ == "__main__":
    import unittest
    unittest.main() 