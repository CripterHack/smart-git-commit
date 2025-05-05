#!/usr/bin/env python3
"""Tests for squash.py module."""

import os
import sys
import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import datetime
import json

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions directly, not a class
from smart_git_commit.squash import (
    CommitInfo,
    SquashStrategy,
    get_recent_commits,
    get_commits_since,
    get_changed_files_for_commit,
    calculate_file_similarity,
    is_conventional_commit,
    calculate_semantic_similarity,
    find_squashable_commits,
    generate_squashed_commit_message,
    squash_commits,
    analyze_commits_for_squashing,
    squash_commit_group,
    run_squash_command
)

class TestSquashingUtilities(unittest.TestCase):
    """Tests for utility functions in squash.py."""
    
    def test_calculate_file_similarity_identical(self):
        """Test calculating similarity between identical file lists."""
        files1 = ["file1.py", "file2.py", "file3.py"]
        files2 = ["file1.py", "file2.py", "file3.py"]
        
        similarity = calculate_file_similarity(files1, files2)
        self.assertEqual(similarity, 1.0)
    
    def test_calculate_file_similarity_partial(self):
        """Test calculating similarity between partially overlapping file lists."""
        files1 = ["file1.py", "file2.py", "file3.py"]
        files2 = ["file1.py", "file2.py", "file4.py"]
        
        similarity = calculate_file_similarity(files1, files2)
        self.assertEqual(similarity, 0.5)
    
    def test_calculate_file_similarity_disjoint(self):
        """Test calculating similarity between completely different file lists."""
        files1 = ["file1.py", "file2.py", "file3.py"]
        files2 = ["file4.py", "file5.py", "file6.py"]
        
        similarity = calculate_file_similarity(files1, files2)
        self.assertEqual(similarity, 0.0)
    
    def test_calculate_file_similarity_empty(self):
        """Test calculating similarity with empty file lists."""
        files1 = []
        files2 = ["file1.py", "file2.py"]
        
        similarity = calculate_file_similarity(files1, files2)
        self.assertEqual(similarity, 0.0)
        
        similarity = calculate_file_similarity(files2, files1)
        self.assertEqual(similarity, 0.0)
        
        similarity = calculate_file_similarity([], [])
        self.assertEqual(similarity, 0.0)

class TestCommitInfo(unittest.TestCase):
    """Tests for the CommitInfo class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.commit = CommitInfo(
            hash="abcdef1234567890",
            author="Test User",
            date=datetime.datetime.now(),
            subject="feat(api): add new endpoint",
            body="This adds a new REST API endpoint for users.",
            changed_files=["src/api/endpoints.py", "tests/test_endpoints.py"]
        )
    
    def test_commit_info_properties(self):
        """Test CommitInfo properties."""
        self.assertEqual(self.commit.hash, "abcdef1234567890")
        self.assertEqual(self.commit.author, "Test User")
        self.assertEqual(self.commit.subject, "feat(api): add new endpoint")
        self.assertEqual(self.commit.body, "This adds a new REST API endpoint for users.")
        self.assertListEqual(self.commit.changed_files, ["src/api/endpoints.py", "tests/test_endpoints.py"])
    
    def test_is_merge_commit(self):
        """Test checking if a commit is a merge commit."""
        # Regular commit
        self.assertFalse(self.commit.is_merge_commit())
        
        # Merge commit
        merge_commit = CommitInfo(
            hash="abcdef1234567890",
            author="Test User",
            date=datetime.datetime.now(),
            subject="Merge branch 'feature' into main",
            body="",
            changed_files=[]
        )
        self.assertTrue(merge_commit.is_merge_commit())

class TestGitCommands(unittest.TestCase):
    """Tests for Git command execution functions."""
    
    def setUp(self):
        """Set up test environment with a temporary Git repository."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo_path, capture_output=True)
        
        # Create and commit a file
        file_path = self.repo_path / "test.txt"
        file_path.write_text("Initial content")
        
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.repo_path, capture_output=True)
    
    def tearDown(self):
        """Clean up the temporary directory."""
        os.chdir(os.path.dirname(__file__))  # Change back to test directory
        shutil.rmtree(self.temp_dir)
    
    @patch('smart_git_commit.squash.get_changed_files_for_commit')
    @patch('subprocess.run')
    def test_get_recent_commits(self, mock_run, mock_get_changed_files):
        """Test getting recent commits."""
        # Mock the git log command to return hashes
        mock_log_output = MagicMock()
        mock_log_output.stdout = "hash1\nhash2"
        mock_log_output.check_returncode = lambda: None

        # Mock the git show command (called inside from_commit_hash)
        mock_show_output_hash1 = MagicMock()
        # Format: %H\n%an\n%at\n%s\n%b\n%P
        mock_show_output_hash1.stdout = f"hash1\nauthor1\n{int(datetime.datetime(2023, 1, 2, 10, 0, 0).timestamp())}\nsubject1\nbody1\nparent1"
        mock_show_output_hash1.check_returncode = lambda: None

        mock_show_output_hash2 = MagicMock()
        mock_show_output_hash2.stdout = f"hash2\nauthor2\n{int(datetime.datetime(2023, 1, 1, 9, 0, 0).timestamp())}\nsubject2\nbody2\nparent2"
        mock_show_output_hash2.check_returncode = lambda: None

        # Configure subprocess.run mock based on command args
        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0]
            if "log" in cmd:
                return mock_log_output
            elif "show" in cmd and "hash1" in cmd:
                return mock_show_output_hash1
            elif "show" in cmd and "hash2" in cmd:
                return mock_show_output_hash2
            # Add mocks for git diff-tree called by get_changed_files_for_commit if not mocking that function directly
            elif "diff-tree" in cmd:
                 mock_diff_tree = MagicMock()
                 mock_diff_tree.stdout = "" # Return empty string for simplicity, or specific files if needed
                 mock_diff_tree.check_returncode = lambda: None
                 return mock_diff_tree
            else:
                # Default mock for other calls like git config, init etc. during setup if needed
                default_mock = MagicMock()
                default_mock.stdout = ""
                default_mock.check_returncode = lambda: None
                return default_mock

        mock_run.side_effect = mock_run_side_effect

        # Mock get_changed_files_for_commit directly to simplify
        def mock_get_files_side_effect(commit_hash, repo_path):
            if commit_hash == "hash1":
                return ["file1.py"]
            elif commit_hash == "hash2":
                return ["file2.py", "file3.py"]
            return []
        mock_get_changed_files.side_effect = mock_get_files_side_effect

        # Call the function
        commits = get_recent_commits(count=2, repo_path=str(self.repo_path))

        # Verify results
        self.assertEqual(len(commits), 2)
        self.assertEqual(commits[0].hash, "hash1")
        self.assertEqual(commits[0].author, "author1")
        self.assertEqual(commits[0].subject, "subject1")
        self.assertEqual(commits[0].body, "body1")
        self.assertEqual(commits[0].changed_files, ["file1.py"])
        self.assertEqual(commits[0].parents, ["parent1"])
        self.assertEqual(commits[0].date, datetime.datetime(2023, 1, 2, 10, 0, 0))

        self.assertEqual(commits[1].hash, "hash2")
        self.assertEqual(commits[1].author, "author2")
        self.assertEqual(commits[1].subject, "subject2")
        self.assertEqual(commits[1].body, "body2")
        self.assertEqual(commits[1].changed_files, ["file2.py", "file3.py"])
        self.assertEqual(commits[1].parents, ["parent2"])
        self.assertEqual(commits[1].date, datetime.datetime(2023, 1, 1, 9, 0, 0))

        # Check that subprocess.run and get_changed_files_for_commit were called
        self.assertTrue(mock_run.called)
        self.assertTrue(mock_get_changed_files.called)

class TestSquashingLogic(unittest.TestCase):
    """Tests for commit squashing logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample commits for testing
        self.commits = [
            CommitInfo(
                hash="hash1",
                author="author1",
                date=datetime.datetime(2023, 1, 3, 10, 0, 0),
                subject="feat(api): add new endpoint",
                body="This adds a new REST API endpoint for users.",
                changed_files=["src/api/endpoints.py", "tests/test_endpoints.py"]
            ),
            CommitInfo(
                hash="hash2",
                author="author1",
                date=datetime.datetime(2023, 1, 2, 10, 0, 0),
                subject="fix(api): handle edge case",
                body="Fix handling of edge case in API endpoint.",
                changed_files=["src/api/endpoints.py"]
            ),
            CommitInfo(
                hash="hash3",
                author="author2",
                date=datetime.datetime(2023, 1, 1, 10, 0, 0),
                subject="docs: update README",
                body="Update README with new information.",
                changed_files=["README.md"]
            )
        ]
    
    def test_find_squashable_commits_related_files(self):
        """Test finding squashable commits with RELATED_FILES strategy."""
        groups = find_squashable_commits(
            self.commits,
            strategy=SquashStrategy.RELATED_FILES,
            similarity_threshold=0.5
        )
        
        # Should find one group of the first two commits (they share endpoints.py)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(groups[0][0].hash, "hash1")
        self.assertEqual(groups[0][1].hash, "hash2")
    
    def test_find_squashable_commits_same_author(self):
        """Test finding squashable commits with SAME_AUTHOR strategy."""
        groups = find_squashable_commits(
            self.commits,
            strategy=SquashStrategy.SAME_AUTHOR,
            time_window_minutes=60*24*7  # 1 week
        )
        
        # Should find one group of the first two commits (same author)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(groups[0][0].hash, "hash1")
        self.assertEqual(groups[0][1].hash, "hash2")
    
    def test_find_squashable_commits_time_window(self):
        """Test finding squashable commits with TIME_WINDOW strategy."""
        groups = find_squashable_commits(
            self.commits,
            strategy=SquashStrategy.TIME_WINDOW,
            time_window_minutes=60*24*7  # 1 week
        )
        
        # Should find one group of all commits (all within a week)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)
    
    def test_find_squashable_commits_conventional(self):
        """Test finding squashable commits with CONVENTIONAL strategy."""
        groups = find_squashable_commits(
            self.commits,
            strategy=SquashStrategy.CONVENTIONAL
        )
        
        # Should find one group of the first two commits (both API related)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(groups[0][0].hash, "hash1")
        self.assertEqual(groups[0][1].hash, "hash2")
    
    def test_find_squashable_commits_semantic(self):
        """Test finding squashable commits with SEMANTIC strategy."""
        # Create a mock workflow with AI client
        mock_workflow = MagicMock()
        mock_ai_client = MagicMock()
        mock_workflow.ai_client = mock_ai_client
        
        # Mock AI client response
        mock_ai_client.generate.return_value = json.dumps({
            "groups": [["hash1", "hash2"]]
        })
        
        groups = find_squashable_commits(
            self.commits,
            strategy=SquashStrategy.SEMANTIC,
            workflow=mock_workflow
        )
        
        # Should find one group of the first two commits (as defined by mock AI)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(groups[0][0].hash, "hash1")
        self.assertEqual(groups[0][1].hash, "hash2")
    
    def test_find_squashable_commits_empty(self):
        """Test finding squashable commits with empty input."""
        groups = find_squashable_commits([])
        self.assertEqual(len(groups), 0)
    
    def test_find_squashable_commits_single(self):
        """Test finding squashable commits with only one commit."""
        groups = find_squashable_commits([self.commits[0]])
        self.assertEqual(len(groups), 0)

    def test_find_squashable_commits_related_files_heuristic(self):
        """Test RELATED_FILES strategy uses heuristic score now."""
        # Commit 2 is now only barely related by files, but close in time and same author
        self.commits[1].changed_files = ["src/api/util.py"] # Less file overlap
        self.commits[1].date=datetime.datetime(2023, 1, 3, 10, 10, 0) # 10 mins later
        
        # Expecting the heuristic score (combining files, time, author) to still pass threshold
        groups = find_squashable_commits(
            self.commits.copy(), # Use copy as the function reverses in place
            strategy=SquashStrategy.RELATED_FILES,
            similarity_threshold=0.3 # Lower threshold for heuristic test
        )
        
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(groups[0][0].hash, "hash1") 
        self.assertEqual(groups[0][1].hash, "hash2")

    def test_find_squashable_commits_same_author_heuristic(self):
        """Test SAME_AUTHOR strategy uses heuristic score now."""
        # Make commit 2 by different author but very related files/time
        self.commits[1].author = "author2"
        self.commits[1].date=datetime.datetime(2023, 1, 3, 10, 5, 0) # 5 mins later
        self.commits[1].changed_files=["src/api/endpoints.py", "tests/test_endpoints.py"]

        groups = find_squashable_commits(
            self.commits.copy(),
            strategy=SquashStrategy.SAME_AUTHOR,
            similarity_threshold=0.5 # Heuristic score needs to pass this
        )
        
        # Should NOT find a group because author doesn't match, even if heuristic is high
        self.assertEqual(len(groups), 0)

    def test_find_squashable_commits_auto_no_ai(self):
        """Test auto strategy fallback without AI."""
        # Make commits non-conventional but related by file
        self.commits[0].subject = "Added endpoint"
        self.commits[1].subject = "Fixed endpoint"
        self.commits[1].changed_files = ["src/api/endpoints.py"]
        self.commits[0].changed_files = ["src/api/endpoints.py"]

        groups = find_squashable_commits(
            self.commits.copy(),
            strategy="auto", # Use string auto
            similarity_threshold=0.5, # Heuristic threshold
            workflow=None # No AI workflow
        )

        # Without AI, auto should fallback to CONVENTIONAL (fails), then HEURISTIC (passes due to file)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

class TestSquashMessageGeneration(unittest.TestCase):
    """Tests for squashed commit message generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample commits for testing
        self.commits = [
            CommitInfo(
                hash="hash1",
                author="author1",
                date=datetime.datetime(2023, 1, 3, 10, 0, 0),
                subject="feat(api): add new endpoint",
                body="This adds a new REST API endpoint for users.",
                changed_files=["src/api/endpoints.py", "tests/test_endpoints.py"]
            ),
            CommitInfo(
                hash="hash2",
                author="author1",
                date=datetime.datetime(2023, 1, 2, 10, 0, 0),
                subject="fix(api): handle edge case",
                body="Fix handling of edge case in API endpoint.",
                changed_files=["src/api/endpoints.py"]
            )
        ]
    
    def test_generate_squashed_commit_message(self):
        """Test generating a squashed commit message."""
        message = generate_squashed_commit_message(self.commits)
        
        # Check content
        self.assertIn("feat(api): add new endpoint", message)
        self.assertIn("fix(api): handle edge case", message)
        self.assertIn("This adds a new REST API endpoint for users.", message)
        self.assertIn("Fix handling of edge case in API endpoint.", message)
        self.assertIn("Files changed:", message)
        self.assertIn("src/api/endpoints.py", message)
        self.assertIn("tests/test_endpoints.py", message)
    
    def test_generate_squashed_commit_message_no_body(self):
        """Test generating a squashed message without body text."""
        commits = [
            CommitInfo(
                hash="hash1",
                author="author1",
                date=datetime.datetime(2023, 1, 3, 10, 0, 0),
                subject="feat(api): add new endpoint",
                body="",
                changed_files=["src/api/endpoints.py"]
            ),
            CommitInfo(
                hash="hash2",
                author="author1",
                date=datetime.datetime(2023, 1, 2, 10, 0, 0),
                subject="fix(api): handle edge case",
                body="",
                changed_files=["src/api/endpoints.py"]
            )
        ]
        
        message = generate_squashed_commit_message(commits)
        
        # Check content
        self.assertIn("feat(api): add new endpoint", message)
        self.assertIn("fix(api): handle edge case", message)
        self.assertIn("Files changed:", message)
        self.assertIn("src/api/endpoints.py", message)
    
    def test_generate_squashed_commit_message_empty(self):
        """Test generating a squashed message with empty commit list."""
        message = generate_squashed_commit_message([])
        self.assertIn("Squashed commit", message)

class TestSquashingOperations(unittest.TestCase):
    """Tests for the actual squashing operations."""
    
    def setUp(self):
        """Set up test environment with a temporary Git repository."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo_path, capture_output=True)
    
    def tearDown(self):
        """Clean up the temporary directory."""
        os.chdir(os.path.dirname(__file__))  # Change back to test directory
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    def test_squash_commits(self, mock_temp_file, mock_run):
        """Test squashing commits."""
        # Setup mock
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.name = "/tmp/mockfile"
        mock_temp_file.return_value = mock_file
        
        # Setup mock subprocess run to return parent commit hash
        mock_run.side_effect = [
            MagicMock(stdout="parent_hash\n"),  # git rev-parse
            MagicMock(),  # interactive rebase
            MagicMock()   # cleanup
        ]
        
        # Create test commits
        commits = [
            CommitInfo(
                hash="hash1",
                author="author1",
                date=datetime.datetime(2023, 1, 3, 10, 0, 0),
                subject="feat(api): add new endpoint",
                body="This adds a new REST API endpoint for users.",
                changed_files=["src/api/endpoints.py"]
            ),
            CommitInfo(
                hash="hash2",
                author="author1",
                date=datetime.datetime(2023, 1, 2, 10, 0, 0),
                subject="fix(api): handle edge case",
                body="Fix handling of edge case in API endpoint.",
                changed_files=["src/api/endpoints.py"]
            )
        ]
        
        # Call the function
        result = squash_commits(commits, repo_path=str(self.repo_path), interactive=True)
        
        # Verify results
        self.assertTrue(result)
        self.assertEqual(mock_run.call_count, 2)
        
        # Verify rebase command
        rebase_call = mock_run.call_args_list[1]
        self.assertIn("rebase", rebase_call[0][0])
    
    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    def test_squash_commits_non_interactive(self, mock_temp_file, mock_run):
        """Test squashing commits in non-interactive mode."""
        # Setup mock
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.name = "/tmp/mockfile"
        mock_temp_file.return_value = mock_file
        
        # Setup mock subprocess run to return parent commit hash
        mock_run.side_effect = [
            MagicMock(stdout="parent_hash\n"),  # git rev-parse
            MagicMock(),  # git reset
            MagicMock()   # git commit
        ]
        
        # Create test commits
        commits = [
            CommitInfo(
                hash="hash1",
                author="author1",
                date=datetime.datetime(2023, 1, 3, 10, 0, 0),
                subject="feat(api): add new endpoint",
                body="This adds a new REST API endpoint for users.",
                changed_files=["src/api/endpoints.py"]
            ),
            CommitInfo(
                hash="hash2",
                author="author1",
                date=datetime.datetime(2023, 1, 2, 10, 0, 0),
                subject="fix(api): handle edge case",
                body="Fix handling of edge case in API endpoint.",
                changed_files=["src/api/endpoints.py"]
            )
        ]
        
        # Call the function
        result = squash_commits(commits, repo_path=str(self.repo_path), interactive=False)
        
        # Verify results
        self.assertTrue(result)
        
        # Verify reset and commit commands
        reset_call = mock_run.call_args_list[1]
        commit_call = mock_run.call_args_list[2]
        
        self.assertIn("reset", reset_call[0][0])
        self.assertIn("commit", commit_call[0][0])
    
    @patch('subprocess.run')
    def test_squash_commits_error(self, mock_run):
        """Test error handling when squashing commits."""
        # Setup mock to raise an exception
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')
        
        # Create test commits
        commits = [
            CommitInfo(
                hash="hash1",
                author="author1",
                date=datetime.datetime(2023, 1, 3, 10, 0, 0),
                subject="feat(api): add new endpoint",
                body="",
                changed_files=["src/api/endpoints.py"]
            ),
            CommitInfo(
                hash="hash2",
                author="author1",
                date=datetime.datetime(2023, 1, 2, 10, 0, 0),
                subject="fix(api): handle edge case",
                body="",
                changed_files=["src/api/endpoints.py"]
            )
        ]
        
        # Call the function
        result = squash_commits(commits, repo_path=str(self.repo_path))
        
        # Verify results
        self.assertFalse(result)
    
    @patch('smart_git_commit.squash.get_recent_commits')
    @patch('smart_git_commit.squash.find_squashable_commits')
    def test_analyze_commits_for_squashing(self, mock_find, mock_get):
        """Test analyzing commits for squashing."""
        # Setup mocks
        mock_commits = [MagicMock(), MagicMock()]
        mock_groups = [[mock_commits[0]], [mock_commits[1]]]
        
        mock_get.return_value = mock_commits
        mock_find.return_value = mock_groups
        
        # Call the function
        result = analyze_commits_for_squashing(
            repo_path=str(self.repo_path),
            count=10,
            strategy=SquashStrategy.RELATED_FILES
        )
        
        # Verify results
        self.assertEqual(result, mock_groups)
        mock_get.assert_called_once_with(10, str(self.repo_path))
        mock_find.assert_called_once()
    
    @patch('smart_git_commit.squash.squash_commits')
    def test_squash_commit_group(self, mock_squash):
        """Test squashing a commit group."""
        # Setup mock
        mock_squash.return_value = True
        
        # Call the function
        result = squash_commit_group(
            [MagicMock(), MagicMock()],
            repo_path=str(self.repo_path),
            interactive=True
        )
        
        # Verify results
        self.assertTrue(result)
        mock_squash.assert_called_once()

class TestSquashCommand(unittest.TestCase):
    """Tests for the squash command interface."""
    
    def setUp(self):
        """Set up test environment."""
        self.repo_path = "/tmp/mock_repo"
    
    @patch('smart_git_commit.squash.analyze_commits_for_squashing')
    @patch('smart_git_commit.squash.squash_commit_group')
    @patch('builtins.input')
    def test_run_squash_command_interactive(self, mock_input, mock_squash, mock_analyze):
        """Test running the squash command in interactive mode."""
        # Setup mocks
        mock_commits = [MagicMock(), MagicMock()]
        mock_groups = [[mock_commits[0], mock_commits[1]]]
        
        mock_analyze.return_value = mock_groups
        mock_input.return_value = 'y'
        mock_squash.return_value = True
        
        # Call the function
        result = run_squash_command(
            repo_path=self.repo_path,
            count=10,
            strategy=SquashStrategy.RELATED_FILES,
            interactive=True
        )
        
        # Verify results
        self.assertTrue(result)
        mock_analyze.assert_called_once()
        mock_input.assert_called_once()
        mock_squash.assert_called_once()
    
    @patch('smart_git_commit.squash.analyze_commits_for_squashing')
    @patch('smart_git_commit.squash.squash_commit_group')
    def test_run_squash_command_non_interactive(self, mock_squash, mock_analyze):
        """Test running the squash command in non-interactive mode."""
        # Setup mocks
        mock_commits = [MagicMock(), MagicMock()]
        mock_groups = [[mock_commits[0], mock_commits[1]]]
        
        mock_analyze.return_value = mock_groups
        mock_squash.return_value = True
        
        # Call the function
        result = run_squash_command(
            repo_path=self.repo_path,
            count=10,
            strategy=SquashStrategy.RELATED_FILES,
            interactive=False
        )
        
        # Verify results
        self.assertTrue(result)
        mock_analyze.assert_called_once()
        mock_squash.assert_called_once()
    
    @patch('smart_git_commit.squash.analyze_commits_for_squashing')
    @patch('smart_git_commit.squash.squash_commit_group')
    def test_run_squash_command_auto_strategy(self, mock_squash, mock_analyze):
        """Test running the squash command with auto strategy selection."""
        # Setup mocks
        mock_commits = [MagicMock(), MagicMock()]
        mock_groups = [[mock_commits[0], mock_commits[1]]]
        
        mock_analyze.return_value = mock_groups
        mock_squash.return_value = True
        
        # Create mock workflow with AI client
        mock_workflow = MagicMock()
        mock_workflow.ai_client = MagicMock()
        
        # Call the function
        result = run_squash_command(
            repo_path=self.repo_path,
            count=10,
            strategy="auto",
            interactive=False,
            workflow=mock_workflow
        )
        
        # Verify results
        self.assertTrue(result)
        mock_analyze.assert_called_once()
        
        # Should use SEMANTIC strategy with AI client
        self.assertEqual(mock_analyze.call_args[1]["strategy"], SquashStrategy.SEMANTIC)
    
    @patch('smart_git_commit.squash.analyze_commits_for_squashing')
    @patch('smart_git_commit.squash.squash_commit_group')
    @patch('builtins.input')
    def test_run_squash_command_quit(self, mock_input, mock_squash, mock_analyze):
        """Test quitting from interactive squash command."""
        # Setup mocks
        mock_commits = [MagicMock(), MagicMock()]
        mock_groups = [[mock_commits[0], mock_commits[1]]]
        
        mock_analyze.return_value = mock_groups
        mock_input.return_value = 'q'
        
        # Call the function
        result = run_squash_command(
            repo_path=self.repo_path,
            count=10,
            strategy=SquashStrategy.RELATED_FILES,
            interactive=True
        )
        
        # Verify results
        self.assertTrue(result)
        mock_analyze.assert_called_once()
        mock_input.assert_called_once()
        mock_squash.assert_not_called()
    
    @patch('smart_git_commit.squash.analyze_commits_for_squashing')
    def test_run_squash_command_no_groups(self, mock_analyze):
        """Test running squash command when no squashable groups are found."""
        # Setup mock
        mock_analyze.return_value = []
        
        # Call the function
        result = run_squash_command(
            repo_path=self.repo_path,
            count=10,
            strategy=SquashStrategy.RELATED_FILES,
            interactive=True
        )
        
        # Verify results
        self.assertTrue(result)
        mock_analyze.assert_called_once()

# Add tests for the new heuristic function
class TestHeuristicSimilarity(unittest.TestCase):
    """Tests for the _calculate_heuristic_similarity function."""
    
    def test_heuristic_similarity_strong_match(self):
        """Test heuristic similarity with strong matching signals."""
        commit1 = CommitInfo(
            hash="h1", author="user", date=datetime.datetime(2023, 1, 1, 10, 0, 0),
            subject="feat: implement initial feature", body="", changed_files=["a.py", "b.py"]
        )
        commit2 = CommitInfo(
            hash="h2", author="user", date=datetime.datetime(2023, 1, 1, 10, 5, 0), # 5 mins later
            subject="fix typo", body="", changed_files=["a.py"] # Related file + trivial msg
        )
        
        # Import the heuristic function directly for testing (it's protected, but needed for unit test)
        from smart_git_commit.squash import _calculate_heuristic_similarity
        score = _calculate_heuristic_similarity(commit1, commit2)
        # Expect high similarity due to same author, close time, related file, trivial message
        self.assertGreater(score, 0.7, "Expected high heuristic similarity")

    def test_heuristic_similarity_weak_match(self):
        """Test heuristic similarity with weak matching signals."""
        commit1 = CommitInfo(
            hash="h1", author="user1", date=datetime.datetime(2023, 1, 1, 10, 0, 0),
            subject="feat: implement feature A", body="", changed_files=["a.py"]
        )
        commit2 = CommitInfo(
            hash="h2", author="user2", date=datetime.datetime(2023, 1, 5, 10, 0, 0), # Days later
            subject="refactor: unrelated module X", body="", changed_files=["x.py"]
        )
        
        from smart_git_commit.squash import _calculate_heuristic_similarity
        score = _calculate_heuristic_similarity(commit1, commit2)
        # Expect low similarity due to different author, time, files, messages
        self.assertLess(score, 0.2, "Expected low heuristic similarity")

if __name__ == '__main__':
    unittest.main() 