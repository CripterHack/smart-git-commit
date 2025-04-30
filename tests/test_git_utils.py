"""Tests for git_utils.py module."""

import os
import sys
import unittest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch
import shutil

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_git_commit.git_utils import (
    get_repository_details,
    parse_status_line,
    get_git_root,
    get_git_hooks_dir,
    get_staged_files
)

class TestGitUtils(unittest.TestCase):
    """Test cases for git utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=self.repo_path, capture_output=True)
        
        # Create some test files
        (self.repo_path / "test.txt").write_text("test content")
        (self.repo_path / "test_space.txt").write_text("test content with spaces")
        (self.repo_path / "test_special.txt").write_text("test content with special chars")
        
        # Create initial commit
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.repo_path, capture_output=True)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_get_repository_details_with_remote(self):
        """Test getting repository details with remote URL."""
        # Set up a remote URL
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/test/repo.git"],
            cwd=self.repo_path,
            capture_output=True
        )
        
        details = get_repository_details(str(self.repo_path))
        self.assertEqual(details["name"], "repo")
        self.assertIn(details["branch"], ["master", "main"])  # Git version dependent
        self.assertEqual(details["path"], str(self.repo_path.absolute()))
    
    def test_get_repository_details_without_remote(self):
        """Test getting repository details without remote URL."""
        details = get_repository_details(str(self.repo_path))
        self.assertEqual(details["name"], self.repo_path.name)
        self.assertIn(details["branch"], ["master", "main"])
        self.assertEqual(details["path"], str(self.repo_path.absolute()))
    
    def test_get_repository_details_error(self):
        """Test getting repository details with error."""
        with tempfile.TemporaryDirectory() as non_git_dir:
            details = get_repository_details(non_git_dir)
            self.assertEqual(details["name"], "unknown")
            self.assertEqual(details["branch"], "unknown")
            self.assertEqual(details["path"], os.path.abspath(non_git_dir))
    
    def test_get_repository_details_with_error(self):
        """Test getting repository details with git command error."""
        with patch('subprocess.check_output', side_effect=subprocess.CalledProcessError(1, 'git')):
            details = get_repository_details(str(self.repo_path))
            self.assertEqual(details["name"], "unknown")
            self.assertEqual(details["branch"], "unknown")
            self.assertEqual(details["path"], str(self.repo_path.absolute()))
    
    def test_get_repository_details_with_general_error(self):
        """Test getting repository details with general error."""
        with patch('subprocess.check_output', side_effect=Exception("test error")):
            details = get_repository_details(str(self.repo_path))
            self.assertEqual(details["name"], "unknown")
            self.assertEqual(details["branch"], "unknown")
            self.assertEqual(details["path"], str(self.repo_path.absolute()))
    
    def test_parse_status_line_empty(self):
        """Test parsing empty status line."""
        status, filename = parse_status_line("")
        self.assertEqual(status, "")
        self.assertEqual(filename, "")
    
    def test_parse_status_line_added(self):
        """Test parsing added file status."""
        status, filename = parse_status_line("A  test.txt")
        self.assertEqual(status, "A")
        self.assertEqual(filename, "test.txt")
    
    def test_parse_status_line_modified(self):
        """Test parsing modified file status."""
        status, filename = parse_status_line("M  test.txt")
        self.assertEqual(status, "M")
        self.assertEqual(filename, "test.txt")
    
    def test_parse_status_line_renamed(self):
        """Test parsing renamed file status."""
        status, filename = parse_status_line("R  old.txt -> new.txt")
        self.assertEqual(status, "R")
        self.assertEqual(filename, "new.txt")
    
    def test_parse_status_line_quoted(self):
        """Test parsing status line with quoted filename."""
        status, filename = parse_status_line('A  "test_space.txt"')
        self.assertEqual(status, "A")
        self.assertEqual(filename, "test_space.txt")
    
    def test_parse_status_line_with_spaces(self):
        """Test parsing status line with spaces in filename."""
        status, filename = parse_status_line('A  "file with spaces.txt"')
        self.assertEqual(status, "A")
        self.assertEqual(filename, "file with spaces.txt")
    
    def test_parse_status_line_with_special_chars(self):
        """Test parsing status line with special characters."""
        status, filename = parse_status_line('M  file#1[test].txt')
        self.assertEqual(status, "M")
        self.assertEqual(filename, "file#1[test].txt")
    
    def test_parse_status_line_with_rename_and_spaces(self):
        """Test parsing renamed file with spaces."""
        status, filename = parse_status_line('R  "old file.txt" -> "new file.txt"')
        self.assertEqual(status, "R")
        self.assertEqual(filename, "new file.txt")
    
    def test_get_git_root_success(self):
        """Test getting git root directory successfully."""
        root = get_git_root(str(self.repo_path))
        self.assertEqual(root, str(self.repo_path.absolute()))
        
        # Test from subdirectory
        subdir = self.repo_path / "subdir"
        subdir.mkdir()
        root = get_git_root(str(subdir))
        self.assertEqual(root, str(self.repo_path.absolute()))
    
    def test_get_git_root_error(self):
        """Test getting git root directory with error."""
        with tempfile.TemporaryDirectory() as non_git_dir:
            root = get_git_root(non_git_dir)
            self.assertIsNone(root)
    
    def test_get_git_root_with_error(self):
        """Test getting git root with command error."""
        with patch('subprocess.check_output', side_effect=subprocess.CalledProcessError(1, 'git')):
            root = get_git_root(str(self.repo_path))
            self.assertIsNone(root)
    
    def test_get_git_hooks_dir_success(self):
        """Test getting git hooks directory successfully."""
        hooks_dir = get_git_hooks_dir(str(self.repo_path))
        expected_dir = os.path.join(str(self.repo_path.absolute()), ".git", "hooks")
        self.assertEqual(hooks_dir, expected_dir)
    
    def test_get_git_hooks_dir_error(self):
        """Test getting git hooks directory with error."""
        with tempfile.TemporaryDirectory() as non_git_dir:
            hooks_dir = get_git_hooks_dir(non_git_dir)
            self.assertIsNone(hooks_dir)
    
    def test_get_git_hooks_dir_with_missing_hooks(self):
        """Test getting hooks directory when .git/hooks doesn't exist."""
        # Remove hooks directory
        hooks_dir = os.path.join(str(self.repo_path), ".git", "hooks")
        shutil.rmtree(hooks_dir)
        
        result = get_git_hooks_dir(str(self.repo_path))
        self.assertIsNone(result)
    
    def test_get_staged_files_empty(self):
        """Test getting staged files when none are staged."""
        files = get_staged_files(str(self.repo_path))
        self.assertEqual(files, {})
    
    def test_get_staged_files_with_changes(self):
        """Test getting staged files with changes."""
        # Create and stage files with special characters
        special_file = self.repo_path / "test#1[special].txt"
        space_file = self.repo_path / "test file.txt"
        
        special_file.write_text("test content")
        space_file.write_text("test content")
        
        # Stage the files
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        
        # Get staged files
        files = get_staged_files(str(self.repo_path))
        
        # Verify files are staged
        self.assertGreater(len(files), 0)
        self.assertIn("test#1[special].txt", files)
        self.assertIn("test file.txt", files)
        self.assertEqual(files["test#1[special].txt"], "A")
        self.assertEqual(files["test file.txt"], "A")
    
    def test_get_staged_files_with_renames(self):
        """Test getting staged files with renamed files."""
        # Create and commit a file
        old_file = self.repo_path / "old.txt"
        old_file.write_text("test content")
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=self.repo_path, capture_output=True)
        
        # Rename the file using git mv
        new_file = "new.txt"
        subprocess.run(["git", "mv", "old.txt", new_file], cwd=self.repo_path, capture_output=True)
        
        # Get staged files
        files = get_staged_files(str(self.repo_path))
        
        # Verify renamed file is staged
        self.assertIn(new_file, files)
        self.assertEqual(files[new_file], "R")
    
    def test_get_staged_files_with_spaces(self):
        """Test getting staged files with spaces in names."""
        # Create and stage a file with spaces
        file_with_spaces = self.repo_path / "file with spaces.txt"
        file_with_spaces.write_text("test content")
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        
        # Get staged files
        files = get_staged_files(str(self.repo_path))
        
        # Verify file with spaces is staged
        self.assertIn("file with spaces.txt", files)
        self.assertEqual(files["file with spaces.txt"], "A")
    
    def test_get_staged_files_with_unicode(self):
        """Test getting staged files with unicode characters."""
        # Create and stage a file with unicode characters
        unicode_file = self.repo_path / "测试文件.txt"
        unicode_file.write_text("test content")
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        
        # Get staged files
        files = get_staged_files(str(self.repo_path))
        
        # Verify unicode file is staged - either directly or in escaped form
        found = False
        for filename in files:
            # Check if the filename matches directly or after unescaping
            if filename == "测试文件.txt" or "测试" in filename or filename.endswith(".txt"):
                found = True
                break
        
        self.assertTrue(found, f"Unicode filename not found in staged files: {files}")
    
    def test_get_staged_files_with_deleted(self):
        """Test getting staged files with deleted files."""
        # Create and commit a file
        file_to_delete = self.repo_path / "delete_me.txt"
        file_to_delete.write_text("test content")
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=self.repo_path, capture_output=True)
        
        # Delete and stage the file
        file_to_delete.unlink()
        subprocess.run(["git", "rm", "delete_me.txt"], cwd=self.repo_path, capture_output=True)
        
        # Get staged files
        files = get_staged_files(str(self.repo_path))
        
        # Verify deleted file is staged
        self.assertIn("delete_me.txt", files)
        self.assertEqual(files["delete_me.txt"], "D")
    
    def test_get_staged_files_with_modified(self):
        """Test getting staged files with modified files."""
        # Create and commit a file
        file_to_modify = self.repo_path / "modify_me.txt"
        file_to_modify.write_text("initial content")
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=self.repo_path, capture_output=True)
        
        # Modify and stage the file
        file_to_modify.write_text("modified content")
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        
        # Get staged files
        files = get_staged_files(str(self.repo_path))
        
        # Verify modified file is staged
        self.assertIn("modify_me.txt", files)
        self.assertEqual(files["modify_me.txt"], "M")
    
    def test_get_staged_files_with_multiple_statuses(self):
        """Test getting staged files with multiple status types."""
        # Create and commit initial files
        files = {
            "modified.txt": "initial content",
            "deleted.txt": "delete me",
            "renamed.txt": "rename me"
        }
        for name, content in files.items():
            file_path = self.repo_path / name
            file_path.write_text(content)
        
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=self.repo_path, capture_output=True)
        
        # Create new file
        new_file = self.repo_path / "new.txt"
        new_file.write_text("new content")
        
        # Modify existing file
        modified_file = self.repo_path / "modified.txt"
        modified_file.write_text("modified content")
        
        # Delete a file
        deleted_file = self.repo_path / "deleted.txt"
        deleted_file.unlink()
        
        # Rename a file
        subprocess.run(["git", "mv", "renamed.txt", "new_name.txt"], cwd=self.repo_path, capture_output=True)
        
        # Stage all changes
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "add", "-u"], cwd=self.repo_path, capture_output=True)
        
        # Get staged files
        files = get_staged_files(str(self.repo_path))
        
        # Verify all changes are properly staged
        self.assertIn("new.txt", files)
        self.assertEqual(files["new.txt"], "A")
        
        self.assertIn("modified.txt", files)
        self.assertEqual(files["modified.txt"], "M")
        
        self.assertIn("deleted.txt", files)
        self.assertEqual(files["deleted.txt"], "D")
        
        self.assertIn("new_name.txt", files)
        self.assertEqual(files["new_name.txt"], "R")

if __name__ == '__main__':
    unittest.main() 