#!/usr/bin/env python3
"""
Tests for the Git Hooks integration functionality.
"""

import os
import sys
import tempfile
import subprocess
import unittest
from unittest import TestCase, mock
import platform
import json
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock, mock_open
import logging

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_git_commit.hooks import (
    list_installed_hooks,
    is_hook_installed,
    install_git_hooks,
    remove_git_hooks,
    get_git_root,
    get_git_hooks_dir,
    create_hook_script,
    run_hook,
    is_windows,
    SUPPORTED_HOOKS,
    check_husky_compatibility,
    install_husky,
    remove_husky,
    GitHook,
    install_hook,
    remove_hook,
    get_hook_path,
    get_hook_script,
    get_hook_types,
    validate_hook_type,
    format_hook_script
)

# Determine if git-dependent tests should be skipped
SKIP_GIT_TESTS = os.environ.get('SKIP_GIT_TESTS', '0') == '1'


class TestHooksUtils(unittest.TestCase):
    """Test utility functions in hooks module."""
    
    def test_is_windows(self):
        """Test is_windows function."""
        # This will always return a boolean regardless of platform
        result = is_windows()
        self.assertIsInstance(result, bool)
        
        # On Windows, should return True
        with patch('platform.system', return_value='Windows'):
            self.assertTrue(is_windows())
        
        # On Unix-like, should return False
        with patch('platform.system', return_value='Linux'):
            self.assertFalse(is_windows())
    
    def test_get_smart_git_commit_command(self):
        """Test the get_smart_git_commit_command function."""
        from smart_git_commit.hooks import get_smart_git_commit_command
        
        # Test for pre-commit
        pre_commit_cmd = get_smart_git_commit_command('pre-commit')
        self.assertIn('smart-git-commit', pre_commit_cmd)
        self.assertIn('--non-interactive', pre_commit_cmd)
        
        # Test for pre-push
        pre_push_cmd = get_smart_git_commit_command('pre-push')
        self.assertIn('smart-git-commit', pre_push_cmd)
        self.assertIn('--non-interactive', pre_push_cmd)
        self.assertIn('--analyze-only', pre_push_cmd)
        
        # Test for post-merge
        post_merge_cmd = get_smart_git_commit_command('post-merge')
        self.assertIn('smart-git-commit', post_merge_cmd)
        self.assertIn('--non-interactive', post_merge_cmd)
        self.assertIn('--analyze-only', post_merge_cmd)
        
        # Test for unsupported hook type
        other_cmd = get_smart_git_commit_command('other-hook')
        self.assertEqual(other_cmd, 'smart-git-commit')


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
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        # Write the file
        with open(path, "w") as f:
            f.write(content)
            
    def skip_test(self):
        """Skip the current test."""
        raise unittest.SkipTest("Test skipped because SKIP_GIT_TESTS is set")


class TestHooksFunctionality(TestCase):
    """Tests for the hooks functionality."""
    
    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    def test_get_git_root(self):
        """Test the get_git_root function."""
        with TestMockGitRepository() as repo_path:
            if repo_path is None:
                return  # Test was skipped
                
            # Get the git root
            root = get_git_root(repo_path)
            
            # Should return the repository path
            self.assertEqual(root, repo_path)
            
            # Test with non-git directory
            with tempfile.TemporaryDirectory() as temp_dir:
                self.assertIsNone(get_git_root(temp_dir))
    
    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    def test_get_git_hooks_dir(self):
        """Test the get_git_hooks_dir function."""
        with TestMockGitRepository() as repo_path:
            if repo_path is None:
                return  # Test was skipped
                
            # Get the hooks directory
            hooks_dir = get_git_hooks_dir(repo_path)
            
            # Should return a path ending with .git/hooks
            self.assertTrue(hooks_dir.endswith('.git/hooks') or hooks_dir.endswith('.git\\hooks'))
            
            # Test with non-git directory
            with tempfile.TemporaryDirectory() as temp_dir:
                self.assertIsNone(get_git_hooks_dir(temp_dir))
    
    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    @mock.patch('os.makedirs')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('os.chmod')
    def test_install_git_hooks(self, mock_chmod, mock_open, mock_makedirs):
        """Test installing git hooks."""
        with TestMockGitRepository() as repo_path:
            if repo_path is None:
                return  # Test was skipped
                
            # Install git hooks
            results = install_git_hooks(repo_path, ["pre-commit"])
            
            # Should have installed the pre-commit hook
            self.assertTrue(results["pre-commit"])
            
            # Verify mkdir was called
            mock_makedirs.assert_called()
            
            # Verify file was created
            mock_open.assert_called()
            
            # Verify chmod was called on Unix or not called on Windows
            if is_windows():
                mock_chmod.assert_not_called()
            else:
                mock_chmod.assert_called()
    
    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    @mock.patch('os.path.exists')
    @mock.patch('builtins.open', new_callable=mock.mock_open(read_data="Smart Git Commit Hook"))
    @mock.patch('os.remove')
    def test_remove_git_hooks(self, mock_remove, mock_open, mock_exists):
        """Test removing git hooks."""
        with TestMockGitRepository() as repo_path:
            if repo_path is None:
                return  # Test was skipped
                
            # Mock hook existence
            mock_exists.return_value = True
            
            # Remove git hooks
            results = remove_git_hooks(repo_path, ["pre-commit"])
            
            # Should have removed the pre-commit hook
            self.assertTrue(results["pre-commit"])
            
            # Verify file was opened and removed
            mock_open.assert_called()
            mock_remove.assert_called()
    
    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock.mock_open(read_data="Smart Git Commit Hook"))
    @mock.patch('os.access')
    def test_is_hook_installed(self, mock_access, mock_open, mock_exists):
        """Test checking if a hook is installed."""
        with TestMockGitRepository() as repo_path:
            if repo_path is None:
                return  # Test was skipped
                
            # Mock hook existence
            mock_exists.return_value = True
            mock_access.return_value = True
            
            # Check all supported hooks
            for hook_type in SUPPORTED_HOOKS:
                result = is_hook_installed(hook_type, repo_path)
                self.assertIsInstance(result, bool)


class TestHuskyFunctionality(TestCase):
    """Tests for the Husky integration."""
    
    @mock.patch('subprocess.run')
    def test_check_husky_compatibility(self, mock_run):
        """Test the check_husky_compatibility function."""
        # Case 1: npm is available and in a git repo
        mock_run.return_value.returncode = 0
        
        with mock.patch('smart_git_commit.hooks.get_git_root', return_value="/fake/git/repo"):
            is_compatible, reason = check_husky_compatibility()
            self.assertTrue(is_compatible)
        
        # Case 2: npm is not available
        mock_run.side_effect = [FileNotFoundError("No npm")]
        
        is_compatible, reason = check_husky_compatibility()
        self.assertFalse(is_compatible)
        self.assertIn("npm is not available", reason)
        
        # Case 3: not in a git repository
        mock_run.side_effect = None
        mock_run.return_value.returncode = 0
        
        with mock.patch('smart_git_commit.hooks.get_git_root', return_value=None):
            is_compatible, reason = check_husky_compatibility()
            self.assertFalse(is_compatible)
            self.assertIn("Not in a Git repository", reason)
    
    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    @mock.patch('os.path.exists')
    @mock.patch('json.load')
    @mock.patch('json.dump')
    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('subprocess.run')
    def test_install_husky(self, mock_run, mock_open, mock_dump, mock_load, mock_exists):
        """Test installing Husky."""
        with TestMockGitRepository() as repo_path:
            if repo_path is None:
                return  # Test was skipped
                
            # Mock package.json existence
            mock_exists.return_value = True
            
            # Mock reading package.json
            mock_load.return_value = {"name": "test-project"}
            
            # Mock npm install success
            mock_run.return_value.returncode = 0
            
            # Install husky
            result = install_husky(repo_path)
            
            # Should have installed successfully
            self.assertTrue(result)
            
            # Verify npm install was called
            npm_call_found = False
            for call in mock_run.call_args_list:
                args, kwargs = call
                if args and "npm" in args[0] and "install" in args[0]:
                    npm_call_found = True
                    break
            
            self.assertTrue(npm_call_found)
    
    @unittest.skipIf(SKIP_GIT_TESTS, "Skipping git-dependent test")
    @mock.patch('os.path.exists')
    @mock.patch('subprocess.run')
    def test_run_hook(self, mock_run, mock_exists):
        """Test running a hook."""
        # Mock hook running
        mock_run.return_value.returncode = 0
        
        # Run hook
        result = run_hook("pre-commit")
        
        # Should have run successfully
        self.assertEqual(result, 0)
        
        # Verify command was run with the right arguments
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertIn("smart-git-commit", args[0])


class TestGitHooks(unittest.TestCase):
    """Test cases for Git hooks functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=self.repo_path, capture_output=True)
        
        # Create hooks directory
        self.hooks_dir = self.repo_path / ".git" / "hooks"
        self.hooks_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_git_hooks_dir_success(self):
        """Test getting hooks directory successfully."""
        hooks_dir = get_git_hooks_dir(str(self.repo_path))
        self.assertEqual(hooks_dir, str(self.hooks_dir))
    
    def test_get_git_hooks_dir_not_git_repo(self):
        """Test getting hooks directory in non-git repository."""
        with tempfile.TemporaryDirectory() as non_git_dir:
            hooks_dir = get_git_hooks_dir(non_git_dir)
            self.assertIsNone(hooks_dir)
    
    def test_get_hook_path(self):
        """Test getting hook path."""
        hook_path = get_hook_path(str(self.hooks_dir), "pre-commit")
        expected_path = str(self.hooks_dir / "pre-commit")
        self.assertEqual(hook_path, expected_path)
    
    def test_validate_hook_type_valid(self):
        """Test validating valid hook type."""
        self.assertTrue(validate_hook_type("pre-commit"))
        self.assertTrue(validate_hook_type("commit-msg"))
    
    def test_validate_hook_type_invalid(self):
        """Test validating invalid hook type."""
        with self.assertRaises(ValueError):
            validate_hook_type("invalid-hook")
    
    def test_get_hook_types(self):
        """Test getting available hook types."""
        hook_types = get_hook_types()
        self.assertIsInstance(hook_types, list)
        self.assertIn("pre-commit", hook_types)
        self.assertIn("commit-msg", hook_types)
    
    def test_get_hook_script(self):
        """Test getting hook script content."""
        script = get_hook_script("pre-commit")
        self.assertIsInstance(script, str)
        self.assertIn("#!/usr/bin/env python3", script)
        self.assertIn("import sys", script)
    
    def test_install_hook_new(self):
        """Test installing a new hook."""
        hook_path = self.hooks_dir / "pre-commit"
        
        # Install the hook
        success = install_hook(str(self.hooks_dir), "pre-commit")
        self.assertTrue(success)
        
        # Verify hook was installed
        self.assertTrue(hook_path.exists())
        self.assertTrue(os.access(str(hook_path), os.X_OK))
        
        # Verify content
        content = hook_path.read_text()
        self.assertIn("#!/usr/bin/env python3", content)
    
    def test_install_hook_existing(self):
        """Test installing hook when one already exists."""
        hook_path = self.hooks_dir / "pre-commit"
        
        # Create existing hook
        hook_path.write_text("#!/bin/sh\necho test")
        hook_path.chmod(0o755)
        
        # Try to install
        success = install_hook(str(self.hooks_dir), "pre-commit")
        self.assertFalse(success)  # Should not overwrite existing hook
        
        # Verify original content remains
        content = hook_path.read_text()
        self.assertIn("echo test", content)
    
    def test_install_hook_error(self):
        """Test installing hook with error."""
        with patch('builtins.open', side_effect=IOError("test error")):
            success = install_hook(str(self.hooks_dir), "pre-commit")
            self.assertFalse(success)
    
    def test_remove_hook_success(self):
        """Test removing an installed hook."""
        hook_path = self.hooks_dir / "pre-commit"
        
        # Install hook first
        hook_path.write_text("#!/bin/sh\necho test")
        hook_path.chmod(0o755)
        
        # Remove the hook
        success = remove_hook(str(self.hooks_dir), "pre-commit")
        self.assertTrue(success)
        self.assertFalse(hook_path.exists())
    
    def test_remove_hook_not_exists(self):
        """Test removing non-existent hook."""
        success = remove_hook(str(self.hooks_dir), "pre-commit")
        self.assertTrue(success)  # Should return True as hook is effectively "removed"
    
    def test_remove_hook_error(self):
        """Test removing hook with error."""
        hook_path = self.hooks_dir / "pre-commit"
        hook_path.write_text("test")
        
        with patch('os.remove', side_effect=OSError("test error")):
            success = remove_hook(str(self.hooks_dir), "pre-commit")
            self.assertFalse(success)
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="# Smart Git Commit Hook")
    def test_is_hook_installed_true(self, mock_file_open, mock_exists):
        """Test checking if hook is installed when it is."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Check if hook is installed
        installed = is_hook_installed("pre-commit", str(self.hooks_dir))
        
        # Verify result
        self.assertTrue(installed)
        mock_exists.assert_called()
        mock_file_open.assert_called()
    
    def test_is_hook_installed_false(self):
        """Test checking if hook is installed when it isn't."""
        self.assertFalse(is_hook_installed("pre-commit", str(self.hooks_dir)))
    
    def test_is_hook_installed_not_executable(self):
        """Test checking if hook is installed but not executable."""
        hook_path = self.hooks_dir / "pre-commit"
        hook_path.write_text("#!/bin/sh\necho test")
        hook_path.chmod(0o644)  # Not executable
        
        self.assertFalse(is_hook_installed("pre-commit", str(self.hooks_dir)))


class TestGitHookClass(unittest.TestCase):
    """Test cases for GitHook class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        
        # Create hooks directory
        self.hooks_dir = self.repo_path / ".git" / "hooks"
        self.hooks_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_hook_init(self):
        """Test GitHook initialization."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        self.assertEqual(hook.repo_path, str(self.repo_path))
        self.assertEqual(hook.hook_type, "pre-commit")
        self.assertEqual(hook.hooks_dir, str(self.hooks_dir))
    
    def test_hook_init_invalid_type(self):
        """Test GitHook initialization with invalid type."""
        with self.assertRaises(ValueError):
            GitHook(str(self.repo_path), "invalid-hook")
    
    def test_hook_init_not_git_repo(self):
        """Test GitHook initialization in non-git repository."""
        with tempfile.TemporaryDirectory() as non_git_dir:
            with self.assertRaises(ValueError):
                GitHook(non_git_dir, "pre-commit")
    
    def test_hook_install(self):
        """Test installing hook through GitHook class."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        success = hook.install()
        self.assertTrue(success)
        
        hook_path = self.hooks_dir / "pre-commit"
        self.assertTrue(hook_path.exists())
        self.assertTrue(os.access(str(hook_path), os.X_OK))
    
    def test_hook_remove(self):
        """Test removing hook through GitHook class."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        # Install first
        hook_path = self.hooks_dir / "pre-commit"
        hook_path.write_text("test")
        hook_path.chmod(0o755)
        
        success = hook.remove()
        self.assertTrue(success)
        self.assertFalse(hook_path.exists())
    
    def test_hook_is_installed(self):
        """Test checking if hook is installed through GitHook class."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        # Initially not installed
        self.assertFalse(hook.is_installed())
        
        # Install hook
        hook_path = self.hooks_dir / "pre-commit"
        hook_path.write_text("test")
        hook_path.chmod(0o755)
        
        # Now should be installed
        self.assertTrue(hook.is_installed())
    
    def test_hook_install_with_custom_script(self):
        """Test installing hook with custom script content."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        custom_script = "#!/bin/sh\necho 'Custom hook script'"
        
        success = hook.install(script=custom_script)
        self.assertTrue(success)
        
        # Verify custom script was installed
        hook_path = self.hooks_dir / "pre-commit"
        self.assertTrue(hook_path.exists())
        content = hook_path.read_text()
        self.assertEqual(content, custom_script)
    
    def test_hook_install_with_backup(self):
        """Test installing hook with backup of existing hook."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        # Create existing hook
        hook_path = self.hooks_dir / "pre-commit"
        original_content = "#!/bin/sh\necho 'Original hook'"
        hook_path.write_text(original_content)
        hook_path.chmod(0o755)
        
        # Install with backup
        success = hook.install(backup=True)
        self.assertTrue(success)
        
        # Verify backup was created
        backup_path = hook_path.with_suffix('.bak')
        self.assertTrue(backup_path.exists())
        self.assertEqual(backup_path.read_text(), original_content)
    
    def test_hook_install_with_force(self):
        """Test force installing hook over existing one."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        # Create existing hook
        hook_path = self.hooks_dir / "pre-commit"
        hook_path.write_text("#!/bin/sh\necho 'Original hook'")
        hook_path.chmod(0o755)
        
        # Force install
        success = hook.install(force=True)
        self.assertTrue(success)
        
        # Verify new hook was installed
        content = hook_path.read_text()
        self.assertIn("smart-git-commit", content)
    
    def test_hook_install_with_template(self):
        """Test installing hook with template variables."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        template = """#!/bin/sh
REPO_PATH="{repo_path}"
HOOK_TYPE="{hook_type}"
echo "Hook for $REPO_PATH of type $HOOK_TYPE"
"""
        
        success = hook.install(template=template)
        self.assertTrue(success)
        
        # Verify template was properly formatted
        hook_path = self.hooks_dir / "pre-commit"
        content = hook_path.read_text()
        self.assertIn(str(self.repo_path), content)
        self.assertIn("pre-commit", content)
    
    def test_hook_install_with_permissions(self):
        """Test installing hook with specific permissions."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        # Install with specific permissions
        success = hook.install(permissions=0o700)
        self.assertTrue(success)
        
        # Verify permissions
        hook_path = self.hooks_dir / "pre-commit"
        if os.name != 'nt':  # Skip on Windows
            self.assertEqual(oct(hook_path.stat().st_mode & 0o777), oct(0o700))
    
    def test_hook_remove_with_backup_restore(self):
        """Test removing hook and restoring from backup."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        # Create original hook and backup
        hook_path = self.hooks_dir / "pre-commit"
        backup_path = hook_path.with_suffix('.bak')
        
        original_content = "#!/bin/sh\necho 'Original hook'"
        backup_path.write_text(original_content)
        
        # Install new hook
        hook.install()
        
        # Remove hook with backup restore
        success = hook.remove(restore_backup=True)
        self.assertTrue(success)
        
        # Verify original content was restored
        self.assertTrue(hook_path.exists())
        self.assertEqual(hook_path.read_text(), original_content)
        self.assertFalse(backup_path.exists())  # Backup should be removed
    
    def test_hook_remove_with_cleanup(self):
        """Test removing hook with cleanup of related files."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        # Create hook and related files
        hook_path = self.hooks_dir / "pre-commit"
        hook_path.write_text("test")
        
        backup_path = hook_path.with_suffix('.bak')
        backup_path.write_text("backup")
        
        temp_path = hook_path.with_suffix('.tmp')
        temp_path.write_text("temp")
        
        # Remove with cleanup
        success = hook.remove(cleanup=True)
        self.assertTrue(success)
        
        # Verify all related files are removed
        self.assertFalse(hook_path.exists())
        self.assertFalse(backup_path.exists())
        self.assertFalse(temp_path.exists())
    
    def test_hook_is_installed_with_content_check(self):
        """Test checking if hook is installed with content verification."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        # Create hook with specific content
        hook_path = self.hooks_dir / "pre-commit"
        hook_path.write_text("#!/bin/sh\necho 'test'")
        hook_path.chmod(0o755)
        
        # Check with content verification
        self.assertTrue(hook.is_installed())
        self.assertTrue(hook.is_installed(verify_content=True))
        
        # Modify content to invalid
        hook_path.write_text("invalid content")
        
        # Should still be considered installed without content verification
        self.assertTrue(hook.is_installed())
        # But not with content verification
        self.assertFalse(hook.is_installed(verify_content=True))
    
    def test_hook_validate_script(self):
        """Test hook script validation."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        # Valid script
        valid_script = """#!/bin/sh
echo "Valid hook script"
exit 0
"""
        self.assertTrue(hook.validate_script(valid_script))
        
        # Invalid script (no shebang)
        invalid_script = "echo 'Invalid script'"
        self.assertFalse(hook.validate_script(invalid_script))
        
        # Invalid script (syntax error)
        invalid_script = """#!/bin/sh
if then else
"""
        self.assertFalse(hook.validate_script(invalid_script))
    
    def test_hook_get_script_variables(self):
        """Test getting hook script variables."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        variables = hook.get_script_variables()
        self.assertIsInstance(variables, dict)
        self.assertEqual(variables['repo_path'], str(self.repo_path))
        self.assertEqual(variables['hook_type'], 'pre-commit')
        self.assertIn('git_dir', variables)
        self.assertIn('hooks_dir', variables)
    
    def test_hook_format_script(self):
        """Test formatting hook script template."""
        hook = GitHook(str(self.repo_path), "pre-commit")
        
        template = """#!/bin/sh
REPO_PATH="{repo_path}"
HOOK_TYPE="{hook_type}"
GIT_DIR="{git_dir}"
HOOKS_DIR="{hooks_dir}"
"""
        
        formatted = hook.format_script(template)
        self.assertIn(str(self.repo_path), formatted)
        self.assertIn('pre-commit', formatted)
        self.assertIn(str(self.hooks_dir), formatted)
        self.assertIn('.git', formatted)


class TestHookScriptGeneration(unittest.TestCase):
    """Test the generation of hook scripts."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        
        # Create hooks directory
        self.hooks_dir = self.repo_path / ".git" / "hooks"
        self.hooks_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_hook_script_simple(self):
        """Test getting a simple hook script."""
        script = get_hook_script("pre-commit")
        self.assertIsInstance(script, str)
        self.assertIn("#!/usr/bin/env python3", script)
        self.assertIn("import sys", script)
        self.assertIn("import os", script)
        self.assertIn("pre-commit", script.lower())
    
    def test_get_hook_script_with_template(self):
        """Test getting a hook script with a template."""
        template = """#!/bin/sh
# Custom hook template
echo "Running {hook_type} hook"
exit 0
"""
        script = get_hook_script("pre-commit", template=template)
        self.assertIn("Running pre-commit hook", script)
        self.assertNotIn("{hook_type}", script)
    
    def test_get_hook_script_with_variables(self):
        """Test getting a hook script with custom variables."""
        template = """#!/bin/sh
REPO="{repo_name}"
BRANCH="{branch}"
CUSTOM="{custom_var}"
"""
        variables = {
            "repo_name": "test-repo",
            "branch": "main",
            "custom_var": "custom-value"
        }
        script = get_hook_script("post-commit", template=template, variables=variables)
        self.assertIn('REPO="test-repo"', script)
        self.assertIn('BRANCH="main"', script)
        self.assertIn('CUSTOM="custom-value"', script)
    
    def test_get_hook_script_missing_variables(self):
        """Test getting a hook script with missing variables."""
        template = """#!/bin/sh
REPO="{repo_name}"
MISSING="{not_provided}"
"""
        variables = {"repo_name": "test-repo"}
        with self.assertRaises(KeyError):
            get_hook_script("pre-push", template=template, variables=variables)
    
    def test_get_hook_script_validate(self):
        """Test validating a hook script."""
        script = get_hook_script("pre-commit")
        self.assertTrue(validate_hook_script(script))
        
        invalid_script = "This is not a valid script without shebang"
        self.assertFalse(validate_hook_script(invalid_script))
    
    def test_format_hook_script(self):
        """Test formatting a hook script template."""
        template = """#!/bin/sh
HOOK_TYPE="{hook_type}"
REPO_PATH="{repo_path}"
"""
        variables = {
            "hook_type": "pre-commit",
            "repo_path": "/path/to/repo"
        }
        
        formatted = format_hook_script(template, variables)
        self.assertEqual(formatted, """#!/bin/sh
HOOK_TYPE="pre-commit"
REPO_PATH="/path/to/repo"
""")


class TestHookInstallation(unittest.TestCase):
    """Tests for hook installation process."""

    def setUp(self):
        """Set up test fixtures."""
        self.repo = MockGitRepository(Path(tempfile.mkdtemp()))
        # Ensure the hooks dir exists for the mock repo
        os.makedirs(self.repo.hooks_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        self.repo.cleanup()

    def test_install_hooks_success(self):
        """Test successful installation of hooks."""
        hooks_to_install = ["pre-commit", "commit-msg"]
        results = install_git_hooks(str(self.repo.path), hooks=hooks_to_install)
        
        self.assertTrue(all(results.values()))
        for hook_type in hooks_to_install:
            # Use the correct hooks_dir from the mock repo object
            hook_path = Path(get_hook_path(str(self.repo.hooks_dir), hook_type))
            self.assertTrue(hook_path.exists())
            # Add check for executable bit on non-windows
            if not is_windows():
                 self.assertTrue(os.access(hook_path, os.X_OK))

    def test_install_hooks_with_custom_script(self):
        """Test installation with custom script."""
        custom_script = "#!/bin/sh\necho \"Custom hook script\""
        results = install_git_hooks(
            str(self.repo.path), 
            hooks=["pre-commit"], 
            script=custom_script
        )
        
        self.assertTrue(results["pre-commit"])
        # Use the correct hooks_dir from the mock repo object
        hook_path = Path(get_hook_path(str(self.repo.hooks_dir), "pre-commit"))
        self.assertTrue(hook_path.exists())
        content = hook_path.read_text()
        self.assertEqual(content, custom_script)

    def test_install_hooks_with_force(self):
        """Test force installation over existing hooks."""
        # Install initial hook
        hook_path = Path(get_hook_path(str(self.repo.hooks_dir), "pre-commit"))
        hook_path.write_text("initial content")
        
        results = install_git_hooks(str(self.repo.path), hooks=["pre-commit"], force=True)
        
        self.assertTrue(results["pre-commit"])
        # Check content was overwritten
        content = hook_path.read_text()
        self.assertIn("smart-git-commit", content)

    def test_install_hooks_without_force(self):
        """Test installation without force option when hook exists."""
        # Install initial hook manually
        hook_path = Path(get_hook_path(str(self.repo.hooks_dir), "pre-commit"))
        initial_content = "initial content"
        # Ensure hooks dir exists
        os.makedirs(hook_path.parent, exist_ok=True)
        hook_path.write_text(initial_content)
        
        results = install_git_hooks(str(self.repo.path), hooks=["pre-commit"], force=False)
        
        self.assertFalse(results["pre-commit"])
        # Check content was NOT overwritten
        content = hook_path.read_text()
        self.assertEqual(content, initial_content)

    def test_install_hooks_with_backup(self):
        """Test installation with backup of existing hook."""
        # Install initial hook manually
        hook_path = Path(get_hook_path(str(self.repo.hooks_dir), "pre-commit"))
        initial_content = "initial content"
        # Ensure hooks dir exists
        os.makedirs(hook_path.parent, exist_ok=True)
        hook_path.write_text(initial_content)
        
        results = install_git_hooks(str(self.repo.path), hooks=["pre-commit"], backup=True)
        
        self.assertTrue(results["pre-commit"])
        # Check backup exists
        backup_path = Path(f"{hook_path}.bak")
        self.assertTrue(backup_path.exists())
        self.assertEqual(backup_path.read_text(), initial_content)
        # Check new hook content
        content = hook_path.read_text()
        self.assertIn("smart-git-commit", content)

    def test_remove_hooks_success(self):
        """Test successful removal of hooks."""
        # Install hook first
        install_git_hooks(str(self.repo.path), hooks=["pre-commit"])
        hook_path = Path(get_hook_path(str(self.repo.hooks_dir), "pre-commit"))
        self.assertTrue(hook_path.exists())
        
        results = remove_git_hooks(str(self.repo.path), hooks=["pre-commit"])
        
        self.assertTrue(results["pre-commit"])
        self.assertFalse(hook_path.exists())

    def test_remove_hooks_with_backup_restore(self):
        """Test removing hooks with backup restoration."""
        # Install initial hook with backup
        hook_path = Path(get_hook_path(str(self.repo.hooks_dir), "pre-commit"))
        original_content = "#!/bin/sh\necho 'Original hook'" # This is the content to be backed up
        os.makedirs(hook_path.parent, exist_ok=True) # Ensure dir exists
        hook_path.write_text(original_content)
        install_git_hooks(str(self.repo.path), hooks=["pre-commit"], backup=True)
        
        # Remove hook and restore backup
        results = remove_git_hooks(str(self.repo.path), hooks=["pre-commit"], restore_backup=True)
        
        self.assertTrue(results["pre-commit"])
        self.assertTrue(hook_path.exists())
        content = hook_path.read_text()
        # Assert that the restored content matches the original content
        self.assertEqual(content, original_content)
        # Check backup file is gone
        backup_path = Path(f"{hook_path}.bak")
        self.assertFalse(backup_path.exists())

    def test_remove_hooks_not_present(self):
        """Test removing hooks that are not present."""
        results = remove_git_hooks(str(self.repo.path), hooks=["pre-commit"])
        self.assertTrue(results["pre-commit"]) # Should succeed even if not present


if __name__ == '__main__':
    unittest.main() 