#!/usr/bin/env python3
"""
Tests for the HuskyIntegration class in hook.py module.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest import mock
import platform
import subprocess
from pathlib import Path

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_git_commit.hook import HuskyIntegration


class TestHuskyIntegration(unittest.TestCase):
    """Test cases for HuskyIntegration class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Mock logger
        self.logger_patcher = mock.patch('smart_git_commit.hook.logger')
        self.mock_logger = self.logger_patcher.start()
        
        # Mock subprocess
        self.subprocess_patcher = mock.patch('smart_git_commit.hook.subprocess.run')
        self.mock_subprocess = self.subprocess_patcher.start()
        
        # Configure mock subprocess to handle git commands
        def mock_run(cmd, **kwargs):
            mock_process = mock.MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = ""
            
            # Handle git rev-parse --show-toplevel
            if cmd[0] == "git" and cmd[1] == "rev-parse" and "--show-toplevel" in cmd:
                mock_process.stdout = self.temp_dir
            # Handle node --version
            elif cmd[0] == "node" and cmd[1] == "--version":
                mock_process.stdout = "v14.17.0"
            # Handle npm --version
            elif cmd[0] == "npm" and cmd[1] == "--version":
                mock_process.stdout = "6.14.13"
            # Handle npm install husky
            elif cmd[0] == "npm" and cmd[1] == "install" and "husky" in cmd:
                mock_process.stdout = "Successfully installed husky"
            # Handle npx husky install
            elif cmd[0] == "npx" and cmd[1] == "husky" and cmd[2] == "install":
                mock_process.stdout = "husky - Git hooks installed"
            
            return mock_process
        
        self.mock_subprocess.side_effect = mock_run
        
        # Create git repository structure
        os.makedirs(os.path.join(self.temp_dir, ".git"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, ".git", "hooks"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, ".git", "refs", "heads"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, ".git", "objects"), exist_ok=True)
        
        # Create basic git config
        with open(os.path.join(self.temp_dir, ".git", "config"), "w") as f:
            f.write("[core]\n\trepositoryformatversion = 0\n\tfilemode = false\n\tbare = false\n")
        
        # Create HEAD file
        with open(os.path.join(self.temp_dir, ".git", "HEAD"), "w") as f:
            f.write("ref: refs/heads/main\n")

    def tearDown(self):
        """Clean up after tests."""
        # Restore original directory
        os.chdir(self.original_dir)
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)
        
        # Stop patchers
        self.logger_patcher.stop()
        self.subprocess_patcher.stop()

    def test_init_success(self):
        """Test successful initialization."""
        husky = HuskyIntegration(self.temp_dir)
        self.assertEqual(husky.repo_path, os.path.abspath(self.temp_dir))
        self.assertEqual(husky.git_root, self.temp_dir)
        self.assertEqual(husky.is_windows, platform.system() == "Windows")
        self.assertEqual(husky.hook_extension, ".ps1" if husky.is_windows else ".sh")

    def test_init_no_git_repo(self):
        """Test initialization with no git repository."""
        # Mock subprocess to fail
        def mock_run_fail(cmd, **kwargs):
            mock_process = mock.MagicMock()
            mock_process.returncode = 1
            mock_process.stderr = "fatal: not a git repository"
            mock_process.stdout = ""
            return mock_process
        
        self.mock_subprocess.side_effect = mock_run_fail
        
        with self.assertRaises(ValueError):
            HuskyIntegration(self.temp_dir)

    def test_get_git_root_success(self):
        """Test successful git root detection."""
        husky = HuskyIntegration(self.temp_dir)
        result = husky._get_git_root()
        self.assertEqual(result, self.temp_dir)

    def test_get_git_root_failure(self):
        """Test git root detection failure."""
        # Mock subprocess to fail
        def mock_run_fail(cmd, **kwargs):
            mock_process = mock.MagicMock()
            mock_process.returncode = 1
            mock_process.stderr = "fatal: not a git repository"
            mock_process.stdout = ""
            return mock_process
        
        self.mock_subprocess.side_effect = mock_run_fail
        
        # Create a new instance with a non-git directory
        non_git_dir = os.path.join(self.temp_dir, "non_git")
        os.makedirs(non_git_dir, exist_ok=True)
        
        with self.assertRaises(ValueError):
            HuskyIntegration(non_git_dir)

    def test_get_git_root_exception(self):
        """Test git root detection with exception."""
        # Mock subprocess to raise exception
        def mock_run_exception(cmd, **kwargs):
            if cmd[0] == "git" and cmd[1] == "rev-parse" and "--show-toplevel" in cmd:
                raise Exception("Test error")
            return mock.MagicMock()
        
        self.mock_subprocess.side_effect = mock_run_exception
        
        with self.assertRaises(ValueError):
            HuskyIntegration(self.temp_dir)

    def test_run_git_command_success(self):
        """Test successful git command execution."""
        husky = HuskyIntegration(self.temp_dir)
        result = husky._run_git_command(["status"])
        self.assertEqual(result, "")  # Mock returns empty string for unknown commands

    def test_run_git_command_no_output(self):
        """Test git command execution without output capture."""
        husky = HuskyIntegration(self.temp_dir)
        result = husky._run_git_command(["status"], check_output=False)
        self.assertIsNone(result)

    def test_run_git_command_failure(self):
        """Test git command execution failure."""
        # Mock subprocess to fail
        def mock_run_fail(cmd, **kwargs):
            if cmd[0] == "git" and cmd[1] == "rev-parse" and "--show-toplevel" in cmd:
                mock_process = mock.MagicMock()
                mock_process.returncode = 0
                mock_process.stdout = self.temp_dir
                return mock_process
            raise subprocess.CalledProcessError(1, "git", stderr="Test error")
        
        self.mock_subprocess.side_effect = mock_run_fail
        
        husky = HuskyIntegration(self.temp_dir)
        result = husky._run_git_command(["status"])
        self.assertIsNone(result)
        self.mock_logger.error.assert_called()

    def test_check_node_installed_true(self):
        """Test node.js installation check when installed."""
        def mock_run_node_installed(cmd, **kwargs):
            mock_process = mock.MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = ""
            
            # Handle git rev-parse --show-toplevel
            if cmd[0] == "git" and cmd[1] == "rev-parse" and "--show-toplevel" in cmd:
                mock_process.stdout = self.temp_dir
            # Handle node --version
            elif cmd[0] == "node" and cmd[1] == "--version":
                mock_process.stdout = "v14.17.0"
            
            return mock_process
        
        self.mock_subprocess.side_effect = mock_run_node_installed
        
        husky = HuskyIntegration(self.temp_dir)
        self.assertTrue(husky._check_node_installed())

    def test_check_node_installed_false(self):
        """Test node.js installation check when not installed."""
        def mock_run_node_not_installed(cmd, **kwargs):
            mock_process = mock.MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = ""
            
            # Handle git rev-parse --show-toplevel
            if cmd[0] == "git" and cmd[1] == "rev-parse" and "--show-toplevel" in cmd:
                mock_process.stdout = self.temp_dir
            # Handle node --version
            elif cmd[0] == "node" and cmd[1] == "--version":
                raise FileNotFoundError("node not found")
            
            return mock_process
        
        self.mock_subprocess.side_effect = mock_run_node_not_installed
        
        husky = HuskyIntegration(self.temp_dir)
        self.assertFalse(husky._check_node_installed())

    def test_check_npm_installed_true(self):
        """Test npm installation check when installed."""
        def mock_run_npm_installed(cmd, **kwargs):
            mock_process = mock.MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = ""
            
            # Handle git rev-parse --show-toplevel
            if cmd[0] == "git" and cmd[1] == "rev-parse" and "--show-toplevel" in cmd:
                mock_process.stdout = self.temp_dir
            # Handle npm --version
            elif cmd[0] == "npm" and cmd[1] == "--version":
                mock_process.stdout = "6.14.13"
            
            return mock_process
        
        self.mock_subprocess.side_effect = mock_run_npm_installed
        
        husky = HuskyIntegration(self.temp_dir)
        self.assertTrue(husky._check_npm_installed())

    def test_check_npm_installed_false(self):
        """Test npm installation check when not installed."""
        def mock_run_npm_not_installed(cmd, **kwargs):
            mock_process = mock.MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = ""
            
            # Handle git rev-parse --show-toplevel
            if cmd[0] == "git" and cmd[1] == "rev-parse" and "--show-toplevel" in cmd:
                mock_process.stdout = self.temp_dir
            # Handle npm --version
            elif cmd[0] == "npm" and cmd[1] == "--version":
                raise FileNotFoundError("npm not found")
            
            return mock_process
        
        self.mock_subprocess.side_effect = mock_run_npm_not_installed
        
        husky = HuskyIntegration(self.temp_dir)
        self.assertFalse(husky._check_npm_installed())

    def test_update_package_json_new(self):
        """Test creating new package.json."""
        husky = HuskyIntegration(self.temp_dir)
        result = husky._update_package_json()
        
        self.assertTrue(result)
        package_path = os.path.join(self.temp_dir, "package.json")
        self.assertTrue(os.path.exists(package_path))
        
        with open(package_path) as f:
            data = json.load(f)
            self.assertEqual(data["scripts"]["prepare"], "husky install")

    def test_update_package_json_existing(self):
        """Test updating existing package.json."""
        # Create existing package.json
        package_path = os.path.join(self.temp_dir, "package.json")
        with open(package_path, 'w') as f:
            json.dump({
                "name": "test",
                "version": "1.0.0",
                "scripts": {}
            }, f)
        
        husky = HuskyIntegration(self.temp_dir)
        result = husky._update_package_json()
        
        self.assertTrue(result)
        with open(package_path) as f:
            data = json.load(f)
            self.assertEqual(data["scripts"]["prepare"], "husky install")
            self.assertEqual(data["name"], "test")

    def test_update_package_json_invalid(self):
        """Test updating invalid package.json."""
        # Create invalid package.json
        package_path = os.path.join(self.temp_dir, "package.json")
        with open(package_path, 'w') as f:
            f.write("invalid json")
        
        husky = HuskyIntegration(self.temp_dir)
        result = husky._update_package_json()
        
        self.assertFalse(result)
        self.mock_logger.error.assert_called_with("Invalid package.json file")

    def test_create_husky_dir(self):
        """Test creating husky directory."""
        husky = HuskyIntegration(self.temp_dir)
        result = husky._create_husky_dir()
        
        self.assertTrue(result)
        husky_dir = os.path.join(self.temp_dir, ".husky")
        self.assertTrue(os.path.exists(husky_dir))
        self.assertTrue(os.path.exists(os.path.join(husky_dir, ".gitignore")))

    def test_create_husky_dir_error(self):
        """Test creating husky directory with error."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Mock os.makedirs to raise error
        with mock.patch('os.makedirs', side_effect=Exception("Test error")):
            result = husky._create_husky_dir()
            
            self.assertFalse(result)
            self.mock_logger.error.assert_called()

    def test_install_husky_npm_success(self):
        """Test successful husky installation via npm."""
        husky = HuskyIntegration(self.temp_dir)
        result = husky.install_husky_npm()
        
        self.assertTrue(result)
        self.mock_logger.info.assert_called_with("Successfully installed Husky via npm")

    def test_install_husky_npm_no_npm(self):
        """Test husky installation when npm is not installed."""
        # Mock npm check to return False
        with mock.patch.object(HuskyIntegration, '_check_npm_installed', return_value=False):
            husky = HuskyIntegration(self.temp_dir)
            result = husky.install_husky_npm()
            
            self.assertFalse(result)
            self.mock_logger.error.assert_called_with("npm is not installed. Cannot install Husky via npm.")

    def test_install_husky_npm_install_error(self):
        """Test husky installation with npm install error."""
        # Mock subprocess to fail on npm install
        self.mock_subprocess.side_effect = [
            mock.DEFAULT,  # For git root check
            subprocess.CalledProcessError(1, "npm", stderr="Test error")  # For npm install
        ]
        
        husky = HuskyIntegration(self.temp_dir)
        result = husky.install_husky_npm()
        
        self.assertFalse(result)
        self.mock_logger.error.assert_called()

    def test_install_husky_manual_success(self):
        """Test successful manual husky installation."""
        husky = HuskyIntegration(self.temp_dir)
        result = husky.install_husky_manual()
        
        self.assertTrue(result)
        husky_dir = os.path.join(self.temp_dir, ".husky")
        self.assertTrue(os.path.exists(husky_dir))
        self.assertTrue(os.path.exists(os.path.join(husky_dir, "_", "husky.sh")))

    def test_install_husky_manual_error(self):
        """Test manual husky installation with error."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Mock _create_husky_dir to fail
        with mock.patch.object(HuskyIntegration, '_create_husky_dir', return_value=False):
            result = husky.install_husky_manual()
            self.assertFalse(result)

    def test_create_hook_unix(self):
        """Test creating a hook file on Unix."""
        # Mock platform to be Unix
        with mock.patch('platform.system', return_value='Linux'):
            husky = HuskyIntegration(self.temp_dir)
            
            # Create .husky directory
            husky_dir = os.path.join(self.temp_dir, ".husky")
            os.makedirs(husky_dir, exist_ok=True)
            
            # Mock file operations and os.chmod
            mock_file = mock.mock_open()
            with mock.patch('builtins.open', mock_file), \
                 mock.patch('os.chmod') as mock_chmod:
                result = husky.create_hook("pre-commit", "echo test")
                
                self.assertTrue(result)
                mock_file.assert_called_with(
                    os.path.join(self.temp_dir, ".husky", "pre-commit"),
                    'w',
                    encoding='utf-8',
                    newline='\n'
                )
                mock_chmod.assert_called_once()

    def test_create_hook_windows(self):
        """Test creating a hook file on Windows."""
        # Mock platform to be Windows
        with mock.patch('platform.system', return_value='Windows'):
            husky = HuskyIntegration(self.temp_dir)
            
            # Create .husky directory
            husky_dir = os.path.join(self.temp_dir, ".husky")
            os.makedirs(husky_dir, exist_ok=True)
            
            # Mock file operations
            mock_file = mock.mock_open()
            with mock.patch('builtins.open', mock_file):
                result = husky.create_hook("pre-commit", "echo test")
                
                self.assertTrue(result)
                mock_file.assert_called_with(
                    os.path.join(self.temp_dir, ".husky", "pre-commit"),
                    'w',
                    encoding='utf-8',
                    newline='\r\n'
                )

    def test_create_hook_error(self):
        """Test creating a hook file with error."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Mock open to raise error
        with mock.patch('builtins.open', side_effect=Exception("Test error")):
            result = husky.create_hook("pre-commit", "echo test")
            
            self.assertFalse(result)
            self.mock_logger.error.assert_called()

    def test_setup_pre_commit_success(self):
        """Test successful pre-commit hook setup."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Create .husky directory
        husky_dir = os.path.join(self.temp_dir, ".husky")
        os.makedirs(husky_dir, exist_ok=True)
        
        # Mock file operations and husky installation
        mock_file = mock.mock_open()
        with mock.patch('builtins.open', mock_file), \
             mock.patch.object(husky, 'is_husky_installed', return_value=True), \
             mock.patch('os.chmod'):
            result = husky.setup_pre_commit()
            
            self.assertTrue(result)
            mock_file.assert_called_with(
                os.path.join(self.temp_dir, ".husky", "pre-commit"),
                'w',
                encoding='utf-8',
                newline='\r\n' if husky.is_windows else '\n'
            )

    def test_setup_pre_commit_npm(self):
        """Test pre-commit hook setup with npm."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Mock file operations and husky installation
        mock_file = mock.mock_open()
        with mock.patch('builtins.open', mock_file), \
             mock.patch.object(husky, 'is_husky_installed', return_value=False), \
             mock.patch.object(husky, '_check_npm_installed', return_value=True), \
             mock.patch.object(husky, 'install_husky_npm', return_value=True), \
             mock.patch('os.chmod'):
            result = husky.setup_pre_commit()
            
            self.assertTrue(result)
            mock_file.assert_called_with(
                os.path.join(self.temp_dir, ".husky", "pre-commit"),
                'w',
                encoding='utf-8',
                newline='\r\n' if husky.is_windows else '\n'
            )

    def test_setup_pre_commit_manual(self):
        """Test pre-commit hook setup without npm."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Mock husky not installed and npm not available
        with mock.patch.object(husky, 'is_husky_installed', return_value=False), \
             mock.patch.object(husky, '_check_npm_installed', return_value=False):
            result = husky.setup_pre_commit()
            
            self.assertTrue(result)
            hook_path = os.path.join(self.temp_dir, ".husky", "pre-commit")
            self.assertTrue(os.path.exists(hook_path))

    def test_setup_pre_commit_install_failure(self):
        """Test pre-commit hook setup with installation failure."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Mock husky not installed and installation failure
        with mock.patch.object(husky, 'is_husky_installed', return_value=False), \
             mock.patch.object(husky, 'install_husky_manual', return_value=False), \
             mock.patch.object(husky, '_check_npm_installed', return_value=False):
            result = husky.setup_pre_commit()
            
            self.assertFalse(result)

    def test_remove_hook_success(self):
        """Test successful hook removal."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Create a hook file first
        hook_path = os.path.join(self.temp_dir, ".husky", "pre-commit")
        os.makedirs(os.path.dirname(hook_path), exist_ok=True)
        with open(hook_path, 'w') as f:
            f.write("test")
        
        result = husky.remove_hook("pre-commit")
        
        self.assertTrue(result)
        self.assertFalse(os.path.exists(hook_path))

    def test_remove_hook_not_exists(self):
        """Test removing non-existent hook."""
        husky = HuskyIntegration(self.temp_dir)
        result = husky.remove_hook("non-existent")
        
        self.assertTrue(result)
        self.mock_logger.warning.assert_called()

    def test_remove_hook_error(self):
        """Test hook removal with error."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Create a hook file first
        hook_path = os.path.join(self.temp_dir, ".husky", "pre-commit")
        os.makedirs(os.path.dirname(hook_path), exist_ok=True)
        with open(hook_path, 'w') as f:
            f.write("test")
        
        # Mock os.remove to raise error
        with mock.patch('os.remove', side_effect=Exception("Test error")):
            result = husky.remove_hook("pre-commit")
            
            self.assertFalse(result)
            self.mock_logger.error.assert_called()

    def test_uninstall_husky_success(self):
        """Test successful husky uninstallation."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Create husky directory and package.json
        husky_dir = os.path.join(self.temp_dir, ".husky")
        os.makedirs(husky_dir, exist_ok=True)
        
        package_path = os.path.join(self.temp_dir, "package.json")
        with open(package_path, 'w') as f:
            json.dump({
                "scripts": {
                    "prepare": "husky install"
                }
            }, f)
        
        result = husky.uninstall_husky()
        
        self.assertTrue(result)
        self.assertFalse(os.path.exists(husky_dir))
        
        with open(package_path) as f:
            data = json.load(f)
            self.assertNotIn("prepare", data.get("scripts", {}))

    def test_uninstall_husky_not_installed(self):
        """Test uninstalling when husky is not installed."""
        husky = HuskyIntegration(self.temp_dir)
        result = husky.uninstall_husky()
        
        self.assertTrue(result)
        self.mock_logger.info.assert_called_with("Husky not installed")

    def test_uninstall_husky_rmtree_error(self):
        """Test husky uninstallation with rmtree error."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Create husky directory
        husky_dir = os.path.join(self.temp_dir, ".husky")
        os.makedirs(husky_dir, exist_ok=True)
        
        # Mock shutil.rmtree to raise error
        with mock.patch('shutil.rmtree', side_effect=Exception("Test error")):
            result = husky.uninstall_husky()
            
            self.assertFalse(result)
            self.mock_logger.error.assert_called()

    def test_get_hook_path(self):
        """Test getting hook path."""
        husky = HuskyIntegration(self.temp_dir)
        hook_path = husky.get_hook_path("pre-commit")
        expected_path = os.path.join(self.temp_dir, ".husky", "pre-commit")
        self.assertEqual(hook_path, expected_path)

    def test_list_hooks_empty(self):
        """Test listing hooks when none are installed."""
        husky = HuskyIntegration(self.temp_dir)
        hooks = husky.list_hooks()
        self.assertEqual(hooks, [])

    def test_list_hooks_with_hooks(self):
        """Test listing installed hooks."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Create some hook files
        husky_dir = os.path.join(self.temp_dir, ".husky")
        os.makedirs(husky_dir, exist_ok=True)
        
        hook_names = ["pre-commit", "pre-push"]
        for hook in hook_names:
            with open(os.path.join(husky_dir, hook), 'w') as f:
                f.write("test")
        
        hooks = husky.list_hooks()
        self.assertEqual(sorted(hooks), sorted(hook_names))

    def test_is_hook_installed_true(self):
        """Test checking if a hook is installed."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Create a hook file
        hook_path = os.path.join(self.temp_dir, ".husky", "pre-commit")
        os.makedirs(os.path.dirname(hook_path), exist_ok=True)
        with open(hook_path, 'w') as f:
            f.write("test")
        
        self.assertTrue(husky.is_hook_installed("pre-commit"))

    def test_is_hook_installed_false(self):
        """Test checking if a non-existent hook is installed."""
        husky = HuskyIntegration(self.temp_dir)
        self.assertFalse(husky.is_hook_installed("non-existent"))

    def test_is_husky_installed_true(self):
        """Test checking if husky is installed."""
        husky = HuskyIntegration(self.temp_dir)
        
        # Create husky files
        husky_dir = os.path.join(self.temp_dir, ".husky", "_")
        os.makedirs(husky_dir, exist_ok=True)
        with open(os.path.join(husky_dir, "husky.sh"), 'w') as f:
            f.write("test")
        
        self.assertTrue(husky.is_husky_installed())

    def test_is_husky_installed_false(self):
        """Test checking if husky is installed when it's not."""
        husky = HuskyIntegration(self.temp_dir)
        self.assertFalse(husky.is_husky_installed())


class TestMain(unittest.TestCase):
    """Test cases for main function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create git repository structure
        os.makedirs(os.path.join(self.temp_dir, ".git"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, ".git", "hooks"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, ".git", "refs", "heads"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, ".git", "objects"), exist_ok=True)
        
        # Create basic git config
        with open(os.path.join(self.temp_dir, ".git", "config"), "w") as f:
            f.write("[core]\n\trepositoryformatversion = 0\n\tfilemode = false\n\tbare = false\n")
        
        # Create HEAD file
        with open(os.path.join(self.temp_dir, ".git", "HEAD"), "w") as f:
            f.write("ref: refs/heads/main\n")
        
        # Mock logger
        self.logger_patcher = mock.patch('smart_git_commit.hook.logger')
        self.mock_logger = self.logger_patcher.start()
        
        # Mock subprocess
        self.subprocess_patcher = mock.patch('smart_git_commit.hook.subprocess.run')
        self.mock_subprocess = self.subprocess_patcher.start()
        
        # Configure mock subprocess to handle git commands
        def mock_run(cmd, **kwargs):
            mock_process = mock.MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = ""
            
            # Handle git rev-parse --show-toplevel
            if cmd[0] == "git" and cmd[1] == "rev-parse" and "--show-toplevel" in cmd:
                mock_process.stdout = self.temp_dir
            # Handle node --version
            elif cmd[0] == "node" and cmd[1] == "--version":
                mock_process.stdout = "v14.17.0"
            # Handle npm --version
            elif cmd[0] == "npm" and cmd[1] == "--version":
                mock_process.stdout = "6.14.13"
            # Handle npm install husky
            elif cmd[0] == "npm" and cmd[1] == "install" and "husky" in cmd:
                mock_process.stdout = "Successfully installed husky"
            # Handle npx husky install
            elif cmd[0] == "npx" and cmd[1] == "husky" and cmd[2] == "install":
                mock_process.stdout = "husky - Git hooks installed"
            
            return mock_process
        
        self.mock_subprocess.side_effect = mock_run
        
        # Mock file operations
        self.file_patcher = mock.patch('builtins.open', mock.mock_open())
        self.mock_file = self.file_patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
        self.logger_patcher.stop()
        self.subprocess_patcher.stop()
        self.file_patcher.stop()

    def test_main_setup(self):
        """Test main function with setup command."""
        # Mock file operations and husky installation
        mock_file = mock.mock_open()
        with mock.patch('sys.argv', ['hook.py', 'setup']), \
             mock.patch('builtins.open', mock_file), \
             mock.patch.object(HuskyIntegration, 'is_husky_installed', return_value=True), \
             mock.patch.object(HuskyIntegration, 'setup_pre_commit', return_value=True), \
             mock.patch('os.chmod'), \
             mock.patch('argparse.ArgumentParser') as mock_parser_class:
            # Set up mock parser
            mock_parser = mock.MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_subparsers = mock.MagicMock()
            mock_parser.add_subparsers.return_value = mock_subparsers
            mock_setup_parser = mock.MagicMock()
            mock_subparsers.add_parser.return_value = mock_setup_parser
            
            # Set up mock args
            mock_args = mock.MagicMock()
            mock_args.command = "setup"
            mock_args.hook_name = "pre-commit"
            mock_args.command_str = "python -m smart_git_commit"
            mock_parser.parse_args.return_value = mock_args
            
            from smart_git_commit.hook import main
            result = main()
            self.assertEqual(result, 0)

    def test_main_setup_custom_command(self):
        """Test main function with setup command and custom command."""
        # Mock file operations and husky installation
        mock_file = mock.mock_open()
        with mock.patch('sys.argv', ['hook.py', 'setup', '--command', 'custom command']), \
             mock.patch('builtins.open', mock_file), \
             mock.patch.object(HuskyIntegration, 'is_husky_installed', return_value=True), \
             mock.patch.object(HuskyIntegration, 'setup_pre_commit', return_value=True), \
             mock.patch('os.chmod'), \
             mock.patch('argparse.ArgumentParser') as mock_parser_class:
            # Set up mock parser
            mock_parser = mock.MagicMock()
            mock_parser_class.return_value = mock_parser
            mock_subparsers = mock.MagicMock()
            mock_parser.add_subparsers.return_value = mock_subparsers
            mock_setup_parser = mock.MagicMock()
            mock_subparsers.add_parser.return_value = mock_setup_parser
            
            # Set up mock args
            mock_args = mock.MagicMock()
            mock_args.command = "setup"
            mock_args.hook_name = "pre-commit"
            mock_args.command_str = "custom command"
            mock_parser.parse_args.return_value = mock_args
            
            from smart_git_commit.hook import main
            result = main()
            self.assertEqual(result, 0)

    def test_main_remove(self):
        """Test main function with remove command."""
        with mock.patch('sys.argv', ['hook.py', 'remove', 'pre-commit']), \
             mock.patch.object(HuskyIntegration, 'remove_hook', return_value=True):
            from smart_git_commit.hook import main
            result = main()
            self.assertEqual(result, 0)

    def test_main_uninstall(self):
        """Test main function with uninstall command."""
        with mock.patch('sys.argv', ['hook.py', 'uninstall']), \
             mock.patch.object(HuskyIntegration, 'uninstall_husky', return_value=True):
            from smart_git_commit.hook import main
            result = main()
            self.assertEqual(result, 0)

    def test_main_list(self):
        """Test main function with list command."""
        with mock.patch('sys.argv', ['hook.py', 'list']), \
             mock.patch.object(HuskyIntegration, 'list_hooks', return_value=['pre-commit']):
            from smart_git_commit.hook import main
            result = main()
            self.assertEqual(result, 0)

    def test_main_no_command(self):
        """Test main function with no command."""
        with mock.patch('sys.argv', ['hook.py']):
            from smart_git_commit.hook import main
            result = main()
            self.assertEqual(result, 1)

    def test_main_invalid_repo(self):
        """Test main function with invalid repository."""
        # Mock git root check to fail
        def mock_run_fail(cmd, **kwargs):
            mock_process = mock.MagicMock()
            mock_process.returncode = 1
            mock_process.stderr = "fatal: not a git repository"
            mock_process.stdout = ""
            return mock_process
        
        self.mock_subprocess.side_effect = mock_run_fail
        
        with mock.patch('sys.argv', ['hook.py', 'setup']):
            from smart_git_commit.hook import main
            result = main()
            self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main() 