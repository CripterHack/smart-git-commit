#!/usr/bin/env python3
"""
Tests for the processor.py module.
"""

import os
import sys
import unittest
from unittest import mock
from typing import Dict, Any
import json
import tempfile
import shutil
import subprocess
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_git_commit.processor import (
    AIClient, OllamaClient, OpenAIClient, CommitProcessor,
    Configuration, get_config_file_path, load_config, get_processor
)
from smart_git_commit.commit_group import CommitGroup, CommitType
from smart_git_commit.change import Change


class TestAIClient(unittest.TestCase):
    """Tests for the AIClient abstract base class."""
    
    def test_generate_not_implemented(self):
        """Test that the generate method raises NotImplementedError."""
        client = AIClient()
        with self.assertRaises(NotImplementedError):
            client.generate("Test prompt")


class TestOllamaClient(unittest.TestCase):
    """Tests for the OllamaClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.patcher_requests = mock.patch('requests.get')
        self.mock_get = self.patcher_requests.start()
        
        # Mock successful API response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3"},
                {"name": "gemma:2b"}
            ]
        }
        self.mock_get.return_value = mock_response
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher_requests.stop()
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        client = OllamaClient()
        self.assertEqual(client.model, "llama3")
        self.assertEqual(client.host, "http://localhost:11434")
    
    def test_validate_setup_success(self):
        """Test successful validation setup."""
        client = OllamaClient(model="llama3")
        self.assertEqual(client.model, "llama3")
    
    def test_validate_setup_model_fallback(self):
        """Test model fallback when specified model is not found."""
        # Mock API response with different models available
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral"},
                {"name": "gemma:2b"}
            ]
        }
        self.mock_get.return_value = mock_response
        
        client = OllamaClient(model="llama3")
        self.assertEqual(client.model, "mistral")
    
    def test_validate_setup_api_error(self):
        """Test handling of API errors during validation."""
        self.mock_get.side_effect = Exception("Connection error")
        
        with mock.patch('smart_git_commit.processor.logger.warning') as mock_warn:
            client = OllamaClient()
            mock_warn.assert_called()
            self.assertEqual(client.model, "llama3")
    
    @mock.patch('requests.post')
    def test_generate_success(self, mock_post):
        """Test successful text generation."""
        # Mock successful generation API response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Generated text from model"
        }
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        result = client.generate("Test prompt", system_prompt="System prompt")
        
        # Check that the result matches the expected output
        self.assertEqual(result, "Generated text from model")
        
        # Check that the API was called with the right parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        self.assertEqual(kwargs['headers'], {"Content-Type": "application/json"})
        
        # Check request data
        request_data = json.loads(kwargs['data'])
        self.assertEqual(request_data['model'], "llama3")
        self.assertEqual(request_data['prompt'], "Test prompt")
        self.assertEqual(request_data['system'], "System prompt")
        self.assertEqual(request_data['options']['num_predict'], 500)
    
    @mock.patch('requests.post')
    def test_generate_error(self, mock_post):
        """Test handling of errors during generation."""
        mock_post.side_effect = Exception("API error")
        
        with mock.patch('smart_git_commit.processor.logger.error') as mock_error:
            client = OllamaClient()
            result = client.generate("Test prompt")
            
            # Should return empty string on error
            self.assertEqual(result, "")
            
            # Error should be logged
            mock_error.assert_called_once()


class TestOpenAIClient(unittest.TestCase):
    """Tests for the OpenAIClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Mock environment variables
        self.patcher_env = mock.patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-api-key',
            'OPENAI_API_MODEL': 'gpt-3.5-turbo',
            'OPENAI_API_PROXY': 'http://test-proxy.example.com'
        })
        self.patcher_env.start()
        
        # Mock OpenAI library
        self.patcher_openai = mock.patch('openai.OpenAI')
        self.mock_openai = self.patcher_openai.start()
        
        # Mock OpenAI chat completion
        self.mock_completion = mock.MagicMock()
        self.mock_completion.choices[0].message.content = "Formatted commit message"
        self.mock_openai.ChatCompletion.create.return_value = self.mock_completion
        
        # Mock logger
        self.patcher_logger = mock.patch('smart_git_commit.processor.logger')
        self.mock_logger = self.patcher_logger.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Restore original directory
        os.chdir(self.original_dir)
        
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
        
        # Stop patchers
        self.patcher_env.stop()
        self.patcher_openai.stop()
        self.patcher_logger.stop()
    
    def test_init_with_key(self):
        """Test initialization with API key."""
        client = OpenAIClient(api_key="direct_key")
        self.assertEqual(client.api_key, "direct_key")
    
    def test_init_with_env_key(self):
        """Test initialization with environment variable."""
        client = OpenAIClient()
        self.assertEqual(client.api_key, "test-api-key")
    
    @mock.patch('openai.OpenAI')
    def test_generate_success(self, mock_openai_class):
        """Test successful text generation."""
        mock_client = mock.MagicMock()
        mock_completion = mock.MagicMock()
        mock_completion.choices = [mock.MagicMock(message=mock.MagicMock(content="Generated text"))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        client = OpenAIClient()
        result = client.generate("Test prompt", system_prompt="System prompt")
        self.assertEqual(result, "Generated text")
    
    @mock.patch('openai.OpenAI')
    def test_generate_error(self, mock_openai_class):
        """Test error handling in text generation."""
        mock_openai_class.side_effect = Exception("API Error")
        
        client = OpenAIClient()
        result = client.generate("Test prompt")
        self.assertEqual(result, "")
    
    def test_generate_no_key(self):
        """Test generation attempt without API key."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        client = OpenAIClient()
        result = client.generate("Test prompt")
        self.assertEqual(result, "")


class TestCommitProcessor(unittest.TestCase):
    """Test the CommitProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=self.repo_path, capture_output=True)
        
        self.mock_config = {
            'openai_api_key': 'test-key',
            'openai_model': 'gpt-3.5-turbo',
            'ollama_host': 'http://localhost:11434',
            'ollama_model': 'llama3',
            'component_rules': {
                'src/(.+?)(/|$)': r'\1',
                'tests/(.+?)(/|$)': 'tests',
                'docs/(.+?)(/|$)': 'docs'
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_determine_component_with_rules(self, mock_repo, mock_config):
        """Test component determination with different rules."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        test_cases = [
            ('src/api/routes.py', 'api'),
            ('src/core/utils/helpers.py', 'core'),
            ('tests/test_api.py', 'tests'),
            ('docs/api.md', 'docs'),
            ('unknown/file.txt', 'misc'),
            ('frontend/components/Button.tsx', 'frontend'),
            ('styles/main.css', 'ui'),
            ('config/settings.json', 'config')
        ]
        
        for filename, expected in test_cases:
            component = processor._determine_component(filename)
            self.assertEqual(component, expected, f"Failed for {filename}")
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_get_diff_error_handling(self, mock_repo, mock_config):
        """Test diff retrieval error handling."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        # Test with nonexistent file
        diff = processor._get_diff('nonexistent.txt', 'M')
        self.assertEqual(diff, '')
        
        # Test with deleted file
        diff = processor._get_diff('deleted.txt', 'D')
        self.assertEqual(diff, '')
        
        # Test with untracked file
        diff = processor._get_diff('new.txt', '??')
        self.assertEqual(diff, '')
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_extract_issues_complex(self, mock_repo, mock_config):
        """Test issue extraction with complex scenarios."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('file1.txt', 'M', 'misc', 'Fix for #123\nRelated to #456'),
            Change('file2.txt', 'M', 'misc', 'See issue #789\nAlso fixes #101'),
            Change('file3.txt', 'A', 'misc', 'New feature #202')
        ]
        
        issues = processor._extract_issues(changes)
        self.assertEqual(issues, {'#123', '#456', '#789', '#101', '#202'})
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_commit_changes_error_handling(self, mock_repo, mock_config):
        """Test error handling in commit_changes method."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        # Test with invalid file
        group = CommitGroup(
            changes=[Change('nonexistent.txt', 'M', 'misc')],
            commit_type=CommitType.FEAT
        )
        success = processor.commit_changes(group, "test commit")
        self.assertFalse(success)
        
        # Test with git error
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'git', stderr=b'error')):
            success = processor.commit_changes(group, "test commit")
            self.assertFalse(success)
        
        # Test with general error
        with patch('subprocess.run', side_effect=Exception('test error')):
            success = processor.commit_changes(group, "test commit")
            self.assertFalse(success)

    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_get_changes_with_git_error(self, mock_repo, mock_config):
        """Test getting changes with git command error."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "git error"
            
            processor = CommitProcessor(use_ai=False)
            changes = processor.get_changes()
            self.assertEqual(len(changes), 0)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_get_changes_with_empty_status(self, mock_repo, mock_config):
        """Test getting changes with empty git status."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            
            processor = CommitProcessor(use_ai=False)
            changes = processor.get_changes()
            self.assertEqual(len(changes), 0)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_get_diff_for_new_file(self, mock_repo, mock_config):
        """Test getting diff for a new file."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        # Test new file (status A)
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "+new content"
            diff = processor._get_diff("new.txt", "A")
            self.assertEqual(diff, "+new content")
            
            # Verify git diff --cached was used
            mock_run.assert_called_with(
                ['git', 'diff', '--cached', '--', 'new.txt'],
                cwd=processor.repo_path,
                capture_output=True,
                text=True
            )
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_get_diff_for_modified_file(self, mock_repo, mock_config):
        """Test getting diff for a modified file."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        # Test modified file (status M)
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "-old\n+new"
            diff = processor._get_diff("modified.txt", "M")
            self.assertEqual(diff, "-old\n+new")
            
            # Verify git diff was used without --cached
            mock_run.assert_called_with(
                ['git', 'diff', '--', 'modified.txt'],
                cwd=processor.repo_path,
                capture_output=True,
                text=True
            )
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_get_diff_with_error(self, mock_repo, mock_config):
        """Test getting diff with command error."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        with patch('subprocess.run', side_effect=Exception("git error")):
            diff = processor._get_diff("file.txt", "M")
            self.assertEqual(diff, "")
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_determine_component_with_capture_group(self, mock_repo, mock_config):
        """Test component determination with capture group in rule."""
        config = self.mock_config.copy()
        config['component_rules'] = {
            'src/(.+?)/.*': r'\1',  # Capture first directory after src/
            'tests/(.+?)/.*': 'tests'
        }
        mock_config.return_value = config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        # Test with capture group rule
        component = processor._determine_component('src/auth/login.py')
        self.assertEqual(component, 'auth')
        
        # Test with non-capture group rule
        component = processor._determine_component('tests/auth/test_login.py')
        self.assertEqual(component, 'tests')
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_determine_component_by_extension(self, mock_repo, mock_config):
        """Test component determination by file extension."""
        mock_config.return_value = {'component_rules': {}}  # No rules
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        test_cases = [
            ('app.js', 'frontend'),
            ('app.ts', 'frontend'),
            ('app.jsx', 'frontend'),
            ('app.tsx', 'frontend'),
            ('app.py', 'backend'),
            ('app.rb', 'backend'),
            ('app.java', 'backend'),
            ('app.go', 'backend'),
            ('app.rs', 'backend'),
            ('app.c', 'backend'),
            ('app.cpp', 'backend'),
            ('app.h', 'backend'),
            ('style.css', 'ui'),
            ('style.scss', 'ui'),
            ('template.html', 'ui'),
            ('icon.svg', 'ui'),
            ('README.md', 'docs'),
            ('doc.rst', 'docs'),
            ('notes.txt', 'docs'),
            ('config.json', 'config'),
            ('config.yml', 'config'),
            ('config.yaml', 'config'),
            ('config.toml', 'config'),
            ('config.ini', 'config'),
            ('unknown.xyz', 'misc')
        ]
        
        for filename, expected in test_cases:
            component = processor._determine_component(filename)
            self.assertEqual(component, expected, f"Failed for {filename}")
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_extract_issues_with_duplicates(self, mock_repo, mock_config):
        """Test issue extraction with duplicate references."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('file1.txt', 'M', 'misc', 'Fix for #123\nAlso fixes #123'),
            Change('file2.txt', 'M', 'misc', 'Related to #123 and #456')
        ]
        
        issues = processor._extract_issues(changes)
        self.assertEqual(issues, {'#123', '#456'})  # Duplicates are removed


class TestConfiguration(unittest.TestCase):
    """Tests for Configuration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_no_config(self):
        """Test initialization without config file."""
        config = Configuration()
        self.assertEqual(config.get('nonexistent'), None)
        self.assertEqual(config.get('nonexistent', 'default'), 'default')
    
    def test_init_with_config(self):
        """Test initialization with valid config file."""
        test_config = {
            'key1': 'value1',
            'key2': {'nested': 'value2'}
        }
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        config = Configuration(self.config_path)
        self.assertEqual(config.get('key1'), 'value1')
        self.assertEqual(config.get('key2')['nested'], 'value2')
    
    def test_init_invalid_json(self):
        """Test initialization with invalid JSON config."""
        with open(self.config_path, 'w') as f:
            f.write('invalid json')
        
        config = Configuration(self.config_path)
        self.assertEqual(config.get('any_key'), None)
    
    def test_set_value(self):
        """Test setting configuration values."""
        config = Configuration(self.config_path)
        config.set('new_key', 'new_value')
        self.assertEqual(config.get('new_key'), 'new_value')
        
        # Verify file was written
        with open(self.config_path, 'r') as f:
            saved_config = json.load(f)
        self.assertEqual(saved_config['new_key'], 'new_value')
    
    def test_set_value_no_path(self):
        """Test setting value without config path."""
        config = Configuration()
        config.set('new_key', 'new_value')
        self.assertEqual(config.get('new_key'), 'new_value')
    
    def test_set_value_invalid_path(self):
        """Test setting value with invalid config path."""
        config = Configuration('/nonexistent/path/config.json')
        config.set('new_key', 'new_value')
        self.assertEqual(config.get('new_key'), 'new_value')


if __name__ == '__main__':
    unittest.main()

def test_get_processor():
    """Test get_processor function."""
    # Test with default configuration (ollama)
    with mock.patch('smart_git_commit.processor.load_config') as mock_load_config:
        mock_config = mock.MagicMock()
        mock_config.get.return_value = 'ollama'
        mock_load_config.return_value = mock_config
        
        processor = get_processor()
        assert isinstance(processor, CommitProcessor)
        assert processor.use_ai
        assert processor.client is not None
        assert isinstance(processor.client, OllamaClient)
    
    # Test with OpenAI configuration
    with mock.patch('smart_git_commit.processor.load_config') as mock_load_config:
        mock_config = mock.MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            'processor': 'openai',
            'openai_api_key': 'test_key'
        }.get(key, default)
        mock_load_config.return_value = mock_config
        
        processor = get_processor()
        assert isinstance(processor, CommitProcessor)
        assert processor.use_ai
        assert processor.client is not None
        assert isinstance(processor.client, OpenAIClient)
    
    # Test with invalid configuration
    with mock.patch('smart_git_commit.processor.load_config') as mock_load_config:
        mock_config = mock.MagicMock()
        mock_config.get.side_effect = lambda key, default=None: {
            'processor': 'invalid'
        }.get(key, default)
        mock_load_config.return_value = mock_config
        
        with pytest.raises(ValueError):
            get_processor()


class TestConfigFunctions(unittest.TestCase):
    """Tests for configuration functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create a mock home directory
        self.home_dir = os.path.join(self.test_dir, 'home')
        os.makedirs(self.home_dir)
        self.patcher_expanduser = mock.patch('os.path.expanduser', return_value=self.home_dir)
        self.mock_expanduser = self.patcher_expanduser.start()
        
        # Create a mock repository
        os.makedirs(os.path.join(self.test_dir, '.git'))
        
        # Mock logger
        self.patcher_logger = mock.patch('smart_git_commit.processor.logger')
        self.mock_logger = self.patcher_logger.start()
        
        # Mock environment variable
        self.patcher_environ = mock.patch.dict('os.environ', {})
        self.mock_environ = self.patcher_environ.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Restore original directory
        os.chdir(self.original_dir)
        
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
        
        # Stop patchers
        self.patcher_expanduser.stop()
        self.patcher_logger.stop()
        self.patcher_environ.stop()
    
    def test_get_config_file_path(self):
        """Test get_config_file_path function."""
        # Create config files
        repo_config_path = os.path.join(self.test_dir, '.smart-git-commit.json')
        with open(repo_config_path, 'w') as f:
            f.write('{"repo_config": true}')
        
        home_config_path = os.path.join(self.home_dir, '.smart-git-commit.json')
        with open(home_config_path, 'w') as f:
            f.write('{"home_config": true}')
        
        # Test repo config is preferred
        result = get_config_file_path()
        self.assertEqual(result, repo_config_path)
        
        # Test home config is used when repo config doesn't exist
        os.remove(repo_config_path)
        result = get_config_file_path()
        self.assertEqual(result, home_config_path)
        
        # Test None is returned when no config exists
        os.remove(home_config_path)
        result = get_config_file_path()
        self.assertIsNone(result)
    
    def test_load_config(self):
        """Test load_config function."""
        # Create config file
        config_path = os.path.join(self.test_dir, '.smart-git-commit.json')
        with open(config_path, 'w') as f:
            f.write('{"api_key": "test_key", "model": "test_model"}')
        
        # Test loading config
        with mock.patch('smart_git_commit.processor.get_config_file_path', return_value=config_path):
            config = load_config()
            
            self.assertEqual(config.get('api_key'), 'test_key')
            self.assertEqual(config.get('model'), 'test_model')
        
        # Test loading invalid JSON
        with open(config_path, 'w') as f:
            f.write('{"invalid JSON"')
        
        with mock.patch('smart_git_commit.processor.get_config_file_path', return_value=config_path):
            config = load_config()
            
            self.assertEqual(config, {})
            self.mock_logger.error.assert_called()
        
        # Test loading non-existent file
        os.remove(config_path)
        with mock.patch('smart_git_commit.processor.get_config_file_path', return_value=config_path):
            config = load_config()
            
            self.assertEqual(config, {})
            self.mock_logger.warning.assert_called()
        
        # Test loading None path
        with mock.patch('smart_git_commit.processor.get_config_file_path', return_value=None):
            config = load_config()
            
            self.assertEqual(config, {})


class TestGetProcessorFunction(unittest.TestCase):
    """Tests for get_processor function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, '.smart-git-commit.json')
        os.environ['OPENAI_API_KEY'] = 'test_key'
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
    
    @patch('smart_git_commit.processor.load_config')
    def test_get_processor_default(self, mock_load_config):
        """Test get_processor with default configuration."""
        mock_load_config.return_value = Configuration()
        
        processor = get_processor()
        self.assertIsInstance(processor, CommitProcessor)
        self.assertTrue(processor.use_ai)
        self.assertIsInstance(processor.ai_client, OllamaClient)
    
    @patch('smart_git_commit.processor.load_config')
    def test_get_processor_openai(self, mock_load_config):
        """Test get_processor with OpenAI configuration."""
        config = Configuration()
        config.set('processor', 'openai')
        config.set('openai_api_key', 'test_key')
        mock_load_config.return_value = config
        
        processor = get_processor()
        self.assertIsInstance(processor, CommitProcessor)
        self.assertTrue(processor.use_ai)
        self.assertIsInstance(processor.ai_client, OpenAIClient)
    
    @patch('smart_git_commit.processor.load_config')
    def test_get_processor_invalid(self, mock_load_config):
        """Test get_processor with invalid configuration."""
        config = Configuration()
        config.set('processor', 'invalid')
        mock_load_config.return_value = config
        
        with self.assertRaises(ValueError) as ctx:
            get_processor()
        self.assertIn('Invalid processor type', str(ctx.exception))


class TestConfigFileHandling(unittest.TestCase):
    """Tests for configuration file handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.home_dir = os.path.join(self.temp_dir, 'home')
        os.makedirs(self.home_dir)
        self.patcher = patch('os.path.expanduser', return_value=self.home_dir)
        self.patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        self.patcher.stop()
    
    def test_get_config_file_path_repo(self):
        """Test getting config path from repository."""
        repo_config = os.path.join(self.temp_dir, '.smart-git-commit.json')
        with open(repo_config, 'w') as f:
            f.write('{}')
        
        with patch('os.getcwd', return_value=self.temp_dir):
            path = get_config_file_path()
            self.assertEqual(path, repo_config)
    
    def test_get_config_file_path_home(self):
        """Test getting config path from home directory."""
        home_config = os.path.join(self.home_dir, '.smart-git-commit.json')
        with open(home_config, 'w') as f:
            f.write('{}')
        
        path = get_config_file_path()
        self.assertEqual(path, home_config)
    
    def test_get_config_file_path_none(self):
        """Test getting config path when no config exists."""
        path = get_config_file_path()
        self.assertIsNone(path)
    
    def test_load_config_repo(self):
        """Test loading config from repository."""
        repo_config = os.path.join(self.temp_dir, '.smart-git-commit.json')
        config_data = {'key': 'value'}
        with open(repo_config, 'w') as f:
            json.dump(config_data, f)
        
        with patch('os.getcwd', return_value=self.temp_dir):
            config = load_config()
            self.assertEqual(config.get('key'), 'value')
    
    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON."""
        repo_config = os.path.join(self.temp_dir, '.smart-git-commit.json')
        with open(repo_config, 'w') as f:
            f.write('invalid json')
        
        with patch('os.getcwd', return_value=self.temp_dir):
            config = load_config()
            self.assertEqual(config.get('any_key'), None)
    
    def test_load_config_no_file(self):
        """Test loading config when no file exists."""
        config = load_config()
        self.assertEqual(config.get('any_key'), None)


class TestAIClients(unittest.TestCase):
    """Tests for AI client classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        os.environ['OPENAI_API_KEY'] = 'test_key'
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
    
    def test_ollama_client_init_custom_host(self):
        """Test OllamaClient initialization with custom host."""
        client = OllamaClient(host='http://custom:11434')
        self.assertEqual(client.host, 'http://custom:11434')
    
    def test_ollama_client_validate_setup_no_models(self):
        """Test OllamaClient setup validation with no models."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'models': []}
            
            client = OllamaClient()
            self.assertEqual(client.model, 'llama3')  # Default model
    
    def test_ollama_client_validate_setup_api_error(self):
        """Test OllamaClient setup validation with API error."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 500
            
            client = OllamaClient()
            self.assertEqual(client.model, 'llama3')  # Default model
    
    def test_ollama_client_generate_with_system_prompt(self):
        """Test OllamaClient text generation with system prompt."""
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {'response': 'Generated text'}
            
            client = OllamaClient()
            result = client.generate('Test prompt', system_prompt='System prompt')
            
            self.assertEqual(result, 'Generated text')
            mock_post.assert_called_once()
            
            # Verify request data
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]['data'])
            self.assertEqual(request_data['system'], 'System prompt')
    
    def test_openai_client_init_with_env_key(self):
        """Test OpenAIClient initialization with environment key."""
        client = OpenAIClient()
        self.assertEqual(client.api_key, 'test_key')
    
    def test_openai_client_init_with_direct_key(self):
        """Test OpenAIClient initialization with direct key."""
        client = OpenAIClient(api_key='direct_key')
        self.assertEqual(client.api_key, 'direct_key')
    
    def test_openai_client_generate_with_system_prompt(self):
        """Test OpenAIClient text generation with system prompt."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_completion = MagicMock()
            mock_completion.choices = [
                MagicMock(message=MagicMock(content='Generated text'))
            ]
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client
            
            client = OpenAIClient()
            result = client.generate('Test prompt', system_prompt='System prompt')
            
            self.assertEqual(result, 'Generated text')
            mock_client.chat.completions.create.assert_called_once()
            
            # Verify request data
            call_args = mock_client.chat.completions.create.call_args
            self.assertEqual(call_args[1]['messages'][0]['content'], 'System prompt')
            self.assertEqual(call_args[1]['messages'][1]['content'], 'Test prompt')

class TestCommitProcessorInit(unittest.TestCase):
    """Tests for CommitProcessor initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, '.smart-git-commit.json')
        self.mock_config = {
            'openai_api_key': 'test-key',
            'openai_model': 'gpt-3.5-turbo',
            'ollama_host': 'http://localhost:11434',
            'ollama_model': 'llama3',
            'component_rules': {
                'src/(.+?)(/|$)': r'\1',
                'tests/(.+?)(/|$)': 'tests'
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_init_no_ai(self, mock_repo, mock_config):
        """Test initialization without AI."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        self.assertFalse(processor.use_ai)
        self.assertIsNone(processor.ai_client)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_init_with_ollama(self, mock_repo, mock_config):
        """Test initialization with Ollama."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        self.assertTrue(processor.use_ai)
        self.assertIsInstance(processor.ai_client, OllamaClient)
        self.assertEqual(processor.ai_client.host, 'http://localhost:11434')
        self.assertEqual(processor.ai_client.model, 'llama3')
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_init_with_openai(self, mock_repo, mock_config):
        """Test initialization with OpenAI."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='openai')
        self.assertTrue(processor.use_ai)
        self.assertIsInstance(processor.ai_client, OpenAIClient)
        self.assertEqual(processor.ai_client.api_key, 'test-key')
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_init_invalid_provider(self, mock_repo, mock_config):
        """Test initialization with invalid AI provider."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        with self.assertRaises(ValueError) as ctx:
            CommitProcessor(use_ai=True, ai_provider='invalid')
        self.assertIn('Unsupported AI provider', str(ctx.exception))
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_init_missing_openai_key(self, mock_repo, mock_config):
        """Test initialization with missing OpenAI key."""
        config = self.mock_config.copy()
        del config['openai_api_key']
        mock_config.return_value = config
        mock_repo.return_value = {'name': 'test-repo'}
        
        with self.assertRaises(ValueError) as ctx:
            CommitProcessor(use_ai=True, ai_provider='openai')
        self.assertIn('OpenAI API key is required', str(ctx.exception))
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_init_with_custom_repo_path(self, mock_repo, mock_config):
        """Test initialization with custom repository path."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo', 'path': '/custom/path'}
        
        processor = CommitProcessor(use_ai=False)
        self.assertEqual(processor.repo_path, os.path.abspath('.'))
        self.assertEqual(processor.repo_details['name'], 'test-repo')
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_init_with_empty_config(self, mock_repo, mock_config):
        """Test initialization with empty configuration."""
        mock_config.return_value = {}
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        self.assertTrue(processor.use_ai)
        self.assertIsInstance(processor.ai_client, OllamaClient)
        self.assertEqual(processor.component_rules, {})

class TestCommitProcessorFunctionality(unittest.TestCase):
    """Tests for CommitProcessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=self.repo_path, capture_output=True)
        
        self.mock_config = {
            'openai_api_key': 'test-key',
            'component_rules': {
                'src/(.+?)(/|$)': r'\1',
                'tests/(.+?)(/|$)': 'tests',
                'docs/(.+?)(/|$)': 'docs'
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_get_changes_with_files(self, mock_repo, mock_config):
        """Test getting changes with actual files."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo', 'path': str(self.repo_path)}
        
        # Create some test files
        (self.repo_path / "src" / "api").mkdir(parents=True)
        (self.repo_path / "src" / "api" / "routes.py").write_text("test content")
        (self.repo_path / "tests").mkdir()
        (self.repo_path / "tests" / "test_api.py").write_text("test content")
        
        # Stage the files
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        
        processor = CommitProcessor(use_ai=False)
        changes = processor.get_changes()
        
        self.assertEqual(len(changes), 2)
        self.assertTrue(any(c.filename == "src/api/routes.py" for c in changes))
        self.assertTrue(any(c.filename == "tests/test_api.py" for c in changes))
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_get_changes_empty_repo(self, mock_repo, mock_config):
        """Test getting changes in empty repository."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo', 'path': str(self.repo_path)}
        
        processor = CommitProcessor(use_ai=False)
        changes = processor.get_changes()
        self.assertEqual(len(changes), 0)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_get_changes_with_renames(self, mock_repo, mock_config):
        """Test getting changes with renamed files."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo', 'path': str(self.repo_path)}
        
        # Create and rename a file
        (self.repo_path / "src").mkdir()
        (self.repo_path / "src" / "old.py").write_text("test content")
        subprocess.run(["git", "add", "."], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=self.repo_path, capture_output=True)
        
        os.rename(
            str(self.repo_path / "src" / "old.py"),
            str(self.repo_path / "src" / "new.py")
        )
        subprocess.run(["git", "add", "-A"], cwd=self.repo_path, capture_output=True)
        
        processor = CommitProcessor(use_ai=False)
        changes = processor.get_changes()
        
        self.assertTrue(any(c.status == "R" and c.filename == "src/new.py" for c in changes))
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_determine_commit_type_mixed(self, mock_repo, mock_config):
        """Test commit type determination with mixed changes."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        # Test with mixed test and non-test files
        changes = [
            Change("src/api/routes.py", "M", "api"),
            Change("tests/test_api.py", "A", "tests"),
            Change("tests/test_routes.py", "A", "tests")
        ]
        
        commit_type = processor._determine_commit_type(changes)
        self.assertEqual(commit_type, CommitType.TEST)  # Majority are test files
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_determine_commit_type_build(self, mock_repo, mock_config):
        """Test commit type determination with build files."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change("build/webpack.config.js", "M", "build"),
            Change("package.json", "M", "build")
        ]
        
        commit_type = processor._determine_commit_type(changes)
        self.assertEqual(commit_type, CommitType.BUILD)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_generate_ai_commit_message_with_diff(self, mock_repo, mock_config):
        """Test AI commit message generation with diff content."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "feat(api): add user authentication"
        
        changes = [
            Change(
                "src/api/auth.py",
                "A",
                "api",
                "diff --git a/src/api/auth.py b/src/api/auth.py\n+def authenticate_user():\n+    pass"
            )
        ]
        group = CommitGroup(changes=changes, commit_type=CommitType.FEAT)
        
        message = processor._generate_ai_commit_message(group)
        self.assertEqual(message, "feat(api): add user authentication")
        
        # Verify AI was called with diff content
        call_args = processor.ai_client.generate.call_args
        self.assertIn("authenticate_user", call_args[0][0])
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_commit_changes_success(self, mock_repo, mock_config):
        """Test successful commit of changes."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo', 'path': str(self.repo_path)}
        
        # Create a test file
        (self.repo_path / "test.txt").write_text("test content")
        
        processor = CommitProcessor(use_ai=False)
        group = CommitGroup(
            changes=[Change("test.txt", "A", "misc")],
            commit_type=CommitType.FEAT
        )
        
        success = processor.commit_changes(group, "test commit")
        self.assertTrue(success)
        
        # Verify commit was made
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        self.assertIn("test commit", result.stdout)

    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_group_changes_by_component(self, mock_repo, mock_config):
        """Test grouping changes by component."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('src/api/routes.py', 'M', 'api'),
            Change('src/api/models.py', 'A', 'api'),
            Change('tests/test_api.py', 'A', 'tests'),
            Change('docs/api.md', 'M', 'docs')
        ]
        
        groups = processor.group_changes(changes)
        
        # Verify groups are created by component
        self.assertEqual(len(groups), 3)  # api, tests, docs
        
        api_group = next(g for g in groups if g.component == 'api')
        test_group = next(g for g in groups if g.component == 'tests')
        docs_group = next(g for g in groups if g.component == 'docs')
        
        self.assertEqual(len(api_group.changes), 2)
        self.assertEqual(len(test_group.changes), 1)
        self.assertEqual(len(docs_group.changes), 1)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_group_changes_empty(self, mock_repo, mock_config):
        """Test grouping empty changes list."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        groups = processor.group_changes([])
        self.assertEqual(len(groups), 0)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_group_changes_single_component(self, mock_repo, mock_config):
        """Test grouping changes from single component."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('src/api/routes.py', 'M', 'api'),
            Change('src/api/models.py', 'A', 'api'),
            Change('src/api/views.py', 'A', 'api')
        ]
        
        groups = processor.group_changes(changes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].component, 'api')
        self.assertEqual(len(groups[0].changes), 3)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_determine_commit_type_style(self, mock_repo, mock_config):
        """Test commit type determination for style changes."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('src/styles/main.css', 'M', 'ui'),
            Change('src/styles/components.scss', 'M', 'ui')
        ]
        
        commit_type = processor._determine_commit_type(changes)
        self.assertEqual(commit_type, CommitType.STYLE)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_determine_commit_type_mixed_with_majority(self, mock_repo, mock_config):
        """Test commit type determination with mixed changes but clear majority."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('src/api/routes.py', 'M', 'api'),
            Change('tests/test_api.py', 'A', 'tests'),
            Change('tests/test_routes.py', 'A', 'tests'),
            Change('tests/test_models.py', 'A', 'tests')
        ]
        
        commit_type = processor._determine_commit_type(changes)
        self.assertEqual(commit_type, CommitType.TEST)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_determine_commit_type_docs(self, mock_repo, mock_config):
        """Test commit type determination for documentation changes."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('docs/api.md', 'M', 'docs'),
            Change('README.md', 'M', 'docs')
        ]
        
        commit_type = processor._determine_commit_type(changes)
        self.assertEqual(commit_type, CommitType.DOCS)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_generate_ai_commit_message_with_context(self, mock_repo, mock_config):
        """Test AI commit message generation with full context."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "feat(auth): implement OAuth2 authentication"
        
        changes = [
            Change(
                'src/auth/oauth.py',
                'A',
                'auth',
                'diff --git a/src/auth/oauth.py b/src/auth/oauth.py\n'
                '+class OAuth2Provider:\n'
                '+    def authenticate(self):\n'
                '+        pass'
            )
        ]
        group = CommitGroup(
            changes=changes,
            commit_type=CommitType.FEAT,
            component='auth',
            scope='auth',
            description='Add OAuth2 support',
            issues={'#789'}
        )
        
        message = processor._generate_ai_commit_message(group)
        
        # Verify AI was called with correct context
        call_args = processor.ai_client.generate.call_args
        prompt = call_args[0][0]
        self.assertIn('src/auth/oauth.py', prompt)
        self.assertIn('OAuth2Provider', prompt)
        self.assertIn('authenticate', prompt)
        self.assertIn('#789', prompt)
        self.assertIn('feat', prompt)
        self.assertIn('auth', prompt)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_generate_ai_commit_message_with_multiple_files(self, mock_repo, mock_config):
        """Test AI commit message generation with multiple files."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "feat(auth): implement user authentication and session management"
        
        changes = [
            Change(
                'src/auth/user.py',
                'A',
                'auth',
                'diff --git a/src/auth/user.py b/src/auth/user.py\n'
                '+def authenticate_user(username, password):\n'
                '+    pass'
            ),
            Change(
                'src/auth/session.py',
                'A',
                'auth',
                'diff --git a/src/auth/session.py b/src/auth/session.py\n'
                '+def create_session(user_id):\n'
                '+    pass'
            )
        ]
        group = CommitGroup(
            changes=changes,
            commit_type=CommitType.FEAT,
            component='auth'
        )
        
        message = processor._generate_ai_commit_message(group)
        
        # Verify AI was called with all files
        call_args = processor.ai_client.generate.call_args
        prompt = call_args[0][0]
        self.assertIn('src/auth/user.py', prompt)
        self.assertIn('src/auth/session.py', prompt)
        self.assertIn('authenticate_user', prompt)
        self.assertIn('create_session', prompt)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_generate_ai_commit_message_with_system_prompt(self, mock_repo, mock_config):
        """Test AI commit message generation includes system prompt."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "feat(core): add new feature"
        
        changes = [Change('src/core/feature.py', 'A', 'core')]
        group = CommitGroup(changes=changes, commit_type=CommitType.FEAT)
        
        processor._generate_ai_commit_message(group)
        
        # Verify system prompt was included
        call_args = processor.ai_client.generate.call_args
        system_prompt = call_args[1].get('system_prompt')
        self.assertIsNotNone(system_prompt)
        self.assertIn('Conventional Commits', system_prompt)
        self.assertIn('imperative mood', system_prompt)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_generate_ai_commit_message_fallback_on_error(self, mock_repo, mock_config):
        """Test AI commit message generation falls back on error."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.side_effect = Exception("AI error")
        
        changes = [
            Change('src/core/feature.py', 'A', 'core', 'Added new feature'),
            Change('src/core/utils.py', 'M', 'core', 'Updated utilities')
        ]
        group = CommitGroup(
            changes=changes,
            commit_type=CommitType.FEAT,
            component='core'
        )
        
        message = processor._generate_ai_commit_message(group)
        
        # Verify fallback message format
        self.assertIn('feat', message)
        self.assertIn('core', message)
        self.assertIn('src/core/feature.py', message)
        self.assertIn('src/core/utils.py', message)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_generate_commit_message_no_ai_with_details(self, mock_repo, mock_config):
        """Test commit message generation without AI but with full details."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('src/auth/oauth.py', 'A', 'auth', 'Implement OAuth'),
            Change('src/auth/user.py', 'M', 'auth', 'Update user model')
        ]
        group = CommitGroup(
            changes=changes,
            commit_type=CommitType.FEAT,
            component='auth',
            scope='auth',
            description='Add OAuth support',
            body='Implement OAuth2 authentication flow\nAdd user model updates',
            breaking=True,
            issues={'#123', '#456'}
        )
        
        message = processor.generate_commit_message(group)
        
        # Verify all components are included
        self.assertIn('feat(auth)!', message)  # Type, scope, and breaking change
        self.assertIn('Add OAuth support', message)  # Description
        self.assertIn('Implement OAuth2 authentication flow', message)  # Body
        self.assertIn('Add user model updates', message)  # Body
        self.assertIn('src/auth/oauth.py', message)  # Files
        self.assertIn('src/auth/user.py', message)  # Files
        self.assertIn('#123', message)  # Issues
        self.assertIn('#456', message)  # Issues

    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_ai_analysis(self, mock_repo, mock_config):
        """Test processing changes with AI analysis."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "These changes implement a new API endpoint and update documentation"
        
        changes = [
            Change('src/api/routes.py', 'A', 'api', 'Added new endpoint'),
            Change('docs/api.md', 'M', 'docs', 'Updated API docs')
        ]
        
        groups = processor.process_changes(changes)
        
        # Verify AI was called for analysis
        processor.ai_client.generate.assert_called_once()
        
        # Verify groups were created
        self.assertEqual(len(groups), 2)  # One for API, one for docs
        
        api_group = next(g for g in groups if g.component == 'api')
        docs_group = next(g for g in groups if g.component == 'docs')
        
        self.assertEqual(len(api_group.changes), 1)
        self.assertEqual(len(docs_group.changes), 1)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_ai_error(self, mock_repo, mock_config):
        """Test processing changes when AI analysis fails."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.side_effect = Exception("AI error")
        
        changes = [
            Change('src/api/routes.py', 'A', 'api', 'Added new endpoint'),
            Change('docs/api.md', 'M', 'docs', 'Updated API docs')
        ]
        
        # Should fall back to rule-based grouping
        groups = processor.process_changes(changes)
        
        self.assertEqual(len(groups), 2)  # Should still group by component
        
        # Verify groups were created correctly despite AI error
        api_group = next(g for g in groups if g.component == 'api')
        docs_group = next(g for g in groups if g.component == 'docs')
        
        self.assertEqual(len(api_group.changes), 1)
        self.assertEqual(len(docs_group.changes), 1)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_importance_analysis(self, mock_repo, mock_config):
        """Test processing changes with importance analysis."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        
        # Mock AI responses for importance analysis
        processor.ai_client.generate.side_effect = [
            "This change adds a critical security fix",  # High importance
            "This change updates documentation",  # Low importance
        ]
        
        changes = [
            Change('src/auth/security.py', 'M', 'auth', 'Fixed security vulnerability'),
            Change('docs/readme.md', 'M', 'docs', 'Updated README')
        ]
        
        groups = processor.process_changes(changes)
        
        # Verify groups were created with correct ordering
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].changes[0].filename, 'src/auth/security.py')  # Security fix should be first
        self.assertEqual(groups[1].changes[0].filename, 'docs/readme.md')
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_breaking_changes(self, mock_repo, mock_config):
        """Test processing changes with breaking changes."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "This is a breaking change that modifies the API interface"
        
        changes = [
            Change('src/api/interface.py', 'M', 'api', 'Changed API interface\nBREAKING CHANGE'),
            Change('src/api/impl.py', 'M', 'api', 'Updated implementation')
        ]
        
        groups = processor.process_changes(changes)
        
        # Verify breaking change was detected
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0].breaking)
        self.assertEqual(len(groups[0].changes), 2)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_related_issues(self, mock_repo, mock_config):
        """Test processing changes with related issue references."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('src/bug.py', 'M', 'core', 'Fixed bug #123'),
            Change('tests/test_bug.py', 'A', 'tests', 'Added tests for #123'),
            Change('docs/fix.md', 'M', 'docs', 'Documented fix for #123 and #456')
        ]
        
        groups = processor.process_changes(changes)
        
        # Verify issue references were collected
        for group in groups:
            self.assertIn('#123', group.issues)
        
        # At least one group should also reference #456
        self.assertTrue(any('#456' in group.issues for group in groups))
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_formatting_only(self, mock_repo, mock_config):
        """Test processing changes that are formatting only."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        # Create changes that are just formatting
        changes = [
            Change(
                'src/code.py',
                'M',
                'core',
                'diff --git a/src/code.py b/src/code.py\n-def func(): return 1\n+def func():\n+    return 1'
            ),
            Change(
                'src/other.py',
                'M',
                'core',
                'diff --git a/src/other.py b/src/other.py\n-x=1\n+x = 1'
            )
        ]
        
        groups = processor.process_changes(changes)
        
        # Should be grouped as style changes
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].commit_type, CommitType.STYLE)
        self.assertEqual(len(groups[0].changes), 2)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_dependencies(self, mock_repo, mock_config):
        """Test processing changes to dependencies."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('package.json', 'M', 'deps', 'Updated dependencies'),
            Change('requirements.txt', 'M', 'deps', 'Updated Python packages'),
            Change('go.mod', 'M', 'deps', 'Updated Go modules')
        ]
        
        groups = processor.process_changes(changes)
        
        # Should be grouped as a single dependency update
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].commit_type, CommitType.BUILD)
        self.assertEqual(len(groups[0].changes), 3)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_mixed_scopes(self, mock_repo, mock_config):
        """Test processing changes with mixed scopes."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('src/api/routes.py', 'M', 'api', 'Updated routes'),
            Change('src/api/models.py', 'M', 'api', 'Updated models'),
            Change('src/core/utils.py', 'M', 'core', 'Updated utils'),
            Change('src/ui/components.js', 'M', 'ui', 'Updated components')
        ]
        
        groups = processor.process_changes(changes)
        
        # Should create separate groups for each scope
        self.assertEqual(len(groups), 3)  # api, core, ui
        
        scopes = [group.scope for group in groups]
        self.assertIn('api', scopes)
        self.assertIn('core', scopes)
        self.assertIn('ui', scopes)
        
        # API group should have 2 changes
        api_group = next(g for g in groups if g.scope == 'api')
        self.assertEqual(len(api_group.changes), 2)

    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_empty_input(self, mock_repo, mock_config):
        """Test processing empty changes list."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        groups = processor.process_changes([])
        self.assertEqual(len(groups), 0)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_unknown_component(self, mock_repo, mock_config):
        """Test processing changes with unknown component."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('unknown/file.txt', 'M', 'unknown', 'Updated file'),
            Change('misc/other.txt', 'M', 'misc', 'Updated other file')
        ]
        
        groups = processor.process_changes(changes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].component, 'misc')
        self.assertEqual(len(groups[0].changes), 2)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_multiple_ai_providers(self, mock_repo, mock_config):
        """Test processing changes with different AI providers."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        # Test with Ollama
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "feat(api): add new endpoint"
        
        changes = [Change('src/api/endpoint.py', 'A', 'api', 'Added endpoint')]
        groups = processor.process_changes(changes)
        self.assertEqual(len(groups), 1)
        
        # Test with OpenAI
        processor = CommitProcessor(use_ai=True, ai_provider='openai')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "feat(api): add new endpoint"
        
        groups = processor.process_changes(changes)
        self.assertEqual(len(groups), 1)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_custom_rules(self, mock_repo, mock_config):
        """Test processing changes with custom component rules."""
        config = self.mock_config.copy()
        config['component_rules'] = {
            'custom/(.+?)/.*': r'\1',
            'special/(.+?)/.*': r'\1'
        }
        mock_config.return_value = config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('custom/module1/file.py', 'M', 'module1', 'Updated file'),
            Change('special/module2/file.py', 'M', 'module2', 'Updated file')
        ]
        
        groups = processor.process_changes(changes)
        self.assertEqual(len(groups), 2)
        components = {g.component for g in groups}
        self.assertEqual(components, {'module1', 'module2'})
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_complex_diffs(self, mock_repo, mock_config):
        """Test processing changes with complex diff content."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "feat(api): implement complex changes"
        
        changes = [
            Change(
                'src/api/complex.py',
                'M',
                'api',
                'diff --git a/src/api/complex.py b/src/api/complex.py\n'
                '-class OldImplementation:\n'
                '-    def method1(self):\n'
                '-        pass\n'
                '+class NewImplementation:\n'
                '+    def method1(self):\n'
                '+        return "new"\n'
                '+    def method2(self):\n'
                '+        return "added"\n'
            )
        ]
        
        groups = processor.process_changes(changes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].commit_type, CommitType.FEAT)
        
        # Verify AI was called with complex diff
        call_args = processor.ai_client.generate.call_args
        prompt = call_args[0][0]
        self.assertIn('OldImplementation', prompt)
        self.assertIn('NewImplementation', prompt)
        self.assertIn('method2', prompt)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_multiple_commit_types(self, mock_repo, mock_config):
        """Test processing changes with multiple commit types."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('src/api/feature.py', 'A', 'api', 'Added new feature'),
            Change('src/api/bug.py', 'M', 'api', 'Fixed critical bug'),
            Change('tests/test_api.py', 'A', 'tests', 'Added tests'),
            Change('docs/api.md', 'M', 'docs', 'Updated docs')
        ]
        
        groups = processor.process_changes(changes)
        
        # Verify different commit types
        commit_types = {g.commit_type for g in groups}
        self.assertTrue(len(commit_types) >= 2)
        self.assertIn(CommitType.TEST, commit_types)
        self.assertIn(CommitType.DOCS, commit_types)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_merge_conflicts(self, mock_repo, mock_config):
        """Test processing changes with merge conflict markers."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "fix(core): resolve merge conflicts"
        
        changes = [
            Change(
                'src/core/conflict.py',
                'M',
                'core',
                'diff --git a/src/core/conflict.py b/src/core/conflict.py\n'
                '<<<<<<< HEAD\n'
                'def feature1():\n'
                '    return "main"\n'
                '=======\n'
                'def feature1():\n'
                '    return "branch"\n'
                '>>>>>>> feature-branch\n'
            )
        ]
        
        groups = processor.process_changes(changes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].commit_type, CommitType.FIX)
        
        # Verify AI was called with conflict markers
        call_args = processor.ai_client.generate.call_args
        prompt = call_args[0][0]
        self.assertIn('<<<<<<< HEAD', prompt)
        self.assertIn('>>>>>>> feature-branch', prompt)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_binary_files(self, mock_repo, mock_config):
        """Test processing changes with binary files."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('images/logo.png', 'A', 'assets', 'Added logo'),
            Change('docs/guide.pdf', 'M', 'docs', 'Updated guide')
        ]
        
        groups = processor.process_changes(changes)
        self.assertEqual(len(groups), 2)
        
        # Verify binary files are handled correctly
        assets_group = next(g for g in groups if g.component == 'assets')
        docs_group = next(g for g in groups if g.component == 'docs')
        
        self.assertEqual(len(assets_group.changes), 1)
        self.assertEqual(len(docs_group.changes), 1)
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_large_changes(self, mock_repo, mock_config):
        """Test processing changes with large number of files."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=True, ai_provider='ollama')
        processor.ai_client = MagicMock()
        processor.ai_client.generate.return_value = "refactor(core): major restructuring"
        
        # Create 50 changes across different components
        changes = []
        for i in range(50):
            component = ['api', 'core', 'ui', 'docs'][i % 4]
            changes.append(
                Change(f'src/{component}/file{i}.py', 'M', component, f'Updated file {i}')
            )
        
        groups = processor.process_changes(changes)
        
        # Verify changes are properly grouped
        self.assertTrue(len(groups) >= 4)  # At least one group per component
        for group in groups:
            self.assertGreater(len(group.changes), 0)
            self.assertLess(len(group.changes), 50)  # No single group has all changes
    
    @patch('smart_git_commit.processor.load_config')
    @patch('smart_git_commit.processor.get_repository_details')
    def test_process_changes_with_special_characters(self, mock_repo, mock_config):
        """Test processing changes with special characters in filenames."""
        mock_config.return_value = self.mock_config
        mock_repo.return_value = {'name': 'test-repo'}
        
        processor = CommitProcessor(use_ai=False)
        
        changes = [
            Change('src/api/test[1].py', 'M', 'api', 'Updated test file'),
            Change('src/api/file with spaces.py', 'M', 'api', 'Updated file'),
            Change('src/api/special#char.py', 'M', 'api', 'Updated special file')
        ]
        
        groups = processor.process_changes(changes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].changes), 3)
        
        # Verify filenames are preserved
        filenames = {c.filename for c in groups[0].changes}
        self.assertIn('src/api/test[1].py', filenames)
        self.assertIn('src/api/file with spaces.py', filenames)
        self.assertIn('src/api/special#char.py', filenames)


if __name__ == '__main__':
    unittest.main() 