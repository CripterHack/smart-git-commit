"""
Tests for the CLI Welcome Wizard.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys

# Add project root to sys.path to allow importing smart_git_commit modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from smart_git_commit.cli_wizard import run_cli_welcome_wizard
from smart_git_commit.config import Configuration

# Mock get_version as it's called in the wizard
@patch('smart_git_commit.cli_wizard.get_version', return_value="0.TEST.0")
class TestCliWelcomeWizard(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        # Use a temporary config file path for tests
        self.config_path = "temp_test_config.json"
        # Ensure no lingering config file from previous failed tests
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

        self.mock_config = Configuration()
        # Mock save to prevent actual file writing during most tests
        self.mock_config.save = MagicMock()
        # Mock the config file path property used in the wizard's success message
        self.mock_config.config_file_path = self.config_path

    def tearDown(self):
        """Clean up after test methods."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    @patch('rich.prompt.Prompt.ask')
    @patch('smart_git_commit.cli_wizard.OllamaClient')
    def test_ollama_setup_success(self, mock_ollama_client, mock_prompt_ask, mock_get_version):
        """Test the wizard flow for Ollama with successful model fetch."""
        # Mock OllamaClient instance and its methods
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_available_models.return_value = ['llama3:latest', 'gemma:2b']
        mock_ollama_client.return_value = mock_ollama_instance

        # Simulate user inputs using Prompt.ask mock
        mock_prompt_ask.side_effect = [
            'ollama',  # AI Provider
            'http://testhost:11434', # Ollama Host
            'llama3:latest', # Ollama Model
            '' # Press Enter to continue
        ]

        run_cli_welcome_wizard(self.mock_config, use_color=False)

        # Assertions
        self.assertEqual(self.mock_config.get('ai_provider'), 'ollama')
        self.assertTrue(self.mock_config.get('use_ai'))
        self.assertEqual(self.mock_config.get('ollama_host'), 'http://testhost:11434')
        self.assertEqual(self.mock_config.get('ollama_model'), 'llama3:latest')
        self.assertTrue(self.mock_config.get('welcome_completed'))
        self.mock_config.save.assert_called_once()
        mock_ollama_client.assert_called_once_with(host='http://testhost:11434', timeout=15)
        mock_ollama_instance.get_available_models.assert_called_once()

    @patch('rich.prompt.Prompt.ask')
    @patch('smart_git_commit.cli_wizard.OllamaClient')
    def test_ollama_setup_no_models(self, mock_ollama_client, mock_prompt_ask, mock_get_version):
        """Test the wizard flow for Ollama when no models are found."""
        mock_ollama_instance = MagicMock()
        mock_ollama_instance.get_available_models.return_value = []
        mock_ollama_client.return_value = mock_ollama_instance

        mock_prompt_ask.side_effect = [
            'ollama',
            'http://localhost:11434', # Default host
            '' # Press Enter
        ]

        run_cli_welcome_wizard(self.mock_config, use_color=False)

        self.assertEqual(self.mock_config.get('ai_provider'), 'ollama')
        self.assertTrue(self.mock_config.get('use_ai'))
        self.assertEqual(self.mock_config.get('ollama_host'), 'http://localhost:11434')
        self.assertIsNone(self.mock_config.get('ollama_model')) # Should be None
        self.assertTrue(self.mock_config.get('welcome_completed'))
        self.mock_config.save.assert_called_once()

    @patch('rich.prompt.Prompt.ask')
    @patch('smart_git_commit.cli_wizard.OllamaClient')
    def test_ollama_setup_connection_error(self, mock_ollama_client, mock_prompt_ask, mock_get_version):
        """Test the wizard flow for Ollama when connection fails."""
        mock_ollama_instance = MagicMock()
        # Simulate requests.exceptions.ConnectionError or similar
        mock_ollama_instance.get_available_models.side_effect = Exception("Connection failed")
        mock_ollama_client.return_value = mock_ollama_instance

        mock_prompt_ask.side_effect = [
            'ollama',
            'http://badhost:11434',
            '' # Press Enter
        ]

        run_cli_welcome_wizard(self.mock_config, use_color=False)

        self.assertEqual(self.mock_config.get('ai_provider'), 'ollama')
        self.assertTrue(self.mock_config.get('use_ai'))
        self.assertEqual(self.mock_config.get('ollama_host'), 'http://badhost:11434')
        self.assertIsNone(self.mock_config.get('ollama_model'))
        self.assertTrue(self.mock_config.get('welcome_completed'))
        self.mock_config.save.assert_called_once()

    @patch('rich.prompt.Prompt.ask')
    def test_openai_setup(self, mock_prompt_ask, mock_get_version):
        """Test the wizard flow for OpenAI setup."""
        mock_prompt_ask.side_effect = [
            'openai', # AI Provider
            'test_api_key', # API Key
            'gpt-4-test', # Model Name
            '' # Press Enter
        ]

        run_cli_welcome_wizard(self.mock_config, use_color=False)

        self.assertEqual(self.mock_config.get('ai_provider'), 'openai')
        self.assertTrue(self.mock_config.get('use_ai'))
        self.assertEqual(self.mock_config.get('openai_api_key'), 'test_api_key')
        self.assertEqual(self.mock_config.get('openai_model'), 'gpt-4-test')
        self.assertTrue(self.mock_config.get('welcome_completed'))
        self.mock_config.save.assert_called_once()

    @patch('rich.prompt.Prompt.ask')
    def test_no_ai_setup(self, mock_prompt_ask, mock_get_version):
        """Test the wizard flow when selecting 'none' for AI provider."""
        mock_prompt_ask.side_effect = [
            'none', # AI Provider
            '' # Press Enter
        ]

        run_cli_welcome_wizard(self.mock_config, use_color=False)

        self.assertEqual(self.mock_config.get('ai_provider'), 'none')
        self.assertFalse(self.mock_config.get('use_ai')) # use_ai should be False
        # Ensure Ollama/OpenAI specific fields weren't set
        self.assertIsNone(self.mock_config.get('ollama_model', None))
        self.assertIsNone(self.mock_config.get('openai_api_key', None))
        self.assertTrue(self.mock_config.get('welcome_completed'))
        self.mock_config.save.assert_called_once()

if __name__ == '__main__':
    unittest.main() 