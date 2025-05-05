#!/usr/bin/env python3
"""
Tests for the config.py module.
"""

import os
import sys
import unittest
from unittest import mock
import tempfile
import shutil
import json

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_git_commit.config import Configuration, DEFAULT_CONFIG


class TestConfiguration(unittest.TestCase):
    """Tests for the Configuration class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Mock logger
        self.patcher_logger = mock.patch('smart_git_commit.config.logger')
        self.mock_logger = self.patcher_logger.start()
        
        # Setup test configuration
        self.config_dir = os.path.join(self.test_dir, '.config')
        self.config_file = 'config.json'
        os.makedirs(self.config_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Restore original directory
        os.chdir(self.original_dir)
        
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
        
        # Stop patchers
        self.patcher_logger.stop()

    def create_config_file(self, config_data):
        """Helper to create a config file with specified data."""
        config_path = os.path.join(self.config_dir, self.config_file)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        return config_path

    def test_default_config(self):
        """Test that Configuration uses default values when no config file exists."""
        config = Configuration(self.config_dir, self.config_file)
        
        # Verify default values
        for key, value in DEFAULT_CONFIG.items():
            self.assertEqual(config.get(key), value)
    
    def test_load_config(self):
        """Test loading config from file."""
        # Create a config file
        test_config = {
            'theme': 'cyberpunk',
            'timeout': 120,
            'ollama_model': 'llama3'
        }
        self.create_config_file(test_config)
        
        # Load config
        config = Configuration(self.config_dir, self.config_file)
        
        # Verify config values were loaded
        self.assertEqual(config.get('theme'), 'cyberpunk')
        self.assertEqual(config.get('timeout'), 120)
        self.assertEqual(config.get('ollama_model'), 'llama3')
        
        # Verify other values defaulted
        self.assertEqual(config.get('security_scan'), DEFAULT_CONFIG['security_scan'])
    
    def test_load_invalid_json(self):
        """Test handling of invalid JSON in config file."""
        # Create an invalid config file
        config_path = os.path.join(self.config_dir, self.config_file)
        with open(config_path, 'w') as f:
            f.write('invalid json')
        
        # Load config
        config = Configuration(self.config_dir, self.config_file)
        
        # Verify default values are used
        self.assertEqual(config.get('theme'), DEFAULT_CONFIG['theme'])
        
        # Verify error was logged
        self.mock_logger.error.assert_called()
    
    def test_get_config_nonexistent(self):
        """Test get method with nonexistent key."""
        config = Configuration(self.config_dir, self.config_file)
        
        # Get with nonexistent key
        result = config.get('nonexistent_key')
        self.assertIsNone(result)
        
        # Get with nonexistent key and default value
        result = config.get('nonexistent_key', 'default_value')
        self.assertEqual(result, 'default_value')
    
    def test_set_config(self):
        """Test set method."""
        config = Configuration(self.config_dir, self.config_file)
        
        # Set a value
        config.set('theme', 'dracula')
        
        # Verify value was set
        self.assertEqual(config.get('theme'), 'dracula')
    
    def test_save_config(self):
        """Test save method."""
        # Load default config
        config = Configuration(self.config_dir, self.config_file)
        
        # Update property
        config.set('theme', 'dracula')
        config.set('timeout', 120)
        
        # Save config
        result = config.save()
        self.assertTrue(result)
        
        # Verify file was created
        config_path = os.path.join(self.config_dir, self.config_file)
        self.assertTrue(os.path.exists(config_path))
        
        # Verify content
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config['theme'], 'dracula')
        self.assertEqual(saved_config['timeout'], 120)
    
    def test_nested_config_get(self):
        """Test get method with nested keys."""
        # Create a config file with nested data
        test_config = {
            'commit_templates': {
                'default': {
                    'subject_template': 'test_subject',
                    'body_template': 'test_body'
                }
            }
        }
        self.create_config_file(test_config)
        
        # Load config
        config = Configuration(self.config_dir, self.config_file)
        
        # Get nested value
        value = config.get('commit_templates.default.subject_template')
        self.assertEqual(value, 'test_subject')
        
        # Get nonexistent nested value
        value = config.get('commit_templates.nonexistent.subject_template')
        self.assertIsNone(value)
        
        # Get nonexistent nested value with default
        value = config.get('commit_templates.nonexistent.subject_template', 'default_value')
        self.assertEqual(value, 'default_value')
    
    def test_nested_config_set(self):
        """Test set method with nested keys."""
        config = Configuration(self.config_dir, self.config_file)
        
        # Set nested value
        config.set('commit_templates.default.subject_template', 'new_template')
        
        # Verify value was set
        self.assertEqual(
            config.get('commit_templates.default.subject_template'), 
            'new_template'
        )
        
        # Set deeply nested value that doesn't exist yet
        config.set('new.deeply.nested.value', 42)
        self.assertEqual(config.get('new.deeply.nested.value'), 42)
    
    def test_reset_config(self):
        """Test reset method."""
        # Create a config file
        test_config = {
            'theme': 'cyberpunk',
            'timeout': 120
        }
        self.create_config_file(test_config)
        
        # Load config
        config = Configuration(self.config_dir, self.config_file)
        
        # Verify modified values
        self.assertEqual(config.get('theme'), 'cyberpunk')
        
        # Reset config
        config.reset()
        
        # Verify values are reset to defaults
        self.assertEqual(config.get('theme'), DEFAULT_CONFIG['theme'])
        self.assertEqual(config.get('timeout'), DEFAULT_CONFIG['timeout'])
    
    def test_commit_template_management(self):
        """Test commit template management methods."""
        config = Configuration(self.config_dir, self.config_file)
        
        # Test adding a template
        config.add_commit_template(
            "test_template",
            "test-subject",
            "test-body",
            "test-footer"
        )
        
        # Verify the template was added
        templates = config.get_commit_templates()
        self.assertIn("test_template", templates)
        self.assertEqual(templates["test_template"]["subject_template"], "test-subject")
        self.assertEqual(templates["test_template"]["body_template"], "test-body")
        self.assertEqual(templates["test_template"]["footer_template"], "test-footer")
        
        # Test setting active template
        self.assertTrue(config.set_active_template("test_template"))
        self.assertEqual(config.get("active_template"), "test_template")
        
        # Test getting the active template
        active_template = config.get_commit_template()
        self.assertEqual(active_template["subject_template"], "test-subject")
        
        # Test getting a specific template
        default_template = config.get_commit_template("default")
        self.assertEqual(
            default_template["subject_template"], 
            DEFAULT_CONFIG["commit_templates"]["default"]["subject_template"]
        )
        
        # Test removing a template
        self.assertTrue(config.remove_commit_template("test_template"))
        templates = config.get_commit_templates()
        self.assertNotIn("test_template", templates)
        
        # Verify active template reverted to default
        self.assertEqual(config.get("active_template"), "default")
        
        # Test that default template cannot be removed
        self.assertFalse(config.remove_commit_template("default"))
    
    def test_save_failure(self):
        """Test handling save failures."""
        config = Configuration(self.config_dir, self.config_file)
        
        # Use mock to simulate IO error during save
        with mock.patch('builtins.open', side_effect=IOError("Test exception")):
            # Try to save
            result = config.save()
            
            # Verify save failed
            self.assertFalse(result)
            self.mock_logger.error.assert_called_once()


if __name__ == "__main__":
    unittest.main() 