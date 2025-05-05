"""
Tests for the Textual UI components of Smart Git Commit.

This module tests the TUI components, screens, and user interactions.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from textual.app import App
from textual.widgets import Button, DataTable, Input, Label, Switch
from textual.screen import Screen
from textual.testing import AppTest

from smart_git_commit.tui import (
    SmartGitCommitApp, MainScreen, CommitScreen, SettingsScreen,
    CommitTemplateScreen, TemplateEditScreen, WelcomeScreen,
    ThemeSelect, ChangesTable, CommitGroupsTable
)
from smart_git_commit.smart_git_commit import GitChange, CommitGroup, CommitType

# Helper function to create mock GitChange objects for testing
def create_mock_changes():
    """Create a list of mock GitChange objects for testing."""
    return [
        GitChange("A", "file1.py", "component1", False),
        GitChange("M", "file2.py", "component2", True, "Contains API key"),
        GitChange("D", "file3.py", "component1", False),
    ]

# Helper function to create mock CommitGroup objects for testing
def create_mock_groups():
    """Create a list of mock CommitGroup objects for testing."""
    changes = create_mock_changes()
    return [
        CommitGroup(CommitType.FEAT, "Add new feature", [changes[0]]),
        CommitGroup(CommitType.FIX, "Fix bug", [changes[1]]),
        CommitGroup(CommitType.REFACTOR, "Remove old code", [changes[2]]),
    ]

class TestThemeSelect:
    """Tests for the ThemeSelect widget."""

    @pytest.mark.asyncio
    async def test_theme_selection(self):
        """Test theme selection functionality."""
        # Create a simple app with the ThemeSelect widget
        class TestApp(App):
            def compose(self):
                yield ThemeSelect()
        
        async with AppTest(TestApp) as pilot:
            # Check that the widget is rendered
            await pilot.pause()
            theme_widget = pilot.app.query_one(ThemeSelect)
            assert theme_widget is not None
            
            # Make sure all theme buttons are present
            theme_buttons = pilot.app.query("Button")
            assert len(theme_buttons) == 5  # Standard, Cyberpunk, Dracula, Nord, Monokai
            
            # Test button press
            with patch('smart_git_commit.tui.get_config') as mock_config:
                mock_config_instance = MagicMock()
                mock_config.return_value = mock_config_instance
                
                # Click on a theme button
                await pilot.click("#theme_cyberpunk")
                
                # Verify config was updated
                mock_config_instance.set.assert_called_with("theme", "cyberpunk")
                mock_config_instance.save.assert_called_once()

class TestChangesTable:
    """Tests for the ChangesTable widget."""

    @pytest.mark.asyncio
    async def test_changes_display(self):
        """Test that changes are correctly displayed in the table."""
        # Create a simple app with the ChangesTable widget
        class TestApp(App):
            def compose(self):
                yield ChangesTable(changes=create_mock_changes())
        
        async with AppTest(TestApp) as pilot:
            await pilot.pause()
            
            # Get the data table and check rows
            table = pilot.app.query_one(DataTable)
            assert table.row_count == 3
            
            # Check column headers
            assert table.columns[0].label == "Status"
            assert table.columns[1].label == "Filename"
            assert table.columns[2].label == "Component"
            assert table.columns[3].label == "Sensitive"
            
            # Check row data
            first_row = table.get_row_at(0)
            assert first_row[0] == "A"  # Status
            assert first_row[1] == "file1.py"  # Filename
            assert first_row[2] == "component1"  # Component
            assert first_row[3] == ""  # Not sensitive
            
            # Check sensitive file row
            second_row = table.get_row_at(1)
            assert second_row[0] == "M"
            assert second_row[1] == "file2.py"
            assert second_row[3] == "⚠️"  # Should be marked as sensitive

class TestCommitGroupsTable:
    """Tests for the CommitGroupsTable widget."""

    @pytest.mark.asyncio
    async def test_groups_display(self):
        """Test that commit groups are correctly displayed in the table."""
        # Create a simple app with the CommitGroupsTable widget
        class TestApp(App):
            def compose(self):
                yield CommitGroupsTable(groups=create_mock_groups())
        
        async with AppTest(TestApp) as pilot:
            await pilot.pause()
            
            # Get the data table and check rows
            table = pilot.app.query_one(DataTable)
            assert table.row_count == 3
            
            # Check column headers
            assert table.columns[0].label == "Type"
            assert table.columns[1].label == "Name"
            assert table.columns[2].label == "Files"
            assert table.columns[3].label == "Component"
            
            # Check row data for the first group
            first_row = table.get_row_at(0)
            assert first_row[0] == "feat"  # Type
            assert first_row[1] == "Add new feature"  # Name
            assert first_row[2] == "1"  # File count

class TestWelcomeScreen:
    """Tests for the WelcomeScreen."""
    
    @pytest.mark.asyncio
    async def test_welcome_navigation(self):
        """Test navigation through welcome wizard steps."""
        class TestApp(App):
            SCREENS = {"welcome": WelcomeScreen}
            
            def on_mount(self):
                self.push_screen("welcome")
        
        async with AppTest(TestApp) as pilot:
            await pilot.pause()
            
            # Verify we're on step 1
            step1 = pilot.app.query_one("#step_1")
            assert "current" in step1.classes
            
            # Navigate to next step
            await pilot.press("ctrl+n")
            step2 = pilot.app.query_one("#step_2")
            assert "current" in step2.classes
            
            # Test UI elements in step 2
            use_ai_switch = pilot.app.query_one("#welcome_use_ai", Switch)
            assert use_ai_switch is not None
            
            # Continue navigation
            await pilot.press("ctrl+n")
            step3 = pilot.app.query_one("#step_3")
            assert "current" in step3.classes
            
            await pilot.press("ctrl+n")
            step4 = pilot.app.query_one("#step_4")
            assert "current" in step4.classes
            
            # Test radio buttons in step 4
            radio_buttons = pilot.app.query("RadioButton")
            assert len(radio_buttons) == 3

            # Test back navigation
            await pilot.press("ctrl+p")
            assert "current" in step3.classes

class TestSettingsScreen:
    """Tests for the SettingsScreen."""

    @pytest.mark.asyncio
    async def test_settings_ui_elements(self):
        """Test that all settings UI elements are present and functioning."""
        with patch('smart_git_commit.tui.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config.return_value = mock_config_instance
            mock_config_instance.get.return_value = True
            
            class TestApp(App):
                SCREENS = {"settings": SettingsScreen}
                
                def on_mount(self):
                    self.push_screen("settings")
            
            async with AppTest(TestApp) as pilot:
                await pilot.pause()
                
                # Check theme selection widget
                theme_select = pilot.app.query_one(ThemeSelect)
                assert theme_select is not None
                
                # Check switches
                security_scan_switch = pilot.app.query_one("#security_scan", Switch)
                assert security_scan_switch is not None
                assert security_scan_switch.value is True
                
                use_ai_switch = pilot.app.query_one("#use_ai", Switch)
                assert use_ai_switch is not None
                
                # Test switch toggle
                await pilot.click("#security_scan")
                mock_config_instance.set.assert_called_with("security_scan", False)
                
                # Check input fields
                ollama_host_input = pilot.app.query_one("#ollama_host", Input)
                assert ollama_host_input is not None
                
                # Test save button
                await pilot.click("#save_settings")
                mock_config_instance.save.assert_called()

class TestMainScreen:
    """Tests for the MainScreen."""
    
    @pytest.mark.asyncio
    @patch('smart_git_commit.tui.SmartGitCommitWorkflow')
    async def test_main_screen_refresh(self, mock_workflow_class):
        """Test refreshing Git changes on the main screen."""
        # Setup mocks
        mock_workflow = MagicMock()
        mock_workflow_class.return_value = mock_workflow
        mock_workflow.changes = create_mock_changes()
        mock_workflow.commit_groups = create_mock_groups()
        
        class TestApp(App):
            SCREENS = {"main": MainScreen}
            
            def on_mount(self):
                self.push_screen("main")
        
        async with AppTest(TestApp) as pilot:
            await pilot.pause()
            
            # Check if tables are populated
            changes_table = pilot.app.query_one("#changes_table > #changes_table", DataTable)
            assert changes_table.row_count == 3
            
            groups_table = pilot.app.query_one("#groups_table > #groups_table", DataTable)
            assert groups_table.row_count == 3
            
            # Test refresh action
            mock_workflow.load_changes.reset_mock()
            await pilot.press("r")  # Refresh shortcut
            mock_workflow.load_changes.assert_called_once()

class TestCommitScreen:
    """Tests for the CommitScreen."""
    
    @pytest.mark.asyncio
    async def test_commit_screen_navigation(self):
        """Test navigation within the commit screen."""
        # Create a mock main screen with workflow and groups
        main_screen = MagicMock(spec=MainScreen)
        main_screen.workflow = MagicMock()
        main_screen.workflow._generate_ai_commit_message.return_value = "feat: Test commit message"
        main_screen.groups = create_mock_groups()
        
        class TestApp(App):
            SCREENS = {"commit": CommitScreen}
            
            def on_mount(self):
                # Add mock main screen to app for querying
                self.mock_main = main_screen
                self.query_one = MagicMock(return_value=main_screen)
                self.push_screen("commit")
        
        with patch('smart_git_commit.tui.CommitScreen.on_mount', 
                   side_effect=lambda self: setattr(self, 'groups', create_mock_groups())):
            async with AppTest(TestApp) as pilot:
                await pilot.pause()
                
                # Check commit group selection
                commit_groups_table = pilot.app.query_one("#commit_groups_table", DataTable)
                assert commit_groups_table.row_count == 3
                
                # Test back navigation
                back_button = pilot.app.query_one("#back_button", Button)
                assert back_button is not None

class TestTemplateScreens:
    """Tests for template management screens."""
    
    @pytest.mark.asyncio
    @patch('smart_git_commit.tui.get_config')
    async def test_template_list_screen(self, mock_config):
        """Test the commit template listing screen."""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        
        # Mock templates data
        templates = {
            "default": {
                "subject_template": "{type}({scope}): {description}",
                "body_template": "{body}",
                "footer_template": "{issues}"
            },
            "detailed": {
                "subject_template": "{type}({scope}): {description}",
                "body_template": "{body}\n\nAffected files:\n{files}",
                "footer_template": "{issues}"
            }
        }
        mock_config_instance.get_commit_templates.return_value = templates
        mock_config_instance.get.return_value = "default"
        
        class TestApp(App):
            SCREENS = {"commit_templates": CommitTemplateScreen}
            
            def on_mount(self):
                self.push_screen("commit_templates")
        
        async with AppTest(TestApp) as pilot:
            await pilot.pause()
            
            # Check table content
            table = pilot.app.query_one("#templates_table", DataTable)
            assert table.row_count == 2
            
            # Test add template button
            add_button = pilot.app.query_one("#add_template", Button)
            assert add_button is not None

class TestSmartGitCommitApp:
    """Tests for the main application class."""
    
    @pytest.mark.asyncio
    @patch('smart_git_commit.tui.get_config')
    async def test_app_initialization(self, mock_config):
        """Test app initialization and screen setup."""
        # Set welcome_completed to True to show main screen
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.get.return_value = True
        
        async with AppTest(SmartGitCommitApp, press_keys=False) as pilot:
            await pilot.pause()
            
            # Check if app initialized correctly
            app = pilot.app
            assert isinstance(app, SmartGitCommitApp)
            
            # Should start with main screen since welcome_completed is True
            current_screen = app.screen
            assert isinstance(current_screen, MainScreen)
            
            # Test help action
            with patch.object(app, 'notify') as mock_notify:
                await pilot.press("f1")  # Help shortcut
                mock_notify.assert_called_once()

    @pytest.mark.asyncio
    @patch('smart_git_commit.tui.get_config')
    async def test_welcome_flow(self, mock_config):
        """Test welcome flow for first-time setup."""
        # Set welcome_completed to False to trigger welcome flow
        mock_config_instance = MagicMock()
            {"label": "Option 1", "value": "opt1"},
            {"label": "Option 2", "value": "opt2"}
        ]
        
        radio_group = self.tui.create_radio_group(
            label="Select one:",
            options=options,
            selected="opt1"
        )
        
        # Render radio group
        self.tui.render_radio_group(radio_group)
        
        # Assertions
        mock_print.assert_called()
        
        # Verify label and options are in the output
        calls = mock_print.call_args_list
        output = ''.join(str(call[0][0]) for call in calls if call[0])
        self.assertIn("Select one:", output)
        self.assertIn("Option 1", output)
        self.assertIn("Option 2", output)
    
    @patch('builtins.print')
    def test_render_progress_bar_component(self, mock_print):
        """Test rendering a progress bar component."""
        # Create progress bar
        progress_bar = self.tui.create_progress_bar(
            label="Loading...",
            value=50,
            max_value=100
        )
        
        # Render progress bar
        self.tui.render_progress_bar(progress_bar)
        
        # Assertions
        mock_print.assert_called()
        
        # Verify label is in the output
        calls = mock_print.call_args_list
        output = ''.join(str(call[0][0]) for call in calls if call[0])
        self.assertIn("Loading...", output)
    
    @patch('builtins.print')
    def test_render_separator(self, mock_print):
        """Test rendering a separator."""
        # Render separator
        self.tui.render_separator()
        
        # Assertions
        mock_print.assert_called()
    
    @patch('builtins.print')
    def test_render_heading(self, mock_print):
        """Test rendering a heading."""
        # Render heading
        self.tui.render_heading("Test Heading")
        
        # Assertions
        mock_print.assert_called()
        
        # Verify heading is in the output
        calls = mock_print.call_args_list
        output = ''.join(str(call[0][0]) for call in calls if call[0])
        self.assertIn("Test Heading", output)
    
    @patch('builtins.print')
    def test_render_subheading(self, mock_print):
        """Test rendering a subheading."""
        # Render subheading
        self.tui.render_subheading("Test Subheading")
        
        # Assertions
        mock_print.assert_called()
        
        # Verify subheading is in the output
        calls = mock_print.call_args_list
        output = ''.join(str(call[0][0]) for call in calls if call[0])
        self.assertIn("Test Subheading", output)


if __name__ == '__main__':
    unittest.main() 