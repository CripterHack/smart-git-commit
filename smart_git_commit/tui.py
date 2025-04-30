"""
Text-based User Interface for Smart Git Commit.

This module provides an interactive TUI for navigating through changes,
managing commit templates, and configuring Smart Git Commit settings.
"""

import os
import sys
import time
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Static, Input, Label
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import DataTable, RadioSet, RadioButton, Switch, Select
from textual.screen import Screen
from textual.binding import Binding
from textual import events
from textual.reactive import reactive

from .config import get_config, Configuration
from .smart_git_commit import (
    Colors, CommitType, GitChange, CommitGroup, 
    SecurityScanner, SmartGitCommitWorkflow
)


class ThemeSelect(Static):
    """Widget for selecting a theme."""
    
    def __init__(self, name: str = None):
        super().__init__(name=name)
        self.themes = [
            "standard", "cyberpunk", "dracula", "nord", "monokai"
        ]
        self.current_theme = get_config().get("theme", "standard")
    
    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Theme")
            with Horizontal():
                for theme in self.themes:
                    selected = theme == self.current_theme
                    yield Button(theme.capitalize(), id=f"theme_{theme}", 
                                 classes="selected" if selected else "")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id
        if button_id and button_id.startswith("theme_"):
            theme = button_id[6:]  # Remove "theme_" prefix
            
            # Update all button styles
            for button in self.query("Button"):
                if button.id == button_id:
                    button.add_class("selected")
                else:
                    button.remove_class("selected")
            
            # Update theme in configuration
            config = get_config()
            config.set("theme", theme)
            config.save()
            self.current_theme = theme
            
            # Apply theme to Colors class
            Colors.set_theme(theme)
            self.app.post_message(events.Message(sender=self, name="theme_changed", theme=theme))


class CommitTemplateScreen(Screen):
    """Screen for managing commit templates."""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("a", "add_template", "Add Template"),
        Binding("d", "delete_template", "Delete Template"),
        Binding("e", "edit_template", "Edit Template"),
        Binding("s", "set_active", "Set Active")
    ]
    
    def __init__(self, name: str = "commit_templates"):
        super().__init__(name=name)
        self.config = get_config()
        self.templates = self.config.get_commit_templates()
        self.active_template = self.config.get("active_template", "default")
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container():
            yield Label("Manage Commit Templates", classes="title")
            
            with Container(id="templates_container"):
                yield DataTable(id="templates_table")
            
            with Horizontal(id="template_buttons"):
                yield Button("Add", id="add_template", variant="primary")
                yield Button("Edit", id="edit_template", variant="default")
                yield Button("Delete", id="delete_template", variant="error")
                yield Button("Set Active", id="set_active", variant="success")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Set up the table when the screen is mounted."""
        table = self.query_one("#templates_table", DataTable)
        table.add_columns("Name", "Subject Template", "Active")
        
        active_template = self.config.get("active_template", "default")
        
        for name, template in self.templates.items():
            is_active = "✓" if name == active_template else ""
            table.add_row(name, template["subject_template"], is_active)
    
    def action_add_template(self) -> None:
        """Action to add a new template."""
        self.app.push_screen("template_edit", {"mode": "add"})
    
    def action_edit_template(self) -> None:
        """Action to edit an existing template."""
        table = self.query_one("#templates_table", DataTable)
        if table.cursor_row is not None:
            template_name = table.get_row_at(table.cursor_row)[0]
            self.app.push_screen("template_edit", {"mode": "edit", "name": template_name})
    
    def action_delete_template(self) -> None:
        """Action to delete a template."""
        table = self.query_one("#templates_table", DataTable)
        if table.cursor_row is not None:
            template_name = table.get_row_at(table.cursor_row)[0]
            if template_name != "default":
                if self.config.remove_commit_template(template_name):
                    self.config.save()
                    table.remove_row(table.cursor_row)
    
    def action_set_active(self) -> None:
        """Action to set the active template."""
        table = self.query_one("#templates_table", DataTable)
        if table.cursor_row is not None:
            template_name = table.get_row_at(table.cursor_row)[0]
            if self.config.set_active_template(template_name):
                self.config.save()
                
                # Update active indicators
                active_template = self.config.get("active_template", "default")
                for row in range(table.row_count):
                    name = table.get_row_at(row)[0]
                    is_active = "✓" if name == active_template else ""
                    table.update_cell(row, 2, is_active)


class TemplateEditScreen(Screen):
    """Screen for editing commit templates."""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]
    
    def __init__(self, name: str = "template_edit"):
        super().__init__(name=name)
        self.mode = "add"
        self.template_name = ""
        self.subject_template = ""
        self.body_template = ""
        self.footer_template = ""
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container():
            yield Label("Edit Commit Template", classes="title", id="edit_title")
            
            with Vertical(id="template_form"):
                yield Label("Template Name")
                yield Input(placeholder="template_name", id="template_name")
                
                yield Label("Subject Template")
                yield Input(placeholder="{type}({scope}): {description}", id="subject_template")
                
                yield Label("Body Template")
                yield Input(placeholder="{body}\n\nAffected files:\n{files}", id="body_template")
                
                yield Label("Footer Template")
                yield Input(placeholder="{issues}", id="footer_template")
            
            with Horizontal(id="form_buttons"):
                yield Button("Save", id="save_template", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")
        
        yield Footer()
    
    def on_screen_resume(self) -> None:
        """Called when the screen is resumed."""
        # Reset the form when the screen is shown again
        self.query_one("#template_name", Input).value = self.template_name
        self.query_one("#subject_template", Input).value = self.subject_template
        self.query_one("#body_template", Input).value = self.body_template
        self.query_one("#footer_template", Input).value = self.footer_template
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # Initialize form with data for edit mode
        if hasattr(self, "data") and self.data:
            self.mode = self.data.get("mode", "add")
            
            if self.mode == "edit":
                self.template_name = self.data.get("name", "")
                template = get_config().get_commit_template(self.template_name)
                self.subject_template = template.get("subject_template", "")
                self.body_template = template.get("body_template", "")
                self.footer_template = template.get("footer_template", "")
                
                # Update title and disable name field for edit mode
                self.query_one("#edit_title", Label).update(f"Edit Template: {self.template_name}")
                self.query_one("#template_name", Input).disabled = True
            else:
                self.query_one("#edit_title", Label).update("Add New Template")
                self.query_one("#template_name", Input).disabled = False
            
            # Set form values
            self.query_one("#template_name", Input).value = self.template_name
            self.query_one("#subject_template", Input).value = self.subject_template
            self.query_one("#body_template", Input).value = self.body_template
            self.query_one("#footer_template", Input).value = self.footer_template
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "save_template":
            self.action_save()
        elif event.button.id == "cancel":
            self.app.pop_screen()
    
    def action_save(self) -> None:
        """Action to save the template."""
        config = get_config()
        
        name = self.query_one("#template_name", Input).value
        subject = self.query_one("#subject_template", Input).value
        body = self.query_one("#body_template", Input).value
        footer = self.query_one("#footer_template", Input).value
        
        if not name or not subject:
            # Show error message
            return
        
        # Add or update the template
        config.add_commit_template(name, subject, body, footer)
        config.save()
        
        # Return to template list
        self.app.pop_screen()


class SettingsScreen(Screen):
    """Screen for managing application settings."""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("s", "save", "Save"),
    ]
    
    def __init__(self, name: str = "settings"):
        super().__init__(name=name)
        self.config = get_config()
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container():
            yield Label("Settings", classes="title")
            
            with Vertical(id="settings_form"):
                # Theme selection
                yield ThemeSelect(name="theme_select")
                
                # Security scanning
                with Horizontal():
                    yield Label("Enable Security Scanning")
                    yield Switch(value=self.config.get("security_scan", True), id="security_scan")
                
                # AI settings
                with Horizontal():
                    yield Label("Use AI for Commit Analysis")
                    yield Switch(value=self.config.get("use_ai", True), id="use_ai")
                
                # Ollama settings
                yield Label("Ollama Host")
                yield Input(value=self.config.get("ollama_host", "http://localhost:11434"), id="ollama_host")
                
                yield Label("Ollama Model (leave empty for auto-selection)")
                yield Input(value=self.config.get("ollama_model", "") or "", id="ollama_model")
                
                # Performance settings
                with Horizontal():
                    yield Label("Enable Parallel Processing")
                    yield Switch(value=self.config.get("parallel", True), id="parallel")
                
                yield Label("Timeout (seconds)")
                yield Input(value=str(self.config.get("timeout", 60)), id="timeout")
                
                # Git hooks
                with Horizontal():
                    yield Label("Skip Git Hooks")
                    yield Switch(value=self.config.get("skip_hooks", False), id="skip_hooks")
            
            with Horizontal(id="settings_buttons"):
                yield Button("Save", id="save_settings", variant="primary")
                yield Button("Reset to Defaults", id="reset_settings", variant="error")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "save_settings":
            self.action_save()
        elif event.button.id == "reset_settings":
            self.config.reset()
            self.config.save()
            self.app.push_screen("settings")  # Refresh the screen
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch toggle events."""
        if event.switch.id == "security_scan":
            self.config.set("security_scan", event.value)
        elif event.switch.id == "use_ai":
            self.config.set("use_ai", event.value)
        elif event.switch.id == "parallel":
            self.config.set("parallel", event.value)
        elif event.switch.id == "skip_hooks":
            self.config.set("skip_hooks", event.value)
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input change events."""
        if event.input.id == "ollama_host":
            self.config.set("ollama_host", event.value)
        elif event.input.id == "ollama_model":
            self.config.set("ollama_model", event.value or None)
        elif event.input.id == "timeout":
            try:
                timeout = int(event.value)
                self.config.set("timeout", timeout)
            except ValueError:
                pass
    
    def action_save(self) -> None:
        """Action to save the settings."""
        if self.config.save():
            self.app.notify("Settings saved successfully", title="Success")
        else:
            self.app.notify("Failed to save settings", title="Error")


class WelcomeScreen(Screen):
    """Welcome screen for first-time setup."""
    
    BINDINGS = [
        Binding("ctrl+n", "next", "Next"),
        Binding("ctrl+p", "prev", "Previous"),
    ]
    
    def __init__(self, name: str = "welcome"):
        super().__init__(name=name)
        self.current_step = 0
        self.total_steps = 4
        self.config = get_config()
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container(id="welcome_container"):
            yield Label("Welcome to Smart Git Commit!", classes="title")
            yield Label("Let's set up your configuration.", classes="subtitle")
            
            with Container(id="step_container"):
                # Step 1: Theme
                with Container(id="step_1", classes="step current"):
                    yield Label("Step 1: Choose a Theme", classes="step_title")
                    yield ThemeSelect(name="welcome_theme_select")
                
                # Step 2: AI Configuration
                with Container(id="step_2", classes="step"):
                    yield Label("Step 2: AI Configuration", classes="step_title")
                    
                    with Vertical():
                        with Horizontal():
                            yield Label("Use AI for Commit Analysis")
                            yield Switch(value=self.config.get("use_ai", True), id="welcome_use_ai")
                        
                        yield Label("Ollama Host")
                        yield Input(value=self.config.get("ollama_host", "http://localhost:11434"), 
                                   id="welcome_ollama_host")
                        
                        yield Label("Ollama Model (leave empty for auto-selection)")
                        yield Input(value=self.config.get("ollama_model", "") or "", 
                                   id="welcome_ollama_model")
                
                # Step 3: Security Settings
                with Container(id="step_3", classes="step"):
                    yield Label("Step 3: Security Settings", classes="step_title")
                    
                    with Vertical():
                        with Horizontal():
                            yield Label("Enable Security Scanning")
                            yield Switch(value=self.config.get("security_scan", True), 
                                        id="welcome_security_scan")
                        
                        yield Label("Security scanning helps prevent sensitive data from being committed.")
                
                # Step 4: Commit Templates
                with Container(id="step_4", classes="step"):
                    yield Label("Step 4: Commit Templates", classes="step_title")
                    
                    with Vertical():
                        yield Label("Choose a default commit template style:")
                        
                        with RadioSet(id="welcome_template"):
                            yield RadioButton("Default", value="default", 
                                           id="welcome_template_default")
                            yield RadioButton("Conventional", value="conventional", 
                                           id="welcome_template_conventional")
                            yield RadioButton("Detailed", value="detailed", 
                                           id="welcome_template_detailed")
                        
                        yield Label("You can customize templates later in Settings.")
            
            with Horizontal(id="navigation_buttons"):
                yield Button("Previous", id="prev_button", disabled=True)
                yield Button("Next", id="next_button")
                yield Button("Finish", id="finish_button", variant="primary", disabled=True)
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # Select the current template in the radio set
        active_template = self.config.get("active_template", "default")
        radio_button = self.query_one(f"#welcome_template_{active_template}", RadioButton)
        if radio_button:
            radio_button.value = True
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "prev_button":
            self.action_prev()
        elif event.button.id == "next_button":
            self.action_next()
        elif event.button.id == "finish_button":
            self._complete_setup()
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio set change events."""
        if event.radio_set.id == "welcome_template":
            self.config.set("active_template", event.value)
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch toggle events."""
        if event.switch.id == "welcome_security_scan":
            self.config.set("security_scan", event.value)
        elif event.switch.id == "welcome_use_ai":
            self.config.set("use_ai", event.value)
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input change events."""
        if event.input.id == "welcome_ollama_host":
            self.config.set("ollama_host", event.value)
        elif event.input.id == "welcome_ollama_model":
            self.config.set("ollama_model", event.value or None)
    
    def _update_step_visibility(self) -> None:
        """Update which step is visible based on current_step."""
        for i in range(1, self.total_steps + 1):
            step = self.query_one(f"#step_{i}", Container)
            if i == self.current_step + 1:
                step.add_class("current")
                step.remove_class("hidden")
            else:
                step.remove_class("current")
                step.add_class("hidden")
        
        # Update button states
        prev_button = self.query_one("#prev_button", Button)
        next_button = self.query_one("#next_button", Button)
        finish_button = self.query_one("#finish_button", Button)
        
        prev_button.disabled = self.current_step == 0
        
        if self.current_step == self.total_steps - 1:
            next_button.disabled = True
            finish_button.disabled = False
        else:
            next_button.disabled = False
            finish_button.disabled = True
    
    def action_next(self) -> None:
        """Move to the next step."""
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
            self._update_step_visibility()
    
    def action_prev(self) -> None:
        """Move to the previous step."""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_step_visibility()
    
    def _complete_setup(self) -> None:
        """Complete the setup and save settings."""
        # Mark welcome as completed
        self.config.set("welcome_completed", True)
        
        # Save all settings
        if self.config.save():
            self.app.notify("Setup completed successfully!", title="Welcome")
            
            # Switch to the main screen
            self.app.pop_screen()
            self.app.push_screen("main")
        else:
            self.app.notify("Failed to save settings", title="Error")


class ChangesTable(Static):
    """Widget for displaying Git changes."""
    
    def __init__(self, name: str = None, changes: List[GitChange] = None):
        super().__init__(name=name)
        self.changes = changes or []
    
    def compose(self) -> ComposeResult:
        yield DataTable(id="changes_table")
    
    def on_mount(self) -> None:
        """Set up the table when the widget is mounted."""
        table = self.query_one("#changes_table", DataTable)
        table.add_columns("Status", "Filename", "Component", "Sensitive")
        
        for change in self.changes:
            sensitive = "⚠️" if change.is_sensitive else ""
            table.add_row(change.status, change.filename, change.component, sensitive)


class CommitGroupsTable(Static):
    """Widget for displaying commit groups."""
    
    def __init__(self, name: str = None, groups: List[CommitGroup] = None):
        super().__init__(name=name)
        self.groups = groups or []
    
    def compose(self) -> ComposeResult:
        yield DataTable(id="groups_table")
    
    def on_mount(self) -> None:
        """Set up the table when the widget is mounted."""
        table = self.query_one("#groups_table", DataTable)
        table.add_columns("Type", "Name", "Files", "Component")
        
        for group in self.groups:
            table.add_row(
                group.commit_type.value, 
                group.name, 
                str(len(group.changes)),
                "/".join(set(change.component for change in group.changes))[:20]
            )


class MainScreen(Screen):
    """Main screen of the application."""
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "commit", "Commit"),
        Binding("s", "settings", "Settings"),
        Binding("t", "templates", "Templates"),
    ]
    
    def __init__(self, name: str = "main"):
        super().__init__(name=name)
        self.workflow = None
        self.changes = []
        self.groups = []
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container():
            yield Label("Smart Git Commit", classes="title")
            
            with Horizontal(id="main_actions"):
                yield Button("Refresh", id="refresh_button", variant="default")
                yield Button("Settings", id="settings_button", variant="default")
                yield Button("Templates", id="templates_button", variant="default")
                yield Button("Commit", id="commit_button", variant="primary")
            
            with Vertical(id="changes_container"):
                yield Label("Staged Changes", classes="section_title")
                yield ChangesTable(name="changes_table")
            
            with Vertical(id="groups_container"):
                yield Label("Commit Groups", classes="section_title")
                yield CommitGroupsTable(name="groups_table")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Set up the screen when it's mounted."""
        # Initialize workflow
        self.refresh_workflow()
    
    def refresh_workflow(self) -> None:
        """Refresh the Git workflow and update UI."""
        config = get_config()
        self.workflow = SmartGitCommitWorkflow(
            repo_path=".",
            ollama_host=config.get("ollama_host", "http://localhost:11434"),
            ollama_model=config.get("ollama_model"),
            use_ai=config.get("use_ai", True),
            timeout=config.get("timeout", 60),
            skip_hooks=config.get("skip_hooks", False),
            parallel=config.get("parallel", True),
            security_scan=config.get("security_scan", True)
        )
        
        # Load and analyze changes
        try:
            self.workflow.load_changes()
            if self.workflow.changes:
                self.workflow.analyze_and_group_changes()
                
                # Update UI
                self.changes = self.workflow.changes
                self.groups = self.workflow.commit_groups
                
                # Update tables
                self._update_changes_table()
                self._update_groups_table()
            else:
                self.app.notify("No changes to commit", title="Git Status")
                self.changes = []
                self.groups = []
                self._update_changes_table()
                self._update_groups_table()
        except Exception as e:
            self.app.notify(f"Error: {str(e)}", title="Git Error")
    
    def _update_changes_table(self) -> None:
        """Update the changes table with current data."""
        table = self.query_one("#changes_table > #changes_table", DataTable)
        table.clear()
        table.add_columns("Status", "Filename", "Component", "Sensitive")
        
        for change in self.changes:
            sensitive = "⚠️" if change.is_sensitive else ""
            table.add_row(change.status, change.filename, change.component, sensitive)
    
    def _update_groups_table(self) -> None:
        """Update the groups table with current data."""
        table = self.query_one("#groups_table > #groups_table", DataTable)
        table.clear()
        table.add_columns("Type", "Name", "Files", "Component")
        
        for group in self.groups:
            table.add_row(
                group.commit_type.value, 
                group.name, 
                str(len(group.changes)),
                "/".join(set(change.component for change in group.changes))[:20]
            )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "refresh_button":
            self.action_refresh()
        elif event.button.id == "settings_button":
            self.action_settings()
        elif event.button.id == "templates_button":
            self.action_templates()
        elif event.button.id == "commit_button":
            self.action_commit()
    
    def action_refresh(self) -> None:
        """Action to refresh Git status."""
        self.refresh_workflow()
    
    def action_settings(self) -> None:
        """Action to open settings screen."""
        self.app.push_screen("settings")
    
    def action_templates(self) -> None:
        """Action to open templates screen."""
        self.app.push_screen("commit_templates")
    
    def action_commit(self) -> None:
        """Action to start commit process."""
        if not self.groups:
            self.app.notify("No changes to commit", title="Git Status")
            return
        
        self.app.push_screen("commit")


class CommitScreen(Screen):
    """Screen for executing commits."""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("c", "commit", "Commit"),
    ]
    
    def __init__(self, name: str = "commit"):
        super().__init__(name=name)
        self.workflow = None
        self.selected_group_index = 0
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container():
            yield Label("Commit Changes", classes="title")
            
            with Horizontal(id="commit_selection"):
                with Vertical(id="groups_list", classes="sidebar"):
                    yield Label("Commit Groups", classes="section_title")
                    yield DataTable(id="commit_groups_table")
                
                with Vertical(id="commit_details"):
                    yield Label("Commit Details", classes="section_title")
                    
                    yield Label("Files in Group", classes="subsection_title")
                    yield DataTable(id="group_files_table")
                    
                    yield Label("Commit Message", classes="subsection_title")
                    yield TextArea(id="commit_message", language="markdown")
            
            with Horizontal(id="commit_buttons"):
                yield Button("Back", id="back_button", variant="default")
                yield Button("Edit Message", id="edit_button", variant="default")
                yield Button("Commit", id="do_commit_button", variant="primary")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Set up the screen when it's mounted."""
        # Get workflow from main screen
        main_screen = self.app.query_one("MainScreen")
        self.workflow = main_screen.workflow
        self.groups = main_screen.groups
        
        # Set up commit groups table
        table = self.query_one("#commit_groups_table", DataTable)
        table.add_columns("Type", "Name", "Files")
        
        for idx, group in enumerate(self.groups):
            table.add_row(
                group.commit_type.value, 
                group.name, 
                str(len(group.changes))
            )
        
        # Select the first group
        if self.groups:
            self._select_group(0)
    
    def _select_group(self, index: int) -> None:
        """Select a commit group and update UI."""
        if 0 <= index < len(self.groups):
            self.selected_group_index = index
            group = self.groups[index]
            
            # Update files table
            files_table = self.query_one("#group_files_table", DataTable)
            files_table.clear()
            files_table.add_columns("Status", "Filename", "Component")
            
            for change in group.changes:
                files_table.add_row(
                    change.status, 
                    change.filename, 
                    change.component
                )
            
            # Update commit message
            message = self.workflow._generate_ai_commit_message(group)
            self.query_one("#commit_message", TextArea).text = message
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the groups table."""
        if event.data_table.id == "commit_groups_table":
            self._select_group(event.row_key.row_index)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "back_button":
            self.app.pop_screen()
        elif event.button.id == "edit_button":
            self._edit_commit_message()
        elif event.button.id == "do_commit_button":
            self.action_commit()
    
    def _edit_commit_message(self) -> None:
        """Open the commit message for editing."""
        # We'll use the text area directly since it's already editable
        pass
    
    def action_commit(self) -> None:
        """Execute the commit."""
        if self.selected_group_index < 0 or self.selected_group_index >= len(self.groups):
            return
        
        group = self.groups[self.selected_group_index]
        message = self.query_one("#commit_message", TextArea).text
        
        success = self.workflow._commit_changes(group, message)
        
        if success:
            self.app.notify("Changes committed successfully", title="Git Commit")
            
            # Remove the committed group and refresh
            self.groups.pop(self.selected_group_index)
            self.selected_group_index = max(0, min(self.selected_group_index, len(self.groups) - 1))
            
            # Update groups table
            table = self.query_one("#commit_groups_table", DataTable)
            table.clear()
            table.add_columns("Type", "Name", "Files")
            
            for idx, group in enumerate(self.groups):
                table.add_row(
                    group.commit_type.value, 
                    group.name, 
                    str(len(group.changes))
                )
            
            # Select next group or pop screen if no more groups
            if self.groups:
                self._select_group(self.selected_group_index)
            else:
                self.app.pop_screen()
                self.app.query_one("MainScreen").refresh_workflow()
        else:
            self.app.notify("Failed to commit changes", title="Git Error")


# Create importable classes to resolve undefined references
class TextArea(Input):
    """Text area widget with multi-line editing."""
    
    def __init__(self, value: str = "", *, id: str = None, language: str = None, classes: str = None):
        super().__init__(value=value, id=id, classes=classes)
        self.language = language
        self.text = value


class SmartGitCommitApp(App):
    """Main Textual application for Smart Git Commit."""
    
    CSS = """
    Screen {
        background: $surface;
        color: $text;
    }
    
    .title {
        text-align: center;
        text-style: bold;
        width: 100%;
        margin: 1 0;
    }
    
    .subtitle {
        text-align: center;
        width: 100%;
        margin: 1 0;
    }
    
    .section_title {
        text-style: bold underline;
        margin: 1 0;
    }
    
    .subsection_title {
        text-style: bold;
        margin: 1 0;
    }
    
    Button.selected {
        background: $accent;
    }
    
    #main_actions {
        width: 100%;
        height: 3;
        margin: 1 0;
    }
    
    #navigation_buttons {
        width: 100%;
        height: 3;
        margin: 1 0;
    }
    
    #settings_buttons {
        width: 100%;
        height: 3;
        margin: 1 0;
    }
    
    #template_buttons {
        width: 100%;
        height: 3;
        margin: 1 0;
    }
    
    #form_buttons {
        width: 100%;
        height: 3;
        margin: 1 0;
    }
    
    #commit_buttons {
        width: 100%;
        height: 3;
        margin: 1 0;
    }
    
    .step {
        display: none;
    }
    
    .step.current {
        display: block;
    }
    
    #commit_selection {
        width: 100%;
        height: 1fr;
    }
    
    #groups_list {
        width: 30%;
        height: 100%;
    }
    
    #commit_details {
        width: 70%;
        height: 100%;
    }
    
    #commit_message {
        height: 10;
        margin: 1 0;
    }
    """
    
    SCREENS = {
        "main": MainScreen,
        "settings": SettingsScreen,
        "commit_templates": CommitTemplateScreen,
        "template_edit": TemplateEditScreen,
        "commit": CommitScreen,
        "welcome": WelcomeScreen,
    }
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("f1", "help", "Help")
    ]
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Check if welcome flow has been completed
        config = get_config()
        if not config.get("welcome_completed", False):
            self.push_screen("welcome")
        else:
            self.push_screen("main")
    
    def action_help(self) -> None:
        """Show help information."""
        self.notify(
            "Smart Git Commit TUI\n\n"
            "Key Bindings:\n"
            "  Ctrl+Q: Quit\n"
            "  F1: Help\n"
            "  Escape: Back (in sub-screens)\n"
            "  R: Refresh\n"
            "  C: Commit\n"
            "  S: Settings\n"
            "  T: Templates",
            title="Help"
        )


def run_tui() -> None:
    """Run the TUI application."""
    app = SmartGitCommitApp()
    app.run()


if __name__ == "__main__":
    run_tui() 