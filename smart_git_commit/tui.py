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
from textual.widgets import Header, Footer, Button, Static, Input, Label, Select, ListView
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import DataTable, RadioSet, RadioButton, Switch
from textual.screen import Screen
from textual.binding import Binding
from textual import events
from textual.reactive import reactive
from textual.message import Message
from textual.widgets import TextArea
from textual.validation import Integer, ValidationResult

from .config import get_config, Configuration
from .smart_git_commit import (
    Colors, CommitType, GitChange, CommitGroup, 
    SecurityScanner, SmartGitCommitWorkflow, OllamaClient, OpenAIClient
)


class ThemeChanged(Message):
    """Message sent when the theme is changed."""
    def __init__(self, theme: str) -> None:
        super().__init__()
        self.theme = theme


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
            try:
                config.save()
            except Exception as e:
                self.app.notify(f"Error saving theme config: {e}", title="Config Error", severity="error")

            self.current_theme = theme
            
            # Apply theme to Colors class
            Colors.set_theme(theme)
            self.app.post_message(ThemeChanged(theme))


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
        
        # Ensure active_template is a string
        if not isinstance(active_template, str):
            self.app.notify(f"Warning: active_template is {type(active_template).__name__}, not string. Using default.", title="Config Warning")
            active_template = "default"
            self.config.set("active_template", active_template)
        
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
                try:
                    if self.config.remove_commit_template(template_name):
                        self.config.save()
                        table.remove_row(table.cursor_row)
                    else:
                        self.app.notify("Failed to remove template.", title="Error", severity="error")
                except Exception as e:
                    self.app.notify(f"Error deleting template: {e}", title="Error", severity="error")
    
    def action_set_active(self) -> None:
        """Action to set the active template."""
        table = self.query_one("#templates_table", DataTable)
        if table.cursor_row is not None:
            template_name = table.get_row_at(table.cursor_row)[0]
            
            # Ensure we're working with a string
            if not isinstance(template_name, str):
                self.app.notify(f"Warning: Template name is {type(template_name).__name__}, not string.", title="Config Warning")
                return
                
            try:
                if self.config.set_active_template(template_name):
                    self.config.save()
                    
                    # Update active indicators
                    active_template = self.config.get("active_template", "default")
                    
                    # Double-check active_template is a string
                    if not isinstance(active_template, str):
                        self.app.notify(f"Warning: active_template is {type(active_template).__name__}, not string. Using default.", title="Config Warning")
                        active_template = "default"
                        self.config.set("active_template", active_template)
                    
                    for row in range(table.row_count):
                        name = table.get_row_at(row)[0]
                        is_active = "✓" if name == active_template else ""
                        table.update_cell(row, 2, is_active)
                else:
                    self.app.notify("Failed to set active template.", title="Config Error", severity="error")
            except Exception as e:
                 self.app.notify(f"Error setting active template: {e}", title="Config Error", severity="error")


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
        try:
            config.save()
            # Return to template list
            self.app.pop_screen()
        except Exception as e:
            self.app.notify(f"Error saving template: {e}", title="Config Error", severity="error")


class SettingsScreen(Screen):
    """Screen for managing application settings."""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("s", "save", "Save"),
    ]
    
    def __init__(self, name: str = "settings"):
        super().__init__(name=name)
        self.config = get_config()
        self.ollama_client = None
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container():
            yield Label("Settings", classes="title")
            
            with Vertical(id="settings_form"):
                # AI Provider Settings
                yield Label("AI Provider", classes="section_title")
                yield Select(
                    [(value, value.capitalize()) for value in ["ollama", "openai", "none"]],
                    id="ai_provider",
                    value=self.config.get("ai_provider", "ollama")
                )
                
                # Ollama Settings
                with Container(id="ollama_settings"):
                    yield Label("Ollama Settings", classes="subsection_title")
                    yield Label("Ollama Host")
                    yield Input(
                        value=self.config.get("ollama_host", "http://localhost:11434"),
                        placeholder="http://localhost:11434",
                        id="ollama_host"
                    )
                    yield Label("Ollama Model")
                    yield Select([], id="ollama_model")
                    yield Label("Status")
                    yield Label("Waiting to connect...", id="ollama_status")
                
                # OpenAI Settings
                with Container(id="openai_settings"):
                    yield Label("OpenAI Settings", classes="subsection_title")
                    yield Label("OpenAI API Key")
                    yield Input(
                        value=self.config.get("openai_api_key", ""),
                        placeholder="sk-...",
                        id="openai_api_key",
                        password=True
                    )
                    yield Label("OpenAI Model")
                    yield Input(
                        value=self.config.get("openai_model", "gpt-3.5-turbo"),
                        placeholder="gpt-3.5-turbo",
                        id="openai_model"
                    )
                
                # General Settings
                yield Label("General Settings", classes="section_title")
                yield Label("Security Scan")
                yield Switch(
                    value=self.config.get("security_scan", True),
                    id="security_scan"
                )
                yield Label("Theme")
                yield ThemeSelect(id="theme_select")
            
            with Horizontal(id="settings_buttons"):
                yield Button("Save", id="save_settings", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # Set initial visibility for provider-specific settings
        self._update_provider_settings()
        # Fetch available Ollama models
        asyncio.create_task(self._fetch_ollama_models())

    async def _fetch_ollama_models(self) -> None:
        """Fetch available Ollama models asynchronously."""
        try:
            select = self.query_one("#ollama_model", Select)
            select.set_options([("", "Loading models...")])
            
            # Get Ollama host from input
            host_input = self.query_one("#ollama_host", Input)
            host = host_input.value
            
            # Create Ollama client
            client = OllamaClient(host=host)
            models = await asyncio.to_thread(client.get_available_models)
            
            if models:
                # Create options from models
                options = [(model, model) for model in models]
                select.set_options(options)
                
                # Set current value if it exists in the options
                current_model = self.config.get("ollama_model")
                if current_model and current_model in models:
                    select.value = current_model
                elif models:
                    select.value = models[0]
                    
                # Update status if label exists
                status_label = self.query_one("#ollama_status", Label, default=None)
                if status_label:
                    status_label.update("Connected successfully, models loaded")
            else:
                select.set_options([("", "No models found")])
                status_label = self.query_one("#ollama_status", Label, default=None)
                if status_label:
                    status_label.update("No models found. Please install at least one model with 'ollama pull modelname'")
        except Exception as e:
            self.app.notify(f"Error fetching Ollama models: {str(e)}", title="Error", severity="error")
            select = self.query_one("#ollama_model", Select)
            select.set_options([("", "Error loading models")])
            status_label = self.query_one("#ollama_status", Label, default=None)
            if status_label:
                status_label.update(f"Connection error: {str(e)}")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changed events."""
        if event.select.id == "ai_provider":
            self._update_provider_settings()

    def _update_provider_settings(self) -> None:
        """Update visibility of provider-specific settings."""
        provider = self.query_one("#ai_provider", Select).value
        
        # Update Ollama settings visibility
        ollama_container = self.query_one("#ollama_settings", Container)
        ollama_container.display = provider == "ollama"
        
        # Update OpenAI settings visibility
        openai_container = self.query_one("#openai_settings", Container)
        openai_container.display = provider == "openai"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "save_settings":
            self.action_save()
        elif event.button.id == "cancel":
            self.app.pop_screen()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changed events."""
        if event.switch.id == "security_scan":
            self.config.set("security_scan", event.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changed events."""
        if event.input.id in ["ollama_host", "openai_api_key", "openai_model"]:
            self.config.set(event.input.id, event.value)

    def action_save(self) -> None:
        """Save settings and return to the previous screen."""
        try:
            # Save AI provider
            provider_select = self.query_one("#ai_provider", Select)
            self.config.set("ai_provider", provider_select.value)
            
            # Save Ollama settings
            if provider_select.value == "ollama":
                ollama_host = self.query_one("#ollama_host", Input).value
                self.config.set("ollama_host", ollama_host)
                
                ollama_model_select = self.query_one("#ollama_model", Select)
                if ollama_model_select.value:
                    self.config.set("ollama_model", ollama_model_select.value)
            
            # Save OpenAI settings
            elif provider_select.value == "openai":
                openai_api_key = self.query_one("#openai_api_key", Input).value
                openai_model = self.query_one("#openai_model", Input).value
                
                if openai_api_key:
                    self.config.set("openai_api_key", openai_api_key)
                
                if openai_model:
                    self.config.set("openai_model", openai_model)
            
            # Save general settings
            security_scan = self.query_one("#security_scan", Switch).value
            self.config.set("security_scan", security_scan)
            
            # Save config
            self.config.save()
            
            # Notify success
            self.app.notify("Settings saved successfully", title="Success")
            
            # Return to previous screen
            self.app.pop_screen()
        except Exception as e:
            self.app.notify(f"Error saving settings: {str(e)}", title="Error", severity="error")


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
        self.ollama_client = None
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container(id="welcome_container"):
            yield Label("Welcome to Smart Git Commit", classes="title")
            yield Label("Let's set up your configuration", classes="subtitle")
            
            # Step 1: AI Provider
            with Container(id="step_1", classes="step current"):
                yield Label("Step 1: Choose AI Provider", classes="section_title")
                yield Label("Select how you want to power the AI analysis:")
                
                with RadioSet(id="ai_provider"):
                    yield RadioButton("Ollama (Local models)", id="provider_ollama", value=True)
                    yield Label("- Fast, privacy-focused, runs locally")
                    yield Label("- Requires Ollama to be installed: https://ollama.ai")
                    yield Label("- Limited by local hardware resources")
                    
                    yield RadioButton("OpenAI (API)", id="provider_openai")
                    yield Label("- High quality analysis and suggestions")
                    yield Label("- Requires an OpenAI API key")
                    yield Label("- Costs based on API usage")
                    
                    yield RadioButton("None (Disable AI)", id="provider_none")
                    yield Label("- Use rule-based analysis only")
                    yield Label("- No AI-powered features")
                    yield Label("- Fastest option, no external dependencies")
            
            # Step 2a: Ollama Configuration
            with Container(id="step_2a", classes="step"):
                yield Label("Step 2: Ollama Configuration", classes="section_title")
                yield Label("Configure Ollama settings:")
                
                yield Label("Ollama Host")
                yield Input(
                    value="http://localhost:11434",
                    placeholder="http://localhost:11434",
                    id="ollama_host"
                )
                
                yield Label("Ollama Model")
                yield Select([], id="ollama_model")
                
                yield Label("Status")
                yield Label("Checking connection...", id="ollama_status")
            
            # Step 2b: OpenAI Configuration
            with Container(id="step_2b", classes="step"):
                yield Label("Step 2: OpenAI Configuration", classes="section_title")
                yield Label("Configure OpenAI settings:")
                
                yield Label("OpenAI API Key")
                yield Input(
                    placeholder="sk-...",
                    id="openai_api_key",
                    password=True
                )
                
                yield Label("OpenAI Model")
                yield Input(
                    value="gpt-3.5-turbo",
                    placeholder="gpt-3.5-turbo",
                    id="openai_model"
                )
            
            # Step 3: Additional Settings
            with Container(id="step_3", classes="step"):
                yield Label("Step 3: Additional Settings", classes="section_title")
                
                yield Label("Security Scan")
                yield Switch(value=True, id="security_scan")
                yield Label("Detect and exclude sensitive data from commits")
                
                yield Label("Theme")
                yield ThemeSelect(id="theme_select")
            
            # Navigation buttons
            with Horizontal(id="navigation_buttons"):
                yield Button("< Previous", id="prev", disabled=True)
                yield Button("Next >", id="next")
                yield Button("Finish", id="finish", disabled=True)
        
        yield Footer()

    def on_mount(self) -> None:
        """Set up the screen when mounted."""
        # Initialize the config
        if not hasattr(self, "config"):
            self.config = get_config()
        
        self.current_step = 1
        self.total_steps = 3
        
        # Load existing settings if available
        self._load_current_settings()
        
        # Start Ollama model fetch if Ollama is selected
        if self.query_one("#provider_ollama", RadioButton).value:
            asyncio.create_task(self._fetch_ollama_models())
        
        # Update step visibility
        self._update_step_visibility()

    async def _fetch_ollama_models(self) -> None:
        """Fetch available Ollama models asynchronously."""
        try:
            select = self.query_one("#ollama_model", Select)
            select.set_options([("", "Loading models...")])
            
            # Get Ollama host from input
            host_input = self.query_one("#ollama_host", Input)
            host = host_input.value
            
            # Create Ollama client
            client = OllamaClient(host=host)
            models = await asyncio.to_thread(client.get_available_models)
            
            if models:
                # Create options from models
                options = [(model, model) for model in models]
                select.set_options(options)
                
                # Set current value if it exists in the options
                current_model = self.config.get("ollama_model")
                if current_model and current_model in models:
                    select.value = current_model
                elif models:
                    select.value = models[0]
                    
                # Update status if label exists
                status_label = self.query_one("#ollama_status", Label, default=None)
                if status_label:
                    status_label.update("Connected successfully, models loaded")
            else:
                select.set_options([("", "No models found")])
                status_label = self.query_one("#ollama_status", Label, default=None)
                if status_label:
                    status_label.update("No models found. Please install at least one model with 'ollama pull modelname'")
        except Exception as e:
            self.app.notify(f"Error fetching Ollama models: {str(e)}", title="Error", severity="error")
            select = self.query_one("#ollama_model", Select)
            select.set_options([("", "Error loading models")])
            status_label = self.query_one("#ollama_status", Label, default=None)
            if status_label:
                status_label.update(f"Connection error: {str(e)}")

    def _load_current_settings(self) -> None:
        """Load current settings from config if available."""
        try:
            # Load AI provider
            provider = self.config.get("ai_provider", "ollama")
            if provider == "ollama":
                self.query_one("#provider_ollama", RadioButton).value = True
            elif provider == "openai":
                self.query_one("#provider_openai", RadioButton).value = True
            elif provider == "none":
                self.query_one("#provider_none", RadioButton).value = True
            
            # Load Ollama settings
            ollama_host = self.config.get("ollama_host")
            if ollama_host:
                self.query_one("#ollama_host", Input).value = ollama_host
            
            # Load OpenAI settings
            openai_api_key = self.config.get("openai_api_key")
            if openai_api_key:
                self.query_one("#openai_api_key", Input).value = openai_api_key
                
            openai_model = self.config.get("openai_model")
            if openai_model:
                self.query_one("#openai_model", Input).value = openai_model
            
            # Load security scan setting
            security_scan = self.config.get("security_scan", True)
            self.query_one("#security_scan", Switch).value = security_scan
        except Exception as e:
            self.app.notify(f"Error loading settings: {str(e)}", title="Warning", severity="warning")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changed events."""
        if event.select.id == "ollama_model":
            self.config.set("ollama_model", event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "next":
            self.action_next()
        elif event.button.id == "prev":
            self.action_prev()
        elif event.button.id == "finish":
            self._complete_setup()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio set changed events."""
        if event.radio_set.id == "ai_provider":
            selected_button = event.pressed
            
            # Update config based on selected provider
            if selected_button.id == "provider_ollama":
                self.config.set("ai_provider", "ollama")
                # Fetch Ollama models
                asyncio.create_task(self._fetch_ollama_models())
            elif selected_button.id == "provider_openai":
                self.config.set("ai_provider", "openai")
            elif selected_button.id == "provider_none":
                self.config.set("ai_provider", "none")
            
            # Update which step 2 to show (a for Ollama, b for OpenAI)
            self._update_step_visibility()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changed events."""
        if event.switch.id == "security_scan":
            self.config.set("security_scan", event.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changed events."""
        if event.input.id in ["ollama_host", "openai_api_key", "openai_model"]:
            self.config.set(event.input.id, event.value)

    def _update_step_visibility(self) -> None:
        """Update which steps are visible based on current state."""
        # Update current step container
        for step in range(1, self.total_steps + 1):
            step_container = self.query_one(f"#step_{step}", Container)
            if step == self.current_step:
                step_container.add_class("current")
                step_container.remove_class("disabled")
            else:
                step_container.remove_class("current")
                step_container.add_class("disabled")
        
        # Special handling for step 2 (AI provider specific)
        if self.current_step == 2:
            provider = "none"
            if self.query_one("#provider_ollama", RadioButton).value:
                provider = "ollama"
            elif self.query_one("#provider_openai", RadioButton).value:
                provider = "openai"
            
            # Show appropriate step 2 container
            step_2a = self.query_one("#step_2a", Container)
            step_2b = self.query_one("#step_2b", Container)
            
            if provider == "ollama":
                step_2a.add_class("current")
                step_2a.remove_class("disabled")
                step_2b.remove_class("current")
                step_2b.add_class("disabled")
            elif provider == "openai":
                step_2a.remove_class("current")
                step_2a.add_class("disabled")
                step_2b.add_class("current")
                step_2b.remove_class("disabled")
            else:
                # Skip to step 3 if "none" is selected
                self.current_step = 3
                self._update_step_visibility()
        
        # Update navigation buttons
        prev_button = self.query_one("#prev", Button)
        next_button = self.query_one("#next", Button)
        finish_button = self.query_one("#finish", Button)
        
        prev_button.disabled = self.current_step == 1
        next_button.disabled = self.current_step == self.total_steps
        finish_button.disabled = self.current_step != self.total_steps

    def action_next(self) -> None:
        """Go to the next step."""
        if self.current_step < self.total_steps:
            self.current_step += 1
            self._update_step_visibility()

    def action_prev(self) -> None:
        """Go to the previous step."""
        if self.current_step > 1:
            self.current_step -= 1
            self._update_step_visibility()

    def _complete_setup(self) -> None:
        """Complete the setup process."""
        try:
            # Save the configuration
            self.config.set("welcome_completed", True)
            self.config.save()
            
            # Mark the welcome as completed for this session
            app = self.app
            app.welcome_completed = True
            
            # Show success message
            self.app.notify("Setup completed successfully", title="Success")
            
            # Go to the main screen
            self.app.pop_screen()
            self.app.push_screen("main")
        except Exception as e:
            self.app.notify(f"Error saving configuration: {str(e)}", title="Error", severity="error")


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
        self.commit_groups = []
    
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
        """Refresh the workflow with the current configuration."""
        try:
            config = get_config()
            
            # Get repository path
            repo_path = self.app.repo_path
            
            # Get AI provider and settings
            ai_provider = config.get("ai_provider", "ollama")
            use_ai = ai_provider != "none"
            
            # Get Ollama settings
            ollama_host = config.get("ollama_host", "http://localhost:11434")
            ollama_model = config.get("ollama_model")
            
            # Get OpenAI settings
            openai_api_key = config.get("openai_api_key")
            openai_model = config.get("openai_model", "gpt-3.5-turbo")
            
            # Get other settings
            security_scan = config.get("security_scan", True)
            skip_hooks = config.get("skip_hooks", False)
            parallel = config.get("parallel", True)
            timeout = config.get("timeout", 60)
            
            # Initialize workflow with all settings
            self.workflow = SmartGitCommitWorkflow(
                repo_path=repo_path,
                use_ai=use_ai,
                ai_provider=ai_provider,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                security_scan=security_scan,
                skip_hooks=skip_hooks,
                parallel=parallel,
                timeout=timeout
            )
            
            # Load changes
            with self.app.batch_update():
                self.app.notify("Loading changes...", title="Status")
                self.workflow.load_changes()
                
                # Update changes table
                self._update_changes_table()
                
                # Analyze and group changes
                if self.workflow.changes:
                    self.app.notify("Analyzing changes...", title="Status")
                    self.commit_groups = self.workflow.analyze_and_group_changes()
                    self._update_groups_table()
                else:
                    self.commit_groups = []
                    self.query_one("#groups_table > #groups_table", DataTable).clear()
                    self.app.notify("No changes to commit", title="Status")
        except Exception as e:
            self.app.notify(f"Error refreshing workflow: {str(e)}", title="Error", severity="error")
    
    def _update_changes_table(self) -> None:
        """Update the changes table with current data."""
        table = self.query_one("#changes_table > #changes_table", DataTable)
        table.clear()
        table.add_columns("Status", "Filename", "Component", "Sensitive")
        
        for change in self.workflow.changes if self.workflow else []:
            sensitive = "⚠️" if change.is_sensitive else ""
            table.add_row(change.status, change.filename, change.component, sensitive)
    
    def _update_groups_table(self) -> None:
        """Update the groups table with current data."""
        table = self.query_one("#groups_table > #groups_table", DataTable)
        table.clear()
        table.add_columns("Type", "Name", "Files", "Component")
        
        for group in self.commit_groups:
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
        if not self.commit_groups:
            self.app.notify("No changes to commit", title="Git Status")
            return
        
        self.app.push_screen("commit")


class CommitScreen(Screen):
    """Screen for committing changes."""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("c", "commit", "Commit"),
    ]
    
    def __init__(self, name: str = "commit"):
        super().__init__(name=name)
        self.workflow = None
        self.groups: List[CommitGroup] = []
        self.selected_group_index = -1
    
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
        try:
            main_screen = self.app.query_one("MainScreen")
            self.workflow = main_screen.workflow
            self.groups = main_screen.commit_groups
        except Exception as e:
            self.app.notify(f"Error connecting to main screen: {str(e)}",
                           title="Error", severity="error")
            self.app.pop_screen()
            return
            
        # Set up commit groups table
        table = self.query_one("#commit_groups_table", DataTable)
        table.clear()
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
            
            # Generate commit message
            message = self._generate_commit_message(group)
            self.query_one("#commit_message", TextArea).text = message
    
    def _generate_commit_message(self, group: CommitGroup) -> str:
        """Generate a commit message for the selected group."""
        try:
            # Use workflow's method if available
            if hasattr(self.workflow, "_generate_commit_message"):
                return self.workflow._generate_commit_message(group)
            
            # Fallback to basic message format
            message = f"{group.commit_type.value}"
            if group.name and group.name != "root":
                message += f"({group.name})"
            
            message += f": Changes in {group.name}\n\n"
            message += "Affected files:\n"
            
            for change in group.changes:
                status_symbol = {
                    'M': 'M', 'A': '+', 'D': '-', 'R': 'R',
                    '??': '?'
                }.get(change.status[:1], ' ')
                message += f"- {status_symbol} {change.filename}\n"
                
            return message
        except Exception as e:
            self.app.notify(f"Error generating commit message: {str(e)}", title="Error", severity="error")
            return f"commit({group.name}): Changes in {len(group.changes)} files"
    
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
        
        success = False
        try:
            # Try to use workflow's commit method if available
            if hasattr(self.workflow, "_commit_changes"):
                success = self.workflow._commit_changes(group, message)
            elif hasattr(self.workflow, "execute_commits"):
                # Generic commit method
                success = self._perform_commit(group, message)
            else:
                self.app.notify("Cannot perform commit: workflow missing required methods", 
                               title="Commit Error", severity="error")
                return
        except Exception as e:
             self.app.notify(f"Error during commit: {str(e)}", title="Git Commit Error", severity="error")
             success = False
        
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
            self.app.notify("Failed to commit changes", title="Git Error", severity="error")
            
    def _perform_commit(self, group: CommitGroup, message: str) -> bool:
        """Fallback method to perform a commit using workflow's basic methods."""
        try:
            # Stage files
            if not hasattr(self.workflow, "_stage_files"):
                self.app.notify("Cannot stage files: workflow missing required methods", 
                               title="Commit Error", severity="error")
                return False
                
            staging_success = self.workflow._stage_files(group.changes)
            if not staging_success:
                return False
                
            # Execute git commit
            if not hasattr(self.workflow, "_execute_commit"):
                self.app.notify("Cannot execute commit: workflow missing required methods", 
                               title="Commit Error", severity="error")
                return False
                
            return self.workflow._execute_commit(message)
        except Exception as e:
            self.app.notify(f"Error in commit process: {str(e)}", 
                           title="Git Error", severity="error")
            return False


class SensitiveFileConfirmScreen(Screen):
    """Modal screen for confirming inclusion of sensitive files."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel")
    ]
    
    def __init__(self, name: str = "sensitive_confirm"):
        super().__init__(name=name)
        self.change = None
        self.on_confirm = None
        self.on_cancel = None
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container():
            yield Label("⚠️ SECURITY WARNING ⚠️", classes="title warning")
            
            with Vertical(id="warning_details"):
                yield Label("The following file has been flagged as potentially containing sensitive data:")
                yield Label("", id="sensitive_filename", classes="sensitive_filename")
                
                yield Label("Reason:", classes="subsection_title")
                yield Label("", id="sensitive_reason", classes="sensitive_reason")
                
                yield Label("", id="warning_message", classes="warning")
                yield Label("Including sensitive data in commits can lead to security breaches.", classes="warning")
                yield Label("Are you ABSOLUTELY SURE you want to include this file?", classes="warning")
            
            with Horizontal(id="confirmation_buttons"):
                yield Button("No, Exclude This File (Recommended)", id="exclude_button", variant="primary")
                yield Button("Yes, Include Despite Warning", id="include_button", variant="error")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Update details when the screen is mounted."""
        if self.change:
            self.query_one("#sensitive_filename", Label).update(self.change.filename)
            self.query_one("#sensitive_reason", Label).update(self.change.sensitive_reason)
            
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "include_button":
            # Double confirmation with more explicit warning
            self.query_one("#warning_message", Label).update("⚠️ FINAL WARNING: This is a high-security risk! ⚠️")
            event.button.label = "Yes, I Understand the Risk (type CONFIRM)"
            event.button.id = "final_confirm_button"
        elif event.button.id == "final_confirm_button":
            if self.on_confirm:
                self.on_confirm()
            self.app.pop_screen()
        elif event.button.id == "exclude_button":
            if self.on_cancel:
                self.on_cancel()
            self.app.pop_screen()
    
    def action_cancel(self) -> None:
        """Handle escape key (cancel)."""
        if self.on_cancel:
            self.on_cancel()
        self.app.pop_screen()


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
    
    .warning {
        color: $error;
        text-style: bold;
    }
    
    .sensitive_filename {
        background: $surface-lighten-1;
        color: $text;
        text-align: center;
        padding: 1;
        margin: 1 0;
    }
    
    .sensitive_reason {
        color: $text-muted;
        margin: 1 0;
    }
    
    #warning_details {
        margin: 1 0;
        padding: 1;
        border: heavy $error;
    }
    
    #confirmation_buttons {
        width: 100%;
        height: 3;
        margin: 2 0;
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
    
    Input.-invalid {
        border: thick $error;
    }
    """
    
    SCREENS = {
        "main": MainScreen,
        "settings": SettingsScreen,
        "commit_templates": CommitTemplateScreen,
        "template_edit": TemplateEditScreen,
        "commit": CommitScreen,
        "welcome": WelcomeScreen,
        "sensitive_confirm": SensitiveFileConfirmScreen,
    }
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("f1", "help", "Help")
    ]
    
    repo_path: str = "."
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Restore original logic: Check if welcome flow has been completed
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

    def action_quit(self) -> None:
        """Clean up and exit the application."""
        # Save configuration before exit
        try:
            config = get_config()
            config.save()
        except Exception as e:
            self.notify(f"Error saving configuration: {str(e)}", title="Config Error")
            
        self.exit()
        
    def on_exception(self, exception: Exception) -> None:
        """Handle any unhandled exceptions globally."""
        # Log the exception or display it. Using notify for visibility.
        import traceback
        exc_info = traceback.format_exc()
        self.notify(f"Unhandled Exception:\n{exc_info}", title="APP CRASH", timeout=20)
        # Optionally, you might want to exit or log to a file here.
        self.exit(1)


def run_tui(repo_path: str = ".") -> int:
    """
    Run the Text-based User Interface (TUI).
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Try to import Textual
        from textual.app import App
        from textual.widgets import Header, Footer, Static, Button, ListView, Label
        from textual.containers import Container, Horizontal, Vertical
        from textual.screen import Screen
        from textual.binding import Binding
    except ImportError:
        logger.error("Failed to import textual. Please install it: pip install textual")
        print("Error: The TUI requires the 'textual' package.")
        print("Please install it using: pip install textual")
        return 1
    
    # Import our workflow
    from .smart_git_commit import SmartGitCommitWorkflow, GitChange, CommitGroup
    
    class CommitScreen(Screen):
        """Screen for committing changes."""
        
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("r", "refresh", "Refresh"),
            Binding("c", "commit", "Commit"),
        ]
        
        def __init__(self, workflow: SmartGitCommitWorkflow):
            """Initialize the commit screen."""
            super().__init__()
            self.workflow = workflow
            self.commit_groups = []
            self.selected_group_index = 0
        
        def on_mount(self) -> None:
            """Handle the screen mount event."""
            self.refresh_data()
        
        def refresh_data(self) -> None:
            """Refresh the data from git."""
            # Load changes
            self.workflow.load_changes()
            
            # Analyze changes
            self.commit_groups = self.workflow.analyze_and_group_changes()
            
            # Update UI
            self.update_commit_groups()
        
        def update_commit_groups(self) -> None:
            """Update the commit groups display."""
            groups_list = self.query_one("#groups-list", ListView)
            groups_list.clear()
            
            for i, group in enumerate(self.commit_groups):
                groups_list.append(f"{group.commit_type.value}({group.name}): {len(group.changes)} files")
            
            if self.commit_groups:
                groups_list.index = self.selected_group_index
                self.update_group_details(self.selected_group_index)
            else:
                self.query_one("#group-details", Static).update("No changes to commit")
        
        def update_group_details(self, index: int) -> None:
            """Update the details for the selected commit group."""
            if not self.commit_groups:
                return
                
            group = self.commit_groups[index]
            
            # Build details text
            details = f"# {group.commit_type.value}({group.name})\n\n"
            
            details += "## Files:\n"
            for change in group.changes:
                status_symbol = {
                    'M': '📝', 'A': '➕', 'D': '❌', 'R': '🔄',
                    '??': '❓'
                }.get(change.status[:1], '•')
                details += f"- {status_symbol} {change.filename}\n"
            
            # Show commit message preview
            details += "\n## Commit Message Preview:\n"
            commit_message = self.workflow._generate_commit_message(group)
            details += f"```\n{commit_message}\n```\n"
            
            self.query_one("#group-details", Static).update(details)
        
        def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
            """Handle list view highlighted event."""
            self.selected_group_index = event.index
            self.update_group_details(event.index)
        
        def action_refresh(self) -> None:
            """Refresh the data."""
            self.refresh_data()
        
        def action_commit(self) -> None:
            """Commit the selected group."""
            if not self.commit_groups:
                self.notify("No changes to commit")
                return
                
            group = self.commit_groups[self.selected_group_index]
            
            # Generate commit message
            commit_message = self.workflow._generate_commit_message(group)
            
            # Commit changes
            success = self.workflow._commit_changes(group, commit_message)
            
            if success:
                self.notify(f"Committed: {group.commit_type.value}({group.name})")
                # Refresh after commit
                self.refresh_data()
            else:
                self.notify("Failed to commit changes")
    
    class SmartGitCommitApp(App):
        """Smart Git Commit TUI Application."""
        
        TITLE = "Smart Git Commit"
        CSS_PATH = "tui.css"
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("r", "refresh", "Refresh"),
        ]
        
        def __init__(self, repo_path: str):
            """Initialize the application."""
            super().__init__()
            self.repo_path = repo_path
            
            # Create workflow
            self.workflow = SmartGitCommitWorkflow(repo_path=repo_path)
        
        def compose(self):
            """Compose the app layout."""
            yield Header()
            
            with Container():
                with Horizontal():
                    with Vertical(id="sidebar"):
                        yield Label("Commit Groups")
                        yield ListView(id="groups-list")
                        yield Button("Commit Selected", id="commit-btn")
                    
                    with Vertical(id="main-content"):
                        yield Static("No changes selected", id="group-details")
            
            yield Footer()
        
        def on_mount(self):
            """Handle the app mount event."""
            # Set up CSS if file doesn't exist
            if not os.path.exists(self.CSS_PATH):
                with open(self.CSS_PATH, "w") as f:
                    f.write("""
                    #sidebar {
                        width: 30%;
                        background: $panel;
                    }
                    
                    #main-content {
                        width: 70%;
                        background: $surface;
                    }
                    
                    #groups-list {
                        height: 100%;
                    }
                    
                    Button {
                        width: 100%;
                    }
                    """)
            
            # Show the commit screen
            self.push_screen(CommitScreen(self.workflow))
            
        def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button pressed event."""
            if event.button.id == "commit-btn":
                self.screen.action_commit()
    
    # Run the app
    try:
        app = SmartGitCommitApp(repo_path)
        app.run()
        return 0
    except Exception as e:
        logger.exception(f"Error running TUI: {str(e)}")
        print(f"Error running TUI: {str(e)}")
        return 1


if __name__ == "__main__":
    run_tui() 