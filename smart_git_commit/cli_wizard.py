"""
Module for the Command-Line Interface (CLI) first-time setup wizard.
"""

import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from . import get_version  # Get version from __init__.py
from .smart_git_commit import OllamaClient
from .colors import Colors

if TYPE_CHECKING:
    from .config import Configuration

logger = logging.getLogger(__name__)


def run_cli_welcome_wizard(config: 'Configuration', use_color: bool) -> bool:
    """
    Runs the first-time setup wizard in the CLI.

    Args:
        config: The configuration object.
        use_color: Whether to use colored output.
        
    Returns:
        True if the wizard completed successfully, False otherwise.
    """
    console = Console()

    console.print(Panel(
        Text.from_markup(f"👋 Welcome to [bold cyan]Smart Git Commit v{get_version()}[/]! Let's get you set up."),
        title="First-Time Setup",
        border_style="blue" if use_color else ""
    ))

    # 1. AI Provider Choice
    console.print("\n[bold]1. Choose your AI Provider:[/]")
    console.print("   - [bold cyan]Ollama[/]: Use local models (Recommended, requires Ollama setup).")
    console.print("   - [bold green]OpenAI[/]: Use OpenAI's API (Requires API Key).")
    console.print("   - [bold yellow]None[/]: Disable AI features (Rule-based mode).")

    ai_provider = Prompt.ask(
        "Select AI Provider",
        choices=["ollama", "openai", "none"],
        default=config.get("ai_provider", "ollama") # Use config default
    ).lower()

    config.set("ai_provider", ai_provider)
    config.set("use_ai", ai_provider != "none")

    # 2. Provider Specific Setup
    if ai_provider == "ollama":
        console.print("\n[bold]2. Ollama Setup:[/]")
        ollama_host = Prompt.ask("Enter Ollama host URL", default=config.get("ollama_host", "http://localhost:11434"))
        config.set("ollama_host", ollama_host)
        # Reuse OllamaClient logic to find models
        try:
            console.print("   Checking for available Ollama models...")
            wizard_timeout = 15
            temp_ollama_client = OllamaClient(host=ollama_host, timeout=wizard_timeout)
            available_models = temp_ollama_client.get_available_models()

            if not available_models:
                console.print(f"[yellow]⚠️ No Ollama models found or could not connect to {ollama_host} within {wizard_timeout}s.[/]")
                console.print("[yellow]   Please ensure Ollama is running and you have pulled a model (e.g., `ollama pull llama3`).[/]")
                console.print("[yellow]   You can change the model later using the CLI or TUI.[/]")
                config.set("ollama_model", None)
            else:
                console.print("   Available Ollama models:")
                for model in available_models:
                    console.print(f"     - {model}")
                current_model = config.get("ollama_model")
                model_choices = sorted(list(set(available_models + ([current_model] if current_model else []) + [""])))

                ollama_model = Prompt.ask(
                    "Enter the Ollama model to use (leave blank to be prompted each time)",
                    choices=model_choices,
                    default=current_model if current_model in model_choices else ""
                )
                config.set("ollama_model", ollama_model if ollama_model else None)
        except Exception as e:
            logger.error(f"Failed to connect to Ollama during wizard: {e}", exc_info=True)
            console.print(f"[red]❌ Error connecting to Ollama at {ollama_host}. Please check if it's running.[/]")
            console.print("[yellow]   Proceeding without selecting a model. You can configure it later.[/]")
            config.set("ollama_model", None)

    elif ai_provider == "openai":
        console.print("\n[bold]2. OpenAI Setup:[/]")
        existing_key = config.get("openai_api_key")
        prompt_text = "Enter your OpenAI API Key"
        if existing_key:
            prompt_text += " (leave blank to keep existing)"

        openai_key_input = Prompt.ask(prompt_text, password=True, default="")
        if openai_key_input:
             config.set("openai_api_key", openai_key_input)
        elif not existing_key:
             console.print("[yellow]⚠️ OpenAI API Key is required to use the OpenAI provider.[/]")

        default_openai_model = config.get("openai_model", "gpt-3.5-turbo")
        openai_model = Prompt.ask(f"Enter OpenAI model name", default=default_openai_model)
        config.set("openai_model", openai_model)

    # Mark wizard as completed
    config.set("welcome_completed", True)
    try:
        config.save()
        # Use getattr to safely access config_file_path with a fallback
        config_path = getattr(config, "config_file_path", "configuration")
        console.print(f"\n[green]✅ Setup complete! Configuration saved to {config_path}[/]")
    except Exception as e:
        logger.error(f"Failed to save configuration during wizard: {e}", exc_info=True)
        console.print(f"[red]❌ Error saving configuration: {e}[/]")

    console.print("\n[bold cyan]Tip:[/bold cyan] You can change these settings later using `smart-git-commit --tui` or by editing the config file.")
    width = 60
    try:
        width = console.width
    except Exception:
        pass
    console.print("-" * width)
    Prompt.ask("[bold]Press Enter to continue...[/]", default="")
    return True 