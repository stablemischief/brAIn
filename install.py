#!/usr/bin/env python3
"""
brAIn v2.0 - Simple Installation Script
Run this to set up your brAIn installation with the AI Configuration Wizard
"""

import sys
import os
import subprocess
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def main():
    """Main installation entry point."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  brAIn v2.0 Installation                      ║
║           AI-Powered Knowledge Management System              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print("🚀 Starting AI Configuration Wizard...\n")

    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9+ required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)

    try:
        # Import and run the wizard
        from backend.app.config.wizard import ConfigurationWizard

        wizard = ConfigurationWizard()

        print("Welcome to the brAIn Configuration Wizard!")
        print("=" * 50)
        print("\nThis wizard will help you:")
        print("  • Set up your database connection")
        print("  • Configure AI providers (OpenAI/Anthropic)")
        print("  • Set security parameters")
        print("  • Configure cost management")
        print("  • Test all connections")
        print("\nYou can choose from pre-configured templates or customize your setup.")
        print("\n" + "=" * 50)

        # Run the interactive wizard
        import asyncio
        asyncio.run(run_wizard(wizard))

    except ImportError as e:
        print(f"\n❌ Missing dependencies. Installing required packages...\n")

        # Auto-install basic requirements
        requirements_file = Path(__file__).parent / "backend" / "requirements.txt"
        if requirements_file.exists():
            print("📦 Installing dependencies from requirements.txt...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "-r", str(requirements_file)
            ])
            print("✅ Dependencies installed. Please run install.py again.")
        else:
            print("❌ Error: requirements.txt not found")
            print("   Please ensure you're in the brAIn project directory")
        sys.exit(1)

async def run_wizard(wizard):
    """Run the configuration wizard."""

    # Check for existing configuration
    config_path = Path("backend/config/settings.json")
    if config_path.exists():
        print("\n⚠️  Existing configuration detected.")
        response = input("Do you want to (R)econfigure, (U)pdate, or (C)ancel? [R/U/C]: ").upper()
        if response == 'C':
            print("Installation cancelled.")
            return
        elif response == 'U':
            print("\n📝 Loading existing configuration for update...")

    # Template selection
    print("\n📋 Configuration Templates Available:")
    print("1. Development - Local development with minimal setup")
    print("2. Production - Full production configuration")
    print("3. Cost-Optimized - Minimal token usage, budget controls")
    print("4. High-Performance - Optimized for speed")
    print("5. Custom - Configure everything manually")

    choice = input("\nSelect template [1-5]: ")

    template_map = {
        "1": "development",
        "2": "production",
        "3": "cost_optimized",
        "4": "high_performance",
        "5": "custom"
    }

    template = template_map.get(choice, "custom")

    if template != "custom":
        print(f"\n✅ Using {template.replace('_', ' ').title()} template")
        config = wizard.templates.get_template(template)

        # Quick customization for template
        print("\n🔧 Quick Setup (press Enter to use defaults):")

        # Database URL
        db_url = input(f"Database URL [{config.database.connection_url}]: ").strip()
        if db_url:
            config.database.connection_url = db_url

        # OpenAI Key (if needed)
        if template != "development":
            api_key = input("OpenAI API Key (optional): ").strip()
            if api_key:
                config.openai.api_key = api_key

    else:
        print("\n📝 Starting custom configuration...")
        # Run full interactive setup
        config = await wizard.run_interactive_setup()

    # Validate configuration
    print("\n🔍 Validating configuration...")
    validation_result = await wizard.validator.validate_config(config)

    if validation_result.is_valid:
        print("✅ Configuration valid!")

        # Test connections
        print("\n🧪 Testing connections...")
        test_results = await wizard.test_configuration(config)

        if test_results.get('all_passed', False):
            print("✅ All connection tests passed!")
        else:
            print("⚠️  Some tests failed:")
            for test, result in test_results.items():
                if test != 'all_passed' and not result.get('success'):
                    print(f"  • {test}: {result.get('error', 'Failed')}")

        # Save configuration
        save = input("\n💾 Save configuration? [Y/n]: ").upper() != 'N'
        if save:
            wizard.save_configuration(config)
            print("✅ Configuration saved to backend/config/settings.json")

            # Generate .env file
            generate_env(config)
            print("✅ Environment variables saved to backend/.env")

            print("\n" + "=" * 50)
            print("🎉 Installation Complete!")
            print("=" * 50)
            print("\nTo start brAIn, run:")
            print("  cd backend")
            print("  python -m uvicorn app.main:app --reload")
            print("\nOr use: python run.py")
    else:
        print("❌ Configuration validation failed:")
        for error in validation_result.errors:
            print(f"  • {error}")

def generate_env(config):
    """Generate .env file from configuration."""
    env_path = Path("backend/.env")

    env_lines = []

    # Database
    if config.database.connection_url != "postgresql://localhost/brain":
        env_lines.append(f"DATABASE_URL={config.database.connection_url}")

    # Supabase
    if config.supabase and config.supabase.url:
        env_lines.append(f"SUPABASE_URL={config.supabase.url}")
        env_lines.append(f"SUPABASE_SERVICE_KEY={config.supabase.service_key}")

    # OpenAI
    if config.openai and config.openai.api_key:
        env_lines.append(f"OPENAI_API_KEY={config.openai.api_key}")

    # Anthropic
    if config.anthropic and config.anthropic.api_key:
        env_lines.append(f"ANTHROPIC_API_KEY={config.anthropic.api_key}")

    # Security
    if config.security:
        env_lines.append(f"JWT_SECRET={config.security.jwt_secret}")

    with env_path.open('w') as f:
        f.write('\n'.join(env_lines))

if __name__ == "__main__":
    main()