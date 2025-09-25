#!/usr/bin/env python3
"""
brAIn v2.0 - Simple Installation Script
Run this to set up your brAIn installation with the AI Configuration Wizard
"""

import sys
import os
import subprocess
from pathlib import Path
import platform

def check_virtual_env():
    """Check if we're in a virtual environment."""
    return sys.prefix != sys.base_prefix

def create_virtual_env():
    """Create and guide user to activate virtual environment."""
    venv_path = Path.cwd() / "venv"

    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created!")

    # Provide activation instructions based on OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"

    print("\n" + "="*60)
    print("⚠️  VIRTUAL ENVIRONMENT REQUIRED")
    print("="*60)
    print("\nModern Python requires a virtual environment for package installation.")
    print("\nPlease run these commands:")
    print(f"\n  1. {activate_cmd}")
    print("  2. python3 install.py")
    print("\n" + "="*60)

    return False

def main():
    """Main installation entry point."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  brAIn v2.0 Installation                      ║
║           AI-Powered Knowledge Management System              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9+ required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)

    # Check for virtual environment on modern Python
    if sys.version_info >= (3, 11) and not check_virtual_env():
        print("🔍 Checking environment...")
        if not create_virtual_env():
            sys.exit(0)

    print("🚀 Starting AI Configuration Wizard...\n")

    # Add backend to path for imports
    sys.path.insert(0, str(Path(__file__).parent / "backend"))

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
        print(f"\n❌ Missing dependencies: {e}\n")

        # Check if we're in a venv now
        if not check_virtual_env():
            print("⚠️  Not in a virtual environment!")
            create_virtual_env()
            sys.exit(1)

        # Auto-install basic requirements
        requirements_file = Path(__file__).parent / "backend" / "requirements.txt"
        if requirements_file.exists():
            print("📦 Installing dependencies from requirements.txt...")
            print("This may take a few minutes on first install...\n")

            result = subprocess.run([
                sys.executable, "-m", "pip", "install",
                "-r", str(requirements_file)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Dependencies installed successfully!")
                print("\n🔄 Restarting installer with dependencies...")
                print("-" * 50 + "\n")
                # Restart the script with dependencies installed
                os.execv(sys.executable, ['python3', __file__] + sys.argv[1:])
            else:
                print("❌ Failed to install dependencies:")
                print(result.stderr)
                sys.exit(1)
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
        "1": "Development Environment",
        "2": "Production Secure",
        "3": "Cost Optimized",
        "4": "High Performance",
        "5": "custom"
    }

    template = template_map.get(choice, "custom")

    if template != "custom":
        print(f"\n✅ Using {template.replace('_', ' ').title()} template")
        config = wizard.templates.get_template_by_name(template)

        # Quick customization for template
        print("\n🔧 Quick Setup (press Enter to use defaults):")

        # Build default database URL from template config
        db_config = config.config.get("database", {})
        default_db_url = f"postgresql://{db_config.get('username', 'user')}:{db_config.get('password', 'password')}@{db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config.get('database', 'brain')}"

        # Database URL
        db_url = input(f"Database URL [{default_db_url}]: ").strip()
        if db_url:
            # Parse and update database config from provided URL
            # For now, store the full URL - this would need URL parsing for full implementation
            config.config["database"]["connection_url"] = db_url

        # OpenAI Key (if needed)
        if template != "Development Environment":
            current_key = config.config.get("openai", {}).get("api_key", "")
            api_key = input(f"OpenAI API Key [{current_key[:10]}...]: ").strip()
            if api_key:
                if "openai" not in config.config:
                    config.config["openai"] = {}
                config.config["openai"]["api_key"] = api_key

    else:
        print("\n📝 Starting custom configuration...")
        # Run full interactive setup
        config = await wizard.run_interactive_setup()

    # For template-based setup, show configuration summary
    print("\n🔍 Configuration Summary:")
    print(f"✅ Template: {config.name}")
    print(f"✅ Environment: {config.config.get('environment', 'development')}")

    if 'database' in config.config:
        db_config = config.config['database']
        print(f"✅ Database: {db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config.get('database', 'brain')}")

    if 'openai' in config.config and config.config['openai'].get('api_key'):
        api_key = config.config['openai']['api_key']
        masked_key = api_key[:10] + "..." if len(api_key) > 10 else "***"
        print(f"✅ OpenAI: {masked_key}")

    print("\n🧪 Connection testing skipped for template setup")
    print("💡 You can test connections after installation completes")

    # Save configuration
    save = input("\n💾 Save configuration? [Y/n]: ").upper() != 'N'
    if save:
        await wizard.save_configuration(config)
        print("✅ Configuration saved to backend/config/settings.json")

        # Generate .env file
        generate_env(config)
        print("✅ Environment variables saved to backend/.env")

    print("\n" + "=" * 50)
    print("🎉 Installation Complete!")
    print("=" * 50)
    print("\nTo start brAIn, run:")
    print("  python3 -m uvicorn app.main:app --reload")
    print("\n(Make sure you're still in the virtual environment)")

def generate_env(config):
    """Generate .env file from configuration."""
    env_path = Path("backend/.env")

    env_lines = []

    # Database
    if 'database' in config.config:
        db_config = config.config['database']
        if 'connection_url' in db_config:
            env_lines.append(f"DATABASE_URL={db_config['connection_url']}")
        else:
            # Build database URL from components
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 5432)
            database = db_config.get('database', 'brain')
            username = db_config.get('username', 'user')
            password = db_config.get('password', 'password')
            db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            env_lines.append(f"DATABASE_URL={db_url}")

    # Supabase
    if 'supabase' in config.config:
        supabase_config = config.config['supabase']
        if supabase_config.get('url'):
            env_lines.append(f"SUPABASE_URL={supabase_config['url']}")
        if supabase_config.get('service_key'):
            env_lines.append(f"SUPABASE_SERVICE_KEY={supabase_config['service_key']}")

    # OpenAI
    if 'openai' in config.config:
        openai_config = config.config['openai']
        if openai_config.get('api_key'):
            env_lines.append(f"OPENAI_API_KEY={openai_config['api_key']}")

    # Anthropic
    if 'anthropic' in config.config:
        anthropic_config = config.config['anthropic']
        if anthropic_config.get('api_key'):
            env_lines.append(f"ANTHROPIC_API_KEY={anthropic_config['api_key']}")

    # Security
    if 'security' in config.config:
        security_config = config.config['security']
        if security_config.get('jwt_secret'):
            env_lines.append(f"JWT_SECRET={security_config['jwt_secret']}")

    with env_path.open('w') as f:
        f.write('\n'.join(env_lines))

if __name__ == "__main__":
    main()