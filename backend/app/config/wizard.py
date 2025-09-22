"""
AI-Powered Configuration Wizard with Pydantic validation.

This module provides the main configuration wizard that guides users
through system setup with AI assistance and comprehensive validation.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

from pydantic import ValidationError

from config.models import (
    SystemConfig,
    ConfigurationExport,
    ValidationResult,
    DatabaseConfig,
    OpenAIConfig,
    AnthropicConfig,
    SupabaseConfig,
    LangfuseConfig,
    SecurityConfig,
    CostManagementConfig,
    ProcessingConfig
)
from config.validators import ConfigurationValidator
from config.templates import ConfigurationTemplates
from config.sql_generator import SQLScriptGenerator


class ConfigurationWizard:
    """AI-Powered configuration wizard for system setup."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration wizard.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or Path("config/settings.json")
        self.current_config: Optional[SystemConfig] = None
        self.validator = ConfigurationValidator()
        self.templates = ConfigurationTemplates()
        self.config_history: List[ConfigurationExport] = []

    async def start_wizard(self, use_ai_assistance: bool = True) -> SystemConfig:
        """
        Start the configuration wizard process.

        Args:
            use_ai_assistance: Whether to use AI assistance

        Returns:
            Validated system configuration
        """
        print("\n" + "="*60)
        print("🚀 brAIn Configuration Wizard")
        print("="*60)

        # Step 1: Template selection
        config_dict = await self._select_template()

        # Step 2: Environment setup
        config_dict = await self._configure_environment(config_dict)

        # Step 3: Database configuration
        config_dict = await self._configure_database(config_dict)

        # Step 4: LLM provider configuration
        config_dict = await self._configure_llm_providers(config_dict)

        # Step 5: Supabase configuration
        config_dict = await self._configure_supabase(config_dict)

        # Step 6: Security configuration
        config_dict = await self._configure_security(config_dict)

        # Step 7: Cost management
        config_dict = await self._configure_cost_management(config_dict)

        # Step 8: Processing configuration
        config_dict = await self._configure_processing(config_dict)

        # Step 9: Optional monitoring (Langfuse)
        config_dict = await self._configure_monitoring(config_dict)

        # Step 10: Validate complete configuration
        self.current_config = await self._validate_and_finalize(config_dict)

        return self.current_config

    async def _select_template(self) -> Dict[str, Any]:
        """Select a configuration template."""
        print("\n📋 Configuration Templates Available:")
        print("-" * 40)

        templates = self.templates.get_all_templates()
        for i, template in enumerate(templates, 1):
            print(f"{i}. {template.name}")
            print(f"   {template.description}")
            print(f"   Tags: {', '.join(template.tags)}")

        print(f"{len(templates) + 1}. Custom Configuration (start from scratch)")

        choice = input("\nSelect template (number): ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(templates):
                template = templates[idx]
                print(f"\n✅ Selected template: {template.name}")
                return template.config.copy()
            else:
                print("\n📝 Starting with custom configuration")
                return self._get_minimal_config()
        except (ValueError, IndexError):
            print("\n⚠️ Invalid choice, starting with custom configuration")
            return self._get_minimal_config()

    async def _configure_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure deployment environment."""
        print("\n🌍 Environment Configuration")
        print("-" * 40)

        environments = ["development", "staging", "production"]
        current = config.get("environment", "development")

        print(f"Current: {current}")
        print("Available: " + ", ".join(environments))

        env = input(f"Environment [{current}]: ").strip().lower()
        if env and env in environments:
            config["environment"] = env
        else:
            config["environment"] = current

        print(f"✅ Environment set to: {config['environment']}")
        return config

    async def _configure_database(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure database settings."""
        print("\n🗄️ Database Configuration")
        print("-" * 40)

        if "database" not in config:
            config["database"] = {}

        db = config["database"]

        # Check for environment variables first
        if os.getenv("DATABASE_URL"):
            use_env = input("Use DATABASE_URL from environment? (y/n) [y]: ").strip().lower()
            if use_env != 'n':
                # Parse DATABASE_URL
                from urllib.parse import urlparse
                url = urlparse(os.getenv("DATABASE_URL"))
                db["host"] = url.hostname or "localhost"
                db["port"] = url.port or 5432
                db["database"] = url.path.lstrip("/") if url.path else "brain_db"
                db["username"] = url.username or "postgres"
                db["password"] = url.password or ""
                print("✅ Database configuration loaded from DATABASE_URL")
                config["database"] = db
                return config

        # Manual configuration
        db["host"] = input(f"Host [{db.get('host', 'localhost')}]: ").strip() or db.get('host', 'localhost')
        db["port"] = int(input(f"Port [{db.get('port', 5432)}]: ").strip() or db.get('port', 5432))
        db["database"] = input(f"Database [{db.get('database', 'brain_db')}]: ").strip() or db.get('database', 'brain_db')
        db["username"] = input(f"Username [{db.get('username', 'postgres')}]: ").strip() or db.get('username', 'postgres')

        # Password input (hidden)
        import getpass
        db["password"] = getpass.getpass("Password: ") or db.get('password', '')

        db["schema"] = input(f"Schema [{db.get('schema', 'public')}]: ").strip() or db.get('schema', 'public')
        db["pool_size"] = int(input(f"Pool size [{db.get('pool_size', 10)}]: ").strip() or db.get('pool_size', 10))
        db["max_overflow"] = int(input(f"Max overflow [{db.get('max_overflow', 5)}]: ").strip() or db.get('max_overflow', 5))

        config["database"] = db
        print("✅ Database configuration complete")
        return config

    async def _configure_llm_providers(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure LLM providers (OpenAI/Anthropic)."""
        print("\n🤖 LLM Provider Configuration")
        print("-" * 40)

        # OpenAI configuration
        configure_openai = input("Configure OpenAI? (y/n) [y]: ").strip().lower()
        if configure_openai != 'n':
            if "openai" not in config:
                config["openai"] = {}

            openai_cfg = config["openai"]

            # Check for environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                use_env = input("Use OPENAI_API_KEY from environment? (y/n) [y]: ").strip().lower()
                if use_env != 'n':
                    openai_cfg["api_key"] = api_key
                else:
                    import getpass
                    openai_cfg["api_key"] = getpass.getpass("OpenAI API Key: ")
            else:
                import getpass
                openai_cfg["api_key"] = getpass.getpass("OpenAI API Key: ")

            openai_cfg["model"] = input(f"Model [{openai_cfg.get('model', 'gpt-4o-mini')}]: ").strip() or openai_cfg.get('model', 'gpt-4o-mini')
            openai_cfg["temperature"] = float(input(f"Temperature [{openai_cfg.get('temperature', 0.7)}]: ").strip() or openai_cfg.get('temperature', 0.7))
            openai_cfg["max_tokens"] = int(input(f"Max tokens [{openai_cfg.get('max_tokens', 4000)}]: ").strip() or openai_cfg.get('max_tokens', 4000))
            openai_cfg["timeout"] = int(input(f"Timeout (seconds) [{openai_cfg.get('timeout', 30)}]: ").strip() or openai_cfg.get('timeout', 30))
            openai_cfg["max_retries"] = int(input(f"Max retries [{openai_cfg.get('max_retries', 3)}]: ").strip() or openai_cfg.get('max_retries', 3))

            config["openai"] = openai_cfg
            print("✅ OpenAI configuration complete")

        # Anthropic configuration
        configure_anthropic = input("\nConfigure Anthropic Claude? (y/n) [n]: ").strip().lower()
        if configure_anthropic == 'y':
            if "anthropic" not in config:
                config["anthropic"] = {}

            anthropic_cfg = config["anthropic"]

            # Check for environment variable
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                use_env = input("Use ANTHROPIC_API_KEY from environment? (y/n) [y]: ").strip().lower()
                if use_env != 'n':
                    anthropic_cfg["api_key"] = api_key
                else:
                    import getpass
                    anthropic_cfg["api_key"] = getpass.getpass("Anthropic API Key: ")
            else:
                import getpass
                anthropic_cfg["api_key"] = getpass.getpass("Anthropic API Key: ")

            anthropic_cfg["model"] = input(f"Model [{anthropic_cfg.get('model', 'claude-3-5-sonnet-20241022')}]: ").strip() or anthropic_cfg.get('model', 'claude-3-5-sonnet-20241022')
            anthropic_cfg["max_tokens"] = int(input(f"Max tokens [{anthropic_cfg.get('max_tokens', 4000)}]: ").strip() or anthropic_cfg.get('max_tokens', 4000))
            anthropic_cfg["temperature"] = float(input(f"Temperature [{anthropic_cfg.get('temperature', 0.7)}]: ").strip() or anthropic_cfg.get('temperature', 0.7))
            anthropic_cfg["timeout"] = int(input(f"Timeout (seconds) [{anthropic_cfg.get('timeout', 60)}]: ").strip() or anthropic_cfg.get('timeout', 60))

            config["anthropic"] = anthropic_cfg
            print("✅ Anthropic configuration complete")

        return config

    async def _configure_supabase(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Supabase settings."""
        print("\n🔥 Supabase Configuration")
        print("-" * 40)

        if "supabase" not in config:
            config["supabase"] = {}

        supabase = config["supabase"]

        # Check for environment variables
        url = os.getenv("SUPABASE_URL")
        anon_key = os.getenv("SUPABASE_ANON_KEY")

        if url and anon_key:
            use_env = input("Use Supabase settings from environment? (y/n) [y]: ").strip().lower()
            if use_env != 'n':
                supabase["url"] = url
                supabase["anon_key"] = anon_key
                supabase["service_key"] = os.getenv("SUPABASE_SERVICE_KEY")
                supabase["jwt_secret"] = os.getenv("SUPABASE_JWT_SECRET")
                print("✅ Supabase configuration loaded from environment")
                config["supabase"] = supabase
                return config

        # Manual configuration
        supabase["url"] = input(f"Supabase URL [{supabase.get('url', '')}]: ").strip() or supabase.get('url', '')

        import getpass
        supabase["anon_key"] = getpass.getpass("Anon Key: ") or supabase.get('anon_key', '')

        configure_service = input("Configure service key? (y/n) [n]: ").strip().lower()
        if configure_service == 'y':
            supabase["service_key"] = getpass.getpass("Service Key: ") or None
            supabase["jwt_secret"] = getpass.getpass("JWT Secret: ") or None

        config["supabase"] = supabase
        print("✅ Supabase configuration complete")
        return config

    async def _configure_security(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure security settings."""
        print("\n🔒 Security Configuration")
        print("-" * 40)

        if "security" not in config:
            config["security"] = {}

        sec = config["security"]
        is_production = config.get("environment") == "production"

        # JWT configuration
        sec["jwt_enabled"] = is_production or input(f"Enable JWT? (y/n) [{sec.get('jwt_enabled', True)}]: ").strip().lower() != 'n'

        if sec["jwt_enabled"]:
            import getpass
            jwt_secret = os.getenv("JWT_SECRET")
            if jwt_secret:
                use_env = input("Use JWT_SECRET from environment? (y/n) [y]: ").strip().lower()
                if use_env != 'n':
                    sec["jwt_secret"] = jwt_secret
                else:
                    sec["jwt_secret"] = getpass.getpass("JWT Secret (min 32 chars): ")
            else:
                sec["jwt_secret"] = getpass.getpass("JWT Secret (min 32 chars): ")

            sec["jwt_algorithm"] = input(f"JWT Algorithm [{sec.get('jwt_algorithm', 'HS256')}]: ").strip() or sec.get('jwt_algorithm', 'HS256')
            sec["jwt_expiry_hours"] = int(input(f"JWT Expiry (hours) [{sec.get('jwt_expiry_hours', 24)}]: ").strip() or sec.get('jwt_expiry_hours', 24))

        # CORS configuration
        sec["cors_enabled"] = True
        origins_input = input(f"CORS Origins (comma-separated) [{','.join(sec.get('cors_origins', ['http://localhost:3000']))}]: ").strip()
        if origins_input:
            sec["cors_origins"] = [origin.strip() for origin in origins_input.split(',')]
        else:
            sec["cors_origins"] = sec.get('cors_origins', ['http://localhost:3000'])

        # Rate limiting
        sec["rate_limit_enabled"] = is_production or input("Enable rate limiting? (y/n) [y]: ").strip().lower() != 'n'
        if sec["rate_limit_enabled"]:
            sec["rate_limit_requests"] = int(input(f"Requests per minute [{sec.get('rate_limit_requests', 100)}]: ").strip() or sec.get('rate_limit_requests', 100))

        # Protection flags
        sec["input_validation_strict"] = True
        sec["sql_injection_protection"] = True
        sec["xss_protection"] = True

        config["security"] = sec
        print("✅ Security configuration complete")
        return config

    async def _configure_cost_management(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure cost management settings."""
        print("\n💰 Cost Management Configuration")
        print("-" * 40)

        if "cost_management" not in config:
            config["cost_management"] = {}

        cost = config["cost_management"]

        cost["daily_budget"] = float(input(f"Daily budget (USD) [{cost.get('daily_budget', 50.0)}]: ").strip() or cost.get('daily_budget', 50.0))
        cost["monthly_budget"] = float(input(f"Monthly budget (USD) [{cost.get('monthly_budget', 1000.0)}]: ").strip() or cost.get('monthly_budget', 1000.0))
        cost["alert_threshold_percent"] = int(input(f"Alert threshold (%) [{cost.get('alert_threshold_percent', 80)}]: ").strip() or cost.get('alert_threshold_percent', 80))

        is_production = config.get("environment") == "production"
        cost["hard_limit_enabled"] = is_production or input("Enable hard budget limits? (y/n) [y]: ").strip().lower() != 'n'

        # Keep default cost per token values
        if "cost_per_1k_tokens" not in cost:
            cost["cost_per_1k_tokens"] = {
                "gpt-4o-mini": 0.00015,
                "gpt-4o": 0.005,
                "claude-3-5-sonnet-20241022": 0.003,
                "text-embedding-3-small": 0.00002
            }

        config["cost_management"] = cost
        print("✅ Cost management configuration complete")
        return config

    async def _configure_processing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure processing settings."""
        print("\n⚙️ Processing Configuration")
        print("-" * 40)

        if "processing" not in config:
            config["processing"] = {}

        proc = config["processing"]

        proc["batch_size"] = int(input(f"Batch size [{proc.get('batch_size', 10)}]: ").strip() or proc.get('batch_size', 10))
        proc["parallel_workers"] = int(input(f"Parallel workers [{proc.get('parallel_workers', 4)}]: ").strip() or proc.get('parallel_workers', 4))
        proc["chunk_size"] = int(input(f"Text chunk size [{proc.get('chunk_size', 1000)}]: ").strip() or proc.get('chunk_size', 1000))
        proc["chunk_overlap"] = int(input(f"Chunk overlap [{proc.get('chunk_overlap', 200)}]: ").strip() or proc.get('chunk_overlap', 200))
        proc["quality_threshold"] = float(input(f"Quality threshold (0-1) [{proc.get('quality_threshold', 0.7)}]: ").strip() or proc.get('quality_threshold', 0.7))
        proc["duplicate_threshold"] = float(input(f"Duplicate threshold (0-1) [{proc.get('duplicate_threshold', 0.95)}]: ").strip() or proc.get('duplicate_threshold', 0.95))
        proc["max_file_size_mb"] = int(input(f"Max file size (MB) [{proc.get('max_file_size_mb', 100)}]: ").strip() or proc.get('max_file_size_mb', 100))

        # Keep default supported formats
        if "supported_formats" not in proc:
            proc["supported_formats"] = ["pdf", "docx", "txt", "md", "html", "json", "csv"]

        config["processing"] = proc
        print("✅ Processing configuration complete")
        return config

    async def _configure_monitoring(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure optional monitoring (Langfuse)."""
        print("\n📊 Monitoring Configuration (Optional)")
        print("-" * 40)

        configure_langfuse = input("Configure Langfuse monitoring? (y/n) [n]: ").strip().lower()
        if configure_langfuse == 'y':
            if "langfuse" not in config:
                config["langfuse"] = {}

            langfuse = config["langfuse"]
            langfuse["enabled"] = True

            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")

            if public_key and secret_key:
                use_env = input("Use Langfuse keys from environment? (y/n) [y]: ").strip().lower()
                if use_env != 'n':
                    langfuse["public_key"] = public_key
                    langfuse["secret_key"] = secret_key
                else:
                    langfuse["public_key"] = input("Langfuse public key: ").strip()
                    import getpass
                    langfuse["secret_key"] = getpass.getpass("Langfuse secret key: ")
            else:
                langfuse["public_key"] = input("Langfuse public key: ").strip()
                import getpass
                langfuse["secret_key"] = getpass.getpass("Langfuse secret key: ")

            custom_host = input("Custom Langfuse host (leave empty for cloud): ").strip()
            if custom_host:
                langfuse["host"] = custom_host

            config["langfuse"] = langfuse
            print("✅ Langfuse configuration complete")
        else:
            config["langfuse"] = {"enabled": False}

        return config

    async def _validate_and_finalize(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """
        Validate and finalize the configuration.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Validated SystemConfig object
        """
        print("\n🔍 Validating Configuration")
        print("-" * 40)

        try:
            # Create SystemConfig object
            system_config = SystemConfig(**config_dict)
            print("✅ Configuration structure valid")

            # Run comprehensive validation
            print("\n🧪 Running validation tests...")
            validation_result = await self.validator.validate_complete_config(system_config)

            # Display validation results
            self._display_validation_results(validation_result)

            if not validation_result.valid:
                print("\n⚠️ Configuration has errors that need to be fixed.")
                retry = input("Would you like to retry configuration? (y/n): ").strip().lower()
                if retry == 'y':
                    return await self.start_wizard()
                else:
                    raise ValueError("Configuration validation failed")

            # Save configuration
            save = input("\n💾 Save configuration? (y/n) [y]: ").strip().lower()
            if save != 'n':
                await self.save_configuration(system_config)

            # Generate SQL scripts
            generate_sql = input("\n📝 Generate SQL setup scripts? (y/n) [y]: ").strip().lower()
            if generate_sql != 'n':
                await self._generate_sql_scripts(system_config)

            print("\n✅ Configuration wizard complete!")
            return system_config

        except ValidationError as e:
            print(f"\n❌ Configuration validation failed:")
            for error in e.errors():
                print(f"  - {error['loc']}: {error['msg']}")
            raise

    def _display_validation_results(self, result: ValidationResult):
        """Display validation results."""
        if result.errors:
            print("\n❌ Errors found:")
            for error in result.errors:
                print(f"  - {error}")

        if result.warnings:
            print("\n⚠️ Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

        if result.tested_components:
            print("\n✔️ Component Tests:")
            for component, status in result.tested_components.items():
                emoji = "✅" if status else "❌"
                print(f"  {emoji} {component}")

        if result.recommendations:
            print("\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")

    async def save_configuration(self, config: SystemConfig):
        """
        Save configuration to file.

        Args:
            config: System configuration to save
        """
        # Create export
        export = ConfigurationExport(
            version="1.0.0",
            config=config
        )
        export.checksum = export.calculate_checksum()

        # Save to file
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(
                export.model_dump(mode='json'),
                f,
                indent=2,
                default=str
            )

        print(f"✅ Configuration saved to {self.config_path}")

        # Also save environment file
        await self._save_env_file(config)

    async def _save_env_file(self, config: SystemConfig):
        """Save environment variables file."""
        env_path = Path(".env.wizard")

        env_content = []
        env_content.append("# Generated by brAIn Configuration Wizard")
        env_content.append(f"# {datetime.utcnow().isoformat()}")
        env_content.append("")

        # Database
        env_content.append(f"DATABASE_HOST={config.database.host}")
        env_content.append(f"DATABASE_PORT={config.database.port}")
        env_content.append(f"DATABASE_NAME={config.database.database}")
        env_content.append(f"DATABASE_USER={config.database.username}")
        env_content.append(f"DATABASE_PASSWORD={config.database.password.get_secret_value()}")
        env_content.append("")

        # LLM Providers
        if config.openai:
            env_content.append(f"OPENAI_API_KEY={config.openai.api_key.get_secret_value()}")
        if config.anthropic:
            env_content.append(f"ANTHROPIC_API_KEY={config.anthropic.api_key.get_secret_value()}")
        env_content.append("")

        # Supabase
        env_content.append(f"SUPABASE_URL={config.supabase.url}")
        env_content.append(f"SUPABASE_ANON_KEY={config.supabase.anon_key.get_secret_value()}")
        if config.supabase.service_key:
            env_content.append(f"SUPABASE_SERVICE_KEY={config.supabase.service_key.get_secret_value()}")
        if config.supabase.jwt_secret:
            env_content.append(f"SUPABASE_JWT_SECRET={config.supabase.jwt_secret.get_secret_value()}")
        env_content.append("")

        # Security
        env_content.append(f"JWT_SECRET={config.security.jwt_secret.get_secret_value()}")
        env_content.append(f"JWT_ALGORITHM={config.security.jwt_algorithm}")
        env_content.append("")

        # Langfuse
        if config.langfuse and config.langfuse.enabled:
            env_content.append(f"LANGFUSE_PUBLIC_KEY={config.langfuse.public_key}")
            if config.langfuse.secret_key:
                env_content.append(f"LANGFUSE_SECRET_KEY={config.langfuse.secret_key.get_secret_value()}")
            if config.langfuse.host:
                env_content.append(f"LANGFUSE_HOST={config.langfuse.host}")

        with open(env_path, 'w') as f:
            f.write("\n".join(env_content))

        print(f"✅ Environment variables saved to {env_path}")

    async def _generate_sql_scripts(self, config: SystemConfig):
        """Generate SQL setup scripts."""
        generator = SQLScriptGenerator(config)

        # Generate setup script
        setup_script = generator.generate_complete_setup()
        setup_path = Path("config/sql/setup.sql")
        setup_path.parent.mkdir(parents=True, exist_ok=True)

        with open(setup_path, 'w') as f:
            f.write(setup_script)

        print(f"✅ SQL setup script saved to {setup_path}")

        # Generate rollback script
        rollback_script = generator.generate_rollback_script()
        rollback_path = Path("config/sql/rollback.sql")

        with open(rollback_path, 'w') as f:
            f.write(rollback_script)

        print(f"✅ SQL rollback script saved to {rollback_path}")

    async def load_configuration(self, path: Optional[Path] = None) -> SystemConfig:
        """
        Load configuration from file.

        Args:
            path: Configuration file path

        Returns:
            Loaded system configuration
        """
        path = path or self.config_path

        with open(path, 'r') as f:
            export_dict = json.load(f)

        export = ConfigurationExport(**export_dict)

        # Verify checksum
        calculated_checksum = export.calculate_checksum()
        if export.checksum and export.checksum != calculated_checksum:
            print("⚠️ Warning: Configuration checksum mismatch")

        return export.config

    def _get_minimal_config(self) -> Dict[str, Any]:
        """Get minimal configuration structure."""
        return {
            "environment": "development",
            "database": {},
            "supabase": {},
            "security": {},
            "cost_management": {},
            "processing": {}
        }


async def main():
    """Main entry point for the configuration wizard."""
    wizard = ConfigurationWizard()

    # Check for existing configuration
    if wizard.config_path.exists():
        print(f"\n📋 Existing configuration found at {wizard.config_path}")
        load_existing = input("Load existing configuration? (y/n) [y]: ").strip().lower()

        if load_existing != 'n':
            try:
                config = await wizard.load_configuration()
                print("✅ Configuration loaded successfully")

                # Optionally re-validate
                revalidate = input("Re-validate configuration? (y/n) [y]: ").strip().lower()
                if revalidate != 'n':
                    validator = ConfigurationValidator()
                    result = await validator.validate_complete_config(config)
                    wizard._display_validation_results(result)

                return config
            except Exception as e:
                print(f"❌ Error loading configuration: {e}")
                print("Starting new configuration...")

    # Start wizard
    config = await wizard.start_wizard()
    return config


if __name__ == "__main__":
    asyncio.run(main())