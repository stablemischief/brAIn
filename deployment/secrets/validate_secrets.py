#!/usr/bin/env python3
"""
Secrets Validation Script for brAIn v2.0 Production Deployment
Validates that all required environment variables are set and properly formatted.
"""

import os
import sys
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SecretValidator:
    """Validates production environment variables."""

    REQUIRED_SECRETS = {
        # Security
        "JWT_SECRET": {"min_length": 32, "pattern": None},
        "JWT_ALGORITHM": {"values": ["HS256", "RS256"]},

        # Database
        "DB_PASSWORD": {"min_length": 16, "pattern": None},
        "DATABASE_URL": {"pattern": r"postgresql://.*"},

        # Supabase
        "SUPABASE_URL": {"pattern": r"https://.*\.supabase\.co"},
        "SUPABASE_ANON_KEY": {"min_length": 40},
        "SUPABASE_SERVICE_KEY": {"min_length": 40},

        # OpenAI
        "OPENAI_API_KEY": {"pattern": r"sk-.*", "min_length": 40},

        # Monitoring
        "GRAFANA_PASSWORD": {"min_length": 12},
        "MONITORING_PASSWORD_HASH": {"pattern": r"\$2[ayb]\$.*"},

        # Domain
        "DOMAIN_NAME": {"pattern": r"^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?\.[a-zA-Z]{2,}$"},
        "SSL_EMAIL": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
    }

    OPTIONAL_SECRETS = {
        "ANTHROPIC_API_KEY": {"pattern": r"sk-ant-.*", "min_length": 40},
        "LANGFUSE_PUBLIC_KEY": {"pattern": r"pk-lf-.*"},
        "LANGFUSE_SECRET_KEY": {"pattern": r"sk-lf-.*"},
        "S3_BACKUP_BUCKET": {"pattern": r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$"},
        "AWS_ACCESS_KEY_ID": {"pattern": r"^[A-Z0-9]{20}$"},
        "AWS_SECRET_ACCESS_KEY": {"min_length": 40},
        "SENTRY_DSN": {"pattern": r"https://.*@.*sentry\.io/.*"},
        "SLACK_WEBHOOK_URL": {"pattern": r"https://hooks\.slack\.com/.*"},
    }

    def __init__(self, env_file: Optional[Path] = None):
        """Initialize validator with optional env file path."""
        self.env_file = env_file or Path("deployment/secrets/.env.production")
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def load_env_file(self) -> Dict[str, str]:
        """Load environment variables from file."""
        if not self.env_file.exists():
            self.errors.append(f"Environment file not found: {self.env_file}")
            return {}

        env_vars = {}
        with open(self.env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip().strip('"').strip("'")

        return env_vars

    def validate_secret(
        self,
        key: str,
        value: str,
        requirements: Dict
    ) -> Tuple[bool, Optional[str]]:
        """Validate a single secret against requirements."""
        if not value or value == f"CHANGE_THIS_{key}":
            return False, f"Not set or still contains placeholder value"

        # Check minimum length
        if "min_length" in requirements:
            if len(value) < requirements["min_length"]:
                return False, f"Too short (min: {requirements['min_length']} chars)"

        # Check allowed values
        if "values" in requirements:
            if value not in requirements["values"]:
                return False, f"Must be one of: {', '.join(requirements['values'])}"

        # Check pattern
        if "pattern" in requirements:
            pattern = requirements["pattern"]
            if not re.match(pattern, value):
                return False, f"Does not match required pattern: {pattern}"

        return True, None

    def validate(self) -> bool:
        """Validate all secrets."""
        print("🔐 Validating Production Secrets...")
        print("=" * 60)

        env_vars = self.load_env_file()

        if not env_vars:
            print("❌ Failed to load environment file")
            return False

        # Validate required secrets
        print("\n📋 Checking Required Secrets:")
        for key, requirements in self.REQUIRED_SECRETS.items():
            value = env_vars.get(key, "")
            is_valid, error = self.validate_secret(key, value, requirements)

            if is_valid:
                print(f"  ✅ {key}: Valid")
            else:
                print(f"  ❌ {key}: {error}")
                self.errors.append(f"{key}: {error}")

        # Validate optional secrets if present
        print("\n📋 Checking Optional Secrets:")
        for key, requirements in self.OPTIONAL_SECRETS.items():
            if key in env_vars:
                value = env_vars[key]
                is_valid, error = self.validate_secret(key, value, requirements)

                if is_valid:
                    print(f"  ✅ {key}: Valid")
                else:
                    print(f"  ⚠️  {key}: {error}")
                    self.warnings.append(f"{key}: {error}")
            else:
                print(f"  ⏭️  {key}: Not configured (optional)")

        # Check for insecure values
        print("\n🔍 Security Checks:")
        security_issues = self.check_security(env_vars)
        if security_issues:
            for issue in security_issues:
                print(f"  ⚠️  {issue}")
                self.warnings.append(issue)
        else:
            print("  ✅ No security issues detected")

        # Summary
        print("\n" + "=" * 60)
        print("📊 Validation Summary:")
        print(f"  Errors: {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")

        if self.errors:
            print("\n❌ Validation Failed! Please fix the following errors:")
            for error in self.errors:
                print(f"  • {error}")
            return False

        if self.warnings:
            print("\n⚠️  Warnings (non-critical):")
            for warning in self.warnings:
                print(f"  • {warning}")

        print("\n✅ All required secrets are valid!")
        return True

    def check_security(self, env_vars: Dict[str, str]) -> List[str]:
        """Check for common security issues."""
        issues = []

        # Check for default/weak passwords
        weak_patterns = ["password", "123456", "admin", "default", "test"]
        password_fields = [k for k in env_vars.keys() if "PASSWORD" in k.upper()]

        for field in password_fields:
            value = env_vars.get(field, "").lower()
            for pattern in weak_patterns:
                if pattern in value:
                    issues.append(f"{field} contains weak pattern: '{pattern}'")

        # Check for exposed production keys in example values
        if "example" in str(self.env_file).lower():
            if any(env_vars.get(k, "").startswith("sk-") for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]):
                issues.append("Real API keys detected in example file!")

        # Check JWT secret randomness
        jwt_secret = env_vars.get("JWT_SECRET", "")
        if jwt_secret and len(set(jwt_secret)) < 10:
            issues.append("JWT_SECRET lacks sufficient randomness")

        return issues

    def generate_missing(self) -> None:
        """Generate secure values for missing secrets."""
        print("\n🔧 Generating secure values for missing secrets...")

        import secrets
        import string

        suggestions = {
            "JWT_SECRET": secrets.token_urlsafe(32),
            "DB_PASSWORD": ''.join(
                secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*")
                for _ in range(24)
            ),
            "GRAFANA_PASSWORD": ''.join(
                secrets.choice(string.ascii_letters + string.digits)
                for _ in range(16)
            ),
        }

        print("\nSuggested secure values:")
        for key, value in suggestions.items():
            print(f"\n{key}={value}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate production secrets for brAIn deployment"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to environment file (default: deployment/secrets/.env.production)"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate secure values for missing secrets"
    )

    args = parser.parse_args()

    validator = SecretValidator(args.env_file)
    is_valid = validator.validate()

    if args.generate and not is_valid:
        validator.generate_missing()

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()