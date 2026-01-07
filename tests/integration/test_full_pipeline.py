"""Integration tests for the full pipeline.

These tests verify that the pipeline scripts exist and can be imported.
They also test the configuration loading with real files.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestPipelineScriptsExist:
    """Test that all pipeline scripts exist and are executable."""

    EXPECTED_SCRIPTS = [
        "parse_mbox.py",
        "import_to_postgres.py",
        "populate_threads.py",
        "enrich_emails_db.py",
        "compute_basic_features.py",
        "compute_embeddings.py",
        "classify_ai_handleability.py",
        "populate_users.py",
        "cluster_emails.py",
        "compute_priority.py",
        "run_llm_classification.py",
        "onboard_data.py",
        "checkpoint.py",
        "query_db.py",
        "validate_data.py",
    ]

    def test_scripts_directory_exists(self) -> None:
        """Test that scripts directory exists."""
        scripts_dir = PROJECT_ROOT / "scripts"
        assert scripts_dir.exists(), f"scripts/ directory not found at {scripts_dir}"

    @pytest.mark.parametrize("script_name", EXPECTED_SCRIPTS)
    def test_script_exists(self, script_name: str) -> None:
        """Test that each expected script exists."""
        script_path = PROJECT_ROOT / "scripts" / script_name
        assert script_path.exists(), f"Script not found: {script_path}"

    @pytest.mark.parametrize("script_name", EXPECTED_SCRIPTS)
    def test_script_has_python_syntax(self, script_name: str) -> None:
        """Test that each script has valid Python syntax."""
        script_path = PROJECT_ROOT / "scripts" / script_name
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error in {script_name}: {result.stderr}"


class TestConfigurationLoading:
    """Test configuration loading integration."""

    def test_config_loads_from_real_env_file(self, tmp_path: Path) -> None:
        """Test that Config can load from a real .env file."""
        from rl_emails.core.config import Config

        # Create a real .env file
        env_file = tmp_path / ".env"
        env_file.write_text(
            """
DATABASE_URL=postgresql://test:test@localhost:5432/testdb
MBOX_PATH=/tmp/test.mbox
YOUR_EMAIL=test@example.com
OPENAI_API_KEY=sk-test-key
ANTHROPIC_API_KEY=ant-test-key
"""
        )

        # Clear environment variables that might interfere
        old_env = {}
        for key in [
            "DATABASE_URL",
            "MBOX_PATH",
            "YOUR_EMAIL",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ]:
            if key in os.environ:
                old_env[key] = os.environ.pop(key)

        try:
            config = Config.from_env(env_file=env_file)

            assert config.database_url == "postgresql://test:test@localhost:5432/testdb"
            assert config.mbox_path == Path("/tmp/test.mbox")
            assert config.your_email == "test@example.com"
            assert config.openai_api_key == "sk-test-key"
            assert config.anthropic_api_key == "ant-test-key"
        finally:
            # Restore environment
            os.environ.update(old_env)


class TestOnboardStatus:
    """Test onboard_data.py --status command."""

    @pytest.mark.skipif(
        "DATABASE_URL" not in os.environ,
        reason="DATABASE_URL not set - skip status check",
    )
    def test_onboard_status_runs(self) -> None:
        """Test that onboard_data.py --status runs without error."""
        script_path = PROJECT_ROOT / "scripts" / "onboard_data.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Status command should work if DATABASE_URL is set
        # It may exit with error if database doesn't exist, but shouldn't crash
        assert "Traceback" not in result.stderr or "DATABASE_URL" in result.stderr
