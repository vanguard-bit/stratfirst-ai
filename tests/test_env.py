from __future__ import annotations

import os

import pytest

from nse_trader.config import ROOT
from nse_trader.env import load_dotenv, require_env

pytestmark = pytest.mark.phase0


RETIRED_GEMINI_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
}


class TestDotenv:
    def test_env_example_exists(self):
        assert (ROOT / ".env.example").exists()

    def test_load_dotenv_does_not_override_existing(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "from-env")
        load_dotenv()
        assert os.environ["GEMINI_MODEL"] == "from-env"

    def test_require_env_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_TEST_KEY_XYZ", raising=False)
        with pytest.raises(RuntimeError, match="NONEXISTENT_TEST_KEY_XYZ"):
            require_env("NONEXISTENT_TEST_KEY_XYZ")

    def test_gemini_model_not_retired(self):
        load_dotenv()
        model = os.environ.get("GEMINI_MODEL", "")
        if model:
            assert model not in RETIRED_GEMINI_MODELS, f"Update GEMINI_MODEL — {model} is retired"

    def test_gemini_api_key_present_in_env_file(self):
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                assert line.split("=", 1)[1].strip(), "GEMINI_API_KEY is empty in .env"
                return
        pytest.fail("GEMINI_API_KEY not found in .env")
