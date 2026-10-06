import subprocess
import sys

import pytest

from infectionpulse.tabpfn_api import verify_api_version


class FakeRegressor:
    model_path = "v3.5_default"


def test_installed_sdk_supports_explicit_35():
    # SDK imports PyTorch, which must not share an OpenMP runtime with LightGBM.
    script = """
from tabpfn_client import TabPFNClassifier, TabPFNRegressor
for cls in [TabPFNClassifier, TabPFNRegressor]:
    model = cls.create_default_for_version('v3.5', n_estimators=8, fit_mode='fit_with_cache')
    assert model.model_path == 'v3.5_default'
    assert hasattr(model, 'save_model') and hasattr(model, 'load_model')
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


def test_reject_other_or_unverified_api_versions():
    model = FakeRegressor()
    model._last_meta = {"tabpfn_config": {"model_path": "v3_default"}}
    with pytest.raises(RuntimeError, match="Expected TabPFN-3.5"):
        verify_api_version(model)
    model._last_meta = {}
    with pytest.raises(RuntimeError, match="unverified"):
        verify_api_version(model)


def test_accept_verified_35_response():
    model = FakeRegressor()
    model._last_meta = {
        "tabpfn_config": {"model_path": "/app/tabpfn-v3.5-20260909.safetensors"},
        "billing_model_version": "v3.5",
    }
    assert verify_api_version(model)["billing_version"] == "v3.5"
