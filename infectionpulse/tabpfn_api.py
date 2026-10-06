"""Guards for hosted TabPFN-3.5 calls: token presence and verified model version."""

import os
import re

TABPFN_VERSION = "v3.5"


def require_api_token():
    if not os.getenv("TABPFN_TOKEN"):
        raise RuntimeError(
            "TABPFN_TOKEN is missing. Copy .env.example to .env and set it locally."
        )


def verify_api_version(estimator):
    meta = getattr(estimator, "_last_meta", {})
    if hasattr(meta, "model_dump"):
        meta = meta.model_dump(mode="json")
    if not isinstance(meta, dict):
        raise RuntimeError("Cannot verify API response model metadata")
    config = meta.get("tabpfn_config", {}) or {}
    resolved = config.get("model_path")
    if not resolved:
        raise RuntimeError(
            "API response did not identify its model; refusing unverified predictions"
        )
    if not re.search(r"(^v3\.5_|-v3\.5-)", resolved) or "v3.5-fast" in resolved:
        raise RuntimeError(f"Expected TabPFN-3.5; API returned {resolved}")
    billed = meta.get("billing_model_version")
    if billed and billed != TABPFN_VERSION:
        raise RuntimeError(f"Unexpected API model version: {billed}")
    return {
        "requested": estimator.model_path,
        "resolved": resolved,
        "billing_version": billed,
    }
