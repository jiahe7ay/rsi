"""Onboarding layer: input a benchmark repo -> analyze, provision-check, elicit
missing resources, smoke-test. Importing this package registers built-in adapters.
"""
from ouroboros.onboard import adapters  # noqa: F401  (side effect: registers adapters)
from ouroboros.onboard.orchestrator import onboard

__all__ = ["onboard"]
