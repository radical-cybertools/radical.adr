"""ADL exception hierarchy."""

from __future__ import annotations


class ADLError(Exception):
    """Base exception for all radical.adl errors."""


class PolicyError(ADLError):
    """Raised when a policy produces an invalid or unexecutable Decision."""


class ActionError(ADLError):
    """Raised when an Action cannot be applied (unknown task name, missing future, etc.)."""


class RegistryError(ADLError):
    """Raised on tool or task registry lookup failures."""
