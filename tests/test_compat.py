"""Tests for the Python 3.10 compatibility shim module."""

from sgr_agent_core._compat import Self


class TestCompatShim:
    def test_self_is_importable_and_truthy(self):
        """Self is resolved (from typing on 3.11+, from typing_extensions on 3.10)."""
        assert Self is not None

    def test_self_usable_as_annotation(self):
        """Self works as a return annotation at runtime."""

        class Foo:
            def clone(self) -> Self:
                return self

        assert isinstance(Foo().clone(), Foo)

    def test_self_supports_union(self):
        """Self | None evaluates at runtime (needed by GlobalConfig ClassVar)."""
        annotation: type | None = type(None)
        # Build a ClassVar-like union involving Self; just assert it doesn't raise.
        _ = Self | None  # noqa: F841
        assert annotation is not None
