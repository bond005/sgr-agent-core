"""Compatibility shims for running the framework on Python 3.10+.

Currently the only construct requiring a shim is :data:`typing.Self`, which was
added in Python 3.11. Import ``Self`` from this module everywhere instead of
``typing`` so the codebase works on 3.10 (via ``typing_extensions``) and 3.11+.
"""

try:
    from typing import Self
except ImportError:  # Python < 3.11
    from typing_extensions import Self

__all__ = ["Self"]
