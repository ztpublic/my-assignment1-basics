from __future__ import annotations

import importlib.metadata


try:
    # Prefer the installed package version when available.
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    # Fall back to a source-tree marker when the package has not been installed.
    __version__ = "0.0.0"
