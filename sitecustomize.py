"""Ensure a writable temp directory is available in sandboxed environments.

Pytest (and the stdlib tempfile module) can fail early if /tmp is not writable or missing.
By setting TMPDIR and tempfile.tempdir here, we point temp usage to a project-local folder.
"""
import os
import pathlib
import tempfile

_tmp_base = pathlib.Path(__file__).parent / ".pytest_tmp"
_tmp_base.mkdir(exist_ok=True)

os.environ.setdefault("TMPDIR", str(_tmp_base))
tempfile.tempdir = str(_tmp_base)
