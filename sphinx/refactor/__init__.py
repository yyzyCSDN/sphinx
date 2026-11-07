"""Source-level refactoring helpers for Sphinx projects.

.. versionadded:: 9.3

The :mod:`sphinx.refactor` package contains *source code* refactoring tools:
they analyse a project using the same parser and domain machinery as a real
build, then edit the reStructuredText *sources* (not generated output) so
that the project keeps building with its bindings intact.

The first available operation moves a section (with all of its subsections)
between documents; see :mod:`sphinx.refactor.sectionmove`.
"""

from __future__ import annotations

from sphinx.refactor.sectionmove import (
    Diagnostic,
    FilePatch,
    MoveSectionResult,
    RefactorAborted,
    move_section,
)

__all__ = (
    'Diagnostic',
    'FilePatch',
    'MoveSectionResult',
    'RefactorAborted',
    'move_section',
)
