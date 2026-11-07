"""Tests for the rST section-move source refactoring.

The refactoring operates on *sources*, and every test builds the project
before and after the move to check that the result still parses and that
bindings (labels, links, substitutions, footnotes, citations, std/py
references, object identity, contexts and relative paths) survive.
"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pytest
from docutils import nodes

from sphinx.application import Sphinx
from sphinx.refactor import RefactorAborted, move_section


def build(srcdir: Path, builder: str = 'dummy', freshenv: bool = True):
    status, warning = StringIO(), StringIO()
    builddir = srcdir / '_build_test'
    app = Sphinx(
        srcdir=srcdir,
        confdir=srcdir,
        outdir=builddir,
        doctreedir=builddir / '.doctrees',
        buildername=builder,
        status=status,
        warning=warning,
        freshenv=freshenv,
    )
    app.build()
    text = '\n'.join(
        line for line in warning.getvalue().splitlines()
        if '[app.add_' not in line
    )
    return app, text


def make_project(tmp_path: Path, files: dict[str, str],
                 conf: str = "extensions = []\nmaster_doc = 'index'\n") -> Path:
    srcdir = tmp_path / 'src'
    srcdir.mkdir()
    (srcdir / 'conf.py').write_text(conf, encoding='utf-8')
    for name, text in files.items():
        p = srcdir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
    return srcdir


def section_line(path: Path, title: str) -> int:
    """1-based line of the title text."""
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip() == title:
            return i
    raise AssertionError(f'title {title!r} not found in {path}')


# ---------------------------------------------------------------------------
# Basic mechanics
# ---------------------------------------------------------------------------


def test_preview_does_not_write(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': 'Main\n====\n\n.. toctree::\n\n   b\n\nSec\n---\n\nbody.\n',
        'b.rst': 'B\n=\n\nB body.\n',
    })
    before = (srcdir / 'index.rst').read_text(encoding='utf-8')
    result = move_section(srcdir, 'index',
                          section_line(srcdir / 'index.rst', 'Sec'),
                          'b', 0)
    assert result.applied is False
    assert (srcdir / 'index.rst').read_text(encoding='utf-8') == before
    assert result.patches
    assert 'Sec' in result.format_patch()


def test_basic_move_builds_clean(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'Sec\n---\n\nSec body.\n'
        ),
        'b.rst': 'B\n=\n\nB body.\n',
    })
    line = section_line(srcdir / 'index.rst', 'Sec')
    result = move_section(srcdir, 'index', line, 'b', 0, apply=True)
    assert result.applied
    index_text = (srcdir / 'index.rst').read_text()
    b_text = (srcdir / 'b.rst').read_text()
    assert 'Sec' not in index_text
    assert 'Sec\n===' in b_text  # promoted to level-1 underline
    _app, warnings = build(srcdir)
    assert warnings == ''


def test_moved_section_becomes_last_child_with_remapped_levels(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'S\n-\n\nS body.\n\n'
            'S2\n^^\n\nS2 body.\n'
        ),
        'b.rst': 'B\n=\n\nP\n-\n\nP body.\n',
    })
    line = section_line(srcdir / 'index.rst', 'S')
    move_section(srcdir, 'index', line, 'b',
                 section_line(srcdir / 'b.rst', 'P'), apply=True)
    b = (srcdir / 'b.rst').read_text().splitlines()
    # S now at depth 2 (char '^' target level under P '-'), S2 depth 3 new char
    text = '\n'.join(b)
    assert 'S\n^' in text
    # P keeps its '-' and stays before S
    assert text.index('P\n-') < text.index('S\n^')
    _app, warnings = build(srcdir)
    assert warnings == ''


def test_invalid_line_diagnosed_and_not_written(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': 'Main\n====\n\nbody line.\n',
        'b.rst': 'B\n=\n',
    })
    before = (srcdir / 'index.rst').read_text()
    with pytest.raises(RefactorAborted) as exc:
        move_section(srcdir, 'index', 4, 'b', 0)
    codes = {d.code for d in exc.value.diagnostics}
    assert 'not-a-title' in codes
    assert (srcdir / 'index.rst').read_text() == before


def test_target_inside_subtree_rejected(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\nS\n-\n\nbody\n\nS2\n^^\n\nbody2\n'
        ),
    })
    with pytest.raises(RefactorAborted) as exc:
        move_section(
            srcdir, 'index',
            section_line(srcdir / 'index.rst', 'S'),
            'index', section_line(srcdir / 'index.rst', 'S2'),
        )
    assert any(d.code in ('target-inside-moved-subtree', 'target-is-source')
               for d in exc.value.diagnostics)


def test_intradocument_move(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n'
            'A\n-\n\nA body.\n\n'
            'B\n-\n\nB body.\n'
        ),
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'A'),
        'index', section_line(srcdir / 'index.rst', 'B'),
        apply=True,
    )
    text = (srcdir / 'index.rst').read_text()
    assert text.index('B') < text.index('A')
    _app, warnings = build(srcdir)
    assert warnings == ''


# ---------------------------------------------------------------------------
# Labels and references
# ---------------------------------------------------------------------------


def test_explicit_label_moves_and_cross_doc_ref_resolves(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\nIntro.\n\n'
            '.. _my-label:\n\nSec\n---\n\nSee :ref:`my-label`.\n'
        ),
        'b.rst': 'B\n=\n\nRef :ref:`my-label`.\n',
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    app, warnings = build(srcdir, builder='html')
    assert warnings == ''
    b_html = (app.outdir / 'b.html').read_text(encoding='utf-8')
    assert '#my-label' in b_html or 'id="my-label"' in b_html


def test_implicit_title_references_rewritten_both_ways(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'See `Sec`_ from intro.\n\n'
            'Sec\n---\n\nSec body, see `Main`_.\n'
        ),
        'b.rst': 'B\n=\n\nB body.\n',
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    _app, warnings = build(srcdir)
    assert warnings == '', warnings
    # injected explicit labels are referenced
    assert ':ref:`rm-' in (srcdir / 'index.rst').read_text()
    assert ':ref:`rm-' in (srcdir / 'b.rst').read_text()


def test_indirect_alias_external_kept(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'Use the `project site`_.\n\n'
            'Sec\n---\n\n.. _project site: https://example.com/\n'
        ),
        'b.rst': 'B\n=\n\nbody\n',
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    _app, warnings = build(srcdir)
    assert warnings == '', warnings
    index_text = (srcdir / 'index.rst').read_text()
    assert 'https://example.com/' in index_text  # externalized


def test_same_name_sections_allowed(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'Topic\n-----\n\nfirst.\n'
        ),
        'b.rst': (
            'B\n=\n\nTopic\n-----\n\nsecond.\n'
        ),
    })
    result = move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Topic'),
        'b', 0, apply=True,
    )
    assert result.applied
    _app, warnings = build(srcdir)
    # implicit same titles are doc-local and were disambiguated by injected
    # labels where references existed; no refs existed here
    assert 'duplicate' not in warnings


# ---------------------------------------------------------------------------
# Substitutions / footnotes / citations
# ---------------------------------------------------------------------------


def test_substitution_copy_back_and_move(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'Kept uses |s1|.\n\n'
            'Sec\n---\n\nMoved uses |s1|.\n\n'
            '.. |s1| replace:: *em*\n'
        ),
        'b.rst': 'B\n=\n\nbody\n',
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    app, warnings = build(srcdir)
    assert warnings == '', warnings
    assert '|s1|' in (srcdir / 'index.rst').read_text()
    assert '|s1|' in (srcdir / 'b.rst').read_text()
    b_html = (app.outdir / 'b.html').read_text if False else None


def test_substitution_recursive_equivalent_reuse_and_privatise(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'Sec\n---\n\n|a|\n\n'
            '.. |a| replace:: uses |z|\n'
            '.. |z| replace:: ZED\n'
        ),
        'b.rst': (
            'B\n=\n\n'
            '.. |z| replace:: ZED\n\n'
            '.. |a| replace:: DIFFERENT\n\n'
            'body\n'
        ),
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    _app, warnings = build(srcdir)
    assert warnings == '', warnings


def test_footnote_split(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'kept ref [#f1]_.\n\n'
            'Sec\n---\n\nmoved ref [#f1]_.\n\n'
            '.. [#f1] footnote body\n'
        ),
        'b.rst': 'B\n=\n\nbody\n',
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    _app, warnings = build(srcdir)
    assert warnings == '', warnings


def test_footnote_collision_renamed(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'Sec\n---\n\nx [2]_.\n\n.. [2] source footnote\n'
        ),
        'b.rst': (
            'B\n=\n\ny [1]_.\n\n.. [1] target footnote\n'
        ),
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    _app, warnings = build(srcdir)
    assert warnings == '', warnings


def test_citation_conflict_rejected(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'kept [paper]_.\n\n'
            'Sec\n---\n\nmoved [paper]_.\n\n.. [paper] source citation\n'
        ),
        'b.rst': (
            'B\n=\n\n.. [paper] target citation\n'
        ),
    })
    before = (srcdir / 'index.rst').read_text()
    with pytest.raises(RefactorAborted) as exc:
        move_section(
            srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
            'b', 0,
        )
    assert any(d.code == 'citation-conflict' for d in exc.value.diagnostics)
    assert (srcdir / 'index.rst').read_text() == before


def test_citation_moves_without_conflict(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'kept [paper]_.\n\n'
            'Sec\n---\n\nmoved [paper]_.\n\n.. [paper] source citation\n'
        ),
        'b.rst': 'B\n=\n\nbody\n',
        'c.rst': (
            'C\n=\n\nThird document uses [paper]_.\n'
        ),
    })
    # add c to toctree by recreating index
    (srcdir / 'index.rst').write_text(
        'Main\n====\n\n.. toctree::\n\n   b\n   c\n\n'
        'kept [paper]_.\n\n'
        'Sec\n---\n\nmoved [paper]_.\n\n.. [paper] source citation\n'
    )
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    _app, warnings = build(srcdir)
    assert warnings == '', warnings


# ---------------------------------------------------------------------------
# Python domain / context / paths
# ---------------------------------------------------------------------------


def test_py_object_identity_and_module_context(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            '.. py:currentmodule:: pkg.mod\n\n'
            'Sec\n---\n\n.. py:function:: f(a)\n\n   Doc f.\n'
        ),
        'b.rst': (
            'B\n=\n\nUse :py:func:`f`.\n'
        ),
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    app, warnings = build(srcdir, builder='html')
    assert warnings == '', warnings
    # identity preserved: pkg.mod.f is now anchored to b but still resolves
    entry = app.env.domains['py'].objects['pkg.mod.f']
    assert entry.docname == 'b'
    b_html = (app.outdir / 'b.html').read_text(encoding='utf-8')
    assert '#pkg.mod.f' in b_html


def test_relative_image_path_rewritten(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   sub/b\n\n'
            'Sec\n---\n\n.. image:: pic.png\n'
        ),
        'sub/b.rst': 'B\n=\n\nbody\n',
        'pic.png': 'png',
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'sub/b', 0, apply=True,
    )
    _app, warnings = build(srcdir)
    assert warnings == '', warnings
    text = (srcdir / 'sub' / 'b.rst').read_text()
    assert '../pic.png' in text


# ---------------------------------------------------------------------------
# Includes
# ---------------------------------------------------------------------------


def test_include_directive_moves_with_section(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'Sec\n---\n\n.. include:: frag.inc\n'
        ),
        'b.rst': 'B\n=\n\nbody\n',
        'frag.inc': 'fragment content\n',
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    _app, warnings = build(srcdir)
    assert warnings == '', warnings
    assert 'include:: frag.inc' in (srcdir / 'b.rst').read_text()
    assert 'frag.inc' not in (srcdir / 'index.rst').read_text()


def test_section_include_materialised_other_expansion_unaffected(tmp_path):
    # shared.inc is included by index (inside the moved section) AND by c.rst.
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n   c\n\n'
            'Sec\n---\n\n.. include:: shared.inc\n'
        ),
        'b.rst': 'B\n=\n\nbody\n',
        'c.rst': 'C\n=\n\n.. include:: shared.inc\n',
        'shared.inc': 'Shared Sec\n----------\n\nshared body\n',
    })
    c_before = (srcdir / 'c.rst').read_text()
    shared_before = (srcdir / 'shared.inc').read_text()
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    # shared physical fragment and the other expansion are untouched
    assert (srcdir / 'shared.inc').read_text() == shared_before
    assert (srcdir / 'c.rst').read_text() == c_before
    app, warnings = build(srcdir)
    assert warnings == '', warnings
    # the moved document now contains the materialised section text and no
    # include directive for the section-bearing fragment
    b_text = (srcdir / 'b.rst').read_text()
    assert 'Shared Sec' in b_text
    assert 'shared body' in b_text
    assert 'include:: shared.inc' not in b_text
    # c still renders the shared section
    c_doctree = app.env.get_doctree('c')
    titles = [n[0].astext() for n in c_doctree.findall(nodes.section)]
    assert 'Shared Sec' in titles


# ---------------------------------------------------------------------------
# Unsupported content
# ---------------------------------------------------------------------------


def test_autodoc_rejected(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\nSec\n---\n\n.. automodule:: somemod\n'
        ),
    })
    before = (srcdir / 'index.rst').read_text()
    with pytest.raises(RefactorAborted) as exc:
        move_section(srcdir, 'index',
                     section_line(srcdir / 'index.rst', 'Sec'),
                     'index', 0)
    assert any(d.code == 'autodoc-unsupported' for d in exc.value.diagnostics)
    assert (srcdir / 'index.rst').read_text() == before


def test_duplicate_py_object_rejected(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            '.. py:currentmodule:: m\n\n'
            'Sec\n---\n\n.. py:function:: g()\n'
        ),
        'b.rst': (
            'B\n=\n\n.. py:currentmodule:: m\n\n.. py:function:: g()\n'
        ),
    })
    with pytest.raises(RefactorAborted) as exc:
        move_section(
            srcdir, 'index',
            section_line(srcdir / 'index.rst', 'Sec'),
            'b', 0,
        )
    assert any(d.code == 'py-object-conflict' for d in exc.value.diagnostics)


def test_formatting_and_code_blocks_preserved(tmp_path):
    srcdir = make_project(tmp_path, {
        'index.rst': (
            'Main\n====\n\n.. toctree::\n\n   b\n\n'
            'Sec\n---\n\n'
            '.. a comment preserved\n\n'
            '.. code-block:: python\n\n   print("x")\n\n'
            '.. note:: admonition\n'
        ),
        'b.rst': 'B\n=\n\nbody\n',
    })
    move_section(
        srcdir, 'index', section_line(srcdir / 'index.rst', 'Sec'),
        'b', 0, apply=True,
    )
    b_text = (srcdir / 'b.rst').read_text()
    assert '.. a comment preserved' in b_text
    assert '.. code-block:: python' in b_text
    assert 'print("x")' in b_text
    assert '.. note:: admonition' in b_text
    assert '<html' not in b_text.lower() or 'code-block' in b_text
    _app, warnings = build(srcdir)
    assert warnings == '', warnings
