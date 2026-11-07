"""Planning/execution of a single section move.

This is an implementation detail of :mod:`sphinx.refactor.sectionmove`.
"""

from __future__ import annotations

import io
import os
import posixpath
import re
import shutil
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from docutils import nodes

from sphinx import addnodes
from sphinx.application import Sphinx

from sphinx.refactor.sectionmove import (
    _ANSI_RE,
    _WARNING_RE,
    Binding,
    Diagnostic,
    FilePatch,
    Issues,
    MoveSectionResult,
    RefactorAborted,
    Session,
    SLine,
    Token,
    _AUTODOC_RE,
    _block_extent,
    _CURRENTMODULE_RE,
    _DEFAULT_DOMAIN_RE,
    _DEFAULT_ROLE_RE,
    _DIRECTIVE_RE,
    _DOWNLOAD_ROLE_RE,
    _is_underline,
    _norm_name,
    _PREFERRED_CHARS,
    _PATH_DIRECTIVE_ARGS,
    _PATH_DIRECTIVE_OPTIONS,
    _ROLE_DIRECTIVE_RE,
    _SUPPORTED_DOMAINS,
)

if __import__('typing').TYPE_CHECKING:
    from typing import Any

    from sphinx.refactor.sectionmove import (
        Binding,
        DefBlock,
        DocModel,
        IncludeInstance,
        SectionInfo,
    )


# ---------------------------------------------------------------------------
# Rendered streams
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    text: str
    origin: Path
    oln: int
    tag: str = 'line'  # 'line' | 'directive' | 'synthetic'
    gidx: int | None = None


class Stream:
    """Mutable list of rendered lines with host-block rewrite support."""

    def __init__(self, cells: list[Cell] | None = None) -> None:
        self.cells: list[Cell] = cells or []

    def text(self) -> str:
        return ''.join(c.text for c in self.cells)

    def find_run(
        self, origin: Path, oln_start: int, oln_end: int, occurrence: int = 0
    ) -> tuple[int, int]:
        """Return [i,j) covering physical lines oln_start..oln_end."""
        starts = [
            idx for idx, c in enumerate(self.cells)
            if c.origin == origin and c.oln == oln_start
        ]
        if not starts:
            raise KeyError((origin, oln_start))
        i = starts[occurrence]
        j = i + 1
        expected = oln_start + 1
        while expected <= oln_end and j < len(self.cells):
            c = self.cells[j]
            if c.origin == origin and c.oln == expected:
                j += 1
                expected += 1
            elif not c.text.strip():
                j += 1  # tolerate injected blank lines
            else:
                break
        return i, j

    def rewrite_host(
        self,
        origin: Path,
        oln_start: int,
        oln_end: int,
        replacements: list[tuple[str, str]],
        occurrence: int = 0,
    ) -> None:
        i, j = self.find_run(origin, oln_start, oln_end, occurrence)
        cells = self.cells[i:j]
        text = ''.join(c.text for c in cells)
        for old, new in replacements:
            if old not in text:
                # tolerate whitespace-normalised rawsources spanning lines
                pattern = re.escape(old).replace(r'\ ', r'\s+')
                text, n = re.subn(pattern, new.replace('\\', r'\\'), text,
                                  count=1)
                if n == 0:
                    raise KeyError(f'rawsource not found: {old!r}')
            else:
                text = text.replace(old, new, 1)
        new_lines = text.splitlines(keepends=True)
        if text and not text.endswith(('\n', '\r')):
            new_lines.append('')
        self.cells[i:j] = [
            Cell(t, origin, cells[min(k, len(cells) - 1)].oln)
            for k, t in enumerate(new_lines)
        ]

    def rewrite_line(
        self, origin: Path, oln: int, old: str, new: str, occurrence: int = 0
    ) -> None:
        matches = [
            idx for idx, c in enumerate(self.cells)
            if c.origin == origin and c.oln == oln and old in c.text
        ]
        if not matches:
            raise KeyError((origin, oln, old))
        idx = matches[occurrence]
        self.cells[idx].text = self.cells[idx].text.replace(old, new, 1)

    def insert_before_line(
        self, origin: Path, oln: int, new_cells: list[Cell], occurrence: int = 0
    ) -> None:
        matches = [
            idx for idx, c in enumerate(self.cells)
            if c.origin == origin and c.oln == oln
        ]
        if not matches:
            raise KeyError((origin, oln))
        idx = matches[occurrence]
        self.cells[idx:idx] = new_cells

    def append_cells(self, new_cells: list[Cell]) -> None:
        self.cells.extend(new_cells)

    def prepend_cells(self, new_cells: list[Cell]) -> None:
        self.cells[0:0] = new_cells

    def origins_set(self) -> set[Path]:
        return {c.origin for c in self.cells}


def _cells_from_model_range(
    model: DocModel, i: int, j: int
) -> list[Cell]:
    return [
        Cell(sl.text, sl.origin, sl.oln)
        for sl in model.lines[i:j]
    ]


# ---------------------------------------------------------------------------
# Include item trees
# ---------------------------------------------------------------------------


@dataclass
class _Item:
    kind: str  # 'line' | 'include'
    index: int = -1
    cell: Cell | None = None
    inst: IncludeInstance | None = None
    directive_cells: list[Cell] = field(default_factory=list)
    children: list[_Item] = field(default_factory=list)


class _ItemTreeBuilder:
    def __init__(self, model: DocModel, file_lines: dict[Path, list[str]]) -> None:
        self.model = model
        self.file_lines = file_lines
        self.insts = list(model.instances.values())
        # only instances with materialised spans
        self.insts = [x for x in self.insts if x.span[0] >= 0]
        self.insts.sort(key=lambda x: x.span[0])

    def _directive_cells(self, inst: IncludeInstance) -> list[Cell]:
        phys = self.file_lines[inst.directive_origin][0]
        start = inst.directive_oln - 1
        end = _block_extent(
            [type('L', (), {'text': t})() for t in phys], start, len(phys)
        )
        return [
            Cell(text=phys[k], origin=inst.directive_origin,
                 oln=k + 1, tag='directive')
            for k in range(start, end)
        ]

    def build(self) -> list[_Item]:
        return self._build(0, len(self.model.lines), self.model.path)

    def _build(self, lo: int, hi: int, host: Path) -> list[_Item]:
        items: list[_Item] = []
        children = [
            x for x in self.insts
            if x.directive_origin == host and x.span[0] >= lo and x.span[1] <= hi
        ]
        children.sort(key=lambda x: x.span[0])
        # remove children nested inside another child
        top: list[IncludeInstance] = []
        for ch in children:
            if top and ch.span[0] < top[-1].span[1]:
                continue
            top.append(ch)
        i = lo
        for ch in top:
            while i < ch.span[0]:
                items.append(_Item(
                    'line', index=i,
                    cell=Cell(self.model.lines[i].text,
                              self.model.lines[i].origin,
                              self.model.lines[i].oln),
                ))
                i += 1
            sub_children = self._build(ch.span[0], ch.span[1], ch.target)
            items.append(_Item(
                'include',
                inst=ch,
                directive_cells=self._directive_cells(ch),
                children=sub_children,
            ))
            i = ch.span[1]
        while i < hi:
            sl = self.model.lines[i]
            items.append(_Item('line', index=i, cell=Cell(sl.text, sl.origin,
                                                         sl.oln)))
            i += 1
        return items


def _render_items(
    items: list[_Item],
    *,
    window: tuple[int, int] | None,
    policy: Callable[[IncludeInstance], str],
    index_shift: int = 0,
) -> list[Cell]:
    """Render items; only leaves with global index inside *window* emitted.

    ``policy(inst)`` returns 'directive' (collapse to the directive block) or
    'materialize' (emit expanded content).
    """
    out: list[Cell] = []
    wlo, whi = window if window is not None else (0, 10**18)

    def render_list(lst: list[_Item], shift: int) -> None:
        for item in lst:
            if item.kind == 'line':
                idx = item.index
                if wlo <= idx < whi:
                    assert item.cell is not None
                    item.cell.gidx = idx
                    out.append(item.cell)
            else:
                assert item.inst is not None
                a, b = item.inst.span
                overlap_lo = max(a, wlo)
                overlap_hi = min(b, whi)
                decision = policy(item.inst)
                if overlap_lo < overlap_hi and decision == 'materialize':
                    render_list(item.children, shift)
                elif wlo <= a < whi or overlap_lo < overlap_hi:
                    if decision == 'directive':
                        out.extend(item.directive_cells)

    render_list(items, index_shift)
    return out


# ---------------------------------------------------------------------------
# Helpers shared with planner
# ---------------------------------------------------------------------------


def _nodes_by_id(model: DocModel) -> dict[str, nodes.Element]:
    # Use the post-read tree: explicit targets are folded into their
    # elements there, and footnote/citation/object ids are assigned.
    result: dict[str, nodes.Element] = {}
    for node in model.read_tree.findall(nodes.Element):
        for nid in node.get('ids', []):
            result.setdefault(nid, node)
    return result


def _node_positions(model: DocModel) -> dict[int, int]:
    """Map id(node) -> expanded index (document order occurrence matching).

    Uses the post-read tree, whose physical source/line attribution is the
    same as the raw parse tree.
    """
    queues: dict[tuple[Path, int], list[int]] = defaultdict(list)
    for idx, sl in enumerate(model.lines):
        queues[(sl.origin, sl.oln)].append(idx)
    for key in queues:
        queues[key].reverse()
    positions: dict[int, int] = {}
    for node in model.read_tree.findall(nodes.Element):
        if node.source is None or node.line is None:
            continue
        q = queues.get((Path(node.source).resolve(), node.line))
        if q:
            positions[id(node)] = q.pop()
    return positions


def _raw_node_positions(model: DocModel) -> dict[int, int]:
    queues: dict[tuple[Path, int], list[int]] = defaultdict(list)
    for idx, sl in enumerate(model.lines):
        queues[(sl.origin, sl.oln)].append(idx)
    for key in queues:
        queues[key].reverse()
    positions: dict[int, int] = {}
    for node in model.raw.findall(nodes.Element):
        if node.source is None or node.line is None:
            continue
        q = queues.get((Path(node.source).resolve(), node.line))
        if q:
            positions[id(node)] = q.pop()
    return positions


def _section_ancestors_ok(section: nodes.section) -> bool:
    parent = section.parent
    while parent is not None:
        if not isinstance(parent, (nodes.section, nodes.document)):
            return False
        parent = parent.parent
    return True


def _physical_block(session: Session, origin: Path, oln: int) -> tuple[int, int]:
    phys = session.file_contents[origin][0]
    start = oln - 1
    end = _block_extent(
        [type('L', (), {'text': t})() for t in phys], start, len(phys)
    )
    return start, end


def _block_text(session: Session, origin: Path, oln: int) -> str:
    a, b = _physical_block(session, origin, oln)
    return ''.join(session.file_contents[origin][0][a:b])


def _norm_block(text: str) -> str:
    return ' '.join(line.strip() for line in text.splitlines() if line.strip())


_FOOTNOTE_LABEL_RE = re.compile(r'^(?P<indent>\s*)\.\.[ \t]+\[(?P<label>[^\]]+)\]')
_CITATION_LABEL_RE = re.compile(r'^(?P<indent>\s*)\.\.[ \t]+\[(?P<label>[^\]]+)\]')
_SUB_LABEL_RE = re.compile(r'^(?P<indent>\s*)\.\.[ \t]+\|(?P<name>[^|]+)\|')
_PATH_TOKEN_RE = re.compile(r'^(?P<prefix>(?P<indent>\s*)\.\.[ \t]+[\w:-]+::\s*)(?P<arg>.+?)\s*$')
_TOCTREE_RE = re.compile(r'^(?P<indent>\s*)\.\.[ \t]+toctree::')
_OPTION_LINE_RE = re.compile(r'^[ \t]+:([\w-]+):(?:[ \t]+(.*))?$')
_ABS_URI_RE = re.compile(r'^([a-zA-Z][a-zA-Z0-9+.-]*:)?#|^[a-zA-Z][a-zA-Z0-9+.-]*://|^mailto:|^\.\s*$')


def _is_external_ref(text: str) -> bool:
    if not text:
        return True
    if '://' in text or text.startswith(('#', 'mailto:', '/')):
        return True
    if text.startswith('<') and text.endswith('>'):
        return True
    return False


# ===========================================================================
# Planner
# ===========================================================================


class Planner:
    def __init__(
        self,
        session: Session,
        src_docname: str,
        src_title_line: int,
        tgt_docname: str,
        tgt_parent_line: int,
    ) -> None:
        self.session = session
        self.issues = session.issues
        self.app = session.app
        self.src_docname = src_docname
        self.tgt_docname = tgt_docname
        self.same_doc = src_docname == tgt_docname
        self.SM = session.model(src_docname)
        self.TM = self.SM if self.same_doc else session.model(tgt_docname)
        self.src_line_no = src_title_line
        self.tgt_line_no = tgt_parent_line

        self.S = self._select_section(self.SM, src_title_line, 'source')
        self.P = (
            self._select_section(self.TM, tgt_parent_line, 'target')
            if tgt_parent_line
            else None
        )
        self.c0 = self._cut_start()
        self.c1 = self.S.end
        self.np = self.S.node
        self.new_root_depth = 0 if self.P is None else self.P.depth + 1

        self.note = self.issues.note

        # resolved caches
        self._resolved_cache: dict[str, dict[tuple[Path, int], list[Any]]] = {}
        self._id_maps: dict[str, dict[str, nodes.Element]] = {}
        self._pos_maps: dict[str, dict[int, int]] = {}
        self._raw_pos_maps: dict[str, dict[int, int]] = {}

        # outputs of planning
        self.label_injections: dict[int, str] = {}  # section gidx -> label name
        self.fresh_labels: set[str] = set()
        self.renamed_xrefs: list[tuple[DocModel, Token, str]] = []
        self.externalize: list[tuple[DocModel, Token, str]] = []
        self.local_ref_rewrite: list[tuple[DocModel, Token, str, str | None]] = []
        self.sub_plans = []
        self.tail_blocks: list[str] = []  # blocks appended back to source
        self.moved_tail_blocks: list[str] = []
        self.context_prefix = ''
        self.context_suffix = ''
        self.context_restore = ''

    # -- selection ---------------------------------------------------------

    def _select_section(
        self, model: DocModel, line: int, where: str
    ) -> SectionInfo:
        path = model.path
        phys = self.session.file_contents[path][0]
        if line < 1 or line > len(phys):
            self.issues.err(
                'line-out-of-range',
                f'{where} title line {line} is outside '
                f'{os.path.relpath(path, self.session.srcdir)} '
                f'({len(phys)} lines)',
                path,
                min(line, len(phys)),
            )
            self.issues.raise_if_errors()
        text_line = phys[line - 1].rstrip('\r\n')
        underline = phys[line].rstrip('\r\n') if line < len(phys) else ''
        ch = _is_underline(underline)
        if ch is None or not text_line.strip() or len(underline.rstrip()) < len(text_line.strip()):
            self.issues.err(
                'not-a-title',
                f'{where} line {line} is not a section title text line '
                f'(expected a non-empty title immediately followed by an '
                f'underline): {text_line[:60]!r}',
                path,
                line,
            )
            self.issues.raise_if_errors()
        candidates = [
            s for s in model.sections
            if s.origin == path and s.title_oln == line
        ]
        if not candidates:
            self.issues.err(
                'title-not-parsed',
                f'{where} line {line} did not parse as a section in the '
                'analysis build',
                path,
                line,
            )
            self.issues.raise_if_errors()
        return candidates[0]

    def _cut_start(self) -> int:
        start = self.S.start
        if self.S.overline_oln is not None:
            idx = self.SM.pos_index.get(
                (self.S.origin, self.S.overline_oln), []
            )
            if idx and idx[0] == start - 1:
                start -= 1
        for oln in self.S.lead_olns:
            idxs = self.SM.pos_index.get((self.S.origin, oln), [])
            if idxs and idxs[0] < start:
                start = min(start, idxs[0])
        return start

    # -- positions ---------------------------------------------------------

    def posmap(self, model: DocModel) -> dict[int, int]:
        if model.docname not in self._pos_maps:
            self._pos_maps[model.docname] = _node_positions(model)
        return self._pos_maps[model.docname]

    def raw_posmap(self, model: DocModel) -> dict[int, int]:
        if model.docname not in self._raw_pos_maps:
            self._raw_pos_maps[model.docname] = _raw_node_positions(model)
        return self._raw_pos_maps[model.docname]

    def idmap(self, model: DocModel) -> dict[str, nodes.Element]:
        if model.docname not in self._id_maps:
            self._id_maps[model.docname] = _nodes_by_id(model)
        return self._id_maps[model.docname]

    def in_x(self, model: DocModel, node: nodes.Node) -> bool:
        if model is not self.SM:
            return False
        pos = self.posmap(model).get(id(node))
        if pos is None:
            pos = self.raw_posmap(model).get(id(node))
        return pos is not None and self.c0 <= pos < self.c1

    def gidx_in_x(self, gidx: int) -> bool:
        return self.c0 <= gidx < self.c1

    # -- resolved cross references ----------------------------------------

    def resolved_xrefs(
        self, model: DocModel
    ) -> dict[tuple[Path, int], list[Any]]:
        if model.docname in self._resolved_cache:
            return self._resolved_cache[model.docname]
        assert self.app is not None
        resolved = model.docname and self.app.env.get_and_resolve_doctree(
            model.docname, self.app.builder
        )
        mapping: dict[tuple[Path, int], list[Any]] = defaultdict(list)
        for host in resolved.findall(nodes.TextElement):
            if host.source is None or host.line is None:
                continue
            key = (Path(host.source).resolve(), host.line)
            for node in host.children:
                if isinstance(node, nodes.reference) and 'xref' in (
                    node.get('classes') or []
                ):
                    mapping[key].append(node)
                elif isinstance(node, addnodes.pending_xref):
                    mapping[key].append(node)
        self._resolved_cache[model.docname] = dict(mapping)
        return dict(mapping)

    # -- main entry --------------------------------------------------------

    def plan(self) -> dict[str, Any]:
        self._validate_structure()
        self._resolve_source_tokens()
        self._check_dynamic_content()
        self._check_custom_roles()
        label_plan = self._plan_global_labels()
        py_plan = self._plan_py_objects()
        streams = self._render_streams()
        self._remap_headings(streams)
        self._plan_context()
        self._plan_substitutions(streams)
        self._plan_footnotes(streams)
        self._plan_citations(streams)
        self._plan_local_references(streams)
        self._apply_xref_renames(streams, label_plan)
        self._rewrite_paths(streams)
        texts = self._assemble(streams)
        return {
            'texts': texts,
            'notes': self.issues.notes,
            'warnings': self.issues.warnings,
            'label_plan': label_plan,
            'py_plan': py_plan,
        }

    # -- structural validation --------------------------------------------

    def _validate_structure(self) -> None:
        if not _section_ancestors_ok(self.S.node):
            parent = self.S.node.parent
            self.issues.err(
                'unsupported-ancestor',
                'the source section is nested inside a non-section node '
                f'({type(parent).__name__}); only document and ordinary '
                'sections may be ancestors',
                self.S.origin,
                self.S.title_oln,
            )
        if self.same_doc:
            if self.P is not None:
                ppos = self.posmap(self.SM).get(id(self.P.node))
                if ppos is None:
                    ppos = self.raw_posmap(self.SM).get(id(self.P.node))
                if ppos is not None and self.c0 <= ppos < self.c1:
                    self.issues.err(
                        'target-inside-moved-subtree',
                        'the target parent section is inside the subtree being '
                        'moved',
                        self.TM.path,
                        self.P.title_oln,
                    )
            if self.P is self.S:
                self.issues.err(
                    'target-is-source',
                    'cannot move a section under itself',
                    self.SM.path,
                    self.S.title_oln,
                )
        # source and target must not introduce unsupported section levels
        # through unparseable titles -- checked by the validation build.
        self.issues.raise_if_errors()

    def _check_dynamic_content(self) -> None:
        # autodoc directives anywhere in the moved expanded text
        protected = self._protected_olns(self.SM)
        for gidx in range(self.c0, self.c1):
            sl = self.SM.lines[gidx]
            text = sl.text.rstrip('\r\n')
            if (sl.origin, sl.oln) in protected:
                continue
            if _AUTODOC_RE.match(text):
                self.issues.err(
                    'autodoc-unsupported',
                    'autodoc directives generate content dynamically and '
                    'cannot be moved; copy the generated directives or use '
                    'autosummary static tables instead',
                    sl.origin,
                    sl.oln,
                )
        # custom domains participating in the move
        for node in self.SM.raw.findall(addnodes.desc):
            if not self.in_x(self.SM, node):
                continue
            domain = node.get('domain')
            if domain not in _SUPPORTED_DOMAINS:
                sig = node.next_node(addnodes.desc_signature)
                self.issues.err(
                    'custom-domain-unsupported',
                    f'object description uses unsupported domain '
                    f'{domain!r}',
                    sig.source if sig else self.SM.path,
                    sig.line if sig else None,
                )
        for tok in self.SM.tokens:
            if tok.kind != 'xref':
                continue
            if not self.gidx_in_x(tok.host_index):
                continue
            domain = tok.node.get('refdomain')
            if domain not in _SUPPORTED_DOMAINS:
                self.issues.err(
                    'custom-domain-unsupported',
                    f'cross reference uses unsupported domain '
                    f'{domain!r}: {tok.rawsource.strip()}',
                    tok.host_origin,
                    tok.host_oln,
                )
        # third-party event hooks that can alter sources
        assert self.app is not None
        listeners = self.app.events.listeners
        for event_name in ('source-read', 'include-read', 'doctree-read'):
            for listener in listeners.get(event_name, []):
                module = getattr(listener.handler, '__module__', '') or ''
                if not module.startswith('sphinx.'):
                    self.issues.err(
                        'event-hook-unsupported',
                        f'a third-party handler for {event_name!r} '
                        f'({module}) may generate dynamic content',
                    )
        self.issues.raise_if_errors()

    def _protected_olns(self, model: DocModel) -> set[tuple[Path, int]]:
        result: set[tuple[Path, int]] = set()
        for node in model.raw.findall(nodes.Element):
            if not isinstance(node, (nodes.literal_block, nodes.doctest_block,
                                     nodes.math_block, nodes.comment)):
                continue
            if node.source is None or node.line is None:
                continue
            origin = Path(node.source).resolve()
            if origin not in self.session.file_contents:
                continue
            a, b = _physical_block(self.session, origin, node.line)
            for k in range(a, b):
                result.add((origin, k + 1))
        return result

    def _check_custom_roles(self) -> None:
        role_lines: list[tuple[int, Path, int, str]] = []
        for gidx, sl in enumerate(self.SM.lines):
            m = _ROLE_DIRECTIVE_RE.match(sl.text.rstrip('\r\n'))
            if m and not sl.text[:1].isspace():
                role_lines.append((gidx, sl.origin, sl.oln,
                                   m.group('arg').split('(')[0].strip()))
        if not role_lines:
            return
        protected = self._protected_olns(self.SM)
        for gidx, origin, oln, name in role_lines:
            inside = self.gidx_in_x(gidx)
            pattern = re.compile(rf':{re.escape(name)}:`')
            other_use = False
            for k, sl2 in enumerate(self.SM.lines):
                if (sl2.origin, sl2.oln) in protected:
                    continue
                if inside and self.gidx_in_x(k):
                    continue
                if not inside and not self.gidx_in_x(k):
                    continue
                if pattern.search(sl2.text):
                    other_use = True
                    break
            if other_use:
                self.issues.err(
                    'custom-role-scope',
                    f'custom role {name!r} is defined on line {oln} but used '
                    'on both sides of the move; move the ".. role::" '
                    'definition together with all its uses, or duplicate it '
                    'in both documents before retrying',
                    origin,
                    oln,
                )
        self.issues.raise_if_errors()

    # -- bindings ----------------------------------------------------------

    def _resolve_source_tokens(self) -> None:
        sm = self.SM
        names = sm.targets_by_name
        # anonymous pairing
        anon_refs = [
            t for t in sm.tokens
            if t.kind == 'reference' and not _ref_name(t.node)
        ]
        anon_targets = [
            n for n in sm.raw.findall(nodes.target)
            if not n.get('names') and (n.get('refuri') or n.get('refname'))
        ]
        anon_pairs = dict(zip([id(t.node) for t in anon_refs],
                              anon_targets, strict=False))
        # footnote auto pairing via read tree
        raw_foots = [
            t for t in sm.tokens if t.kind == 'footnote'
        ]
        read_foots = list(sm.read_tree.findall(nodes.footnote_reference))
        foot_ids: dict[int, str] = {}
        for tok, read in zip(raw_foots, read_foots, strict=False):
            if read.get('ids'):
                foot_ids[id(tok.node)] = read['ids'][0]
        # resolved xref alignment per host
        xref_resolved = self.resolved_xrefs(sm)
        xref_seen: dict[tuple[Path, int], int] = defaultdict(int)

        for tok in sm.tokens:
            node = tok.node
            if tok.kind == 'reference':
                key = _ref_name(node)
                target_node = anon_pairs.get(id(node))
                if key is None and target_node is not None:
                    tok.binding = self._binding_from_target(sm, target_node)
                elif key is not None:
                    tok.binding = self._resolve_named(sm, names, key)
                elif node.get('refuri'):
                    tok.binding = Binding('exturi', uri=node['refuri'])
            elif tok.kind == 'footnote':
                refname = node.get('refname')
                if refname:
                    block = (
                        sm.defs_by_name.get(refname, [None])[0]
                        or sm.defs_by_id.get(refname)
                    )
                    if block:
                        tok.binding = Binding(
                            'footnote', sm.docname, block.node,
                            block.origin, block.oln,
                            id=block.extra['ids'][0]
                            if block.extra['ids'] else refname,
                            name=block.name,
                        )
                else:
                    fid = foot_ids.get(id(node))
                    block = sm.defs_by_id.get(fid) if fid else None
                    if block:
                        tok.binding = Binding(
                            'footnote', sm.docname, block.node,
                            block.origin, block.oln, id=fid,
                            name=block.name,
                        )
            elif tok.kind == 'citation':
                cname = _node_name(node)
                block = sm.defs_by_name.get(cname, [None])[0]
                if block:
                    tok.binding = Binding(
                        'citation', sm.docname, block.node, block.origin,
                        block.oln, id=block.extra['ids'][0], name=block.name,
                    )
            elif tok.kind == 'substitution':
                blocks = sm.defs_by_name.get(_node_name(node), [])
                if blocks:
                    block = blocks[0]
                    tok.binding = Binding(
                        'substitution', sm.docname, block.node, block.origin,
                        block.oln, name=block.name,
                    )
            elif tok.kind == 'xref':
                tok.binding = self._resolve_xref(
                    sm, tok, xref_resolved, xref_seen
                )

    def _resolve_xref(
        self,
        model: DocModel,
        tok: Token,
        resolved: dict[tuple[Path, int], list[Any]],
        seen: dict[tuple[Path, int], int],
    ) -> Binding | None:
        assert self.app is not None
        node = tok.node
        domain = node.get('refdomain', '')
        reftype = node.get('reftype')
        target = node.get('reftarget', '')
        env = self.app.env
        key = (tok.host_origin, tok.host_oln)
        seq = resolved.get(key, [])
        idx = seen[key]
        seen[key] += 1
        resolved_node = seq[idx] if idx < len(seq) else None
        if isinstance(resolved_node, addnodes.pending_xref):
            resolved_node = None
        if isinstance(resolved_node, nodes.reference):
            refuri = resolved_node.get('refuri', '')
            if refuri and '://' in refuri:
                return Binding('exturi', uri=refuri)
            if refuri and '#' in refuri:
                doc, anchor = refuri.split('#', 1)
                return Binding('element', doc or model.docname,
                               id=anchor, name=target)
            if refuri and not resolved_node.get('internal', True):
                return Binding('exturi', uri=refuri)
            anchor = resolved_node.get('refid') or refuri.lstrip('#')
            if anchor:
                return Binding('element', model.docname, id=anchor,
                               name=target)
        # fall back to domain inventories
        if domain == 'std':
            labels = env.domains['std'].labels
            anon = env.domains['std'].anonlabels
            if reftype == 'doc':
                return Binding('doc', name=target)
            if target in labels:
                doc, anchor, _title = labels[target]
                return Binding('element', doc, id=anchor, name=target)
            if target in anon:
                doc, anchor = anon[target]
                return Binding('element', doc, id=anchor, name=target)
        if domain == 'py':
            entry = env.domains['py'].objects.get(target)
            if entry:
                return Binding('pyobj', entry.docname, id=entry.node_id,
                               name=target)
        return None

    def _resolve_named(
        self,
        model: DocModel,
        names: dict[str, nodes.Element],
        key: str,
        _seen: set[str] | None = None,
    ) -> Binding | None:
        _seen = _seen or set()
        norm = _norm_name(key)
        if norm in _seen:
            return None
        _seen.add(norm)
        node = names.get(norm)
        if node is None:
            return None
        if isinstance(node, nodes.target):
            if node.get('refname'):
                return self._resolve_named(model, names, node['refname'],
                                           _seen)
            if node.get('refuri'):
                return Binding('exturi', uri=node['refuri'])
            if node.get('refid'):
                target = self.idmap(model).get(node['refid'])
                if target is not None:
                    return Binding('element-local', model.docname, target,
                                   node.source and Path(node.source),
                                   node.line, id=node['refid'])
        kind = ('footnote' if isinstance(node, nodes.footnote)
                else 'citation' if isinstance(node, nodes.citation)
                else 'element')
        origin = Path(node.source) if node.source else None
        return Binding(kind if kind != 'element' else 'element-local',
                       model.docname, node, origin,
                       getattr(node, 'line', None),
                       id=(node.get('ids') or [None])[0], name=key)

    def _binding_from_target(
        self, model: DocModel, target: nodes.target
    ) -> Binding | None:
        if target.get('refname'):
            return self._resolve_named(model, model.targets_by_name,
                                       target['refname'])
        if target.get('refuri'):
            return Binding('exturi', uri=target['refuri'])
        return None

    # -- global labels / moving identity -----------------------------------

    def _plan_global_labels(self) -> dict[str, Any]:
        assert self.app is not None
        env = self.app.env
        std = env.domains['std']
        labels = std.labels
        moved_ids: set[str] = set()
        moved_labels: dict[str, tuple[str, str]] = {}  # name -> (doc,id)
        sm = self.SM
        tm = self.TM
        sp = self.posmap(sm)
        for name, entry in labels.items():
            doc, anchor, _title = entry
            if doc != sm.docname:
                continue
            node = self.idmap(sm).get(anchor)
            if node is None:
                continue
            pos = sp.get(id(node))
            if pos is None:
                # id may live on a signode child
                continue
            if self.c0 <= pos < self.c1:
                moved_labels[name] = (doc, anchor)
                moved_ids.add(anchor)
        target_anchored = {
            name for name, (doc, _a, _t) in labels.items()
            if doc == tm.docname
        }
        rename_map: dict[str, str] = {}
        # model all docs lazily for consumer analysis
        all_docnames = list(env.all_docs)
        for name, (doc, anchor) in sorted(moved_labels.items()):
            collision = name in target_anchored
            prefixed = ':' in name and self.SM.sectionlabel_prefix
            third_consumers = []
            inscope_consumers: list[tuple[DocModel, Token]] = []
            for dn in all_docnames:
                dm = sm if dn == sm.docname else (
                    tm if dn == tm.docname else self.session.model(dn)
                )
                for tok in dm.tokens:
                    if tok.kind != 'xref':
                        continue
                    if self._xref_targets(tok, doc, anchor):
                        if dm is sm or dm is tm:
                            inscope_consumers.append((dm, tok))
                        else:
                            third_consumers.append((dn, tok))
            if collision:
                if third_consumers:
                    for dn, tok in third_consumers[:5]:
                        self.issues.err(
                            'label-conflict-external',
                            f'label {name!r} is referenced from {dn} but '
                            'already exists in the target document; rename '
                            'one of the labels before retrying',
                            tok.host_origin,
                            tok.host_oln,
                        )
                    self.issues.raise_if_errors()
                fresh = self._fresh_label(name)
                rename_map[name] = fresh
                self.note(f'injected explicit target {fresh!r} for moved '
                          f'label {name!r} (name already used in target)')
                for dm, tok in inscope_consumers:
                    self.renamed_xrefs.append((dm, tok, fresh))
            elif prefixed:
                # srcdoc:title -> tgtdoc:title
                new_name = self._prefixed_name(tm.docname, name)
                rename_map[name] = new_name
                if third_consumers:
                    for dn, tok in third_consumers[:5]:
                        self.issues.err(
                            'autosectionlabel-external',
                            f'prefixed label {name!r} is referenced from '
                            f'{dn}; move or update that reference first',
                            tok.host_origin,
                            tok.host_oln,
                        )
                    self.issues.raise_if_errors()
                for dm, tok in inscope_consumers:
                    self.renamed_xrefs.append((dm, tok, new_name))
                self.note(f'updated prefixed section label {name!r} -> '
                          f'{new_name!r}')
        # non-section object labels colliding in target -> hard error
        for name, (_doc, anchor) in moved_labels.items():
            node = self.idmap(sm).get(anchor)
            if node is not None and not isinstance(
                node, (nodes.section, nodes.citation)
            ) and name in target_anchored:
                self.issues.err(
                    'object-label-conflict',
                    f'object label {name!r} already exists in the target '
                    'document; rename one object before retrying',
                    getattr(node, 'source', None),
                    getattr(node, 'line', None),
                )
        self.issues.raise_if_errors()
        # remember injections per moved section element
        for name, fresh in rename_map.items():
            anchor = moved_labels[name][1]
            node = self.idmap(sm).get(anchor)
            if isinstance(node, nodes.section):
                self.label_injections[sp[id(node)]] = fresh
        self.rename_map = rename_map
        return {'moved_labels': moved_labels, 'rename_map': rename_map}

    def _prefixed_name(self, tgt_docname: str, name: str) -> str:
        if ':' in name:
            _src, title = name.split(':', 1)
            return f'{tgt_docname}:{title}'
        return name

    def _xref_targets(
        self, tok: Token, doc: str, anchor: str
    ) -> bool:
        node = tok.node
        if tok.kind != 'xref':
            return False
        b = tok.binding
        if b is None:
            return False
        if b.kind in ('element',) and b.docname == doc and b.id == anchor:
            return True
        # compare against pending xref inventory directly
        if node.get('reftarget') == anchor:
            return True
        return False

    def _fresh_label(self, base: str) -> str:
        assert self.app is not None
        existing = set(self.app.env.domains['std'].labels) | self.fresh_labels
        slug = nodes.make_id(base) or 'label'
        candidate = f'rm-{slug}'
        n = 1
        while candidate in existing:
            n += 1
            candidate = f'rm-{slug}-{n}'
        self.fresh_labels.add(candidate)
        return candidate

    # -- python domain objects --------------------------------------------

    def _plan_py_objects(self) -> dict[str, Any]:
        assert self.app is not None
        py = self.app.env.domains['py']
        id2full: dict[str, str] = {}
        for fullname, entry in py.objects.items():
            id2full.setdefault(entry.node_id, fullname)
        moved_full: dict[str, nodes.Element] = {}
        sm = self.SM
        sp = self.raw_posmap(sm)
        for desc in sm.raw.findall(addnodes.desc):
            if desc.get('domain') != 'py':
                continue
            pos = sp.get(id(desc))
            if pos is None or not self.c0 <= pos < self.c1:
                continue
            for sig in desc.findall(addnodes.desc_signature):
                for sid in sig.get('ids', []):
                    full = id2full.get(sid)
                    if full:
                        moved_full[full] = desc
        if moved_full and not self.same_doc:
            tm = self.TM
            tp = self.posmap(tm)
            target_full: set[str] = set()
            for desc in tm.raw.findall(addnodes.desc):
                if desc.get('domain') != 'py':
                    continue
                for sig in desc.findall(addnodes.desc_signature):
                    for sid in sig.get('ids', []):
                        full = id2full.get(sid)
                        if full:
                            target_full.add(full)
            for full in sorted(moved_full):
                if full in target_full:
                    desc = moved_full[full]
                    self.issues.err(
                        'py-object-conflict',
                        f'Python object {full!r} is already defined in the '
                        'target document; moving would create a duplicate '
                        'definition',
                        desc.source,
                        desc.line,
                    )
        self.issues.raise_if_errors()
        return {'moved_fullnames': sorted(moved_full)}

    # -- rendering ---------------------------------------------------------

    def _section_bearing(self, inst: Any) -> bool:
        a, b = inst.span
        return any(a <= s.start < b for s in self.SM.sections)

    def _render_streams(self) -> dict[str, Stream]:
        sm, tm = self.SM, self.TM
        file_lines = self.session.file_contents
        src_builder = _ItemTreeBuilder(sm, file_lines)
        src_items = src_builder.build()
        cut = self.c1

        def moved_policy(inst: Any) -> str:
            a, b = inst.span
            if not (a < self.c1 and b > self.c0):
                return 'directive'
            if self._section_bearing(inst) or (a < cut < b):
                return 'materialize'
            return 'directive'

        def kept_policy(inst: Any) -> str:
            a, b = inst.span
            if self.c0 <= a and b <= self.c1:
                return 'skip'
            if a < cut < b:
                return 'materialize'
            return 'directive'

        moved_cells = _render_items(
            src_items, window=(self.c0, self.c1), policy=moved_policy
        )
        kept_before = _render_items(
            src_items, window=(0, self.c0), policy=kept_policy
        )
        kept_after = _render_items(
            src_items, window=(self.c1, len(sm.lines)), policy=kept_policy
        )
        streams = {
            'moved': Stream(moved_cells),
            'before': Stream(kept_before),
            'after': Stream(kept_after),
        }
        return streams

    # -- heading remap -----------------------------------------------------

    def _remap_headings(self, streams: dict[str, Stream]) -> None:
        moved = streams['moved']
        tm = self.TM
        used = {s.char for s in tm.sections}
        # existing depths in target excluding, for same-doc moves, the moved
        # subtree itself
        base: dict[int, str] = {}
        for sec in tm.sections:
            if self.same_doc and self.c0 <= sec.start < self.c1:
                continue
            base.setdefault(sec.depth, sec.char)
            used.add(sec.char)
        pool = [c for c in _PREFERRED_CHARS if c not in used]

        def char_for(depth: int) -> str:
            if depth in base and base[depth]:
                return base[depth]
            for d in range(len(base), depth + 1):
                if not pool:
                    self.issues.err(
                        'heading-chars-exhausted',
                        'no distinct underline characters left for the new '
                        'section depth',
                    )
                    self.issues.raise_if_errors()
                ch = pool.pop(0)
                base[d] = ch
                used.add(ch)
            return base[depth]

        moved_sections = [
            s for s in self.SM.sections
            if self.c0 <= s.start < self.c1
        ]
        for sec in moved_sections:
            new_depth = self.new_root_depth + (sec.depth - self.S.depth)
            new_char = char_for(new_depth)
            if new_char == sec.char:
                continue
            for cell in moved.cells:
                if cell.origin == sec.origin and cell.oln == sec.under_oln:
                    cell.text = self._reunderline(cell.text, new_char)
                if (sec.overline_oln is not None
                        and cell.origin == sec.origin
                        and cell.oln == sec.overline_oln):
                    cell.text = self._reunderline(cell.text, new_char)
            self.note(f'changed heading underline of {self._title(sec)!r} '
                      f'{sec.char!r} -> {new_char!r}')

    @staticmethod
    def _reunderline(line: str, char: str) -> str:
        nl = '\r\n' if line.endswith('\r\n') else '\n'
        body = line.rstrip('\r\n')
        return char * max(1, len(body)) + nl

    @staticmethod
    def _title(sec: SectionInfo) -> str:
        return sec.node[0].astext() if len(sec.node) else ''

    # -- context (currentmodule / default-domain / default-role) ----------

    def _state_at(self, model: DocModel, index: int) -> dict[str, str | None]:
        state = {'module': None, 'domain': 'rst',
                 'role': self.app.config.default_role or 'title-reference'}
        for i, key, value in model.contexts:
            if i >= index:
                break
            state[key] = value
        return state

    def _plan_context(self) -> None:
        sm, tm = self.SM, self.TM
        src_before = self._state_at(sm, self.c0)
        src_after = self._state_at(sm, self.c1)
        if self.P is None:
            tgt_index = len(tm.lines)
        else:
            tgt_index = self.P.start
        tgt_state = self._state_at(tm, tgt_index)
        self.context_prefix = self._context_lines(
            [('domain', src_before['domain']),
             ('module', src_before['module']),
             ('role', src_before['role'])],
            tgt_state,
        )
        self.context_suffix = self._context_lines(
            [('domain', tgt_state['domain']),
             ('module', tgt_state['module']),
             ('role', tgt_state['role'])],
            src_after,
            invert=True,
        )
        # restore state for the source document after the removed block
        self.context_restore = self._context_lines(
            [('domain', src_before['domain']),
             ('module', src_before['module']),
             ('role', src_before['role'])],
            src_after,
            invert=True,
        )

    @staticmethod
    def _context_lines(
        order: list[tuple[str, str | None]],
        reference: dict[str, str | None],
        *,
        invert: bool = False,
    ) -> str:
        # emit directives that change state away from reference (invert=True:
        # emit directives setting reference values where current differs)
        lines: list[str] = []
        for key, value in order:
            ref = reference[key]
            diff = value != ref
            if invert:
                # suffix/restore: set BACK to reference
                if value != ref:
                    lines.append(_context_directive(key, ref))
            else:
                if diff:
                    lines.append(_context_directive(key, value))
        if not lines:
            return ''
        return '\n'.join(lines) + '\n\n'

    # -----------------------------------------------------------------
    # substitutions
    # -----------------------------------------------------------------

    def _sub_closure(
        self, sm: DocModel, name: str, seen: set[str] | None = None
    ) -> list[str]:
        seen = seen if seen is not None else set()
        if name in seen:
            return []
        seen.add(name)
        out = [name]
        blocks = sm.defs_by_name.get(name, [])
        if not blocks:
            return out
        for inner in blocks[0].node.findall(nodes.substitution_reference):
            for n in self._sub_closure(sm, _node_name(inner), seen):
                if n not in out:
                    out.append(n)
        return out

    def _plan_substitutions(self, streams: dict[str, Stream]) -> None:
        sm, tm = self.SM, self.TM
        moved = streams['moved']
        x_tokens = [t for t in sm.tokens
                    if t.kind == 'substitution' and self.gidx_in_x(t.host_index)]
        kept_tokens = [t for t in sm.tokens
                       if t.kind == 'substitution'
                       and not self.gidx_in_x(t.host_index)]
        target_names = {
            name for name, blocks in tm.defs_by_name.items()
            if any(b.kind == 'substitution' for b in blocks)
        }
        used_new: set[str] = set()
        delete_runs: list[tuple[Path, int, int]] = []
        moved_deps: list[tuple[str, str]] = []  # name, block text (possibly renamed)
        copy_back: list[str] = []
        token_renames: dict[int, tuple[str, str]] = {}  # id(token) -> old,new

        def fresh_sub(name: str) -> str:
            base = nodes.make_id(name) or 'sub'
            candidate = f'{base}-moved'
            n = 1
            while candidate in target_names or candidate in used_new:
                n += 1
                candidate = f'{base}-moved-{n}'
            used_new.add(candidate)
            return candidate

        def block_in_x(block) -> bool:
            return self.c0 <= block.extra['expanded_start'] < self.c1

        # names needed inside the moved block
        needed: list[str] = []
        for tok in x_tokens:
            nm = _node_name(tok.node)
            if nm not in needed:
                needed.append(nm)
        # moved definitions survive (even unreferenced ones move with text)
        moved_def_blocks = [
            b for blocks in sm.defs_by_name.values() for b in blocks
            if b.kind == 'substitution' and block_in_x(b)
        ]
        target_def_text = {
            name: _norm_block(bl[0].text) for name, bl in tm.defs_by_name.items()
            if bl and bl[0].kind == 'substitution'
        }

        # renames for moved definitions colliding in target
        moved_renames: dict[str, str] = {}
        line_renames: list[tuple[Path, int, str, str]] = []
        for block in moved_def_blocks:
            name = block.name
            if name in target_names:
                if _norm_block(block.text) == target_def_text.get(name):
                    # equivalent definition already present: drop moved def
                    delete_runs.append(
                        (block.origin, block.oln, block.end_oln)
                    )
                    self.note(f'reused equivalent substitution |{name}| '
                              'already defined in target')
                else:
                    new = fresh_sub(name)
                    moved_renames[name] = new
                    line_renames.append(
                        (block.origin, block.oln,
                         f'|{name}|', f'|{new}|')
                    )
                    self.note(f'privatised moved substitution |{name}| as '
                              f'|{new}|')
        # tokens: rename where required
        for tok in x_tokens:
            name = _node_name(tok.node)
            blocks = sm.defs_by_name.get(name, [])
            in_x_block = next((b for b in blocks if block_in_x(b)), None)
            if in_x_block is not None:
                if name in moved_renames:
                    token_renames[id(tok)] = (f'|{name}|',
                                             f'|{moved_renames[name]}|')
                continue
            # definition stays in source: copy closure into moved deps
            for cn in self._sub_closure(sm, name):
                cblocks = sm.defs_by_name.get(cn, [])
                cb = next((b for b in cblocks if not block_in_x(b)), None)
                if cb is None:
                    continue
                if cn in target_names and _norm_block(
                    cb.text
                ) == target_def_text.get(cn):
                    continue
                new = moved_renames.get(cn)
                if new is None and cn in target_names:
                    new = fresh_sub(cn)
                    moved_renames[cn] = new
                if new is not None:
                    token_renames[id(tok)] = (
                        f'|{name}|', f'|{moved_renames.get(name, new)}|'
                    )
                text = self._block_renamed(cb, new) if new else cb.text
                if not any(n == cn for n, _t in moved_deps):
                    moved_deps.append((cn, text))
        # rename *all* in-X tokens for renamed defs (including those that
        # reference moved defs)
        for tok in x_tokens:
            if id(tok) in token_renames:
                continue
            name = _node_name(tok.node)
            if name in moved_renames:
                token_renames[id(tok)] = (
                    f'|{name}|', f'|{moved_renames[name]}|'
                )
        # kept tokens whose definition moved away: copy back
        kept_needs: list[str] = []
        for tok in kept_tokens:
            name = _node_name(tok.node)
            blocks = sm.defs_by_name.get(name, [])
            moved_block = next((b for b in blocks if block_in_x(b)), None)
            if moved_block is None:
                continue
            # still another definition remaining?
            if any(not block_in_x(b) for b in blocks):
                continue
            for cn in self._sub_closure(sm, name):
                if cn not in kept_needs:
                    kept_needs.append(cn)
        for cn in kept_needs:
            blocks = sm.defs_by_name.get(cn, [])
            block = next((b for b in blocks if block_in_x(b)), None)
            if block is not None:
                copy_back.append(block.text)
                self.note(f'copied back substitution |{cn}| definition for '
                          'remaining references')
        # apply
        for origin, oln, end_oln in delete_runs:
            self._delete_run(streams, origin, oln, end_oln, self.c0, self.c1)
        self._apply_token_renames(streams, token_renames)
        if moved_deps:
            block_text = '\n' + '\n'.join(t.rstrip('\n') for _n, t in moved_deps)
            self.moved_tail_blocks.append(block_text + '\n')
        if copy_back:
            block_text = '\n'.join(t.rstrip('\n') for t in copy_back)
            self.tail_blocks.append(block_text + '\n')

    def _block_renamed(self, block, new_name: str) -> str:
        phys = self.session.file_contents[block.origin][0]
        lines = list(phys[block.oln - 1:block.end_oln])
        if block.kind == 'substitution':
            lines[0] = lines[0].replace(f'|{block.name}|', f'|{new_name}|', 1)
        elif block.kind in ('footnote', 'citation'):
            lines[0] = re.sub(
                r'^(\s*\.\.\s+)\[[^\]]+\]',
                lambda m: m.group(1) + f'[{new_name}]',
                lines[0],
                count=1,
            )
        return ''.join(lines)

    def _defs_in_x(self, kind: str) -> set[str]:
        return {
            b.name
            for blocks in self.SM.defs_by_name.values()
            for b in blocks
            if b.kind == kind
            and self.c0 <= b.extra['expanded_start'] < self.c1
        }

    # -----------------------------------------------------------------
    # footnotes
    # -----------------------------------------------------------------

    def _plan_footnotes(self, streams: dict[str, Stream]) -> None:
        sm, tm = self.SM, self.TM
        moved = streams['moved']
        target_labels = {
            b.name for blocks in tm.defs_by_name.values() for b in blocks
            if b.kind == 'footnote'
        }
        used: set[str] = set()
        token_renames: dict[int, tuple[str, str]] = {}
        line_renames: list[tuple[Path, int, str, str]] = []
        moved_deps: list[str] = []
        copy_back: list[str] = []

        def free_label(old: str) -> str:
            if old.startswith('#') and old != '#':
                base = old[1:]
                candidate = f'#{base}-moved'
            elif old.isdigit():
                taken = {x for x in target_labels | used if x.isdigit()}
                taken.add(old)
                nums = [int(n) for n in taken]
                return str(max(nums + [0]) + 1)
            else:
                candidate = f'{old}-moved'
            n = 1
            base_candidate = candidate
            while candidate.lstrip('#') in {x.lstrip('#') for x in
                                             target_labels | used}:
                n += 1
                candidate = f'{base_candidate}-{n}'
            return candidate

        def label_of(block) -> str:
            first = self.session.file_contents[block.origin][0][block.oln - 1]
            m = _FOOTNOTE_LABEL_RE.match(first)
            return m.group('label') if m else block.name

        def in_x_block(b) -> bool:
            return self.c0 <= b.extra['expanded_start'] < self.c1

        defs = [
            b for blocks in sm.defs_by_name.values() for b in blocks
            if b.kind == 'footnote'
        ]
        for block in defs:
            label = label_of(block)
            refs = [
                t for t in sm.tokens if t.kind == 'footnote'
                and t.binding is not None and t.binding.id in block.extra['ids']
            ]
            x_refs = [t for t in refs if self.gidx_in_x(t.host_index)]
            kept_refs = [t for t in refs if not self.gidx_in_x(t.host_index)]
            moved_block = in_x_block(block)
            needs_rename = False
            new_label: str | None = None
            if moved_block:
                # moves with the block; rename on collision with target
                if label.lstrip('#') in {x.lstrip('#') for x in target_labels}:
                    new_label = free_label(label)
                    needs_rename = True
                if label == '#' and (x_refs or kept_refs):
                    new_label = new_label or f'#rm{len(used) + 1}'
                    needs_rename = True
                if needs_rename:
                    used.add(new_label)
                    line_renames.append(
                        (block.origin, block.oln,
                         f'[{label}]', f'[{new_label}]')
                    )
                    for t in x_refs:
                        token_renames[id(t)] = (
                            t.rawsource, new_ref_token(t, new_label)
                        )
                    self.note(f'renamed footnote [{label}] -> [{new_label}] '
                              'in moved content')
                if kept_refs:
                    copy_back.append(block.text)
                    self.note('copied back footnote definition for remaining '
                              'references')
            else:
                # definition stays; references inside moved block need a copy
                if x_refs:
                    if label in target_labels:
                        new_label = free_label(label)
                        used.add(new_label)
                        text = self._block_renamed(block, new_label)
                    else:
                        text = block.text
                    moved_deps.append(text)
                    for t in x_refs:
                        token_renames[id(t)] = (
                            t.rawsource,
                            new_ref_token(t, new_label)
                            if new_label else t.rawsource,
                        )
                    self.note('copied footnote definition into moved section')
        # apply
        self._apply_token_renames(streams, token_renames)
        for origin, oln, old, new in line_renames:
            for stream_name in ('moved',):
                try:
                    streams[stream_name].rewrite_line(
                        origin, oln, old, new
                    )
                except KeyError:
                    pass
        if moved_deps:
            self.moved_tail_blocks.append(
                '\n' + '\n'.join(t.rstrip('\n') for t in moved_deps) + '\n'
            )
        if copy_back:
            self.tail_blocks.append(
                '\n'.join(t.rstrip('\n') for t in copy_back) + '\n'
            )

    # -----------------------------------------------------------------
    # citations (global labels)
    # -----------------------------------------------------------------

    def _plan_citations(self, streams: dict[str, Stream]) -> None:
        sm, tm = self.SM, self.TM
        target = {
            b.name: b for blocks in tm.defs_by_name.values() for b in blocks
            if b.kind == 'citation'
        }
        token_renames: dict[int, tuple[str, str]] = {}
        moved_deps: list[str] = []
        source_copybacks: list[str] = []
        used: set[str] = set()

        defs = [
            b for blocks in sm.defs_by_name.values() for b in blocks
            if b.kind == 'citation'
        ]

        def in_x_block(b) -> bool:
            return self.c0 <= b.extra['expanded_start'] < self.c1

        def fresh(base: str) -> str:
            candidate = f'{base}-moved'
            n = 1
            while candidate in target or candidate in used:
                n += 1
                candidate = f'{base}-moved-{n}'
            used.add(candidate)
            return candidate

        def citation_consumers_elsewhere(name: str) -> list[str]:
            docs = []
            assert self.app is not None
            anon = self.app.env.domains['std'].anonlabels
            resolved_doc, resolved_id = anon.get(name, (None, None))
            if resolved_doc != sm.docname:
                return docs
            for dn in self.app.env.all_docs:
                if dn in (sm.docname, tm.docname):
                    continue
                dm = self.session.model(dn)
                for tok in dm.tokens:
                    if tok.kind == 'citation' and _node_name(tok.node) == name:
                        docs.append(dn)
                        break
            return docs

        for block in defs:
            refs = [
                t for t in sm.tokens if t.kind == 'citation'
                and t.binding is not None and t.binding.name == block.name
            ]
            x_refs = [t for t in refs if self.gidx_in_x(t.host_index)]
            kept_refs = [t for t in refs
                         if not self.gidx_in_x(t.host_index)]
            if in_x_block(block):
                other = target.get(block.name)
                if other is None:
                    continue
                if _norm_block(other.text) == _norm_block(block.text):
                    continue
                # Same global citation name with a different body already
                # exists in the target: ambiguous before the move and cannot
                # be disambiguated mechanically without changing bindings.
                self.issues.err(
                    'citation-conflict',
                    f'citation [{block.name}] is already defined with '
                    'different content in the target document; rename one '
                    'of the citations before retrying',
                    block.origin,
                    block.oln,
                )
            elif x_refs and block.name in target and _norm_block(
                block.text
            ) != _norm_block(target[block.name].text):
                self.issues.err(
                    'citation-conflict',
                    f'citation [{block.name}] referenced from the moved '
                    'section is already defined with different content in '
                    'the target document; rename one citation before '
                    'retrying',
                    block.origin,
                    block.oln,
                )
        self.issues.raise_if_errors()
        self._apply_token_renames(streams, token_renames)
        if moved_deps:
            self.moved_tail_blocks.append(
                '\n' + '\n'.join(t.rstrip('\n') for t in moved_deps) + '\n'
            )
        if source_copybacks:
            self.tail_blocks.append(
                '\n'.join(t.rstrip('\n') for t in source_copybacks) + '\n'
            )

    # -----------------------------------------------------------------
    # local docutils references
    # -----------------------------------------------------------------

    def _plan_local_references(self, streams: dict[str, Stream]) -> None:
        sm = self.SM
        token_renames: dict[int, str] = {}
        injected: dict[tuple[str, int], str] = {}

        def section_endpoint(binding) -> nodes.section | None:
            node = binding.node
            if isinstance(node, nodes.section):
                return node
            anc = node
            while anc is not None:
                if isinstance(anc, nodes.section):
                    return anc
                anc = anc.parent
            return None

        for tok in sm.tokens:
            if tok.kind != 'reference' or tok.binding is None:
                continue
            b = tok.binding
            tok_in_x = self.gidx_in_x(tok.host_index)
            if b.kind == 'exturi':
                display = _reference_display(tok)
                token_renames[id(tok)] = f'`{display} <{b.uri}>`_'
                continue
            if b.kind in ('footnote', 'citation', 'substitution'):
                continue
            if b.kind not in ('element-local',):
                continue
            section = section_endpoint(b)
            if section is None:
                self.issues.err(
                    'unsupported-reference-target',
                    'reference target is not a section; add an explicit '
                    'global target manually before retrying: '
                    f'{tok.rawsource.strip()}',
                    tok.host_origin,
                    tok.host_oln,
                )
                continue
            ep_in_x = self.in_x(sm, section)
            if tok_in_x == ep_in_x:
                continue  # moves together / stays together
            info = next((s for s in sm.sections if s.node is section), None)
            label = self._section_explicit_name(section)
            if info is not None:
                implicit = nodes.make_id(
                    info.node[0].astext() if len(info.node) else ''
                )
                explicit = [n for n in info.names if n != implicit]
                label = explicit[0] if explicit else None
            if label is None:
                label = self._fresh_label(
                    section[0].astext() if len(section) else 'section'
                )
                key = ('x' if ep_in_x else 'k'), id(section)
                injected[key] = label
            display = _reference_display(tok)
            new = (f':ref:`{display} <{label}>`' if display
                   and display != label else f':ref:`{label}`')
            token_renames[id(tok)] = new
        self.issues.raise_if_errors()
        # resolve sections by walking SM sections
        id_to_sec = {id(s.node): s for s in sm.sections}
        for (side, node_id), label in injected.items():
            sec = id_to_sec.get(node_id)
            if sec is None:
                continue
            block = f'.. _{label}:\n\n'
            if side == 'x':
                streams['moved'].insert_before_line(
                    sec.origin, sec.title_oln,
                    [Cell(line, sec.origin, 0, 'synthetic')
                     for line in (f'.. _{label}:\n', '\n')]
                )
            else:
                target_stream = (streams['before']
                                 if sec.start < self.c0 else streams['after'])
                target_stream.insert_before_line(
                    sec.origin, sec.title_oln,
                    [Cell(line, sec.origin, 0, 'synthetic')
                     for line in (f'.. _{label}:\n', '\n')]
                )
            self.note(f'added explicit target {label!r} for section '
                      f'{self._title(sec)!r}')
        # also injections for renamed global labels (collisions)
        for gidx, label in self.label_injections.items():
            sec = next((s for s in sm.sections if s.start == gidx), None)
            if sec is None:
                continue
            streams['moved'].insert_before_line(
                sec.origin, sec.title_oln,
                [Cell(line, sec.origin, 0, 'synthetic')
                 for line in (f'.. _{label}:\n', '\n')]
            )
        # token rewrites
        self._apply_token_renames(streams, {
            tid: ('', new) for tid, new in token_renames.items()
        }, raw_new=True)

    def _section_explicit_name(self, section: nodes.section) -> str | None:
        implicit = nodes.make_id(section[0].astext() if len(section) else '')
        for name in section.get('names', []):
            if name != implicit:
                return name
        return None

    # -----------------------------------------------------------------
    # shared token rewrite application
    # -----------------------------------------------------------------

    def _apply_token_renames(
        self,
        streams: dict[str, Stream],
        renames: dict[int, tuple[str, str]],
        *,
        raw_new: bool = False,
    ) -> None:
        by_host_stream: dict[str, dict[tuple[Path, int], list[tuple[str, str]]]] = {
            'moved': defaultdict(list),
            'before': defaultdict(list),
            'after': defaultdict(list),
        }
        host_seen: dict[int, int] = defaultdict(int)
        sm = self.SM
        for tok in sm.tokens:
            spec = renames.get(id(tok))
            if spec is None:
                continue
            if self.gidx_in_x(tok.host_index):
                stream_name = 'moved'
            elif tok.host_index < self.c0:
                stream_name = 'before'
            else:
                stream_name = 'after'
            if raw_new:
                old, new = tok.rawsource, spec[1]
            else:
                old, new = spec
            by_host_stream[stream_name][
                (tok.host_origin, tok.host_oln)
            ].append((old, new))
        for stream_name, hosts in by_host_stream.items():
            stream = streams[stream_name]
            for (origin, oln), specs in hosts.items():
                # find host end oln from any token
                toks = sm.tokens_by_host[(origin, oln)]
                end_oln = max(t.host_end_oln for t in toks)
                occurrence = self._host_occurrence(stream_name, origin, oln)
                stream.rewrite_host(origin, oln, end_oln, specs,
                                   occurrence=occurrence)

    def _host_occurrence(
        self, stream_name: str, origin: Path, oln: int
    ) -> int:
        stream_map = {
            'moved': (self.c0, self.c1),
            'before': (0, self.c0),
            'after': (self.c1, len(self.SM.lines)),
        }
        lo, _hi = stream_map[stream_name]
        target_gidx = self._token_host_gidx(origin, oln, stream_name)
        count = 0
        for i in range(lo, target_gidx):
            sl = self.SM.lines[i]
            if sl.origin == origin and sl.oln == oln:
                count += 1
        return count

    def _first_host_gidx(self, origin: Path, oln: int, stream_name: str) -> int:
        return self._token_host_gidx(origin, oln, stream_name)

    def _token_host_gidx(self, origin: Path, oln: int, stream_name: str) -> int:
        for t in self.SM.tokens:
            if t.host_origin == origin and t.host_oln == oln:
                return t.host_index
        return 0

    # -----------------------------------------------------------------
    # xref renames (global labels)
    # -----------------------------------------------------------------

    def _apply_xref_renames(
        self, streams: dict[str, Stream], label_plan: dict[str, Any]
    ) -> None:
        if not self.renamed_xrefs:
            return
        grouped: dict[str, dict[tuple[Path, int], list[tuple[str, str]]]] = {
            'moved': defaultdict(list),
            'before': defaultdict(list),
            'after': defaultdict(list),
        }
        for dm, tok, new_label in self.renamed_xrefs:
            node = tok.node
            if dm is self.SM:
                stream_name = (
                    'moved' if self.gidx_in_x(tok.host_index)
                    else 'before' if tok.host_index < self.c0
                    else 'after'
                )
            else:
                continue  # target-side consumers are not rewritten
            display = node.astext() if node.get('refexplicit') else ''
            new = (f':ref:`{display} <{new_label}>`' if display
                   else f':ref:`{new_label}`')
            grouped[stream_name][
                (tok.host_origin, tok.host_oln)
            ].append((tok.rawsource, new))
        sm = self.SM
        for stream_name, hosts in grouped.items():
            stream = streams[stream_name]
            for (origin, oln), specs in hosts.items():
                toks = sm.tokens_by_host[(origin, oln)]
                end_oln = max(t.host_end_oln for t in toks)
                occurrence = self._host_occurrence(stream_name, origin, oln)
                stream.rewrite_host(origin, oln, end_oln, specs,
                                   occurrence=occurrence)

    # -----------------------------------------------------------------
    # relative resource / doc paths
    # -----------------------------------------------------------------

    def _rewrite_paths(self, streams: dict[str, Stream]) -> None:
        src_dir = posixpath.dirname(self.src_docname)
        tgt_dir = posixpath.dirname(self.tgt_docname)
        if src_dir == tgt_dir:
            return
        moved = streams['moved']
        env_docs = set(self.app.env.all_docs)
        cells = moved.cells
        i = 0
        while i < len(cells):
            text = cells[i].text.rstrip('\r\n')
            m = _DIRECTIVE_RE.match(text)
            if m:
                name = m.group('name').strip()
                canonical = name.split(':')[-1]
                arg = m.group('arg') or ''
                if canonical in _PATH_DIRECTIVE_ARGS and arg:
                    new_arg = self._rel_resource(arg, src_dir, tgt_dir)
                    if new_arg != arg:
                        cells[i].text = text.replace(arg, new_arg, 1) + (
                            '\r\n' if cells[i].text.endswith('\r\n') else '\n'
                        )
                # option lines
                j = i + 1
                while (j < len(cells) and cells[j].origin == cells[i].origin
                       and _OPTION_LINE_RE.match(cells[j].text.rstrip('\r\n'))):
                    om = _OPTION_LINE_RE.match(cells[j].text.rstrip('\r\n'))
                    opt = om.group(1)
                    val = om.group(2) or ''
                    if canonical in _PATH_DIRECTIVE_OPTIONS and opt in \
                            _PATH_DIRECTIVE_OPTIONS[canonical] and val:
                        new_val = self._rel_resource(val, src_dir, tgt_dir)
                        if new_val != val:
                            nl = '\r\n' if cells[j].text.endswith('\r\n') else '\n'
                            cells[j].text = cells[j].text.replace(val, new_val, 1)
                    j += 1
                if canonical == 'toctree':
                    self._rewrite_toctree_entries(cells, i, j, src_dir,
                                                  tgt_dir, env_docs)
                i = j
                continue
            # :download: roles
            if ':download:`' in text:
                nl = '\r\n' if cells[i].text.endswith('\r\n') else '\n'
                cells[i].text = _DOWNLOAD_ROLE_RE.sub(
                    lambda mm: ':download:`'
                    + self._rewrite_download_body(mm.group(1), src_dir,
                                                  tgt_dir)
                    + '`',
                    cells[i].text.rstrip('\r\n'),
                ) + nl
            i += 1

    def _rewrite_toctree_entries(
        self, cells, i, j, src_dir, tgt_dir, env_docs
    ) -> None:
        k = i + 1
        while k < j:
            raw = cells[k].text.rstrip('\r\n')
            stripped = raw.strip()
            if not stripped or stripped.startswith(':') or stripped.startswith('..'):
                k += 1
                continue
            if '://' in stripped:
                k += 1
                continue
            entry, title = _parse_toctree_entry(stripped)
            if entry is None:
                k += 1
                continue
            if posixpath.normpath(posixpath.join(src_dir, entry)) in env_docs:
                new_entry = _doc_relative(src_dir, tgt_dir, entry)
                nl = '\r\n' if cells[k].text.endswith('\r\n') else '\n'
                indent = raw[:len(raw) - len(raw.lstrip())]
                rendered = (f'{indent}{title} <{new_entry}>' if title
                            else f'{indent}{new_entry}') + nl
                cells[k].text = rendered
            k += 1

    @staticmethod
    def _rel_resource(arg: str, src_dir: str, tgt_dir: str) -> str:
        token = arg.strip()
        if _is_external_ref(token):
            return arg
        if token.startswith('<') and token.endswith('>'):
            return arg
        abs_doc = posixpath.normpath(posixpath.join(src_dir, token))
        new = posixpath.relpath(abs_doc, tgt_dir)
        new = new.replace(os.sep, '/')
        if new == '.':
            new = './'
        return new

    @staticmethod
    def _rewrite_download_body(body: str, src_dir: str, tgt_dir: str) -> str:
        m = re.match(r'^(.*[^<])\s+<([^>]+)>$', body)
        if m:
            label, target = m.group(1), m.group(2)
            return f'{label} <{Planner._rel_resource(target, src_dir, tgt_dir)}>'
        return Planner._rel_resource(body, src_dir, tgt_dir)

    # -----------------------------------------------------------------
    # deletion of runs (equivalent reused definitions)
    # -----------------------------------------------------------------

    def _delete_run(
        self, streams, origin: Path, oln: int, end_oln: int,
        c0: int, c1: int
    ) -> None:
        moved = streams['moved']
        try:
            i, j = moved.find_run(origin, oln, end_oln)
        except KeyError:
            return
        del moved.cells[i:j]

    # -----------------------------------------------------------------
    # assembly
    # -----------------------------------------------------------------

    def _assemble(self, streams: dict[str, Stream]) -> dict[str, str]:
        moved_text = self._finalize_moved(streams)
        if self.same_doc:
            source_text = self._assemble_same_doc(streams, moved_text)
            return {self.SM.path: source_text}
        source_text = self._finalize_source(streams)
        target_text = self._finalize_target(moved_text)
        return {self.SM.path: source_text, self.TM.path: target_text}

    def _finalize_moved(self, streams: dict[str, Stream]) -> str:
        moved = streams['moved']
        parts: list[str] = []
        if self.context_prefix:
            parts.append(self.context_prefix)
        parts.append(moved.text())
        for block in self.moved_tail_blocks:
            parts.append(block.rstrip('\n') + '\n')
        if self.context_suffix:
            parts.append(self.context_suffix)
        text = ''.join(parts)
        return _ensure_blank_bounded(text)

    def _finalize_source(self, streams: dict[str, Stream]) -> str:
        before = streams['before'].text()
        after = streams['after'].text()
        restore = self.context_restore
        head = before
        if restore:
            head = _join_blocks(head, restore)
        body = _join_blocks(head, after)
        if self.tail_blocks:
            tail = '\n\n'.join(b.strip('\n') for b in self.tail_blocks)
            body = _join_blocks(body, tail + '\n')
        return _normalize_document(body)

    def _assemble_same_doc(
        self, streams: dict[str, Stream], moved_text: str
    ) -> str:
        before = streams['before']
        after = streams['after']
        moved_cells = [
            Cell(line, self.SM.path, 0, 'synthetic')
            for line in moved_text.splitlines(keepends=True)
        ]
        if self.P is None:
            after.append_cells(moved_cells)
        else:
            # insert at the *end* of the parent subtree, i.e. immediately
            # before the next sibling/outer section title (or at EOF)
            end = self.P.end
            boundary = self.SM.section_at.get(end)
            if boundary is None or end >= len(self.SM.lines):
                after.append_cells(moved_cells)
            else:
                target_stream = (before if end < self.c0 else after)
                matches = [
                    idx for idx, c in enumerate(target_stream.cells)
                    if c.origin == boundary.origin
                    and c.oln == boundary.title_oln
                ]
                if not matches:
                    # boundary fell inside the removed subtree: append in after
                    after.append_cells(moved_cells)
                else:
                    idx = self._include_leads(target_stream, matches[0])
                    target_stream.cells[idx:idx] = moved_cells
        body = _join_blocks(before.text(),
                            _join_blocks(self.context_restore, after.text()))
        if self.tail_blocks:
            tail = '\n\n'.join(b.strip('\n') for b in self.tail_blocks)
            body = _join_blocks(body, tail + '\n')
        return _normalize_document(body)

    @staticmethod
    def _include_leads(stream: Stream, idx: int) -> int:
        k = idx
        while k - 1 >= 0:
            prev = stream.cells[k - 1].text.rstrip('\r\n')
            if not prev.strip():
                k -= 1
                continue
            if prev.lstrip().startswith('.. _'):
                k -= 1
                continue
            break
        return k

    def _finalize_target(self, moved_text: str) -> str:
        tm = self.TM
        phys = self.session.file_contents[tm.path][0]
        original = ''.join(phys)
        if self.P is None:
            return _normalize_document(_join_blocks(original, moved_text))
        end = self.P.end
        if end >= len(tm.lines):
            return _normalize_document(_join_blocks(original, moved_text))
        boundary = tm.section_at.get(end)
        if boundary is not None and boundary.origin != tm.path:
            return self._finalize_target_materialized(boundary, moved_text)
        # boundary title lives in the physical target file: insert before it
        # (and before its attached lead targets / overline)
        lead_oln = boundary.title_oln if boundary is not None else None
        if boundary is not None:
            candidate = boundary.title_oln
            if boundary.overline_oln is not None:
                candidate = min(candidate, boundary.overline_oln)
            for oln in boundary.lead_olns:
                candidate = min(candidate, oln)
            lead_oln = candidate
        if lead_oln is None:
            return _normalize_document(_join_blocks(original, moved_text))
        head = ''.join(phys[:lead_oln - 1])
        tail = ''.join(phys[lead_oln - 1:])
        return _normalize_document(_join_blocks(_join_blocks(head, moved_text),
                                                tail))

    def _finalize_target_materialized(
        self, boundary: SectionInfo, moved_text: str
    ) -> str:
        tm = self.TM
        builder = _ItemTreeBuilder(tm, self.session.file_contents)
        items = builder.build()
        end = boundary.start

        def policy(inst: Any) -> str:
            a, b = inst.span
            return 'materialize' if a <= end < b else 'directive'

        cells = _render_items(items, window=None, policy=policy)
        stream = Stream(cells)
        matches = [
            idx for idx, c in enumerate(stream.cells)
            if c.origin == boundary.origin and c.oln == boundary.title_oln
        ]
        if not matches:
            self.issues.err(
                'target-insertion-failed',
                'could not locate the insertion point while materialising '
                'an include in the target document',
                boundary.origin,
                boundary.title_oln,
            )
            self.issues.raise_if_errors()
        new_cells = [
            Cell(line, tm.path, 0, 'synthetic')
            for line in moved_text.splitlines(keepends=True)
        ]
        stream.cells[matches[0]:matches[0]] = new_cells
        self.note('materialised one include expansion at the target insertion '
                  'point; the shared fragment file is unchanged')
        return _normalize_document(stream.text())


def _context_directive(key: str, value: str | None) -> str:
    if key == 'module':
        return f'.. py:currentmodule:: {value or ""}\n'
    if key == 'domain':
        return f'.. default-domain:: {value or "rst"}\n'
    return f'.. default-role:: {value or "title-reference"}\n'


def _ref_name(node: nodes.Node) -> str | None:
    name = node.get('refname')
    if name:
        return name
    name = node.get('name')
    if name:
        return name
    return None


def _node_name(node: nodes.Node) -> str:
    return node.get('refname') or node.get('name') or ''


def _parse_toctree_entry(stripped: str) -> tuple[str | None, str | None]:
    m = re.match(r'^(?P<title>.+?)\s+<(?P<target>[^>]+)>$', stripped)
    if m:
        return m.group('target').strip(), m.group('title').strip()
    return stripped.strip(), None


def _doc_relative(src_dir: str, tgt_dir: str, entry: str) -> str:
    abs_doc = posixpath.normpath(posixpath.join(src_dir, entry))
    new = posixpath.relpath(abs_doc, tgt_dir).replace(os.sep, '/')
    return new


def _reference_display(tok: Token) -> str:
    node = tok.node
    if node.get('refexplicit'):
        return node.astext()
    raw = tok.rawsource
    m = re.match(r'`(?P<disp>.+?)\s+<[^>]+>`_$', raw, re.DOTALL)
    if m:
        return m.group('disp')
    return ''


def new_ref_token(tok: Token, label: str | None) -> str:
    if label is None:
        return tok.rawsource
    if label.startswith('#'):
        return f'[{label}]_'
    return f'[{label}]_'


def _ensure_blank_bounded(text: str) -> str:
    if not text.endswith('\n'):
        text += '\n'
    if not text.endswith('\n\n'):
        text += '\n'
    return text


def _join_blocks(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    if a.endswith('\n\n') and not b.startswith('\n'):
        return a + b
    if a.endswith('\n') and not b.startswith('\n'):
        return a + '\n' + b
    if not a.endswith('\n'):
        return a + '\n\n' + b
    return a + b


def _normalize_document(text: str) -> str:
    text = text.replace('\r\n', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    if not text.endswith('\n'):
        text += '\n'
    return text


# ===========================================================================
# Validation build
# ===========================================================================


def validate_result(
    session: Session,
    texts: dict[Path, str],
    issues: Issues,
) -> None:
    """Build a temporary copy with the proposed texts.

    Raises RefactorAborted on any *new* diagnostic; leaves real sources
    untouched.
    """
    import subprocess
    import sys

    srcdir = session.srcdir
    tmp = Path(tempfile.mkdtemp(prefix='sphinx-refactor-verify-'))
    target_src = tmp / 'src'
    ignore = shutil.ignore_patterns('_build', '.doctrees', '__pycache__')
    shutil.copytree(srcdir, target_src, ignore=ignore)
    for path, text in texts.items():
        rel = os.path.relpath(path, srcdir)
        out = target_src / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8', newline='')
    builddir = tmp / 'build'
    # Use a clean subprocess: instantiating a second Sphinx app in-process
    # re-registers directives/nodes and produces unrelated setup warnings.
    env = dict(os.environ)
    py_path = os.pathsep.join(sys.path)
    if env.get('PYTHONPATH'):
        py_path = env['PYTHONPATH'] + os.pathsep + py_path
    env['PYTHONPATH'] = py_path
    config_args: list[str] = []
    for key, value in (session.confoverrides or {}).items():
        config_args += ['-D', f'{key}={value}']
    proc = subprocess.run(
        [sys.executable, '-m', 'sphinx.cmd.build', '-b', 'dummy', '-E',
         *config_args, str(target_src), str(builddir)],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        tail = '\n'.join((proc.stderr or proc.stdout).splitlines()[-20:])
        issues.err('verification-build-failed',
                   f'sphinx-build exited with {proc.returncode}:\n{tail}')
        issues.raise_if_errors()
    post = _parse_subprocess_warnings(proc.stderr, target_src)
    _compare_messages(session, post, issues, texts)
    issues.raise_if_errors()
    shutil.rmtree(tmp, ignore_errors=True)


def _parse_subprocess_warnings(
    stream: str, srcdir: Path
) -> dict[str, list[tuple[str, str, str]]]:
    result: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for raw in stream.splitlines():
        line = _ANSI_RE.sub('', raw)
        m = _WARNING_RE.match(line)
        if not m:
            continue
        f = m.group('file')
        try:
            rel = os.path.relpath(Path(f), srcdir).replace(os.sep, '/')
        except ValueError:
            rel = f
        msg = re.sub(r'["\u2018\u2019].*?["\u2018\u2019]', '""', m.group('msg'))
        result[rel].append((m.group('level'), m.group('subtype') or '', msg))
    return dict(result)


def _compare_messages(
    session: Session,
    post: dict[str, list[tuple[str, str, str]]],
    issues: Issues,
    texts: dict[Path, str],
) -> None:
    baseline = session.baseline_messages
    edited_rel = {
        os.path.relpath(p, session.srcdir).replace(os.sep, '/') for p in texts
    }
    all_docs = set(baseline) | set(post)
    for doc in all_docs:
        base = list(baseline.get(doc, []))
        new = list(post.get(doc, []))
        if doc in edited_rel:
            # same multiset of (level, subtype, masked message); line numbers
            # are allowed to shift
            for item in new:
                if item in base:
                    base.remove(item)
                else:
                    issues.err(
                        'new-build-diagnostic',
                        f'{item[0]} {item[2]} [{item[1]}] in {doc} after move',
                    )
        else:
            for item in new:
                if item not in base:
                    issues.err(
                        'diagnostic-changed-elsewhere',
                        f'{item[0]} {item[2]} [{item[1]}] in untouched '
                        f'document {doc}',
                    )


def _compare_inventories(
    session: Session,
    app: Sphinx,
    issues: Issues,
    texts: dict[Path, str],
) -> None:
    base_env = session.app.env
    base_py = base_env.domains['py'].objects
    post_py = app.env.domains['py'].objects
    missing = sorted(set(base_py) - set(post_py))
    if missing:
        issues.err(
            'object-lost',
            'Python objects disappeared after the move: '
            + ', '.join(missing[:10]),
        )
    base_labels = base_env.domains['std'].labels
    post_labels = app.env.domains['std'].labels
    lost_labels = sorted(set(base_labels) - set(post_labels))
    if lost_labels:
        issues.err(
            'label-lost',
            'labels disappeared after the move: '
            + ', '.join(lost_labels[:10]),
        )


