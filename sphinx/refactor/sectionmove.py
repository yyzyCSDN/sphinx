"""Move an rST section (and its subsections) between source documents.

The operation is a *source refactoring*: it takes

* a project source directory,
* a source docname and a **1-based title-text line number** of the section to
  move,
* a target docname and a 1-based title-text line number of the future parent
  section (``0`` means the document root),

and produces a patch that moves the selected section so that it becomes the
last subsection of the target parent, adjusting the underline hierarchy of
every moved heading.

Bindings are taken from a real read-only analysis build performed *before*
the move (parser, standard and Python domains and ordinary ``include``
expansions).  Local links and indirect aliases, substitutions and their
recursive dependencies, footnotes, standard/Python references and object
identity, the active module/default-role/default-domain context and relative
resource paths are preserved by supplementing explicit targets or context,
rewriting references, copying (and, on collision, renaming) local definitions,
and, when necessary, materialising a single expansion of a shared
``include`` fragment.

By default a unified source patch is returned and nothing is written.  Files
are only modified when the caller explicitly asks to *apply* the move, and
even then the result is verified by a build in a temporary copy first.

Only native reStructuredText, the ``std`` and ``py`` domains and ordinary
includes are supported.  Custom domains that participate in the move,
``autodoc`` directives and dynamically generated body content from event
hooks are rejected with a locatable diagnostic.
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
import io
import os
import posixpath
import re
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from docutils import nodes
from docutils.readers import standalone

from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.util import get_filetype, logging
from sphinx.util.docutils import (
    LoggingReporter,
    _get_settings,
    sphinx_domains,
)

if False:
    # TYPE_CHECKING block -- populated below for type checkers only.
    pass

if __import__('typing').TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class Diagnostic:
    """A single diagnostic produced while planning a section move."""

    severity: str  # 'error' or 'warning'
    code: str
    message: str
    location: str | None = None

    def format(self) -> str:
        loc = f'{self.location}: ' if self.location else ''
        return f'{loc}{self.severity.upper()}: [{self.code}] {self.message}'


class RefactorAborted(SphinxError):
    """Raised when a section move cannot be performed safely.

    No source file has been modified when this exception is raised.
    """

    def __init__(self, diagnostics: Diagnostic | list[Diagnostic]) -> None:
        if isinstance(diagnostics, Diagnostic):
            diagnostics = [diagnostics]
        self.diagnostics: list[Diagnostic] = list(diagnostics)
        super().__init__('\n'.join(d.format() for d in self.diagnostics))


@dataclass
class FilePatch:
    """A proposed modification of a single source file."""

    path: Path
    old_text: str
    new_text: str

    def unified_diff(self, *, context: int = 3) -> str:
        old = self.old_text.splitlines(keepends=True)
        new = self.new_text.splitlines(keepends=True)
        rel = self.path.name
        diff = difflib.unified_diff(
            old,
            new,
            fromfile=f'a/{rel}',
            tofile=f'b/{rel}',
            n=context,
        )
        return ''.join(diff)


@dataclass
class MoveSectionResult:
    """Result of :func:`move_section`."""

    source_docname: str
    target_docname: str
    source_title: str
    target_parent_title: str | None
    patches: list[FilePatch] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    applied: bool = False

    def format_patch(self) -> str:
        return '\n'.join(p.unified_diff() for p in self.patches).strip('\n') + '\n'


# ---------------------------------------------------------------------------
# Constants / regular expressions
# ---------------------------------------------------------------------------

_RST_UNDERLINE_CHARS = set("= - ` : ' \" ~ ^ _ * + # < >")
_PREFERRED_CHARS = ['=', '-', '^', '"', '~', '+', '#', "'", '`', '@',
                    ':', '.', '_', '*']

_DIRECTIVE_RE = re.compile(r'^(?P<indent>[ \t]*)\.\.[ \t]+(?P<name>[^\s:`][^:`]*?)::'
                           r'(?:[ \t]*(?P<arg>.*?))?[ \t]*$')
_COMMENT_RE = re.compile(r'^[ \t]*\.\.[ \t]*$')
_EXPLICIT_TARGET_RE = re.compile(r'^[ \t]*\.\.[ \t]+_(?P<name>[^:]+):[ \t]*$')
_INCLUDE_RE = re.compile(r'^(?P<indent>[ \t]*)\.\.[ \t]+include::[ \t]+'
                         r'(?P<arg>.+?)[ \t]*$')
_OPTION_RE = re.compile(r'^[ \t]+:[A-Za-z][\w-]*:')

_CURRENTMODULE_RE = re.compile(
    r'^[ \t]*\.\.[ \t]+(?:py:)?(?P<what>currentmodule|module)::[ \t]*(?P<arg>.*?)\s*$'
)
_DEFAULT_DOMAIN_RE = re.compile(
    r'^[ \t]*\.\.[ \t]+default-domain::[ \t]*(?P<arg>[^\s]+)\s*$'
)
_DEFAULT_ROLE_RE = re.compile(
    r'^[ \t]*\.\.[ \t]+default-role::[ \t]*(?P<arg>\S+)\s*$'
)
_ROLE_DIRECTIVE_RE = re.compile(
    r'^[ \t]*\.\.[ \t]+role::[ \t]+(?P<arg>\S+)\s*$'
)

_AUTODOC_RE = re.compile(
    r'^[ \t]*\.\.[ \t]+(?:[\w-]+:)*auto[a-z]*::'
)

_PATH_DIRECTIVE_ARGS = {
    'include',
    'literalinclude',
    'image',
    'figure',
}
_PATH_DIRECTIVE_OPTIONS = {
    'raw': {'url', 'file'},
    'csv-table': {'url', 'file'},
    'list-table': {'url'},  # list-table has no file option, kept empty for safety
}
_DOWNLOAD_ROLE_RE = re.compile(r':download:`(?P<body>.*?)`', re.DOTALL)
_ROLE_TARGET_RE = re.compile(r'^(?:[^<\n]*<(?P<target>[^>]+)>)$')

_SUPPORTED_DOMAINS = {'', 'rst', 'std', 'py'}

_INCLUDE_SUPPORTED_OPTIONS = {
    'start-line', 'end-line', 'start-after', 'end-before',
    'encoding', 'tab-width', 'number-lines', 'class', 'name',
    'literal', 'code',
}

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_WARNING_RE = re.compile(
    r'^(?P<file>.+?):(?:(?P<line>\d+)(?::(?P<col>\d+))?:)? '
    r'(?P<level>DEBUG|INFO|WARNING|ERROR|SEVERE): '
    r'(?P<msg>.*?)(?: \[(?P<subtype>[\w.\-/]+)\])?$'
)


def _norm_name(name: str) -> str:
    return nodes.fully_normalize_name(name)


# ---------------------------------------------------------------------------
# Annotated source lines
# ---------------------------------------------------------------------------


@dataclass
class SLine:
    """One expanded source line."""

    text: str
    origin: Path
    oln: int  # 1-based line number in *origin*
    uid: int
    instance: int | None = None  # innermost include instance that produced it


@dataclass
class IncludeInstance:
    uid: int
    directive_origin: Path
    directive_oln: int
    target: Path | None  # None for opaque-external <...> includes / unreadable
    standard: bool
    opaque: bool  # :literal:/:code: include -- expansion is one literal block
    options: dict[str, str] = field(default_factory=dict)
    span: tuple[int, int] = (-1, -1)  # expanded content indices (exclusive)
    parent: int | None = None


# ---------------------------------------------------------------------------
# Diagnostics collection
# ---------------------------------------------------------------------------


class Issues:
    def __init__(self, srcdir: Path) -> None:
        self.srcdir = srcdir
        self.errors: list[Diagnostic] = []
        self.warnings: list[Diagnostic] = []
        self.notes: list[str] = []

    def loc(self, path: Path | str, line: int | None = None) -> str:
        p = Path(path)
        try:
            name = os.path.relpath(p, self.srcdir).replace(os.sep, '/')
        except ValueError:
            name = str(p)
        return f'{name}:{line}' if line is not None else name

    def err(self, code: str, msg: str,
            path: Path | str | None = None, line: int | None = None) -> None:
        loc = self.loc(path, line) if path is not None else None
        self.errors.append(Diagnostic('error', code, msg, loc))

    def warn(self, code: str, msg: str,
             path: Path | str | None = None, line: int | None = None) -> None:
        loc = self.loc(path, line) if path is not None else None
        self.warnings.append(Diagnostic('warning', code, msg, loc))

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def raise_if_errors(self) -> None:
        if self.errors:
            raise RefactorAborted(self.errors)


# ---------------------------------------------------------------------------
# Raw rST helpers
# ---------------------------------------------------------------------------


def _line_indent(text: str) -> int:
    return len(text) - len(text.lstrip(' \t'))


def _is_underline(text: str) -> str | None:
    stripped = text.rstrip('\r\n')
    if not stripped:
        return None
    ch = stripped[0]
    if ch not in _RST_UNDERLINE_CHARS:
        return None
    if any(c != ch for c in stripped):
        return None
    if len(stripped) < 1:
        return None
    return ch


def _block_extent(lines: SequenceLike, i: int, bound: int | None = None) -> int:
    """Exclusive end index of a directive/comment/target block at *i*.

    Consumes the marker line and following indented content (with blank lines
    only while a more-indented non-blank line follows).
    """
    if bound is None:
        bound = len(lines)
    text = lines[i].text if hasattr(lines[i], 'text') else lines[i]
    indent = _line_indent(text)
    j = i + 1
    last = i
    while j < bound:
        t = lines[j].text if hasattr(lines[j], 'text') else lines[j]
        if not t.strip():
            k = j
            while k < bound and not (lines[k].text if hasattr(lines[k], 'text')
                                     else lines[k]).strip():
                k += 1
            if k < bound and _line_indent(
                lines[k].text if hasattr(lines[k], 'text') else lines[k]
            ) > indent:
                j = k
                continue
            break
        if _line_indent(t) > indent:
            last = j
            j += 1
            continue
        break
    return last + 1


def _paragraph_extent(lines: SequenceLike, i: int, bound: int) -> int:
    indent = _line_indent(lines[i].text)
    j = i
    while j < bound:
        t = lines[j].text
        if not t.strip():
            break
        if j != i and _line_indent(t) < indent:
            break
        ch = _is_underline(t)
        if ch is not None and j == i + 1:
            # would be a title, stop defensively
            break
        j += 1
    return j


def _read_physical(path: Path) -> tuple[list[str], str]:
    """Return (lines with ends, newline style) for a physical source file."""
    with open(path, encoding='utf-8', newline='') as f:
        text = f.read()
    if '\r\n' in text:
        newline = '\r\n'
    else:
        newline = '\n'
    lines = text.splitlines(keepends=True)
    return lines, newline


# ---------------------------------------------------------------------------
# Include expansion
# ---------------------------------------------------------------------------


class ExpansionError(Exception):
    def __init__(self, path: Path, line: int, msg: str) -> None:
        self.path = path
        self.line = line
        super().__init__(f'{path}:{line}: {msg}')


class _Expander:
    def __init__(
        self,
        srcdir: Path,
        issues: Issues,
        file_contents: dict[Path, tuple[list[str], str]],
        protected: dict[Path, list[tuple[int, int]]] | None = None,
        docdir: Path | None = None,
    ) -> None:
        self.srcdir = srcdir
        self.issues = issues
        self.file_contents = file_contents
        self.protected = protected or {}
        # Sphinx resolves include arguments relative to the directory of the
        # top-level document being parsed, regardless of nesting depth.
        self.docdir = docdir if docdir is not None else srcdir
        self.uid_counter = 0
        self.instances: dict[int, IncludeInstance] = {}
        self._stack: list[Path] = []

    def _new_uid(self) -> int:
        self.uid_counter += 1
        return self.uid_counter

    def _lines_of(self, path: Path) -> list[str]:
        if path not in self.file_contents:
            try:
                self.file_contents[path] = _read_physical(path)
            except OSError:
                self.file_contents[path] = ([], '\n')
        return self.file_contents[path][0]

    def _is_protected(self, origin: Path, oln: int) -> bool:
        return any(a <= oln < b for a, b in self.protected.get(origin, ()))

    def expand(self, docfile: Path) -> list[SLine]:
        physical = self._lines_of(docfile)
        return self._expand_file(
            docfile,
            [SLine(text=t, origin=docfile, oln=i + 1,
                   uid=self._new_uid()) for i, t in enumerate(physical)],
        )

    def _expand_file(
        self, origin: Path, base: list[SLine]
    ) -> list[SLine]:
        if origin in self._stack:
            msg = 'circular inclusion'
            raise ExpansionError(origin, 1, msg)
        self._stack.append(origin)
        try:
            out: list[SLine] = []
            i = 0
            n = len(base)
            while i < n:
                sl = base[i]
                text = sl.text.rstrip('\r\n')
                m = _INCLUDE_RE.match(text)
                if (m is not None and not self._is_protected(origin, sl.oln)):
                    inst = self._make_instance(sl, m)
                    self.instances[inst.uid] = inst
                    if inst.standard or inst.opaque or inst.target is None:
                        # external/standard/literal include: keep the directive
                        out.append(sl)
                        i += 1
                        continue
                    included = self._read_included(inst)
                    if included is None:
                        out.append(sl)
                        i += 1
                        continue
                    content_base = [
                        SLine(text=t, origin=inst.target, oln=oln,
                              uid=self._new_uid(), instance=inst.uid)
                        for oln, t in included
                    ]
                    start = len(out)
                    out.extend(self._expand_file(inst.target, content_base))
                    inst.span = (start, len(out))
                    i += 1
                else:
                    out.append(sl)
                    i += 1
            return out
        finally:
            self._stack.pop()

    def _make_instance(
        self, directive_line: SLine, match: re.Match[str]
    ) -> IncludeInstance:
        raw_arg = match.group('arg').strip()
        standard = raw_arg.startswith('<') and raw_arg.endswith('>')
        options: dict[str, str] = {}
        phys = self._lines_of(directive_line.origin)
        ext = _block_extent(
            [type('L', (), {'text': t})() for t in phys],
            directive_line.oln - 1,
        )
        for t in phys[directive_line.oln:ext]:
            mm = re.match(r'^[ \t]+:([\w-]+):(?:[ \t]*(.*))?$',
                          t.rstrip('\r\n'))
            if mm:
                options[mm.group(1)] = (mm.group(2) or '').strip()
        target: Path | None = None
        if not standard:
            target = self._resolve_include_path(directive_line.origin, raw_arg)
        opaque = 'literal' in options or 'code' in options
        return IncludeInstance(
            uid=self._new_uid(),
            directive_origin=directive_line.origin,
            directive_oln=directive_line.oln,
            target=target,
            standard=standard,
            opaque=opaque,
            options=options,
        )

    def _resolve_include_path(self, host: Path, arg: str) -> Path:
        if arg.startswith(('/', '\\')):
            return (self.srcdir / arg.lstrip('/\\')).resolve()
        return (self.docdir / arg).resolve()

    def _read_included(
        self, inst: IncludeInstance
    ) -> list[tuple[int, str]] | None:
        """Return list of (physical 1-based line number, text with NL)."""
        assert inst.target is not None
        if inst.opaque:
            return None
        if 'parser' in inst.options:
            msg = ('the :parser: option of include is not supported by the '
                   'section move refactoring')
            raise ExpansionError(inst.directive_origin, inst.directive_oln, msg)
        unknown = set(inst.options) - _INCLUDE_SUPPORTED_OPTIONS
        if unknown:
            msg = f'unsupported include option(s): {sorted(unknown)!r}'
            raise ExpansionError(inst.directive_origin, inst.directive_oln, msg)
        raw_lines = self._lines_of(inst.target)
        indexed = list(enumerate(raw_lines, start=1))
        start_line = int(inst.options.get('start-line', 0) or 0) + 1
        end_raw = inst.options.get('end-line')
        end_line = int(end_raw) + 1 if end_raw not in (None, '') else None
        indexed = [(ln, t) for ln, t in indexed
                   if ln >= start_line and (end_line is None or ln <= end_line)]
        marker = inst.options.get('start-after')
        if marker:
            for idx, (_, t) in enumerate(indexed):
                if marker in t:
                    indexed = indexed[idx + 1:]
                    break
        marker = inst.options.get('end-before')
        if marker:
            for idx, (_, t) in enumerate(indexed):
                if marker in t:
                    indexed = indexed[:idx]
                    break
        tab_width = int(inst.options.get('tab-width', 8) or 8)
        out: list[tuple[int, str]] = []
        for ln, t in indexed:
            nl = '\r\n' if t.endswith('\r\n') else ('\n' if t else '\n')
            body = t[:-2] if t.endswith('\r\n') else (t[:-1] if t.endswith('\n')
                                                     else t)
            out.append((ln, body.expandtabs(tab_width) + nl))
        return out


# Type alias declared late to avoid forward-reference noise.
from collections.abc import Sequence as _Sequence  # noqa: E402

SequenceLike = _Sequence


# ---------------------------------------------------------------------------
# Analysis session
# ---------------------------------------------------------------------------


@dataclass
class SectionInfo:
    node: nodes.section
    depth: int
    origin: Path
    title_oln: int      # title text line (1-based)
    under_oln: int
    char: str
    overline_oln: int | None
    start: int         # expanded index of title text
    end: int           # expanded index of next section start / EOF
    names: list[str]
    ids: list[str]
    lead_olns: list[int] = field(default_factory=list)


@dataclass
class Token:
    kind: str  # 'xref','reference','citation','footnote','substitution'
    node: Any
    host_origin: Path
    host_oln: int
    host_end_oln: int
    ordinal: int
    rawsource: str
    host_index: int = -1
    host_end_index: int = -1
    binding: Any = None  # Binding | None


@dataclass
class Binding:
    kind: str
    docname: str | None = None
    node: Any = None
    origin: Path | None = None
    oln: int | None = None
    id: str | None = None
    name: str | None = None
    uri: str | None = None


@dataclass
class DefBlock:
    kind: str  # 'substitution','footnote','citation','target'
    name: str
    node: Any
    origin: Path
    oln: int
    end_oln: int
    text: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocModel:
    docname: str
    path: Path
    lines: list[SLine]
    instances: dict[int, IncludeInstance]
    raw: nodes.document
    read_tree: nodes.document
    sections: list[SectionInfo]
    section_at: dict[int, SectionInfo]  # expanded index of title
    defs_by_name: dict[str, list[DefBlock]]
    defs_by_id: dict[str, DefBlock]
    tokens: list[Token]
    tokens_by_host: dict[tuple[Path, int], list[Token]]
    targets_by_name: dict[str, Any]
    pos_index: dict[tuple[Path, int], list[int]]
    occ: dict[tuple[Path, int], list[int]]
    chars_by_depth: list[str]
    contexts: list[tuple[int, str, str | None]]
    autosection: bool
    sectionlabel_prefix: bool


class Session:
    def __init__(self, srcdir: Path, confoverrides: dict[str, Any] | None,
                 issues: Issues) -> None:
        self.srcdir = srcdir.resolve()
        self.issues = issues
        self.confoverrides = confoverrides or {}
        self.app: Sphinx | None = None
        self.file_contents: dict[Path, tuple[list[str], str]] = {}
        self.docs: dict[str, DocModel] = {}
        self.baseline_messages: dict[str, list[tuple[str, str, str]]] = {}
        self.newlines: dict[Path, str] = {}

    # -- build -------------------------------------------------------------

    def build(self) -> None:
        import warnings

        status = io.StringIO()
        warning = io.StringIO()
        builddir = Path(tempfile.mkdtemp(prefix='sphinx-refactor-'))
        with warnings.catch_warnings(record=True) as caught:
            app = Sphinx(
                srcdir=self.srcdir,
                confdir=self.srcdir,
                outdir=builddir,
                doctreedir=builddir / '.doctrees',
                buildername='dummy',
                status=status,
                warning=warning,
                freshenv=True,
                confoverrides=self.confoverrides,
            )
            app.build()
        self.app = app
        self._caught_warnings = caught
        self.baseline_status = status.getvalue()
        self.baseline_warning = warning.getvalue()
        self.baseline_messages = self._parse_warning_stream(
            self.baseline_warning, app
        )

    @staticmethod
    def _parse_warning_stream(
        stream: str, app: Sphinx
    ) -> dict[str, list[tuple[str, str, str]]]:
        result: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for raw in stream.splitlines():
            line = _ANSI_RE.sub('', raw)
            m = _WARNING_RE.match(line)
            if not m:
                continue
            f = m.group('file')
            path = Path(f)
            try:
                rel = os.path.relpath(path, app.srcdir).replace(os.sep, '/')
            except ValueError:
                rel = f
            msg = re.sub(r'["\u2018\u2019].*?["\u2018\u2019]', '""', m.group('msg'))
            key = (m.group('level'), m.group('subtype') or '', msg)
            result[rel].append(key)
        return dict(result)

    # -- raw parsing -------------------------------------------------------

    def _raw_parse(self, docname: str) -> nodes.document:
        assert self.app is not None
        env = self.app.env
        env.prepare_settings(docname)
        filename = env.doc2path(docname)
        content = filename.read_text(encoding='utf-8')
        filetype = get_filetype(self.app.config.source_suffix, filename)
        parser = self.app.registry.create_source_parser(
            filetype, config=self.app.config, env=env
        )
        settings = _get_settings(
            standalone.Reader,
            parser,
            defaults={'traceback': True, **env.settings},
            read_config_files=True,
        )
        settings._source = str(filename)
        reporter = LoggingReporter(
            source=str(filename),
            report_level=5,
            halt_level=5,
            debug=0,
            error_handler=lambda x: '?',
        )
        document = nodes.document(settings, reporter, source=str(filename))
        document.note_source(str(filename), -1)
        saved_current = env.current_document
        saved_ref_context = dict(env.ref_context)
        try:
            with sphinx_domains(env):
                parser.parse(content, document)
        finally:
            # mirror Builder.read_doc cleanup so parser/context state does
            # not leak into later build phases
            from sphinx.environment import _CurrentDocument
            env.current_document = _CurrentDocument()
            env.ref_context.clear()
        document.current_source = document.current_line = None
        return document

    # -- per-doc model -----------------------------------------------------

    def model(self, docname: str) -> DocModel:
        if docname in self.docs:
            return self.docs[docname]
        assert self.app is not None
        env = self.app.env
        if docname not in env.all_docs:
            self.issues.err(
                'doc-not-found',
                f'document {docname!r} is not part of the analysed project '
                '(not reached from the master document/toctrees)',
            )
            self.issues.raise_if_errors()
        path = Path(env.doc2path(docname)).resolve()
        if path not in self.file_contents:
            self.file_contents[path] = _read_physical(path)
            self.newlines[path] = self.file_contents[path][1]
        filetype = get_filetype(self.app.config.source_suffix, path)
        parser_obj = self.app.registry.get_source_parser(filetype)
        from sphinx.parsers import RSTParser

        if not issubclass(parser_obj, RSTParser):
            self.issues.err(
                'not-rst',
                f'document {docname!r} uses parser {parser_obj.__name__!r}; '
                'only native reStructuredText documents are supported',
                path,
            )
            self.issues.raise_if_errors()

        raw = self._raw_parse(docname)
        read_tree = env.get_doctree(docname)
        lines = self._build_expansion(docname, path, raw)
        pos_index = self._build_pos_index(lines)
        occ: dict[tuple[Path, int], list[int]] = {
            key: list(reversed(idxs)) for key, idxs in pos_index.items()
        }
        sections = self._build_sections(raw, read_tree, lines, occ)
        section_at: dict[int, SectionInfo] = {}
        for sec in sections:
            section_at[sec.start] = sec

        defs_by_name, defs_by_id = self._build_defs(
            raw, read_tree, lines, occ
        )
        targets_by_name = self._build_targets(raw)
        contexts = self._build_contexts(lines)
        chars_by_depth: list[str] = []
        for sec in sections:
            if sec.depth >= len(chars_by_depth):
                chars_by_depth.extend([''] * (sec.depth + 1 - len(chars_by_depth)))
            if not chars_by_depth[sec.depth]:
                chars_by_depth[sec.depth] = sec.char
        tokens_by_host, token_list = self._build_tokens(raw, lines, occ)
        autosection = 'sphinx.ext.autosectionlabel' in (
            self.app.config.extensions or []
        )
        prefix = bool(
            getattr(self.app.config, 'autosectionlabel_prefix_document', False)
        ) if autosection else False
        model = DocModel(
            docname=docname,
            path=path,
            lines=lines,
            instances=self._expander_instances,
            raw=raw,
            read_tree=read_tree,
            sections=sections,
            section_at=section_at,
            defs_by_name=defs_by_name,
            defs_by_id=defs_by_id,
            tokens=token_list,
            tokens_by_host=tokens_by_host,
            targets_by_name=targets_by_name,
            pos_index=pos_index,
            occ=occ,
            chars_by_depth=chars_by_depth,
            contexts=contexts,
            autosection=autosection,
            sectionlabel_prefix=prefix,
        )
        self.docs[docname] = model
        return model

    def _build_expansion(
        self, docname: str, path: Path, raw: nodes.document
    ) -> list[SLine]:
        protected = self._protected_ranges(raw)
        expander = _Expander(self.srcdir, self.issues, self.file_contents,
                             protected, docdir=path.parent)
        try:
            lines = expander.expand(path)
        except ExpansionError as exc:
            self.issues.err('include-error', str(exc).split(': ', 1)[-1],
                            exc.path, exc.line)
            self.issues.raise_if_errors()
        self._expander_instances = expander.instances
        self._validate_expansion(docname, path, raw, lines, expander)
        return lines

    def _protected_ranges(
        self, raw: nodes.document
    ) -> dict[Path, list[tuple[int, int]]]:
        """Physical line ranges that must not be scanned for directives.

        Literal blocks, doctest/math blocks and comments may contain text
        looking like an ``include`` directive.
        """
        protected_classes = (
            nodes.literal_block,
            nodes.doctest_block,
            nodes.math_block,
            nodes.comment,
        )
        ranges: dict[Path, list[tuple[int, int]]] = defaultdict(list)
        for node in raw.findall(nodes.Element):
            if not isinstance(node, protected_classes):
                continue
            if not node.source or node.line is None:
                continue
            origin = Path(node.source).resolve()
            if origin not in self.file_contents:
                try:
                    self.file_contents[origin] = _read_physical(origin)
                except OSError:
                    continue
            phys = self.file_contents[origin][0]
            start = node.line - 1
            end = _block_extent(
                [type('L', (), {'text': t})() for t in phys],
                start,
                len(phys),
            )
            # literal_block nodes mark the first content line, not a possible
            # ``::`` introducer; expand to cover the introducing line too.
            ranges[origin].append((start + 1, end + 1))
        return dict(ranges)

    def _validate_expansion(
        self,
        docname: str,
        path: Path,
        raw: nodes.document,
        lines: list[SLine],
        expander: _Expander,
    ) -> None:
        """Cross-check parsed section origins against the expansion map."""
        origin_occ: dict[Path, dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx, sl in enumerate(lines):
            origin_occ[sl.origin][sl.oln].append(idx)
        queues: dict[tuple[Path, int], list[int]] = {
            (o, oln): list(idxs)
            for o, byline in origin_occ.items()
            for oln, idxs in byline.items()
        }
        used: set[int] = set()
        problems: list[str] = []
        for node in raw.findall(nodes.section):
            if node.source is None or node.line is None:
                continue
            origin = Path(node.source).resolve()
            title_oln = node.line - 1
            key = (origin, title_oln)
            avail = [i for i in queues.get(key, []) if i not in used]
            if not avail:
                problems.append(f'section from {origin.name}:{title_oln}')
                continue
            used.add(avail[0])
        if problems:
            self.issues.err(
                'include-mismatch',
                'the expanded include structure does not match the parsed '
                'document (include content is modified by an event hook or '
                'uses unsupported options): ' + '; '.join(problems),
                path,
            )
        # nodes coming from outside the source directory (standard includes,
        # data files) are not supported inside movable content.
        for node in raw.findall(nodes.Element):
            if node.source and _is_outside(Path(node.source).resolve(),
                                          self.srcdir):
                self.issues.err(
                    'external-origin',
                    f'content originates from outside the source directory: '
                    f'{node.source}',
                    node.source,
                    node.line,
                )

    def _build_sections(
        self,
        raw: nodes.document,
        read_tree: nodes.document,
        lines: list[SLine],
        occ: dict[tuple[Path, int], list[int]],
    ) -> list[SectionInfo]:
        # The post-read tree carries the merged explicit target names/ids
        # (attached targets are folded into section nodes by transforms).
        read_sections = {
            (Path(s.source).resolve() if s.source else None, s.line): s
            for s in read_tree.findall(nodes.section)
            if s.source is not None and s.line is not None
        }
        infos: list[SectionInfo] = []
        for node in raw.findall(nodes.section):
            if node.source is None or node.line is None:
                continue
            origin = Path(node.source).resolve()
            under_oln = node.line
            title_oln = under_oln - 1
            q = occ.get((origin, title_oln))
            if not q:
                continue
            start = q.pop()
            depth = _depth_of(raw, node)
            title_text = lines[start].text.rstrip('\r\n')
            underline_text = (
                lines[start + 1].text.rstrip('\r\n') if start + 1 < len(lines)
                else ''
            )
            char = _is_underline(underline_text) or '='
            overline_oln: int | None = None
            if start - 1 >= 0:
                prev = lines[start - 1].text.rstrip('\r\n')
                if (prev and _is_underline(prev) == char
                        and lines[start - 1].origin == origin
                        and len(prev) >= len(title_text)):
                    overline_oln = lines[start - 1].oln
            read_node = read_sections.get((origin, under_oln), node)
            infos.append(SectionInfo(
                node=node,
                depth=depth,
                origin=origin,
                title_oln=title_oln,
                under_oln=under_oln,
                char=char,
                overline_oln=overline_oln,
                start=start,
                end=len(lines),
                names=list(read_node['names']),
                ids=list(read_node['ids']),
            ))
        infos.sort(key=lambda s: s.start)
        # ends + lead targets
        for idx, sec in enumerate(infos):
            for later in infos[idx + 1:]:
                if later.depth <= sec.depth:
                    sec.end = later.start
                    break
            else:
                sec.end = len(lines)
            sec.lead_olns = self._lead_targets(sec, lines, infos, idx)
        return infos

    def _lead_targets(
        self,
        sec: SectionInfo,
        lines: list[SLine],
        infos: list[SectionInfo],
        idx: int,
    ) -> list[int]:
        """Physical explicit-target lines attached to the section title."""
        result: list[int] = []
        if sec.origin != lines[sec.start].origin:
            return result
        # walk backwards over immediately adjacent blank/target lines
        i = sec.start - 1
        while i >= 0:
            sl = lines[i]
            if sl.origin != sec.origin:
                break
            text = sl.text.rstrip('\r\n')
            if not text.strip():
                i -= 1
                continue
            m = _EXPLICIT_TARGET_RE.match(text)
            if m and _norm_name(m.group('name')) in {
                _norm_name(n) for n in sec.names
            }:
                result.append(sl.oln)
                i -= 1
                continue
            break
        result.reverse()
        return result

    def _build_pos_index(
        self, lines: list[SLine]
    ) -> dict[tuple[Path, int], list[int]]:
        idx_map: dict[tuple[Path, int], list[int]] = defaultdict(list)
        for i, sl in enumerate(lines):
            idx_map[(sl.origin, sl.oln)].append(i)
        return dict(idx_map)

    def _build_defs(
        self,
        raw: nodes.document,
        read_tree: nodes.document,
        lines: list[SLine],
        occ: dict[tuple[Path, int], list[int]],
    ) -> tuple[dict[str, list[DefBlock]], dict[str, DefBlock]]:
        by_name: dict[str, list[DefBlock]] = defaultdict(list)
        by_id: dict[str, DefBlock] = {}
        read_by_pos: dict[
            tuple[type, Path, int], nodes.Element
        ] = {}
        for rnode in read_tree.findall(nodes.Element):
            if not isinstance(rnode, (
                nodes.footnote, nodes.citation,
                nodes.substitution_definition,
            )):
                continue
            if rnode.source is None or rnode.line is None:
                continue
            key = (type(rnode), Path(rnode.source).resolve(), rnode.line)
            read_by_pos.setdefault(key, rnode)
        for node in raw.findall(nodes.Element):
            kind: str | None = None
            name: str | None = None
            read_node2 = read_by_pos.get((
                type(node),
                Path(node.source).resolve() if node.source else None,
                node.line,
            )) if node.source and node.line else None
            if isinstance(node, nodes.substitution_definition):
                kind = 'substitution'
                if read_node2 is not None and read_node2.get('names'):
                    name = read_node2['names'][0]
                else:
                    name = node['names'][0] if node.get('names') else None
            elif isinstance(node, nodes.footnote):
                kind = 'footnote'
                if read_node2 is not None and read_node2.get('names'):
                    name = read_node2['names'][0]
                else:
                    label = node.next_node(nodes.label)
                    name = label.astext() if label is not None else None
            elif isinstance(node, nodes.citation):
                kind = 'citation'
                if read_node2 is not None and read_node2.get('names'):
                    name = read_node2['names'][0]
                else:
                    label = node.next_node(nodes.label)
                    name = label.astext() if label is not None else None
            if kind is None or name is None or node.line is None or not node.source:
                continue
            origin = Path(node.source).resolve()
            start_oln = node.line
            q = occ.get((origin, start_oln))
            if not q:
                continue
            start_i = q.pop()
            end_i = _block_extent(lines, start_i, len(lines))
            text = ''.join(sl.text for sl in lines[start_i:end_i])
            read_node = read_by_pos.get((type(node), origin, start_oln))
            ids = list(read_node['ids']) if read_node is not None else list(
                node['ids']
            )
            names = (
                list(read_node['names']) if read_node is not None
                else list(node['names'])
            )
            block = DefBlock(
                kind=kind,
                name=name,
                node=node,
                origin=origin,
                oln=start_oln,
                end_oln=lines[end_i - 1].oln if end_i > start_i else start_oln,
                text=text,
                extra={'ids': ids,
                       'names': names,
                       'expanded_start': start_i,
                       'expanded_end': end_i},
            )
            by_name[name].append(block)
            for nid in ids:
                by_id[nid] = block
        return dict(by_name), by_id

    def _build_targets(self, raw: nodes.document) -> dict[str, Any]:
        targets: dict[str, Any] = {}
        for node in raw.findall(nodes.Element):
            for name in node.get('names', []):  # noqa: SIM118
                targets.setdefault(_norm_name(name), node)
        return targets

    def _build_contexts(
        self, lines: list[SLine]
    ) -> list[tuple[int, str, str | None]]:
        events: list[tuple[int, str, str | None]] = []
        for i, sl in enumerate(lines):
            text = sl.text.rstrip('\r\n')
            if not text or text[0].isspace():
                continue
            m = _CURRENTMODULE_RE.match(text)
            if m:
                events.append((i, 'module', m.group('arg').strip() or None))
                continue
            m = _DEFAULT_DOMAIN_RE.match(text)
            if m:
                events.append((i, 'domain', m.group('arg')))
                continue
            m = _DEFAULT_ROLE_RE.match(text)
            if m:
                events.append((i, 'role', m.group('arg')))
        return events

    def _build_tokens(
        self,
        raw: nodes.document,
        lines: list[SLine],
        occ: dict[tuple[Path, int], list[int]],
    ) -> tuple[dict[tuple[Path, int], list[Token]], list[Token]]:
        tokens: list[Token] = []
        # walk document-order; group per host block
        for node in raw.findall(nodes.TextElement):
            if node.source is None or node.line is None:
                continue
            host_origin = Path(node.source).resolve()
            host_oln = node.line
            q = occ.get((host_origin, host_oln))
            if not q:
                continue
            start_i = q[0]  # do not consume: block anchors stay reusable
            if isinstance(node, nodes.title):
                end_i = min(start_i + 3, len(lines))
            else:
                bound = len(lines)
                end_i = _paragraph_extent(lines, start_i, bound)
            end_oln = lines[end_i - 1].oln
            ordinals: dict[str, int] = defaultdict(int)
            for child in node.children:
                kind = self._token_kind(child)
                if kind is None:
                    continue
                rawsource = child.rawsource or self._rawsource_fallback(child)
                if not rawsource:
                    continue
                ordn = ordinals[kind]
                ordinals[kind] += 1
                tokens.append(Token(
                    kind=kind,
                    node=child,
                    host_origin=host_origin,
                    host_oln=host_oln,
                    host_end_oln=end_oln,
                    ordinal=ordn,
                    rawsource=rawsource,
                    host_index=start_i,
                    host_end_index=end_i,
                ))
        by_host: dict[tuple[Path, int], list[Token]] = defaultdict(list)
        for tok in tokens:
            by_host[(tok.host_origin, tok.host_oln)].append(tok)
        return by_host, tokens

    @staticmethod
    def _token_kind(node: Any) -> str | None:
        if isinstance(node, addnodes.pending_xref):
            return 'xref'
        if isinstance(node, nodes.reference):
            return 'reference'
        if isinstance(node, nodes.citation_reference):
            return 'citation'
        if isinstance(node, nodes.footnote_reference):
            return 'footnote'
        if isinstance(node, nodes.substitution_reference):
            return 'substitution'
        return None

    @staticmethod
    def _rawsource_fallback(node: Any) -> str:
        name = node.get('refname') or node.get('name')
        if isinstance(node, nodes.substitution_reference):
            return f'|{name}|'
        if isinstance(node, nodes.footnote_reference):
            return f'[#{name}]_' if name else '[#]_'
        if isinstance(node, nodes.citation_reference):
            return f'[{name}]_'
        return ''


def _depth_of(doc: nodes.document, node: nodes.Node) -> int:
    depth = 0
    parent = node.parent
    while parent is not None and parent is not doc:
        if isinstance(parent, nodes.section):
            depth += 1
        parent = parent.parent
    return depth


def _is_outside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def move_section(
    srcdir: str | os.PathLike[str],
    source_docname: str,
    source_line: int,
    target_docname: str,
    target_parent_line: int,
    *,
    apply: bool = False,
    confoverrides: dict[str, Any] | None = None,
) -> MoveSectionResult:
    """Move a section from one rST document to another.

    :param srcdir: Sphinx project source directory.
    :param source_docname: docname containing the section to move.
    :param source_line: 1-based line number of the section **title text** in
        the source document's physical file.
    :param target_docname: docname of the destination document.
    :param target_parent_line: 1-based title-text line of the section that
        becomes the parent, or ``0`` to move to the document root.
    :param apply: when false (the default) only return a patch preview and
        write nothing; when true, write the files after verifying the result
        with a throwaway build.
    :param confoverrides: optional Sphinx configuration overrides used for
        the analysis and verification builds.

    The analysis uses the bindings of a real build performed *before* the
    move.  When the selection is invalid, the target lies inside the moved
    subtree, or a required dependency is unsupported,
    :class:`RefactorAborted` is raised with locatable diagnostics and no
    source file is modified.
    """
    from sphinx.refactor._planner import Planner, validate_result

    srcdir_path = Path(srcdir).resolve()
    if not srcdir_path.is_dir():
        msg = f'source directory does not exist: {srcdir_path}'
        raise RefactorAborted(Diagnostic('error', 'no-srcdir', msg))
    issues = Issues(srcdir_path)
    session = Session(srcdir_path, confoverrides, issues)
    session.build()
    planner = Planner(
        session,
        source_docname,
        int(source_line),
        target_docname,
        int(target_parent_line),
    )
    plan = planner.plan()
    issues.raise_if_errors()
    texts: dict[Path, str] = plan['texts']

    # Verification build in a temporary copy -- always run, even for preview,
    # so that a proposed patch is known to build.
    validate_result(session, texts, issues)
    issues.raise_if_errors()

    patches: list[FilePatch] = []
    for path, new_text in sorted(texts.items()):
        old_text, _nl = session.file_contents[path]
        patches.append(FilePatch(path=path, old_text=''.join(old_text),
                                 new_text=new_text))

    result = MoveSectionResult(
        source_docname=source_docname,
        target_docname=target_docname,
        source_title=planner._title(planner.S),
        target_parent_title=(planner._title(planner.P)
                             if planner.P is not None else None),
        patches=patches,
        diagnostics=list(issues.warnings),
        notes=list(issues.notes),
        applied=False,
    )

    if apply:
        for patch in patches:
            newline = session.newlines.get(patch.path, '\n')
            text = patch.new_text
            if newline == '\r\n':
                text = text.replace('\r\n', '\n').replace('\n', '\r\n')
            with open(patch.path, 'w', encoding='utf-8', newline='') as f:
                f.write(text)
        result.applied = True
    return result

