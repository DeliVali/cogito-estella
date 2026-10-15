"""Code to graph by syntax, not by model: (subject, defines|contains|calls|imports|inherits,
object) with the line each fact comes from. The prose heads were trained on sentences and
name only a closed vocabulary; a parse names every function, class and import exactly, in
any language tree-sitter reads (the `[full]` extra)."""
from __future__ import annotations

import bisect
from pathlib import Path

LANGUAGES = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".swift": "swift", ".go": "go",
    ".rs": "rust", ".rb": "ruby", ".php": "php", ".cs": "csharp", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".m": "objc", ".mm": "objc", ".dart": "dart",
    ".scala": "scala", ".lua": "lua", ".sh": "bash", ".bash": "bash", ".ex": "elixir",
    ".exs": "elixir", ".hs": "haskell", ".jl": "julia", ".r": "r", ".vue": "vue",
    ".svelte": "svelte", ".gradle": "groovy", ".groovy": "groovy", ".zig": "zig",
}
_DEF_SUFFIXES = ("_definition", "_declaration", "_signature", "function_item", "struct_item",
                 "enum_item", "trait_item", "impl_item")
_DEF_WORDS = ("function", "method", "class", "interface", "struct", "enum", "trait",
              "protocol", "object", "module", "constructor")
_CALLS = {"call", "call_expression", "method_invocation", "function_call",
          "invocation_expression", "method_call", "message_expression"}
_IMPORTS = {"import_statement", "import_from_statement", "import_declaration", "import_header",
            "import_spec", "library_import", "use_declaration", "using_directive",
            "preproc_include", "namespace_use_declaration", "import"}
_BASES = {"superclasses", "delegation_specifier", "superclass", "class_heritage", "delegation_specifiers",
          "inheritance_specifier", "type_inheritance_clause", "base_list", "super_interfaces",
          "extends_clause", "implements_clause", "superclass_list"}
_NAMES = {"identifier", "type_identifier", "simple_identifier", "property_identifier",
          "field_identifier", "name", "constant", "user_type", "scoped_identifier"}


def language(name: str) -> str | None:
    return LANGUAGES.get(Path(name).suffix.lower())


def available() -> bool:
    try:
        import tree_sitter_language_pack  # noqa: F401
    except ImportError:
        return False
    return True


def _is_def(t: str) -> bool:
    return t.endswith(_DEF_SUFFIXES) and any(w in t for w in _DEF_WORDS) \
        and "parameter" not in t and "variable" not in t


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "ignore")


def _name(node, src: bytes) -> str | None:
    """The declared name: the `name` field, else the first identifier-like child."""
    n = node.child_by_field_name("name")
    if n is not None:
        return _text(n, src).strip()
    for c in node.children:
        if c.type in _NAMES:
            return _text(c, src).strip()
    return None


def _last_name(node, src: bytes) -> str | None:
    """`client.post(url)` calls `post`: the last identifier before the argument list."""
    target = node.child_by_field_name("function") or node.child_by_field_name("name") \
        or (node.children[0] if node.children else None)
    if target is None:
        return None
    best = None
    stack = [target]
    while stack:
        n = stack.pop()
        if (n.type in _NAMES or n.type.endswith("identifier")) \
                and (best is None or n.start_byte >= best.start_byte):
            best = n
        if "argument" not in n.type and n.type not in ("call_suffix", "annotated_lambda"):
            stack.extend(n.children)
    return _text(best, src).strip() if best is not None else None


def _import_target(node, src: bytes) -> str:
    raw = " ".join(_text(node, src).split())
    for kw in ("import", "from", "use", "using", "#include", "package:", "library"):
        raw = raw.replace(kw, " ")
    return " ".join(raw.replace(";", " ").replace("'", " ").replace('"', " ").split())[:120]


def extract(text: str, source: str, offsets: list[int]) -> list[dict]:
    """Edges for one code file; `offsets` are the start offsets of its stored lines, so each
    edge points at the line (`sent_idx`) that holds it. Empty when the language is unknown
    or tree-sitter is not installed."""
    lang = language(source)
    if lang is None or not offsets:
        return []
    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(lang)
    except Exception:                             # noqa: BLE001 - an unsupported grammar is no graph
        return []
    src = text.encode("utf-8")
    root = parser.parse(src).root_node
    module = Path(source).name
    # byte -> char offset, for the line lookup over character offsets
    char_at = {}

    def line_of(node) -> int:
        b = node.start_byte
        if b not in char_at:
            char_at[b] = len(src[:b].decode("utf-8", "ignore"))
        return max(0, bisect.bisect_right(offsets, char_at[b]) - 1)

    edges: list[dict] = []
    seen: set[tuple] = set()

    def add(s, r, o, node):
        if not s or not o or s == o:
            return
        key = (s, r, o, line_of(node))
        if key in seen:
            return
        seen.add(key)
        edges.append({"s": s, "r": r, "o": o, "s_span": None, "o_span": None, "r_class": r,
                      "r_lex": r, "pattern": "syntax", "swapped": False,
                      "sent_idx": key[3]})

    def walk(node, scope: str):
        t = node.type
        here = scope
        if _is_def(t):
            name = _name(node, src)
            if name:
                add(scope, "contains" if scope != module else "defines", name, node)
                for c in node.children:
                    if c.type in _BASES or (lang == "python" and c.type == "argument_list"):
                        for b in c.children:
                            bn = _name(b, src) if b.type not in _NAMES else _text(b, src)
                            if bn and b.type not in ("(", ")", ",", ":"):
                                add(name, "inherits", bn.strip(), b)
                here = name
        elif t in _CALLS:
            callee = _last_name(node, src)
            if callee:
                add(scope, "calls", callee, node)
        elif t in _IMPORTS and node.parent is not None and node.parent.type not in _IMPORTS:
            target = _import_target(node, src)
            if target:
                add(module, "imports", target, node)
        for c in node.children:
            walk(c, here)

    walk(root, module)
    return edges
