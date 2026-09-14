"""Direction 4, Phase A: build the CodeFact items (Python next-token counterfactuals with a category taxonomy).

Sources (data/codefact/raw/, downloaded from the Hugging Face Hub; see build_stats.md for licences):
  HumanEval (openai/openai_humaneval, MIT): prompt + canonical_solution;
  MBPP (google-research-datasets/mbpp, CC-BY-4.0): `code` of the full split (train/test/validation/prompt);
  CodeSearchNet Python test split (code-search-net/code_search_net; corpus restricted at collection time to GitHub
  repositories whose licence permits redistribution; per-function repo/URL recorded): a seed-0 sample of functions.
Pre-processing: CRLF -> LF, trailing whitespace stripped, docstrings removed, units must parse with `ast`.

Categories (prefix ends just before the true token; foil is appended to the same prefix; subject = span to noise):
  S1 closing bracket   ')' ']' '}' vs another closer; subject = the matching opener.
  S2 block keyword     else/elif/except/finally at block start vs a sibling keyword; subject = the if/for/while/try head.
  S3 keyword completion ' in' after 'for x' (foil ','); ':' after an if/elif/while header (foil ' and');
                       ' import' after 'from pkg' (foil '.'); subject = the for/if/while/from token.
  R1 variable recall   a Name load bound earlier in the function (params, assignments, loop targets), foil = the most
                       recently bound other name; subject = the definition site. n_prior_occurrences <= 2 recorded.
  R2 attribute recall  attribute of a local with a literal-typed definition (list/dict/set/str) or of an imported
                       module; foil = another attribute of the same type/module; subject = the literal / module name.
  R3 constant recall   a str / single-digit int literal that occurred earlier; foil = another earlier literal of the
                       same kind; subject = the first occurrence.
Every candidate is then resolved with the Qwen3 and Mixtral tokenizers (moetrace.ext4_data.resolve_item) to record
single-token validity; the items file keeps ALL candidates (validity flags per tokenizer), capped per unit and category.

Usage: python scripts/ext4_build_codefact.py [--csn-sample 4000] [--cap-per-unit 2] [--max-per-category 1200]
Writes data/codefact/items.jsonl, build_stats.md, samples.md, csn_sample_repos.csv
"""
import argparse, ast, builtins, collections, io, json, keyword, os, random, sys, time, tokenize, warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd

ROOT = "/home/ubuntu/MOE"
RAW = os.path.join(ROOT, "data", "codefact", "raw")
OUT = os.path.join(ROOT, "data", "codefact")
BUILTINS = set(dir(builtins)) | set(keyword.kwlist) | {"self", "cls", "_"}
CLOSER_OF = {"(": ")", "[": "]", "{": "}"}
OPENER_OF = {v: k for k, v in CLOSER_OF.items()}
DEFAULT_FOIL_CLOSER = {")": "]", "]": ")", "}": "]"}
TYPE_ATTRS = {
    "list": ["append", "extend", "insert", "pop", "remove", "sort", "index", "count", "reverse", "clear", "copy"],
    "dict": ["get", "items", "keys", "values", "pop", "update", "setdefault", "copy", "clear"],
    "set": ["add", "remove", "discard", "update", "union", "intersection", "difference", "pop", "clear"],
    "str": ["join", "split", "strip", "replace", "format", "lower", "upper", "startswith", "endswith", "find", "encode"],
}
MODULE_ATTRS = {
    "numpy": ["array", "zeros", "ones", "arange", "linspace", "sum", "mean", "dot", "sqrt", "exp", "log", "max", "min", "abs",
              "where", "concatenate", "reshape", "random", "argmax", "empty", "std", "pi", "float32", "int32", "ndarray", "asarray"],
    "os": ["path", "getcwd", "listdir", "environ", "makedirs", "remove", "mkdir", "getenv", "sep", "walk", "chdir", "stat", "name", "system"],
    "os.path": ["join", "exists", "isfile", "isdir", "basename", "dirname", "abspath", "splitext", "expanduser", "getsize", "split", "realpath"],
    "re": ["match", "search", "sub", "compile", "findall", "split", "finditer", "fullmatch", "escape", "IGNORECASE", "MULTILINE", "DOTALL"],
    "math": ["sqrt", "floor", "ceil", "pi", "log", "exp", "pow", "fabs", "sin", "cos", "factorial", "gcd", "inf", "isnan", "log2", "log10", "e", "atan2", "tan", "hypot", "radians", "degrees"],
    "json": ["loads", "dumps", "load", "dump", "JSONDecodeError", "JSONEncoder"],
    "random": ["random", "randint", "choice", "shuffle", "seed", "uniform", "sample", "randrange", "gauss", "choices"],
    "sys": ["argv", "exit", "path", "stdout", "stderr", "stdin", "version", "platform", "modules", "maxsize", "version_info"],
    "time": ["time", "sleep", "strftime", "localtime", "gmtime", "mktime", "perf_counter", "monotonic", "ctime", "strptime"],
    "collections": ["defaultdict", "Counter", "OrderedDict", "deque", "namedtuple", "ChainMap"],
    "itertools": ["chain", "product", "permutations", "combinations", "groupby", "islice", "count", "zip_longest", "accumulate", "cycle", "repeat", "starmap", "tee", "takewhile", "dropwhile", "compress"],
    "heapq": ["heappush", "heappop", "heapify", "nlargest", "nsmallest", "heappushpop", "heapreplace", "merge"],
    "string": ["ascii_lowercase", "ascii_uppercase", "ascii_letters", "digits", "punctuation", "whitespace", "printable", "Template", "hexdigits"],
    "datetime": ["datetime", "date", "timedelta", "time", "timezone"],
    "logging": ["getLogger", "info", "debug", "warning", "error", "basicConfig", "DEBUG", "INFO", "WARNING", "ERROR", "exception", "Formatter", "StreamHandler"],
    "subprocess": ["Popen", "PIPE", "run", "check_output", "call", "check_call", "STDOUT", "CalledProcessError"],
    "functools": ["partial", "wraps", "reduce", "lru_cache", "cmp_to_key", "total_ordering"],
    "copy": ["deepcopy", "copy"],
    "pandas": ["DataFrame", "Series", "read_csv", "concat", "merge", "to_datetime", "isnull", "Timestamp", "read_excel", "notnull", "MultiIndex", "Index"],
    "torch": ["tensor", "zeros", "ones", "cat", "stack", "nn", "randn", "no_grad", "from_numpy", "arange", "sum", "mean", "exp", "log", "matmul", "sigmoid", "softmax", "float32", "device", "cuda", "manual_seed"],
    "shutil": ["copy", "copyfile", "rmtree", "move", "copytree", "which", "copy2", "make_archive"],
    "struct": ["pack", "unpack", "calcsize", "unpack_from", "pack_into"],
    "base64": ["b64encode", "b64decode", "urlsafe_b64encode", "urlsafe_b64decode", "encodebytes", "decodebytes"],
    "hashlib": ["md5", "sha1", "sha256", "sha512", "new", "sha224"],
    "socket": ["socket", "AF_INET", "SOCK_STREAM", "error", "timeout", "gethostname", "gethostbyname", "SOCK_DGRAM", "inet_aton"],
    "urllib": ["parse", "request", "error"],
    "pickle": ["load", "dump", "loads", "dumps", "HIGHEST_PROTOCOL"],
    "operator": ["itemgetter", "attrgetter", "add", "mul", "sub", "methodcaller", "or_", "and_", "not_", "eq"],
    "inspect": ["getmembers", "isclass", "isfunction", "signature", "getargspec", "getsource", "ismethod", "getmodule", "stack", "currentframe", "getfullargspec", "iscoroutinefunction", "isroutine", "getdoc", "getfile", "ismodule"],
    "warnings": ["warn", "filterwarnings", "simplefilter", "catch_warnings", "resetwarnings"],
    "tempfile": ["mkdtemp", "NamedTemporaryFile", "mkstemp", "gettempdir", "TemporaryDirectory", "TemporaryFile"],
    "glob": ["glob", "iglob", "escape"],
    "codecs": ["open", "encode", "decode", "getwriter", "getreader", "lookup", "BOM_UTF8"],
    "traceback": ["format_exc", "print_exc", "extract_stack", "format_exception", "print_stack", "extract_tb", "format_tb"],
    "threading": ["Thread", "Lock", "Event", "current_thread", "RLock", "Timer", "Condition", "Semaphore", "local"],
    "unittest": ["TestCase", "main", "mock", "skip", "skipIf", "TestSuite", "TextTestRunner", "expectedFailure"],
    "signal": ["signal", "SIGINT", "SIGTERM", "SIG_IGN", "alarm", "SIGALRM", "SIGKILL", "SIG_DFL", "getsignal", "SIGHUP"],
    "uuid": ["uuid4", "uuid1", "UUID", "uuid5", "uuid3", "NAMESPACE_DNS"],
    "types": ["FunctionType", "ModuleType", "MethodType", "GeneratorType", "SimpleNamespace", "BuiltinFunctionType", "LambdaType"],
    "io": ["BytesIO", "StringIO", "open", "TextIOWrapper", "IOBase", "BufferedReader", "SEEK_END", "SEEK_SET"],
    "abc": ["ABC", "abstractmethod", "ABCMeta", "abstractproperty"],
    "csv": ["reader", "writer", "DictReader", "DictWriter", "QUOTE_ALL", "QUOTE_MINIMAL", "Sniffer", "excel"],
    "zipfile": ["ZipFile", "ZIP_DEFLATED", "is_zipfile", "ZIP_STORED", "BadZipFile", "ZipInfo"],
    "platform": ["system", "machine", "python_version", "node", "release", "platform", "architecture", "uname", "version", "processor"],
    "errno": ["ENOENT", "EEXIST", "EACCES", "EINTR", "EAGAIN", "EPIPE", "EINVAL", "ECONNREFUSED", "ENOTDIR", "EISDIR"],
    "textwrap": ["dedent", "wrap", "fill", "indent", "TextWrapper", "shorten"],
    "difflib": ["SequenceMatcher", "unified_diff", "get_close_matches", "Differ", "ndiff", "HtmlDiff"],
    "getpass": ["getuser", "getpass"],
    "fnmatch": ["fnmatch", "filter", "fnmatchcase", "translate"],
    "binascii": ["hexlify", "unhexlify", "b2a_hex", "a2b_hex", "crc32", "b2a_base64", "a2b_base64", "Error"],
    "argparse": ["ArgumentParser", "Namespace", "FileType", "SUPPRESS", "RawTextHelpFormatter", "ArgumentTypeError", "REMAINDER", "Action"],
    "bisect": ["bisect_left", "bisect_right", "insort", "bisect", "insort_left", "insort_right"],
    "decimal": ["Decimal", "getcontext", "ROUND_HALF_UP", "InvalidOperation", "ROUND_DOWN", "localcontext"],
    "fractions": ["Fraction", "gcd"],
    "statistics": ["mean", "median", "stdev", "variance", "mode", "pstdev", "pvariance", "harmonic_mean"],
    "cmath": ["sqrt", "exp", "log", "phase", "polar", "rect", "pi", "sin", "cos"],
}
STR_TYPES = {ast.List: "list", ast.ListComp: "list", ast.Dict: "dict", ast.DictComp: "dict", ast.Set: "set", ast.SetComp: "set"}
KW_ELSE_LIKE = {"else", "elif", "except", "finally"}
HEADS = {"if", "for", "while", "try"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# sources and pre-processing
# ---------------------------------------------------------------------------------------------------------------
def normalise(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.rstrip() for l in code.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def strip_docstrings(src: str) -> str | None:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    spans = []  # (lineno, end_lineno, indent_of_docstring_line, body_len)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            st = body[0]
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                spans.append((st.lineno, st.end_lineno, st.col_offset, len(body)))
    if not spans:
        return src
    lines = src.split("\n")
    for ln, end, col, blen in sorted(spans, reverse=True):
        repl = [" " * col + "pass"] if blen == 1 else []
        lines[ln - 1: end] = repl
    out = "\n".join(lines)
    try:
        ast.parse(out)
    except (SyntaxError, ValueError):
        return None
    return out


def load_units(csn_sample: int, seed: int = 0, csn_extra: int = 0) -> list[dict]:
    units = []
    he = pd.read_parquet(os.path.join(RAW, "humaneval_test.parquet"))
    for _, r in he.iterrows():
        units.append(dict(unit_id=f"he:{r.task_id}", source="humaneval", licence="MIT", code=r.prompt + r.canonical_solution))
    for split in ("train", "test", "validation", "prompt"):
        mb = pd.read_parquet(os.path.join(RAW, f"mbpp_full_{split}.parquet"))
        for _, r in mb.iterrows():
            units.append(dict(unit_id=f"mbpp:{r.task_id}", source="mbpp", licence="CC-BY-4.0", code=r.code))
    csn = pd.read_parquet(os.path.join(RAW, "csn_python_test.parquet"))
    csn = csn[(csn.whole_func_string.str.len() <= 1500) & csn.whole_func_string.map(lambda s: s.isascii())]
    idx = list(csn.index)
    random.Random(seed).shuffle(idx)
    for i in idx[:csn_sample]:
        r = csn.loc[i]
        units.append(dict(unit_id=f"csn:{i}", source="csn", licence="per-repository (CodeSearchNet: redistribution-permitting licences only)",
                          code=r.whole_func_string, repo=r.repository_name, url=r.func_code_url))
    if csn_extra:
        ex = pd.read_parquet(os.path.join(RAW, "csn_python_validation.parquet"))
        ex = ex[(ex.whole_func_string.str.len() <= 1500) & ex.whole_func_string.map(lambda s: s.isascii())]
        idx = list(ex.index)
        random.Random(seed + 7).shuffle(idx)
        for i in idx[:csn_extra]:
            r = ex.loc[i]
            units.append(dict(unit_id=f"csnv:{i}", source="csn", licence="per-repository (CodeSearchNet: redistribution-permitting licences only)",
                              code=r.whole_func_string, repo=r.repository_name, url=r.func_code_url, extra=True))
    out = []
    n_bad = collections.Counter()
    for u in units:
        code = normalise(u["code"])
        code = strip_docstrings(code)
        if code is None:
            n_bad[u["source"]] += 1
            continue
        code = normalise(code)
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            n_bad[u["source"]] += 1
            continue
        u = dict(u, code=code, toks=toks, tree=ast.parse(code))
        out.append(u)
    log(f"units: {len(out)} usable ({collections.Counter(u['source'] for u in out)}), unusable {dict(n_bad)}")
    return out


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
class Unit:
    def __init__(self, u: dict):
        self.u = u
        self.code = u["code"]
        self.toks = u["toks"]
        self.tree = u["tree"]
        self.line_off = [0]
        for l in self.code.split("\n")[:-1]:
            self.line_off.append(self.line_off[-1] + len(l) + 1)

    def off(self, rc) -> int:
        r, c = rc
        return self.line_off[r - 1] + c

    def node_span(self, node) -> tuple[int, int]:
        return self.off((node.lineno, node.col_offset)), self.off((node.end_lineno, node.end_col_offset))

    def name_tokens_before(self, name: str, pos: int) -> int:
        return sum(1 for t in self.toks if t.type == tokenize.NAME and t.string == name and self.off(t.start) < pos)

    def first_on_line(self, i: int) -> bool:
        t = self.toks[i]
        for j in range(i - 1, -1, -1):
            p = self.toks[j]
            if p.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.ENCODING):
                if p.type == tokenize.COMMENT:
                    continue
                return True
            return p.start[0] != t.start[0]
        return True


def item(unit: Unit, category: str, subcategory: str, ans_start: int, true: str, foil: str, s0: int, s1: int, **kw) -> dict:
    return dict(category=category, subcategory=subcategory, source=unit.u["source"], licence=unit.u["licence"], unit_id=unit.u["unit_id"],
                repo=unit.u.get("repo", ""), url=unit.u.get("url", ""), prefix_text=unit.code[:ans_start], true_str=true, foil_str=foil,
                subject_start=s0, subject_end=s1, subject_text=unit.code[s0:s1], ans_char=ans_start, **kw)


# ---------------------------------------------------------------------------------------------------------------
# S1 closing bracket
# ---------------------------------------------------------------------------------------------------------------
def extract_s1(unit: Unit) -> list[dict]:
    out, stack = [], []
    for i, t in enumerate(unit.toks):
        if t.type != tokenize.OP:
            continue
        if t.string in CLOSER_OF:
            stack.append((t.string, unit.off(t.start), i))
        elif t.string in OPENER_OF:
            if not stack or stack[-1][0] != OPENER_OF[t.string]:
                return out  # malformed
            opener, o_off, o_i = stack.pop()
            if i - o_i < 2:
                continue  # empty brackets
            others = [CLOSER_OF[s[0]] for s in reversed(stack) if CLOSER_OF[s[0]] != t.string]
            foil = others[0] if others else DEFAULT_FOIL_CLOSER[t.string]
            out.append(item(unit, "S1", t.string, unit.off(t.start), t.string, foil, o_off, o_off + 1,
                            foil_kind="enclosing_open" if others else "default", n_inner_tokens=i - o_i - 1, depth=len(stack) + 1))
    return out


# ---------------------------------------------------------------------------------------------------------------
# S2 block keyword
# ---------------------------------------------------------------------------------------------------------------
def extract_s2(unit: Unit) -> list[dict]:
    out = []
    toks = unit.toks
    for i, t in enumerate(toks):
        if t.type != tokenize.NAME or t.string not in KW_ELSE_LIKE or not unit.first_on_line(i):
            continue
        col = t.start[1]
        head = None
        for j in range(i - 1, -1, -1):
            p = toks[j]
            if p.type != tokenize.NAME or p.start[1] != col or not unit.first_on_line(j):
                continue
            if p.string in HEADS:
                head = p
                break
            if p.string in ("elif", "except", "else"):
                continue
            break
        if head is None:
            continue
        kw, hd = t.string, head.string
        if kw == "else":
            foil = "finally" if hd == "try" else "elif"
        elif kw == "elif":
            if hd != "if":
                continue
            foil = "else"
        elif kw == "except":
            if hd != "try":
                continue
            foil = "finally"
        else:
            if hd != "try":
                continue
            foil = "except"
        h0 = unit.off(head.start)
        out.append(item(unit, "S2", f"{kw}_after_{hd}", unit.off(t.start), kw, foil, h0, h0 + len(hd), foil_kind="sibling_keyword",
                        head=hd, n_lines_between=t.start[0] - head.start[0]))
    return out


# ---------------------------------------------------------------------------------------------------------------
# S3 keyword completion
# ---------------------------------------------------------------------------------------------------------------
def extract_s3(unit: Unit) -> list[dict]:
    out = []
    toks = unit.toks
    for i, t in enumerate(toks):
        if t.type != tokenize.NAME:
            continue
        if t.string == "for" and i + 2 < len(toks):
            tgt, nxt = toks[i + 1], toks[i + 2]
            if tgt.type == tokenize.NAME and nxt.type == tokenize.NAME and nxt.string == "in" and tgt.string not in BUILTINS:
                a = unit.off(tgt.end)
                true = unit.code[a: unit.off(nxt.end)]
                if true.strip() != "in":
                    continue
                f0 = unit.off(t.start)
                out.append(item(unit, "S3", "for_in", a, true, ",", f0, f0 + 3, foil_kind="tuple_target", target=tgt.string))
        elif t.string in ("if", "elif", "while") and unit.first_on_line(i):
            # header: find the ':' ending the header on the same logical line at depth 0
            depth = 0
            j = i + 1
            end_tok = None
            while j < len(toks):
                p = toks[j]
                if p.type == tokenize.OP and p.string in CLOSER_OF:
                    depth += 1
                elif p.type == tokenize.OP and p.string in OPENER_OF:
                    depth -= 1
                elif p.type == tokenize.OP and p.string == ":" and depth == 0:
                    end_tok = p
                    break
                elif p.type in (tokenize.NEWLINE, tokenize.NL) and depth == 0:
                    break
                j += 1
            if end_tok is None or j - i < 3:
                continue
            prev = toks[j - 1]
            if prev.type not in (tokenize.NAME, tokenize.NUMBER):
                continue  # ')' + ':' merge into one token in both tokenizers; keep headers ending in a name/number
            if prev.type == tokenize.NAME and prev.string in ("None", "True", "False", "not", "and", "or", "in", "is"):
                if prev.string not in ("None", "True", "False"):
                    continue
            a = unit.off(prev.end)
            if unit.code[a: unit.off(end_tok.end)] != ":":
                continue
            f0 = unit.off(t.start)
            out.append(item(unit, "S3", f"{t.string}_colon", a, ":", " and", f0, f0 + len(t.string), foil_kind="conjunction",
                            n_header_tokens=j - i - 1))
        elif t.string == "from" and unit.first_on_line(i) and i + 2 < len(toks):
            j = i + 1
            while j < len(toks) and (toks[j].type == tokenize.NAME or (toks[j].type == tokenize.OP and toks[j].string == ".")):
                j += 1
            if j == i + 1 or toks[j].type != tokenize.NAME or toks[j].string != "import":
                continue
            last = toks[j - 1]
            if last.type != tokenize.NAME:
                continue
            a = unit.off(last.end)
            true = unit.code[a: unit.off(toks[j].end)]
            if true.strip() != "import":
                continue
            f0 = unit.off(t.start)
            out.append(item(unit, "S3", "from_import", a, true, ".", f0, f0 + 4, foil_kind="dotted_module", n_module_parts=(j - i) // 2 + 1))
    return out


# ---------------------------------------------------------------------------------------------------------------
# scopes for R1 / R2
# ---------------------------------------------------------------------------------------------------------------
def function_bindings(unit: Unit, fn) -> list[tuple[str, int, int, str]]:
    """(name, start, end, kind) of the binding occurrences inside a function, in source order (params first)."""
    b = []
    a = fn.args
    for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs) + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
        s0 = unit.off((arg.lineno, arg.col_offset))
        b.append((arg.arg, s0, s0 + len(arg.arg), "param"))
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            s0, s1 = unit.node_span(node)
            b.append((node.id, s0, s1, "store"))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                nm = (al.asname or al.name).split(".")[0]
                s0, s1 = unit.node_span(node)
                b.append((nm, s0, s1, "import"))
    b.sort(key=lambda x: x[1])
    return b


def module_imports(unit: Unit) -> dict[str, tuple[str, int, int]]:
    """alias -> (module, span of the module name in the import statement) for module-level `import x [as y]`."""
    out = {}
    for node in unit.tree.body:
        if isinstance(node, ast.Import):
            for al in node.names:
                alias = al.asname or al.name.split(".")[0]
                mod = al.name
                if mod in MODULE_ATTRS:
                    s0, s1 = unit.node_span(node)
                    seg = unit.code[s0:s1]
                    k = seg.find(mod)
                    if k >= 0:
                        out[alias] = (mod, s0 + k, s0 + k + len(mod))
    return out


def extract_r1(unit: Unit) -> list[dict]:
    out = []
    for fn in ast.walk(unit.tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        binds = function_bindings(unit, fn)
        if not binds:
            continue
        first_def = {}
        for nm, s0, s1, kind in binds:
            first_def.setdefault(nm, (s0, s1, kind))
        loads = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
        loads.sort(key=lambda n: (n.lineno, n.col_offset))
        for n in loads:
            nm = n.id
            if nm in BUILTINS or nm not in first_def or len(nm) < 2:
                continue
            a, _ = unit.node_span(n)
            d0, d1, kind = first_def[nm]
            if d1 > a - 1:
                continue
            prior = unit.name_tokens_before(nm, a)
            if prior < 1 or prior > 2:
                continue
            # foil: most recently bound other name before the usage
            others = [(s0, o) for o, s0, s1, k in binds if o != nm and s1 <= a and o not in BUILTINS and len(o) >= 2]
            if not others:
                continue
            foil = max(others)[1]
            lead = ""
            out.append(item(unit, "R1", kind, a, nm, foil, d0, d1, foil_kind="most_recent_other_binding",
                            n_prior_occurrences=prior, n_in_scope=len(set(o for _, o in others))))
    return out


def literal_type(node) -> str | None:
    for cls, ty in STR_TYPES.items():
        if isinstance(node, cls):
            return ty
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "str"
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("list", "dict", "set", "str", "sorted"):
        return "list" if node.func.id == "sorted" else node.func.id
    return None


def extract_r2(unit: Unit) -> list[dict]:
    out = []
    mods = module_imports(unit)
    for fn in ast.walk(unit.tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        typed = {}  # name -> (type, subject span, def pos)
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                ty = literal_type(node.value)
                if ty:
                    s0, s1 = unit.node_span(node.value)
                    nm = node.targets[0].id
                    if nm not in typed or s0 < typed[nm][1]:
                        typed[nm] = (ty, s0, s1)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                ty = literal_type(node.value)
                if ty:
                    s0, s1 = unit.node_span(node.value)
                    typed.setdefault(node.target.id, (ty, s0, s1))
        attrs = [n for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)]
        attrs.sort(key=lambda n: (n.lineno, n.col_offset))
        for n in attrs:
            base, attr = n.value.id, n.attr
            v0, v1 = unit.node_span(n.value)
            a = v1 + 1  # after the '.'
            if unit.code[v1:a] != "." or unit.code[a: a + len(attr)] != attr:
                continue
            if base in typed and typed[base][1] < v0:
                ty, s0, s1 = typed[base]
                cands = [x for x in TYPE_ATTRS[ty] if x != attr]
                sub = f"local_{ty}"
                known = attr in TYPE_ATTRS[ty]
            elif base in mods and mods[base][1] < v0:
                mod, s0, s1 = mods[base]
                cands = [x for x in MODULE_ATTRS[mod] if x != attr]
                sub = f"module_{mod}"
                known = attr in MODULE_ATTRS[mod]
            else:
                continue
            if not cands:
                continue
            out.append(item(unit, "R2", sub, a, attr, cands[0], s0, s1, foil_kind="same_type_attribute", attr_known=known, base_name=base))
    return out


def extract_r3(unit: Unit) -> list[dict]:
    out = []
    consts = [n for n in ast.walk(unit.tree) if isinstance(n, ast.Constant) and (isinstance(n.value, str) or (type(n.value) is int and 2 <= n.value <= 9))]
    consts.sort(key=lambda n: (n.lineno, n.col_offset))
    seen = {}  # (kind, value) -> first content span
    parent = {}
    for node in ast.walk(unit.tree):
        for ch in ast.iter_child_nodes(node):
            parent[ch] = node
    for n in consts:
        s0, s1 = unit.node_span(n)
        seg = unit.code[s0:s1]
        if isinstance(n.value, str):
            kind = "str"
            if len(seg) < 3 or seg[0] not in "\"'" or seg[-1] != seg[0] or seg[:3] in ('"""', "'''"):
                continue
            content = seg[1:-1]
            if not content or len(content) > 16 or "\\" in content or content != n.value:
                continue
            c0, c1 = s0 + 1, s1 - 1
        else:
            kind = "int"
            if seg != str(n.value):
                continue
            content, c0, c1 = seg, s0, s1
        key = (kind, n.value)
        if key not in seen:
            p = parent.get(n)
            seen[key] = (c0, c1, isinstance(p, (ast.Assign, ast.AnnAssign)))
            continue
        d0, d1, is_assign = seen[key]
        others = [(v[0], k[1]) for k, v in seen.items() if k[0] == kind and k != key and v[0] < c0]
        if not others:
            continue
        foil_val = max(others)[1]
        foil = str(foil_val) if kind == "int" else foil_val
        out.append(item(unit, "R3", kind, c0, content, foil, d0, d1, foil_kind="other_earlier_literal", def_is_assignment=is_assign,
                        n_prior_occurrences=sum(1 for _ in [0])))
    return out


EXTRACTORS = {"S1": extract_s1, "S2": extract_s2, "S3": extract_s3, "R1": extract_r1, "R2": extract_r2, "R3": extract_r3}


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csn-sample", type=int, default=4000)
    ap.add_argument("--cap-per-unit", type=int, default=2)
    ap.add_argument("--max-per-category", type=int, default=1200)
    ap.add_argument("--min-prefix-chars", type=int, default=20)
    ap.add_argument("--max-prefix-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csn-extra-sample", type=int, default=0, help="additional CSN units (validation split) used only for --extra-categories")
    ap.add_argument("--extra-categories", default="R3")
    ap.add_argument("--extra-cap-per-unit", type=int, default=3)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    from moetrace.models import MODELS
    from moetrace.ext4_data import resolve_item, CATEGORIES
    units = load_units(args.csn_sample, args.seed, args.csn_extra_sample)
    extra_cats = set(args.extra_categories.split(",")) if args.csn_extra_sample else set()
    rng = random.Random(args.seed)
    raw_counts = collections.Counter()
    cands = {c: [] for c in CATEGORIES}
    for u in units:
        unit = Unit(u)
        for cat, fn in EXTRACTORS.items():
            if u.get("extra") and cat not in extra_cats:
                continue
            try:
                its = fn(unit)
            except Exception as e:  # noqa
                its = []
            its = [it for it in its if len(it["prefix_text"]) >= args.min_prefix_chars]
            raw_counts[cat] += len(its)
            rng.shuffle(its)
            cands[cat].extend(its[: (args.extra_cap_per_unit if u.get("extra") else args.cap_per_unit)])
    log("raw candidates:", dict(raw_counts), "after per-unit cap:", {c: len(v) for c, v in cands.items()})
    toks = {k: AutoTokenizer.from_pretrained(snapshot_dir(MODELS[k]["repo"])) for k in ("qwen3", "mixtral")}
    items = []
    stats = {c: collections.Counter() for c in CATEGORIES}
    for cat in CATEGORIES:
        pool = cands[cat]
        rng.shuffle(pool)
        kept = 0
        for it in pool:
            if kept >= args.max_per_category:
                break
            ok = {}
            for k, tk in toks.items():
                c, why = resolve_item(it, tk, special_tokens=(k != "mixtral"), case_id=0, max_tokens=args.max_prefix_tokens)
                ok[k] = why
                if c is not None:
                    it[f"{k}_backoff"] = c.backoff
                    it[f"{k}_n_tokens"] = len(c.ids)
                    it[f"{k}_n_subject_tokens"] = len(c.subject_pos)
                    it[f"{k}_true_tok"] = c.true_tok
                    it[f"{k}_foil_tok"] = c.foil_tok
                    it[f"{k}_subject_tok"] = c.subject_tok
                stats[cat][f"{k}:{why}"] += 1
            it["ok_qwen3"] = ok["qwen3"] == "ok"
            it["ok_mixtral"] = ok["mixtral"] == "ok"
            if not (it["ok_qwen3"] or it["ok_mixtral"]):
                continue
            kept += 1
            items.append(it)
    for i, it in enumerate(items):
        it["case_id"] = i
        it["item_id"] = f"{it['category']}-{i:05d}"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    df = pd.DataFrame(items)
    # ---- stats
    lines = [f"# CodeFact build statistics ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n",
             f"Command: `python scripts/ext4_build_codefact.py --csn-sample {args.csn_sample} --cap-per-unit {args.cap_per_unit} "
             f"--max-per-category {args.max_per_category} --max-prefix-tokens {args.max_prefix_tokens} --csn-extra-sample {args.csn_extra_sample} --extra-categories {args.extra_categories}`\n",
             "## Sources and licences\n",
             "| Source | Units used | Licence | Notes |", "|---|---|---|---|"]
    uc = collections.Counter(u["source"] for u in units if not u.get("extra"))
    n_extra = sum(1 for u in units if u.get("extra"))
    lines += [f"| HumanEval (openai/openai_humaneval) | {uc['humaneval']} | MIT | prompt + canonical_solution, docstrings removed |",
              f"| MBPP full (google-research-datasets/mbpp; train/test/validation/prompt) | {uc['mbpp']} | CC-BY-4.0 | `code` field; CRLF normalised |",
              f"| CodeSearchNet Python, test split (code-search-net/code_search_net) | {uc['csn']} | per repository; the corpus was restricted at collection time to "
              f"repositories whose licence permits redistribution (Husain et al. 2019); per-function repo + URL in items.jsonl and csn_sample_repos.csv | "
              f"seed-{args.seed} sample of {args.csn_sample} functions <= 1500 chars, ASCII, docstrings removed |",
              f"| CodeSearchNet Python, validation split (extra units for {args.extra_categories} only) | {n_extra} | as above | seed-{args.seed + 7} sample of {args.csn_extra_sample} functions, cap {args.extra_cap_per_unit} per unit |",
              "\nThe Stack was requested as the volume source but is gated on the Hub (needs an authenticated account that accepted the terms); "
              "no token is available on this machine, so CodeSearchNet (the documented fallback) was used.\n",
              "## Yields per category\n",
              "| Category | Raw candidates | After per-unit cap | Written (single-token in >= 1 tokenizer) | Qwen3 ok | Mixtral ok | both ok | Qwen3 rejects | Mixtral rejects |",
              "|---|---|---|---|---|---|---|---|---|"]
    for cat in CATEGORIES:
        sub = df[df.category == cat] if len(df) else df
        q = int(sub.ok_qwen3.sum()) if len(sub) else 0
        m = int(sub.ok_mixtral.sum()) if len(sub) else 0
        b = int((sub.ok_qwen3 & sub.ok_mixtral).sum()) if len(sub) else 0
        rq = {k.split(":")[1]: v for k, v in stats[cat].items() if k.startswith("qwen3:") and not k.endswith(":ok")}
        rm = {k.split(":")[1]: v for k, v in stats[cat].items() if k.startswith("mixtral:") and not k.endswith(":ok")}
        lines.append(f"| {cat} {__import__('moetrace.ext4_data', fromlist=['x']).CATEGORY_LABEL[cat]} | {raw_counts[cat]} | {len(cands[cat])} | {len(sub)} | {q} | {m} | {b} | {rq} | {rm} |")
    lines += ["\nRejects: `multi_token` = true or foil is not a single token as a continuation of the prefix (after up to 4 characters of "
              "boundary back-off), `copy` = the true token occurs in the last 3 prefix tokens, `subject_is_final` = the subject token is the last "
              "prefix token, `too_long` = prefix > max tokens, `same_token` = true and foil map to the same token.\n",
              "## Sub-categories (written items)\n", "| Category | Sub-category | n | Qwen3 ok | Mixtral ok |", "|---|---|---|---|---|"]
    if len(df):
        for (cat, sub), g in df.groupby(["category", "subcategory"]):
            lines.append(f"| {cat} | {sub} | {len(g)} | {int(g.ok_qwen3.sum())} | {int(g.ok_mixtral.sum())} |")
        lines += ["\n## Prefix length (Qwen3 tokens, items ok for Qwen3)\n", "| Category | n | min | 25% | median | 75% | max | mean subject tokens |", "|---|---|---|---|---|---|---|---|"]
        for cat in CATEGORIES:
            g = df[(df.category == cat) & df.ok_qwen3]
            if len(g):
                d = g.qwen3_n_tokens.describe()
                lines.append(f"| {cat} | {len(g)} | {int(d['min'])} | {int(d['25%'])} | {int(d['50%'])} | {int(d['75%'])} | {int(d['max'])} | {g.qwen3_n_subject_tokens.mean():.2f} |")
        lines += ["\n## Boundary back-off (Qwen3 / Mixtral, items ok): how many characters the prefix/answer boundary moved back\n",
                  "| Category | Qwen3 backoff 0 / 1 / 2+ | Mixtral backoff 0 / 1 / 2+ |", "|---|---|---|"]
        for cat in CATEGORIES:
            g = df[df.category == cat]
            def bo(col, g):
                v = g[col].dropna().astype(int)
                return f"{int((v == 0).sum())} / {int((v == 1).sum())} / {int((v >= 2).sum())}"
            lines.append(f"| {cat} | {bo('qwen3_backoff', g[g.ok_qwen3])} | {bo('mixtral_backoff', g[g.ok_mixtral])} |")
        lines += ["\n## Source mix of written items\n", "| Category | humaneval | mbpp | csn |", "|---|---|---|---|"]
        for cat in CATEGORIES:
            g = df[df.category == cat].source.value_counts()
            lines.append(f"| {cat} | {int(g.get('humaneval', 0))} | {int(g.get('mbpp', 0))} | {int(g.get('csn', 0))} |")
    with open(os.path.join(OUT, "build_stats.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # ---- repos
    if len(df):
        rep = df[df.source == "csn"].groupby("repo").agg(n_items=("item_id", "size"), example_url=("url", "first")).reset_index()
        rep.to_csv(os.path.join(OUT, "csn_sample_repos.csv"), index=False)
    # ---- samples for eyeballing (20 random per category, Qwen3-valid)
    srng = random.Random(1)
    sl = [f"# CodeFact samples: 20 random Qwen3-valid items per category ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n",
          "For each item: the last lines of the prefix (`⟂` marks the split point), the true and foil tokens as the Qwen3 tokenizer sees them, "
          "and the subject token(s) that receive the noise. Check: is the subject the determinant of the answer?\n"]
    for cat in CATEGORIES:
        g = df[(df.category == cat) & df.ok_qwen3]
        pick = srng.sample(list(g.index), min(20, len(g)))
        sl.append(f"\n## {cat} ({__import__('moetrace.ext4_data', fromlist=['x']).CATEGORY_LABEL[cat]})\n")
        for j, idx in enumerate(pick):
            it = df.loc[idx]
            tail = it.prefix_text[-220:]
            if "\n" in tail and len(it.prefix_text) > 220:
                tail = tail[tail.index("\n") + 1:]
            sl.append(f"**{j + 1}. {it.item_id}** ({it.subcategory}, {it.source}) true `{it.qwen3_true_tok}` foil `{it.qwen3_foil_tok}` "
                      f"subject `{it.qwen3_subject_tok}` (\"{it.subject_text}\" at char {it.subject_start}) backoff {int(it.qwen3_backoff)}\n")
            sl.append("```python\n" + tail + "⟂\n```\n")
    with open(os.path.join(OUT, "samples.md"), "w") as f:
        f.write("\n".join(sl))
    log(f"wrote {len(items)} items -> {OUT}/items.jsonl; per category: {df.category.value_counts().to_dict() if len(df) else {}}")
    log("qwen3 ok per category:", df[df.ok_qwen3].category.value_counts().to_dict() if len(df) else {})
    log("mixtral ok per category:", df[df.ok_mixtral].category.value_counts().to_dict() if len(df) else {})


if __name__ == "__main__":
    main()
