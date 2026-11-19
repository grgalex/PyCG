#
# Copyright (c) 2020 Vitalis Salis.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""Native code reached from Python: compiled modules and dlopen'ed libraries.

The registry is filled by the preprocessor (extension-module imports) and
the dynamic-load pass (ctypes/cffi style loads), and read by the call-graph
pass, which turns the records into edges, and by the FASTEN formatter.
"""
import importlib.machinery
import os
import re

# Suffixes that make a file an importable extension module, longest first
# so that `_ext.cpython-311-x86_64-linux-gnu.so` loses the whole tag.
EXTENSION_SUFFIXES = sorted(
    set(importlib.machinery.EXTENSION_SUFFIXES) | {".so", ".pyd", ".abi3.so"},
    key=len,
    reverse=True,
)

# An extension built for another interpreter (`cpython-311` seen from 3.10,
# `cp313-win_amd64.pyd`) is still an extension module of the package.
EXTENSION_TAG_RE = re.compile(
    r"\.(?:cpython-\d+[A-Za-z0-9_-]*|cp\d+[A-Za-z0-9_-]*|abi3|pypy[A-Za-z0-9_-]*)?\.?(?:so|pyd)$"
)

# Suffixes a shared library may carry; `.so.1.2` is handled by prefix match.
LIBRARY_SUFFIXES = (".so", ".pyd", ".dylib", ".dll")

# Functions that load a shared library, with the index of the positional
# argument naming it. Names are the dotted names PyCG resolves callees to.
PRIMITIVE_LOADERS = {
    "ctypes.CDLL": 0,
    "ctypes.PyDLL": 0,
    "ctypes.WinDLL": 0,
    "ctypes.OleDLL": 0,
    "ctypes.cdll.LoadLibrary": 0,
    "ctypes.pydll.LoadLibrary": 0,
    "ctypes.windll.LoadLibrary": 0,
    "ctypes.oledll.LoadLibrary": 0,
    "cffi.FFI.dlopen": 0,
    "cffi.api.FFI.dlopen": 0,
    "_cffi_backend.FFI.dlopen": 0,
    "numpy.ctypeslib.load_library": 0,
}

# Keyword names under which loaders accept the library.
LIBRARY_KEYWORDS = ("name", "libname", "path", "library", "filename")

FIND_LIBRARY = {"ctypes.util.find_library"}

PATH_JOINERS = {"os.path.join", "posixpath.join", "ntpath.join"}
PATH_PASSTHROUGH = {
    "os.path.abspath",
    "os.path.realpath",
    "os.fspath",
    "str",
    "pathlib.Path",
}

DYNAMIC_IMPORTERS = {"importlib.import_module", "<builtin>.__import__"}


def strip_extension_suffix(filename):
    """`_ext.cpython-311-x86_64-linux-gnu.so` -> `_ext`; None if not one."""
    # The tagged form first: the plain `.so` suffix would also match it and
    # leave the ABI tag in the stem.
    m = EXTENSION_TAG_RE.search(filename)
    if m and m.start() > 0:
        return filename[: m.start()]
    for suffix in EXTENSION_SUFFIXES:
        if filename.endswith(suffix) and len(filename) > len(suffix):
            return filename[: -len(suffix)]
    return None


_SUFFIX_CACHE = {}


def extension_suffixes_in(root):
    """Every extension suffix used by a file under `root`, plus the
    interpreter's own, so the import hook finds modules built for another
    Python. Cached per root: the hook is installed once per entry point."""
    root = os.path.abspath(root) if root else root
    if root in _SUFFIX_CACHE:
        return _SUFFIX_CACHE[root]
    suffixes = set(importlib.machinery.all_suffixes())
    if root and os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                stem = strip_extension_suffix(fn)
                if stem is not None and stem:
                    suffixes.add(fn[len(stem):])
    _SUFFIX_CACHE[root] = sorted(suffixes, key=len, reverse=True)
    return _SUFFIX_CACHE[root]


def is_extension_file(path):
    return bool(path) and strip_extension_suffix(os.path.basename(path)) is not None


class LoaderSummary:
    """What a package-internal wrapper around a loader does.

    `param_index` is the parameter the library name comes from (None when the
    wrapper names a fixed library), `fixed` the library when it is fixed,
    `primitives` the loaders the wrapper ends in, `via` the wrappers between
    this one and the primitive, `resolution` how `param_index` was found:
    "parameter" when the name flows from the parameter, "argument" when the
    name could not be followed and the first string argument is assumed.
    """

    def __init__(self, param_index, fixed, primitives, via, resolution):
        self.param_index = param_index
        self.fixed = fixed
        self.primitives = set(primitives)
        self.via = list(via)
        self.resolution = resolution

    def key(self):
        return (
            self.param_index,
            tuple(sorted(self.fixed or [])),
            tuple(sorted(self.primitives)),
            tuple(self.via),
            self.resolution,
        )


class NativeRegistry:
    def __init__(self, package_root=None):
        self.package_root = os.path.abspath(package_root) if package_root else None
        # dotted module name -> path relative to the package root
        self.extension_modules = {}
        # (module name, line) of an import statement -> [dotted modules it
        # imports], compiled or not
        self.import_sites = {}
        # library name as written -> record
        self.libraries = {}
        # (module, line, col) -> load site record
        self.load_sites = {}
        # definition namespace -> {library names}
        self.handles = {}
        # (module, line, col) of a loader call -> {library names}
        self.call_handles = {}
        # function namespace -> LoaderSummary
        self.loaders = {}
        # definition namespace -> [LoadInfo] with no library yet
        self.pending = {}
        self._file_index = None

    # ---- extension modules -------------------------------------------------

    def relpath(self, path):
        if not path:
            return ""
        path = os.path.abspath(path)
        if self.package_root and path.startswith(self.package_root + os.sep):
            return os.path.relpath(path, self.package_root)
        return path

    def add_extension_module(self, dotted, path):
        self.extension_modules.setdefault(dotted, self.relpath(path))

    def add_import_site(self, modname, lineno, dotted):
        sites = self.import_sites.setdefault((modname, lineno), [])
        if dotted not in sites:
            sites.append(dotted)

    def extension_module_of(self, name):
        """The registered extension module `name` belongs to, or None.

        `pkg._ext.new.convert` -> `pkg._ext`; the longest match wins.
        """
        best = None
        for dotted in self.extension_modules:
            if name == dotted or name.startswith(dotted + "."):
                if best is None or len(dotted) > len(best):
                    best = dotted
        return best

    def find_extension_module(self, dotted):
        """Resolve a dotted name to an extension file under the root."""
        if dotted in self.extension_modules:
            return self.extension_modules[dotted]
        if not self.package_root:
            return None
        rel = dotted.replace(".", os.sep)
        for suffix in EXTENSION_SUFFIXES:
            candidate = os.path.join(self.package_root, rel + suffix)
            if os.path.isfile(candidate):
                self.extension_modules[dotted] = rel + suffix
                return rel + suffix
        # any ABI tag, not only this interpreter's
        parent = os.path.join(self.package_root, os.path.dirname(rel))
        base = os.path.basename(rel)
        if os.path.isdir(parent):
            for fn in sorted(os.listdir(parent)):
                if strip_extension_suffix(fn) == base:
                    self.extension_modules[dotted] = os.path.join(os.path.dirname(rel), fn)
                    return self.extension_modules[dotted]
        return None

    # ---- dynamic loads ----------------------------------------------------

    def _index_files(self):
        if self._file_index is not None:
            return self._file_index
        index = {}
        if self.package_root:
            for dirpath, dirnames, filenames in os.walk(self.package_root):
                dirnames.sort()
                for fn in sorted(filenames):
                    if ".so" in fn or fn.endswith(LIBRARY_SUFFIXES[1:]):
                        rel = os.path.relpath(os.path.join(dirpath, fn), self.package_root)
                        index.setdefault(fn, []).append(rel)
        self._file_index = index
        return index

    def resolve_library_file(self, library, near=None):
        """The file under the package root a library name denotes, or None.

        Accepts a path (absolute or relative), a dotted module name
        (`Crypto.Cipher._raw_aes`), a bare name (`foo`, `libfoo`), or a
        file name (`libfoo.so.1`). `near` is the directory of the loading
        module, preferred when several files match.
        """
        if not library or not self.package_root:
            return None
        root = self.package_root
        if os.path.isabs(library):
            if os.path.isfile(library) and library.startswith(root + os.sep):
                return os.path.relpath(library, root)
            library = os.path.basename(library)
        if os.sep in library:
            for base in ([near] if near else []) + [root]:
                candidate = os.path.normpath(os.path.join(base, library))
                if os.path.isfile(candidate) and candidate.startswith(root + os.sep):
                    return os.path.relpath(candidate, root)
            library = os.path.basename(library)
        if "." in library and not any(library.endswith(s) for s in LIBRARY_SUFFIXES) and ".so." not in library:
            found = self.find_extension_module(library)
            if found:
                return found
        index = self._index_files()
        names = [library]
        if not library.startswith("lib"):
            names.append("lib" + library)
        matches = []
        for fn, rels in index.items():
            stem = strip_extension_suffix(fn)
            for name in names:
                if fn == name or stem == name or fn.startswith(name + ".so"):
                    matches.extend(rels)
                    break
        if not matches:
            return None
        if near:
            near_rel = os.path.relpath(near, root)
            for rel in matches:
                if os.path.dirname(rel) == near_rel:
                    return rel
        return sorted(matches, key=lambda r: (r.count(os.sep), r))[0]

    def library_record(self, library):
        rec = self.libraries.get(library)
        if rec is None:
            rec = {
                "library": library,
                "file": None,
                "loaders": set(),
                "sites": [],
                "symbols": {},
            }
            self.libraries[library] = rec
        return rec

    def add_load_site(self, modname, lineno, col, namespace, loader, via, libraries, files, expr, resolution):
        """Record a load call. `libraries` may be empty (unresolved)."""
        key = (modname, lineno, col)
        site = self.load_sites.get(key)
        if site is None:
            site = {
                "namespace": namespace,
                "module": modname,
                "line": lineno,
                "loaders": set(),
                "via": list(via),
                "libraries": {},
                "expr": expr,
                "resolution": resolution,
            }
            self.load_sites[key] = site
        site["loaders"].update(loader if isinstance(loader, (set, list, tuple)) else [loader])
        if not site["via"] and via:
            site["via"] = list(via)
        for library, path in zip(libraries, files):
            site["libraries"].setdefault(library, path)
            rec = self.library_record(library)
            if path and not rec["file"]:
                rec["file"] = path
            rec["loaders"].update(site["loaders"])
            where = {"namespace": namespace, "line": lineno}
            if where not in rec["sites"]:
                rec["sites"].append(where)
        return site

    def bind_handle(self, def_ns, libraries):
        if not libraries:
            return
        self.handles.setdefault(def_ns, set()).update(libraries)

    def bind_call_handle(self, modname, lineno, col, libraries):
        if not libraries:
            return
        self.call_handles.setdefault((modname, lineno, col), set()).update(libraries)

    def libraries_of(self, names):
        libs = set()
        for name in names:
            libs.update(self.handles.get(name, ()))
        return libs

    def libraries_reachable(self, def_manager, names):
        """Handles bound to `names` or to anything they point at.

        The transitive closure keeps only the leaves of a name chain
        (`l -> lib -> x -> ctypes.CDLL` closes to `ctypes.CDLL`), and a
        handle is bound to `x`, so follow the name pointers themselves.
        """
        libs = set()
        seen = set()
        todo = list(names)
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            libs.update(self.handles.get(name, ()))
            defi = def_manager.get(name)
            if not defi:
                continue
            todo.extend(defi.get_name_pointer().get())
        return libs

    def bind_pending(self, def_ns, info):
        """A handle whose library is not known yet (a wrapper's parameter)."""
        self.pending.setdefault(def_ns, []).append(info)

    def pending_reachable(self, def_manager, names):
        out = []
        seen = set()
        todo = list(names)
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            out.extend(self.pending.get(name, ()))
            defi = def_manager.get(name)
            if not defi:
                continue
            todo.extend(defi.get_name_pointer().get())
        return out

    def add_symbol(self, library, symbol, caller):
        rec = self.library_record(library)
        rec["symbols"].setdefault(symbol, set()).add(caller)

    def set_loader(self, fn_ns, summary):
        old = self.loaders.get(fn_ns)
        if old is not None and old.key() == summary.key():
            return False
        self.loaders[fn_ns] = summary
        return True

    def state_key(self):
        return (
            len(self.loaders),
            tuple(sorted((k, s.key()) for k, s in self.loaders.items())),
            len(self.handles),
            sum(len(v) for v in self.handles.values()),
            len(self.load_sites),
            sum(len(s["libraries"]) for s in self.load_sites.values()),
        )

    # ---- output -----------------------------------------------------------

    def output_dynamic_loads(self):
        out = []
        for (modname, lineno, col), site in sorted(self.load_sites.items()):
            libs = sorted(site["libraries"].items())
            entry = {
                "namespace": site["namespace"],
                "line": lineno,
                "loader": sorted(site["loaders"])[0] if len(site["loaders"]) == 1 else sorted(site["loaders"]),
                "via": site["via"],
                "library": libs[0][0] if len(libs) == 1 else ([l for l, _ in libs] if libs else None),
                "file": libs[0][1] if len(libs) == 1 else ([f for _, f in libs] if libs else None),
                "expr": site["expr"],
                "resolution": site["resolution"],
            }
            out.append(entry)
        return out

    def library_metadata(self, library):
        rec = self.libraries.get(library)
        if rec is None:
            return {"kind": "dynamic-library", "library": library}
        return {
            "kind": "dynamic-library",
            "library": library,
            "loaders": sorted(rec["loaders"]),
            "sites": list(rec["sites"]),
        }
