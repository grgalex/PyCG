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
"""Native code reached from Python: the package's compiled modules.

The registry is filled by the preprocessor (imports) and read by the
call-graph pass, which turns the records into edges, and by the FASTEN
formatter.
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


class NativeRegistry:
    def __init__(self, package_root=None):
        self.package_root = os.path.abspath(package_root) if package_root else None
        # dotted module name -> path relative to the package root
        self.extension_modules = {}
        # (module name, line) of an import statement -> [dotted modules it
        # imports], compiled or not
        self.import_sites = {}
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

