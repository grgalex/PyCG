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
"""The dynamic-load pass: which names hold a handle to which library.

Runs after the pointer analysis and before the call-graph pass. It finds
calls to loaders (`ctypes.CDLL`, `ffi.dlopen`, ...), resolves the library
they name, records the load site, binds the handle to the names it is
assigned to, and summarises package-internal wrappers (`def load(name):
return CDLL(name)`) so that calls to them count as loads too. The pass is
run to a fixpoint because a wrapper may call another wrapper.
"""
import ast
import os

from pycg import utils
from pycg.machinery import native
from pycg.machinery.definitions import Definition
from pycg.processing.base import ProcessingBase


# candidate library names one expression may produce; beyond this the
# argument is a computed name, not a choice between a few
MAX_CANDIDATES = 32


class LoadInfo:
    def __init__(self, primitives, via, libraries, files, resolution, param_index=None,
                 slots=None):
        self.primitives = set(primitives)
        self.via = list(via)
        self.libraries = list(libraries)
        self.files = list(files)
        self.resolution = resolution
        self.param_index = param_index
        # positions of the handle when the call returns a tuple
        self.slots = set(slots) if slots else None


class NativeLoadProcessor(ProcessingBase):
    def __init__(
        self,
        filename,
        modname,
        import_manager,
        scope_manager,
        def_manager,
        class_manager,
        module_manager,
        native_registry,
        modules_analyzed=None,
    ):
        super().__init__(filename, modname, modules_analyzed)
        self.import_manager = import_manager
        self.scope_manager = scope_manager
        self.def_manager = def_manager
        self.class_manager = class_manager
        self.module_manager = module_manager
        self.native = native_registry
        self.closured = self.def_manager.transitive_closure()
        self.fn_stack = []

    # ---- traversal ---------------------------------------------------------

    def visit_FunctionDef(self, node):
        self.fn_stack.append(node)
        super().visit_FunctionDef(node)
        self.fn_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        counter = self.scope_manager.get_scope(self.current_ns).inc_lambda_counter()
        super().visit_Lambda(node, utils.get_lambda_name(counter))

    def visit_Assign(self, node):
        self.generic_visit(node)
        self._bind_targets(node.value, node.targets)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.value is not None:
            self._bind_targets(node.value, [node.target])

    def visit_Return(self, node):
        self.generic_visit(node)
        if node.value is None:
            return
        return_ns = utils.join_ns(self.current_ns, utils.constants.RETURN_NAME)
        if isinstance(node.value, ast.Tuple):
            # `return lib, True`: the handle sits in a slot of the tuple
            for i, elt in enumerate(node.value.elts):
                info = self._load_info_of_value(elt)
                if info is None:
                    continue
                self.native.bind_handle(return_ns, info.libraries)
                self._summarise_wrapper(info, slots={i})
            return
        info = self._load_info_of_value(node.value)
        if info is None:
            return
        self.native.bind_handle(return_ns, info.libraries)
        self._summarise_wrapper(info)

    def visit_Call(self, node):
        self.generic_visit(node)
        self._load_info(node)

    # ---- loads -------------------------------------------------------------

    def _bind_targets(self, value, targets):
        info = self._load_info_of_value(value)
        if info is None:
            # not a handle; perhaps a library name (`_libname =
            # find_library('sndfile')`, `path = os.path.join(d, name)`)
            libs, res, _pidx = self._resolve_library(value)
            # only what the literal pointers cannot carry; binding every
            # string-valued name fed products back into products across
            # the fixpoint iterations until memory ran out (pycryptodome)
            if libs and res in ("find_library", "path-join", "pattern"):
                for target in targets:
                    if isinstance(target, ast.Tuple):
                        continue
                    for tns in self._get_target_ns(target):
                        if tns:
                            self.native.bind_libname(tns, libs)
            return
        for target in targets:
            if isinstance(target, ast.Tuple):
                # `lib, flag = load()` where the wrapper returns the handle
                # in a known slot
                for i in sorted(info.slots or ()):
                    if i < len(target.elts) and not isinstance(
                            target.elts[i], (ast.Tuple, ast.Starred)):
                        for tns in self._get_target_ns(target.elts[i]):
                            if tns and info.libraries:
                                self.native.bind_handle(tns, info.libraries)
                continue
            if isinstance(target, ast.Subscript):
                # `handles['x'] = CDLL(...)`: the handle is bound to the
                # container; a read of any item of it yields the handle
                target = target.value
            for tns in self._get_target_ns(target):
                if not tns:
                    continue
                if info.libraries:
                    self.native.bind_handle(tns, info.libraries)
                else:
                    self.native.bind_pending(tns, info)

    def _load_info_of_value(self, value):
        """LoadInfo for an expression that may evaluate to a handle."""
        if isinstance(value, ast.Call):
            return self._load_info(value)
        if isinstance(value, ast.IfExp):
            infos = [self._load_info_of_value(v) for v in (value.body, value.orelse)]
            infos = [i for i in infos if i is not None]
            if not infos:
                return None
            merged = LoadInfo(set(), infos[0].via, [], [], infos[0].resolution)
            for i in infos:
                merged.primitives |= i.primitives
                merged.libraries += i.libraries
                merged.files += i.files
            return merged
        if isinstance(value, ast.Subscript):
            value = value.value
        if isinstance(value, (ast.Name, ast.Attribute)):
            names = set()
            for d in self.decode_node(value):
                if isinstance(d, Definition):
                    names.add(d.get_ns())
            libs = self.native.libraries_reachable(self.def_manager, names)
            pending = self.native.pending_reachable(self.def_manager, names)
            if not libs and not pending:
                return None
            merged = LoadInfo(set(), [], sorted(libs), [None] * len(libs), "alias")
            for info in pending:
                merged.primitives |= info.primitives
                if not merged.via:
                    merged.via = list(info.via)
                if merged.resolution == "alias":
                    merged.resolution = info.resolution
                    merged.param_index = info.param_index
            return merged
        return None

    def _callee_names(self, node):
        names = set(self.retrieve_call_names(node) or ())
        if not names and isinstance(node.func, ast.Attribute):
            names.update(self._full_attr_names(node.func))
        if not names and isinstance(node.func, ast.Name):
            names.add(utils.join_ns(utils.constants.BUILTIN_NAME, node.func.id))
        return names

    def _full_attr_names(self, node):
        """`a.b.c` -> every `<closure of a>.b.c` (external attribute chains)."""
        attrs = []
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return set()
        defi = self.scope_manager.get_def(self.current_ns, node.id)
        if not defi:
            return set()
        names = set()
        for base in self.closured.get(defi.get_ns(), ()) | {defi.get_ns()}:
            names.add(".".join([base] + attrs))
        return names

    def _load_info(self, node):
        """Classify a call; record a site and a call handle when it loads."""
        key = (self.modname, node.lineno, node.col_offset)
        names = self._callee_names(node)
        primitives = set()
        wrappers = []
        for name in names:
            if name in native.PRIMITIVE_LOADERS:
                primitives.add(name)
            elif name in self.native.loaders:
                wrappers.append(name)
        if not primitives and not wrappers:
            return None

        libraries = []
        files = []
        resolution = "unresolved"
        param_index = None
        via = []
        all_primitives = set(primitives)

        if primitives:
            arg = self._library_arg(node, 0)
            libraries, resolution, param_index = self._resolve_library(arg)
        slots = set()
        for wrapper in wrappers:
            summary = self.native.loaders[wrapper]
            all_primitives |= summary.primitives
            via = [wrapper] + summary.via
            if summary.handle_slots:
                slots |= summary.handle_slots
            if summary.fixed:
                libraries = libraries + list(summary.fixed)
                resolution = "wrapper-constant"
            if summary.param_index is None:
                continue
            arg = self._library_arg(node, summary.param_index)
            libs, res, pidx = self._resolve_library(arg)
            if libs:
                libraries = libraries + libs
                resolution = "argument" if summary.resolution == "argument" else res
                param_index = pidx if pidx is not None else param_index
            elif res == "parameter":
                resolution = "parameter"
                param_index = pidx

        libraries = sorted(set(l for l in libraries if l))
        near = os.path.dirname(self.filename)
        resolved = []
        for lib in libraries:
            if "*" in lib:
                # a name with a hole stands for the files it matches
                resolved += self.native.resolve_library_pattern(lib, near=near)
            else:
                resolved.append(lib)
        libraries = sorted(set(resolved))
        files = [self.native.resolve_library_file(lib, near=near) for lib in libraries]
        try:
            expr = ast.unparse(node)
        except Exception:
            expr = ""
        self.native.add_load_site(
            self.modname,
            node.lineno,
            node.col_offset,
            self.current_method,
            all_primitives,
            via,
            libraries,
            files,
            expr,
            resolution,
        )
        self.native.bind_call_handle(self.modname, node.lineno, node.col_offset, libraries)
        info = LoadInfo(all_primitives, via, libraries, files, resolution, param_index,
                        slots=slots)
        if self.fn_stack and libraries and resolution not in (
                "unresolved", "alias", "parameter"):
            # a function that loads a known library is a loader whatever it
            # returns: calling it is where the library gets loaded from the
            # caller's side (`_ghash_portable = _get_ghash_portable()`)
            self._summarise_wrapper(info)
        return info

    def _library_arg(self, node, index):
        if index is not None and len(node.args) > index:
            arg = node.args[index]
            if isinstance(arg, ast.Starred):
                return None
            return arg
        for kw in node.keywords:
            if kw.arg in native.LIBRARY_KEYWORDS:
                return kw.value
        return None

    # ---- the library argument ----------------------------------------------

    def _resolve_library(self, node):
        """-> ([library names], resolution, parameter index or None)."""
        if node is None:
            return [], "unresolved", None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value], "literal", None
        if isinstance(node, ast.JoinedStr):
            return [], "unresolved", None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, lres, _ = self._resolve_library(node.left)
            right, rres, _ = self._resolve_library(node.right)
            if left and right:
                if len(left) * len(right) > MAX_CANDIDATES:
                    return [], "unresolved", None
                res = "pattern" if "pattern" in (lres, rres) else "literal"
                return [a + b for a in left for b in right], res, None
            # one side is computed (`'libsndfile_' + machine() + '.so'`):
            # a name with a hole, matched against the package's files. The
            # hole must follow a stem: `*.abi3.so` alone names every file.
            if left and lres != "parameter":
                return [a if a.endswith("*") else a + "*" for a in left], "pattern", None
            if right and rres != "parameter" and all(
                    b.startswith("*") and len(b.lstrip("*").split(".")[0]) >= 3 for b in right):
                return right, "pattern", None
            return [], "unresolved", None
        if isinstance(node, ast.Call):
            names = self._callee_names(node)
            if names & native.FIND_LIBRARY:
                libs, res, pidx = self._resolve_library(self._library_arg(node, 0))
                return libs, ("find_library" if libs else res), pidx
            if names & native.PATH_JOINERS:
                parts = [self._resolve_library(a) for a in node.args]
                # keep the trailing run of resolved components; each may
                # have several candidates (a platform choice)
                tail = []
                pattern = False
                for libs, res, _ in reversed(parts):
                    if libs and len(libs) <= 8:
                        tail.insert(0, libs)
                        pattern = pattern or res == "pattern"
                    else:
                        break
                if tail:
                    import itertools
                    joined = [os.path.join(*combo) for combo in itertools.product(*tail)]
                    return sorted(set(joined))[:MAX_CANDIDATES], ("pattern" if pattern else "path-join"), None
                return [], "unresolved", None
            if names & native.PATH_PASSTHROUGH and node.args:
                return self._resolve_library(node.args[0])
            return [], "unresolved", None
        if isinstance(node, ast.Name):
            pidx = self._parameter_index(node.id)
            if pidx is not None:
                return [], "parameter", pidx
        if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            # a literal with a format hole ('./%s.dll', '{}.so') is a
            # template the name was built from, not a library name
            lits = {l for l in self._string_literals_of(node)
                    if "%" not in l and "{" not in l}
            names = set()
            for d in self.decode_node(node):
                if isinstance(d, Definition):
                    names.add(d.get_ns())
            bound = self.native.libnames_reachable(self.def_manager, names)
            if lits or bound:
                res = "name-constant"
                if bound and any("*" in b for b in bound):
                    res = "pattern"
                # what the name was assigned, when that was seen, rather
                # than the string pieces of those assignments (the literal
                # pointer of `'lib' + f() + '.so'` holds 'lib' and '.so')
                return sorted(bound or lits)[:MAX_CANDIDATES], res, None
        return [], "unresolved", None

    def _parameter_index(self, name):
        if not self.fn_stack:
            return None
        fn = self.fn_stack[-1]
        posargs = list(getattr(fn.args, "posonlyargs", None) or []) + list(fn.args.args)
        for i, arg in enumerate(posargs):
            if arg.arg == name:
                # `self`/`cls` was dropped from the pointer's positions by
                # the preprocessor; keep the AST position, callers are
                # resolved against it by `_library_arg`.
                return i
        return None

    def _string_literals_of(self, node):
        """String literals the expression can hold, following name pointers
        through every definition on the way (the transitive closure keeps
        only the leaves of a chain, and a generator's yielded literal sits
        on its `<RETURN>` definition, which also points at other names)."""
        lits = set()
        todo = []
        for d in self.decode_node(node):
            if isinstance(d, str):
                lits.add(d)
            elif isinstance(d, Definition):
                todo.append(d.get_ns())
        seen = set()
        while todo:
            ns = todo.pop()
            if ns in seen:
                continue
            seen.add(ns)
            defi = self.def_manager.get(ns)
            if not defi:
                continue
            for lit in defi.get_lit_pointer().get():
                if isinstance(lit, str):
                    lits.add(lit)
            todo.extend(defi.get_name_pointer().get())
        return lits

    # ---- wrappers ----------------------------------------------------------

    def _summarise_wrapper(self, info, slots=None):
        """The current function returns a handle (in tuple `slots`, when
        given) or loads a known library: make it a loader."""
        if not self.fn_stack:
            return
        fn_ns = self.current_ns
        fixed = None
        param_index = None
        resolution = info.resolution
        if info.resolution == "parameter" and info.param_index is not None:
            param_index = info.param_index
            # a method's `self` is counted in the AST but not by callers
            if self._is_method(self.fn_stack[-1]):
                param_index -= 1
        elif info.libraries and info.resolution not in ("unresolved", "alias", "pattern"):
            fixed = list(info.libraries)
        else:
            fn = self.fn_stack[-1]
            posargs = list(getattr(fn.args, "posonlyargs", None) or []) + list(fn.args.args)
            if self._is_method(fn):
                posargs = posargs[1:]
            if posargs:
                param_index = 0
                resolution = "argument"
            elif info.libraries:
                fixed = list(info.libraries)
        if fixed is None and param_index is None:
            return
        primitives = set(info.primitives)
        for v in info.via:
            summary = self.native.loaders.get(v)
            if summary:
                primitives |= summary.primitives
        self.native.set_loader(
            fn_ns,
            native.LoaderSummary(param_index, fixed, primitives, info.via, resolution,
                                 handle_slots=slots),
        )

    def _is_method(self, fn):
        parent = self.def_manager.get(".".join(self.name_stack[:-1]))
        if not parent or parent.get_type() != utils.constants.CLS_DEF:
            return False
        for decorator in fn.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "staticmethod":
                return False
        return True

    # ---- driver ------------------------------------------------------------

    def analyze_submodules(self):
        super().analyze_submodules(
            NativeLoadProcessor,
            self.import_manager,
            self.scope_manager,
            self.def_manager,
            self.class_manager,
            self.module_manager,
            self.native,
            modules_analyzed=self.get_modules_analyzed(),
        )

    def analyze(self):
        self.visit(ast.parse(self.contents, self.filename))
        self.analyze_submodules()
