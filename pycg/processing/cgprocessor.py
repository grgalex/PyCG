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
import ast
import re
import os

from pycg import utils
from pycg.machinery import native
from pycg.machinery.definitions import Definition
from pycg.processing.base import ProcessingBase


class CallGraphProcessor(ProcessingBase):
    def __init__(
        self,
        filename,
        modname,
        import_manager,
        scope_manager,
        def_manager,
        class_manager,
        module_manager,
        call_graph=None,
        native_registry=None,
        modules_analyzed=None,
    ):
        super().__init__(filename, modname, modules_analyzed)
        # parent directory of file
        self.parent_dir = os.path.dirname(filename)

        self.import_manager = import_manager
        self.scope_manager = scope_manager
        self.def_manager = def_manager
        self.class_manager = class_manager
        self.module_manager = module_manager

        self.call_graph = call_graph
        self.native = native_registry

        self.closured = self.def_manager.transitive_closure()

    def visit_Module(self, node):
        self.call_graph.add_node(self.modname, self.modname)
        super().visit_Module(node)

    def visit_For(self, node):
        self.visit(node.iter)
        self.visit(node.target)
        # assign target.id to the return value of __next__ of node.iter.it
        # we need to have a visit for on the postprocessor also
        iter_decoded = self.decode_node(node.iter)
        for item in iter_decoded:
            if not isinstance(item, Definition):
                continue
            names = self.closured.get(item.get_ns(), [])
            for name in names:
                iter_ns = utils.join_ns(name, utils.constants.ITER_METHOD)
                next_ns = utils.join_ns(name, utils.constants.NEXT_METHOD)
                if self.def_manager.get(iter_ns):
                    self.call_graph.add_edge(self.current_method, iter_ns)
                if self.def_manager.get(next_ns):
                    self.call_graph.add_edge(self.current_method, next_ns)

        super().visit_For(node)

    def visit_Lambda(self, node):
        counter = self.scope_manager.get_scope(self.current_ns).inc_lambda_counter()
        lambda_name = utils.get_lambda_name(counter)
        lambda_fullns = utils.join_ns(self.current_ns, lambda_name)

        self.call_graph.add_node(lambda_fullns, self.modname)

        super().visit_Lambda(node, lambda_name)

    def visit_Raise(self, node):
        if not node.exc:
            return
        self.visit(node.exc)
        decoded = self.decode_node(node.exc)
        for d in decoded:
            if not isinstance(d, Definition):
                continue
            names = self.closured.get(d.get_ns(), [])
            for name in names:
                pointer_def = self.def_manager.get(name)
                if pointer_def.get_type() == utils.constants.CLS_DEF:
                    init_ns = self.find_cls_fun_ns(name, utils.constants.CLS_INIT)
                    for ns in init_ns:
                        self.call_graph.add_edge(self.current_method, ns)
                if pointer_def.get_type() == utils.constants.EXT_DEF:
                    self.call_graph.add_edge(self.current_method, name)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
            decoded = self.decode_node(decorator)
            for d in decoded:
                if not isinstance(d, Definition):
                    continue
                names = self.closured.get(d.get_ns(), [])
                for name in names:
                    self.call_graph.add_edge(self.current_method, name)

        self.call_graph.add_node(
            utils.join_ns(self.current_ns, node.name), self.modname
        )
        super().visit_FunctionDef(node)

    # ---- native code -------------------------------------------------------

    def visit_Import(self, node):
        self._import_edges(node)

    def visit_ImportFrom(self, node):
        self._import_edges(node)

    def _import_edges(self, node):
        """An import is a call of the imported module's body."""
        if self.native is None:
            return
        for dotted in self.native.import_sites.get((self.modname, node.lineno), ()):
            self._module_edge(dotted)

    def _module_edge(self, dotted):
        if dotted in self.native.extension_modules:
            self._compiled_module(dotted)
        elif not self.def_manager.get(dotted):
            return
        self.call_graph.add_node(dotted, dotted)
        self.call_graph.add_edge(self.current_method, dotted)

    def _compiled_module(self, dotted):
        """A compiled module of the package is an internal module whose
        file is the binary and whose namespaces carry no line numbers:
        the consumer resolves them against the file."""
        mod = self.module_manager.get(dotted)
        if not mod:
            mod = self.module_manager.create(dotted, self.native.extension_modules[dotted])
            mod.add_method(dotted)
        return mod

    def _ext_node(self, name, ext_modname, metadata=None):
        """Register `name` under external module `ext_modname`.

        Dynamically loaded libraries carry their load sites so a consumer
        can tell them from PyPI dependencies.
        """
        ext_mod = self.module_manager.get(ext_modname)
        if not ext_mod:
            ext_mod = self.module_manager.create(ext_modname, None, external=True)
            ext_mod.add_method(ext_modname)
        if self.native is not None and ext_modname.startswith("dlopen:"):
            library = ext_modname[len("dlopen:") :]
            rec = self.native.libraries.get(library)
            if rec and rec.get("file") and not ext_mod.get_filename():
                ext_mod.set_filename(rec["file"])
            ext_mod.set_metadata(self.native.library_metadata(library))
        ext_mod.add_method(name, metadata=metadata)

    def _callee_module(self, name):
        """The module a callee name belongs to: a compiled module of the
        package when the name is under one, else the first component."""
        if self.native is not None:
            dotted = self.native.extension_module_of(name)
            if dotted:
                return dotted, True
        return name.split(".")[0], False

    def visit_Attribute(self, node):
        # `f = handle.sym` is a symbol lookup (ctypes/cffi dlsym on attribute
        # access) even when the call comes later through `f`; record it where
        # the attribute is read. An UPPER_CASE read is a cffi cdef constant
        # (`lib.SFC_GET_LOG_INFO`), not a function.
        if not re.fullmatch(r"[A-Z0-9_]+", node.attr):
            self._dynamic_symbol_ref(node)
        self.generic_visit(node)

    def _dynamic_symbol_edges(self, node):
        """`handle.sym(...)` where `handle` holds a dlopen'ed library."""
        if not isinstance(node.func, ast.Attribute):
            return False
        return self._dynamic_symbol_ref(node.func)

    def _dynamic_symbol_ref(self, attr):
        """`handle.sym`, read or called, where `handle` holds a dlopen'ed
        library: a `dlopen:<library>.sym` node and an edge to it."""
        if self.native is None:
            return False
        node = attr
        base = attr.value
        if isinstance(base, ast.Subscript):
            # `handles['x'].sym`: the handle was bound to the container
            base = base.value
        libraries = set()
        if isinstance(base, ast.Call):
            libraries |= self.native.call_handles.get(
                (self.modname, base.lineno, base.col_offset), set()
            )
        names = set()
        for d in self.decode_node(base):
            if isinstance(d, Definition):
                names.add(d.get_ns())
        if isinstance(base, ast.Attribute):
            # `self._lib`: decode_node hands back what the attribute closes
            # to (the loader), not the attribute itself, and the handle is
            # bound to the attribute `pkg.mod.W._lib`.
            for parent in self._retrieve_parent_names(base):
                cls = self.class_manager.get(parent)
                for owner in (cls.get_mro() if cls else [parent]):
                    names.add(utils.join_ns(owner, base.attr))
        libraries |= self.native.libraries_reachable(self.def_manager, names)
        if not libraries:
            return False
        symbol = attr.attr
        for library in sorted(libraries):
            ext_modname = "dlopen:" + library
            name = utils.join_ns(ext_modname, symbol)
            if not self.def_manager.get(name):
                self.def_manager.create(name, utils.constants.EXT_DEF)
            self.native.add_symbol(library, symbol, self.current_method)
            self._ext_node(name, ext_modname, {"kind": "dynamic-symbol", "symbol": symbol})
            self.call_graph.add_node(name, ext_modname)
            self.call_graph.add_edge(self.current_method, name)
        return True

    def _dynamic_import_edges(self, node, names):
        """`importlib.import_module("pkg.mod")` is an import too."""
        if self.native is None or not names & native.DYNAMIC_IMPORTERS:
            return
        if not node.args or not isinstance(node.args[0], ast.Constant):
            return
        dotted = node.args[0].value
        if not isinstance(dotted, str):
            return
        if dotted.startswith("."):
            pkg = None
            for kw in node.keywords:
                if kw.arg == "package" and isinstance(kw.value, ast.Constant):
                    pkg = kw.value.value
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                pkg = node.args[1].value
            if not isinstance(pkg, str):
                return
            dotted = pkg + dotted if dotted.count(".") == 1 else pkg.rsplit(".", dotted.count(".") - 1)[0] + "." + dotted.lstrip(".")
        self.native.find_extension_module(dotted)
        self._module_edge(dotted)

    def visit_Call(self, node):
        def create_ext_edge(name, ext_modname):
            ext_modname, compiled = self._callee_module(name)
            if compiled:
                # an attribute of the package's own compiled module: an
                # internal namespace without lines, under that module
                self._compiled_module(ext_modname).add_method(name)
            else:
                self._ext_node(name, ext_modname)
            self.call_graph.add_node(name, ext_modname)
            self.call_graph.add_edge(self.current_method, name)

        # First visit the child function so that on the case of
        #       func()()()
        # we first visit the call to func and then the other calls
        for arg in node.args:
            self.visit(arg)

        for keyword in node.keywords:
            self.visit(keyword.value)

        self.visit(node.func)

        if self._dynamic_symbol_edges(node):
            return

        names = self.retrieve_call_names(node)
        if names:
            self._dynamic_import_edges(node, names)
        elif isinstance(node.func, ast.Name):
            self._dynamic_import_edges(
                node, {utils.join_ns(utils.constants.BUILTIN_NAME, node.func.id)}
            )
        if not names:
            if isinstance(node.func, ast.Attribute) and self.has_ext_parent(node.func):
                # TODO: This doesn't work for cases
                # where there is an assignment of an attribute
                # i.e. import os; lala = os.path; lala.dirname()
                for name in self.get_full_attr_names(node.func):
                    ext_modname = name.split(".")[0]
                    create_ext_edge(name, ext_modname)
            elif getattr(node.func, "id", None) and self.is_builtin(node.func.id):
                name = utils.join_ns(utils.constants.BUILTIN_NAME, node.func.id)
                create_ext_edge(name, utils.constants.BUILTIN_NAME)
            return

        self.last_called_names = names
        for pointer in names:
            pointer_def = self.def_manager.get(pointer)
            if not pointer_def or not isinstance(pointer_def, Definition):
                continue
            if pointer_def.is_callable():
                if pointer_def.get_type() == utils.constants.EXT_DEF:
                    ext_modname = pointer.split(".")[0]
                    create_ext_edge(pointer, ext_modname)
                    continue
                self.call_graph.add_edge(self.current_method, pointer)

            # TODO: This doesn't work
            # and leads to calls from the decorators
            # themselves to the function,
            # creating edges to the first decorator
            # for decorator in pointer_def.decorator_names:
            #   dec_names = self.closured.get(decorator, [])
            #   for dec_name in dec_names:
            #       if self.def_manager.get(dec_name).
            #               get_type() == utils.constants.FUN_DEF:
            #           self.call_graph.add_edge(self.current_ns, dec_name)

            if pointer_def.get_type() == utils.constants.CLS_DEF:
                init_ns = self.find_cls_fun_ns(pointer, utils.constants.CLS_INIT)

                for ns in init_ns:
                    self.call_graph.add_edge(self.current_method, ns)

    def analyze_submodules(self):
        super().analyze_submodules(
            CallGraphProcessor,
            self.import_manager,
            self.scope_manager,
            self.def_manager,
            self.class_manager,
            self.module_manager,
            call_graph=self.call_graph,
            native_registry=self.native,
            modules_analyzed=self.get_modules_analyzed(),
        )

    def analyze(self):
        self.visit(ast.parse(self.contents, self.filename))
        self.analyze_submodules()

    def get_all_reachable_functions(self):
        reachable = set()
        names = set()
        current_scope = self.scope_manager.get_scope(self.current_ns)
        while current_scope:
            for name, defi in current_scope.get_defs().items():
                if defi.is_function_def() and name not in names:
                    closured = self.closured.get(defi.get_ns())
                    for item in closured:
                        reachable.add(item)
                    names.add(name)
            current_scope = current_scope.parent

        return reachable

    def has_ext_parent(self, node):
        if not isinstance(node, ast.Attribute):
            return False

        while isinstance(node, ast.Attribute):
            parents = self._retrieve_parent_names(node)
            for parent in parents:
                for name in self.closured.get(parent, []):
                    defi = self.def_manager.get(name)
                    if defi and defi.is_ext_def():
                        return True
            node = node.value
        return False

    def get_full_attr_names(self, node):
        name = ""
        while isinstance(node, ast.Attribute):
            if not name:
                name = node.attr
            else:
                name = node.attr + "." + name
            node = node.value

        names = []
        if getattr(node, "id", None) is None:
            return names

        defi = self.scope_manager.get_def(self.current_ns, node.id)
        if defi and self.closured.get(defi.get_ns()):
            for id in self.closured.get(defi.get_ns()):
                names.append(id + "." + name)

        return names

    def is_builtin(self, name):
        return name in __builtins__
