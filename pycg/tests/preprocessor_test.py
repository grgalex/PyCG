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
import os
import shutil
import tempfile

from base import TestBase

from pycg import utils
from pycg.machinery.classes import ClassManager
from pycg.machinery.definitions import DefinitionManager
from pycg.machinery.imports import ImportManager
from pycg.machinery.modules import ModuleManager
from pycg.machinery.scopes import ScopeManager
from pycg.processing.preprocessor import PreProcessor
from pycg.pycg import CallGraphGenerator


class PreProcessorSourceTest(TestBase):
    """Run the preprocessor over a real module and inspect what it found.

    `_get_fun_defaults` used to index `node.args.args` with an offset derived
    from `len(node.args.args) - len(node.args.defaults)`. `ast.arguments`
    stores PEP 570 positional-only parameters separately, in `posonlyargs`,
    while `defaults` covers `posonlyargs + args` together, so any function with
    a `/` and a default raised `IndexError` and took its whole module with it.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name, contents):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(contents)
        return path

    def _preprocess(self, contents, modname="mod"):
        path = self._write(modname + ".py", contents)

        import_manager = ImportManager()
        import_manager.set_pkg(self.tmpdir)
        scope_manager = ScopeManager()
        def_manager = DefinitionManager()
        class_manager = ClassManager()
        module_manager = ModuleManager()

        import_manager.create_node(modname)
        import_manager.set_filepath(modname, path)
        import_manager.set_current_mod(modname, path)

        processor = PreProcessor(
            path,
            modname,
            import_manager,
            scope_manager,
            def_manager,
            class_manager,
            module_manager,
            modules_analyzed=set(),
        )
        processor.analyze()
        return processor, module_manager, def_manager

    def _methods(self, module_manager, modname="mod"):
        mod = module_manager.get(modname)
        self.assertIsNotNone(mod)
        return set(mod.get_methods().keys())

    def test_positional_only_args_with_defaults(self):
        # Before the fix this raised IndexError and the module kept nothing.
        contents = "\n".join(
            [
                "def posonly(a, b=1, /):",
                "    return a",
                "",
                "def other():",
                "    return posonly(1)",
                "",
            ]
        )
        _, module_manager, _ = self._preprocess(contents)

        self.assertEqual(
            self._methods(module_manager), set(["mod", "mod.posonly", "mod.other"])
        )

    def test_mixed_posonly_regular_and_kwonly_defaults(self):
        contents = "\n".join(
            [
                "def f(a, b=1, /, c=2, *, d=3):",
                "    return a",
                "",
            ]
        )
        processor, module_manager, def_manager = self._preprocess(contents)

        self.assertEqual(self._methods(module_manager), set(["mod", "mod.f"]))

        # every parameter, positional-only ones included, gets a definition
        for arg in ("a", "b", "c", "d"):
            self.assertIsNotNone(
                def_manager.get("mod.f." + arg),
                "no definition for parameter %s" % arg,
            )

        # positional indices count posonlyargs first: a=0, b=1, c=2, and the
        # keyword-only `d` gets no position at all
        name_pointer = def_manager.get("mod.f").get_name_pointer()
        self.assertEqual(name_pointer.get_pos_names(), {0: "a", 1: "b", 2: "c"})

    def test_get_fun_defaults_maps_defaults_to_the_right_names(self):
        contents = "\n".join(
            [
                "def one(): pass",
                "def two(): pass",
                "def three(): pass",
                "def f(a, b=one, /, c=two, *, d=three): pass",
                "",
            ]
        )
        processor, _, def_manager = self._preprocess(contents)

        # the names `defaults` is keyed by: `a` has none, and the offset must
        # be counted over posonlyargs + args, not args alone
        node = ast.parse(contents).body[-1]
        self.assertEqual(
            set(processor._get_fun_defaults(node).keys()), set(["b", "c", "d"])
        )

        # and each parameter really points at the function used as its default
        for name, expected in (("b", "one"), ("c", "two"), ("d", "three")):
            arg_def = def_manager.get("mod.f." + name)
            self.assertIsNotNone(arg_def, "no definition for parameter %s" % name)
            self.assertIn("mod." + expected, arg_def.get_name_pointer().get())

    def test_positional_only_self_is_bound_to_the_class(self):
        contents = "\n".join(
            [
                "class C:",
                "    def m(self, /, x=1):",
                "        return x",
                "",
            ]
        )
        _, module_manager, def_manager = self._preprocess(contents)

        self.assertEqual(
            self._methods(module_manager), set(["mod", "mod.C", "mod.C.m"])
        )
        self_def = def_manager.get("mod.C.m.self")
        self.assertIsNotNone(self_def)
        self.assertIn("mod.C", self_def.get_name_pointer().get())


class PosOnlyFullPassTest(TestBase):
    """The end-to-end shape of the bug: a `/` in a signature used to cost the
    whole module, with `do_pass`'s bare `except Exception` hiding the crash."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _entry_point(self, name, contents):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(contents)
        return path

    def test_posonly_module_survives_a_full_pass(self):
        # The end-to-end shape of the bug: one entry point whose only oddity
        # is a `/` in a signature used to yield a module with no methods and
        # no warning at all.
        path = self._entry_point(
            "posonly_mod.py",
            "\n".join(
                [
                    # every positional parameter is positional-only, so
                    # `node.args.args` is empty while `defaults` is not: the
                    # old offset indexed an empty list and raised IndexError
                    "def f(a, b=1, /):",
                    "    return a",
                    "",
                    "def g():",
                    "    return f(1)",
                    "",
                ]
            ),
        )
        cg = CallGraphGenerator(
            [path], self.tmpdir, 1, utils.constants.CALL_GRAPH_OP, False
        )
        cg.analyze()

        mod = cg.module_manager.get("posonly_mod")
        self.assertIsNotNone(mod)
        self.assertEqual(
            set(mod.get_methods().keys()),
            set(["posonly_mod", "posonly_mod.f", "posonly_mod.g"]),
        )
