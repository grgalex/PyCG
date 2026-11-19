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
"""Native code reached from Python: compiled-module imports."""
import glob
import os
import shutil
import tempfile

from base import TestBase
from pycg.formats import Fasten
from pycg.machinery import native
from pycg.pycg import CallGraphGenerator


class NativeTestBase(TestBase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        native._SUFFIX_CACHE.clear()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, rel, contents=""):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(contents)
        return path

    def generate(self):
        files = sorted(glob.glob(os.path.join(self.root, "**", "*.py"), recursive=True))
        cg = CallGraphGenerator(files, self.root, 1, "call-graph", False)
        cg.analyze()
        return cg

    def fasten(self, cg):
        return Fasten(cg, self.root, "pkg", "PyPI", "0", 0).generate()

    @staticmethod
    def edges(cg):
        return {(src, dst) for src, dsts in cg.cg.get().items() for dst in dsts}

    @staticmethod
    def external(doc, modname):
        mod = doc["modules"]["external"][modname]
        return {
            ns["namespace"].split("//")[-1].rstrip("()"): ns["metadata"]
            for ns in mod["namespaces"].values()
        }


class CompiledModuleTest(NativeTestBase):
    def test_imports_are_edges_and_compiled_modules_are_modules_with_a_file(self):
        self.write("pkg/__init__.py")
        self.write("pkg/_ext.cpython-311-x86_64-linux-gnu.so")
        self.write("pkg/sub/__init__.py")
        self.write("pkg/sub/_ext2.abi3.so")
        self.write("pkg/helper.py", "def h():\n    return 1\n")
        self.write(
            "pkg/mod.py",
            "\n".join(
                [
                    "import importlib",
                    "import pkg._ext",
                    "from . import _ext as core",
                    "from pkg._ext import f",
                    "from .sub import _ext2",
                    "from . import helper",
                    "dyn = importlib.import_module('pkg._ext')",
                    "",
                    "def lazy():",
                    "    import pkg._ext as inner",
                    "    return inner.lazy_call()",
                    "",
                    "def use():",
                    "    core.blend()",
                    "    pkg._ext.run()",
                    "    f()",
                    "    _ext2.helper()",
                    "    core.new(1).convert()",
                    "    helper.h()",
                    "",
                ]
            ),
        )
        cg = self.generate()
        edges = self.edges(cg)
        # imports: an edge to the module, compiled or not
        self.assertIn(("pkg.mod", "pkg._ext"), edges)
        self.assertIn(("pkg.mod.lazy", "pkg._ext"), edges)
        self.assertIn(("pkg.mod", "pkg.sub._ext2"), edges)
        self.assertIn(("pkg.mod", "pkg.helper"), edges)
        for callee in ("pkg._ext.blend", "pkg._ext.run", "pkg._ext.f", "pkg.sub._ext2.helper", "pkg._ext.new.convert"):
            self.assertIn(("pkg.mod.use", callee), edges)
        self.assertIn(("pkg.mod.lazy", "pkg._ext.lazy_call"), edges)
        self.assertIn(("pkg.mod.use", "pkg.helper.h"), edges)
        self.assertNotIn("_ext", cg.module_manager.get_external_modules())
        self.assertNotIn("pkg._ext", cg.module_manager.get_external_modules())

        doc = self.fasten(cg)
        internal = doc["modules"]["internal"]
        ext = internal["/pkg._ext/"]
        self.assertEqual(ext["sourceFile"], "pkg/_ext.cpython-311-x86_64-linux-gnu.so")
        names = {ns["namespace"]: ns["metadata"] for ns in ext["namespaces"].values()}
        self.assertIn("/pkg._ext/", names)
        self.assertIn("/pkg._ext/blend", names)
        self.assertIn("/pkg._ext/new.convert", names)
        self.assertEqual(names["/pkg._ext/blend"], {"first": None, "last": None})
        # the import edge is an internal call into the module namespace
        ids = {str(k): v["namespace"] for m in internal.values() for k, v in m["namespaces"].items()}
        calls = {(ids[a], ids[b]) for a, b, _ in doc["graph"]["internalCalls"]}
        self.assertIn(("/pkg.mod/", "/pkg._ext/"), calls)
        self.assertIn(("/pkg.mod/", "/pkg.helper/"), calls)
        self.assertIn(("/pkg.mod/use()", "/pkg._ext/blend"), calls)
        # nothing in the output decides what the file is
        text = __import__("json").dumps(doc)
        self.assertNotIn("PyInit", text)
        self.assertNotIn("extension-", text)

    def test_from_import_names_the_module_not_a_same_named_class(self):
        self.write("pkg/__init__.py")
        self.write("pkg/_imaging.so")
        self.write(
            "pkg/Image.py",
            "\n".join(
                [
                    "from . import _imaging as core",
                    "",
                    "class Image:",
                    "    pass",
                    "",
                    "def new():",
                    "    return core.fill()",
                    "",
                ]
            ),
        )
        self.write(
            "pkg/Plugin.py",
            "\n".join(
                [
                    "from . import Image",
                    "",
                    "def blank():",
                    "    return Image.core.fill()",
                    "",
                    "def img():",
                    "    return Image.new()",
                    "",
                ]
            ),
        )
        edges = self.edges(self.generate())
        self.assertIn(("pkg.Plugin.blank", "pkg._imaging.fill"), edges)
        self.assertIn(("pkg.Plugin.img", "pkg.Image.new"), edges)
        self.assertIn(("pkg.Plugin", "pkg.Image"), edges)

    def test_import_inside_the_packages_init_is_not_an_alias_of_itself(self):
        self.write("pkg/__init__.py")
        self.write("pkg/core/_speedups.cpython-311-x86_64-linux-gnu.so")
        self.write("pkg/core/__init__.py", "from . import _speedups\n")
        self.write(
            "pkg/api.py",
            "\n".join(
                [
                    "from pkg.core import _speedups",
                    "",
                    "def total(xs):",
                    "    return _speedups.fast_sum(xs)",
                    "",
                ]
            ),
        )
        edges = self.edges(self.generate())
        self.assertIn(("pkg.api.total", "pkg.core._speedups.fast_sum"), edges)
        self.assertIn(("pkg.core", "pkg.core._speedups"), edges)

    def test_packed_submodules_stay_under_the_file_module(self):
        """One .so may hold several modules; PyCG does not know that and
        does not pretend to: every name stays under the file's module."""
        self.write("pkg/__init__.py")
        self.write("pkg/_rust.abi3.so")
        self.write("pkg/_rust/openssl/ec.pyi")
        self.write(
            "pkg/ec.py",
            "\n".join(
                [
                    "from pkg._rust import openssl as rust_openssl",
                    "",
                    "def supported(c):",
                    "    return rust_openssl.ec.curve_supported(c)",
                    "",
                ]
            ),
        )
        cg = self.generate()
        edges = self.edges(cg)
        self.assertIn(("pkg.ec", "pkg._rust"), edges)
        self.assertIn(("pkg.ec.supported", "pkg._rust.openssl.ec.curve_supported"), edges)
        doc = self.fasten(cg)
        self.assertIn("/pkg._rust/", doc["modules"]["internal"])
        self.assertNotIn("/pkg._rust.openssl/", doc["modules"]["internal"])


class SuffixTest(TestBase):
    def test_strip_extension_suffix(self):
        cases = {
            "_ext.cpython-311-x86_64-linux-gnu.so": "_ext",
            "_ext.abi3.so": "_ext",
            "m.cp313-win_amd64.pyd": "m",
            "x.so": "x",
            "foo.bar.so": "foo.bar",
            "libfoo.so.1": None,
            ".so": None,
        }
        for name, stem in cases.items():
            self.assertEqual(native.strip_extension_suffix(name), stem, name)

