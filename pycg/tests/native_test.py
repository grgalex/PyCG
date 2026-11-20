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
"""Native code reached from Python: compiled-module imports and dlopen."""
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


class DynamicLoadTest(NativeTestBase):
    def test_loaders_and_library_name_forms(self):
        self.write("pkg/__init__.py")
        self.write("pkg/libfoo.so")
        self.write(
            "pkg/mod.py",
            "\n".join(
                [
                    "import ctypes, ctypes.util, os",
                    "from ctypes import CDLL",
                    "import cffi",
                    "_LIB = 'libbar.so'",
                    "a = ctypes.CDLL('foo')",
                    "b = CDLL(ctypes.util.find_library('m'))",
                    "c = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libfoo.so'))",
                    "d = ctypes.PyDLL(_LIB)",
                    "ffi = cffi.FFI()",
                    "e = ffi.dlopen('libbaz.so.1')",
                    "",
                    "def use():",
                    "    a.fa()",
                    "    b.fb()",
                    "    c.fc()",
                    "    d.fd()",
                    "    e.fe()",
                    "    ctypes.CDLL('libx.so').fx()",
                    "",
                ]
            ),
        )
        cg = self.generate()
        edges = self.edges(cg)
        for lib, sym in (("foo", "fa"), ("m", "fb"), ("libfoo.so", "fc"), ("libbar.so", "fd"), ("libbaz.so.1", "fe"), ("libx.so", "fx")):
            self.assertIn(("pkg.mod.use", "dlopen:%s.%s" % (lib, sym)), edges)
        # the loader itself is not a callee any more
        self.assertFalse(any(dst.startswith("ctypes.CDLL.") for _, dst in edges))

        doc = self.fasten(cg)
        loads = {(l["namespace"], l["line"]): l for l in doc["dynamicLoads"]}
        self.assertEqual(loads[("pkg.mod", 5)]["resolution"], "literal")
        self.assertEqual(loads[("pkg.mod", 6)]["resolution"], "find_library")
        self.assertEqual(loads[("pkg.mod", 7)]["resolution"], "path-join")
        self.assertEqual(loads[("pkg.mod", 7)]["file"], "pkg/libfoo.so")
        self.assertEqual(loads[("pkg.mod", 8)]["resolution"], "name-constant")
        self.assertEqual(loads[("pkg.mod", 10)]["loader"], "cffi.FFI.dlopen")
        self.assertEqual(loads[("pkg.mod.use", 18)]["library"], "libx.so")
        mod = doc["modules"]["external"]["dlopen:libfoo.so"]
        self.assertEqual(mod["sourceFile"], "pkg/libfoo.so")
        self.assertEqual(mod["metadata"]["kind"], "dynamic-library")
        self.assertEqual(mod["metadata"]["loaders"], ["ctypes.cdll.LoadLibrary"])
        self.assertEqual(self.external(doc, "dlopen:foo")["dlopen:foo.fa"], {"kind": "dynamic-symbol", "symbol": "fa"})

    def test_handles_follow_aliases_parameters_and_attributes(self):
        self.write("pkg/__init__.py")
        self.write(
            "pkg/mod.py",
            "\n".join(
                [
                    "import ctypes",
                    "x = ctypes.CDLL('libfoo.so')",
                    "lib = x",
                    "",
                    "def call_it(l):",
                    "    l.foo_param()",
                    "",
                    "call_it(lib)",
                    "",
                    "class W:",
                    "    _lib = x",
                    "    def run(self):",
                    "        return self._lib.foo_method()",
                    "",
                ]
            ),
        )
        edges = self.edges(self.generate())
        self.assertIn(("pkg.mod.call_it", "dlopen:libfoo.so.foo_param"), edges)
        self.assertIn(("pkg.mod.W.run", "dlopen:libfoo.so.foo_method"), edges)

    def test_wrappers_around_loaders_are_loaders(self):
        self.write("pkg/__init__.py")
        self.write("pkg/sub/__init__.py")
        self.write("pkg/sub/_raw_x.abi3.so")
        self.write(
            "pkg/_raw_api.py",
            "\n".join(
                [
                    "import os",
                    "from cffi import FFI",
                    "ffi = FFI()",
                    "",
                    "def load_lib(name, cdecl):",
                    "    lib = ffi.dlopen(name)",
                    "    ffi.cdef(cdecl)",
                    "    return lib",
                    "",
                    "def load_raw(name, cdecl):",
                    "    split = name.split('.')",
                    "    full_name = os.path.join('/x', *split[:-1], split[-1] + '.abi3.so')",
                    "    return load_lib(full_name, cdecl)",
                    "",
                ]
            ),
        )
        self.write(
            "pkg/sub/x.py",
            "\n".join(
                [
                    "from pkg._raw_api import load_raw",
                    "_raw_x = load_raw('pkg.sub._raw_x', 'int x_run(int);')",
                    "",
                    "def run(v):",
                    "    return _raw_x.x_run(v)",
                    "",
                ]
            ),
        )
        cg = self.generate()
        self.assertIn(("pkg.sub.x.run", "dlopen:pkg.sub._raw_x.x_run"), self.edges(cg))
        doc = self.fasten(cg)
        site = [l for l in doc["dynamicLoads"] if l["namespace"] == "pkg.sub.x"][0]
        self.assertEqual(site["loader"], "cffi.FFI.dlopen")
        self.assertEqual(site["via"], ["pkg._raw_api.load_raw", "pkg._raw_api.load_lib"])
        self.assertEqual(site["file"], "pkg/sub/_raw_x.abi3.so")
        self.assertEqual(site["resolution"], "argument")
        body = [l for l in doc["dynamicLoads"] if l["namespace"] == "pkg._raw_api.load_lib"][0]
        self.assertEqual(body["resolution"], "parameter")
        self.assertIsNone(body["library"])

    def test_unresolved_name_is_still_a_site(self):
        self.write("pkg/__init__.py")
        self.write(
            "pkg/mod.py",
            "\n".join(
                [
                    "import ctypes",
                    "def load(v):",
                    "    return ctypes.CDLL(f'lib{v}.so')",
                    "",
                ]
            ),
        )
        doc = self.fasten(self.generate())
        self.assertEqual(len(doc["dynamicLoads"]), 1)
        self.assertEqual(doc["dynamicLoads"][0]["resolution"], "unresolved")
        self.assertIsNone(doc["dynamicLoads"][0]["library"])


class ExtensionAliasTest(NativeTestBase):
    """A module-level name bound to an attribute of a compiled module
    (`HMAC = rust_openssl.hmac.HMAC`) is that attribute wherever it is
    called from, and an object it returns carries its methods."""

    def test_alias_of_a_compiled_attribute(self):
        self.write("pkg/__init__.py")
        self.write("pkg/bindings/__init__.py")
        self.write("pkg/bindings/_rust.abi3.so")
        self.write(
            "pkg/hmac.py",
            "\n".join(
                [
                    "from pkg.bindings._rust import openssl as rust_openssl",
                    "HMAC = rust_openssl.hmac.HMAC",
                    "Numbers = rust_openssl.dsa.Numbers",
                    "",
                ]
            ),
        )
        # processed before pkg.hmac (alphabetical), within one iteration
        self.write(
            "pkg/fernet.py",
            "\n".join(
                [
                    "from pkg.hmac import HMAC",
                    "from pkg import hmac as hm",
                    "def enc(k):",
                    "    h = HMAC(k)",
                    "    h.update(k)",
                    "    return hm.Numbers(k)",
                    "",
                ]
            ),
        )
        edges = self.edges(self.generate())
        self.assertIn(("pkg.fernet.enc", "pkg.bindings._rust.openssl.hmac.HMAC"), edges)
        self.assertIn(("pkg.fernet.enc", "pkg.bindings._rust.openssl.hmac.HMAC.update"), edges)
        self.assertIn(("pkg.fernet.enc", "pkg.bindings._rust.openssl.dsa.Numbers"), edges)


class LoadArgumentTest(NativeTestBase):
    """Library names that reach the loader through a variable, a loop, a
    name with a hole, a tuple, or a function that loads on the side."""

    def _loads(self, cg):
        return {(l["namespace"], l["line"]): l for l in self.fasten(cg)["dynamicLoads"]}

    def test_loop_over_a_generator_of_names(self):
        # python-magic: `for lib in _lib_candidates(): ctypes.CDLL(lib)`;
        # the yielded literal sits on the generator's return definition
        self.write("pkg/__init__.py")
        self.write(
            "pkg/loader.py",
            "\n".join(
                [
                    "import ctypes",
                    "from ctypes.util import find_library",
                    "def _candidates():",
                    "    yield find_library('magic')",
                    "    yield './%s.dll' % 'libmagic'",
                    "    yield 'libmagic.so.1'",
                    "def load_lib():",
                    "    for lib in _candidates():",
                    "        return ctypes.CDLL(lib)",
                    "",
                ]
            ),
        )
        self.write(
            "pkg/__init__.py",
            "\n".join(
                [
                    "from . import loader",
                    "libmagic = loader.load_lib()",
                    "_libraries = {}",
                    "_libraries['magic'] = loader.load_lib()",
                    "magic_open = libmagic.magic_open",
                    "def check():",
                    "    return _libraries['magic'].magic_check()",
                    "",
                ]
            ),
        )
        cg = self.generate()
        loads = self._loads(cg)
        self.assertEqual(loads[("pkg.loader.load_lib", 9)]["library"], "libmagic.so.1")
        # the template './%s.dll' is not a library name
        self.assertEqual(loads[("pkg.loader.load_lib", 9)]["resolution"], "name-constant")
        # a call of the loader function is a site of its own, via it
        self.assertEqual(loads[("pkg", 2)]["via"], ["pkg.loader.load_lib"])
        edges = self.edges(cg)
        # a read of the symbol counts, not only a call of it
        self.assertIn(("pkg", "dlopen:libmagic.so.1.magic_open"), edges)
        # a handle stored in a container item is found through the container
        self.assertIn(("pkg.check", "dlopen:libmagic.so.1.magic_check"), edges)

    def test_name_from_find_library_and_platform_chain(self):
        # soundfile: `_libname = find_library('sndfile'); ffi.dlopen(_libname)`
        self.write("pkg/__init__.py")
        self.write(
            "pkg/mod.py",
            "\n".join(
                [
                    "import sys, ctypes",
                    "from ctypes.util import find_library",
                    "_libname = find_library('sndfile')",
                    "_snd = ctypes.CDLL(_libname)",
                    "if sys.platform == 'darwin':",
                    "    _explicit = 'libsndfile.dylib'",
                    "else:",
                    "    _explicit = 'libsndfile.so'",
                    "_snd2 = ctypes.CDLL(_explicit)",
                    "version = _snd.sf_version_string()",
                    "FLAG = _snd.SFC_GET_LOG_INFO",
                    "",
                ]
            ),
        )
        cg = self.generate()
        loads = self._loads(cg)
        self.assertEqual(loads[("pkg.mod", 4)]["library"], "sndfile")
        self.assertEqual(loads[("pkg.mod", 4)]["resolution"], "name-constant")
        self.assertEqual(loads[("pkg.mod", 9)]["library"], ["libsndfile.dylib", "libsndfile.so"])
        edges = self.edges(cg)
        self.assertIn(("pkg.mod", "dlopen:sndfile.sf_version_string"), edges)
        # an UPPER_CASE read is a cdef constant, not a symbol
        self.assertNotIn(("pkg.mod", "dlopen:sndfile.SFC_GET_LOG_INFO"), edges)

    def test_name_with_a_hole_matches_a_packaged_file(self):
        # soundfile: os.path.join(dirname(__file__), 'libsndfile_' + machine() + '.so')
        self.write("pkg/__init__.py")
        self.write("pkg/_data/__init__.py")
        self.write("pkg/_data/libsnd_x86_64.so")
        self.write(
            "pkg/mod.py",
            "\n".join(
                [
                    "import os, ctypes",
                    "from platform import machine",
                    "from pkg import _data",
                    "_name = 'libsnd_' + machine() + '.so'",
                    "_path = os.path.join(os.path.dirname(_data.__file__), _name)",
                    "_snd = ctypes.CDLL(_path)",
                    "",
                ]
            ),
        )
        cg = self.generate()
        site = self._loads(cg)[("pkg.mod", 6)]
        self.assertEqual(site["file"], "pkg/_data/libsnd_x86_64.so")
        self.assertEqual(site["resolution"], "pattern")

    def test_suffix_only_hole_matches_nothing(self):
        self.write("pkg/__init__.py")
        self.write("pkg/_raw_x.abi3.so")
        self.write(
            "pkg/mod.py",
            "\n".join(
                [
                    "import ctypes",
                    "def load(name):",
                    "    return ctypes.CDLL(name.split('.')[-1] + '.abi3.so')",
                    "",
                ]
            ),
        )
        site = self._loads(self.generate())[("pkg.mod.load", 3)]
        self.assertIsNone(site["library"])

    def test_annotated_assignment_defines_the_ffi(self):
        # pycryptodome: `ffi: Any = cffi.FFI()` then `ffi.dlopen(name)`
        self.write("pkg/__init__.py")
        self.write(
            "pkg/mod.py",
            "\n".join(
                [
                    "from typing import Any",
                    "import cffi",
                    "ffi: Any = cffi.FFI()",
                    "def load_lib(name):",
                    "    return ffi.dlopen(name)",
                    "",
                ]
            ),
        )
        site = self._loads(self.generate())[("pkg.mod.load_lib", 5)]
        self.assertEqual(site["loader"], "cffi.FFI.dlopen")
        self.assertEqual(site["resolution"], "parameter")

    def test_loader_function_and_tuple_slot(self):
        # pycryptodome: `_ec_lib, _bmi2 = load_ec_lib(cdecl)` where load_ec_lib
        # loads a fixed library and returns (handle, flag); and a function
        # that loads but returns something else is still a loader
        self.write("pkg/__init__.py")
        self.write("pkg/_ec_nat.abi3.so")
        self.write(
            "pkg/_ec_lib.py",
            "\n".join(
                [
                    "import ctypes",
                    "def load_ec_lib(cdecl):",
                    "    return ctypes.CDLL('pkg._ec_nat'), True",
                    "def build():",
                    "    lib = ctypes.CDLL('pkg._ec_nat')",
                    "    return [lib.ec_new]",
                    "",
                ]
            ),
        )
        self.write(
            "pkg/ecc.py",
            "\n".join(
                [
                    "from pkg._ec_lib import load_ec_lib, build",
                    "_ec_lib, _bmi2 = load_ec_lib('int ec_new(void);')",
                    "_impl = build()",
                    "def new():",
                    "    return _ec_lib.ec_new()",
                    "",
                ]
            ),
        )
        cg = self.generate()
        loads = self._loads(cg)
        self.assertEqual(loads[("pkg.ecc", 2)]["via"], ["pkg._ec_lib.load_ec_lib"])
        self.assertEqual(loads[("pkg.ecc", 2)]["file"], "pkg/_ec_nat.abi3.so")
        self.assertEqual(loads[("pkg.ecc", 3)]["via"], ["pkg._ec_lib.build"])
        self.assertIn(("pkg.ecc.new", "dlopen:pkg._ec_nat.ec_new"), self.edges(cg))

    def test_lambda_passed_to_an_unknown_callee_is_defined(self):
        # soundfile: `_errorcode = property(lambda self: _snd.sf_error(self._file))`
        self.write("pkg/__init__.py")
        self.write(
            "pkg/mod.py",
            "\n".join(
                [
                    "import ctypes",
                    "_snd = ctypes.CDLL('libsndfile.so')",
                    "class SoundFile:",
                    "    name = property(lambda self: self._name)",
                    "    _errorcode = property(lambda self: _snd.sf_error(self._file))",
                    "",
                ]
            ),
        )
        self.assertIn(
            ("pkg.mod.SoundFile.<lambda2>", "dlopen:libsndfile.so.sf_error"),
            self.edges(self.generate()),
        )


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

