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
import os
import shutil
import tempfile

from base import TestBase

from pycg import utils
from pycg.processing.preprocessor import PreProcessor
from pycg.pycg import CallGraphGenerator


class DoPassFailureTest(TestBase):
    """A module that blows up mid-analysis must not disappear quietly.

    `CallGraphGenerator.do_pass` wraps every entry point in a bare
    `except Exception: pass`. The module has already been added to
    `modules_analyzed` by `ProcessingBase.__init__`, so it is never revisited
    and the output keeps nothing but its module namespace -- with no hint at
    all that anything went wrong.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _entry_point(self, name, contents):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(contents)
        return path

    def test_failure_is_logged_with_module_and_traceback(self):
        path = self._entry_point(
            "boom.py",
            "\n".join(["def f(a, b=1, /, c=2):", "    return a", ""]),
        )
        cg = CallGraphGenerator(
            [path], self.tmpdir, 1, utils.constants.CALL_GRAPH_OP, False
        )

        boom = RuntimeError("kaboom")

        def explode(self):
            raise boom

        original = PreProcessor.analyze
        PreProcessor.analyze = explode
        try:
            with self.assertLogs("pycg.pycg", level="WARNING") as ctx:
                cg.do_pass(
                    PreProcessor,
                    True,
                    cg.import_manager,
                    cg.scope_manager,
                    cg.def_manager,
                    cg.class_manager,
                    cg.module_manager,
                )
        finally:
            PreProcessor.analyze = original

        message = "\n".join(ctx.output)
        self.assertIn("PreProcessor", message)
        self.assertIn("boom", message)  # names the module that was lost
        self.assertIn("kaboom", message)  # carries the traceback
        self.assertIn("Traceback", message)
