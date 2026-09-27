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
from pycg.formats import Fasten
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


class TimeBudgetTest(TestBase):
    """A slow package must still produce a call graph.

    `CallGraphGenerator.analyze` used to wrap `complete_definitions` in
    `signal.alarm(60 * 30)` whose handler called `sys.exit(1)`. A package
    whose pointer fixpoint took longer than half an hour therefore produced
    no output at all -- not a coarser graph, nothing -- and the caller saw
    only a missing file and a non-zero exit code. The budget replaces it:
    the iterative phases stop iterating, say so, and the graph built so far
    is written out.
    """

    SNIPPET = "\n".join(
        [
            "def callee():",
            "    pass",
            "",
            "",
            "def caller():",
            "    callee()",
            "",
            "",
            "caller()",
            "",
        ]
    )

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "mod.py")
        with open(self.path, "w") as f:
            f.write(self.SNIPPET)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _generate(self, **kwargs):
        cg = CallGraphGenerator(
            [self.path], self.tmpdir, -1, utils.constants.CALL_GRAPH_OP, False,
            **kwargs
        )
        return cg

    def _fasten(self, cg):
        return Fasten(cg, self.tmpdir, "prod", "PyPI", "1.0", 0).generate()

    def test_no_budget_runs_to_completion(self):
        cg = self._generate()
        cg.analyze()
        self.assertIsNone(cg.incomplete)
        self.assertIn("mod.caller", cg.output())
        self.assertIn("mod.callee", cg.output()["mod.caller"])
        self.assertNotIn("incomplete", self._fasten(cg))

    def test_exhausted_budget_still_outputs_and_is_recorded(self):
        # A deadline already in the past when the first iterative phase is
        # reached: every one of them must bail out on its first check.
        cg = self._generate(time_budget=1e-9)

        with self.assertLogs("pycg.pycg", level="WARNING") as ctx:
            cg.analyze()

        message = "\n".join(ctx.output)
        self.assertIn("time budget", message)
        self.assertIn("fixpoint-in-progress", message)

        self.assertIsNotNone(cg.incomplete)
        self.assertIn("complete_definitions", cg.incomplete["phase"])
        self.assertEqual(cg.incomplete["iterations"], 0)
        self.assertEqual(cg.incomplete["timeBudgetSeconds"], 1e-9)
        self.assertTrue(cg.incomplete["phasesStopped"])

        # The whole point: there is still a call graph, and it is still the
        # output of a normal run -- nothing was thrown away.
        output = self._fasten(cg)
        self.assertEqual(output["incomplete"], cg.incomplete)
        self.assertTrue(output["modules"]["internal"])
        self.assertTrue(output["graph"]["internalCalls"])

    def test_generous_budget_is_not_hit(self):
        cg = self._generate(time_budget=600)
        cg.analyze()
        self.assertIsNone(cg.incomplete)
        self.assertNotIn("incomplete", self._fasten(cg))
