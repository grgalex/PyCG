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
import logging
import os
import traceback

from pycg import utils
from pycg.machinery.callgraph import CallGraph
from pycg.machinery.classes import ClassManager
from pycg.machinery.definitions import DefinitionManager
from pycg.machinery.imports import ImportManager
from pycg.machinery.key_err import KeyErrors
from pycg.machinery.modules import ModuleManager
from pycg.machinery.scopes import ScopeManager
from pycg.processing.cgprocessor import CallGraphProcessor
from pycg.processing.keyerrprocessor import KeyErrProcessor
from pycg.processing.postprocessor import PostProcessor
from pycg.processing.preprocessor import PreProcessor
# import tracemalloc
# tracemalloc.start()
# import objgraph
import time


log = logging.getLogger(__name__)


class CallGraphGenerator(object):
    def __init__(
        self,
        entry_points,
        package,
        max_iter,
        operation,
        no_analyze_external,
        time_budget=0,
    ):
        self.entry_points = entry_points
        self.package = package
        self.no_analyze_external = no_analyze_external
        self.state = None
        self.max_iter = max_iter
        self.operation = operation
        # Wall-clock seconds the iterative phases may spend in total, or 0
        # for no limit. `analyze` turns it into `self.deadline`.
        self.time_budget = time_budget or 0
        self.deadline = None
        # Set by `_record_incomplete` when a phase runs out of budget; the
        # FASTEN formatter copies it into the output so a consumer can tell a
        # converged graph from a fixpoint-in-progress.
        self.incomplete = None
        self.setUp()
        self.defs_per_module = {}

    def _budget_exhausted(self):
        return self.deadline is not None and time.monotonic() > self.deadline

    def _record_incomplete(self, phase, iterations):
        """Note that `phase` stopped early, and say so loudly.

        The graph built so far is kept and written out: every phase bounded
        here only ever adds names to points-to sets or edges to the graph, so
        stopping early loses precision (missing edges) and never soundness
        (no edge that should not be there). This used to be a
        `signal.alarm(30 * 60)` whose handler called `sys.exit(1)`, which
        threw the whole analysis away and left the caller with no output file
        at all.
        """
        log.warning(
            "time budget of %ss exhausted during %s after %d iteration(s); "
            "the call graph is a sound but less precise "
            "fixpoint-in-progress and is being written out as is",
            self.time_budget,
            phase,
            iterations,
        )
        if self.incomplete is None:
            self.incomplete = {
                "phase": phase,
                "iterations": iterations,
                "timeBudgetSeconds": self.time_budget,
                "phasesStopped": [],
            }
        self.incomplete["phasesStopped"].append(
            {"phase": phase, "iterations": iterations}
        )

    def setUp(self):
        self.import_manager = ImportManager()
        self.scope_manager = ScopeManager()
        self.def_manager = DefinitionManager()
        self.class_manager = ClassManager()
        self.module_manager = ModuleManager()
        self.cg = CallGraph()
        self.key_errs = KeyErrors()

    def extract_state(self):
        state = {}
        state["defs"] = {}
        for key, defi in self.def_manager.get_defs().items():
            state["defs"][key] = {
                "names": defi.get_name_pointer().get().copy(),
                "lit": defi.get_lit_pointer().get().copy(),
            }

        state["scopes"] = {}
        for key, scope in self.scope_manager.get_scopes().items():
            state["scopes"][key] = set([
                x.get_ns() for (_, x) in scope.get_defs().items()
            ])

        state["classes"] = {}
        for key, ch in self.class_manager.get_classes().items():
            state["classes"][key] = ch.get_mro().copy()
        return state

    def reset_counters(self):
        for key, scope in self.scope_manager.get_scopes().items():
            scope.reset_counters()

    def has_converged(self):
        if not self.state:
            return False

        curr_state = self.extract_state()

        # check defs
        for key, defi in curr_state["defs"].items():
            if key not in self.state["defs"]:
                return False
            if defi["names"] != self.state["defs"][key]["names"]:
                return False
            if defi["lit"] != self.state["defs"][key]["lit"]:
                return False

        # check scopes
        for key, scope in curr_state["scopes"].items():
            if key not in self.state["scopes"]:
                return False
            if scope != self.state["scopes"][key]:
                return False

        # check classes
        for key, ch in curr_state["classes"].items():
            if key not in self.state["classes"]:
                return False
            if ch != self.state["classes"][key]:
                return False

        return True

    def remove_import_hooks(self):
        self.import_manager.remove_hooks()

    def tearDown(self):
        self.remove_import_hooks()

    def _get_mod_name(self, entry, pkg):
        # We do this because we want __init__ modules to
        # only contain the parent module
        # since pycg can't differentiate between functions
        # coming from __init__ files.

        input_mod = utils.to_mod_name(os.path.relpath(entry, pkg))
        if input_mod.endswith("__init__"):
            input_mod = ".".join(input_mod.split(".")[:-1])

        return input_mod

    def do_pass(self, cls, install_hooks=False, *args, _budgeted=False, **kwargs):
        modules_analyzed = set()
        entry_points_done = 0
        # count = 0
        for entry_point in self.entry_points:
            # A budgeted pass checks the deadline between entry points: one
            # entry point is one unit of work here, and the passes that opt
            # in (post-processing) only refine what is already there.
            if _budgeted and self._budget_exhausted():
                self._record_incomplete(
                    "the %s pass (%d of %d entry points)"
                    % (cls.__name__, entry_points_done, len(self.entry_points)),
                    entry_points_done,
                )
                break
            entry_points_done += 1
            # m1 = tracemalloc.take_snapshot()
            input_mod = None
            try:
                # print(entry_point)
                # old_len_defs = len(self.def_manager.defs)
                input_pkg = self.package
                input_mod = self._get_mod_name(entry_point, input_pkg)
                input_file = os.path.abspath(entry_point)

                if not input_mod:
                    log.info(f'no mod_name for entry_point {entry_point}')
                    continue

                if not input_pkg:
                    input_pkg = os.path.dirname(input_file)

                if input_mod not in modules_analyzed:
                    if install_hooks:
                        self.import_manager.set_pkg(input_pkg)
                        self.import_manager.install_hooks()

                    processor = cls(
                        input_file,
                        input_mod,
                        modules_analyzed=modules_analyzed,
                        *args,
                        **kwargs,
                    )

                    processor.analyze()

                    modules_analyzed = modules_analyzed.union(
                        processor.get_modules_analyzed()
                    )

                    if install_hooks:
                        self.remove_import_hooks()

            except Exception:
                if install_hooks:
                    self.remove_import_hooks()
                # Recovery is unchanged -- we still move on to the next entry
                # point -- but the failure is no longer silent. Whatever the
                # processor had not reached yet is simply missing from the
                # call graph, and `input_mod` was already added to
                # `modules_analyzed` by ProcessingBase.__init__, so the module
                # is never revisited: without this warning a module could end
                # up in the output holding nothing but its own namespace and
                # nobody would know.
                log.warning(
                    "%s pass failed for entry point %s (module %r); its "
                    "definitions are missing from the call graph:\n%s",
                    cls.__name__,
                    entry_point,
                    input_mod,
                    traceback.format_exc(),
                )
            # new_len_defs = len(self.def_manager.defs)
            # defs_added = new_len_defs - old_len_defs
            # if defs_added > 0:
            #     self.defs_per_module[entry_point] = defs_added
            # m2 = tracemalloc.take_snapshot()
            # top_stats = m2.compare_to(m1, 'lineno')
            # for stat in top_stats[:2]:
            #     print(stat)

    def analyze(self):
        if self.time_budget > 0:
            self.deadline = time.monotonic() + self.time_budget
            log.info(
                "iterative phases have a time budget of %ss", self.time_budget
            )
        # objgraph.show_growth(limit=5)
        self.do_pass(
            PreProcessor,
            True,
            self.import_manager,
            self.scope_manager,
            self.def_manager,
            self.class_manager,
            self.module_manager,
        )
        # objgraph.show_growth(limit=5)
        # self.defs_per_module = dict(sorted(self.defs_per_module.items(), key=lambda item: item[1]))
        # print(f'{self.defs_per_module}')
        # self.defs_per_module = {}
        # print(f'Completing definitions, len = {len(self.def_manager.defs)}')

        sweeps, timed_out = self.def_manager.complete_definitions(
            False, deadline=self.deadline
        )
        if timed_out:
            self._record_incomplete("the initial complete_definitions", sweeps)
        # objgraph.show_growth(limit=5)
        iter_cnt = 0
        while self.max_iter < 0 or iter_cnt < self.max_iter:
            # The budget is checked before `has_converged`, which rebuilds the
            # whole state snapshot and is itself expensive on a large package.
            if self._budget_exhausted():
                self._record_incomplete("the pointer refinement loop", iter_cnt)
                break
            if self.has_converged():
                break
            # objgraph.show_growth(limit=5)
            self.state = self.extract_state()
            self.reset_counters()
            # objgraph.show_growth(limit=5)
            # print('ENTERING do_pass PostProcessor')
            self.do_pass(
                PostProcessor,
                False,
                self.import_manager,
                self.scope_manager,
                self.def_manager,
                self.class_manager,
                self.module_manager,
                _budgeted=True,
            )
            # objgraph.show_growth(limit=5)

            self.defs_per_module = dict(sorted(self.defs_per_module.items(), key=lambda item: item[1]))
            # print(f'{self.defs_per_module}')
            self.defs_per_module = {}

            # print(f'Completing definitions, len = {len(self.def_manager.defs)}')
            sweeps, timed_out = self.def_manager.complete_definitions(
                self.no_analyze_external, deadline=self.deadline
            )
            iter_cnt += 1
            if timed_out:
                self._record_incomplete(
                    "complete_definitions of refinement iteration %d" % iter_cnt,
                    sweeps,
                )

        self.reset_counters()
        if self.operation == utils.constants.CALL_GRAPH_OP:
            self.do_pass(
                CallGraphProcessor,
                False,
                self.import_manager,
                self.scope_manager,
                self.def_manager,
                self.class_manager,
                self.module_manager,
                call_graph=self.cg,
            )
        elif self.operation == utils.constants.KEY_ERR_OP:
            self.do_pass(
                KeyErrProcessor,
                False,
                self.import_manager,
                self.scope_manager,
                self.def_manager,
                self.class_manager,
                self.key_errs,
            )
        else:
            raise Exception("Invalid operation: " + self.operation)

    def output(self):
        return self.cg.get()

    def output_key_errs(self):
        return self.key_errs.get()

    # Redefined in line 227
    # def output_edges(self):
    #     return self.key_errors

    def output_edges(self):
        return self.cg.get_edges()

    def _generate_mods(self, mods):
        res = {}
        for mod, node in mods.items():
            res[mod] = {
                "filename": (
                    os.path.relpath(node.get_filename(), self.package)
                    if node.get_filename()
                    else None
                ),
                "methods": node.get_methods(),
            }
        return res

    def output_internal_mods(self):
        return self._generate_mods(self.module_manager.get_internal_modules())

    def output_external_mods(self):
        return self._generate_mods(self.module_manager.get_external_modules())

    def output_functions(self):
        functions = []
        for ns, defi in self.def_manager.get_defs().items():
            if defi.is_function_def():
                functions.append(ns)
        return functions

    def output_classes(self):
        classes = {}
        for cls, node in self.class_manager.get_classes().items():
            classes[cls] = {"mro": node.get_mro(), "module": node.get_module()}
        return classes

    def get_as_graph(self):
        return self.def_manager.get_defs().items()
