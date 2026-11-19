"""Scoring policy tests that need neither a model nor the optional Ko checkers."""

import copy
import importlib.util
import sys
import types
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import patch


SCORER = Path(__file__).resolve().parents[3] / "src/lighteval/tasks/tasks/ifeval_ko/scoring.py"
VERIFIER = Path(__file__).resolve().parents[2] / "ifeval_ko/verify_scoring.py"


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.events, self.instances = [], []
        self.predicate = lambda instruction, value: False
        self.error = None
        owner = self

        class Instruction:
            def __init__(self, iid):
                self.id = iid
                self.kwargs = {}
                self.builds = []
                owner.instances.append(self)
                owner.events.append(("create", self))

            def build_description(self, **kwargs):
                self.builds.append(kwargs)
                self.kwargs.update(kwargs)
                owner.events.append(("build", self))

            def get_instruction_args(self):
                return self.kwargs.copy()

            def check_following(self, value):
                owner.events.append(("check", self, value))
                if owner.error is not None:
                    raise owner.error
                return owner.predicate(self, value)

        package = types.ModuleType("_test_ifeval_ko")
        package.__path__ = [str(SCORER.parent)]
        registry = types.ModuleType(package.__name__ + ".instructions_registry")
        registry.INSTRUCTION_DICT = {"same": Instruction, "other": Instruction}
        package.instructions_registry = registry
        self.modules = patch.dict(sys.modules, {package.__name__: package, registry.__name__: registry})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(package.__name__ + ".scoring", SCORER)
        self.scoring = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.scoring
        spec.loader.exec_module(self.scoring)

    def test_repeated_ids_keep_index_and_all_strict_precedes_all_loose(self):
        self.predicate = lambda instruction, value: instruction.kwargs["target"] in value
        result = self.scoring.score_response("raw prompt", ["same", "same"], [{"target": "a"}, {"target": "b"}], "a")
        self.assertEqual(
            result,
            {
                "prompt_level_strict_acc": False,
                "inst_level_strict_acc": [True, False],
                "prompt_level_loose_acc": False,
                "inst_level_loose_acc": [True, False],
            },
        )
        self.assertEqual(len(self.instances), 4)
        self.assertEqual([i.kwargs["target"] for i in self.instances], ["a", "b", "a", "b"])
        checks = [event[1] for event in self.events if event[0] == "check"]
        self.assertEqual(checks[:2], self.instances[:2])
        self.assertEqual(checks[2:], [self.instances[2], self.instances[3], self.instances[3]])
        self.assertEqual(len({id(i) for i in self.instances}), 4)

    def test_nfc_runs_at_each_phase_entry(self):
        raw = unicodedata.normalize("NFD", "한글")
        normalize = unicodedata.normalize
        self.predicate = lambda instruction, value: value == "한글"
        with patch.object(self.scoring.unicodedata, "normalize", wraps=normalize) as observed:
            result = self.scoring.score_response("p", ["same"], [{}], raw)
        self.assertEqual(observed.call_args_list, [unittest.mock.call("NFC", raw)] * 2)
        self.assertTrue(result["prompt_level_strict_acc"])
        self.assertTrue(result["prompt_level_loose_acc"])

    def test_raw_prompt_kwargs_and_reference_prompt_rebuild(self):
        raw = unicodedata.normalize("NFD", "한글")
        kwargs = [
            {
                "prompt": "placeholder",
                "keyword": raw,
                "zero": 0,
                "false": False,
                "null": None,
                "empty": "",
                "list": [],
                "dict": {},
                "negative": -1,
            }
        ]
        before = copy.deepcopy(kwargs)
        self.scoring.score_response(raw, ["same"], kwargs, "text")
        for instance in self.instances:
            self.assertEqual(
                instance.builds, [{"prompt": "placeholder", "keyword": raw, "negative": -1}, {"prompt": raw}]
            )
        self.assertEqual(kwargs, before)

    def test_loose_candidate_order_preserves_spaces_markers_and_duplicates(self):
        raw = " *첫* \n >중간* \n #끝* "
        self.scoring.score_loose("p", ["same"], [{}], raw)
        self.assertEqual(
            [e[2] for e in self.events if e[0] == "check"],
            [
                raw,
                " 첫 \n >중간 \n #끝 ",
                ">중간* \n #끝*",
                "*첫* \n >중간*",
                ">중간*",
                ">중간 \n #끝",
                "첫 \n >중간",
                ">중간",
            ],
        )
        self.events.clear()
        self.scoring.score_loose("p", ["same"], [{}], "same")
        self.assertEqual([e[2] for e in self.events if e[0] == "check"], ["same", "same"])

    def test_loose_is_independent_or_for_each_instruction(self):
        self.predicate = lambda instruction, value: value == instruction.kwargs["target"]
        result = self.scoring.score_response("p", ["same", "other"], [{"target": "*yes*"}, {"target": "yes"}], "*yes*")
        self.assertEqual(result["inst_level_strict_acc"], [True, False])
        self.assertEqual(result["inst_level_loose_acc"], [True, True])
        self.assertTrue(result["prompt_level_loose_acc"])

    def test_blank_response_builds_checkers_without_checking(self):
        for response in ["", " \t\n "]:
            with self.subTest(response=response):
                self.events.clear()
                result = self.scoring.score_response("p", ["same"], [{}], response)
                self.assertEqual(result["inst_level_strict_acc"], [False])
                self.assertEqual(result["inst_level_loose_acc"], [False])
                self.assertEqual([e[0] for e in self.events], ["create", "build", "create", "build"])
        # Registry lookup still happens, before indexing kwargs, even when blank.
        for fn in [self.scoring.score_strict, self.scoring.score_loose]:
            with self.assertRaises(KeyError):
                fn("p", ["unknown"], [], "")

    def test_empty_instruction_list_preserves_reference_all_semantics(self):
        result = self.scoring.score_response("p", [], [], "")
        self.assertEqual(
            result,
            {
                "prompt_level_strict_acc": True,
                "inst_level_strict_acc": [],
                "prompt_level_loose_acc": True,
                "inst_level_loose_acc": [],
            },
        )

    def test_external_error_propagates_and_combined_stops_before_loose(self):
        self.error = ValueError("checker failure")
        for fn in [self.scoring.score_strict, self.scoring.score_loose, self.scoring.score_response]:
            self.instances.clear()
            with self.assertRaises(ValueError) as caught:
                fn("p", ["same", "same"], [{}, {}], "text")
            self.assertIs(caught.exception, self.error)
            self.assertEqual(len(self.instances), 1)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_test_ko_verifier", VERIFIER)
        cls.verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verifier)

    def test_incorrect_second_repeated_id_is_reported_with_its_index(self):
        case = {"case_id": "repeat", "key": 1203, "ko_instruction_id_list": ["same", "same"]}
        reference = {
            "inst_level_strict_acc": [True, False],
            "prompt_level_strict_acc": False,
            "inst_level_loose_acc": [True, False],
            "prompt_level_loose_acc": False,
        }
        for wrong in [[True, True], [True], [True, 0], [True, False, False]]:
            actual = copy.deepcopy(reference)
            actual["inst_level_strict_acc"] = wrong
            mismatches = []
            self.verifier.compare_flags(case, actual, reference, mismatches, "negative control")
            self.assertTrue(mismatches)
            self.assertEqual(mismatches[0]["key"], 1203)
            self.assertEqual(mismatches[0]["instruction_index"], 2 if len(wrong) == 3 else 1)

    def test_metric_denominators_flatten_uneven_instruction_counts(self):
        values = [
            {
                f"{level}_level_{kind}_acc": all(flags) if level == "prompt" else flags
                for kind in ["strict", "loose"]
                for level in ["inst", "prompt"]
            }
            for flags in [[True, True, True], [False]]
        ]
        counts = self.verifier.counts(values)
        for kind in ["strict", "loose"]:
            self.assertEqual(counts[f"prompt_level_{kind}_acc"], {"numerator": 1, "denominator": 2, "value": 0.5})
            self.assertEqual(counts[f"inst_level_{kind}_acc"], {"numerator": 3, "denominator": 4, "value": 0.75})

    def test_missing_and_duplicate_case_ids_fail(self):
        for records in [[{"case_id": "a"}], [{"case_id": "a"}, {"case_id": "a"}]]:
            with self.assertRaises(ValueError):
                self.verifier.index_records(records, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
