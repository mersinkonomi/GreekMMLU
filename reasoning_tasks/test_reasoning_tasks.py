"""Regression checks for the isolated reasoning evaluation protocol.

Run with the evaluation environment and its offline dataset cache, for example:
    PYTHONPATH=lm-evaluation-harness python reasoning_tasks/test_reasoning_tasks.py
"""

from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from greekmmlu import reasoning_utils as helpers
from lm_eval.tasks import TaskManager
from lm_eval.utils import load_yaml_config


ROOT = Path(__file__).resolve().parent


class ReasoningProtocolTest(unittest.TestCase):
    def setUp(self):
        flags = patch.dict(os.environ, {helpers.REQUIRE_THINKING_CLOSE_ENV: "0"})
        flags.start()
        self.addCleanup(flags.stop)

    def test_prompt_never_reads_gold_and_only_lists_available_options(self):
        for count in (2, 3, 4):
            doc = {
                "question": "Ποια επιλογή ταιριάζει;",
                "choices": [f"Επιλογή {i}" for i in range(count)],
                "subject": "Mathematics",
            }
            prompt = helpers.doc_to_text(doc)
            for answer in range(count):
                self.assertEqual(prompt, helpers.doc_to_text({**doc, "answer": answer}))
            for label in helpers.LABELS[:count]:
                self.assertIn(f"{label}. Επιλογή", prompt)
            for label in helpers.LABELS[count:]:
                self.assertNotIn(f"{label}. ", prompt)
            self.assertIn("εξήγησε τον συλλογισμό", prompt)
            self.assertIn(r"Τελική απάντηση: \boxed{γράμμα}", prompt)
            self.assertNotIn("\b", prompt)

    def test_extracts_terminal_final_answer_and_preserves_raw_text(self):
        cases = [
            ("Εξετάζω την Α.\nΤελική απάντηση: \\boxed{Β}", 4, "Β"),
            ("Αρχικά \\boxed{Α}.\nΤελική απάντηση: \\boxed{δ.}", 4, "Δ"),
            ("Σκέψη.\nΤελική απάντηση: \\boxed{c}", 4, "Γ"),
            ("**Τελική απάντηση:** $\\boxed{B}$", 2, "Β"),
            ("Τελική απάντηση: \\boxed{Δ}", 2, helpers.INVALID),
            ("Αρχικά \\boxed{Α}, αλλά η ανάλυση δεν έχει τελειώσει", 4, helpers.INVALID),
            ("Τελική απάντηση: \\boxed{Α}\nΣυνεχίζω χωρίς συμπέρασμα", 4, helpers.INVALID),
            ("Σκέψη.\nΕρώτηση: επόμενη\nΤελική απάντηση: \\boxed{Γ}", 4, helpers.INVALID),
            ("Σκέψη.\nΤελική απάντηση: \\boxed{Β}\nΕρώτηση: επόμενη\nΤελική απάντηση: \\boxed{Γ}", 4, "Β"),
            ("Τελική απάντηση: \\boxed{", 4, helpers.INVALID),
            ("<think>Η ανάλυση δεν τελείωσε.\nΤελική απάντηση: \\boxed{Α}", 4, helpers.INVALID),
            ("<think>Τελική απάντηση: \\boxed{Α}</think>", 4, helpers.INVALID),
            ("<think>Τελική απάντηση: \\boxed{Α}</think>\nΤελική απάντηση: \\boxed{Β}", 4, "Β"),
            ("Τελική απάντηση: \\boxed{Β}<|im_end|>", 4, "Β"),
            ("Τελική απάντηση: \\boxed{Β}<|ifm|im_end|>", 4, "Β"),
            ("<think>Σκέψη.</think>\nΤελική απάντηση: \\boxed{Β}\nΕρώτηση: επόμενη\n<think>Άλλη σκέψη.</think>\nΤελική απάντηση: \\boxed{Γ}", 4, "Β"),
            (None, 4, helpers.INVALID),
        ]
        for text, count, expected in cases:
            with self.subTest(text=text):
                responses = [[text]]
                before = deepcopy(responses)
                self.assertEqual(
                    helpers.extract_final_answers(responses, [{"choices": [None] * count}]),
                    [[expected]],
                )
                self.assertEqual(responses, before)

    def test_native_preopened_thinking_must_close_before_final_answer(self):
        final = "Τελική απάντηση: \\boxed{Β}"
        with patch.dict(os.environ, {helpers.REQUIRE_THINKING_CLOSE_ENV: "1"}):
            self.assertEqual(helpers.extract_final_answer(final, 4), helpers.INVALID)
            for opener, closer in helpers.THINKING_PAIRS.items():
                self.assertEqual(helpers.extract_final_answer("Σκέψη. " + closer + "\n" + final, 4), "Β")
                self.assertEqual(helpers.extract_final_answer(final + closer, 4), helpers.INVALID)
                self.assertEqual(helpers.extract_final_answer(closer + "\n" + opener + "\n" + final, 4), helpers.INVALID)
                self.assertTrue(helpers.reasoning_is_closed("Σκέψη. " + closer))
                self.assertFalse(helpers.reasoning_is_closed(opener + final))
                self.assertEqual(
                    helpers.extract_final_answer(
                        opener + "\nΕρώτηση: παράθεση της αρχικής\nΣυλλογισμός.\n"
                        + closer + "\n" + final,
                        4,
                    ),
                    "Β",
                )
                self.assertEqual(
                    helpers.extract_final_answer(
                        "\nΕρώτηση: παράθεση της αρχικής\nΣυλλογισμός.\n"
                        + closer + "\n" + final,
                        4,
                    ),
                    "Β",
                )
                self.assertEqual(
                    helpers.extract_final_answer(
                        "Συλλογισμός.\n" + closer + "\n" + final
                        + "\nΕρώτηση: επόμενη\n" + opener + "Άλλη σκέψη." + closer
                        + "\nΤελική απάντηση: \\boxed{Γ}",
                        4,
                    ),
                    "Β",
                )

    def test_task_manager_overrides_all_45_tasks_and_five_groups(self):
        manager = TaskManager(include_path=str(ROOT))
        leaves = list((ROOT / "greekmmlu").glob("greekmmlu_*.yaml"))
        groups = list((ROOT / "greekmmlu").glob("_greekmmlu*.yaml"))
        self.assertEqual(len(leaves), 45)
        self.assertEqual(len(groups), 5)
        for path in leaves:
            config = load_yaml_config(str(path))
            self.assertEqual(Path(manager.task_index[config["task"]]["yaml_path"]), path)
            self.assertEqual(config["output_type"], "generate_until")
            self.assertEqual(config["generation_kwargs"]["max_gen_toks"], 32768)
            self.assertFalse(any("Ερώτηση:" in stop for stop in config["generation_kwargs"]["until"]))
            self.assertEqual(config["metadata"]["protocol"], "reasoning_final_box_v1")
            self.assertEqual(config["metric_list"][0]["metric"], "exact_match")
            self.assertEqual(config["filter_list"][0]["name"], "boxed-extract")
        for path in groups:
            config = load_yaml_config(str(path))
            self.assertEqual(Path(manager.task_index[config["group"]]["yaml_path"]), path)
            metric = config["aggregate_metric_list"][0]
            self.assertEqual(metric["metric"], "exact_match")
            self.assertEqual(metric["filter_list"], "boxed-extract")

    def test_real_dataset_zero_and_five_shot_requests_and_scoring(self):
        manager = TaskManager(include_path=str(ROOT))
        task = manager.load_task_or_group(["greekmmlu_mathematics"])["greekmmlu_mathematics"]
        task.set_fewshot_seed(1234)
        doc = next(iter(task.test_docs()))
        for shots in (0, 5):
            context = task.fewshot_context(doc, shots)
            # Removing/changing the current sample's target cannot affect context.
            changed_gold = {**doc, "answer": (int(doc["answer"]) + 1) % len(doc["choices"])}
            self.assertEqual(context, task.fewshot_context(changed_gold, shots))
            self.assertTrue(context.endswith(helpers.doc_to_text(doc)))
            self.assertEqual(context.count("Ερώτηση: "), shots + 1)
            # Only the dev examples have real boxed gold letters; the user-facing
            # format instruction uses the generic placeholder 'γράμμα'.
            real_boxes = sum(context.count(f"\\boxed{{{label}}}") for label in helpers.LABELS)
            self.assertEqual(real_boxes, shots)
            request = task.construct_requests(
                doc, context, metadata=("greekmmlu_mathematics", 0, 1)
            )
            self.assertEqual(request.request_type, "generate_until")
            self.assertEqual(request.args[1]["max_gen_toks"], 32768)
            raw = "Αρχικά \\boxed{Α}, έπειτα ελέγχω.\n" + helpers.doc_to_fewshot_target(doc)
            request.resps = [raw]
            task._filters[0].apply([request])
            answer = request.filtered_resps["boxed-extract"]
            self.assertEqual(answer, task.doc_to_target(doc))
            self.assertEqual(task.process_results(doc, [answer])["exact_match"], 1.0)
            self.assertEqual(request.resps, [raw])


if __name__ == "__main__":
    unittest.main(verbosity=2)
