import unittest
import os
import json
from backend.ai.dpo_pipeline import DPOPipeline

class TestDPOThreshold(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_dpo_dataset.jsonl"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        self.pipeline = DPOPipeline(dataset_path=self.test_db)

    def test_collapse_lenient_success(self):
        # Even if score is low (0.1 and 0.05), it should pick them if there is signal
        results = [
            {"rule": "best_rule", "score": 0.1, "is_success": False},
            {"rule": "worst_rule", "score": 0.05, "is_success": False}
        ]
        prompt = "Move right"
        entry = self.pipeline.collapse(results, prompt)
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry["chosen"], "best_rule")
        self.assertEqual(entry["rejected"], "worst_rule")
        
        # Verify file write
        with open(self.test_db, "r") as f:
            data = json.loads(f.readline())
            self.assertEqual(data["chosen"], "best_rule")

    def test_collapse_no_signal(self):
        # If scores are too close (0.1 and 0.099), it should skip
        results = [
            {"rule": "rule1", "score": 0.1, "is_success": False},
            {"rule": "rule2", "score": 0.099, "is_success": False}
        ]
        entry = self.pipeline.collapse(results, "Move right")
        self.assertIsNone(entry)

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

if __name__ == "__main__":
    unittest.main()
