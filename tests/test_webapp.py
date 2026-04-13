import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from webapp.app import app


class WebAppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint_reports_ok_and_model_path(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("fixture_demo_best.pt", payload["model_path"])

    def test_index_page_renders_supported_scope_and_samples(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("Analyze clean logic schematics in stable and beta modes.", body)
        self.assertIn("Half Adder", body)
        self.assertIn("Half Subtractor", body)
        self.assertIn("Full Adder", body)
        self.assertIn("Symbol Beta", body)
        self.assertIn("Try Supported Symbol Beta Cases", body)
        self.assertIn("Full Adder TP", body)

    def test_favicon_route_returns_no_content(self) -> None:
        response = self.client.get("/favicon.ico")

        self.assertEqual(response.status_code, 204)

    def test_sample_analysis_route_renders_mocked_result(self) -> None:
        mocked_payload = {
            "result": {
                "filename": "half_adder.png",
                "mode": "fixture",
                "classification": {
                    "label": "half_adder",
                    "confidence": 0.5,
                    "reasoning": "mocked reasoning",
                },
                "gates": [
                    {
                        "gate_id": "gate_0",
                        "gate_type": "XOR",
                        "confidence": 0.9,
                        "bbox": [0, 0, 10, 10],
                    }
                ],
                "warnings": [],
                "expressions": {"OUT0": "(IN0 XOR IN1)"},
                "truth_table": [{"IN0": 0, "IN1": 0, "OUT0": 0}],
                "analysis_image_url": "data:image/png;base64,ZmFrZQ==",
                "debug_image_url": "data:image/png;base64,ZmFrZQ==",
            }
        }

        with patch("webapp.app._run_analysis", return_value=mocked_payload):
            response = self.client.post("/analyze/sample", data={"sample": "half_adder"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("mocked reasoning", response.text)
        self.assertIn("half_adder.png", response.text)

    def test_symbol_sample_analysis_route_uses_symbol_beta_mode(self) -> None:
        mocked_payload = {
            "result": {
                "filename": "full_adder_tp.jpg",
                "mode": "symbol_beta",
                "classification": {
                    "label": "full_adder",
                    "confidence": 0.99,
                    "reasoning": "supported symbol beta sample",
                },
                "gates": [],
                "selected_gate_ids": ["proposal_23", "proposal_30"],
                "warnings": [],
                "expressions": {"SUM": "(A XOR B XOR CIN)"},
                "truth_table": [{"A": 0, "B": 0, "CIN": 0, "SUM": 0, "COUT": 0}],
                "analysis_image_url": "data:image/png;base64,ZmFrZQ==",
                "debug_image_url": "data:image/png;base64,ZmFrZQ==",
            }
        }

        with patch("webapp.app._run_analysis", return_value=mocked_payload) as mocked_run:
            response = self.client.post(
                "/analyze/sample",
                data={"sample": "full_adder_tp", "mode": "symbol_beta"},
            )

        self.assertEqual(response.status_code, 200)
        mocked_run.assert_called_once()
        self.assertEqual(mocked_run.call_args.kwargs["mode"], "symbol_beta")
        self.assertIn("supported symbol beta sample", response.text)
        self.assertIn("full_adder_tp.jpg", response.text)

    def test_api_analyze_rejects_unsupported_extension(self) -> None:
        response = self.client.post(
            "/api/analyze",
            files={"image": ("bad.txt", b"not-an-image", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["error"])

    def test_api_symbol_beta_rejects_unsupported_extension(self) -> None:
        response = self.client.post(
            "/api/analyze/symbol-beta",
            files={"image": ("bad.txt", b"not-an-image", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
