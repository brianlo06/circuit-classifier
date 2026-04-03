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
        self.assertIn("Upload a clean fixture-style logic schematic.", body)
        self.assertIn("Half Adder", body)
        self.assertIn("Half Subtractor", body)
        self.assertIn("Full Adder", body)
        self.assertIn("Symbol-style inputs are planned", body)

    def test_favicon_route_returns_no_content(self) -> None:
        response = self.client.get("/favicon.ico")

        self.assertEqual(response.status_code, 204)

    def test_sample_analysis_route_renders_mocked_result(self) -> None:
        mocked_payload = {
            "result": {
                "filename": "half_adder.png",
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

    def test_api_analyze_rejects_unsupported_extension(self) -> None:
        response = self.client.post(
            "/api/analyze",
            files={"image": ("bad.txt", b"not-an-image", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
