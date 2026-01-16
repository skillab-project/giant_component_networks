import unittest
from fastapi.testclient import TestClient

# IMPORTANT:
# change "main" to the filename where your FastAPI app lives
# e.g. if file is api.py → from api import app
from gianti_ending import app

client = TestClient(app)


class TestGiantAPI(unittest.TestCase):


    # -------------------------------
    # BASIC HEALTH CHECK
    # -------------------------------
    def test_jobs_endpoint_exists(self):
        response = client.get(
            "/api/jobs_mapped_ultra",
            params={"keywords": "ai"}
        )
        self.assertIn(response.status_code, [200, 422])

    # -------------------------------
    # KU CO-OCCURRENCE ENDPOINT
    # -------------------------------
    def test_ku_cooccurrence_endpoint(self):
        response = client.get(
            "/ku-co-occurrence",
            params={
                "start_date": "2024-01",
                "end_date": "2024-12",
                "max_edges": 10,
                "max_nodes": 10
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), dict)

    # -------------------------------
    # JOBS NETWORK ENDPOINT
    # -------------------------------
    def test_jobs_mapped_ultra(self):
        response = client.get(
            "/api/jobs_mapped_ultra",
            params={
                "keywords": "data,software",
                "max_pages": 1,
                "max_edges": 10,
                "max_nodes": 10
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    # -------------------------------
    # LAW POLICIES ENDPOINT
    # -------------------------------
    def test_law_policies_mapped(self):
        response = client.get(
            "/api/law-policies_mapped",
            params={
                "keywords": "data,ai",
                "max_edges": 10,
                "max_nodes": 10
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("summary", response.json())


if __name__ == "__main__":
    unittest.main()
