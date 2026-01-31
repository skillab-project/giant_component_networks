import pytest
import os
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from giant_component_networks import app

load_dotenv()
client = TestClient(app)

TRACKER_CREDS = os.getenv("TRACKER_USERNAME") and os.getenv("TRACKER_PASSWORD")
KU_API_URL = os.getenv("KU_API_URL")

@pytest.mark.skipif(not TRACKER_CREDS, reason="Tracker credentials missing")
class TestTrackerNetworkIntegration:
    def test_esco_cache_initialization(self):
        """Verify that the ESCO skill cache was generated on startup."""
        # This is triggered by the import of the app
        assert os.path.exists("esco_skills_cache.json"), "ESCO cache file was not created on app startup."

    def test_jobs_network_generation(self):
        """Test co-occurrence network generation from Job postings."""
        response = client.get(
            "/api/jobs_mapped_ultra",
            params={"keywords": "software", "max_pages": 1, "max_edges": 30}
        )
        assert response.status_code == 200
        data = response.json()
        assert "giant_component" in data
        assert len(data["giant_component"]["nodes"]) > 0

    def test_profiles_network_generation(self):
        """Test co-occurrence network generation from Profiles."""
        response = client.get(
            "/api/profiles_mapped",
            params={"keywords": "software", "max_pages": 1}
        )
        assert response.status_code == 200
        assert "giant_component" in response.json()

    def test_law_policies_network_generation(self):
        """Test co-occurrence network generation from Law/Policies."""
        response = client.get(
            "/api/law-policies_mapped",
            params={"keywords": "software", "source": "eur_lex"}
        )
        assert response.status_code == 200
        # The network might be empty depending on the keywords, 
        # but the API response structure must be correct.
        assert "summary" in response.json()


@pytest.mark.skipif(not KU_API_URL, reason="KU_API_URL not configured in environment")
class TestKUNetworkIntegration:
    def test_ku_cooccurrence_network(self):
        """Test network generation from Knowledge Units."""
        response = client.get("/ku-co-occurrence")
        # We check for 200 OK. If no data exists, the API returns a 'message' key.
        assert response.status_code == 200
        data = response.json()
        assert any(key in data for key in ["giant_component", "message"])

    def test_ku_forecasting_logic(self):
        """Test the link prediction (Adamic-Adar) for KUs."""
        response = client.get(
            "/ku-link-prediction",
            params={"method": "adamic_adar", "top_k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_links" in data
        assert "summary" in data