import pytest
import os
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from giant_component_networks import app
import requests

load_dotenv()
client = TestClient(app)

TRACKER_CREDS = os.getenv("TRACKER_USERNAME") and os.getenv("TRACKER_PASSWORD")

def is_ku_service_online(url):
    """Checks if the KU service is reachable."""
    if not url:
        return False
    try:
        response = requests.get(url, timeout=3)
        return response.status_code < 500
    except requests.RequestException:
        return False

KU_API_URL = os.getenv("KU_API_URL")
KU_ONLINE = is_ku_service_online(KU_API_URL)
print(f"KU Service Online: {KU_ONLINE} at {KU_API_URL}")


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


@pytest.mark.skipif(not KU_API_URL or not KU_ONLINE, reason=f"KU Service is offline or URL missing at {KU_API_URL}")
class TestKUNetworkIntegration:
    def test_ku_cooccurrence_network(self):
        """Test network generation from Knowledge Units."""
        response = client.get("/ku-co-occurrence")
        # We check for 200 OK. If no data exists, the API returns a 'message' key.
        assert response.status_code == 200
        data = response.json()
        
        # We accept EITHER a full network OR a 'no data' message/error
        valid_keys = ["giant_component", "message", "error"]
        assert any(key in data for key in valid_keys), f"Unexpected response keys: {data.keys()}"

    def test_ku_forecasting_logic(self):
        """Test the link prediction for KUs."""
        response = client.get(
            "/ku-link-prediction",
            params={"method": "adamic_adar", "top_k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        
        # If there is data, check for links. If not, check for the error/message.
        if "predicted_links" in data:
            assert "summary" in data
            assert isinstance(data["predicted_links"], list)
        else:
            # If no data yet, ensure the API explained why 
            assert any(key in data for key in ["error", "message"]), "API returned 200 but no links or error message."