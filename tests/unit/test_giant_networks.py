import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import networkx as nx

# 1. Import the module
import giant_component_networks

# 2. Directly inject mock data into the module's global variables
# This ensures that 'labelize' and intersection logic ('valid_skills') work correctly.
mock_labels = {
    "uri:1": "Python",
    "uri:2": "FastAPI",
    "uri:3": "Docker",
    "uri:4": "Cloud",
    "uri:5": "Linux"
}
giant_component_networks.ESCO_LABELS = mock_labels
giant_component_networks.PILLARS = {
    "skill": {"uri:1", "uri:2"}, 
    "knowledge": {"uri:3", "uri:4", "uri:5"}, 
    "transversal": set()
}

# Now we can safely import app and functions
from giant_component_networks import app, build_cooccurrence, process_documents_with_limits

client = TestClient(app)

# ==========================================
# 1. UNIT TESTS FOR LOGIC
# ==========================================

def test_build_cooccurrence_logic():
    """Verify the Eij calculation: (cij^2) / (ci * cj)"""
    skills_per_doc = [
        ["uri:1", "uri:2"], # Doc 1
        ["uri:1", "uri:2"], # Doc 2
        ["uri:1"]           # Doc 3
    ]
    # Use the keys from our mock
    valid_skills = {"uri:1", "uri:2"}
    
    # Calculation:
    # ci (uri:1) = 3
    # cj (uri:2) = 2
    # cij (1,2)  = 2
    # Expected weight = (2^2) / (3 * 2) = 4 / 6 = 0.6667
    
    edges = build_cooccurrence(skills_per_doc, valid_skills)
    
    assert len(edges) == 1
    assert edges[0]["source"] == "Python"
    assert edges[0]["target"] == "FastAPI"
    assert edges[0]["value"] == 0.6667

def test_process_documents_giant_component():
    """Test that it correctly extracts the largest connected component."""
    # Island 1: 1-2
    # Island 2: 3-4, 4-5 (Connects 3-4-5) -> This is the Giant Component (3 nodes)
    documents = [
        {"skills": ["uri:1", "uri:2"]},
        {"skills": ["uri:3", "uri:4"]},
        {"skills": ["uri:4", "uri:5"]},
    ]
    
    # This now works because uri:4 and uri:5 are in giant_component_networks.ESCO_LABELS
    gc_result, _, _, _ = process_documents_with_limits(documents)
    
    assert gc_result is not None, "GC result should not be None"
    nodes, edges = gc_result
    
    # Island 2 should be the giant component (3 nodes: Docker, Cloud, Linux)
    assert len(nodes) == 3 
    assert "Docker" in nodes
    assert "Cloud" in nodes
    assert "Linux" in nodes

# ==========================================
# 2. ENDPOINT INTEGRATION TESTS
# ==========================================

@patch("requests.post")
def test_law_policies_endpoint(mock_post):
    """Test the Law/Policies network endpoint with mocked API."""
    
    # Mock Login
    mock_login = MagicMock()
    mock_login.text = '"fake_token"'
    mock_login.status_code = 200
    
    # Mock Policy Response
    mock_policy = MagicMock()
    mock_policy.json.return_value = {
        "items": [
            {"skills": ["uri:1", "uri:2"]},
            {"skills": ["uri:1", "uri:2"]}
        ]
    }
    mock_policy.status_code = 200
    
    # Provide responses for the login and the paginated data fetch
    mock_post.side_effect = [mock_login, mock_policy]

    response = client.get("/api/law-policies_mapped?keywords=ai")
    
    assert response.status_code == 200
    data = response.json()
    assert "giant_component" in data
    assert len(data["giant_component"]["nodes"]) == 2
    # Verify label mapping worked in the API response
    node_ids = [n["id"] for n in data["giant_component"]["nodes"]]
    assert "Python" in node_ids
    assert "FastAPI" in node_ids

@patch("requests.get")
def test_ku_cooccurrence_endpoint(mock_get):
    """Test the KU co-occurrence endpoint."""
    
    mock_res = MagicMock()
    mock_res.json.return_value = [
        {"detected_kus": {"K1": "1", "K2": "1"}, "organization": "OrgA"},
        {"detected_kus": {"K1": "1", "K2": "1"}, "organization": "OrgA"}
    ]
    mock_res.status_code = 200
    mock_get.return_value = mock_res

    response = client.get("/ku-co-occurrence?organization=OrgA")
    
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["Giant Component Nodes"] == 2
    assert data["giant_component"]["edges"][0]["source"] == "K1"

@patch("requests.get")
def test_ku_link_prediction_endpoint(mock_get):
    """Test the Link Prediction endpoint logic."""
    
    # K1-K2, K1-K3, K2-K3. A small complete triangle.
    mock_res = MagicMock()
    mock_res.json.return_value = [
        {"detected_kus": {"K1": "1", "K2": "1"}},
        {"detected_kus": {"K1": "1", "K3": "1"}},
        {"detected_kus": {"K2": "1", "K3": "1"}}
    ]
    mock_res.status_code = 200
    mock_get.return_value = mock_res

    response = client.get("/ku-link-prediction?method=jaccard")
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_links" in data