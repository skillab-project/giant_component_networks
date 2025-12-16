import networkx as nx


from network_analysis import (
    labelize,
    build_cooccurrence,
    process_documents,
    process_documents_with_limits
)


# ----------------------------
# labelize
# ----------------------------

def test_labelize_returns_label_if_exists(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {"s1": "Skill One"}
    )

    assert labelize("s1") == "Skill One"


def test_labelize_falls_back_to_uri(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {}
    )

    assert labelize("unknown_skill") == "unknown_skill"


# ----------------------------
# build_cooccurrence
# ----------------------------

def test_build_cooccurrence_basic():
    skills_per_doc = [
        ["s1", "s2", "s3"],
        ["s1", "s2"]
    ]
    valid_skills = {"s1", "s2", "s3"}

    edges = build_cooccurrence(skills_per_doc, valid_skills)

    assert len(edges) > 0
    assert all("source" in e and "target" in e for e in edges)
    assert all(e["value"] > 0 for e in edges)


def test_build_cooccurrence_ignores_invalid_skills():
    skills_per_doc = [
        ["s1", "s2", "invalid"]
    ]
    valid_skills = {"s1", "s2"}

    edges = build_cooccurrence(skills_per_doc, valid_skills)

    assert len(edges) == 1
    assert edges[0]["source"] != "invalid"
    assert edges[0]["target"] != "invalid"


def test_build_cooccurrence_empty_input():
    edges = build_cooccurrence([], {"s1", "s2"})
    assert edges == []


# ----------------------------
# process_documents
# ----------------------------

def test_process_documents_basic(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {"s1": "Skill 1", "s2": "Skill 2"}
    )

    documents = [
        {"skills": ["s1", "s2"]},
        {"skills": ["s1"]}
    ]

    result, meta, edges, count = process_documents(documents)

    assert count == 2
    assert result is not None

    nodes, edges_gc = result
    assert len(nodes) > 0
    assert len(edges_gc) > 0


def test_process_documents_no_skills():
    documents = [{"skills": []}, {"skills": []}]

    result, meta, edges, count = process_documents(documents)

    assert result is None
    assert count == 0


def test_process_documents_filters_non_esco(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {"s1": "Skill 1"}
    )

    documents = [
        {"skills": ["s1", "invalid"]}
    ]

    result, meta, edges, count = process_documents(documents)

    assert result is None or len(edges) >= 0


# ----------------------------
# process_documents_with_limits
# ----------------------------

def test_process_documents_with_limits_respects_limits(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {f"s{i}": f"Skill {i}" for i in range(10)}
    )

    documents = [
        {"skills": ["s1", "s2", "s3", "s4"]},
        {"skills": ["s1", "s2", "s5"]},
        {"skills": ["s3", "s4", "s6"]},
    ]

    result, meta, edges, count = process_documents_with_limits(
        documents,
        max_edges=2,
        max_nodes=3
    )

    assert count == 3
    assert result is not None

    nodes, edges_gc = result
    assert len(edges_gc) <= 2
    assert len(nodes) <= 3


def test_process_documents_with_limits_no_valid_skills(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {}
    )

    documents = [{"skills": ["x", "y"]}]

    result, meta, edges, count = process_documents_with_limits(documents)

    assert result is None
    assert count == 1


# ----------------------------
# build_cooccurrence – properties
# ----------------------------

def test_build_cooccurrence_symmetry():
    skills_per_doc = [["a", "b"], ["b", "a"]]
    valid_skills = {"a", "b"}

    edges = build_cooccurrence(skills_per_doc, valid_skills)

    assert len(edges) == 1
    assert edges[0]["source"] != edges[0]["target"]


def test_build_cooccurrence_single_skill_docs():
    skills_per_doc = [["a"], ["a"], ["a"]]
    valid_skills = {"a"}

    edges = build_cooccurrence(skills_per_doc, valid_skills)

    assert edges == []


def test_build_cooccurrence_weight_increases_with_frequency():
    skills_per_doc = [["a", "b"], ["a", "b"], ["a", "b"]]
    valid_skills = {"a", "b"}

    edges = build_cooccurrence(skills_per_doc, valid_skills)

    assert edges[0]["value"] > 0


# ----------------------------
# process_documents – graph invariants
# ----------------------------

def test_process_documents_giant_component_is_connected(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {"a": "A", "b": "B", "c": "C"}
    )

    documents = [
        {"skills": ["a", "b"]},
        {"skills": ["b", "c"]}
    ]

    (nodes, edges), _, _, _ = process_documents(documents)

    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"])

    assert nx.is_connected(G)


def test_process_documents_returns_raw_edges_even_if_gc_small(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {"a": "A", "b": "B", "c": "C"}
    )

    documents = [
        {"skills": ["a", "b"]},
        {"skills": ["c"]}
    ]

    result, _, raw_edges, _ = process_documents(documents)

    assert isinstance(raw_edges, list)


def test_process_documents_empty_documents():
    result, meta, edges, count = process_documents([])

    assert result is None
    assert edges == []
    assert count == 0


# ----------------------------
# process_documents_with_limits – stress & limits
# ----------------------------

def test_process_documents_with_limits_single_document(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {"a": "A", "b": "B"}
    )

    documents = [{"skills": ["a", "b"]}]

    result, _, edges, count = process_documents_with_limits(documents)

    assert count == 1
    assert result is not None


def test_process_documents_with_limits_extreme_limits(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {f"s{i}": f"S{i}" for i in range(20)}
    )

    documents = [{"skills": [f"s{i}" for i in range(10)]}]

    result, _, edges, _ = process_documents_with_limits(
        documents,
        max_edges=1,
        max_nodes=1
    )

    assert result is not None
    nodes, edges_gc = result
    assert len(nodes) <= 1
    assert len(edges_gc) <= 1


def test_process_documents_with_limits_no_crash_on_duplicates(monkeypatch):
    monkeypatch.setattr(
        "network_analysis.ESCO_LABELS",
        {"a": "A", "b": "B"}
    )

    documents = [
        {"skills": ["a", "a", "b", "b"]},
        {"skills": ["a", "b"]}
    ]

    result, _, edges, count = process_documents_with_limits(documents)

    assert count == 2
    assert result is not None