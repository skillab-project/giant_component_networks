# analysis.py
from collections import defaultdict
from itertools import combinations
import networkx as nx


# This will be injected / monkeypatched in tests or set by API
ESCO_LABELS = {}


# ----------------------------
# Helpers
# ----------------------------

def labelize(skill_uri: str) -> str:
    """Convert ESCO skill URI → human-readable label."""
    return ESCO_LABELS.get(skill_uri, skill_uri)


# ----------------------------
# Core analysis
# ----------------------------

def build_cooccurrence(skills_per_doc, valid_skills):
    """
    Build weighted co-occurrence edges between skills.
    """
    co_counts = defaultdict(int)
    skill_counts = defaultdict(int)

    for skills in skills_per_doc:
        filtered = [s for s in skills if s in valid_skills]

        for s in set(filtered):
            skill_counts[s] += 1

        for s1, s2 in combinations(sorted(set(filtered)), 2):
            co_counts[(s1, s2)] += 1

    edges = []
    for (s1, s2), cij in co_counts.items():
        ci, cj = skill_counts[s1], skill_counts[s2]
        if ci == 0 or cj == 0:
            continue

        eij = (cij ** 2) / (ci * cj)
        edges.append({
            "source": labelize(s1),
            "target": labelize(s2),
            "value": round(eij, 4),
            "raw_count": cij
        })

    return edges


def process_documents(documents):
    """
    Build a co-occurrence network and extract the giant component.
    """
    skills_per_doc = [d["skills"] for d in documents if d.get("skills")]

    if not skills_per_doc:
        return None, {}, [], 0

    all_skills = set(sum(skills_per_doc, []))
    valid_skills = all_skills & set(ESCO_LABELS.keys())

    if not valid_skills:
        return None, {}, [], len(skills_per_doc)

    edges = build_cooccurrence(skills_per_doc, valid_skills)
    if not edges:
        return None, {}, [], len(skills_per_doc)

    # Build graph
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"], value=e["value"])

    if G.number_of_nodes() == 0:
        return None, {}, edges, len(skills_per_doc)

    GC = max(nx.connected_components(G), key=len)
    subG = G.subgraph(GC).copy()

    nodes = list(subG.nodes())
    edges_gc = [
        {"source": u, "target": v, "value": d["value"]}
        for u, v, d in subG.edges(data=True)
    ]

    return (nodes, edges_gc), {}, edges, len(skills_per_doc)


def process_documents_with_limits(documents, max_edges=100, max_nodes=200):
    """
    Same as process_documents, but limits graph size.
    """
    skills_per_doc = [d["skills"] for d in documents if d.get("skills")]

    if not skills_per_doc:
        return None, {}, [], 0

    all_skills = set(sum(skills_per_doc, []))
    valid_skills = all_skills & set(ESCO_LABELS.keys())

    if not valid_skills:
        return None, {}, [], len(skills_per_doc)

    co_counts = defaultdict(int)
    skill_counts = defaultdict(int)

    for skills in skills_per_doc:
        filtered = [s for s in skills if s in valid_skills]

        for s in set(filtered):
            skill_counts[s] += 1

        for s1, s2 in combinations(sorted(set(filtered)), 2):
            co_counts[(s1, s2)] += 1

    edges = []
    for (s1, s2), cij in co_counts.items():
        ci, cj = skill_counts[s1], skill_counts[s2]
        if ci == 0 or cj == 0:
            continue

        eij = (cij ** 2) / (ci * cj)
        edges.append({
            "source": labelize(s1),
            "target": labelize(s2),
            "value": round(eij, 4),
            "raw_count": cij
        })

    if not edges:
        return None, {}, [], len(skills_per_doc)

    # Limit edges
    edges = sorted(edges, key=lambda x: x["value"], reverse=True)[:max_edges]

    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"], value=e["value"])

    # Limit nodes
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = {n for n, _ in top_nodes}
        G = G.subgraph(keep).copy()

    if G.number_of_nodes() == 0:
        return None, {}, edges, len(skills_per_doc)

    GC = max(nx.connected_components(G), key=len)
    subG = G.subgraph(GC).copy()

    nodes = list(subG.nodes())
    edges_gc = [
        {"source": u, "target": v, "value": d["value"]}
        for u, v, d in subG.edges(data=True)
    ]

    return (nodes, edges_gc), {}, edges, len(skills_per_doc)
