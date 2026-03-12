from fastapi import FastAPI, Query
from typing import List, Optional
import pandas as pd
from collections import defaultdict
from itertools import combinations
import networkx as nx
import json
import math
import re
from pathlib import Path
import requests
import os
import time
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

API      = os.getenv("TRACKER_API")
USERNAME = os.getenv("TRACKER_USERNAME")
PASSWORD = os.getenv("TRACKER_PASSWORD")

app = FastAPI(
    title="Giant Component - Skill Network API",
    root_path="/giant-component-networks"
)


# === 1️⃣ Authentication Helper ===
def get_token():
    print(f"🔐 Authenticating with {API} ...")
    try:
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=10)
        res.raise_for_status()
        token = res.text.replace('"', "")
        print("✅ Authenticated successfully.")
        return token
    except Exception as e:
        print(f"❌ Login failed: {e}")
        raise


# === 2️⃣ Fetch ALL ESCO Skills from Tracker (all sources, auto-paginated) ===
def fetch_all_skills_from_tracker():
    """
    Fetch every ESCO skill from the Tracker API — no keyword/source filter,
    all pages auto-paginated using the `count` field from the first response.
    Reads credentials exclusively from environment variables (.env).
    """
    api_url  = os.getenv("TRACKER_API")
    username = os.getenv("TRACKER_USERNAME")
    password = os.getenv("TRACKER_PASSWORD")

    page_size   = 100
    max_retries = 3
    backoff     = 10
    timeout     = 180

    print(f"🔐 Authenticating to {api_url} ...")
    res = requests.post(f"{api_url}/login", json={"username": username, "password": password}, timeout=15)
    res.raise_for_status()
    token   = res.text.replace('"', "")
    headers = {"Authorization": f"Bearer {token}"}
    print(f"✅ Authenticated as {username}")

    # ── helper: fetch one page with retry ──────────────────────────────────
    def fetch_page(page_num: int) -> dict:
        url = f"{api_url}/skills?page={page_num}&page_size={page_size}"
        for attempt in range(1, max_retries + 1):
            try:
                print(f"   ↪ Attempt {attempt}/{max_retries} — page {page_num} (timeout={timeout}s)...")
                r = requests.post(url, headers=headers, timeout=timeout)
                if r.status_code != 200:
                    print(f"   ⚠️ HTTP {r.status_code} on page {page_num}: {r.text[:200]}")
                    return {}
                return r.json()
            except requests.exceptions.ReadTimeout:
                print(f"   ⏱️ Timeout on page {page_num}, attempt {attempt}/{max_retries}.")
                if attempt < max_retries:
                    print(f"   🔄 Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    print(f"   ❌ All {max_retries} attempts exhausted — skipping page {page_num}.")
                    return {}
            except Exception as ex:
                print(f"   ❌ {type(ex).__name__}: {ex}")
                return {}

    # ── probe page 1 → read total count ────────────────────────────────────
    print("🔍 Probing page 1 to determine total skill count...")
    probe = fetch_page(1)
    if not probe:
        raise RuntimeError("❌ Probe request (page 1) failed. Cannot load ESCO skills.")

    total_count = probe.get("count", 0)
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
    print(f"📊 Total ESCO skills available: {total_count} → {total_pages} page(s)")

    all_items = list(probe.get("items", []))
    seen_ids  = {i.get("id") for i in all_items}
    print(f"📦 Page 1/{total_pages}: {len(all_items)} skills")

    # ── fetch remaining pages ───────────────────────────────────────────────
    for page in range(2, total_pages + 1):
        print(f"📄 Fetching page {page}/{total_pages}...")
        data  = fetch_page(page)
        items = data.get("items", []) if data else []

        if not items:
            print(f"✅ No items on page {page} — stopping.")
            break

        new_items = [i for i in items if i.get("id") not in seen_ids]
        seen_ids.update(i.get("id") for i in new_items)
        all_items.extend(new_items)
        print(f"📦 Page {page}/{total_pages}: {len(new_items)} new skills (running total: {len(all_items)})")

        if len(items) < page_size:
            print("✅ Last page reached.")
            break

    print(f"\n🎯 Total unique ESCO skills fetched: {len(all_items)} / {total_count}")

    # ── build mapping & pillar sets ────────────────────────────────────────
    mapping = {s["id"]: s.get("label", s["id"]) for s in all_items}
    pillars = {"skill": set(), "knowledge": set(), "transversal": set()}
    for s in all_items:
        sid = s["id"]
        if s.get("skill_levels"):
            pillars["skill"].add(sid)
        if s.get("knowledge_levels"):
            pillars["knowledge"].add(sid)
        if s.get("traversal_levels"):
            pillars["transversal"].add(sid)

    print("📊 Pillar distribution:")
    for p, v in pillars.items():
        print(f"  {p}: {len(v)} items")

    return mapping, pillars


# === 3️⃣ Cached loading system ===
CACHE_FILE = Path("esco_skills_cache.json")

def load_or_fetch_skills():
    if CACHE_FILE.exists():
        print("📦 Loading ESCO skills from local cache...")
        data = json.loads(CACHE_FILE.read_text())
        mapping = data["mapping"]
        pillars = {
            "skill":       set(data["pillars"]["skill"]),
            "knowledge":   set(data["pillars"]["knowledge"]),
            "transversal": set(data["pillars"]["transversal"]),
        }
        print(f"✅ Cache loaded: {len(mapping)} skills.")
        return mapping, pillars

    print("🌐 No cache found — fetching from API (this may take a while)...")
    mapping, pillars = fetch_all_skills_from_tracker()
    CACHE_FILE.write_text(json.dumps({
        "mapping": mapping,
        "pillars": {k: list(v) for k, v in pillars.items()}
    }, indent=2))
    print(f"✅ Cached {len(mapping)} skills → {CACHE_FILE}")
    return mapping, pillars


ESCO_LABELS, PILLARS = load_or_fetch_skills()
print(f"🎯 Total cached ESCO skills: {len(ESCO_LABELS)}")


# === Helper Functions ===
def labelize(skill_uri):
    return ESCO_LABELS.get(skill_uri, skill_uri)

def get_total_jobs_in_tracker():
    try:
        print("🔐 Authenticating to fetch total job count...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authentication successful.")
        print("📊 Fetching total job count from Tracker...")
        res = requests.post(f"{API}/jobs", headers=headers, json={}, timeout=60)
        res.raise_for_status()
        data = res.json()
        total_jobs = data.get("count", 0)
        print(f"📦 Total jobs available in Tracker: {total_jobs}")
        return total_jobs
    except Exception as e:
        print(f"❌ Failed to fetch total job count: {e}")
        return 0


def fetch_all_items(endpoint: str, payload_base, max_pages=50, page_size=100):
    res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=10)
    res.raise_for_status()
    token = res.text.replace('"', "")
    headers = {"Authorization": f"Bearer {token}"}

    print(f"🔑 Authenticated → querying filtered {endpoint}")
    print(f"📦 Payload:\n{json.dumps(payload_base, indent=2)}")

    all_items, seen_ids = [], set()

    for page in range(1, max_pages + 1):
        try:
            res = requests.post(
                f"{API}/{endpoint}",
                headers=headers,
                params={"page": page, "page_size": page_size},
                json=payload_base,
                timeout=60
            )
            if res.status_code != 200:
                print(f"⚠️ Error {res.status_code}: {res.text[:200]}")
                break
            data = res.json()
            items = data.get("items", [])
            print(f"📄 Page {page}: {len(items)} items")
            if not items:
                break
            new_items = [i for i in items if i.get("id") not in seen_ids]
            seen_ids.update(i.get("id") for i in new_items)
            all_items.extend(new_items)
            if len(items) < page_size:
                break
        except Exception as e:
            print(f"❌ Error fetching page {page}: {e}")
            break

    print(f"🎯 Retrieved {len(all_items)} filtered items from {endpoint}")
    return {"items": all_items}


# === Network Creation ===
def build_cooccurrence(skills_per_doc, valid_skills):
    co_counts    = defaultdict(int)
    skill_counts = defaultdict(int)

    for skills in skills_per_doc:
        filtered = [s for s in skills if s in valid_skills]
        for skill in set(filtered):
            skill_counts[skill] += 1
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
    print(f"🧩 Processing {len(documents)} documents...")
    skills_per_doc = [doc["skills"] for doc in documents if doc.get("skills")]
    print(f"📊 Documents with skill lists: {len(skills_per_doc)}")
    if not skills_per_doc:
        return None, {}, [], 0

    all_mentioned_skills = set(sum(skills_per_doc, []))
    valid_skills = all_mentioned_skills & set(ESCO_LABELS.keys())
    print(f"✅ Found {len(valid_skills)} valid ESCO skills in these documents.")
    if not valid_skills:
        print("⚠️ No valid ESCO skills found in the current subset.")
        return None, {}, [], len(skills_per_doc)

    edges = build_cooccurrence(skills_per_doc, valid_skills)
    if not edges:
        print("⚠️ No co-occurrence edges built.")
        return None, {}, [], len(skills_per_doc)

    edges = sorted(edges, key=lambda x: x["value"], reverse=True)[:300]
    print(f"🔝 Retained top {len(edges)} strongest edges")

    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"], value=e["value"])
    if G.number_of_nodes() == 0:
        return None, {}, edges, len(skills_per_doc)

    GC   = max(nx.connected_components(G), key=len)
    subG = G.subgraph(GC).copy()
    nodes    = list(subG.nodes())
    edges_gc = [
        {"source": u, "target": v, "value": round(d["value"], 4)}
        for u, v, d in subG.edges(data=True)
    ]
    print(f"🕸️ Giant Component — Nodes: {len(nodes)}, Edges: {len(edges_gc)}")
    return (nodes, edges_gc), {}, edges, len(skills_per_doc)


def process_documents_with_limits(documents, max_edges=100, max_nodes=200):
    print(f"🧩 Processing {len(documents)} documents with limits: edges={max_edges}, nodes={max_nodes}")
    skills_per_doc = [doc["skills"] for doc in documents if doc.get("skills")]
    print(f"📊 Documents with skill lists: {len(skills_per_doc)}")
    if not skills_per_doc:
        return None, {}, [], 0

    all_mentioned_skills = set(sum(skills_per_doc, []))
    valid_skills = all_mentioned_skills & set(ESCO_LABELS.keys())
    if not valid_skills:
        print("⚠️ No valid ESCO skills found.")
        return None, {}, [], len(skills_per_doc)

    co_counts    = defaultdict(int)
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
            "source": ESCO_LABELS.get(s1, s1),
            "target": ESCO_LABELS.get(s2, s2),
            "value": round(eij, 4),
            "raw_count": cij
        })

    if not edges:
        return None, {}, [], len(skills_per_doc)

    edges = sorted(edges, key=lambda x: x["value"], reverse=True)[:max_edges]
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"], value=e["value"])

    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph({n for n, _ in top_nodes}).copy()

    if G.number_of_nodes() == 0:
        return None, {}, edges, len(skills_per_doc)

    GC   = max(nx.connected_components(G), key=len)
    subG = G.subgraph(GC).copy()
    nodes    = list(subG.nodes())
    edges_gc = [
        {"source": u, "target": v, "value": round(d["value"], 4)}
        for u, v, d in subG.edges(data=True)
    ]
    print(f"🕸️ Giant Component — Nodes: {len(nodes)}, Edges: {len(edges_gc)}")
    return (nodes, edges_gc), {}, edges, len(skills_per_doc)


# ============================================================
#  ENDPOINTS — all unchanged except .env cleanup
# ============================================================

@app.get("/ku-co-occurrence")
def ku_cooccurrence_network(
    start_date:   str = Query(None, description="Start date in YYYY-MM format"),
    end_date:     str = Query(None, description="End date in YYYY-MM format"),
    kus:          str = Query(None, description="Comma-separated list of KU IDs to include, e.g., K1,K5,K10"),
    organization: str = Query(None, description="Optional organization name to filter KU results by"),
    max_edges:    int = Query(100,  description="Maximum number of top edges to retain"),
    max_nodes:    int = Query(200,  description="Maximum number of nodes in the network")
):
    """Fetch KU analysis results and build a KU co-occurrence network."""
    BASE_URL = os.getenv("KU_API_URL", "").rstrip("/")
    api_url  = f"{BASE_URL}/analysis_results"

    params = {}
    if start_date:   params["start_date"]   = start_date
    if end_date:     params["end_date"]     = end_date
    if organization: params["organization"] = organization

    try:
        print(f"🔍 Fetching KU analysis data from: {api_url} with params {params}")
        response = requests.get(api_url, params=params, timeout=90)
        response.raise_for_status()
        ku_data = response.json()

        if not ku_data:
            return {"error": "No KU analysis data found for the given filters."}

        print(f"✅ Retrieved {len(ku_data)} KU records from SKILLAB portal")

        selected_kus = set(kus.split(",")) if kus else None
        ku_docs = []

        for record in ku_data:
            detected_kus = record.get("detected_kus", {})
            org       = record.get("organization", "Unknown")
            timestamp = record.get("timestamp", "")
            if organization and org.lower() != organization.lower():
                continue
            active_kus = [ku for ku, val in detected_kus.items() if str(val) == "1"]
            if selected_kus:
                active_kus = [ku for ku in active_kus if ku in selected_kus]
            if active_kus:
                ku_docs.append({"organization": org, "timestamp": timestamp, "kus": active_kus})

        print(f"📊 Documents with KU detections: {len(ku_docs)}")
        if not ku_docs:
            return {"message": "No KU detections found for the selected filters."}

        co_counts = defaultdict(int)
        ku_counts = defaultdict(int)
        for doc in ku_docs:
            kus_in_doc = doc["kus"]
            for ku in set(kus_in_doc):
                ku_counts[ku] += 1
            for ku1, ku2 in combinations(sorted(set(kus_in_doc)), 2):
                co_counts[(ku1, ku2)] += 1

        edges = []
        for (ku1, ku2), cij in co_counts.items():
            ci, cj = ku_counts[ku1], ku_counts[ku2]
            if ci == 0 or cj == 0:
                continue
            eij = (cij ** 2) / (ci * cj)
            edges.append({"source": ku1, "target": ku2, "value": round(eij, 4), "raw_count": cij})

        if not edges:
            return {"message": "No co-occurrence edges found among KUs."}

        edges = sorted(edges, key=lambda x: x["value"], reverse=True)[:max_edges]
        G = nx.Graph()
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], value=edge["value"])

        if G.number_of_nodes() > max_nodes:
            top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph({n for n, _ in top_nodes}).copy()

        if G.number_of_nodes() == 0:
            return {"message": "No network could be built."}

        GC   = max(nx.connected_components(G), key=len)
        subG = G.subgraph(GC).copy()
        nodes    = list(subG.nodes())
        edges_gc = [{"source": u, "target": v, "value": round(d["value"], 4)} for u, v, d in subG.edges(data=True)]
        print(f"🕸️ Giant Component — Nodes: {len(nodes)}, Edges: {len(edges_gc)}")

        return {
            "message": "✅ KU co-occurrence network successfully created.",
            "filters_used": {"start_date": start_date, "end_date": end_date, "kus": kus, "organization": organization},
            "summary": {
                "Total KU Records": len(ku_data),
                "Documents with KUs": len(ku_docs),
                "Unique KUs": len(ku_counts),
                "Giant Component Nodes": len(nodes),
                "Giant Component Edges": len(edges_gc)
            },
            "giant_component": {"nodes": nodes, "edges": edges_gc}
        }

    except Exception as e:
        return {"error": f"KU co-occurrence analysis failed: {str(e)}"}


@app.get("/api/law-policies_mapped")
def law_policies_mapped(
    keywords:  str = Query(...,       description="Comma-separated keywords (e.g. data,ai,green)"),
    source:    str = Query("eur_lex", description="Source of the policies (default: eur_lex)"),
    max_edges: int = Query(100,       description="Maximum number of top edges to retain"),
    max_nodes: int = Query(200,       description="Maximum number of nodes in the network")
):
    """Fetch filtered law/policy documents, extract skills, build a URI→label co-occurrence network."""
    try:
        print("🔐 Authenticating for law policy retrieval...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token   = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authenticated successfully.")

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {"keywords": keywords_list, "keywords_logic": "or", "sources": [source]}
        print(f"📡 Querying /law-policies with {len(keywords_list)} keywords: {keywords_list}")

        all_docs  = []
        max_pages = 50
        for page in range(1, max_pages + 1):
            try:
                url = f"{API}/law-policies?page={page}&page_size=100"
                print(f"📡 Fetching page {page} ...")
                res = requests.post(url, headers=headers, data=payload, timeout=90)
                if res.status_code != 200:
                    print(f"⚠️ Page {page}: HTTP {res.status_code} → {res.text[:200]}")
                    break
                data  = res.json()
                items = data.get("items", [])
                print(f"📄 Page {page}: Retrieved {len(items)} items")
                if not items:
                    print("✅ No more results, stopping.")
                    break
                all_docs.extend(items)
                if len(items) < 100:
                    print("✅ Last page reached (less than 100 results).")
                    break
            except Exception as e:
                print(f"❌ Error on page {page}: {e}")
                break

        print(f"🎯 Total documents retrieved: {len(all_docs)}")
        if not all_docs:
            return {"message": f"No policy documents found for {keywords_list} from {source}."}

        skills_per_doc = [doc.get("skills", []) for doc in all_docs if doc.get("skills")]
        print(f"📊 Documents containing skills: {len(skills_per_doc)}")
        if not skills_per_doc:
            return {"message": "⚠️ No skills found in the retrieved documents."}

        co_counts    = defaultdict(int)
        skill_counts = defaultdict(int)
        for skills in skills_per_doc:
            unique_skills = set(skills)
            for s in unique_skills:
                skill_counts[s] += 1
            for s1, s2 in combinations(sorted(unique_skills), 2):
                co_counts[(s1, s2)] += 1

        edges = []
        for (s1, s2), cij in co_counts.items():
            ci, cj = skill_counts[s1], skill_counts[s2]
            if ci and cj:
                eij = (cij ** 2) / (ci * cj)
                edges.append({"source": labelize(s1), "target": labelize(s2), "value": round(eij, 4)})

        if not edges:
            return {"message": "⚠️ No co-occurrence relationships found."}

        print(f"🕸️ Raw network: {len(co_counts)} edges among {len(skill_counts)} skills")
        max_edges = max(max_edges, 1000)
        max_nodes = max(max_nodes, 600)

        edges = sorted(edges, key=lambda x: x["value"], reverse=True)[:max_edges]
        G = nx.Graph()
        for e in edges:
            G.add_edge(e["source"], e["target"], value=e["value"])

        print(f"📈 Initial Graph — Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

        if len(G.nodes()) > max_nodes:
            node_strength = {n: sum(d.get("value", 1) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
            top_nodes     = sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G             = G.subgraph({n for n, _ in top_nodes}).copy()
            print(f"⚙️ Simplified: retained top {len(G.nodes())} nodes by weighted degree")

        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)
            print(f"🧹 Removed {len(isolates)} isolated nodes")

        if G.number_of_nodes() == 0:
            return {"message": "⚠️ No valid network could be built."}

        components = list(nx.connected_components(G))
        if len(components) > 1:
            GC   = max(components, key=len)
            subG = G.subgraph(GC).copy()
            print(f"🔗 Selected giant component with {len(subG.nodes())} nodes")
        else:
            subG = G

        print("📈 Computing network metrics...")
        try:
            degree_dict     = dict(subG.degree())
            weighted_degree = dict(subG.degree(weight="value"))
            betweenness     = nx.betweenness_centrality(subG, weight="value", normalized=True)
            closeness       = nx.closeness_centrality(subG)
            clustering      = nx.clustering(subG, weight="value")
            try:
                eigenvector = nx.eigenvector_centrality(subG, weight="value", max_iter=500)
            except nx.PowerIterationFailedConvergence:
                eigenvector = {n: 0 for n in subG.nodes()}
                print("⚠️ Eigenvector centrality did not converge — set to 0")
        except Exception as e:
            print(f"⚠️ Metric computation error: {e}")
            degree_dict = dict(subG.degree())
            weighted_degree = dict(subG.degree(weight="value"))
            betweenness = closeness = clustering = eigenvector = {n: 0 for n in subG.nodes()}

        nodes_out = [
            {
                "id": n,
                "degree": degree_dict.get(n, 0),
                "weighted_degree": round(weighted_degree.get(n, 0), 4),
                "betweenness": round(betweenness.get(n, 0), 5),
                "closeness": round(closeness.get(n, 0), 5),
                "clustering": round(clustering.get(n, 0), 5),
                "eigenvector": round(eigenvector.get(n, 0), 5)
            }
            for n in subG.nodes()
        ]
        edges_gc = [{"source": u, "target": v, "value": round(d["value"], 4)} for u, v, d in subG.edges(data=True)]

        print(f"🌐 Final network — Nodes: {len(nodes_out)}, Edges: {len(edges_gc)}")
        for n in nodes_out[:5]:
            print(f"  {n['id']} — degree={n['degree']}, betweenness={n['betweenness']:.4f}")

        return {
            "message": f"✅ Skill co-occurrence network built for {len(all_docs)} policies.",
            "filters_used": {"keywords": keywords_list, "source": source},
            "summary": {
                "Documents Retrieved": len(all_docs),
                "Documents with Skills": len(skills_per_doc),
                "Giant Component Nodes": len(nodes_out),
                "Giant Component Edges": len(edges_gc)
            },
            "giant_component": {"nodes": nodes_out, "edges": edges_gc}
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"error": str(e)}


@app.get("/api/courses_mapped")
def courses_mapped(
    keywords:  str = Query(...,        description="Comma-separated keywords (e.g. data, ai, green)"),
    source:    str = Query("coursera", description="Source of the courses (default: coursera)"),
    max_edges: int = Query(200,        description="Maximum number of edges to keep"),
    max_nodes: int = Query(200,        description="Maximum number of nodes to keep")
):
    """Fetch filtered courses, build a skill co-occurrence network with centrality metrics."""
    try:
        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token   = res.text.replace('"', "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        print("✅ Authenticated successfully.")

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        print(f"📡 Fetching courses matching keywords: {keywords_list}")

        all_courses = []
        for page in range(1, 51):
            form_data = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("sources", source)]
            for kw in keywords_list:
                form_data.append(("keywords", kw))

            url = f"{API}/courses?page={page}&page_size=100"
            print(f"📄 Fetching page {page}...")
            res = requests.post(url, headers=headers, data=form_data, timeout=90)

            if res.status_code != 200:
                print(f"⚠️ Page {page}: HTTP {res.status_code} → {res.text[:200]}")
                break

            data  = res.json()
            items = data.get("items", [])
            print(f"📦 Page {page}: Retrieved {len(items)} courses")

            if not items:
                print("✅ No more results — stopping.")
                break
            all_courses.extend(items)
            if len(items) < 100:
                print("✅ Last page reached.")
                break

        print(f"🎯 Total courses retrieved: {len(all_courses)}")
        if not all_courses:
            return {"message": f"No courses found for {keywords_list} from {source}."}

        skills_per_doc = [doc.get("skills", []) for doc in all_courses if doc.get("skills")]
        print(f"📊 Courses containing skills: {len(skills_per_doc)}")
        if not skills_per_doc:
            return {"message": "⚠️ No skills found in the retrieved courses."}

        co_counts    = defaultdict(int)
        skill_counts = defaultdict(int)
        for skills in skills_per_doc:
            unique_skills = set(skills)
            for s in unique_skills:
                skill_counts[s] += 1
            for s1, s2 in combinations(sorted(unique_skills), 2):
                co_counts[(s1, s2)] += 1

        edges = []
        for (s1, s2), cij in co_counts.items():
            ci, cj = skill_counts[s1], skill_counts[s2]
            if ci and cj:
                eij = (cij ** 2) / (ci * cj)
                edges.append({"source": labelize(s1), "target": labelize(s2), "value": round(eij, 4)})

        if not edges:
            return {"message": "⚠️ No co-occurrence relationships found."}

        edges = sorted(edges, key=lambda x: x["value"], reverse=True)[:max_edges]
        G = nx.Graph()
        for e in edges:
            G.add_edge(e["source"], e["target"], value=e["value"])

        if len(G.nodes()) > max_nodes:
            node_strength = {n: sum(d.get("value", 1) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
            top_nodes     = sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G             = G.subgraph({n for n, _ in top_nodes}).copy()
            print(f"⚙️ Simplified: retained top {len(G.nodes())} nodes")

        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)
            print(f"🧹 Removed {len(isolates)} isolated nodes")

        if G.number_of_nodes() == 0:
            return {"message": "⚠️ No valid network could be built."}

        print("📈 Computing centrality metrics...")
        try:
            degree_dict     = dict(G.degree())
            weighted_degree = dict(G.degree(weight="value"))
            betweenness     = nx.betweenness_centrality(G, weight="value", normalized=True)
            closeness       = nx.closeness_centrality(G)
            clustering      = nx.clustering(G, weight="value")
            try:
                eigenvector = nx.eigenvector_centrality(G, weight="value", max_iter=500)
            except nx.PowerIterationFailedConvergence:
                eigenvector = {n: 0 for n in G.nodes()}
                print("⚠️ Eigenvector centrality did not converge.")
        except Exception as e:
            print(f"⚠️ Metric computation error: {e}")
            degree_dict = weighted_degree = betweenness = closeness = clustering = eigenvector = {}

        nodes_out = [
            {
                "id": n,
                "degree": degree_dict.get(n, 0),
                "weighted_degree": round(weighted_degree.get(n, 0), 4),
                "betweenness": round(betweenness.get(n, 0), 5),
                "closeness": round(closeness.get(n, 0), 5),
                "clustering": round(clustering.get(n, 0), 5),
                "eigenvector": round(eigenvector.get(n, 0), 5)
            }
            for n in G.nodes()
        ]
        edges_gc = [{"source": u, "target": v, "value": round(d["value"], 4)} for u, v, d in G.edges(data=True)]

        print(f"🌐 Final course network — Nodes: {len(nodes_out)}, Edges: {len(edges_gc)}")
        return {
            "message": f"✅ Skill co-occurrence network built for {len(all_courses)} courses.",
            "filters_used": {"keywords": keywords_list, "source": source},
            "summary": {
                "Courses Retrieved": len(all_courses),
                "Courses with Skills": len(skills_per_doc),
                "Nodes": len(nodes_out),
                "Edges": len(edges_gc)
            },
            "giant_component": {"nodes": nodes_out, "edges": edges_gc}
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"error": str(e)}


@app.get("/api/articles_mapped_kalos")
def articles_mapped(
    keywords:             str = Query(...,     description="Comma-separated keywords (e.g. AI, data, education)"),
    source:               str = Query("cordis",description="Source of the articles (default: cordis)"),
    publication_date_min: str = Query(None,    description="Minimum publication date (YYYY-MM-DD)"),
    publication_date_max: str = Query(None,    description="Maximum publication date (YYYY-MM-DD)"),
    max_edges:            int = Query(200,     description="Maximum number of edges to keep"),
    max_nodes:            int = Query(200,     description="Maximum number of nodes to keep"),
    max_pages:            int = Query(15,      description="Maximum number of pages to fetch (each page = 100 articles)")
):
    """Fetch filtered articles, build a skill co-occurrence network with centrality metrics."""
    try:
        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token   = res.text.replace('"', "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        print("✅ Authenticated successfully.")

        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        print(f"📡 Fetching articles matching keywords: {keywords_list}")

        all_articles = []
        for page in range(1, max_pages + 1):
            form_data = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("sources", source)]
            for kw in keywords_list:
                form_data.append(("keywords", kw))
            if publication_date_min:
                form_data.append(("publication_date_min", publication_date_min))
            if publication_date_max:
                form_data.append(("publication_date_max", publication_date_max))

            url = f"{API}/articles?page={page}&page_size=100"
            print(f"📄 Fetching page {page}/{max_pages}...")
            res = requests.post(url, headers=headers, data=form_data, timeout=90)

            if res.status_code != 200:
                print(f"⚠️ Page {page}: HTTP {res.status_code} → {res.text[:200]}")
                break

            data  = res.json()
            items = data.get("items", [])
            print(f"📦 Page {page}: Retrieved {len(items)} articles")

            if not items:
                print("✅ No more results — stopping.")
                break
            all_articles.extend(items)
            if len(items) < 100:
                print("✅ Last page reached (less than 100 results).")
                break

        print(f"🎯 Total articles retrieved: {len(all_articles)} (max {max_pages * 100})")
        if not all_articles:
            return {"message": f"No articles found for {keywords_list} from {source}."}

        skills_per_doc = [doc.get("skills", []) for doc in all_articles if doc.get("skills")]
        print(f"📊 Articles containing skills: {len(skills_per_doc)}")
        if not skills_per_doc:
            return {"message": "⚠️ No skills found in the retrieved articles."}

        co_counts    = defaultdict(int)
        skill_counts = defaultdict(int)
        for skills in skills_per_doc:
            unique_skills = set(skills)
            for s in unique_skills:
                skill_counts[s] += 1
            for s1, s2 in combinations(sorted(unique_skills), 2):
                co_counts[(s1, s2)] += 1

        edges = []
        for (s1, s2), cij in co_counts.items():
            ci, cj = skill_counts[s1], skill_counts[s2]
            if ci and cj:
                eij = (cij ** 2) / (ci * cj)
                edges.append({"source": labelize(s1), "target": labelize(s2), "value": round(eij, 4)})

        if not edges:
            return {"message": "⚠️ No co-occurrence relationships found."}

        edges = sorted(edges, key=lambda x: x["value"], reverse=True)[:max_edges]
        G = nx.Graph()
        for e in edges:
            G.add_edge(e["source"], e["target"], value=e["value"])

        if len(G.nodes()) > max_nodes:
            node_strength = {n: sum(d.get("value", 1) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
            top_nodes     = sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G             = G.subgraph({n for n, _ in top_nodes}).copy()
            print(f"⚙️ Simplified: retained top {len(G.nodes())} nodes")

        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)
            print(f"🧹 Removed {len(isolates)} isolated nodes")

        if G.number_of_nodes() == 0:
            return {"message": "⚠️ No valid network could be built."}

        print("📈 Computing centrality metrics...")
        try:
            degree_dict     = dict(G.degree())
            weighted_degree = dict(G.degree(weight="value"))
            betweenness     = nx.betweenness_centrality(G, weight="value", normalized=True)
            closeness       = nx.closeness_centrality(G)
            clustering      = nx.clustering(G, weight="value")
            try:
                eigenvector = nx.eigenvector_centrality(G, weight="value", max_iter=500)
            except nx.PowerIterationFailedConvergence:
                eigenvector = {n: 0 for n in G.nodes()}
                print("⚠️ Eigenvector centrality did not converge.")
        except Exception as e:
            print(f"⚠️ Metric computation error: {e}")
            degree_dict = weighted_degree = betweenness = closeness = clustering = eigenvector = {}

        nodes_out = [
            {
                "id": n,
                "degree": degree_dict.get(n, 0),
                "weighted_degree": round(weighted_degree.get(n, 0), 4),
                "betweenness": round(betweenness.get(n, 0), 5),
                "closeness": round(closeness.get(n, 0), 5),
                "clustering": round(clustering.get(n, 0), 5),
                "eigenvector": round(eigenvector.get(n, 0), 5)
            }
            for n in G.nodes()
        ]
        edges_gc = [{"source": u, "target": v, "value": round(d["value"], 4)} for u, v, d in G.edges(data=True)]

        print(f"🌐 Final article network — Nodes: {len(nodes_out)}, Edges: {len(edges_gc)}")
        return {
            "message": f"✅ Skill co-occurrence network built for {len(all_articles)} articles (up to {max_pages} pages).",
            "filters_used": {
                "keywords": keywords_list,
                "source": source,
                "publication_date_min": publication_date_min,
                "publication_date_max": publication_date_max,
                "max_pages": max_pages
            },
            "summary": {
                "Articles Retrieved": len(all_articles),
                "Articles with Skills": len(skills_per_doc),
                "Nodes": len(nodes_out),
                "Edges": len(edges_gc)
            },
            "giant_component": {"nodes": nodes_out, "edges": edges_gc}
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"error": str(e)}


@app.get("/api/profiles_mapped")
def profiles_mapped(
    keywords:  str = Query(...,  description="Comma-separated keywords (e.g. AI, data, education)"),
    source:    str = Query(None, description="Optional source filter for profiles (e.g. linkedin, eurofound)"),
    max_edges: int = Query(200,  description="Maximum number of edges to keep"),
    max_nodes: int = Query(200,  description="Maximum number of nodes to keep"),
):
    """
    Fetch ALL available pages of filtered profiles. Cached in 'Completed_Analyses/'.
    """
    folder = Path("Completed_Analyses")
    folder.mkdir(parents=True, exist_ok=True)

    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
    filename = "completed_analysis_profiles_mapped"
    for kw in keywords_list:
        filename += f"_{kw}"
    if source:
        filename += f"_{source}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache file path: {file_path}")

    if file_path.exists():
        print(f"✅ Cache hit — loading results from '{file_path}' (skipping API call).")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    print(f"🌐 No cache found — running full analysis from API...")

    try:
        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token   = res.text.replace('"', "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        print("✅ Authenticated successfully.")
        print(f"📡 Query — keywords: {keywords_list}")
        if source:
            print(f"🗂️ Source filter: {source}")

        def build_form_data():
            fd = [("keywords_logic", "or"), ("skill_ids_logic", "or")]
            for kw in keywords_list:
                fd.append(("keywords", kw))
            if source:
                fd.append(("sources", source))
            return fd

        page_size       = 100
        REQUEST_TIMEOUT = 180
        MAX_RETRIES     = 3
        RETRY_BACKOFF   = 10

        def fetch_page_with_retry(page_num: int) -> dict:
            url = f"{API}/profiles?page={page_num}&page_size={page_size}"
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"   ↪ Attempt {attempt}/{MAX_RETRIES} for page {page_num}...")
                    r = requests.post(url, headers=headers, data=build_form_data(), timeout=REQUEST_TIMEOUT)
                    if r.status_code != 200:
                        print(f"   ⚠️ HTTP {r.status_code} on page {page_num}: {r.text[:300]}")
                        return {}
                    return r.json()
                except requests.exceptions.ReadTimeout:
                    print(f"   ⏱️ ReadTimeout on page {page_num}, attempt {attempt}/{MAX_RETRIES}.")
                    if attempt < MAX_RETRIES:
                        print(f"   🔄 Retrying in {RETRY_BACKOFF}s...")
                        time.sleep(RETRY_BACKOFF)
                    else:
                        print(f"   ❌ All {MAX_RETRIES} attempts exhausted for page {page_num} — skipping.")
                        return {}
                except Exception as ex:
                    print(f"   ❌ {type(ex).__name__}: {ex}")
                    return {}

        print("🔍 Probing API page 1 to determine total record count...")
        probe_data = fetch_page_with_retry(1)
        if not probe_data:
            return {"error": "❌ Probe request (page 1) failed after all retries."}

        total_count = probe_data.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total records available: {total_count} → {total_pages} page(s)")

        if total_count == 0:
            return {"message": "No profiles found for the given filters."}

        all_profiles = list(probe_data.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(all_profiles)} profiles loaded.")

        for page in range(2, total_pages + 1):
            print(f"📄 Fetching page {page}/{total_pages}...")
            data  = fetch_page_with_retry(page)
            if not data:
                print(f"⚠️ Page {page} returned no data — stopping.")
                break
            items = data.get("items", [])
            print(f"📦 Page {page}/{total_pages}: {len(items)} profiles (running total: {len(all_profiles) + len(items)})")
            if not items:
                break
            all_profiles.extend(items)
            if len(items) < page_size:
                print("✅ Last page reached.")
                break

        print(f"🎯 Total profiles retrieved: {len(all_profiles)} / {total_count}")
        if not all_profiles:
            return {"message": f"No profiles found for {keywords_list}."}

        skills_per_doc = [doc.get("skills", []) for doc in all_profiles if doc.get("skills")]
        print(f"📊 Profiles containing skills: {len(skills_per_doc)} / {len(all_profiles)}")
        if not skills_per_doc:
            return {"message": "⚠️ No skills found in the retrieved profiles."}

        print("🕸️ Building skill co-occurrence matrix...")
        co_counts    = defaultdict(int)
        skill_counts = defaultdict(int)
        for skills in skills_per_doc:
            unique_skills = set(skills)
            for s in unique_skills:
                skill_counts[s] += 1
            for s1, s2 in combinations(sorted(unique_skills), 2):
                co_counts[(s1, s2)] += 1

        print(f"🔗 Raw co-occurrence pairs: {len(co_counts)} | Unique skills seen: {len(skill_counts)}")

        edges = []
        for (s1, s2), cij in co_counts.items():
            ci, cj = skill_counts[s1], skill_counts[s2]
            if ci and cj:
                eij = (cij ** 2) / (ci * cj)
                edges.append({"source": labelize(s1), "target": labelize(s2), "value": round(eij, 4)})

        if not edges:
            return {"message": "⚠️ No co-occurrence relationships found."}

        print(f"✂️ Trimming to top {max_edges} edges (from {len(edges)} total)...")
        edges = sorted(edges, key=lambda x: x["value"], reverse=True)[:max_edges]

        G = nx.Graph()
        for e in edges:
            G.add_edge(e["source"], e["target"], value=e["value"])

        print(f"📈 Initial graph — Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

        if len(G.nodes()) > max_nodes:
            node_strength = {n: sum(d.get("value", 1) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
            top_nodes     = sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G             = G.subgraph({n for n, _ in top_nodes}).copy()
            print(f"⚙️ Trimmed to top {len(G.nodes())} nodes by weighted degree")

        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)
            print(f"🧹 Removed {len(isolates)} isolated nodes")

        if G.number_of_nodes() == 0:
            return {"message": "⚠️ No valid network could be built."}

        print("📈 Computing centrality metrics...")
        try:
            degree_dict     = dict(G.degree())
            weighted_degree = dict(G.degree(weight="value"))
            betweenness     = nx.betweenness_centrality(G, weight="value", normalized=True)
            closeness       = nx.closeness_centrality(G)
            clustering      = nx.clustering(G, weight="value")
            try:
                eigenvector = nx.eigenvector_centrality(G, weight="value", max_iter=500)
                print("✅ Eigenvector centrality converged.")
            except nx.PowerIterationFailedConvergence:
                eigenvector = {n: 0 for n in G.nodes()}
                print("⚠️ Eigenvector centrality did not converge — defaulting to 0.")
        except Exception as e:
            print(f"⚠️ Metric computation error: {e} — falling back to zeros.")
            degree_dict     = dict(G.degree())
            weighted_degree = dict(G.degree(weight="value"))
            betweenness = closeness = clustering = eigenvector = {n: 0 for n in G.nodes()}

        nodes_out = [
            {
                "id": n,
                "degree": degree_dict.get(n, 0),
                "weighted_degree": round(weighted_degree.get(n, 0), 4),
                "betweenness": round(betweenness.get(n, 0), 5),
                "closeness": round(closeness.get(n, 0), 5),
                "clustering": round(clustering.get(n, 0), 5),
                "eigenvector": round(eigenvector.get(n, 0), 5)
            }
            for n in G.nodes()
        ]
        edges_gc = [{"source": u, "target": v, "value": round(d["value"], 4)} for u, v, d in G.edges(data=True)]

        print(f"🌐 Final profile network — Nodes: {len(nodes_out)}, Edges: {len(edges_gc)}")

        result = {
            "message": f"✅ Skill co-occurrence network built for {len(all_profiles)} profiles ({total_pages} pages, {total_count} total available).",
            "filters_used": {"keywords": keywords_list, "source": source},
            "summary": {
                "Profiles Retrieved": len(all_profiles),
                "Total Profiles Available": total_count,
                "Pages Fetched": total_pages,
                "Profiles with Skills": len(skills_per_doc),
                "Nodes": len(nodes_out),
                "Edges": len(edges_gc)
            },
            "giant_component": {"nodes": nodes_out, "edges": edges_gc}
        }

        print(f"💾 Saving results to cache: '{file_path}'...")
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)
        print(f"✅ Results cached successfully to '{file_path}'.")
        return result

    except Exception as e:
        print(f"❌ ERROR in profiles_mapped: {e}")
        return {"error": str(e)}


@app.get("/api/jobs_mapped_ultra")
def jobs_mapped(
    keywords:        Optional[str] = Query(None, description="Comma-separated keywords (e.g. AI, data, software)"),
    source:          str           = Query(None, description="Optional source filter for jobs (e.g. linkedin, indeed)"),
    occupation_ids:  Optional[str] = Query(None, description="Comma-separated occupation IDs (e.g. http://data.europa.eu/esco/occupation/...)"),
    min_upload_date: str           = Query(None, description="Minimum upload date (YYYY-MM-DD)"),
    max_upload_date: str           = Query(None, description="Maximum upload date (YYYY-MM-DD)"),
    max_edges:       int           = Query(200,  description="Maximum number of edges to keep"),
    max_nodes:       int           = Query(200,  description="Maximum number of nodes to keep"),
):
    """
    Fetch ALL available pages of filtered jobs. Cached in 'Completed_Analyses/'.
    Builds a skill co-occurrence network (URI → label) with centrality metrics.
    """
    folder = Path("Completed_Analyses")
    folder.mkdir(parents=True, exist_ok=True)

    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    occ_ids_list  = [o.strip() for o in occupation_ids.split(",") if o.strip()] if occupation_ids else []

    filename = "completed_analysis_jobs_mapped_ultra"
    for kw in keywords_list:
        filename += f"_{kw}"
    for occ in occ_ids_list:
        match = re.search(r'C\d+$', occ)
        filename += f"_{match.group(0)}" if match else f"_{occ.replace('/', '_').replace(':', '').replace('.', '')}"
    if source:
        filename += f"_{source}"
    if min_upload_date:
        filename += f"_from{min_upload_date}"
    if max_upload_date:
        filename += f"_to{max_upload_date}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache file path: {file_path}")

    if file_path.exists():
        print(f"✅ Cache hit — loading results from '{file_path}' (skipping API call).")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    print(f"🌐 No cache found — running full analysis from API...")

    try:
        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token   = res.text.replace('"', "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        print("✅ Authenticated successfully.")
        print(f"📡 Query — keywords: {keywords_list if keywords_list else '(none)'}")
        if occ_ids_list:
            print(f"🏢 Occupation IDs filter: {occ_ids_list}")
        if source:
            print(f"🗂️ Source filter: {source}")
        if min_upload_date or max_upload_date:
            print(f"📅 Date range: {min_upload_date or 'any'} → {max_upload_date or 'any'}")

        def build_form_data():
            fd = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("occupation_ids_logic", "or")]
            for kw in keywords_list:
                fd.append(("keywords", kw))
            for occ in occ_ids_list:
                fd.append(("occupation_ids", occ))
            if source:
                fd.append(("sources", source))
            if min_upload_date:
                fd.append(("min_upload_date", min_upload_date))
            if max_upload_date:
                fd.append(("max_upload_date", max_upload_date))
            return fd

        page_size       = 100
        REQUEST_TIMEOUT = 180
        MAX_RETRIES     = 3
        RETRY_BACKOFF   = 10

        def fetch_page_with_retry(page_num: int) -> dict:
            url = f"{API}/jobs?page={page_num}&page_size={page_size}"
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"   ↪ Attempt {attempt}/{MAX_RETRIES} for page {page_num} (timeout={REQUEST_TIMEOUT}s)...")
                    r = requests.post(url, headers=headers, data=build_form_data(), timeout=REQUEST_TIMEOUT)
                    if r.status_code != 200:
                        print(f"   ⚠️ HTTP {r.status_code} on page {page_num}: {r.text[:300]}")
                        return {}
                    return r.json()
                except requests.exceptions.ReadTimeout:
                    print(f"   ⏱️ ReadTimeout on page {page_num}, attempt {attempt}/{MAX_RETRIES}.")
                    if attempt < MAX_RETRIES:
                        print(f"   🔄 Retrying in {RETRY_BACKOFF}s...")
                        time.sleep(RETRY_BACKOFF)
                    else:
                        print(f"   ❌ All {MAX_RETRIES} attempts exhausted for page {page_num} — skipping.")
                        return {}
                except Exception as ex:
                    print(f"   ❌ {type(ex).__name__}: {ex}")
                    return {}

        print("🔍 Probing API page 1 to determine total record count...")
        probe_data = fetch_page_with_retry(1)
        if not probe_data:
            return {"error": "❌ Probe request (page 1) failed after all retries."}

        total_count = probe_data.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total records available: {total_count} → {total_pages} page(s)")

        if total_count == 0:
            return {"message": "No job postings found for the given filters."}

        all_jobs = list(probe_data.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(all_jobs)} job postings loaded from probe.")

        for page in range(2, total_pages + 1):
            print(f"📄 Fetching page {page}/{total_pages}...")
            data = fetch_page_with_retry(page)
            if not data:
                print(f"⚠️ Page {page} returned no data — stopping.")
                break
            items = data.get("items", [])
            print(f"📦 Page {page}/{total_pages}: {len(items)} job postings (running total: {len(all_jobs) + len(items)})")
            if not items:
                print("✅ No more results — stopping early.")
                break
            all_jobs.extend(items)
            if len(items) < page_size:
                print("✅ Last page reached.")
                break

        print(f"🎯 Total jobs retrieved: {len(all_jobs)} / {total_count} available")
        if not all_jobs:
            return {"message": f"No job postings found for {keywords_list}."}

        skills_per_doc = [doc.get("skills", []) for doc in all_jobs if doc.get("skills")]
        print(f"📊 Job postings containing skills: {len(skills_per_doc)} / {len(all_jobs)}")
        if not skills_per_doc:
            return {"message": "⚠️ No skills found in the retrieved job postings."}

        print("🕸️ Building skill co-occurrence matrix...")
        co_counts    = defaultdict(int)
        skill_counts = defaultdict(int)
        for skills in skills_per_doc:
            unique_skills = set(skills)
            for s in unique_skills:
                skill_counts[s] += 1
            for s1, s2 in combinations(sorted(unique_skills), 2):
                co_counts[(s1, s2)] += 1

        print(f"🔗 Raw co-occurrence pairs: {len(co_counts)} | Unique skills seen: {len(skill_counts)}")

        edges = []
        for (s1, s2), cij in co_counts.items():
            ci, cj = skill_counts[s1], skill_counts[s2]
            if ci and cj:
                eij = (cij ** 2) / (ci * cj)
                edges.append({"source": labelize(s1), "target": labelize(s2), "value": round(eij, 4)})

        if not edges:
            return {"message": "⚠️ No co-occurrence relationships found."}

        print(f"✂️ Trimming to top {max_edges} edges (from {len(edges)} total)...")
        edges = sorted(edges, key=lambda x: x["value"], reverse=True)[:max_edges]

        G = nx.Graph()
        for e in edges:
            G.add_edge(e["source"], e["target"], value=e["value"])

        print(f"📈 Initial graph — Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

        if len(G.nodes()) > max_nodes:
            node_strength = {n: sum(d.get("value", 1) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
            top_nodes     = sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G             = G.subgraph({n for n, _ in top_nodes}).copy()
            print(f"⚙️ Trimmed to top {len(G.nodes())} nodes by weighted degree")

        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)
            print(f"🧹 Removed {len(isolates)} isolated nodes")

        if G.number_of_nodes() == 0:
            return {"message": "⚠️ No valid network could be built."}

        print("📈 Computing centrality metrics...")
        try:
            degree_dict     = dict(G.degree())
            weighted_degree = dict(G.degree(weight="value"))
            betweenness     = nx.betweenness_centrality(G, weight="value", normalized=True)
            closeness       = nx.closeness_centrality(G)
            clustering      = nx.clustering(G, weight="value")
            try:
                eigenvector = nx.eigenvector_centrality(G, weight="value", max_iter=500)
                print("✅ Eigenvector centrality converged.")
            except nx.PowerIterationFailedConvergence:
                eigenvector = {n: 0 for n in G.nodes()}
                print("⚠️ Eigenvector centrality did not converge — defaulting to 0.")
        except Exception as e:
            print(f"⚠️ Metric computation error: {e} — falling back to zeros.")
            degree_dict     = dict(G.degree())
            weighted_degree = dict(G.degree(weight="value"))
            betweenness = closeness = clustering = eigenvector = {n: 0 for n in G.nodes()}

        nodes_out = [
            {
                "id": n,
                "degree": degree_dict.get(n, 0),
                "weighted_degree": round(weighted_degree.get(n, 0), 4),
                "betweenness": round(betweenness.get(n, 0), 5),
                "closeness": round(closeness.get(n, 0), 5),
                "clustering": round(clustering.get(n, 0), 5),
                "eigenvector": round(eigenvector.get(n, 0), 5)
            }
            for n in G.nodes()
        ]
        edges_gc = [{"source": u, "target": v, "value": round(d["value"], 4)} for u, v, d in G.edges(data=True)]

        print(f"🌐 Final job network — Nodes: {len(nodes_out)}, Edges: {len(edges_gc)}")

        result = {
            "message": f"✅ Skill co-occurrence network built for {len(all_jobs)} job postings ({total_pages} pages, {total_count} total available).",
            "filters_used": {
                "keywords": keywords_list,
                "occupation_ids": occ_ids_list if occ_ids_list else None,
                "source": source,
                "min_upload_date": min_upload_date,
                "max_upload_date": max_upload_date,
            },
            "summary": {
                "Jobs Retrieved": len(all_jobs),
                "Total Jobs Available": total_count,
                "Pages Fetched": total_pages,
                "Jobs with Skills": len(skills_per_doc),
                "Nodes": len(nodes_out),
                "Edges": len(edges_gc)
            },
            "giant_component": {"nodes": nodes_out, "edges": edges_gc}
        }

        print(f"💾 Saving results to cache: '{file_path}'...")
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)
        print(f"✅ Results cached successfully to '{file_path}'.")
        return result

    except Exception as e:
        print(f"❌ ERROR in jobs_mapped_ultra: {e}")
        return {"error": str(e)}
