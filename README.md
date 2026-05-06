# Giant Component Networks Back-End

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/skillab-project/giant_component_networks)

## Description

This project implements the backend API for **Giant Component Networks**, an open-source framework that constructs **skill co-occurrence networks** from multiple SkillLab data sources and extracts the **giant connected component** — the largest maximally connected subgraph — along with a full suite of network centrality metrics.

It is built with FastAPI (Python) and exposes endpoints for:

- Fetching ESCO-tagged jobs, professional profiles, online courses, research articles, and EU law/policy documents from the SkillLab Tracker API.
- Fetching Knowledge Unit (KU) detection results from the SkillLab KU API.
- Building skill co-occurrence networks using a normalised edge weight formula: `eij = (co_count²) / (count_i × count_j)`.
- Extracting and returning the giant component of each network.
- Computing node-level centrality metrics: degree, weighted degree, betweenness, closeness, clustering coefficient, and eigenvector centrality.
- Caching completed analyses to `Completed_Analyses/` with in-progress lock detection.

A local ESCO skills cache (`esco_skills_cache.json`) is auto-populated on first startup by fetching all skills from the Tracker API and is reused on subsequent runs.

The service is part of the [SkillLab](https://github.com/skillab-project) EU Horizon Europe project.

---

## Getting Started Guide

### Prerequisites

- **Python 3.11** or newer ([Download Python](https://www.python.org/downloads/))
- **Git** ([Download Git](https://git-scm.com/downloads))
- **Access to the SkillLab Tracker API** — credentials for `TRACKER_API`, `TRACKER_USERNAME`, and `TRACKER_PASSWORD`.
- **Optional:** Access to the SkillLab KU API (`KU_API_URL`) for KU co-occurrence network endpoints.

---

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/skillab-project/giant_component_networks.git
   cd giant_component_networks
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your `.env` file:**

   ```env
   TRACKER_API=https://skillab-tracker.csd.auth.gr/api
   TRACKER_USERNAME=your_username
   TRACKER_PASSWORD=your_password
   KU_API_URL=https://your-ku-api-host/api   # Optional, for KU endpoints
   ```

---

## Running the Application

### Locally

```bash
uvicorn giant_component_networks:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible at `http://localhost:8000`. Swagger UI is at `http://localhost:8000/docs`.

> **First run:** On startup the app fetches all ESCO skills from the Tracker API and saves them to `esco_skills_cache.json`. This may take a few minutes. Subsequent startups load the cache instantly.

### With Docker

```bash
docker-compose up --build
```

Or manually:

```bash
docker build -t giant-component-networks .
docker run -p 8007:8000 --env-file .env giant-component-networks
```

---

## API Endpoints

All endpoints return a `giant_component` object with `nodes` and `edges`, a `summary`, and `filters_used`.

Each node carries: `id`, `degree`, `weighted_degree`, `betweenness`, `closeness`, `clustering`, `eigenvector`.

Each edge carries: `source`, `target`, `value` (normalised co-occurrence weight).

### `GET /api/law-policies_mapped`

Skill co-occurrence network from EU law/policy documents.

| Parameter    | Default    | Description                                   |
|--------------|------------|-----------------------------------------------|
| `keywords`   | required   | Comma-separated keywords                      |
| `source`     | `eur_lex`  | Document source filter                        |
| `max_edges`  | `100`      | Top edges to retain (law policies auto-expand)|
| `max_nodes`  | `200`      | Max nodes to retain                           |

### `GET /api/courses_mapped`

Skill co-occurrence network from online courses.

| Parameter    | Default     | Description               |
|--------------|-------------|---------------------------|
| `keywords`   | required    | Comma-separated keywords  |
| `source`     | `coursera`  | Course source filter      |
| `max_edges`  | `200`       |                           |
| `max_nodes`  | `200`       |                           |

### `GET /api/articles_mapped_kalos`

Skill co-occurrence network from research articles.

| Parameter              | Default   | Description                          |
|------------------------|-----------|--------------------------------------|
| `keywords`             | required  |                                      |
| `source`               | `cordis`  |                                      |
| `publication_date_min` | —         | `YYYY-MM-DD`                         |
| `publication_date_max` | —         | `YYYY-MM-DD`                         |
| `max_pages`            | `15`      | Max API pages to fetch (100 per page)|

### `GET /api/profiles_mapped`

Skill co-occurrence network from professional profiles. Results are cached.

### `GET /ku-co-occurrence`

KU co-occurrence network from Knowledge Unit detection results.

| Parameter      | Default | Description                              |
|----------------|---------|------------------------------------------|
| `start_date`   | —       | `YYYY-MM`                                |
| `end_date`     | —       | `YYYY-MM`                                |
| `kus`          | —       | Comma-separated KU IDs to filter         |
| `organization` | —       | Filter by organization name              |

Additional job-based endpoints are available in `giant_component_networks.py` — see the Swagger UI for the full list.

---

## Running the Tests

```bash
pytest tests/
```

---

## Project Structure

```
giant_component_networks/
├── giant_component_networks.py  # FastAPI app, network construction, all endpoints
├── network_analysis.py          # Shared network analysis utilities
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Compose configuration
├── .env                         # Environment variables (fill in before running)
├── esco_skills_cache.json       # Auto-generated ESCO skill cache (created at first run)
├── Completed_Analyses/          # Auto-created cache folder for completed analyses
├── jenkins/                     # CI/CD pipeline configuration
└── tests/                       # Test suite
```

---

## Technologies

- **Python 3.11**
- **FastAPI** — REST API framework
- **Uvicorn** — ASGI server
- **NetworkX** — Graph construction, giant component extraction, centrality metrics
- **pandas / NumPy** — Data processing
- **python-dotenv** — Environment variable management
- **Docker / Docker Compose** — Containerised deployment

---

## License

This project is licensed under the **Eclipse Public License 2.0 (EPL-2.0)**. See the [LICENSE](LICENSE) file for details.
