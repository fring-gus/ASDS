# ASDS — Automated Similarity Detection System

ASDS is an academic project-similarity detection platform built with Flask, SQLAlchemy, and Sentence Transformers. It allows faculty to manage classes, collect student project abstracts (PDF/DOCX), and detect problem-domain overlaps using a two-stage neural similarity engine.



ASDS similarity pipeline:

1. **Two-Stage Screening & Replacement**  
   - **Stage 1 (Screening)**: Fast pairwise semantic embeddings via `all-mpnet-base-v2` (768-dim) with dynamic GPU (64) / CPU (16) batch sizing.  
   - **High-Confidence Bypass (`score >= 0.80`)**: Auto-flagged high-confidence duplicates skip cross-encoder reranking, saving computation.  
   - **Stage 2 (Cross-Encoder Reranking)**: Borderline pairs (`threshold <= score < 0.80`) are evaluated in a vectorized batch using `cross-encoder/stsb-roberta-base` Cross-encoder scores directly replace bi-encoder scores for borderline pairs, eliminating arbitrary heuristic weighting.

2. **Domain-Specific Stopwords Filtering (`TECH_STOPWORDS`)**  
   Strips ~677 curated computer science, language, framework, database, infrastructure, and academic filler terms during preprocessing. This prevents projects from being falsely flagged simply because they share common tech stacks (e.g., *Python, React, Flask, Docker, PostgreSQL*), focusing comparison strictly on the **problem domain**.
  

## Core Features

- **Role-Based Portals**: Separate dashboards for Students, Faculty, and Administrators.
- **Class Management**: Faculty generate unique join codes, open/close submission windows, and review rosters.
- **Document Ingestion**: Automatic plain-text extraction from student PDF and DOCX uploads.
- **Priority Resolution**: When duplicate abstracts are detected, the earlier submission remains `accepted`; the later submission is marked `rejected`.
- **Live Progress Polling**: Non-blocking background comparisons with real-time status steps in UI modals.
- **Manual Comparison Sandbox**: Faculty can upload and compare up to 120 arbitrary documents in-memory without affecting class records.
- **Reporting**: Export structured Excel reports (.xlsx) detailing submission statuses and collision pairings.

---

## How the Engine Works

```
Raw Submissions (PDF / DOCX)
         │
         ▼
[Text Extraction & Preprocessing]
  └── Strip whitespace, punctuation, and ~677 TECH_STOPWORDS
         │
         ├──► [Semantic Mode]
         │      │
         │      ▼
         │    Stage 1: all-mpnet-base-v2 (GPU/CPU batched embeddings)
         │      │
         │      ├── Bi-score >= 0.80 ──► [Auto-Flagged (Skip Cross-Encoder)]
         │      ├── Bi-score < 0.65  ──► [Marked Safe (Skip Cross-Encoder)]
         │      └── 0.65 <= Bi-score < 0.80 (Borderline Gray Zone)
         │               │
         │               ▼
         │             Stage 2: cross-encoder/stsb-roberta-base
         │               │
         │               └── CE-Score replaces Bi-Score ──► Flag if CE >= 0.65
         │
         └──► [Lexical Mode]
                │
                ▼
              TF-IDF (1-2 n-grams, max 5000 features) ──► Cosine Similarity
```

---

## Quick Start

### 1. Environment Setup

```powershell
python -m venv asds
.\asds\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

Create a `.env` file in the project root:

```ini
SECRET_KEY=change-this-to-a-secure-random-key
FLASK_ENV=development
HOST=127.0.0.1
PORT=5000
```

### 3. Run the Application

```powershell
python run.py
```

Access the app at [http://127.0.0.1:5000](http://127.0.0.1:5000).

> **Note**: On the very first run, Sentence Transformers automatically caches `all-mpnet-base-v2` and `cross-encoder/stsb-roberta-base` from Hugging Face. Subsequent runs load from local cache.

---

## User Workflows

- **Faculty**:
  1. Register/Login &rarr; Create a class section &rarr; Share the 8-character class code.
  2. Monitor student submissions &rarr; Click **Close Uploads** when the deadline passes.
  3. Click **Run Comparison** &rarr; Monitor real-time progress.
  4. Review flagged vs. safe pairs &rarr; Click **Export Excel** for the final grade report.
  5. Use **Manual File Comparison** to evaluate ad-hoc files directly from disk without modifying class records.

- **Students**:
  1. Register/Login &rarr; Join a class via class code.
  2. Submit project title & upload PDF/DOCX abstract.
  3. View acceptance/rejection status after faculty runs comparison.

- **Administrators**:
  - Access `/admin/login` to manage user accounts, change passwords, and view system statistics.

---

## Configuration Reference

| Environment Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | `dev-only-change-me` | Session encryption and CSRF signing key. |
| `DATABASE_URL` | `sqlite:///app.db` | SQLAlchemy database connection URI. |
| `FLASK_ENV` | `development` | Environment mode (`development` or `production`). |
| `SECURE_COOKIES` | `True` (production) | Restricts session cookies to HTTPS connections. |
| `HOST` / `PORT` | `127.0.0.1:5000` | Local server binding address and port. |

---

## Repository Structure

```
├── app/
│   ├── __init__.py           # Flask app factory, extensions, and error handlers
│   ├── engine.py             # Similarity engine (v2.1-ce-replace)
│   ├── tech_stopwords.py     # Curated technical & academic stopword dictionary
│   ├── models.py             # SQLAlchemy models (User, Class, Submission, Result)
│   ├── faculty.py            # Faculty routes (classes, comparison, exports)
│   ├── student.py            # Student routes (enrolling, upload, submissions)
│   ├── admin.py              # Administrator management routes
│   ├── auth.py               # Authentication, login, registration
│   ├── routes.py             # Main index and landing redirects
│   ├── static/               # CSS styles and front-end assets
│   └── templates/            # Jinja2 HTML templates
├── instance/
│   └── app.db                # SQLite database
├── uploads/                  # Uploaded student submission files
├── config.py                 # Application configuration classes
├── run.py                    # Entry point script
└── requirements.txt          # Production dependencies
```
## Development hygiene

Do not commit local runtime data or secrets. At minimum, ignore:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
instance/*.db
uploads/
asds.log
```

## License

No license is currently declared. Add one before redistributing the project.
