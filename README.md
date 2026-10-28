# ASDS — Automated Similarity Detection System

ASDS is a Flask web application for managing project-abstract submissions and identifying potentially similar work. Faculty create classes, students submit PDF or DOCX abstracts, and faculty run a similarity report after submissions close.

The application uses a hybrid comparison pipeline that combines semantic embeddings, lexical overlap, and character n-gram structure. It is intended to highlight submissions that deserve human review; it is not a plagiarism verdicting system.

## Features

- Role-based accounts for students, faculty, and administrators.
- Faculty-managed classes with unique join codes.
- Student enrollment and PDF/DOCX abstract uploads.
- Text extraction from PDF and Word documents.
- Hybrid similarity analysis:
  - semantic embeddings using `all-mpnet-base-v2`;
  - TF-IDF cosine similarity using unigrams and bigrams;
  - character-trigram Jaccard similarity;
  - cross-encoder reranking of borderline matches.
- Configurable 65% default similarity threshold.
- Background comparison with live progress updates.
- Manual, faculty-only comparison of arbitrary PDF/DOCX files without changing class submissions.
- Downloadable original documents and Excel comparison reports.
- SQLite by default, with support for an alternate SQLAlchemy `DATABASE_URL`.
- Health endpoint at `GET /health`.

## Architecture

```text
Browser
  │
  ▼
Flask blueprints and Jinja templates
  ├── Authentication and role checks
  ├── Student submissions
  ├── Faculty class/report management
  └── Administrator account management
  │
  ├── SQLAlchemy database
  │     User ── Class ── Submission ── Result
  │
  ├── Local upload storage
  │
  └── Similarity engine
        preprocessing → embeddings / TF-IDF / n-grams
                      → optional cross-encoder → stored results
```

## Requirements

- Python with a compatible PyTorch wheel for your platform.
- Internet access for the first semantic comparison, unless the Hugging Face models are already cached.
- Sufficient CPU, RAM, and disk space for PyTorch and the downloaded language models.

Core dependencies are listed and pinned in [`requirements.txt`](requirements.txt).

## Quick start

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local environment file named `.env`:

```dotenv
SECRET_KEY=replace-with-a-long-random-secret
FLASK_ENV=development
HOST=127.0.0.1
PORT=5000
```

Start the application:

```powershell
python run.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

On first use of semantic comparison, Sentence Transformers downloads the `all-mpnet-base-v2` model. The cross-encoder model downloads when a borderline pair needs reranking, so the first comparison can take longer than later runs.

## User workflow

### Faculty

1. Register as a faculty user and sign in.
2. Create a class; share its generated code with students.
3. Review enrolled students and their submissions.
4. Close uploads once submission is complete.
5. Run comparison and wait for the progress indicator to finish.
6. Review flagged pairs and export the Excel report if needed.

### Students

1. Register as a student and sign in.
2. Join a class using the faculty-provided code.
3. Upload a project title and a text-based PDF or DOCX abstract.
4. Re-upload while the class remains open, if permitted.

### Administrators

Administrators can create users, reset passwords, and remove accounts with associated data. The application currently has no command to bootstrap its first administrator; create or promote the initial admin through controlled database administration before deploying.

## Similarity scoring

For each submission pair, ASDS calculates:

| Signal | Weight | Purpose |
|---|---:|---|
| Semantic embeddings | 55% | Finds related meaning and paraphrased content |
| TF-IDF cosine similarity | 25% | Finds lexical overlap and copied wording |
| Character-trigram Jaccard score | 20% | Finds structurally similar phrasing |

Pairs with hybrid scores between 0.50 and 0.80 receive cross-encoder verification. Scores at or above the configured threshold—0.65 by default—are flagged. When a class is processed, the later submitted item in a flagged pair is marked `rejected`; the earlier submission has priority.

Thresholds and weights are defined in [`app/engine.py`](app/engine.py). Treat every match as a review signal, not automated evidence of misconduct.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | Development fallback | Flask session/CSRF signing key. Set a strong value in every deployed environment. |
| `FLASK_ENV` | `development` | Use `production` to select production configuration. |
| `HOST` | `127.0.0.1` | Bind address used by `run.py`. |
| `PORT` | `5000` | Bind port used by `run.py`. |
| `DATABASE_URL` | `sqlite:///app.db` | SQLAlchemy connection URL. |
| `SECURE_COOKIES` | `True` in production | Send session cookies only over HTTPS. |

Uploads are saved under `uploads/` by default. The global request limit is 500 MB; configure a stricter limit if that better suits the deployment.

## Production notes

For production, serve the application behind a reverse proxy and a WSGI server rather than Flask’s development server:

```bash
gunicorn run:app
```

Set at least these environment variables in production:

```dotenv
FLASK_ENV=production
SECRET_KEY=long-random-secret
DATABASE_URL=your-production-database-url
SECURE_COOKIES=True
```

The background progress implementation keeps locks and progress state in process memory. It is suitable for a single application process. For multi-worker or horizontally scaled deployments, move analysis to a durable job queue and persist job state.

## Health check

```text
GET /health
```

Response:

```json
{"status":"ok","service":"ASDS"}
```

## Repository layout

```text
app/
  admin.py                 Administrator routes
  auth.py                  Registration, login, role checks
  engine.py                Hybrid similarity engine
  faculty.py               Class, progress, export, manual comparison routes
  models.py                SQLAlchemy models
  student.py               Enrollment and submission routes
  templates/               Jinja templates
  static/                  CSS and browser assets
config.py                  Application configuration
run.py                     Application entry point
requirements.txt           Python dependencies
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
