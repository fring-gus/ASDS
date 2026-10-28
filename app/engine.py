"""
ASDS — Automated Similarity Detection System
Engine Version: v2.1-ce-replace

Multi-Stage Similarity Engine with Domain-Aware Stopwords Filtering:
  1. Technical & Academic Stopwords Filtering:
     Strips ~400 curated CS/Engineering terms and academic filler words
     via TECH_STOPWORDS so similarity focuses on the *problem domain*
     rather than shared libraries/frameworks.
  2. Two-Stage Semantic Pipeline (v2.1-ce-replace):
     - Stage 1: Bi-encoder screening with all-mpnet-base-v2 (768-dim)
       using dynamic batch sizing (64 GPU, 16 CPU).
     - Cutoffs:
       * High-confidence (score >= 0.80): Auto-flagged. Skips cross-encoder.
       * Low-confidence (score < threshold): Safe. Skips cross-encoder.
       * Borderline (threshold <= score < 0.80): Cross-encoder reranking
         with cross-encoder/stsb-roberta-base in a vectorized batch.
     - CE-Replace: The cross-encoder score directly replaces the bi-encoder
       score for borderline pairs.
  3. Optional Lexical Mode:
     - TF-IDF unigram + bigram cosine similarity for assignment checking.
  4. Integration & Concurrency:
     - Thread-safe per-class/task locks.
     - Live progress tracking polled by Flask frontend modals.
     - Database transaction safety (savepoint + rollback).
"""

import os
import re
import shutil
import logging
import threading
from itertools import combinations

from sentence_transformers import SentenceTransformer, CrossEncoder, util as st_util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

try:
    from app.tech_stopwords import TECH_STOPWORDS
except ImportError:
    from tech_stopwords import TECH_STOPWORDS

# ── Version & Configuration ──
ENGINE_VERSION = 'v2.1-ce-replace'
SIMILARITY_THRESHOLD = 0.65       # Default decision threshold (0.0 – 1.0)
CE_SKIP = 0.80                    # High-confidence threshold (skips cross-encoder)
MIN_TOKENS = 15                   # Minimum extracted tokens to consider valid

# Pre-compiled regexes for text normalization
_RE_WHITESPACE = re.compile(r'\s+')
_RE_PUNCTUATION = re.compile(r'[^\w\s]')

# ML models cached for the life of the process
_bi_encoder = None
_cross_encoder = None
_BATCH_SIZE = None

# Concurrency locks
_comparison_locks = {}            # {id: threading.Lock()}
_locks_guard = threading.Lock()   # protects _comparison_locks dict

# Progress tracking for live UI polling
_comparison_progress = {}         # {id: {step, percent, message, status, [result]}}
_progress_lock = threading.Lock()

logger = logging.getLogger(__name__)


# ── GPU / CPU Batch Sizing ──

def _get_batch_size():
    """Select encode batch size: 64 on GPU, 16 on CPU. Cached after first call."""
    global _BATCH_SIZE
    if _BATCH_SIZE is not None:
        return _BATCH_SIZE
    try:
        import torch
        _BATCH_SIZE = 64 if torch.cuda.is_available() else 16
    except Exception:
        _BATCH_SIZE = 16
    return _BATCH_SIZE


# ── Model Loaders with Error Recovery ──

def _get_bi_encoder():
    """Load the bi-encoder model with automatic cache recovery on retry."""
    global _bi_encoder
    if _bi_encoder is None:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                _bi_encoder = SentenceTransformer('all-mpnet-base-v2')
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to load bi-encoder model: {e}")
                    raise
                # Clear corrupted huggingface cache if any
                hf_cache = os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface')
                hub_path = os.path.join(hf_cache, 'hub')
                if os.path.exists(hub_path):
                    try:
                        shutil.rmtree(hub_path)
                    except Exception:
                        pass
    return _bi_encoder


def _get_cross_encoder():
    """Load the cross-encoder model (cached after first call)."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')
    return _cross_encoder


# ── Text Preprocessing ──

def preprocess_text(text):
    """Normalize text before similarity analysis.

    Lowercases text, collapses whitespace, strips punctuation, and removes
    domain-specific technical & academic stopwords so comparison focuses on
    the problem domain rather than technology buzzwords.
    """
    if not text:
        return ''
    cleaned = text.lower()
    cleaned = _RE_WHITESPACE.sub(' ', cleaned).strip()
    cleaned = _RE_PUNCTUATION.sub('', cleaned)
    words = [w for w in cleaned.split() if w not in TECH_STOPWORDS]
    if len(words) < 3:
        # Fallback to cleaned words if tech stopwords stripped almost everything
        words = cleaned.split()
    return ' '.join(words)


# ── Concurrency & Progress Tracking ──

def _get_class_lock(lock_id):
    """Get or create a per-resource lock to prevent concurrent comparisons."""
    with _locks_guard:
        if lock_id not in _comparison_locks:
            _comparison_locks[lock_id] = threading.Lock()
        return _comparison_locks[lock_id]


def _update_progress(task_id, step, percent, message):
    """Update progress state for polling endpoints."""
    with _progress_lock:
        existing = _comparison_progress.get(task_id, {})
        existing.update({
            'step': step,
            'percent': percent,
            'message': message,
            'status': 'running',
        })
        _comparison_progress[task_id] = existing


def _finish_progress(task_id, status='done', message='Comparison complete.'):
    """Mark progress as completed or failed."""
    with _progress_lock:
        existing = _comparison_progress.get(task_id, {})
        existing.update({
            'step': 'done' if status == 'done' else 'error',
            'percent': 100 if status == 'done' else existing.get('percent', 0),
            'message': message,
            'status': status,
        })
        _comparison_progress[task_id] = existing


def get_progress(task_id):
    """Retrieve current progress state for a class or manual comparison task."""
    with _progress_lock:
        return _comparison_progress.get(task_id)


# ── Core Analysis Engines ──

def _run_semantic_pipeline(processed_texts, threshold, progress_fn=None):
    """Run two-stage semantic similarity (v2.1-ce-replace).

    Stage 1: Encode texts with bi-encoder (all-mpnet-base-v2) and compute cosine similarity.
    Stage 2: Cross-encoder reranking (stsb-roberta-base) for borderline pairs
             where threshold <= bi_score < CE_SKIP. The cross-encoder score
             replaces the bi-encoder score.
    """
    n = len(processed_texts)
    total_pairs = n * (n - 1) // 2

    if progress_fn:
        progress_fn('semantic', 30, 'Computing semantic embeddings (all-mpnet-base-v2)...')

    bi_model = _get_bi_encoder()
    embeddings = bi_model.encode(
        processed_texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=_get_batch_size()
    )
    semantic_matrix = st_util.cos_sim(embeddings, embeddings)

    if progress_fn:
        progress_fn('comparing', 50, f'Screening {total_pairs} pairs...')

    results = []
    for i, j in combinations(range(n), 2):
        bi_score = round(float(semantic_matrix[i][j]), 4)
        results.append({
            'i': i,
            'j': j,
            'score': bi_score,
            'is_flagged': bi_score >= threshold,
            'mode': 'semantic',
        })

    # Identify borderline pairs for cross-encoder reranking
    gray_zone = [
        idx for idx, r in enumerate(results)
        if r['is_flagged'] and r['score'] < CE_SKIP
    ]
    auto_flag = sum(1 for r in results if r['score'] >= CE_SKIP)
    cross_encoder_calls = len(gray_zone)

    if gray_zone:
        if progress_fn:
            progress_fn('comparing', 80,
                        f'Cross-encoding {len(gray_zone)} borderline pairs ({auto_flag} high-confidence skipped)...')

        cross_model = _get_cross_encoder()
        pairs = [[processed_texts[results[idx]['i']], processed_texts[results[idx]['j']]] for idx in gray_zone]
        ce_scores = cross_model.predict(pairs, show_progress_bar=False)

        # Handle scalar or 1D array return safely
        if hasattr(ce_scores, 'ndim') and ce_scores.ndim == 0:
            ce_scores = [float(ce_scores)]
        elif not hasattr(ce_scores, '__len__'):
            ce_scores = [float(ce_scores)]

        # Cross-encoder score replaces bi-encoder score for borderline pairs
        for k, idx in enumerate(gray_zone):
            ce_score = round(float(ce_scores[k]), 4)
            ce_score = max(0.0, min(1.0, ce_score))
            results[idx]['score'] = ce_score
            results[idx]['is_flagged'] = ce_score >= threshold

    return results, cross_encoder_calls


def _run_lexical_pipeline(processed_texts, threshold, progress_fn=None):
    """Run TF-IDF lexical similarity pipeline."""
    n = len(processed_texts)
    total_pairs = n * (n - 1) // 2

    if progress_fn:
        progress_fn('tfidf', 35, 'Computing TF-IDF lexical similarity...')

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    sim_matrix = sk_cosine(tfidf_matrix)

    if progress_fn:
        progress_fn('comparing', 60, f'Evaluating {total_pairs} pairs...')

    results = []
    for i, j in combinations(range(n), 2):
        score = round(float(sim_matrix[i][j]), 4)
        results.append({
            'i': i,
            'j': j,
            'score': score,
            'is_flagged': score >= threshold,
            'mode': 'lexical',
        })

    return results, 0


# ── Database Comparison (Class Submissions) ──

def run_comparison(class_id, mode='semantic', threshold=None):
    """Run similarity comparison pipeline for all submissions in a class.

    Safety:
      - Per-class mutex lock prevents duplicate concurrent runs.
      - Savepoint rollback ensures existing results are preserved if an error occurs.
    """
    from app.models import db, Submission, Result

    lock = _get_class_lock(class_id)
    if not lock.acquire(blocking=False):
        return {
            'error': 'A comparison is already running for this class. Please wait for it to finish.'
        }

    effective_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
    _update_progress(class_id, 'init', 0, f'Starting similarity engine ({ENGINE_VERSION})...')

    try:
        result = _run_comparison_inner(class_id, db, Submission, Result, mode, effective_threshold)
        if 'error' in result:
            _finish_progress(class_id, 'error', result['error'])
        else:
            _finish_progress(
                class_id, 'done',
                f"Complete! {result['flagged_pairs']} flagged out of {result['total_pairs']} pairs."
            )
        return result
    except Exception as e:
        db.session.rollback()
        logger.exception(f'Comparison failed for class {class_id}: {e}')
        msg = f'Comparison failed: {str(e)}. Previous results were preserved.'
        _finish_progress(class_id, 'error', msg)
        return {'error': msg}
    finally:
        lock.release()


def _run_comparison_inner(class_id, db, Submission, Result, mode, threshold):
    """Inner comparison logic for class submissions."""
    # 1. Fetch submissions ordered by upload time (earlier upload has priority)
    submissions = Submission.query.filter_by(class_id=class_id)\
        .order_by(Submission.uploaded_at.asc()).all()

    if len(submissions) < 2:
        return {'error': 'Need at least 2 submissions to compare.'}

    savepoint = db.session.begin_nested()
    try:
        _update_progress(class_id, 'preparing', 5, 'Clearing previous results...')
        Result.query.filter_by(class_id=class_id).delete()

        for sub in submissions:
            sub.status = 'accepted'

        # 2. Preprocess texts with TECH_STOPWORDS
        _update_progress(class_id, 'preprocessing', 15,
                         f'Preprocessing {len(submissions)} abstracts with tech stopwords filtering...')

        raw_texts = [sub.abstract_text or '' for sub in submissions]
        processed_texts = [preprocess_text(t) or '[empty document]' for t in raw_texts]

        readable_count = sum(1 for t in processed_texts if t != '[empty document]')
        if readable_count < 2:
            savepoint.rollback()
            return {'error': 'Need at least 2 abstracts with readable text to compare.'}

        # 3. Execute analysis pipeline
        def progress_cb(step, pct, msg):
            _update_progress(class_id, step, pct, msg)

        if mode == 'lexical':
            pair_results, ce_calls = _run_lexical_pipeline(processed_texts, threshold, progress_cb)
        else:
            pair_results, ce_calls = _run_semantic_pipeline(processed_texts, threshold, progress_cb)

        # 4. Map results to database records and determine submission rejections
        _update_progress(class_id, 'saving', 92, 'Saving results to database...')

        rejected_ids = set()
        db_results = []
        flagged_count = 0

        for r in pair_results:
            sub_1 = submissions[r['i']]
            sub_2 = submissions[r['j']]
            is_flagged = r['is_flagged']

            if is_flagged:
                flagged_count += 1
                # The later submission (sub_2) is rejected; the earlier submission (sub_1) remains accepted
                rejected_ids.add(sub_2.id)

            db_results.append(Result(
                class_id=class_id,
                submission_1_id=sub_1.id,
                submission_2_id=sub_2.id,
                similarity_score=r['score'],
                is_similar=is_flagged,
            ))

        db.session.add_all(db_results)

        # 5. Update submission statuses
        _update_progress(class_id, 'finalizing', 96, 'Updating submission statuses...')
        for sub in submissions:
            sub.status = 'rejected' if sub.id in rejected_ids else 'accepted'

        savepoint.commit()

    except Exception:
        savepoint.rollback()
        raise

    db.session.commit()

    total_pairs = len(pair_results)
    return {
        'total_submissions': len(submissions),
        'total_pairs': total_pairs,
        'flagged_pairs': flagged_count,
        'safe_pairs': total_pairs - flagged_count,
        'threshold': threshold,
        'cross_encoder_calls': ce_calls,
        'mode': mode,
        'engine_version': ENGINE_VERSION,
    }


# ── In-Memory Manual Comparison ──

def compare_texts_manual(task_id, items, mode='semantic', threshold=None):
    """Run similarity pipeline on arbitrary in-memory texts without touching the DB.

    Args:
        task_id: Unique identifier for progress polling.
        items: List of dicts: [{'label': str, 'text': str}, ...]
        mode: 'semantic' (default) or 'lexical'.
        threshold: Score cutoff for flagging (default: 0.65).
    """
    lock = _get_class_lock(task_id)
    if not lock.acquire(blocking=False):
        return {'error': 'A comparison is already running for this task. Please wait.'}

    effective_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
    _update_progress(task_id, 'init', 0, f'Starting manual comparison ({ENGINE_VERSION})...')

    try:
        result = _compare_texts_manual_inner(task_id, items, mode, effective_threshold)
        if 'error' in result:
            _finish_progress(task_id, 'error', result['error'])
        else:
            _finish_progress(
                task_id, 'done',
                f"Complete! {result['flagged']} flagged out of {result['total']} pairs."
            )
        return result
    except Exception as e:
        logger.exception(f'Manual comparison failed (task {task_id}): {e}')
        msg = f'Comparison failed: {str(e)}'
        _finish_progress(task_id, 'error', msg)
        return {'error': msg}
    finally:
        lock.release()


def _compare_texts_manual_inner(task_id, items, mode, threshold):
    """Inner logic for in-memory text comparison."""
    if len(items) < 2:
        return {'error': 'Need at least 2 files to compare.'}

    labels = [item.get('label') or f'Document {i+1}' for i, item in enumerate(items)]
    raw_texts = [item.get('text') or '' for item in items]

    _update_progress(task_id, 'preprocessing', 15,
                     f'Preprocessing {len(items)} texts with tech stopwords filtering...')

    processed_texts = [preprocess_text(t) or '[empty document]' for t in raw_texts]

    readable_count = sum(1 for t in processed_texts if t != '[empty document]')
    if readable_count < 2:
        return {'error': 'Need at least 2 files with readable text content.'}

    def progress_cb(step, pct, msg):
        _update_progress(task_id, step, pct, msg)

    if mode == 'lexical':
        pair_results, _ = _run_lexical_pipeline(processed_texts, threshold, progress_cb)
    else:
        pair_results, _ = _run_semantic_pipeline(processed_texts, threshold, progress_cb)

    _update_progress(task_id, 'finalizing', 98, 'Finalizing results...')

    pairs = []
    flagged = 0
    for r in pair_results:
        is_sim = r['is_flagged']
        if is_sim:
            flagged += 1

        pairs.append({
            'label_a': labels[r['i']],
            'label_b': labels[r['j']],
            'score': r['score'],
            'is_similar': is_sim,
        })

    # Sort: flagged pairs first, then descending by score
    pairs.sort(key=lambda p: (not p['is_similar'], -p['score']))

    return {
        'pairs': pairs,
        'flagged': flagged,
        'total': len(pairs),
        'threshold': threshold,
        'mode': mode,
        'engine_version': ENGINE_VERSION,
    }
