"""
Similarity Engine — Enhanced Multi-Signal Pipeline.

Improvements over the original:
  1. Upgraded model: all-mpnet-base-v2 (768-dim, much better quality)
  2. Hybrid scoring: combines semantic (embedding) + lexical (TF-IDF) + structural (n-gram Jaccard)
  3. Cross-encoder re-ranking: precise verification for borderline pairs
  4. Text preprocessing: normalization before comparison

Pipeline:
  Bi-encoder screening  →  Hybrid score  →  Cross-encoder for borderline  →  Final decision
"""

import re
import logging
import threading
from itertools import combinations

from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ── Configuration ──
SIMILARITY_THRESHOLD = 0.65       # Final decision threshold
CROSS_ENCODER_LOW = 0.50          # Below this — skip cross-encoder (clearly safe)
CROSS_ENCODER_HIGH = 0.80         # Above this — skip cross-encoder (clearly similar)

# Hybrid score weights (must sum to 1.0)
W_SEMANTIC = 0.55
W_TFIDF = 0.25
W_NGRAM = 0.20

# Lazy-loaded models (cached after first call)
_bi_encoder = None
_cross_encoder = None

# Per-class locks to prevent concurrent comparisons on the same class
_comparison_locks = {}          # {class_id: threading.Lock()}
_locks_guard = threading.Lock() # protects the _comparison_locks dict itself

# Progress tracking for each class comparison
_comparison_progress = {}       # {class_id: {step, percent, message, status}}
_progress_lock = threading.Lock()

logger = logging.getLogger(__name__)


def _update_progress(class_id, step, percent, message):
    """Update the progress state for a class comparison."""
    with _progress_lock:
        _comparison_progress[class_id] = {
            'step': step,
            'percent': percent,
            'message': message,
            'status': 'running',
        }


def _finish_progress(class_id, status='done', message='Comparison complete.'):
    """Mark the comparison as finished (done or error)."""
    with _progress_lock:
        _comparison_progress[class_id] = {
            'step': 'done' if status == 'done' else 'error',
            'percent': 100 if status == 'done' else _comparison_progress.get(class_id, {}).get('percent', 0),
            'message': message,
            'status': status,
        }


def get_progress(class_id):
    """Get the current progress for a class comparison. Returns None if no comparison is running."""
    with _progress_lock:
        return _comparison_progress.get(class_id)


# ── Model Loaders ──

def _get_bi_encoder():
    """Load the bi-encoder model (cached after first call)."""
    global _bi_encoder
    if _bi_encoder is None:
        # Improvement #1: Upgraded from all-MiniLM-L6-v2 (384-dim)
        #                  to all-mpnet-base-v2 (768-dim, higher quality)
        _bi_encoder = SentenceTransformer('all-mpnet-base-v2')
    return _bi_encoder


def _get_cross_encoder():
    """Load the cross-encoder model (cached after first call)."""
    global _cross_encoder
    if _cross_encoder is None:
        # Improvement #3: Cross-encoder reads both texts together
        #                  for much more precise similarity judgment
        _cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')
    return _cross_encoder


# ── Improvement #4: Text Preprocessing ──

def preprocess_text(text):
    """
    Normalize text before comparison.
    - Lowercase
    - Remove extra whitespace
    - Remove punctuation
    - Remove common stopwords that don't carry meaning
    """
    if not text:
        return ''

    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()          # collapse whitespace
    text = re.sub(r'[^\w\s]', '', text)                # remove punctuation

    # Remove common academic stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'this', 'that', 'these', 'those', 'it', 'its',
        'of', 'in', 'for', 'to', 'and', 'with', 'on', 'by', 'from', 'at',
        'or', 'as', 'but', 'not', 'has', 'have', 'had', 'will', 'would',
        'can', 'could', 'may', 'also', 'which', 'who', 'whom',
        'we', 'our', 'us', 'they', 'their', 'them',
    }

    words = text.split()
    words = [w for w in words if w not in stopwords]
    return ' '.join(words)


# ── Improvement #2: TF-IDF + N-gram Jaccard ──

def compute_tfidf_similarity(texts):
    """
    Compute pairwise TF-IDF cosine similarity.
    Captures lexical overlap — good for catching copy-paste.
    """
    if len(texts) < 2:
        return None

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=5000,
        stop_words='english',
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = sklearn_cosine(tfidf_matrix)
    return sim_matrix


def ngram_jaccard(text_a, text_b, n=3):
    """
    Character-level n-gram Jaccard similarity.
    Catches structural similarity even after minor edits.
    """
    if not text_a or not text_b:
        return 0.0

    a_ngrams = set(text_a[i:i+n] for i in range(len(text_a) - n + 1))
    b_ngrams = set(text_b[i:i+n] for i in range(len(text_b) - n + 1))

    if not a_ngrams or not b_ngrams:
        return 0.0

    intersection = len(a_ngrams & b_ngrams)
    union = len(a_ngrams | b_ngrams)
    return intersection / union


# ── Core: Semantic Similarity Matrix ──

def compute_semantic_matrix(texts):
    """
    Encode texts into semantic embeddings (768-dim) and compute
    pairwise cosine similarity.
    """
    if len(texts) < 2:
        return None

    model = _get_bi_encoder()
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    return sim_matrix


# ── Cross-Encoder Verification ──

def cross_encoder_score(text_a, text_b):
    """
    Use a cross-encoder to precisely score a single text pair.
    Cross-encoders read both texts together (via attention) so they
    are significantly more accurate than bi-encoders, but slower.
    Only used for borderline cases.
    """
    model = _get_cross_encoder()
    score = model.predict([(text_a, text_b)])[0]
    # stsb-roberta-base already outputs scores in 0–1 range.
    # Clamp to [0, 1] as a safety measure.
    score = float(score)
    return max(0.0, min(1.0, score))


# ── Concurrency Lock ──

def _get_class_lock(class_id):
    """Get or create a per-class lock to prevent concurrent comparisons."""
    with _locks_guard:
        if class_id not in _comparison_locks:
            _comparison_locks[class_id] = threading.Lock()
        return _comparison_locks[class_id]


# ── Main Pipeline ──

def run_comparison(class_id):
    """
    Run the enhanced similarity comparison pipeline for a class.

    Safety features:
      - Per-class lock prevents concurrent runs on the same class
      - DB savepoint + rollback ensures a crash doesn't wipe existing results

    Pipeline:
      1. Fetch all submissions (ordered by upload time)
      2. Preprocess abstract texts
      3. Compute semantic similarity matrix     (bi-encoder)
      4. Compute TF-IDF similarity matrix       (lexical)
      5. For each pair:
         a. Compute hybrid score (semantic + TF-IDF + n-gram Jaccard)
         b. If borderline → refine with cross-encoder
         c. Make final accept/reject decision
      6. Save results to DB

    Returns:
        dict with summary stats
    """
    from app.models import db, Submission, Result

    # ── Concurrency guard: only one comparison per class at a time ──
    lock = _get_class_lock(class_id)

    if not lock.acquire(blocking=False):
        return {
            'error': 'A comparison is already running for this class. '
                     'Please wait for it to finish.'
        }

    _update_progress(class_id, 'init', 0, 'Starting comparison...')

    try:
        result = _run_comparison_inner(class_id, db, Submission, Result)
        if 'error' in result:
            _finish_progress(class_id, 'error', result['error'])
        else:
            _finish_progress(class_id, 'done',
                f"Complete! {result['flagged_pairs']} flagged out of {result['total_pairs']} pairs.")
        return result
    except Exception as e:
        # ── Error recovery: rollback so previous results are preserved ──
        db.session.rollback()
        logger.exception(f'Comparison failed for class {class_id}: {e}')
        msg = f'Comparison failed: {str(e)}. Previous results were preserved.'
        _finish_progress(class_id, 'error', msg)
        return {'error': msg}
    finally:
        lock.release()


def _run_comparison_inner(class_id, db, Submission, Result):
    """Inner comparison logic, wrapped by run_comparison for safety."""

    # 1. Fetch submissions ordered by upload time (first uploaded = priority)
    submissions = Submission.query.filter_by(class_id=class_id)\
        .order_by(Submission.uploaded_at.asc()).all()

    if len(submissions) < 2:
        return {'error': 'Need at least 2 submissions to compare.'}

    # ── Use a savepoint so we can rollback without losing previous data ──
    # Save old results/statuses first, delete only after new results succeed
    savepoint = db.session.begin_nested()

    try:
        _update_progress(class_id, 'preparing', 5, 'Clearing previous results...')

        # Clear any previous results for this class
        Result.query.filter_by(class_id=class_id).delete()

        # Reset all statuses to 'accepted' initially
        for sub in submissions:
            sub.status = 'accepted'

        # 2. Preprocess texts
        _update_progress(class_id, 'preprocessing', 10,
            f'Preprocessing {len(submissions)} abstracts...')

        raw_texts = [sub.abstract_text or '' for sub in submissions]
        processed_texts = [preprocess_text(t) for t in raw_texts]

        # Filter out empty abstracts
        valid_indices = [i for i, t in enumerate(processed_texts) if t.strip()]
        if len(valid_indices) < 2:
            savepoint.rollback()
            return {'error': 'Need at least 2 non-empty abstracts to compare.'}

        # 3. Compute semantic similarity matrix (bi-encoder, 768-dim)
        _update_progress(class_id, 'semantic', 20,
            'Computing semantic embeddings (bi-encoder)...')

        semantic_matrix = compute_semantic_matrix(processed_texts)
        if semantic_matrix is None:
            savepoint.rollback()
            return {'error': 'Could not compute semantic similarity.'}

        # 4. Compute TF-IDF similarity matrix (lexical)
        _update_progress(class_id, 'tfidf', 40,
            'Computing TF-IDF lexical similarity...')

        tfidf_matrix = compute_tfidf_similarity(processed_texts)

        # 5. Pairwise comparison with hybrid scoring
        n = len(submissions)
        expected_pairs = n * (n - 1) // 2
        total_pairs = 0
        flagged_pairs = 0
        cross_encoder_calls = 0
        results = []
        rejected_ids = set()

        _update_progress(class_id, 'comparing', 50,
            f'Comparing {expected_pairs} pairs (hybrid scoring)...')

        for i, j in combinations(range(n), 2):
            # ── Hybrid Score ──
            semantic_score = float(semantic_matrix[i][j])

            tfidf_score = float(tfidf_matrix[i][j]) if tfidf_matrix is not None else 0.0

            jaccard_score = ngram_jaccard(processed_texts[i], processed_texts[j])

            hybrid_score = (
                W_SEMANTIC * semantic_score +
                W_TFIDF * tfidf_score +
                W_NGRAM * jaccard_score
            )

            # ── Cross-encoder for borderline cases ──
            final_score = hybrid_score

            if CROSS_ENCODER_LOW <= hybrid_score <= CROSS_ENCODER_HIGH:
                # Borderline — use cross-encoder for precise judgment
                ce_score = cross_encoder_score(raw_texts[i], raw_texts[j])
                # Blend: give cross-encoder 60% weight in borderline zone
                final_score = 0.4 * hybrid_score + 0.6 * ce_score
                cross_encoder_calls += 1

            # ── Decision ──
            is_similar = final_score >= SIMILARITY_THRESHOLD

            if is_similar:
                flagged_pairs += 1
                # Later submission (j) gets rejected — i was submitted first
                rejected_ids.add(submissions[j].id)

            total_pairs += 1

            # Update progress every 10 pairs or on last pair
            if total_pairs % 10 == 0 or total_pairs == expected_pairs:
                pair_progress = 50 + int((total_pairs / expected_pairs) * 40)  # 50-90%
                _update_progress(class_id, 'comparing', pair_progress,
                    f'Comparing pairs... {total_pairs}/{expected_pairs} '
                    f'({flagged_pairs} flagged)')

            # Save result to DB (store the final blended score)
            result = Result(
                class_id=class_id,
                submission_1_id=submissions[i].id,
                submission_2_id=submissions[j].id,
                similarity_score=round(final_score, 4),
                is_similar=is_similar,
            )
            results.append(result)

        _update_progress(class_id, 'saving', 92, 'Saving results to database...')

        db.session.add_all(results)

        # 6. Update submission statuses
        _update_progress(class_id, 'finalizing', 96, 'Updating submission statuses...')

        for sub in submissions:
            sub.status = 'rejected' if sub.id in rejected_ids else 'accepted'

        # Everything succeeded — commit the savepoint
        savepoint.commit()

    except Exception:
        # Something went wrong — rollback to savepoint (preserves old results)
        savepoint.rollback()
        raise  # re-raise so the outer try/except in run_comparison handles it

    # Final commit to make everything permanent
    db.session.commit()

    return {
        'total_submissions': n,
        'total_pairs': total_pairs,
        'flagged_pairs': flagged_pairs,
        'safe_pairs': total_pairs - flagged_pairs,
        'threshold': SIMILARITY_THRESHOLD,
        'cross_encoder_calls': cross_encoder_calls,
    }


# ── Manual (DB-free) Comparison ──

def compare_texts_manual(task_id, items):
    """
    Run the full hybrid similarity pipeline on arbitrary in-memory texts.
    No database reads or writes are performed.

    Args:
        task_id:  Any hashable key used for progress tracking (e.g. a UUID string).
        items:    List of dicts: [{"label": str, "text": str}, ...]
                  Must contain at least 2 items with non-empty text.

    Returns:
        On success:
            {
                "pairs": [
                    {
                        "label_a": str,
                        "label_b": str,
                        "score": float,        # 0.0 – 1.0
                        "is_similar": bool,
                    }, ...
                ],
                "flagged": int,
                "total": int,
                "threshold": float,
            }
        On failure:
            {"error": str}
    """
    lock = _get_class_lock(task_id)
    if not lock.acquire(blocking=False):
        return {'error': 'A comparison is already running for this task. Please wait.'}

    _update_progress(task_id, 'init', 0, 'Starting manual comparison...')

    try:
        result = _compare_texts_manual_inner(task_id, items)
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


def _compare_texts_manual_inner(task_id, items):
    """Inner logic for compare_texts_manual — no DB, pure in-memory."""

    if len(items) < 2:
        return {'error': 'Need at least 2 files to compare.'}

    labels = [item['label'] for item in items]
    raw_texts = [item['text'] for item in items]

    # 1. Preprocess
    _update_progress(task_id, 'preprocessing', 10,
                     f'Preprocessing {len(items)} texts...')

    processed_texts = [preprocess_text(t) for t in raw_texts]

    valid_indices = [i for i, t in enumerate(processed_texts) if t.strip()]
    if len(valid_indices) < 2:
        return {'error': 'Need at least 2 files with readable text content.'}

    # 2. Semantic embeddings
    _update_progress(task_id, 'semantic', 20,
                     'Computing semantic embeddings (bi-encoder)...')

    semantic_matrix = compute_semantic_matrix(processed_texts)
    if semantic_matrix is None:
        return {'error': 'Could not compute semantic similarity.'}

    # 3. TF-IDF
    _update_progress(task_id, 'tfidf', 40,
                     'Computing TF-IDF lexical similarity...')

    tfidf_matrix = compute_tfidf_similarity(processed_texts)

    # 4. Pairwise scoring
    n = len(items)
    expected_pairs = n * (n - 1) // 2
    total_pairs = 0
    flagged = 0
    cross_encoder_calls = 0
    pairs = []

    _update_progress(task_id, 'comparing', 50,
                     f'Comparing {expected_pairs} pairs (hybrid scoring)...')

    for i, j in combinations(range(n), 2):
        semantic_score = float(semantic_matrix[i][j])
        tfidf_score = float(tfidf_matrix[i][j]) if tfidf_matrix is not None else 0.0
        jaccard_score = ngram_jaccard(processed_texts[i], processed_texts[j])

        hybrid_score = (
            W_SEMANTIC * semantic_score +
            W_TFIDF    * tfidf_score    +
            W_NGRAM    * jaccard_score
        )

        final_score = hybrid_score
        if CROSS_ENCODER_LOW <= hybrid_score <= CROSS_ENCODER_HIGH:
            ce_score = cross_encoder_score(raw_texts[i], raw_texts[j])
            final_score = 0.4 * hybrid_score + 0.6 * ce_score
            cross_encoder_calls += 1

        is_similar = final_score >= SIMILARITY_THRESHOLD
        if is_similar:
            flagged += 1

        total_pairs += 1

        pairs.append({
            'label_a': labels[i],
            'label_b': labels[j],
            'score': round(final_score, 4),
            'is_similar': is_similar,
        })

        if total_pairs % 10 == 0 or total_pairs == expected_pairs:
            pair_progress = 50 + int((total_pairs / expected_pairs) * 40)
            _update_progress(task_id, 'comparing', pair_progress,
                             f'Comparing pairs… {total_pairs}/{expected_pairs} '
                             f'({flagged} flagged)')

    _update_progress(task_id, 'finalizing', 98, 'Finalising results...')

    # Sort: flagged first, then by score desc
    pairs.sort(key=lambda p: (not p['is_similar'], -p['score']))

    return {
        'pairs': pairs,
        'flagged': flagged,
        'total': total_pairs,
        'threshold': SIMILARITY_THRESHOLD,
    }

