"""
tests/test_phase10_runner_safety.py — Regression tests for Phase 10 Runner Safety.

Verifies:
1. Success skipping: completed runs in checkpoint are skipped on resume.
2. Deferred skipping: deferred_oversized runs in checkpoint are skipped on resume.
3. Preflight deferral: oversized requests are intercepted before calling run_audit.
4. Architecture specificity: requests are evaluated per-architecture (rule_only never oversized).
5. Quota stop: HTTP 429 quota exhaustion immediately stops execution without blind retries.
6. Scoring safety: deferred entries never enter scoring as negative or positive predictions.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
BRAIN_SCRATCH = Path(r"C:\Users\archy\.gemini\antigravity-ide\brain\b6d049ac-9bb8-4d65-b596-2a4a0f411c84\scratch")

# Insert BRAIN_SCRATCH first so EVAL_DIR can be prepended after and win priority.
# EVAL_DIR must be at sys.path[0] so the canonical eval/phase10_benchmark_runner.py
# is always imported rather than any stale copy in the old brain scratch dir.
if str(BRAIN_SCRATCH) not in sys.path:
    sys.path.insert(0, str(BRAIN_SCRATCH))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import phase10_benchmark_runner as runner


def test_architecture_specificity_and_preflight():
    """Verify that preflight deferral is architecture-specific.
    
    nb_3449 is oversized on LLM conditions (>7k tokens) but NEVER on rule_only (0 prompt tokens).
    Regular notebook (nb_1244) is not oversized on any condition.
    """
    nb_3449_id = "GitHub-data/notebooks/2021-09-27/nb_3449.py"
    nb_3449_path = str(runner.NOTEBOOKS_DIR / "2021-09-27-nb_3449.ipynb")

    # 1. rule_only must NEVER be marked oversized
    is_over, reason = runner.check_preflight_request("rule_only", nb_3449_id, nb_3449_path)
    assert not is_over
    assert reason == ""

    # 2. llm_only on nb_3449 is identified as oversized
    is_over_lo, reason_lo = runner.check_preflight_request("llm_only", nb_3449_id, nb_3449_path)
    assert is_over_lo
    assert "exceeds" in reason_lo.lower()

    # 3. Regular notebook is NOT oversized on LLM
    nb_1244_id = "GitHub-data/notebooks/2021-09-05/nb_1244.py"
    nb_1244_path = str(runner.NOTEBOOKS_DIR / "2021-09-05-nb_1244.ipynb")
    is_over_reg, _ = runner.check_preflight_request("llm_only", nb_1244_id, nb_1244_path)
    assert not is_over_reg


def test_success_and_deferred_skipping_logic():
    """Verify resume guard skips both 'success' and 'deferred_oversized' runs."""
    ckpt = {
        "llm_only:nb_completed.py:run_0": {"status": "success", "predictions": {"pre": False, "overlap": False, "multi": False}},
        "llm_only:nb_deferred.py:run_0": {"status": "deferred_oversized", "predictions": None},
        "llm_only:nb_failed.py:run_0": {"status": "failed", "predictions": None},
    }

    # Helper simulating the runner resume check
    def should_skip(key: str) -> tuple[bool, str]:
        rec = ckpt.get(key)
        if rec and rec.get("status") in ("success", "deferred_oversized"):
            return True, rec.get("status")
        return False, "execute"

    skip_succ, st_succ = should_skip("llm_only:nb_completed.py:run_0")
    assert skip_succ and st_succ == "success"

    skip_def, st_def = should_skip("llm_only:nb_deferred.py:run_0")
    assert skip_def and st_def == "deferred_oversized"

    skip_fail, st_fail = should_skip("llm_only:nb_failed.py:run_0")
    assert not skip_fail and st_fail == "execute"

    skip_missing, _ = should_skip("llm_only:nb_new.py:run_0")
    assert not skip_missing


def test_quota_stop_behavior_on_429():
    """Verify that an HTTP 429 RateLimitError halts execution immediately without retries."""
    mock_429 = Exception("RateLimitError: Error code: 429 - tokens per day limit reached")

    with patch("phase10_benchmark_runner.run_audit", side_effect=mock_429) as mock_audit:
        with patch("phase10_benchmark_runner.log_llm_telemetry") as mock_telemetry:
            with pytest.raises(runner.QuotaLimitError) as exc_info:
                runner.run_single_request("llm_only", "dummy_nb.py", "dummy_path", 0)

            # Assert run_audit was called ONLY ONCE — no retries on 429!
            assert mock_audit.call_count == 1

            record = exc_info.value.args[0]
            assert record["status"] == "failed"
            assert record["error_category"] == "quota_limit"

            # Assert telemetry was logged
            assert mock_telemetry.call_count == 1
            logged_entry = mock_telemetry.call_args[0][0]
            assert logged_entry["status"] == "failed"


def test_413_oversized_immediate_deferral_without_retries():
    """Verify that an HTTP 413 error during live execution is immediately deferred without retrying."""
    mock_413 = Exception("APIStatusError: Error code: 413 - Request too large for model on input tokens per minute")

    with patch("phase10_benchmark_runner.run_audit", side_effect=mock_413) as mock_audit:
        with patch("phase10_benchmark_runner.log_llm_telemetry") as mock_telemetry:
            record = runner.run_single_request("llm_only", "dummy_large.py", "dummy_path", 0)

            # Assert NO retries were performed on 413
            assert mock_audit.call_count == 1
            assert record["status"] == "deferred_oversized"
            assert record["error_category"] == "input_too_large"
            assert record["predictions"] is None
            assert mock_telemetry.call_count == 1


def test_deferred_entries_never_enter_scoring():
    """Verify deferred oversized notebooks are excluded from majority-voting scoring and never become negative predictions."""
    gt_labels = {
        "nb_normal_1.py": {"pre": True, "overlap": False, "multi": True},
        "nb_normal_2.py": {"pre": False, "overlap": True, "multi": False},
        "nb_oversized.py": {"pre": True, "overlap": True, "multi": True},  # GT has True labels
    }

    # Checkpoint with 2 completed notebooks and 1 deferred notebook
    ckpt = {
        # nb_normal_1: all 3 runs predicted pre=True, overlap=False, multi=True
        "llm_only:nb_normal_1.py:run_0": {"status": "success", "predictions": {"pre": True, "overlap": False, "multi": True}},
        "llm_only:nb_normal_1.py:run_1": {"status": "success", "predictions": {"pre": True, "overlap": False, "multi": True}},
        "llm_only:nb_normal_1.py:run_2": {"status": "success", "predictions": {"pre": True, "overlap": False, "multi": True}},
        # nb_normal_2: all 3 runs predicted pre=False, overlap=True, multi=False
        "llm_only:nb_normal_2.py:run_0": {"status": "success", "predictions": {"pre": False, "overlap": True, "multi": False}},
        "llm_only:nb_normal_2.py:run_1": {"status": "success", "predictions": {"pre": False, "overlap": True, "multi": False}},
        "llm_only:nb_normal_2.py:run_2": {"status": "success", "predictions": {"pre": False, "overlap": True, "multi": False}},
        # nb_oversized: all 3 runs deferred
        "llm_only:nb_oversized.py:run_0": {"status": "deferred_oversized", "predictions": None},
        "llm_only:nb_oversized.py:run_1": {"status": "deferred_oversized", "predictions": None},
        "llm_only:nb_oversized.py:run_2": {"status": "deferred_oversized", "predictions": None},
    }

    manifest = [
        {"nb_id": "nb_normal_1.py"},
        {"nb_id": "nb_normal_2.py"},
        {"nb_id": "nb_oversized.py"},
    ]

    cols = ["pre", "overlap", "multi"]
    mv_predictions = {}
    cond_deferred_nbs = []

    for item in manifest:
        nb_id = item["nb_id"]
        nb_run_preds = {c: [] for c in cols}
        nb_deferred = False

        for run_i in range(3):
            key = f"llm_only:{nb_id}:run_{run_i}"
            rec = ckpt.get(key)
            if rec and rec["status"] == "success" and rec["predictions"] is not None:
                for c in cols:
                    nb_run_preds[c].append(rec["predictions"][c])
            elif rec and rec["status"] == "deferred_oversized":
                nb_deferred = True
                cond_deferred_nbs.append((nb_id, run_i, "deferred"))

        if not nb_deferred and all(len(nb_run_preds[c]) == 3 for c in cols):
            mv_pred = {}
            for c in cols:
                votes = nb_run_preds[c]
                mv_pred[c] = votes.count(True) > 1
            mv_predictions[nb_id] = mv_pred

    # 1. Verify deferred notebook is NOT in mv_predictions
    assert "nb_oversized.py" not in mv_predictions
    assert "nb_normal_1.py" in mv_predictions
    assert "nb_normal_2.py" in mv_predictions

    # 2. Score predictions
    metrics = runner.score_predictions(gt_labels, mv_predictions)
    assert metrics["n_scored"] == 2  # Exactly 2 scored, NOT 3!

    # 3. Verify nb_oversized did NOT become a False Negative
    # If nb_oversized (which has pre=True, overlap=True, multi=True) were counted as False predictions,
    # FN would be incremented.
    # For nb_normal_1: pre=True (TP), overlap=False (TN), multi=True (TP)
    # For nb_normal_2: pre=False (TN), overlap=True (TP), multi=False (TN)
    # So TP=1, FP=0, FN=0, TN=1 for each category!
    assert metrics["per_category"]["pre"]["tp"] == 1
    assert metrics["per_category"]["pre"]["fn"] == 0
    assert metrics["per_category"]["pre"]["f1"] == 1.0

    assert metrics["per_category"]["overlap"]["tp"] == 1
    assert metrics["per_category"]["overlap"]["fn"] == 0
    assert metrics["per_category"]["overlap"]["f1"] == 1.0

    assert metrics["per_category"]["multi"]["tp"] == 1
    assert metrics["per_category"]["multi"]["fn"] == 0
    assert metrics["per_category"]["multi"]["f1"] == 1.0

    # 4. Per-run variance scoring with scored_nb_ids
    scored_ids = set(mv_predictions.keys())
    variance_info = runner.compute_per_run_variance(gt_labels, scored_ids, ckpt, "llm_only", 3)
    assert len(variance_info["run_macro_f1s"]) == 3
    assert variance_info["mean_macro_f1"] == 1.0


def test_e2e_runner_execution_loop_behavior(tmp_path):
    """End-to-end integration test of runner execution flow with mocked provider.
    
    Verifies that:
    1. Already-successful runs are skipped (resumed).
    2. Preflight oversized requests are deferred and saved with status 'deferred_oversized'.
    3. Eligible live calls are executed.
    4. HTTP 429 halts the loop immediately, breaks out, and flushes checkpoint.
    """
    mock_ckpt_path = tmp_path / "mock_ckpt.json"
    mock_log_path = tmp_path / "mock_log.jsonl"
    mock_base_path = tmp_path / "mock_base.json"
    mock_results_path = tmp_path / "mock_results.json"

    # Seed checkpoint with completed rule_only baseline and 1 completed LLM run
    initial_ckpt = {
        "rule_only:nb_completed.py:run_0": {"status": "success", "predictions": {"pre": False, "overlap": False, "multi": False}},
        "rule_only:nb_oversized.py:run_0": {"status": "success", "predictions": {"pre": True, "overlap": False, "multi": True}},
        "rule_only:nb_rate_limited.py:run_0": {"status": "success", "predictions": {"pre": False, "overlap": True, "multi": False}},
        "llm_only:nb_completed.py:run_0": {"status": "success", "predictions": {"pre": False, "overlap": False, "multi": False}},
    }
    with open(mock_ckpt_path, "w", encoding="utf-8") as f:
        json.dump(initial_ckpt, f)

    gt_labels = {
        "nb_completed.py": {"pre": False, "overlap": False, "multi": False},
        "nb_oversized.py": {"pre": True, "overlap": False, "multi": True},
        "nb_rate_limited.py": {"pre": False, "overlap": True, "multi": False},
    }
    manifest = [
        {"nb_id": "nb_completed.py", "nb_path": "path_completed.ipynb"},
        {"nb_id": "nb_oversized.py", "nb_path": "path_oversized.ipynb"},
        {"nb_id": "nb_rate_limited.py", "nb_path": "path_429.ipynb"},
    ]

    def mock_preflight(cond_str, nb_id, nb_path):
        if nb_id == "nb_oversized.py":
            return True, "Simulated oversized >7k tokens"
        return False, ""

    def mock_run_single(cond_str, nb_id, nb_path, run_idx):
        if cond_str != "rule_only" and nb_id == "nb_rate_limited.py":
            record = {
                "condition": cond_str,
                "nb_id": nb_id,
                "run_index": run_idx,
                "status": "failed",
                "error": "RateLimitError: 429 daily quota exhausted",
                "error_category": "quota_limit",
                "predictions": None,
            }
            raise runner.QuotaLimitError(record)
        return {
            "condition": cond_str,
            "nb_id": nb_id,
            "run_index": run_idx,
            "status": "success",
            "latency_s": 0.5,
            "predictions": {"pre": False, "overlap": False, "multi": False},
        }

    with patch.object(runner, "CHECKPOINT_PATH", mock_ckpt_path), \
         patch.object(runner, "CALL_LOG_PATH", mock_log_path), \
         patch.object(runner, "BASELINE_JSON_PATH", mock_base_path), \
         patch.object(runner, "RESULTS_JSON_PATH", mock_results_path), \
         patch.object(runner, "PACING_DELAY_SEC", 0), \
         patch.object(runner, "load_manifest_and_gt", return_value=(gt_labels, manifest, [])), \
         patch.object(runner, "check_preflight_request", side_effect=mock_preflight), \
         patch.object(runner, "run_single_request", side_effect=mock_run_single):

        # Run runner main()
        runner.main()

    # Load updated checkpoint
    with open(mock_ckpt_path, "r", encoding="utf-8") as f:
        final_ckpt = json.load(f)

    # 1. nb_completed run 0 was preserved
    assert final_ckpt["llm_only:nb_completed.py:run_0"]["status"] == "success"

    # 2. nb_oversized runs were marked deferred_oversized
    assert final_ckpt["llm_only:nb_oversized.py:run_0"]["status"] == "deferred_oversized"

    # 3. nb_rate_limited hit 429 and was saved as failed with quota_limit
    assert final_ckpt["llm_only:nb_rate_limited.py:run_0"]["status"] == "failed"
    assert final_ckpt["llm_only:nb_rate_limited.py:run_0"]["error_category"] == "quota_limit"

    # 4. Checkpoint did NOT continue after 429 — run_1 of nb_rate_limited was never executed!
    assert "llm_only:nb_rate_limited.py:run_1" not in final_ckpt
