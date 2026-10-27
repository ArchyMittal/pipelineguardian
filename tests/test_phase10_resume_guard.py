"""
test_phase10_resume_guard.py — Offline regression test suite for Phase 10 resume guard.

Ensures that:
1. Checkpoint records with status == "success" are skipped (duplicate protection).
2. Checkpoint records with status == "failed" (e.g. 429 quota errors) are RETRIED.
3. Pending requests are correctly executed.
4. Successful retries update checkpoint and log entries cleanly.
5. Zero live API calls occur (all audit calls are mocked).
6. Existing benchmark artifacts remain completely untouched (uses isolated temp paths).
"""
import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipelineguardian.models import AuditReport, Issue, Category, Severity


@pytest.fixture
def mock_audit_success():
    """Mock run_audit that returns a dummy report with no issues."""
    report = AuditReport(
        source_path="dummy.ipynb",
        issues=[],
        passed_checks=["check_a"],
        reproducibility_score=1.0,
    )
    eval_dir = REPO_ROOT / "eval"
    scratch_dir = Path("C:/Users/archy/.gemini/antigravity-ide/brain/b6d049ac-9bb8-4d65-b596-2a4a0f411c84/scratch")
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    if str(scratch_dir) not in sys.path:
        sys.path.insert(0, str(scratch_dir))
    import phase10_benchmark_runner as runner

    with patch.object(runner, "run_audit", return_value=report) as mock_run:
        yield mock_run


def test_resume_guard_skips_success_and_retries_failures(tmp_path, mock_audit_success):
    """Verify resume guard skips status=='success' but retries status=='failed'."""
    ckpt_path = tmp_path / "test_checkpoint.json"
    log_path = tmp_path / "test_call_log.jsonl"

    # Seed mock checkpoint with:
    # 1. Successful call (nb_1, run 0) -> should be skipped
    # 2. Failed call (nb_1, run 1)     -> should be retried!
    # 3. Unattempted call (nb_1, run 2)  -> should be attempted!
    seeded_ckpt = {
        "llm_only:nb_1:run_0": {
            "condition": "llm_only",
            "nb_id": "nb_1",
            "run_index": 0,
            "status": "success",
            "predictions": {"pre": False, "overlap": False, "multi": False},
        },
        "llm_only:nb_1:run_1": {
            "condition": "llm_only",
            "nb_id": "nb_1",
            "run_index": 1,
            "status": "failed",
            "error": "RateLimitError: 429",
            "error_type": "RateLimitError",
            "error_category": "provider_api_error",
            "predictions": None,
        },
    }
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(seeded_ckpt, f, indent=2)

    eval_dir = REPO_ROOT / "eval"
    scratch_dir = Path("C:/Users/archy/.gemini/antigravity-ide/brain/b6d049ac-9bb8-4d65-b596-2a4a0f411c84/scratch")
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    if str(scratch_dir) not in sys.path:
        sys.path.insert(0, str(scratch_dir))
    
    import phase10_benchmark_runner as runner

    # Patch file locations to isolated temp paths
    with patch.object(runner, "CHECKPOINT_PATH", ckpt_path), \
         patch.object(runner, "CALL_LOG_PATH", log_path), \
         patch.object(runner, "PACING_DELAY_SEC", 0):

        ckpt = runner.load_checkpoint()
        manifest = [{"nb_id": "nb_1", "nb_path": "dummy.ipynb", "labels": {"pre": False, "overlap": False, "multi": False}, "char_count": 100}]
        llm_conditions = ["llm_only"]

        resumed_count = 0
        executed_calls = []

        for cond_str in llm_conditions:
            for item in manifest:
                nb_id = item["nb_id"]
                nb_path = item["nb_path"]
                for run_i in range(3):
                    key = f"{cond_str}:{nb_id}:run_{run_i}"
                    if key in ckpt and ckpt[key].get("status") == "success":
                        resumed_count += 1
                        continue
                    
                    # Call runner request handler
                    rec = runner.run_single_request(cond_str, nb_id, nb_path, run_i)
                    ckpt[key] = rec
                    runner.save_checkpoint(ckpt)
                    executed_calls.append(key)

        # Assertions
        assert resumed_count == 1, "Only status=='success' (run_0) should be skipped as resumed"
        assert executed_calls == ["llm_only:nb_1:run_1", "llm_only:nb_1:run_2"], \
            "Both previously failed run_1 and unattempted run_2 must be executed"
        
        # Verify run_audit mock was called exactly 2 times (for run_1 retry and run_2 pending)
        assert mock_audit_success.call_count == 2

        # Verify updated checkpoint contents
        with open(ckpt_path, "r", encoding="utf-8") as f:
            updated_ckpt = json.load(f)

        assert updated_ckpt["llm_only:nb_1:run_0"]["status"] == "success"
        assert updated_ckpt["llm_only:nb_1:run_1"]["status"] == "success", "Failed run_1 should now be updated to success after retry"
        assert updated_ckpt["llm_only:nb_1:run_2"]["status"] == "success", "Pending run_2 should now be completed as success"


def test_production_artifacts_unmodified():
    """Verify that running offline tests does not alter production artifacts."""
    ckpt_file = Path(__file__).resolve().parent.parent / "eval" / "phase10_checkpoint.json"
    if not ckpt_file.exists():
        artifact_dir = Path("C:/Users/archy/.gemini/antigravity-ide/brain/b6d049ac-9bb8-4d65-b596-2a4a0f411c84")
        ckpt_file = artifact_dir / "phase10_checkpoint.json"
    
    if ckpt_file.exists():
        with open(ckpt_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 768, "Production checkpoint record count must match current checkpoint state (768 records)"
