"""
test_yang_eval.py — unit tests per §9 coverage.

Verifies:
- load_ground_truth() binarization rule: 'Y', 'Y (mean)', 'N', blank → correct booleans
- 'model' column is never read
- score_condition() accumulates TP/FP/FN correctly on a tiny inline fixture
"""

import csv
import io
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from eval.yang_eval import load_ground_truth


# ---------------------------------------------------------------------------
# Binarization rule tests (§6.2 LOCKED)
# ---------------------------------------------------------------------------

def _write_temp_csv(rows: list[dict]) -> str:
    """Write rows to a temp CSV and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    )
    fieldnames = ["nb", "model", "pre", "overlap", "multi"]
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    tmp.close()
    return tmp.name


def test_binarization_y_is_true():
    path = _write_temp_csv([
        {"nb": "nb1.py", "model": "Y", "pre": "Y", "overlap": "Y", "multi": "Y"},
    ])
    gt = load_ground_truth(path)
    assert gt["nb1.py"]["pre"] is True
    assert gt["nb1.py"]["overlap"] is True
    assert gt["nb1.py"]["multi"] is True
    os.unlink(path)


def test_binarization_y_mean_is_true():
    """Qualitative suffix 'Y (mean)' must binarize to True."""
    path = _write_temp_csv([
        {"nb": "nb2.py", "model": "Y", "pre": "Y (mean)", "overlap": "N", "multi": ""},
    ])
    gt = load_ground_truth(path)
    assert gt["nb2.py"]["pre"] is True
    os.unlink(path)


def test_binarization_y_minmaxscaler_is_true():
    """Qualitative suffix 'Y (MinMaxScaler)' must binarize to True."""
    path = _write_temp_csv([
        {"nb": "nb3.py", "model": "Y", "pre": "Y (MinMaxScaler)", "overlap": "N", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    assert gt["nb3.py"]["pre"] is True
    os.unlink(path)


def test_binarization_n_is_false():
    path = _write_temp_csv([
        {"nb": "nb4.py", "model": "Y", "pre": "N", "overlap": "N", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    assert gt["nb4.py"]["pre"] is False
    assert gt["nb4.py"]["overlap"] is False
    assert gt["nb4.py"]["multi"] is False
    os.unlink(path)


def test_binarization_blank_is_false():
    """Blank cells binarize to False."""
    path = _write_temp_csv([
        {"nb": "nb5.py", "model": "N", "pre": "", "overlap": "", "multi": ""},
    ])
    gt = load_ground_truth(path)
    assert gt["nb5.py"]["pre"] is False
    assert gt["nb5.py"]["overlap"] is False
    assert gt["nb5.py"]["multi"] is False
    os.unlink(path)


def test_model_column_never_read():
    """The 'model' column must never appear in the returned dict."""
    path = _write_temp_csv([
        {"nb": "nb6.py", "model": "Y", "pre": "Y", "overlap": "N", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    assert "model" not in gt["nb6.py"], (
        "'model' column must never be returned — it is an annotation flag, not a label"
    )
    os.unlink(path)


def test_multiple_rows_loaded_correctly():
    path = _write_temp_csv([
        {"nb": "nb1.py", "model": "Y", "pre": "Y (mean)", "overlap": "N", "multi": "Y"},
        {"nb": "nb2.py", "model": "N", "pre": "", "overlap": "", "multi": ""},
        {"nb": "nb3.py", "model": "Y", "pre": "Y", "overlap": "Y", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    assert len(gt) == 3
    assert gt["nb1.py"] == {"pre": True, "overlap": False, "multi": True}
    assert gt["nb2.py"] == {"pre": False, "overlap": False, "multi": False}
    assert gt["nb3.py"] == {"pre": True, "overlap": True, "multi": False}
    os.unlink(path)


def test_case_insensitive_y_detection():
    """Binarization should handle uppercase Y variants — startswith('Y') is LOCKED."""
    path = _write_temp_csv([
        {"nb": "nb7.py", "model": "Y", "pre": "y", "overlap": "Y (oversampling)", "multi": "N"},
    ])
    gt = load_ground_truth(path)
    # 'y'.strip().upper().startswith('Y') → True
    assert gt["nb7.py"]["pre"] is True
    assert gt["nb7.py"]["overlap"] is True
    os.unlink(path)


# ---------------------------------------------------------------------------
# McNemar's test & architecture comparison tests
# ---------------------------------------------------------------------------

def test_mcnemar_test_basic():
    from eval.yang_eval import mcnemar_test
    # Condition A: [T, F], Condition B: [F, T], GT: [T, T]
    # A correct: [True, False], B correct: [False, True]
    # n01 = 1 (A wrong, B right), n10 = 1 (A right, B wrong)
    res = mcnemar_test([True, False], [False, True], [True, True])
    assert res["n01"] == 1
    assert res["n10"] == 1
    assert res["p_value"] == 1.0
    assert "statistic" in res


def test_mcnemar_test_zero_discordant():
    from eval.yang_eval import mcnemar_test
    # Both identical predictions
    res = mcnemar_test([True, False], [True, False], [True, True])
    assert res["n01"] == 0
    assert res["n10"] == 0
    assert res["p_value"] == 1.0
    assert res["statistic"] == 0.0


def test_compare_architectures_helper():
    from eval.yang_eval import compare_architectures
    gt = {
        "nb1.py": {"pre": True, "overlap": False, "multi": False},
        "nb2.py": {"pre": False, "overlap": True, "multi": False},
    }
    res_a = {
        "condition": "rule_only",
        "per_notebook": [
            {"nb_id": "nb1.py", "pre_pred": True, "overlap_pred": False, "multi_pred": False},
            {"nb_id": "nb2.py", "pre_pred": False, "overlap_pred": False, "multi_pred": False},
        ]
    }
    res_b = {
        "condition": "hybrid",
        "per_notebook": [
            {"nb_id": "nb1.py", "pre_pred": True, "overlap_pred": False, "multi_pred": False},
            {"nb_id": "nb2.py", "pre_pred": False, "overlap_pred": True, "multi_pred": False},
        ]
    }
    comps = compare_architectures({"rule_only": res_a, "hybrid": res_b}, gt)
    assert "rule_only_vs_hybrid" in comps
    assert "overlap" in comps["rule_only_vs_hybrid"]
    assert comps["rule_only_vs_hybrid"]["overlap"]["n01"] == 1
    assert comps["rule_only_vs_hybrid"]["overlap"]["n10"] == 0


def test_failed_llm_audit_returns_none_and_logs_failure(tmp_path, monkeypatch):
    import json
    import eval.yang_eval as yang_eval

    log_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(yang_eval, "LLM_CALL_LOG_PATH", str(log_path))
    monkeypatch.setattr(
        yang_eval,
        "run_audit",
        MagicMock(side_effect=RuntimeError("simulated API failure")),
    )

    result = yang_eval._predict_notebook(
        "sample.ipynb",
        yang_eval.AuditCondition.LLM_ONLY,
        run_index=0,
    )

    assert result is None

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["status"] == "failed"
    assert entry["run_index"] == 0
    assert "simulated API failure" in entry["error"]


def test_score_condition_excludes_notebook_with_failed_run(monkeypatch):
    import eval.yang_eval as yang_eval

    ground_truth = {
        "good.ipynb": {"pre": True, "overlap": False, "multi": False},
        "failed.ipynb": {"pre": True, "overlap": False, "multi": False},
    }

    monkeypatch.setattr(
        yang_eval,
        "_resolve_notebook_path",
        lambda nb_id, notebooks_dir: nb_id,
    )

    def fake_predict(nb_path, condition, llm=None, run_index=0):
        if nb_path == "failed.ipynb" and run_index == 1:
            return None
        return {"pre": True, "overlap": False, "multi": False}

    monkeypatch.setattr(yang_eval, "_predict_notebook", fake_predict)

    result = yang_eval.score_condition(
        condition=yang_eval.AuditCondition.LLM_ONLY,
        ground_truth=ground_truth,
        runs=2,
    )

    assert [row["nb_id"] for row in result["per_notebook"]] == ["good.ipynb"]
    assert result["n_scored"] == 1
    assert result["n_failed_notebooks"] == 1
    assert result["n_failed_runs"] == 1
    assert result["n_skipped"] == 1
    assert result["failed_notebooks"] == ["failed.ipynb"]


# ---------------------------------------------------------------------------
# Rigorous Phase 1 Regression Tests (§7 coverage)
# ---------------------------------------------------------------------------

def test_successful_audit_no_issues_gives_genuine_false(tmp_path, monkeypatch):
    """1. Successful audit with no issues gives genuine False predictions."""
    import json
    import eval.yang_eval as yang_eval
    from pipelineguardian.models import AuditReport

    log_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(yang_eval, "LLM_CALL_LOG_PATH", str(log_path))
    monkeypatch.setattr(
        yang_eval,
        "run_audit",
        MagicMock(return_value=AuditReport(source_path="clean_nb.ipynb", issues=[])),
    )

    pred = yang_eval._predict_notebook(
        "clean_nb.ipynb",
        yang_eval.AuditCondition.LLM_ONLY,
        run_index=0,
    )

    assert pred == {"pre": False, "overlap": False, "multi": False}

    log_text = log_path.read_text(encoding="utf-8").strip()
    entry = json.loads(log_text)
    assert entry["status"] == "success"
    assert entry["run_index"] == 0
    assert "error" not in entry


def test_api_failure_returns_explicit_failure_not_negative_predictions(tmp_path, monkeypatch):
    """2. API failure returns explicit failure (None), never fabricated False predictions."""
    import eval.yang_eval as yang_eval

    log_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(yang_eval, "LLM_CALL_LOG_PATH", str(log_path))
    monkeypatch.setattr(
        yang_eval,
        "run_audit",
        MagicMock(side_effect=ConnectionError("Groq 503 Service Unavailable")),
    )

    result = yang_eval._predict_notebook(
        "api_fail.ipynb",
        yang_eval.AuditCondition.LLM_ONLY,
        run_index=1,
    )

    assert result is None
    assert result != {"pre": False, "overlap": False, "multi": False}


def test_failed_attempt_logging_metadata_and_secret_redaction(tmp_path, monkeypatch):
    """3. Failed attempt logging records status, error type, category, run index, latency, and redacts secrets."""
    import json
    import eval.yang_eval as yang_eval

    log_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(yang_eval, "LLM_CALL_LOG_PATH", str(log_path))
    monkeypatch.setenv("GROQ_API_KEY", "super_secret_groq_key_999")

    # Simulate an error with leaked Groq key and Bearer auth token
    error_with_secrets = Exception(
        "Groq API error with key gsk_1234567890abcdef1234567890 and Bearer secret_token_xyz and super_secret_groq_key_999"
    )
    monkeypatch.setattr(
        yang_eval,
        "run_audit",
        MagicMock(side_effect=error_with_secrets),
    )

    result = yang_eval._predict_notebook(
        "secret_test.ipynb",
        yang_eval.AuditCondition.LLM_ONLY,
        run_index=2,
    )

    assert result is None
    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["status"] == "failed"
    assert entry["run_index"] == 2
    assert entry["error_type"] == "Exception"
    assert entry["error_category"] == "provider_api_error"
    assert entry["latency_s"] >= 0.0

    # Ensure secrets are sanitized and not leaked
    assert "gsk_1234567890abcdef1234567890" not in entry["error"]
    assert "gsk_[REDACTED]" in entry["error"]
    assert "Bearer secret_token_xyz" not in entry["error"]
    assert "Bearer [REDACTED]" in entry["error"]
    assert "super_secret_groq_key_999" not in entry["error"]
    assert "[REDACTED_API_KEY]" in entry["error"]
    assert len(entry["error"]) <= 500


def test_notebook_with_all_requested_runs_successful_is_included(monkeypatch):
    """5. A notebook with all requested runs successful is included in scoring."""
    import eval.yang_eval as yang_eval

    ground_truth = {
        "nb_success.ipynb": {"pre": True, "overlap": False, "multi": False},
    }

    monkeypatch.setattr(
        yang_eval,
        "_resolve_notebook_path",
        lambda nb_id, notebooks_dir: nb_id,
    )

    call_count = 0
    def fake_predict(nb_path, condition, llm=None, run_index=0):
        nonlocal call_count
        call_count += 1
        return {"pre": True, "overlap": False, "multi": False}

    monkeypatch.setattr(yang_eval, "_predict_notebook", fake_predict)

    result = yang_eval.score_condition(
        condition=yang_eval.AuditCondition.LLM_ONLY,
        ground_truth=ground_truth,
        runs=3,
    )

    assert call_count == 3
    assert result["n_scored"] == 1
    assert result["n_skipped"] == 0
    assert result["n_failed_notebooks"] == 0
    assert result["n_successful_runs"] == 3
    assert result["n_failed_runs"] == 0
    assert len(result["per_notebook"]) == 1
    assert result["per_notebook"][0]["nb_id"] == "nb_success.ipynb"


def test_counters_and_denominators_strictly_consistent(monkeypatch):
    """6. Successful, failed, missing, requested, and completed run counters are consistent."""
    import eval.yang_eval as yang_eval

    ground_truth = {
        "nb_missing.ipynb": {"pre": True, "overlap": False, "multi": False},
        "nb_all_pass.ipynb": {"pre": True, "overlap": False, "multi": False},
        "nb_one_fail.ipynb": {"pre": True, "overlap": False, "multi": False},
        "nb_all_fail.ipynb": {"pre": True, "overlap": False, "multi": False},
    }

    def fake_resolve(nb_id, notebooks_dir):
        if nb_id == "nb_missing.ipynb":
            return None
        return nb_id

    monkeypatch.setattr(yang_eval, "_resolve_notebook_path", fake_resolve)

    def fake_predict(nb_path, condition, llm=None, run_index=0):
        if nb_path == "nb_all_pass.ipynb":
            return {"pre": True, "overlap": False, "multi": False}
        elif nb_path == "nb_one_fail.ipynb":
            return None if run_index == 1 else {"pre": True, "overlap": False, "multi": False}
        elif nb_path == "nb_all_fail.ipynb":
            return None
        return None

    monkeypatch.setattr(yang_eval, "_predict_notebook", fake_predict)

    result = yang_eval.score_condition(
        condition=yang_eval.AuditCondition.LLM_ONLY,
        ground_truth=ground_truth,
        runs=3,
    )

    # Notebook level counters
    assert result["n_requested_notebooks"] == 4
    assert result["n_missing"] == 1
    assert result["n_failed_notebooks"] == 2  # nb_one_fail and nb_all_fail
    assert result["n_scored"] == 1           # nb_all_pass
    assert result["n_skipped"] == 3          # n_missing + n_failed_notebooks

    # Mathematical consistency invariants
    assert result["n_scored"] + result["n_skipped"] == result["n_requested_notebooks"]
    assert result["n_missing"] + result["n_failed_notebooks"] == result["n_skipped"]

    # Run level counters:
    # nb_all_pass: 3 success, 0 fail
    # nb_one_fail: 2 success, 1 fail
    # nb_all_fail: 0 success, 3 fail
    assert result["n_successful_runs"] == 5
    assert result["n_failed_runs"] == 4
    assert result["n_attempted_runs"] == 9
    assert result["n_attempted_runs"] == result["n_successful_runs"] + result["n_failed_runs"]
    assert result["n_requested_runs"] == 12  # 4 notebooks * 3 runs

    assert result["failed_notebooks"] == ["nb_one_fail.ipynb", "nb_all_fail.ipynb"]
    assert len(result["failed_runs"]) == 4


def test_majority_voting_uses_complete_successful_runs_only(monkeypatch):
    """7. Majority voting uses complete successful runs only, never failed runs."""
    import eval.yang_eval as yang_eval

    ground_truth = {
        "nb_true.ipynb": {"pre": True, "overlap": False, "multi": False},
        "nb_false.ipynb": {"pre": False, "overlap": False, "multi": False},
        "nb_incomplete.ipynb": {"pre": True, "overlap": False, "multi": False},
    }

    monkeypatch.setattr(
        yang_eval,
        "_resolve_notebook_path",
        lambda nb_id, notebooks_dir: nb_id,
    )

    def fake_predict(nb_path, condition, llm=None, run_index=0):
        if nb_path == "nb_true.ipynb":
            # 2 True, 1 False -> majority True
            return {"pre": (run_index < 2), "overlap": False, "multi": False}
        elif nb_path == "nb_false.ipynb":
            # 1 True, 2 False -> majority False
            return {"pre": (run_index == 0), "overlap": False, "multi": False}
        elif nb_path == "nb_incomplete.ipynb":
            # Fails on run 1
            if run_index == 1:
                return None
            return {"pre": True, "overlap": False, "multi": False}
        return None

    monkeypatch.setattr(yang_eval, "_predict_notebook", fake_predict)

    result = yang_eval.score_condition(
        condition=yang_eval.AuditCondition.LLM_ONLY,
        ground_truth=ground_truth,
        runs=3,
    )

    scored_ids = [r["nb_id"] for r in result["per_notebook"]]
    assert scored_ids == ["nb_true.ipynb", "nb_false.ipynb"]
    assert "nb_incomplete.ipynb" not in scored_ids

    # nb_true voted True, nb_false voted False
    pre_preds = {r["nb_id"]: r["pre_pred"] for r in result["per_notebook"]}
    assert pre_preds["nb_true.ipynb"] is True
    assert pre_preds["nb_false.ipynb"] is False

    # Metrics on valid population:
    # nb_true: GT True, pred True -> TP=1
    # nb_false: GT False, pred False -> TN=1
    # FP=0, FN=0 -> Precision=1.0, Recall=1.0, F1=1.0
    pre_m = result["per_category"]["pre"]
    assert pre_m["tp"] == 1
    assert pre_m["fp"] == 0
    assert pre_m["fn"] == 0
    assert pre_m["tn"] == 1
    assert pre_m["f1"] == 1.0


def test_per_run_variance_excludes_incomplete_notebooks(monkeypatch):
    """8. Per-run variance excludes incomplete notebooks and evaluates the same eligible population."""
    import eval.yang_eval as yang_eval

    ground_truth = {
        "nb_complete.ipynb": {"pre": True, "overlap": False, "multi": False},
        "nb_partial.ipynb": {"pre": True, "overlap": False, "multi": False},
    }

    monkeypatch.setattr(
        yang_eval,
        "_resolve_notebook_path",
        lambda nb_id, notebooks_dir: nb_id,
    )

    def fake_predict(nb_path, condition, llm=None, run_index=0):
        if nb_path == "nb_complete.ipynb":
            # run 0: True (F1=1.0), run 1: True (F1=1.0), run 2: False (F1=0.0)
            return {"pre": (run_index < 2), "overlap": False, "multi": False}
        elif nb_path == "nb_partial.ipynb":
            if run_index == 2:
                return None
            return {"pre": True, "overlap": False, "multi": False}
        return None

    monkeypatch.setattr(yang_eval, "_predict_notebook", fake_predict)

    result = yang_eval.score_condition(
        condition=yang_eval.AuditCondition.LLM_ONLY,
        ground_truth=ground_truth,
        runs=3,
    )

    assert result["n_scored"] == 1
    assert result["failed_notebooks"] == ["nb_partial.ipynb"]

    # Variance must only be calculated on nb_complete.ipynb
    # Run 0: TP=1, FP=0, FN=0 -> F1=1.0
    # Run 1: TP=1, FP=0, FN=0 -> F1=1.0
    # Run 2: TP=0, FP=0, FN=1 -> F1=0.0
    v = result["run_variance"]["pre"]
    assert v["run_f1s"] == [1.0, 1.0, 0.0]
    expected_mean = round((1.0 + 1.0 + 0.0) / 3, 4)
    assert v["mean_f1"] == expected_mean


def test_mcnemar_comparison_uses_only_shared_valid_notebooks():
    """9. McNemar comparison uses only shared valid notebook IDs and handles empty paired sets explicitly."""
    from eval.yang_eval import compare_architectures, mcnemar_test

    gt = {
        "nb1.py": {"pre": True, "overlap": False, "multi": False},
        "nb2.py": {"pre": True, "overlap": True, "multi": False},
        "nb3.py": {"pre": False, "overlap": False, "multi": False},
    }

    # Cond A scored nb1 and nb2 (nb3 failed)
    res_a = {
        "condition": "llm_only",
        "per_notebook": [
            {"nb_id": "nb1.py", "pre_pred": True, "overlap_pred": False, "multi_pred": False},
            {"nb_id": "nb2.py", "pre_pred": False, "overlap_pred": True, "multi_pred": False},
        ]
    }
    # Cond B scored nb2 and nb3 (nb1 failed)
    res_b = {
        "condition": "hybrid",
        "per_notebook": [
            {"nb_id": "nb2.py", "pre_pred": True, "overlap_pred": True, "multi_pred": False},
            {"nb_id": "nb3.py", "pre_pred": False, "overlap_pred": False, "multi_pred": False},
        ]
    }

    comps = compare_architectures({"llm_only": res_a, "hybrid": res_b}, gt)
    pair = comps["llm_only_vs_hybrid"]

    # Only nb2.py is in the intersection
    assert pair["n_paired"] == 1
    assert pair["pre"]["n_paired"] == 1

    # On nb2.py: GT pre is True.
    # Cond A predicted False (wrong) -> Cond B predicted True (right)
    # n01 = 1 (A wrong, B right), n10 = 0 (A right, B wrong)
    assert pair["pre"]["n01"] == 1
    assert pair["pre"]["n10"] == 0

    # Disjoint condition test (0 shared notebooks)
    res_c = {
        "condition": "cond_c",
        "per_notebook": [
            {"nb_id": "nb4_other.py", "pre_pred": True, "overlap_pred": False, "multi_pred": False},
        ]
    }
    empty_comp = compare_architectures({"llm_only": res_a, "cond_c": res_c}, gt)
    empty_pair = empty_comp["llm_only_vs_cond_c"]
    assert empty_pair["n_paired"] == 0
    assert empty_pair["pre"]["n_paired"] == 0
    assert empty_pair["pre"]["statistic"] is None
    assert empty_pair["pre"]["p_value"] is None


def test_zero_eligible_notebooks_produce_explicit_no_data_outcome(monkeypatch):
    """10. Zero eligible notebooks produce an explicit no-data outcome, never misleading F1=1.0."""
    import eval.yang_eval as yang_eval

    ground_truth = {
        "nb_fail_1.ipynb": {"pre": True, "overlap": False, "multi": False},
        "nb_fail_2.ipynb": {"pre": False, "overlap": True, "multi": False},
    }

    monkeypatch.setattr(
        yang_eval,
        "_resolve_notebook_path",
        lambda nb_id, notebooks_dir: nb_id,
    )
    # All audits fail
    monkeypatch.setattr(
        yang_eval,
        "_predict_notebook",
        lambda nb_path, condition, llm=None, run_index=0: None,
    )

    result = yang_eval.score_condition(
        condition=yang_eval.AuditCondition.LLM_ONLY,
        ground_truth=ground_truth,
        runs=3,
    )

    assert result["n_scored"] == 0
    assert result["n_failed_notebooks"] == 2
    assert result["no_data"] is True
    assert result["macro_f1"] is None

    for col in ("pre", "overlap", "multi"):
        m = result["per_category"][col]
        assert m["precision"] is None
        assert m["recall"] is None
        assert m["f1"] is None
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["tn"] == 0

        v = result["run_variance"][col]
        assert v["mean_f1"] is None
        assert v["std_f1"] is None
        assert v["run_f1s"] == []


def test_rule_only_behavior_and_no_llm_logging(tmp_path, monkeypatch):
    """11. Existing successful rule_only behavior remains unchanged with single run and no LLM call log."""
    import eval.yang_eval as yang_eval

    ground_truth = {
        "nb_rule.ipynb": {"pre": True, "overlap": False, "multi": False},
    }

    log_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(yang_eval, "LLM_CALL_LOG_PATH", str(log_path))
    monkeypatch.setattr(
        yang_eval,
        "_resolve_notebook_path",
        lambda nb_id, notebooks_dir: nb_id,
    )

    call_count = 0
    def fake_predict(nb_path, condition, llm=None, run_index=0):
        nonlocal call_count
        call_count += 1
        return {"pre": True, "overlap": False, "multi": False}

    monkeypatch.setattr(yang_eval, "_predict_notebook", fake_predict)

    # Even if runs=3 is requested, rule_only must only execute 1 effective run
    result = yang_eval.score_condition(
        condition=yang_eval.AuditCondition.RULE_ONLY,
        ground_truth=ground_truth,
        runs=3,
    )

    assert call_count == 1
    assert result["n_scored"] == 1
    assert result["n_successful_runs"] == 1
    assert result["n_attempted_runs"] == 1
    assert result["n_requested_runs"] == 1
    assert result["run_variance"] == {}

    # LLM call log must not be created or written for rule_only
    assert not log_path.exists()


def test_logging_failure_does_not_corrupt_audit_results(monkeypatch):
    """Logging exceptions must not conceal audit failures or corrupt successful audit predictions."""
    import eval.yang_eval as yang_eval
    from pipelineguardian.models import AuditReport

    # Monkeypatch _log_llm_call to simulate an unexpected logging failure
    monkeypatch.setattr(
        yang_eval,
        "LLM_CALL_LOG_PATH",
        "/invalid/directory/path/that/cannot/be/written.jsonl",
    )

    # 1. Successful audit: must return genuine predictions, not crash or return None
    monkeypatch.setattr(
        yang_eval,
        "run_audit",
        MagicMock(return_value=AuditReport(source_path="clean.ipynb", issues=[])),
    )
    pred = yang_eval._predict_notebook(
        "clean.ipynb",
        yang_eval.AuditCondition.LLM_ONLY,
        run_index=0,
    )
    assert pred == {"pre": False, "overlap": False, "multi": False}

    # 2. Failed audit: must still return explicit None, never negative predictions
    monkeypatch.setattr(
        yang_eval,
        "run_audit",
        MagicMock(side_effect=RuntimeError("Audit failed")),
    )
    pred_fail = yang_eval._predict_notebook(
        "failed.ipynb",
        yang_eval.AuditCondition.LLM_ONLY,
        run_index=0,
    )
    assert pred_fail is None


def test_subsequent_successful_audit_clears_stale_failure_metadata(tmp_path, monkeypatch):
    """Subsequent successful audit for the same (nb_path, run_index) clears stale failure metadata."""
    import eval.yang_eval as yang_eval
    from pipelineguardian.models import AuditReport

    log_path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(yang_eval, "LLM_CALL_LOG_PATH", str(log_path))

    nb_path = "retry_test.ipynb"
    run_idx = 0

    # 1. First attempt fails
    monkeypatch.setattr(
        yang_eval,
        "run_audit",
        MagicMock(side_effect=RuntimeError("Transient API failure")),
    )
    first_res = yang_eval._predict_notebook(
        nb_path,
        yang_eval.AuditCondition.LLM_ONLY,
        run_index=run_idx,
    )
    assert first_res is None
    assert (nb_path, run_idx) in yang_eval._LAST_AUDIT_FAILURES
    assert "Transient API failure" in yang_eval._LAST_AUDIT_FAILURES[(nb_path, run_idx)]["error"]

    # 2. Subsequent attempt for the same notebook and run index succeeds
    monkeypatch.setattr(
        yang_eval,
        "run_audit",
        MagicMock(return_value=AuditReport(source_path=nb_path, issues=[])),
    )
    second_res = yang_eval._predict_notebook(
        nb_path,
        yang_eval.AuditCondition.LLM_ONLY,
        run_index=run_idx,
    )

    # Verify predictions are returned normally and failure metadata was cleared
    assert second_res == {"pre": False, "overlap": False, "multi": False}
    assert (nb_path, run_idx) not in yang_eval._LAST_AUDIT_FAILURES


def test_repaired_yang_notebook_resolves_to_repair_copy():
    from pathlib import Path

    from eval.yang_eval import _resolve_notebook_path

    nb_id = "GitHub-data/notebooks/2021-09-09/nb_334.py"
    repaired_path = (
        Path("eval")
        / "yang_corpus"
        / "repairs"
        / "2021-09-09-nb_334.ipynb"
    ).resolve()

    resolved = _resolve_notebook_path(nb_id)

    assert resolved is not None
    assert Path(resolved).resolve() == repaired_path
