"""
yang_eval.py — runs a given AuditCondition against every notebook in the Yang
corpus and scores category-level precision/recall/F1 against ground-truth.csv.

Scope: PREPROCESSING, OVERLAP, MULTI_TEST only — these are the only categories
Yang's ground truth labels (columns 'pre', 'overlap', 'multi'). REPRODUCIBILITY
and VALIDATION_STRATEGY have no Yang ground truth and are evaluated only via the
synthetic corpus (eval.py) — this is a scoring-availability decision, not a scope
reduction; all conditions still run those checks and report them, they are just
not part of this particular comparison table.

The 'model' column is confirmed (see project notes) to be an applicability/
annotation flag, not a fourth leakage category. It is never used as a
ground-truth label here.

Exclusion: target_column_in_feature_list has no corresponding column in Yang's
ground truth. It is excluded entirely from this scorer — never mapped to any of
pre/overlap/multi, never counted toward any Yang category's TP/FP/FN. It
remains scored only in the synthetic corpus (eval.py).

Run:
    python -m eval.yang_eval --condition rule_only
    python -m eval.yang_eval --condition hybrid --runs 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from pipelineguardian.conditions import AuditCondition, run_audit
from pipelineguardian.models import Category
from pipelineguardian.llm_config import LLM_RUNS_PER_NOTEBOOK

HERE = os.path.dirname(__file__)
YANG_DIR = os.path.join(HERE, "yang_corpus")
NOTEBOOKS_DIR = os.path.join(YANG_DIR, "notebooks")
REPAIRED_NOTEBOOKS = {
    "GitHub-data/notebooks/2021-09-09/nb_334.py":
        os.path.join(YANG_DIR, "repairs", "2021-09-09-nb_334.ipynb"),
}
GROUND_TRUTH_PATH = os.path.join(YANG_DIR, "ground_truth", "ground-truth.csv")
RESULTS_MD_PATH = os.path.join(HERE, "yang_results.md")
RESULTS_JSON_PATH = os.path.join(HERE, "yang_results.json")
RUN_MANIFEST_PATH = os.path.join(HERE, "run_manifest.jsonl")
LLM_CALL_LOG_PATH = os.path.join(HERE, "llm_call_log.jsonl")

# Mapping Yang ground-truth CSV column → Category enum
YANG_CATEGORY_TO_COLUMN: dict[Category, str] = {
    Category.PREPROCESSING: "pre",
    Category.OVERLAP: "overlap",
    Category.MULTI_TEST: "multi",
}

# LLM-touching conditions — need multiple runs for variance estimation
LLM_CONDITIONS = {AuditCondition.LLM_ONLY, AuditCondition.LLM_WITH_CONTEXT, AuditCondition.HYBRID}


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(path: str = GROUND_TRUTH_PATH) -> dict[str, dict[str, bool]]:
    """Return {notebook_id: {'pre': bool, 'overlap': bool, 'multi': bool}}.

    Binarization rule (LOCKED — see spec §6.2):
    A cell is positive if its value starts with 'Y' (covers 'Y', 'Y (mean)',
    'Y (MinMaxScaler)', etc.) — qualitative suffixes are NOT treated as separate
    labels. Blank and 'N' both binarize to False. The 'model' column is never
    read here.
    """
    gt: dict[str, dict[str, bool]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nb_id = row["nb"].strip()
            gt[nb_id] = {
                col: row[col].strip().upper().startswith("Y")
                for col in ("pre", "overlap", "multi")
            }
    return gt


# ---------------------------------------------------------------------------
# Notebook path resolution
# ---------------------------------------------------------------------------

def _resolve_notebook_path(nb_id: str, notebooks_dir: str = NOTEBOOKS_DIR) -> Optional[str]:
    """Resolve a ground-truth notebook ID to an actual file path.

    Ground-truth IDs look like 'GitHub-data/notebooks/2021-09-05/nb_1244.py'.
    We look for the file relative to notebooks_dir after stripping the
    'GitHub-data/notebooks/' prefix if present.
    """
    # Explicitly route the approved repaired Yang notebook.
    repaired_path = REPAIRED_NOTEBOOKS.get(nb_id)
    if repaired_path is not None and os.path.isfile(repaired_path):
        return repaired_path
    # Try direct join first
    candidate = os.path.join(notebooks_dir, nb_id)
    if os.path.exists(candidate):
        return candidate

    # Strip leading 'GitHub-data/notebooks/' prefix
    stripped = nb_id
    for prefix in ("GitHub-data/notebooks/", "GitHub-data\\notebooks\\"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    candidate = os.path.join(notebooks_dir, stripped)
    if os.path.exists(candidate):
        return candidate

    # Last resort: search by basename or flattened hyphenated form (e.g. 2021-09-05-nb_1244.ipynb)
    basename = os.path.basename(nb_id)
    stem, ext = os.path.splitext(basename)

    # Flattened name format: 2021-09-05/nb_1244.py -> 2021-09-05-nb_1244.ipynb
    flattened = stripped.replace("/", "-").replace("\\", "-")
    if flattened.endswith(".py"):
        flattened_ipynb = flattened[:-3] + ".ipynb"
    else:
        flattened_ipynb = flattened + ".ipynb"

    for candidate_name in (flattened, flattened_ipynb, basename, stem + ".ipynb"):
        candidate = os.path.join(notebooks_dir, candidate_name)
        if os.path.exists(candidate):
            return candidate

    for root, dirs, files in os.walk(notebooks_dir):
        if basename in files:
            return os.path.join(root, basename)
        if (stem + ".ipynb") in files:
            return os.path.join(root, stem + ".ipynb")

    return None


# ---------------------------------------------------------------------------
# Single-notebook prediction
# ---------------------------------------------------------------------------

_LAST_AUDIT_FAILURES: dict[tuple[str, int], dict] = {}


def _classify_error(e: Exception) -> str:
    """Classify exception type into provider API, parse, or execution failure."""
    name = type(e).__name__.lower()
    msg = str(e).lower()
    mod = getattr(type(e), "__module__", "").lower()

    if (
        any(k in name for k in ("api", "ratelimit", "authentication", "auth", "connection", "timeout", "serviceunavailable"))
        or any(k in mod for k in ("groq", "http", "urllib", "requests"))
        or any(k in msg for k in ("groq", "rate limit", "503", "429", "api key", "unauthorized", "connection error"))
        or isinstance(e, (ConnectionError, TimeoutError))
    ):
        return "provider_api_error"
    elif (
        any(k in name for k in ("notjson", "jsondecode", "syntaxerror", "nbformat"))
        or ("json" in msg or ("notebook" in msg and "parse" in msg))
    ):
        return "notebook_parse_error"
    else:
        return "audit_execution_error"


def _sanitize_error_message(msg: str) -> str:
    """Sanitize error messages to avoid leaking API keys or secrets."""
    sanitized = re.sub(r"gsk_[A-Za-z0-9_\-]+", "gsk_[REDACTED]", msg)
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", sanitized, flags=re.IGNORECASE)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key and groq_key in sanitized:
        sanitized = sanitized.replace(groq_key, "[REDACTED_API_KEY]")
    return sanitized


def _predict_notebook(
    nb_path: str,
    condition: AuditCondition,
    llm=None,
    run_index: int = 0,
) -> Optional[dict[str, bool]]:
    """Return category predictions, or None if the audit failed."""
    _LAST_AUDIT_FAILURES.pop((nb_path, run_index), None)
    t0 = time.time()

    try:
        report = run_audit(nb_path, condition, llm=llm)
    except Exception as e:
        elapsed = time.time() - t0
        err_type = type(e).__name__
        err_cat = _classify_error(e)
        raw_error = f"{err_type}: {e}"
        sanitized_error = _sanitize_error_message(raw_error)
        print(f"  [WARN] {os.path.basename(nb_path)}: {sanitized_error}")

        _LAST_AUDIT_FAILURES[(nb_path, run_index)] = {
            "error_type": err_type,
            "error_category": err_cat,
            "error": sanitized_error,
        }

        if condition in LLM_CONDITIONS:
            _log_llm_call(
                nb_path, condition, run_index, elapsed,
                status="failed", error=sanitized_error,
                error_type=err_type, error_category=err_cat,
            )
        return None

    elapsed = time.time() - t0

    if condition in LLM_CONDITIONS:
        _log_llm_call(
            nb_path, condition, run_index, elapsed,
            status="success",
        )

    pred: dict[str, bool] = {}
    for cat, col in YANG_CATEGORY_TO_COLUMN.items():
        pred[col] = any(i.category == cat.value for i in report.issues)
    return pred


def _log_llm_call(
    nb_path: str,
    condition: AuditCondition,
    run_index: int,
    elapsed: float,
    status: str = "success",
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    error_category: Optional[str] = None,
):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notebook": os.path.basename(nb_path),
        "condition": condition.value,
        "run_index": run_index,
        "latency_s": round(elapsed, 3),
        "status": status,
    }
    if error_type is not None:
        entry["error_type"] = error_type
    if error_category is not None:
        entry["error_category"] = error_category
    if error is not None:
        entry["error"] = _sanitize_error_message(error)[:500]

    try:
        with open(LLM_CALL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as log_err:
        print(f"  [WARN] Failed to write LLM call log: {log_err}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_condition(
    condition: AuditCondition,
    ground_truth: dict[str, dict[str, bool]],
    notebooks_dir: str = NOTEBOOKS_DIR,
    runs: int = 1,
    llm=None,
) -> dict:
    """Run condition over every notebook in ground_truth and compute metrics.

    For LLM-backed conditions, runs each notebook `runs` times and uses
    majority vote as the point estimate fed into precision/recall/F1 and
    McNemar's test. Mean ± std over runs is also reported.

    Returns a dict with:
      per_category: {col: {precision, recall, f1, tp, fp, fn}}
      per_notebook: [{nb_id, pre_pred, overlap_pred, multi_pred}]
      run_details: raw per-run predictions (for variance reporting)
    """
    cols = list(YANG_CATEGORY_TO_COLUMN.values())
    effective_runs = runs if condition in LLM_CONDITIONS else 1
    n_requested_notebooks = len(ground_truth)
    n_requested_runs = n_requested_notebooks * effective_runs

    all_run_preds: dict[str, dict[str, list[bool]]] = {}

    missing_notebooks = []
    resolved_nbs: list[tuple[str, str]] = []  # [(nb_id, nb_path)]

    for nb_id in ground_truth:
        nb_path = _resolve_notebook_path(nb_id, notebooks_dir)
        if nb_path is None:
            missing_notebooks.append(nb_id)
            continue
        resolved_nbs.append((nb_id, nb_path))

    if missing_notebooks:
        print(f"  [WARN] Could not resolve {len(missing_notebooks)} notebooks — they will be skipped.")

    failed_notebooks: list[str] = []
    failed_runs: list[dict] = []
    failed_notebook_reasons: dict[str, str] = {}
    successful_runs = 0
    attempted_runs = 0

    for nb_id, nb_path in resolved_nbs:
        notebook_run_preds = {col: [] for col in cols}
        notebook_failed = False
        nb_failures = []

        for run_i in range(effective_runs):
            attempted_runs += 1
            pred = _predict_notebook(
                nb_path, condition, llm=llm, run_index=run_i
            )

            if pred is None:
                notebook_failed = True
                failure_info = _LAST_AUDIT_FAILURES.pop((nb_path, run_i), {})
                fail_entry = {
                    "nb_id": nb_id,
                    "run_index": run_i,
                    "error_type": failure_info.get("error_type", "AuditFailure"),
                    "error_category": failure_info.get("error_category", "unknown"),
                    "error": failure_info.get("error", "Audit failed or returned None"),
                }
                failed_runs.append(fail_entry)
                nb_failures.append(fail_entry)
                continue

            successful_runs += 1
            for col in cols:
                notebook_run_preds[col].append(pred[col])

        if notebook_failed:
            failed_notebooks.append(nb_id)
            reasons = [
                f"run {f['run_index']}: {f['error_type']} ({f['error_category']})"
                for f in nb_failures
            ]
            failed_notebook_reasons[nb_id] = "; ".join(reasons)
            print(
                f"  [WARN] Excluding {nb_id} from scoring: "
                f"at least one audit run failed ({failed_notebook_reasons[nb_id]})."
            )
            continue

        all_run_preds[nb_id] = notebook_run_preds

    # Majority vote across runs → single point prediction per (nb, col)
    per_notebook = []
    for nb_id, run_preds in all_run_preds.items():
        row = {"nb_id": nb_id}
        for col in cols:
            votes = run_preds[col]
            row[f"{col}_pred"] = votes.count(True) > len(votes) / 2
        per_notebook.append(row)

    # Accumulate TP/FP/FN per category
    per_category: dict[str, dict] = {}
    if not per_notebook:
        for col in cols:
            per_category[col] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            }
        macro_f1 = None
    else:
        for col in cols:
            tp = fp = fn = tn = 0
            for row in per_notebook:
                gt_val = ground_truth.get(row["nb_id"], {}).get(col, False)
                pred_val = row[f"{col}_pred"]
                if gt_val and pred_val:
                    tp += 1
                elif not gt_val and pred_val:
                    fp += 1
                elif gt_val and not pred_val:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            per_category[col] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }

        macro_f1 = sum(m["f1"] for m in per_category.values()) / len(per_category) if per_category else 0.0

    # LLM variance: mean ± std over runs per category
    run_variance: dict[str, dict] = {}
    if condition in LLM_CONDITIONS and runs > 1:
        if not all_run_preds:
            for col in cols:
                run_variance[col] = {
                    "mean_f1": None,
                    "std_f1": None,
                    "run_f1s": [],
                }
        else:
            for col in cols:
                run_f1s = []
                for run_i in range(runs):
                    tp = fp = fn = 0
                    for nb_id, run_preds in all_run_preds.items():
                        if run_i >= len(run_preds[col]):
                            continue
                        pred_val = run_preds[col][run_i]
                        gt_val = ground_truth.get(nb_id, {}).get(col, False)
                        if gt_val and pred_val:
                            tp += 1
                        elif not gt_val and pred_val:
                            fp += 1
                        elif gt_val and not pred_val:
                            fn += 1
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    rec = tp / (tp + fn) if (tp + fn) else 0.0
                    f1_r = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                    run_f1s.append(f1_r)
                mean_f1 = sum(run_f1s) / len(run_f1s) if run_f1s else 0.0
                var = sum((x - mean_f1) ** 2 for x in run_f1s) / len(run_f1s) if run_f1s else 0.0
                std_f1 = var ** 0.5
                run_variance[col] = {
                    "mean_f1": round(mean_f1, 4),
                    "std_f1": round(std_f1, 4),
                    "run_f1s": [round(x, 4) for x in run_f1s],
                }

    return {
        "condition": condition.value,
        "per_category": per_category,
        "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
        "per_notebook": per_notebook,
        "run_variance": run_variance,
        "n_scored": len(per_notebook),
        "n_skipped": len(missing_notebooks) + len(failed_notebooks),
        "n_missing": len(missing_notebooks),
        "n_failed_notebooks": len(failed_notebooks),
        "n_failed_runs": len(failed_runs),
        "n_successful_runs": successful_runs,
        "n_attempted_runs": attempted_runs,
        "n_requested_runs": n_requested_runs,
        "n_requested_notebooks": n_requested_notebooks,
        "failed_notebooks": failed_notebooks,
        "failed_notebook_reasons": failed_notebook_reasons,
        "failed_runs": failed_runs,
        "no_data": len(per_notebook) == 0,
        "out_of_scope_categories": [Category.REPRODUCIBILITY.value, Category.VALIDATION_STRATEGY.value],
    }


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    cond_a_preds: list[bool],
    cond_b_preds: list[bool],
    gt: list[bool],
) -> dict:
    """Compute McNemar's test comparing two conditions on paired notebook outcomes.

    Returns {statistic, p_value, n01, n10, n_paired} where n01 = A wrong, B right and
    n10 = A right, B wrong. Uses scipy.stats binomtest (or exact fallback) for p-value.
    If n_paired == 0, returns explicit no-data outcome with None for statistic and p_value.
    """
    if len(cond_a_preds) != len(cond_b_preds) or len(cond_a_preds) != len(gt):
        raise ValueError("Input lists must have identical lengths for paired McNemar test")

    n_paired = len(cond_a_preds)
    if n_paired == 0:
        return {
            "statistic": None,
            "p_value": None,
            "n01": 0,
            "n10": 0,
            "n_paired": 0,
        }

    a_correct = [p == g for p, g in zip(cond_a_preds, gt)]
    b_correct = [p == g for p, g in zip(cond_b_preds, gt)]

    n00 = sum(1 for a, b in zip(a_correct, b_correct) if not a and not b)
    n01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    n10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n11 = sum(1 for a, b in zip(a_correct, b_correct) if a and b)

    n = n01 + n10
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n01": 0,
            "n10": 0,
            "n_paired": n_paired,
        }

    statistic = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if abs(n10 - n01) >= 1 else 0.0

    p_value = 1.0
    try:
        from scipy.stats import binomtest
        p_value = float(binomtest(n10, n, 0.5).pvalue)
    except (ImportError, AttributeError):
        try:
            from scipy.stats import binom_test
            p_value = float(binom_test(n10, n, 0.5))
        except (ImportError, AttributeError):
            import math
            prob_k = math.comb(n, n10) * (0.5 ** n)
            p_value = min(1.0, 2.0 * sum(math.comb(n, i) * (0.5 ** n) for i in range(min(n10, n - n10) + 1)))

    return {
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 4),
        "n01": n01,
        "n10": n10,
        "n_paired": n_paired,
    }


def compare_architectures(
    results_by_condition: dict[str, dict],
    ground_truth: dict[str, dict[str, bool]],
) -> dict:
    """Perform paired per-notebook McNemar comparisons across conditions for each category.

    results_by_condition maps condition_name -> output from score_condition().
    Returns dict of pairwise McNemar test results.
    """
    cols = list(YANG_CATEGORY_TO_COLUMN.values())
    conditions = list(results_by_condition.keys())
    comparisons = {}

    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            cond_a = conditions[i]
            cond_b = conditions[j]
            pair_key = f"{cond_a}_vs_{cond_b}"
            comparisons[pair_key] = {}

            # Build aligned prediction vectors per notebook using only scored notebooks
            nb_ids_a = {row["nb_id"]: row for row in results_by_condition[cond_a].get("per_notebook", [])}
            nb_ids_b = {row["nb_id"]: row for row in results_by_condition[cond_b].get("per_notebook", [])}
            common_nbs = sorted(set(nb_ids_a.keys()) & set(nb_ids_b.keys()))
            comparisons[pair_key]["n_paired"] = len(common_nbs)

            for col in cols:
                preds_a = [nb_ids_a[nb][f"{col}_pred"] for nb in common_nbs]
                preds_b = [nb_ids_b[nb][f"{col}_pred"] for nb in common_nbs]
                gt_vec = [ground_truth.get(nb, {}).get(col, False) for nb in common_nbs]

                comparisons[pair_key][col] = mcnemar_test(preds_a, preds_b, gt_vec)

    return comparisons


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------

def _append_run_manifest(condition: AuditCondition, config: dict):
    git_hash = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE,
        )
        git_hash = result.stdout.strip()
    except Exception:
        pass

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "condition": condition.value,
        "config": config,
    }
    with open(RUN_MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def write_results(results: dict):
    cond_name = results.get("condition", "unknown")
    cond_json_path = os.path.join(HERE, f"yang_results_{cond_name}.json")
    with open(cond_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cols_display = {"pre": "Preprocessing", "overlap": "Overlap", "multi": "Multi-test"}
    macro_f1_disp = results.get("macro_f1")
    macro_f1_str = f"{macro_f1_disp:.4f}" if macro_f1_disp is not None else "N/A"
    lines = [
        "# Yang-Corpus Evaluation Results\n",
        f"Condition: **{results.get('condition', '?')}**  "
        f"| Notebooks scored: {results.get('n_scored', '?')}  "
        f"| Skipped: {results.get('n_skipped', 0)}  "
        f"| **Macro-F1: {macro_f1_str}**\n",
        "> [!NOTE]\n"
        "> Categories REPRODUCIBILITY and VALIDATION_STRATEGY have no Yang ground-truth labels "
        "> and are evaluated via dedicated separate benchmarks.\n",
        "## Per-Category Metrics\n",
        "| Category | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for col, name in cols_display.items():
        m = results.get("per_category", {}).get(col, {})
        p = f"{m['precision']:.4f}" if m.get("precision") is not None else "N/A"
        r = f"{m['recall']:.4f}" if m.get("recall") is not None else "N/A"
        f1_val = f"{m['f1']:.4f}" if m.get("f1") is not None else "N/A"
        lines.append(
            f"| {name} | {p} | {r} | "
            f"{f1_val} | {m.get('tp','?')} | {m.get('fp','?')} | {m.get('fn','?')} |"
        )
    if results.get("run_variance"):
        lines.append("\n## LLM Variance (mean ± std F1 over runs)\n")
        for col, name in cols_display.items():
            v = results["run_variance"].get(col, {})
            if v and v.get("mean_f1") is not None:
                lines.append(f"- {name}: {v.get('mean_f1'):.4f} ± {v.get('std_f1'):.4f}")
            elif v:
                lines.append(f"- {name}: N/A")

    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Results written to {RESULTS_MD_PATH} and {RESULTS_JSON_PATH}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pipelineguardian.llm_config import (
        LLM_MODEL, LLM_TEMPERATURE, PROMPT_VERSION, CONTEXT_SCHEMA_VERSION
    )

    parser = argparse.ArgumentParser(description="Yang-corpus category-level evaluator")
    parser.add_argument("--condition", required=True, choices=[c.value for c in AuditCondition])
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Number of LLM runs per notebook (default: LLM_RUNS_PER_NOTEBOOK for LLM conditions, 1 for rule_only)",
    )
    args = parser.parse_args()

    condition = AuditCondition(args.condition)
    runs = args.runs if args.runs is not None else (
        LLM_RUNS_PER_NOTEBOOK if condition in LLM_CONDITIONS else 1
    )

    config = {
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "runs": runs,
    }

    print(f"Running Yang-corpus eval: condition={condition.value}, runs={runs}")
    gt = load_ground_truth()
    results = score_condition(condition, gt, runs=runs)
    results["condition"] = condition.value
    results["config"] = config

    write_results(results)
    _append_run_manifest(condition, config)

    for col in ("pre", "overlap", "multi"):
        m = results["per_category"][col]
        p_val = f"{m['precision']:.4f}" if m["precision"] is not None else "N/A"
        r_val = f"{m['recall']:.4f}" if m["recall"] is not None else "N/A"
        f1_val = f"{m['f1']:.4f}" if m["f1"] is not None else "N/A"
        print(f"  {col}: P={p_val} R={r_val} F1={f1_val}")
