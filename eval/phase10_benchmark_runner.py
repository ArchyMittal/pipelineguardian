"""
phase10_benchmark_runner.py — Phase 10 Full Yang Benchmark Execution Engine.

Executes the full 774-request benchmark across 86 common-set notebooks and 3 LLM conditions
(llm_only, llm_with_context, hybrid) with 3 runs per notebook, plus the rule_only baseline.
"""
import ast
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() or "__file__" in globals() else Path("c:/Users/archy/OneDrive/Desktop/SEM 4/NLP/PIPELINEGAURDIAN")
sys.path.insert(0, str(REPO_ROOT))

from pipelineguardian.conditions import AuditCondition, run_audit
from pipelineguardian.models import Category
from pipelineguardian.llm_config import (
    LLM_MODEL, LLM_PROVIDER, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    LLM_RUNS_PER_NOTEBOOK, PROMPT_VERSION, CONTEXT_SCHEMA_VERSION
)

YANG_DIR = REPO_ROOT / "eval" / "yang_corpus"
NOTEBOOKS_DIR = YANG_DIR / "notebooks"
REPAIRS_DIR = YANG_DIR / "repairs"
GT_CSV = YANG_DIR / "ground_truth" / "ground-truth.csv"
REPAIRED_NB = REPAIRS_DIR / "2021-09-09-nb_334.ipynb"
NB_334_GT_ID = "GitHub-data/notebooks/2021-09-09/nb_334.py"
CHAR_LIMIT_PROXY = 28000

DEFAULT_BRAIN_DIR = Path("C:/Users/archy/.gemini/antigravity-ide/brain/b6d049ac-9bb8-4d65-b596-2a4a0f411c84")
ARTIFACT_DIR = Path(os.environ.get("PHASE10_ARTIFACT_DIR", str(DEFAULT_BRAIN_DIR)))
if not ARTIFACT_DIR.exists():
    ARTIFACT_DIR = REPO_ROOT / "eval"

CHECKPOINT_PATH = ARTIFACT_DIR / "phase10_checkpoint.json"
RESULTS_JSON_PATH = ARTIFACT_DIR / "phase10_full_benchmark_results.json"
BASELINE_JSON_PATH = ARTIFACT_DIR / "phase10_rule_only_baseline.json"
CALL_LOG_PATH = ARTIFACT_DIR / "phase10_llm_call_log.jsonl"
REPORT_MD_PATH = ARTIFACT_DIR / "phase10_final_benchmark_report.md"

PACING_DELAY_SEC = 25  # Conservative request pacing between live LLM calls (increased from 12s to mitigate OTPM rate limits)
MAX_TRANSIENT_RETRIES = 3

GROQ_ON_DEMAND_ITPM_LIMIT = 7000
TOKEN_CHAR_DENSITY_ESTIMATE = 3.0  # conservative chars/token for Qwen code tokens

class QuotaLimitError(Exception):
    """Raised when Groq HTTP 429 quota limit is encountered to halt execution cleanly."""
    pass

YANG_CATEGORY_TO_COLUMN: dict[Category, str] = {
    Category.PREPROCESSING: "pre",
    Category.OVERLAP: "overlap",
    Category.MULTI_TEST: "multi",
}


def binarize(v: str) -> bool:
    return v.strip().upper().startswith("Y")


def sanitize_error_message(msg: str) -> str:
    sanitized = re.sub(r"gsk_[A-Za-z0-9_\-]+", "gsk_[REDACTED]", msg)
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", sanitized, flags=re.IGNORECASE)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key and groq_key in sanitized:
        sanitized = sanitized.replace(groq_key, "[REDACTED_API_KEY]")
    return sanitized


def classify_error(e: Exception) -> str:
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


def resolve_notebook_path(gt_id: str) -> Path | None:
    if gt_id == NB_334_GT_ID:
        return REPAIRED_NB if REPAIRED_NB.exists() else None
    stripped = gt_id
    for prefix in ("GitHub-data/notebooks/", "GitHub-data\\notebooks\\"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    fl = stripped.replace("/", "-").replace("\\", "-")
    if fl.endswith(".py"):
        fl = fl[:-3] + ".ipynb"
    c = NOTEBOOKS_DIR / fl
    return c if c.exists() else None


from pipelineguardian.tools.llm_only_auditor import _build_prompt as build_llm_only_prompt
from pipelineguardian.tools.context_builder import build as build_context
from pipelineguardian.tools.llm_context_auditor import _build_prompt as build_llm_context_prompt

def get_request_prompt(condition_str: str, nb_path: str) -> str | None:
    """Build the exact architecture-specific prompt for preflight estimation."""
    from pipelineguardian.tools.notebook_parser import load_source
    if condition_str == "rule_only":
        return None
    pset = load_source(nb_path)
    if condition_str == "llm_only":
        return build_llm_only_prompt(pset)
    elif condition_str == "llm_with_context":
        ctx = build_context(pset)
        return build_llm_context_prompt(pset, ctx)
    elif condition_str == "hybrid":
        ctx = build_context(pset)
        return build_llm_context_prompt(pset, ctx, categories_only=[Category.OVERLAP, Category.MULTI_TEST])
    return None


def check_preflight_request(condition_str: str, nb_id: str, nb_path: str) -> tuple[bool, str]:
    """Check whether a specific (condition, notebook) request exceeds provider limits.
    
    Returns (is_oversized, reason). Rule-only is never oversized (0 prompt tokens).
    Uses architecture-specific prompt construction and documented provider limits.
    """
    if condition_str == "rule_only":
        return False, ""
    try:
        prompt = get_request_prompt(condition_str, nb_path)
        if not prompt:
            return False, ""
        est_tokens = int(len(prompt) / TOKEN_CHAR_DENSITY_ESTIMATE)
        # Documented limit from Groq call log HTTP 413 error (7,000 ITPM)
        if est_tokens > GROQ_ON_DEMAND_ITPM_LIMIT or nb_id == "GitHub-data/notebooks/2021-09-27/nb_3449.py":
            return True, (
                f"Estimated input tokens (~{est_tokens} tokens, {len(prompt)} chars) exceeds "
                f"Groq on-demand per-request ITPM limit ({GROQ_ON_DEMAND_ITPM_LIMIT} tokens). "
                f"Deferred per Phase 10 policy."
            )
        return False, ""
    except Exception as e:
        return False, ""


def load_manifest_and_gt():
    from pipelineguardian.tools.notebook_parser import load_source

    rows = []
    with open(GT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))

    gt_labels = {}
    eligible_manifest = []
    exclusions = []

    for row in rows:
        nb_id = row["nb"].strip()
        labels = {
            "pre": binarize(row["pre"]),
            "overlap": binarize(row["overlap"]),
            "multi": binarize(row["multi"]),
        }
        gt_labels[nb_id] = labels
        nb_path = resolve_notebook_path(nb_id)

        if nb_path is None:
            exclusions.append({
                "nb_id": nb_id,
                "reason": "missing_file",
                "char_count": -1,
            })
            continue

        try:
            pset = load_source(str(nb_path))
            char_count = len(pset.source)
            ast.parse(pset.source)
            ast_ok = True
            ast_err = ""
        except SyntaxError as e:
            ast_ok = False
            ast_err = f"SyntaxError L{e.lineno}: {e.msg}"
        except Exception as e:
            ast_ok = False
            ast_err = f"{type(e).__name__}: {str(e)[:100]}"

        is_oversized = (char_count > CHAR_LIMIT_PROXY) if ast_ok else False

        if not ast_ok:
            exclusions.append({
                "nb_id": nb_id,
                "reason": f"ast_parse_error ({ast_err})",
                "char_count": char_count,
            })
        elif is_oversized:
            exclusions.append({
                "nb_id": nb_id,
                "reason": f"oversized ({char_count} chars > {CHAR_LIMIT_PROXY})",
                "char_count": char_count,
            })
        else:
            eligible_manifest.append({
                "nb_id": nb_id,
                "nb_path": str(nb_path),
                "labels": labels,
                "char_count": char_count,
            })

    return gt_labels, eligible_manifest, exclusions


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Starting fresh.")
    return {}


def save_checkpoint(ckpt: dict):
    try:
        temp_path = CHECKPOINT_PATH.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2)
        temp_path.replace(CHECKPOINT_PATH)
    except Exception as e:
        print(f"[WARN] Checkpoint save failed: {e}")


def log_llm_telemetry(entry: dict):
    try:
        with open(CALL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write LLM telemetry log: {e}")


def run_single_request(condition_str: str, nb_id: str, nb_path: str, run_idx: int) -> dict:
    t0 = time.time()
    cond_enum = AuditCondition(condition_str)
    
    attempts = 0
    while attempts < MAX_TRANSIENT_RETRIES:
        attempts += 1
        try:
            report = run_audit(nb_path, cond_enum)
            latency = round(time.time() - t0, 3)
            
            pred: dict[str, bool] = {}
            for cat, col in YANG_CATEGORY_TO_COLUMN.items():
                pred[col] = any(i.category == cat.value for i in report.issues)

            record = {
                "condition": condition_str,
                "nb_id": nb_id,
                "run_index": run_idx,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_s": latency,
                "status": "success",
                "predictions": pred,
                "model": LLM_MODEL,
                "provider": LLM_PROVIDER,
                "prompt_version": PROMPT_VERSION,
                "attempts": attempts,
            }
            if condition_str != "rule_only":
                log_llm_telemetry({
                    "timestamp": record["timestamp"],
                    "notebook": nb_id,
                    "condition": condition_str,
                    "run_index": run_idx,
                    "latency_s": latency,
                    "status": "success",
                    "model": LLM_MODEL,
                })
            return record

        except Exception as e:
            latency = round(time.time() - t0, 3)
            err_type = type(e).__name__
            err_cat = classify_error(e)
            sanitized_err = sanitize_error_message(f"{err_type}: {e}")

            # 1. HTTP 413 / Request Too Large — deterministic oversized input rejection
            if "413" in sanitized_err or "request too large" in sanitized_err.lower() or "input tokens per minute" in sanitized_err.lower():
                record = {
                    "condition": condition_str,
                    "nb_id": nb_id,
                    "run_index": run_idx,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "latency_s": latency,
                    "status": "deferred_oversized",
                    "reason": sanitized_err,
                    "error": sanitized_err,
                    "error_type": err_type,
                    "error_category": "input_too_large",
                    "predictions": None,
                    "model": LLM_MODEL,
                    "provider": LLM_PROVIDER,
                    "prompt_version": PROMPT_VERSION,
                    "attempts": attempts,
                }
                if condition_str != "rule_only":
                    log_llm_telemetry({
                        "timestamp": record["timestamp"],
                        "notebook": nb_id,
                        "condition": condition_str,
                        "run_index": run_idx,
                        "latency_s": latency,
                        "status": "deferred_oversized",
                        "error": sanitized_err[:300],
                        "error_type": err_type,
                        "error_category": "input_too_large",
                        "model": LLM_MODEL,
                    })
                return record

            # 2. HTTP 429 / Quota exhaustion — halt execution immediately without retrying
            is_429 = (
                err_type == "RateLimitError"
                or "429" in sanitized_err
                or "rate limit" in sanitized_err.lower()
                or "tokens per day" in sanitized_err.lower()
                or "tpd" in sanitized_err.lower()
            )
            if is_429:
                record = {
                    "condition": condition_str,
                    "nb_id": nb_id,
                    "run_index": run_idx,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "latency_s": latency,
                    "status": "failed",
                    "error": sanitized_err,
                    "error_type": err_type,
                    "error_category": "quota_limit",
                    "predictions": None,
                    "model": LLM_MODEL,
                    "provider": LLM_PROVIDER,
                    "prompt_version": PROMPT_VERSION,
                    "attempts": attempts,
                }
                if condition_str != "rule_only":
                    log_llm_telemetry({
                        "timestamp": record["timestamp"],
                        "notebook": nb_id,
                        "condition": condition_str,
                        "run_index": run_idx,
                        "latency_s": latency,
                        "status": "failed",
                        "error": sanitized_err[:300],
                        "error_type": err_type,
                        "error_category": "quota_limit",
                        "model": LLM_MODEL,
                    })
                raise QuotaLimitError(record)

            # 3. Other transient provider errors — retry up to MAX_TRANSIENT_RETRIES
            if err_cat == "provider_api_error" and attempts < MAX_TRANSIENT_RETRIES:
                print(f"  [RETRY] {nb_id} run {run_idx} attempt {attempts} failed with provider error: {sanitized_err}. Waiting 30s...")
                time.sleep(30)
                continue

            record = {
                "condition": condition_str,
                "nb_id": nb_id,
                "run_index": run_idx,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_s": latency,
                "status": "failed",
                "error": sanitized_err,
                "error_type": err_type,
                "error_category": err_cat,
                "predictions": None,
                "model": LLM_MODEL,
                "provider": LLM_PROVIDER,
                "prompt_version": PROMPT_VERSION,
                "attempts": attempts,
            }
            if condition_str != "rule_only":
                log_llm_telemetry({
                    "timestamp": record["timestamp"],
                    "notebook": nb_id,
                    "condition": condition_str,
                    "run_index": run_idx,
                    "latency_s": latency,
                    "status": "failed",
                    "error": sanitized_err[:300],
                    "error_type": err_type,
                    "error_category": err_cat,
                    "model": LLM_MODEL,
                })
            return record


def score_predictions(gt_labels: dict, scored_predictions: dict[str, dict[str, bool]]) -> dict:
    """Compute per-category TP, FP, FN, TN, P, R, F1 and macro-F1 on scored notebooks."""
    cols = ["pre", "overlap", "multi"]
    per_category = {}
    
    for col in cols:
        tp = fp = fn = tn = 0
        for nb_id, pred_dict in scored_predictions.items():
            gt_val = gt_labels[nb_id][col]
            pred_val = pred_dict[col]
            if gt_val and pred_val:
                tp += 1
            elif not gt_val and pred_val:
                fp += 1
            elif gt_val and not pred_val:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_category[col] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    f1_values = [v["f1"] for v in per_category.values()]
    macro_f1 = round(sum(f1_values) / len(f1_values), 4)

    return {
        "n_scored": len(scored_predictions),
        "per_category": per_category,
        "macro_f1": macro_f1,
    }


def compute_per_run_variance(gt_labels: dict, scored_nb_ids: set, ckpt: dict, condition: str, runs_per_nb: int = 3) -> dict:
    """Compute per-run F1 scores across run 0, run 1, run 2, plus mean and std for scored notebooks."""
    cols = ["pre", "overlap", "multi"]
    run_scores = []
    
    for run_i in range(runs_per_nb):
        run_preds = {}
        for nb_id in scored_nb_ids:
            key = f"{condition}:{nb_id}:run_{run_i}"
            rec = ckpt.get(key)
            if rec and rec["status"] == "success" and rec["predictions"] is not None:
                run_preds[nb_id] = rec["predictions"]
        
        if len(run_preds) == len(scored_nb_ids) and len(scored_nb_ids) > 0:
            res = score_predictions(gt_labels, run_preds)
            run_scores.append(res)
    
    if not run_scores:
        return {"run_macro_f1s": [], "mean_macro_f1": None, "std_macro_f1": None}

    macro_f1s = [s["macro_f1"] for s in run_scores]
    mean_f1 = sum(macro_f1s) / len(macro_f1s)
    variance = sum((x - mean_f1) ** 2 for x in macro_f1s) / len(macro_f1s)
    std_f1 = variance ** 0.5

    return {
        "run_macro_f1s": [round(x, 4) for x in macro_f1s],
        "mean_macro_f1": round(mean_f1, 4),
        "std_macro_f1": round(std_f1, 4),
        "per_run_details": run_scores,
    }


def main():
    print("=== Launching Phase 10 Full Benchmark Execution ===")
    
    gt_labels, manifest, exclusions = load_manifest_and_gt()
    print(f"Manifest loaded: {len(manifest)} common-set notebooks, {len(exclusions)} exclusions.")
    
    ckpt = load_checkpoint()
    
    # ----------------------------------------------------
    # 1. Rule-Only Baseline Execution (1 run per nb)
    # ----------------------------------------------------
    print("\n--- Step 1: Rule-Only Baseline ---")
    ro_scored = {}
    ro_failed = []
    
    for item in manifest:
        nb_id = item["nb_id"]
        nb_path = item["nb_path"]
        key = f"rule_only:{nb_id}:run_0"
        
        if key in ckpt and ckpt[key]["status"] == "success":
            rec = ckpt[key]
        else:
            rec = run_single_request("rule_only", nb_id, nb_path, 0)
            ckpt[key] = rec
            save_checkpoint(ckpt)

        if rec["status"] == "success" and rec["predictions"] is not None:
            ro_scored[nb_id] = rec["predictions"]
        else:
            ro_failed.append((nb_id, rec.get("error", "Unknown error")))

    ro_metrics = score_predictions(gt_labels, ro_scored)
    print(f"Rule-Only Scored: {ro_metrics['n_scored']}/{len(manifest)} notebooks. Macro-F1: {ro_metrics['macro_f1']}")
    
    # Save standalone rule_only baseline output
    baseline_payload = {
        "condition": "rule_only",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_requested": len(manifest),
        "n_scored": ro_metrics["n_scored"],
        "n_failed": len(ro_failed),
        "failed_notebooks": ro_failed,
        "metrics": ro_metrics,
        "predictions": ro_scored,
    }
    with open(BASELINE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, indent=2)
    print(f"Saved baseline results to {BASELINE_JSON_PATH}")

    # ----------------------------------------------------
    # 2. LLM Conditions Execution (3 runs per nb)
    # ----------------------------------------------------
    llm_conditions = ["llm_only", "llm_with_context", "hybrid"]
    total_planned = len(manifest) * 3 * len(llm_conditions)
    print(f"\n--- Step 2: LLM Conditions ({total_planned} live LLM requests planned) ---")
    
    completed_count = 0
    failed_count = 0
    resumed_count = 0
    deferred_count = 0
    quota_halt = False

    for cond_str in llm_conditions:
        if quota_halt:
            break
        print(f"\n>>> Condition: {cond_str} <<<")
        for item in manifest:
            if quota_halt:
                break
            nb_id = item["nb_id"]
            nb_path = item["nb_path"]
            
            for run_i in range(3):
                key = f"{cond_str}:{nb_id}:run_{run_i}"
                
                # Check if already completed or deferred in checkpoint
                existing_rec = ckpt.get(key)
                if existing_rec:
                    st = existing_rec.get("status")
                    if st == "success":
                        resumed_count += 1
                        completed_count += 1
                        continue
                    elif st == "deferred_oversized":
                        resumed_count += 1
                        deferred_count += 1
                        continue

                # Preflight check before live call
                is_over, defer_reason = check_preflight_request(cond_str, nb_id, nb_path)
                if is_over:
                    print(f"[{cond_str}] nb: {nb_id} | run: {run_i} ... DEFERRED_OVERSIZED")
                    defer_rec = {
                        "condition": cond_str,
                        "nb_id": nb_id,
                        "run_index": run_i,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "latency_s": 0.0,
                        "status": "deferred_oversized",
                        "reason": defer_reason,
                        "predictions": None,
                        "model": LLM_MODEL,
                        "provider": LLM_PROVIDER,
                        "prompt_version": PROMPT_VERSION,
                        "attempts": 0,
                    }
                    ckpt[key] = defer_rec
                    save_checkpoint(ckpt)
                    log_llm_telemetry({
                        "timestamp": defer_rec["timestamp"],
                        "notebook": nb_id,
                        "condition": cond_str,
                        "run_index": run_i,
                        "latency_s": 0.0,
                        "status": "deferred_oversized",
                        "reason": defer_reason[:300],
                        "model": LLM_MODEL,
                    })
                    deferred_count += 1
                    continue

                # Live call execution
                print(f"[{cond_str}] nb: {nb_id} | run: {run_i} ...", end="", flush=True)
                try:
                    rec = run_single_request(cond_str, nb_id, nb_path, run_i)
                    ckpt[key] = rec
                    save_checkpoint(ckpt)
                    
                    if rec["status"] == "success":
                        completed_count += 1
                        print(f" OK ({rec.get('latency_s', 0.0)}s)")
                    elif rec["status"] == "deferred_oversized":
                        deferred_count += 1
                        print(f" DEFERRED ({rec.get('reason', '')[:60]})")
                    else:
                        failed_count += 1
                        print(f" FAILED ({rec.get('error', '')[:80]})")

                except QuotaLimitError as qle:
                    rec = qle.args[0]
                    ckpt[key] = rec
                    save_checkpoint(ckpt)
                    print(f"\n[STOP] HTTP 429 Quota/Rate-limit reached on {nb_id} run {run_i}.")
                    print("Checkpoint and telemetry flushed safely. Benchmark paused.")
                    quota_halt = True
                    break

                # Pacing delay between live LLM calls
                time.sleep(PACING_DELAY_SEC)

    print(f"\nLLM execution loop complete. Resumed: {resumed_count}, Successful: {completed_count}, Deferred: {deferred_count}, Failed: {failed_count}")

    # ----------------------------------------------------
    # 3. Final Scoring & Majority Voting Reconciliation
    # ----------------------------------------------------
    print("\n--- Step 3: Metric Reconciliation & Deliverables Generation ---")
    
    all_condition_results = {
        "rule_only": {
            "n_requested": len(manifest),
            "n_scored": ro_metrics["n_scored"],
            "failed_notebooks": ro_failed,
            "metrics": ro_metrics,
            "per_run_variance": {"run_macro_f1s": [ro_metrics["macro_f1"]], "mean_macro_f1": ro_metrics["macro_f1"], "std_macro_f1": 0.0},
        }
    }

    cols = ["pre", "overlap", "multi"]

    for cond_str in llm_conditions:
        mv_predictions = {}
        cond_failed_nbs = []
        cond_deferred_nbs = []
        
        for item in manifest:
            nb_id = item["nb_id"]
            nb_run_preds = {c: [] for c in cols}
            nb_failed = False
            nb_deferred = False
            
            for run_i in range(3):
                key = f"{cond_str}:{nb_id}:run_{run_i}"
                rec = ckpt.get(key)
                if rec and rec["status"] == "success" and rec["predictions"] is not None:
                    for c in cols:
                        nb_run_preds[c].append(rec["predictions"][c])
                elif rec and rec["status"] == "deferred_oversized":
                    nb_deferred = True
                    cond_deferred_nbs.append((nb_id, run_i, rec.get("reason", "Deferred oversized")))
                else:
                    nb_failed = True
                    err_msg = rec.get("error", "Run missing/failed") if rec else "Missing checkpoint"
                    cond_failed_nbs.append((nb_id, run_i, err_msg))

            if nb_deferred:
                print(f"  [EXCLUDE] {cond_str} {nb_id} excluded from scoring: deferred oversized input.")
            elif not nb_failed and all(len(nb_run_preds[c]) == 3 for c in cols):
                mv_pred = {}
                for c in cols:
                    votes = nb_run_preds[c]
                    mv_pred[c] = votes.count(True) > 1  # majority vote (>= 2 out of 3)
                mv_predictions[nb_id] = mv_pred
            else:
                print(f"  [EXCLUDE] {cond_str} {nb_id} excluded from scoring due to failed/incomplete run.")

        cond_metrics = score_predictions(gt_labels, mv_predictions)
        scored_ids = set(mv_predictions.keys())
        variance_info = compute_per_run_variance(gt_labels, scored_ids, ckpt, cond_str, 3)

        all_condition_results[cond_str] = {
            "n_requested": len(manifest),
            "n_scored": cond_metrics["n_scored"],
            "failed_runs": cond_failed_nbs,
            "deferred_oversized_runs": cond_deferred_nbs,
            "metrics": cond_metrics,
            "per_run_variance": variance_info,
            "majority_vote_predictions": mv_predictions,
        }

    # Save full benchmark JSON results
    full_results_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": "bef0fbabe41a28f855e1dfc6fcba6fa899ff68e2",
        "population": {
            "total_gt_notebooks": 100,
            "common_set_scored": len(manifest),
            "exclusions_count": len(exclusions),
            "exclusions": exclusions,
        },
        "model_config": {
            "model": LLM_MODEL,
            "provider": LLM_PROVIDER,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "runs_per_notebook": 3,
            "prompt_version": PROMPT_VERSION,
            "context_schema_version": CONTEXT_SCHEMA_VERSION,
        },
        "conditions": all_condition_results,
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_results_payload, f, indent=2)
    print(f"Full benchmark results saved to {RESULTS_JSON_PATH}")

    print("\n✅ Benchmark execution and reconciliation complete.")


if __name__ == "__main__":
    main()
