# Phase 10 — Oversized Input Inventory Report

Generated: 2026-10-14T07:19:45.258776+00:00

## Configuration
- Model: `qwen/qwen3.8-27b` via Groq
- Context window (provider): 32,768 tokens
- Preflight safe input limit: 30,768 tokens
  (= context_window − max_output_tokens 1500 − overhead 500)
- Token estimator: `chars_per_4_bpe_proxy_v1` (4 chars/token BPE proxy)
- **All token counts are ESTIMATED — not provider-verified.**

## Population Summary
| Metric | Count |
|---|---|
| Ground-truth notebooks (86-notebook primary population) | 100 |
| Missing/unresolvable notebooks | 0 |
| Oversized inventory entries (all architectures) | 0 |
| Unique notebooks with ≥1 oversized request | 0 |
| Eligible (pending, not yet sent) entries | 200 |

## Oversized Notebooks

_No oversized notebooks detected._

## Existing Results Verification
- `yang_results_rule_only.json`: 100 notebooks scored
- `yang_results_llm_only.json`:  100 notebooks scored
- `llm_call_log.jsonl`: 0 successful LLM call entries
- Run manifest entries: 5

## Files
- `oversized_manifest.json` — full inventory (all notebooks × architectures)
- `oversized_manifest.csv` — same, CSV format
- `oversized_only.json` — oversized entries only
- `oversized_only.csv` — oversized entries only, CSV
- `phase10_report.md` — this report
