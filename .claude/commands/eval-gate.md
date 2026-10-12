Before merging any change to routing/model_tiers.py, routing/model_router.py, or any agent prompt:

1. Run observability/judge_eval.py against the saved topic set in eval/topics/.
2. Compare the LLM-as-Judge scores (citation accuracy, coverage, hallucination rate) against the
   last known-good baseline stored in eval/baseline_scores.json.
3. If any score regresses, report which topic/dimension regressed and do NOT recommend shipping.
4. If all scores are equal or better, update eval/baseline_scores.json with the new results and
   report a summary table.

This is a hard gate per CLAUDE.md — do not skip it for "small" prompt tweaks.
