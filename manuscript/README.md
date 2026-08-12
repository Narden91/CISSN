# Manuscript inputs

No manuscript claims are frozen before the publication grid completes. Generate the evidence appendix from the current result artifacts:

```powershell
uv run python scripts/generate_reproducibility_appendix.py --results_root ./results/publication --output ./manuscript/reproducibility_appendix.md
```

Use `RUNBOOK.md` as the sole protocol reference. Report the documented limitations on temporal dependence and distinguish conformalized from raw uncertainty intervals.
