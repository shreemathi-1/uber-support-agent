# Borrowed patterns

One line per adaptation, per CLAUDE.md rule 7.

- `pipeline/llm.py`: retry-with-exponential-backoff decorator shape from the tenacity docs, https://tenacity.readthedocs.io/en/latest/ (`retry_if_exception`, `stop_after_attempt`, `wait_exponential`, `reraise=True`).
- `scripts/eval_intents.py`: `ConfusionMatrixDisplay(...).plot(ax=...)` usage from the scikit-learn examples, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html.
