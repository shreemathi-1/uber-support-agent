# Borrowed patterns

One line per adaptation, per CLAUDE.md rule 7.

- `pipeline/llm.py`: retry-with-exponential-backoff decorator shape from the tenacity docs, https://tenacity.readthedocs.io/en/latest/ (`retry_if_exception`, `stop_after_attempt`, `wait_exponential`, `reraise=True`).
- `scripts/eval_intents.py`: `ConfusionMatrixDisplay(...).plot(ax=...)` usage from the scikit-learn examples, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html.
- `baselines/simple.py`: TF-IDF + LogisticRegression pipeline shape from the scikit-learn "Working with text data" tutorial, https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html.
- `eval/kappa.py`: `sklearn.metrics.cohen_kappa_score`, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html.
- `eval/judge.py`: Gemini JSON-mode call (`response_mime_type="application/json"`, `system_instruction`) from the google-generativeai README, https://github.com/google-gemini/deprecated-generative-ai-python.
