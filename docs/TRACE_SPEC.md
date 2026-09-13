# Trace spec

Every request through `pipeline.run` builds one `trace` dict and stores it as JSON in `tickets.trace`.
It is a record of what each step saw and produced. It never feeds back into a decision.
The Trace tab (`frontend/src/Trace.jsx`) renders it; `GET /traces/{ticket_id}` returns it.

Keys, in pipeline order. The first eight always exist; `baselines` is extra.

## `request`
| field | type | meaning |
|---|---|---|
| message | str | the customer message, untouched |
| history | list[{role, content}] | the ≤2 turns sent with it |
| conversation_id | str | given or generated |
| received_at | str | UTC ISO-8601, seconds |

## `enrich`  (LLM step, see "LLM step fields" below)
| field | type | meaning |
|---|---|---|
| parsed | Enrichment as dict | what the rules and the drafter received |
| retried | bool | true when the first answer failed validation and one retry was made |

## `rules`
| field | type | meaning |
|---|---|---|
| reason | str \| null | the first rule that fired (`rules.evaluate`), null = auto |
| rows | list | one row per rule in `rules.RULES`, in order, whether or not an earlier rule already fired |
| rows[].rule | str | rule name |
| rows[].fired | bool | the predicate result for this message |
| rows[].value | any | the value the rule looked at (flag, amount, urgency, confidence, intent) |
| rows[].threshold | str | the condition, as text (`> 10`, `< 0.7`, `== high`, `is true`, `not in AUTO_ALLOW`) |
| rows[].margin | float 0..1 | how far the value is from firing; 0 = fired, 1 = nowhere near. Booleans are 0 or 1 |
| rows[].decisive | bool | true on the one row `evaluate()` returned |

## `retrieve`
| field | type | meaning |
|---|---|---|
| ran | bool | false when a rule fired first or intent = other |
| latency_ms | int | embed + both Chroma queries |
| intent_filter | str | the intent used for the filtered query |
| max_fact_distance | float | the drafter's cutoff for help chunks (0.6) |
| pairs | list | `{id, distance, source, intent, customer_text, reply_text}` nearest first |
| chunks | list | `{id, title, distance, intent, source_url, text, usable}`; usable = distance ≤ cutoff |

## `draft`  (LLM step)
| field | type | meaning |
|---|---|---|
| raw_output | str \| null | exactly what the model returned |
| final_reply | str \| null | after `draft.postprocess`; the canned line for intent = other |
| post_process_notes | list[str] | which post-processing steps changed the text (`stripped_urls`, `collapsed_whitespace`, `added_repeat_contact_prefix`, `cut_to_280_chars`) |
| canned | bool | true for the fixed intent = other reply (no LLM) |

## `check`  (LLM step)
| field | type | meaning |
|---|---|---|
| safe | bool \| null | null when the check did not run |
| why | str \| null | the checker's sentence, `empty_draft`, or `check_error: ...` |

## `action`
| field | type | meaning |
|---|---|---|
| action | "auto" \| "escalate" | |
| reason | str | rule name, `unsafe_draft`, or `auto` |
| reply | str | the text the customer saw (escalation template with ticket reference, or the draft) |
| ticket_id | int | `tickets.id` |
| escalation_id | int \| null | `escalations.id` when escalated |

## `totals`
| field | type | meaning |
|---|---|---|
| llm_calls | int | sum over enrich, draft, check |
| cache_hits | int | of those, answered from `llm_cache` |
| prompt_tokens | int | estimated, see token_counter |
| completion_tokens | int | estimated |
| latency_ms | int | whole pipeline, baselines excluded |
| estimated_paid_cost_usd | float | what the LLM calls would cost at Groq list price (`pipeline/pricing.py`); the free tier pays 0 |
| token_counter | str | how tokens were counted. Currently `chars/4 estimate`: the cache stores no usage and tiktoken needs a network download (DECISIONS.md #45) |

## `baselines`  (not one of the eight pipeline steps)
| field | type | meaning |
|---|---|---|
| trivial | dict | `baselines.trivial.predict_one(message)`: `{available, intent, escalate, reason, reply}` |
| simple | dict | `baselines.simple.predict_one(message)`: same shape, or `{available: false, why}` until `eval/results/simple_intent.pkl` exists |
| latency_ms | int | time spent on the baselines, outside `totals.latency_ms` |

## LLM step fields (shared by enrich, draft, check)
| field | type | meaning |
|---|---|---|
| ran | bool | the step executed (a step can run with zero LLM calls: canned draft, empty-draft check) |
| model | str \| null | model id of the last call |
| llm_calls | int | calls made through `cache.cached_llm` in this step (enrich can make 2 on a validation retry) |
| cache_hits | int | of those, cache hits |
| prompt_tokens / completion_tokens | int | estimated |
| estimated_paid_cost_usd | float | per-call price × tokens, summed |
| latency_ms | int | wall clock for the step |
| raw_output | str \| null | the last call's response text |
