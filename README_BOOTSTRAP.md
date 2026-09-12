# Bootstrap files — provisional data so the pipeline can be built now

Copy into the repo:

    pipeline/taxonomy.py          -> repo/pipeline/taxonomy.py
    data/help_center/*.md         -> repo/data/help_center/

Everything here is PROVISIONAL and marked so:

- `taxonomy.py` — 8 intents guessed from a first look at the data. Revise after reading
  100 threads. `AMOUNT_LIMIT` and `AUTO_ALLOW` are guesses; set them from golden-set results.
- `help_center/*.md` — 12 articles written from general knowledge of Uber's rider flows,
  NOT copied from help.uber.com. `date_checked: UNVERIFIED-PLACEHOLDER` on every file.
  Replace each with a hand-checked paraphrase of the real article and set the date.

The pipeline treats these as ordinary inputs, so swapping them later needs only `make index`.
Add a line to docs/DECISIONS.md: "Built the pipeline on a provisional taxonomy and 12
placeholder help articles so evaluation tooling existed before the manual data work;
replaced on <date>."
