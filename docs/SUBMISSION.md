# Submission format and the scoring environment

Everything here comes from the task statement plus reports from participants in
the competition thread. The platform fails opaquely, so these are worth keeping
in one place.

## Archive

Build the archive with:

```bash
python scripts/make_submission.py --out submission.zip
python scripts/make_submission.py --check      # validate without building
```

Hard requirements, each of which has been reported to break a run:

| Requirement | Why |
| --- | --- |
| **ZIP** — not `tar`, not `tar.gz` | Both were reported to fail at stage 08. |
| **No wrapping directory** | The platform runs `<extracted>/scripts/predict.py`, so `pyproject.toml` and `scripts/` must be at the archive root. |
| **No `metrics/` directory** | Explicitly requested by the organisers. |
| **Directories `0755`, files `0644`** | A zip built on Windows, or via `zipfile`, can store directories as `0600`; the unpacker then cannot enter them and stage 08 fails. |
| **ASCII filename** | Upload tooling has been reported to mis-handle non-ASCII names. |
| **Under 40 GB** | The limit applies to the uploaded archive, not to the unpacked size. |

The archive ships the judge weights in `model/`: the scoring environment
provides nothing solution-specific, so weights must be inside. `--check` warns
when `model/` has no `.safetensors`/`.bin`.

## Predictions

**The scorer reads Parquet, even when `--output` ends in `.csv`.** The task text
says `predictions.csv`; the reader says otherwise, and honouring the extension
was the cause of stage 22 ("metric calculation") failures for several teams.
`write_predictions` therefore ignores the extension and writes Parquet to that
exact path. Set `GUARDIAN_OUTPUT_FORMAT=csv` for local debugging only.

Columns are exactly `id` and `label`, one row per input row, ids unchanged from
the input file.

## Runtime

* **vLLM is 0.19.1** on the scoring machine, A100 80 GB. `pyproject.toml`
  dependencies are **not** installed — there is no network at scoring time and
  no `pip install` step. `scripts/predict.py` therefore prepends `src/` to
  `sys.path`, and runs from an uninstalled checkout.
* Only `pandas`, `pyarrow`, `numpy` and the preinstalled torch/vLLM are
  available. Do not add a dependency that must be fetched.
* If inference does not fit in the time limit, the run fails and the attempt is
  **not** counted. It is safe to be generous with the budget.

## Data layout

Verified on the public sample:

* policies and tool catalogues are in **English**;
* the dialogue and the agent's answers are frequently in **Russian**;
* tool names, ids and values are Latin, never transliterated.

The rule engine and the judge rubric both account for this; see the language
note in `guardian.prompts`.

## Scoring rubric, as clarified by the organisers

* Reasoning text is **not** graded on its own: the action and the meaningful
  part of the response are graded, insofar as it affects the user.
* Making **several tool calls in one turn** is not a hallucination, even where a
  policy asks for one call at a time — it does not change what the user gets.
* Where the task description and the examples disagree on transcript syntax,
  **trust the examples**.
* Public and private label distributions match.