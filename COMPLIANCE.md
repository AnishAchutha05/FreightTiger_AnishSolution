# Case Study Compliance Checklist

This checklist maps the FreightTiger case-study requirements to the project implementation and its supporting artifacts.

| Requirement | Implementation / evidence | Status |
|---|---|---|
| Group shipments by directed route and supplied `route_type` | `src/aggregate.py`; `outputs/weekly_aggregation.csv` | Implemented |
| Use Monday–Sunday weeks, with `week_of` as Monday | `src/aggregate.py`; `tests/test_core.py` | Implemented |
| Calculate weekly cost using total freight cost divided by total tonne-km | `src/aggregate.py`; numerical reference checks | Implemented |
| Compare with the preceding up-to-eight weeks, excluding the current week and without look-ahead | `src/baselines.py`; `tests/test_core.py`; `outputs/weekly_analysis.csv` | Implemented |
| Compare with the same-week average of other routes sharing the supplied route type | `src/baselines.py`; `tests/test_core.py` | Implemented |
| Identify unusual increases with a documented numerical rule | `src/flagging.py`; `src/config.py` | Implemented; 15% thresholds are heuristics |
| Retrieve context notes using embeddings, LangChain and a persistent ChromaDB store | `src/notes.py`, `src/retrieve.py`, `data/notes_structured.json` | Implemented |
| Verify supporting notes against route, date window, cost effect and exclusions | `src/verify.py`; `eval/run_eval.py` | Implemented |
| Cite the exact note for justified increases; leave note ID blank when unexplained | `src/report.py`; `outputs/output.csv` | Implemented |
| Generate concise evidence-grounded explanations and use a deterministic fallback | `src/explain.py` | Implemented |
| Match the required submission CSV schema and column order | `src/config.py`, `src/report.py`, `outputs/output.csv` | Implemented |
| Provide a lightweight labelled evaluation | `eval/labelled.csv`, `eval/run_eval.py`, `eval/EVAL_REPORT.md` | Implemented; limited regression set |
| Show repeatability across three runs | `scripts/repro_check.py`, `repro/REPRO_REPORT.md`, `repro/` | Reported pass for recorded runs |
| Record model calls, token usage and cost information | `logs/token_log.json` | Implemented; hosted comparison depends on its documented published price and date |
| Provide clear documentation and a walkthrough-ready architecture | `README.md`, `assets/architecture.png`, `assets/flowchart.svg` | Included |
| Optional natural-language Q&A over findings | Not included | Stretch goal deferred |

## Validation commands

Run these commands from the repository root after installing dependencies and starting Ollama:

```bash
python -m pytest -q
PYTHONPATH=. python eval/run_eval.py
PYTHONPATH=. python scripts/repro_check.py