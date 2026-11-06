# RuleSight executor (P3)

Runs the experiments of docs/RULESIGHT_DESIGN.md §13 (v8). The console (P3 project in Atelier) imports `RuleSight_data.json` and `results.json`.

## Files
- `rulekb/rules.json` — RuleKB-GUI v2: 16 rule cards (D1–D8 display defects, E1–E8 ergonomic criteria) with decisions, thresholds, sources,
  and `category_map` (UI-Lens tasks / UICrit wording / GLIB → rules). Exemplars come from Rico-synth only, never from a test set.
- `prompts/` — system prompt and the templates of S0, B-CoT, S1, S2, S3 (JSON output schema is fixed).
- `sources.py` — the data sources; `python sources.py check` tells what this machine can reach and whether HF_TOKEN works; `RemoteZip` reads single
  files out of a zip on a server with range requests (GLIB on Zenodo, 10 GB, of which we need ~400 files).
- `fetch_data.py` — `--get-ui-lens` (HuggingFace, gated → HF_TOKEN), `--get-rico` (Google Storage tarball streamed, or HuggingFace parquet),
  then builds the manifest: UI-Lens (Dev 300 pages / Test, UI-Lens-400 subset), UICrit (critiques → rule ids + boxes), GLIB (partial), Rico-synth, ErgoHMI, F-Wild; md5 of everything.
- `inject_defects.py` — injects D1–D8 into Rico screens using their view hierarchies; exact boxes.
- `measure.py` — measurement table of one screenshot: OCR boxes (PaddleOCR, fallback edge boxes), element boxes, WCAG contrast, px→mm/arcmin, CVD and glare simulation, geometric facts for D1/D2/D4/D5/D6/D7/D8.
- `verify.py` — the verifier: ergonomic claims need a threshold-crossing measurement, display claims a geometric or textual fact (D5 may rest on the model); drops the rest as "suspected".
- `run_api.py` — runner for every setting (S0, B-CoT, B-Zoom, S1, S2, S3, four ablations, two injections); OpenRouter or any OpenAI-compatible endpoint (`--base-url http://127.0.0.1:8000/v1` for vLLM); resume; `--dry-run` exercises the pipeline with a stub.
- `score.py` — T1 P/R/F1, T2 type macro-F1, T3 AP@0.5, T4 rule / evidence rates, cost and latency; UI-Lens per-task P/R/F1 (comparable with the UI-Lens paper); UICrit per-rule P/R/F1; paired bootstrap with Holm; H2 non-inferiority; writes `results.json` for the console.
- `fwild_capture.py` — F-Wild screenshots (50 sites × 3 viewports) with Playwright.

## On AutoDL — automatic (as ErgoHMI)
Nothing is typed on AutoDL. `Atelier.bat` / `AutoDL.bat` uploads `autodl/rulesight.zip.b64` with setup.sh and status.py; `setup.sh start` installs the
executor in the background whenever the upload changed (its own venv `rs_venv`, PaddleOCR included; about 2 minutes). The console's P3 pages carry an
"AutoDL 上的执行器" card that talks to the status server through the tunnel (`/rs/status`, `/rs/run`, `/rs/stop`, `/rs/log`, `/rs/manifest`, `/rs/results`):
- HuggingFace token: pasted once into the card (kept with the other global settings); the OpenRouter key is the one in the workbench.
- 一键准备 = `rs.sh prepare` (check → fetch → dev); 一键正式运行 = `rs.sh formal` (trial → main → local); every step also has its own button.
- The manifest is imported into the console when `fetch` produced a new one; the results after every scoring step.
- Budget: `ledger.json` in `rs_data/` holds the spend of every run; a run stops before the total passes the project's budget (100 USD).
Files on the instance: `/root/autodl-tmp/atelier/rulesight` (code), `rs_data/` (data, runs, results, ledger, .env with the keys, mode 600), `rs.log`, `rs_install.log`.

After any change to this folder run `python autodl/build_rulesight.py` (fixed timestamps: the same content gives the same md5, so no needless reinstall).

## By hand (fallback only)
```bash
tmux kill-session -t rulesight 2>/dev/null; pkill -f run_api.py; pkill -f fetch_data.py
cd /root/autodl-tmp/atelier && bash setup.sh rs-install && bash setup.sh rs-run prepare && tail -f rs.log
```
