# RuleSight

RuleSight is a test oracle for GUI display defects. It reads one screenshot and returns a JSON report that names each defect, its category, its bounding box, the rule it violates and the evidence for it. One call of a multimodal large language model (MLLM) is grounded in three components:

* **Rule and exemplar retrieval (RER)** selects the rule cards in scope and three labeled exemplars.
* **Measurement routing (MR)** runs deterministic tools (OCR, element boxes, contrast, visual angle, color vision and glare simulation) and passes their outputs only to the quantitative rules that need them.
* **Rule-grounded verification (RGV)** keeps a reported defect only if it belongs to a rule in scope and no measurement contradicts it.

This repository holds the code, the rule knowledge base RuleKB-GUI, the prompts, every stored answer of the evaluation and the scripts that recompute the tables and figures of the paper.

## Layout

| Path | Content |
|---|---|
| `rulesight/` | the oracle: `run_api.py` (prompting and calls), `measure.py` (tools), `verify.py` (RGV), `score.py` (scoring), `fetch_data.py`, `precompute.py` and the dataset helpers |
| `rulesight/rulekb/rules.json` | RuleKB-GUI v2.2: 8 display rules (D1 to D8) and 8 ergonomic rules (E1 to E8) with criteria and thresholds |
| `rulesight/prompts/` | system prompt and the prompts of every configuration |
| `rulesight/prompts_uilens/` | the per-category prompts published with UI-Lens, used by the UI-Lens prompting baseline (see `SOURCE.md` there) |
| `data/records.csv` | one scored answer per model, configuration and instance (53,510 rows) |
| `data/answers.jsonl.gz` | the raw model answers with token usage, cost and latency |
| `data/splits.json` | the development split (300), the test split (3,635) and the 400-instance subset of UI-Lens, seed 20261101 |
| `analysis/reproduce_tables.py` | recomputes the main tables from the stored answers and checks them against the paper |
| `analysis/check_prompt_version.py` | confirms that the released prompts and rule cards are the ones that produced the stored answers |
| `figures/` | plotting scripts and the data behind the result figures (Figs. 6 to 17) and the supplementary figures |

## Reproduce the tables

```bash
pip install numpy
python analysis/check_prompt_version.py    # prints pv3-2198a396 ... matches the stored answers
python analysis/reproduce_tables.py        # prints Tables 4, 5, 6 and 9 and the non-inferiority test, and ends with "36 of 36 checks against the paper passed"
```

Configuration keys in the data: `s0` zero-shot, `bcot` chain-of-thought, `uilens` UI-Lens prompting, `s1` few-shot, `s2` rule cards and exemplars, `s3nomeas` RuleSight, `s3` RuleSight without routing, `s3norule` without rule cards, `s3noex` without exemplars, `s3raw` without routing and without the verifier. Cost and latency are averaged over the calls with recorded token usage.

## Run RuleSight on screenshots

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # any OpenAI-compatible endpoint works through --base-url
export HF_TOKEN=...                  # UI-Lens is downloaded from Hugging Face
python rulesight/fetch_data.py --root data/raw --get-ui-lens                  # writes data/raw/RuleSight_data.json, the frozen manifest
python rulesight/precompute.py --manifest data/raw/RuleSight_data.json        # OCR and measurement tables, cached per screenshot
python rulesight/run_api.py --manifest data/raw/RuleSight_data.json --model anthropic/claude-haiku-5.5 --tag haiku \
       --settings s0,s3nomeas --datasets ui_lens --out runs/haiku.jsonl
python rulesight/score.py --manifest data/raw/RuleSight_data.json --runs runs/haiku.jsonl --out runs/scores.json
```

Each script lists its options with `--help`. Open-source models are served locally by vLLM and reached through `--base-url`. UI-Lens, UICrit and the other benchmarks are downloaded from their original releases and are not redistributed here.

## Figures

`figures/make_all.py` redraws the result figures of the paper (Figs. 6 to 17) and the supplementary figures from the JSON files in `figures/`, and writes them to `figures/out/`. The case figures (Figs. 12 and 13 and Figs. S1 and S2) also need the UI-Lens screenshots; set `RULESIGHT_CURATED` to the folder that holds them. `figures/prep_extra.py` rebuilds `extra.json` from the stored records, and `figures/tables_extra.py` prints the tables behind the figures. The method figures (Figs. 1 to 5) are illustrations and are not generated from the data.

## License

Code and rule cards: MIT. The stored answers are released for research use; the benchmark screenshots remain under the licenses of their original releases.
