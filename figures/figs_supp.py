# -*- coding: utf-8 -*-
"""Figures of the supplementary material: FigS1 and FigS2 (the further comparison pages, in the cells of Figs. 10 and 11),
FigS3 and FigS4 (threshold ribbons of the other four models) and FigS5 (bars and bubbles of the last two models)."""
import json, os, sys
import figs_cases as FC, figs_3drow as R
from house import OUT
CUR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RULESIGHT_CURATED", "/tmp/exp5/curated")
cases = json.load(open(os.path.join(CUR, "cases.json"), encoding="utf-8")); A, B = FC.choose(cases); used = {c["sid"] for c in A + B}
n = lambda c, s: sum(1 for m in FC.MODELS if c["models"].get(m) and c["models"][m][s] and c["models"][m][s]["correct"])
pool = [c for c in cases if c["models_compared"] == 6 and c["sid"] not in used and (c["ground_truth"] or c["sub"] == "defect_free") and n(c, "rulesight") > n(c, "zero_shot")]
pool = sorted(pool, key=lambda c: (c["sub"] != "defective", c["task"], -(n(c, "rulesight") - n(c, "zero_shot")), c["sid"]))[:18]
L = "abcdefghijkl"
for k, c in enumerate(pool): FC.cell(c, CUR, ("FigS1" if k < 12 else "FigS2") + L[k % 12], L[k % 12])
json.dump([["FigS1", min(12, len(pool))], ["FigS2", max(0, len(pool) - 12)]], open(os.path.join(OUT, "supp_cases.json"), "w"))
for k, (m, st, t) in enumerate((("qwen3vl8b", "s0", "Qwen3-VL-8B, zero-shot"), ("qwen3vl8b", "s3nomeas", "Qwen3-VL-8B, RuleSight"), ("gpt54mini", "s0", "GPT-5.4 mini, zero-shot"), ("gpt54mini", "s3nomeas", "GPT-5.4 mini, RuleSight"))): R.ribbons(f"FigS3{L[k]}", m, st, t, L[k])
for k, (m, st, t) in enumerate((("gemini38flash", "s0", "Gemini 3.8 Flash, zero-shot"), ("gemini38flash", "s3nomeas", "Gemini 3.8 Flash, RuleSight"), ("internvl35", "s0", "InternVL3.5-8B, zero-shot"), ("internvl35", "s3nomeas", "InternVL3.5-8B, RuleSight"))): R.ribbons(f"FigS4{L[k]}", m, st, t, "efgh"[k])
R.bars("FigS5a", "gemini38flash", "Gemini 3.8 Flash", "a"); R.bars("FigS5b", "internvl35", "InternVL3.5-8B", "b"); R.bubbles("FigS5c", "gemini38flash", "Gemini 3.8 Flash", "c"); R.bubbles("FigS5d", "internvl35", "InternVL3.5-8B", "d")
print("supplementary figures written", len(pool))
