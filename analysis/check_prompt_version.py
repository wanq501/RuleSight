"""Recompute the prompt version of the released prompts and rule cards (pv3-2198a396 for every answer in data/)."""
import hashlib, re
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "rulesight"
pipeline = re.search(r'^PIPELINE = "([^"]*)"', (R / "run_api.py").read_text(encoding="utf-8"), flags=re.M).group(1)
prompts = {p.stem: p.read_text(encoding="utf-8") for p in (R / "prompts").glob("*.txt")}
pv = "pv3-" + hashlib.md5(("".join(prompts[k] for k in sorted(prompts)) + (R / "rulekb" / "rules.json").read_text(encoding="utf-8") + pipeline).encode("utf-8")).hexdigest()[:8]
print(pv, "matches the stored answers" if pv == "pv3-2198a396" else "differs from the stored answers (pv3-2198a396)")
