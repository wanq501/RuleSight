# -*- coding: utf-8 -*-
"""Builds the result figures of the paper (Figs. 6 to 17) and the figures of the supplementary material.
The method figures (Figs. 1 to 5) are illustrations and are not generated from the data.
    python make_all.py "<export>/curated"      (the curated folder of the platform export)"""
import subprocess, sys, os
CUR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RULESIGHT_CURATED", "/tmp/exp5/curated")
HERE = os.path.dirname(os.path.abspath(__file__))
import figs_clean as C, figs_rich as RICH
RICH.fig6()                                                    # paper Fig. 5, per-instance confidence
C.fig7(); C.fig8(); C.fig9(); C.fig10(); C.fig11(); C.fig13()   # paper Figs. 6 to 9, 12 and 15
for cmd in (["figs_3drow.py"], ["figs_cases.py", "--curated", CUR], ["figs_supp.py", CUR]):
    subprocess.run([sys.executable] + cmd, check=True, cwd=HERE)
print("all figures built")
