"""The figure and table scripts of the paper.

The inputs are read from results/ and the outputs are written to figs_out/, both
relative to the repository root. The numbers, the labels and the computations
are the ones of the paper.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGS_OUT = REPO / "figs_out"
FIGS_OUT.mkdir(parents=True, exist_ok=True)

import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BLUE="#2f6fdb"; INK="#222222"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#999999","axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","grid.color":"#e6e6e6","grid.linewidth":0.6})
P=[("p=59, one layer","fig1/sweep_summary_p59.csv"),("p=97, one layer","fig1/sweep_summary_p97.csv"),("p=59, two layers","fig1/sweep_summary_two_layers.csv")]
fig,axs=plt.subplots(1,3,figsize=(9.6,3.3),sharey=True)
for ax,(t,f) in zip(axs,P):
    d=pd.read_csv(RESULTS/f); r=spearmanr(d.source_dev_mean,d.final_dev_mean)[0]
    for s,g in d.groupby("seed"): ax.scatter(g.source_dev_mean,g.final_dev_mean,s=6,alpha=0.5,color=BLUE,lw=0)
    ax.set_title(f"{t} (n={len(d)}, rank corr {r:.2f})",fontsize=8.5); ax.set_xlabel("source held-out accuracy"); ax.grid(True)
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02); print(t,len(d),round(r,3))
axs[0].set_ylabel("transfer accuracy after 2,000 steps")
fig.text(0.01,0.99,"Transfer rises with the source's held-out accuracy, checkpoint by checkpoint",ha="left",va="top",fontsize=10,fontweight="bold")
fig.tight_layout(rect=(0,0,1,0.92))
fig.savefig(str(FIGS_OUT/"fig1_sweep.png"),dpi=150); fig.savefig(str(FIGS_OUT/"fig1_sweep.pdf"))
