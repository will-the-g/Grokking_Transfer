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

import pandas as pd, numpy as np, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE="#2f6fdb"; ORANGE="#b85c00"; GREY="#8a8a8a"; LGREY="#c9c9c9"; INK="#222222"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#999999","axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
  "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","savefig.dpi":300,"grid.color":"#e6e6e6","grid.linewidth":0.6,"legend.frameon":False})
OUT=str(FIGS_OUT)+"/"
def title(fig,txt,y=0.99): fig.text(0.01,y,txt,ha="left",va="top",fontsize=10,fontweight="bold",color=INK)
# ---- Table 1 with Pre-LN combined
a=pd.read_csv(str(RESULTS/"table1/p1_preln.csv")); b=pd.read_csv(str(RESULTS/"table1/k2_preln.csv"))
pl=pd.concat([a,b]); print(pl.stage.value_counts().to_dict(), sorted(pl.seed.unique()))
print(pl.groupby("stage").transfer_dev_mean.agg(["mean","min","max","count"]).round(3))
# ---- S5 check
t5=pd.read_csv(str(RESULTS/"table3/t5_hard.csv"))
print(t5.columns.tolist()); 
x=t5[t5.source_task.astype(str).str.contains("S5|s5")] if "source_task" in t5.columns else t5
print(x.groupby([c for c in ["source_task","target_task","target_fraction","source_kind"] if c in x.columns]).dev_acc.mean().round(3).to_string()[:2500])
# ---- Figure 2 specificity (from summary.csv, p=59, affine target)
d=pd.read_csv(str(RESULTS/"summary_runs_nseed.csv"))
t=d[(d.run_kind.isin(["seed","specificity_aged","specificity"]))&(d.task=="affine")]
groups=[("addition source,\ngrokked",d[(d.run_kind=="seed")&(d.stage=="transfer")&(d.task=="affine")&(d.init=="grokked")].final_dev_mean.values,BLUE),
        ("copy-a source,\ngrokked",d[d.init=="from_copy_a_grok"].drop_duplicates(["seed","init"]).final_dev_mean.values,GREY),
        ("selector source,\ngrokked",d[d.init=="from_selector_grok"].drop_duplicates(["seed","init"]).final_dev_mean.values,GREY),
        ("copy-a source,\naged",np.append(d[d.init=="from_copy_a_aged"].drop_duplicates(["seed","init"]).final_dev_mean.values,0.8391442155309033),GREY),
        ("selector source,\naged",d[d.init=="from_selector_aged"].drop_duplicates(["seed","init"]).final_dev_mean.values,GREY)]
for n,v,c in groups: print(n.replace("\n"," "),len(v),np.round(v.mean(),3),np.round(v.min(),3),np.round(v.max(),3))
fig,ax=plt.subplots(figsize=(6.4,3.4))
rng=np.random.default_rng(0)
for i,(n,v,c) in enumerate(groups):
    ax.bar(i,v.mean(),0.6,color=c if c==BLUE else LGREY)
    ax.scatter(i+rng.uniform(-0.15,0.15,len(v)),v,s=10,color=INK,alpha=0.6,zorder=3)
    ax.text(i+0.34,v.mean()+0.01,f"{v.mean():.2f}",ha="left",fontsize=8)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups],fontsize=8)
ax.set_ylabel("affine-target accuracy after 2,000 steps"); ax.set_ylim(0,1.0); ax.grid(axis="y")
title(fig,"Copy and selector sources give no transfer (p=59; points are seeds)")
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(OUT+"fig2_specificity.png",dpi=150); fig.savefig(OUT+"fig2_specificity.pdf"); plt.close(fig)
# ---- Figure 3 LoRA checkpoint sweep (q12)
q=pd.read_csv(str(RESULTS/"fig3/pythia_ckpt_sweep.csv")); print(q.source.value_counts().to_dict(), q.revision.nunique(), q.model.nunique())
q["m"]=q.model.str.split("/").str[-1]
fig,axs=plt.subplots(1,2,figsize=(8.0,3.5),sharey=True)
for ax,mname in zip(axs,["pythia-1b","pythia-1.4b"]):
    g=q[q.m==mname]
    for src,col,lab in [("add",ORANGE,"generalizing source (a+b)"),("add_mem",GREY,"memorizing source (100 examples)")]:
        s=g[g.source==src]
        ax.scatter(s.heldout_ab_acc,s.gain_over_best_direct,s=9,color=col,alpha=0.35,lw=0)
        mm=s.groupby("revision").agg(h=("heldout_ab_acc","mean"),y=("gain_over_best_direct","mean")).sort_values("h")
        ax.plot(mm.h,mm.y,color=col,lw=1.6,marker="o",ms=4)
    ax.axhline(0,color=INK,lw=0.6); ax.axvspan(0.4,0.6,color="#eef2fb",zorder=0)
    ax.set_title("Pythia-"+mname.split("-")[1].upper()); ax.set_xlabel("source held-out a+b accuracy"); ax.grid(axis="y")
axs[0].set_ylabel("gain in a+2b accuracy over\nthe better direct baseline")
h=[Line2D([0],[0],color=ORANGE,marker="o",ms=4),Line2D([0],[0],color=GREY,marker="o",ms=4)]
fig.legend(h,["generalizing source (4,000 examples)","memorizing source (100 examples)"],loc="lower center",ncol=2,fontsize=8,bbox_to_anchor=(0.5,-0.01))
title(fig,"At matched held-out accuracy (shaded), only generalizing sources gain")
fig.tight_layout(rect=(0,0.08,1,0.93)); fig.savefig(OUT+"fig3_pythia_checkpoints.png",dpi=150); fig.savefig(OUT+"fig3_pythia_checkpoints.pdf"); plt.close(fig)
