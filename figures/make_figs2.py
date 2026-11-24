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
from matplotlib.lines import Line2D
BLUE="#2f6fdb"; ORANGE="#b85c00"; DORANGE="#6b2f00"; GREY="#8a8a8a"; LGREY="#c9c9c9"; INK="#222222"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#999999","axes.linewidth":0.8,
  "axes.spines.top":False,"axes.spines.right":False,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
  "axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","figure.dpi":150,"savefig.dpi":300,
  "grid.color":"#e6e6e6","grid.linewidth":0.6,"legend.frameon":False})
OUT=str(FIGS_OUT)+"/"
def title(fig,txt,y=0.99): fig.text(0.01,y,txt,ha="left",va="top",fontsize=10,fontweight="bold",color=INK)

# ---------- Figure 4 ----------
t=pd.read_csv(str(RESULTS/"fig4/p18_traj.csv")); d=pd.read_csv(str(RESULTS/"fig4/p18_direct.csv"))
b=t[(t.stage2_target=="add2")&(t.stage2_budget==600)]
fig,axs=plt.subplots(1,2,figsize=(7.4,3.7),sharey=True)
for ax,(v,ttl,mod) in zip(axs,[("C","Pythia-410M, p=59 (5 seeds)","410m"),("D","Pythia-160M, p=97 (5 seeds)","160m")]):
    g=b[(b.variant==v)&(b.source=="gen")]; m=b[(b.variant==v)&(b.source=="mem")]
    for sd,gg in g.groupby("seed"):
        gg=gg.sort_values("step"); ax.plot(gg.step,gg.stage2_acc,color=ORANGE,alpha=0.16,lw=0.8)
    gm=g.groupby("step")[["train_acc","heldout_acc","stage2_acc"]].mean()
    ax.plot(gm.index,gm.train_acc,color=GREY,lw=1.2,ls="--")
    ax.plot(gm.index,gm.heldout_acc,color=BLUE,lw=1.8)
    ax.plot(gm.index,gm.stage2_acc,color=ORANGE,lw=1.8)
    mm=m.groupby("step").stage2_acc.mean(); ax.plot(mm.index,mm.values,color=GREY,lw=1.2)
    dd=d[(d.model==mod)&(d.target=="add2")].acc.max(); ax.axhline(dd,color=GREY,lw=0.8,ls=":")
    ax.set_title(ttl,fontsize=9); ax.set_xlabel("stage-1 step"); ax.set_xlim(0,5500); ax.set_ylim(0,1.02); ax.grid(axis="y")
axs[0].set_ylabel("accuracy")
axs[0].annotate("one seed diverged\nat step 4,500",(4500,0.80),xytext=(2150,0.54),fontsize=7,color=GREY,arrowprops=dict(arrowstyle="-",color=LGREY,lw=0.8))
h=[Line2D([0],[0],color=GREY,ls="--",lw=1.2),Line2D([0],[0],color=BLUE,lw=1.8),Line2D([0],[0],color=ORANGE,lw=1.8),
   Line2D([0],[0],color=ORANGE,lw=0.8,alpha=0.35),Line2D([0],[0],color=GREY,lw=1.2),Line2D([0],[0],color=GREY,ls=":",lw=0.9)]
l=["train accuracy","held-out accuracy","transfer to a+2b (mean)","transfer, single seeds","transfer from memorizing source","direct training (max over budgets)"]
fig.legend(h,l,loc="lower center",ncol=3,fontsize=7.5,bbox_to_anchor=(0.5,-0.01))
title(fig,"Transfer rises with held-out accuracy; memorizing sources and direct training stay at chance")
fig.tight_layout(rect=(0,0.1,1,0.93)); fig.savefig(OUT+"fig4_pythia_trajectories.png"); fig.savefig(OUT+"fig4_pythia_trajectories.pdf"); plt.close(fig)

# ---------- Figure 5 ----------
e20=pd.read_csv(str(RESULTS/"fig5/m20_edits.csv")); e19=pd.read_csv(str(RESULTS/"fig5/m19_edits.csv"))
s=e20[(e20.step==5000)&(e20.stage2_lr==3e-4)].copy(); s["e"]=(s.edit_energy_in+s.edit_energy_out)/2
def agg(df,col):
    g=df.groupby("edit").agg(m=(col,"mean"),sd=(col,"std"),n=(col,"size"),e=("e","mean")); g["se"]=g.sd/np.sqrt(g.n); return g
H=agg(s,"heldout_acc"); T=agg(s,"stage2_acc")
key=["key_e25_both","key_e50_both","key_e75_both","key_e100_both"]; non=["rank6_10_both","rank6_15_both"]
fig,axs=plt.subplots(1,3,figsize=(10.4,3.8))
for ax,A,ylabel,lab,col,ylim in [(axs[0],H,"held-out accuracy","(a) source held-out",BLUE,1.0),(axs[1],T,"transfer to a+2b","(b) transfer",ORANGE,0.8)]:
    xs=[0]+[A.loc[k,"e"] for k in key]; ys=[A.loc["none","m"]]+[A.loc[k,"m"] for k in key]; ses=[A.loc["none","se"]]+[A.loc[k,"se"] for k in key]
    ax.errorbar(xs,ys,yerr=ses,color=col,marker="o",ms=5,lw=1.6,capsize=2)
    xn=[A.loc[k,"e"] for k in non]; yn=[A.loc[k,"m"] for k in non]; sn=[A.loc[k,"se"] for k in non]
    ax.errorbar(xn,yn,yerr=sn,color=GREY,marker="s",ms=5,lw=0,capsize=2,elinewidth=1)
    for xx,yy,k in zip(xn,yn,non):
        lab5=("5" if "10" in k else "10")
        off=(0,11) if col==BLUE else ((0,24) if lab5=="5" else (9,-3))
        ax.annotate(lab5,(xx,yy),xytext=off,textcoords="offset points",ha="center" if (col==BLUE or lab5=="5") else "left",fontsize=7.5,color=GREY)
    ax.set_xlabel("fraction of total energy removed"); ax.set_ylabel(ylabel); ax.set_ylim(0,ylim); ax.set_xlim(-0.02,0.58)
    ax.set_title(lab); ax.grid(axis="y")
x5=e19[(e19.step==5000)&(e19.stage2_lr==3e-4)&(e19.model=="160m")]
cats=[("none","intact"),("key_out","unembedding\nonly"),("key_in","input\nonly"),("key_both","both")]
hh=[x5[x5.edit==k].heldout_acc.mean() for k,_ in cats]; tt=[x5[x5.edit==k].stage2_acc.mean() for k,_ in cats]
ax=axs[2]; w=0.38; xs=np.arange(len(cats))
ax.bar(xs-w/2,hh,w,color=BLUE); ax.bar(xs+w/2,tt,w,color=ORANGE)
for i,(h_,t_) in enumerate(zip(hh,tt)):
    ax.text(i-w/2,h_+0.015,f"{h_:.2f}",ha="center",fontsize=7); ax.text(i+w/2,t_+0.015,f"{t_:.2f}",ha="center",fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels([c for _,c in cats],fontsize=8); ax.set_ylim(0,1.0); ax.set_ylabel("accuracy")
ax.set_title("(c) key frequencies deleted from"); ax.grid(axis="y")
h=[Line2D([0],[0],color=BLUE,marker="o",ms=5,lw=1.6),Line2D([0],[0],color=ORANGE,marker="o",ms=5,lw=1.6),
   Line2D([0],[0],color=GREY,marker="s",ms=5,lw=0),]
l=["held-out accuracy (key set removed: 0, 25, 50, 75, 100% of its energy)","transfer to a+2b (same edits)","non-key frequencies deleted (5 and 10; label = count)"]
fig.legend(h,l,loc="lower center",ncol=3,fontsize=7.5,bbox_to_anchor=(0.5,-0.01))
title(fig,"Deleting the key directions breaks the circuit; removing their energy or other frequencies does not")
fig.tight_layout(rect=(0,0.08,1,0.93)); fig.savefig(OUT+"fig5_frequency_edits.png"); fig.savefig(OUT+"fig5_frequency_edits.pdf"); plt.close(fig)

# ---------- Figure 6 ----------
l_=pd.read_csv(str(RESULTS/"fig6/p19_lr.csv"))
fig,axs=plt.subplots(1,2,figsize=(7.8,3.7))
ax=axs[0]; g=l_[l_.source=="gen"]; mm=l_[l_.source=="mem"]
styles={3e-5:("#9bb9ee",":"),1e-4:(ORANGE,"-"),3e-4:(DORANGE,"-"),1e-3:(LGREY,"-")}
for lr in [3e-5,1e-4,3e-4,1e-3]:
    gg=g[g.stage2_lr==lr].groupby("step").stage2_acc.mean(); c,ls=styles[lr]
    ax.plot(gg.index,gg.values,color=c,ls=ls,lw=1.8 if lr in (1e-4,3e-4) else 1.3,marker="o",ms=3,label=f"{lr:.0e}".replace("e-0","e-"))
mmm=mm.groupby("step").stage2_acc.mean(); ax.plot(mmm.index,mmm.values,color=GREY,lw=1,ls="--",label="memorizing source (all rates)")
ax.set_xlim(3900,8100); ax.set_ylim(0,0.9); ax.set_xlabel("stage-1 step"); ax.set_ylabel("transfer to a+2b"); ax.grid(axis="y")
ax.legend(title="stage-2 learning rate",title_fontsize=7.5,fontsize=7.5,loc="upper right",ncol=2)
ax.set_title("(a) transfer by learning rate")
ax=axs[1]; hh=g[g.stage2_lr==3e-4].groupby("step")[["heldout_acc","param_norm"]].mean()
ax.plot(hh.index,hh.heldout_acc,color=BLUE,lw=1.8,marker="o",ms=3); ax.set_ylim(0,1.12); ax.set_xlim(3800,8300)
for st,v,n in zip(hh.index,hh.heldout_acc,hh.param_norm): ax.text(st+(120 if st==4000 else 0),v+(-0.09 if st==4000 else 0.05),f"norm {n:.0f}",ha="center",fontsize=7,color=GREY)
ax.set_xlabel("stage-1 step"); ax.set_ylabel("held-out accuracy"); ax.set_title("(b) source held-out and weight norm"); ax.grid(axis="y")
title(fig,"The best stage-2 learning rate falls as the weight norm shrinks")
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(OUT+"fig6_stage2_lr.png"); fig.savefig(OUT+"fig6_stage2_lr.pdf"); plt.close(fig)
print("ok")
