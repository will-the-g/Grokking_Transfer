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

import pandas as pd, numpy as np, json, glob, os
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)

# ---- Table 1
rows=[]
d=pd.read_csv(str(RESULTS/"summary_runs_nseed.csv"))
t=d[(d.run_kind=="seed")&(d.stage=="transfer")&(d.task=="affine")&(d.init.isin(["memorized","norm_control","transition","grokked"]))]
for init,g in t.groupby("init"):
    for sd,v in zip(g.seed,g.final_dev_mean): rows.append(("p=59, one layer",init,sd,v))
for name,lab in [("table1/runs_nseed_p97","p=97, one layer"),("table1/runs_nseed_L2","p=59, two layers")]:
    for f in sorted(glob.glob(str(RESULTS / (name + "/seed_*/summary.json")))):
        j=json.load(open(f)); 
        for k,v in j["transfers"].items():
            task,init=k.split("|")
            if task=="affine" and init in ["memorized","norm_control","transition","grokked"] and "final_dev_mean" in v:
                rows.append((lab,init,j["seed"],v["final_dev_mean"]))
pl=pd.concat([pd.read_csv(str(RESULTS/"table1/p1_preln.csv")),pd.read_csv(str(RESULTS/"table1/k2_preln.csv"))])
for _,r in pl.iterrows(): rows.append(("p=59, Pre-LN",r.stage,int(r.seed),r.transfer_dev_mean))
T1=pd.DataFrame(rows,columns=["setting","source","seed","transfer"])
t1=T1.groupby(["setting","source"]).transfer.agg(["mean","min","max","count"]).round(3)
print("TABLE 1"); print(t1.to_string()); T1.to_csv(str(FIGS_OUT/"table1_data.csv"),index=False)
# ---- Figure 2 data (specificity)
sp=[]
for dname in ["specificity","specificity_aged","specificity_aged_init100","specificity_aged_init200"]:
    for f in sorted(glob.glob(str(RESULTS / ("specificity/" + dname + "/*/summary.json")))):
        j=json.load(open(f))
        for k,v in j.get("transfers",{}).items():
            if "final_dev_mean" in v: sp.append((dname,os.path.basename(os.path.dirname(f)),k,v["final_dev_mean"]))
        # the specificity runs write their transfers under `jobs`, not under
        # `transfers`: the plain runs hold one value per source, the aged ones
        # hold one value per stage
        for k,v in (j.get("jobs") or {}).items():
            if "final_dev_mean" in v: sp.append((dname,os.path.basename(os.path.dirname(f)),k,v["final_dev_mean"]))
            for stage,sv in (v.get("stages") or {}).items():
                if "final_dev_mean" in sv: sp.append((dname,os.path.basename(os.path.dirname(f)),k+"_"+stage,sv["final_dev_mean"]))
SP=pd.DataFrame(sp,columns=["dir","run","transfer_key","final_dev"])
print("SPECIFICITY"); print(SP.groupby(["dir","transfer_key"]).final_dev.agg(["mean","min","max","count"]).round(3).to_string())
SP.to_csv(str(FIGS_OUT/"fig2_data.csv"),index=False)
# ---- Table 3 (queue13 matrix)
b=pd.read_csv(str(RESULTS/"table3/b1_matrix.csv"))
fin=b.sort_values("step").groupby(["source_task","target_task","seed","source_kind"]).dev_acc.last().reset_index()
m=fin.groupby(["source_task","target_task","source_kind"]).dev_acc.agg(["mean","count"]).round(3).unstack("source_kind")
print("TABLE 3 (queue13)"); print(m.to_string())
print(b.groupby(['source_task','target_task']).seed.nunique().to_string())
