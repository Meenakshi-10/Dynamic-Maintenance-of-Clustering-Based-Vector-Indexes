import csv
import matplotlib.pyplot as plt

gt_ntotal, frozen_ntotal = [], []
frozen = {"R@1": [], "R@10": [], "R@100": []}
mutable = {"R@1": [], "R@10": [], "R@100": []}

with open("recall_metrics.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gt_ntotal.append(int(row["gt_ntotal"]))
        frozen_ntotal.append(int(row["frozen_ntotal"]))
        frozen["R@1"].append(float(row["frozen_R@1"]))
        frozen["R@10"].append(float(row["frozen_R@10"]))
        frozen["R@100"].append(float(row["frozen_R@100"]))
        mutable["R@1"].append(float(row["mutable_R@1"]))
        mutable["R@10"].append(float(row["mutable_R@10"]))
        mutable["R@100"].append(float(row["mutable_R@100"]))

# Same color per metric (identity), solid vs dashed line for frozen vs mutable (status).
colors = {"R@1": "#2a78d6", "R@10": "#1baf7a", "R@100": "#008300"}

plt.figure(figsize=(10, 6))
for metric in ("R@1", "R@10", "R@100"):
    plt.plot(gt_ntotal, frozen[metric], label=f"Frozen {metric}",
              color=colors[metric], linestyle="-", linewidth=2)
    plt.plot(gt_ntotal, mutable[metric], label=f"Mutable {metric}",
              color=colors[metric], linestyle="--", linewidth=2)

plt.axvline(x=frozen_ntotal[0], color="red", linestyle=":", alpha=0.7,
            label=f"Snapshot (frozen index built at {frozen_ntotal[0]:,})")

plt.xlabel("Dataset size (gt_ntotal)")
plt.ylabel("Recall")
plt.title("Recall: Frozen vs. Mutable Index as Dataset Mutates (IVF4096,Flat  nprobe=64)\n"
          "R@X = fraction of true top-X neighbors recovered; solid = frozen, dashed = mutable")
plt.legend(ncol=2)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("recall_plot.png", dpi=150)
plt.show()
print("Saved recall_plot.png")