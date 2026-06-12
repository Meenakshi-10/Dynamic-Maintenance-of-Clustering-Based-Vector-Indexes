import csv
import matplotlib.pyplot as plt

index_ntotal, gt_ntotal, r1, r10, r100 = [], [], [], [], []

with open("recall_metrics.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        index_ntotal.append(int(row["index_ntotal"]))
        gt_ntotal.append(int(row["gt_ntotal"]))
        r1.append(float(row["R@1"]))
        r10.append(float(row["R@10"]))
        r100.append(float(row["R@100"]))

plt.figure(figsize=(10, 6))
plt.plot(gt_ntotal, r1,   label="R@1",   marker="o", markersize=3)
plt.plot(gt_ntotal, r10,  label="R@10",  marker="o", markersize=3)
plt.plot(gt_ntotal, r100, label="R@100", marker="o", markersize=3)

plt.axvline(x=index_ntotal[0], color="red", linestyle="--", alpha=0.7,
            label=f"Snapshot (index frozen at {index_ntotal[0]:,})")

plt.xlabel("Dataset size (gt_ntotal)")
plt.ylabel("Recall")
plt.title("Recall Degradation as Dataset Mutates (Frozen IVF4096,Flat  nprobe=64)\n"
          "R@X = fraction of true top-X neighbors recovered")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("recall_plot.png", dpi=150)
plt.show()
print("Saved recall_plot.png")