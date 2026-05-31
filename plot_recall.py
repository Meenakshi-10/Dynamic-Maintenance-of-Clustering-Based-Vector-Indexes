import csv
import matplotlib.pyplot as plt

ntotal, r1, r10, r100 = [], [], [], []

with open("recall_metrics.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ntotal.append(int(row["ntotal"]))
        r1.append(float(row["R@1"]))
        r10.append(float(row["R@10"]))
        r100.append(float(row["R@100"]))

plt.figure(figsize=(10, 6))
plt.plot(ntotal, r1,   label="R@1",   marker="o", markersize=3)
plt.plot(ntotal, r10,  label="R@10",  marker="o", markersize=3)
plt.plot(ntotal, r100, label="R@100", marker="o", markersize=3)

plt.xlabel("Index size (ntotal)")
plt.ylabel("Recall")
plt.title("Recall vs Index Size (IVF4096,Flat  nprobe=64)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("recall_plot.png", dpi=150)
plt.show()
print("Saved recall_plot.png")
