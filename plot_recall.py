import csv
import glob
import re

import matplotlib.pyplot as plt

METRICS = ("R@1", "R@10", "R@100")
COLORS = {"R@1": "#2a78d6", "R@10": "#1baf7a", "R@100": "#008300"}


def read_run_csv(path):
    gt_ntotal, frozen_ntotal = [], []
    frozen = {m: [] for m in METRICS}
    mutable = {m: [] for m in METRICS}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_ntotal.append(int(row["gt_ntotal"]))
            frozen_ntotal.append(int(row["frozen_ntotal"]))
            for m in METRICS:
                frozen[m].append(float(row[f"frozen_{m}"]))
                mutable[m].append(float(row[f"mutable_{m}"]))
    return gt_ntotal, frozen_ntotal, frozen, mutable


def plot_single_run(csv_path, png_path, title_suffix):
    gt_ntotal, frozen_ntotal, frozen, mutable = read_run_csv(csv_path)

    plt.figure(figsize=(10, 6))
    for metric in METRICS:
        plt.plot(gt_ntotal, frozen[metric], label=f"Frozen {metric}",
                  color=COLORS[metric], linestyle="-", linewidth=2)
        plt.plot(gt_ntotal, mutable[metric], label=f"Mutable {metric}",
                  color=COLORS[metric], linestyle="--", linewidth=2)

    plt.axvline(x=frozen_ntotal[0], color="red", linestyle=":", alpha=0.7,
                label=f"Snapshot (frozen index built at {frozen_ntotal[0]:,})")

    plt.xlabel("Dataset size (gt_ntotal)")
    plt.ylabel("Recall")
    plt.title("Recall: Frozen vs. Mutable Index as Dataset Mutates (IVF1024,Flat  nprobe=64)\n"
               f"{title_suffix}\n"
               "R@X = fraction of true top-X neighbors recovered; solid = frozen, dashed = mutable")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved {png_path}")


RESOURCE_COLUMNS = (
    "frozen_latency_ms", "mutable_latency_ms",
    "cpu_time_s", "cpu_delta_s",
    "frozen_cluster_mean", "frozen_cluster_std", "frozen_cluster_min", "frozen_cluster_max",
    "mutable_cluster_mean", "mutable_cluster_std", "mutable_cluster_min", "mutable_cluster_max",
)


def has_resource_columns(path):
    with open(path) as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    return all(c in fieldnames for c in RESOURCE_COLUMNS)


def read_resource_csv(path):
    gt_ntotal = []
    cols = {c: [] for c in RESOURCE_COLUMNS}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_ntotal.append(int(row["gt_ntotal"]))
            for c in RESOURCE_COLUMNS:
                cols[c].append(float(row[c]))
    return gt_ntotal, cols


def plot_resource_metrics(csv_path, png_path, title_suffix):
    gt_ntotal, c = read_resource_csv(csv_path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax = axes[0]
    ax.plot(gt_ntotal, c["frozen_cluster_std"], label="Frozen cluster-size std",
            color="#2a78d6", linewidth=2)
    ax.plot(gt_ntotal, c["mutable_cluster_std"], label="Mutable cluster-size std",
            color="#008300", linestyle="--", linewidth=2)
    ax.plot(gt_ntotal, c["frozen_cluster_max"], label="Frozen cluster-size max",
            color="#2a78d6", alpha=0.4, linewidth=1)
    ax.plot(gt_ntotal, c["mutable_cluster_max"], label="Mutable cluster-size max",
            color="#008300", alpha=0.4, linestyle="--", linewidth=1)
    ax.set_ylabel("Cluster size (vectors/list)")
    ax.set_title("Inverted-list size imbalance (std) and max list size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gt_ntotal, c["frozen_latency_ms"], label="Frozen search latency",
            color="#2a78d6", linewidth=2)
    ax.plot(gt_ntotal, c["mutable_latency_ms"], label="Mutable search latency",
            color="#008300", linestyle="--", linewidth=2)
    ax.set_ylabel("Latency (ms/query)")
    ax.set_title("Search latency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(gt_ntotal, c["cpu_delta_s"], label="CPU time since previous checkpoint",
            color="#c0392b", linewidth=2)
    ax.set_xlabel("Dataset size (gt_ntotal)")
    ax.set_ylabel("CPU time (s)")
    ax.set_title("Process CPU time consumed between checkpoints")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Resource metrics — {title_suffix}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path}")


def read_avg_csv(path):
    gt_ntotal, frozen_ntotal = [], []
    frozen_mean = {m: [] for m in METRICS}
    frozen_std = {m: [] for m in METRICS}
    mutable_mean = {m: [] for m in METRICS}
    mutable_std = {m: [] for m in METRICS}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_ntotal.append(int(row["gt_ntotal"]))
            frozen_ntotal.append(int(row["frozen_ntotal"]))
            for m in METRICS:
                frozen_mean[m].append(float(row[f"frozen_{m}_mean"]))
                frozen_std[m].append(float(row[f"frozen_{m}_std"]))
                mutable_mean[m].append(float(row[f"mutable_{m}_mean"]))
                mutable_std[m].append(float(row[f"mutable_{m}_std"]))
    return gt_ntotal, frozen_ntotal, frozen_mean, frozen_std, mutable_mean, mutable_std


def plot_average(csv_path, png_path, n_runs):
    (gt_ntotal, frozen_ntotal, frozen_mean, frozen_std,
     mutable_mean, mutable_std) = read_avg_csv(csv_path)

    plt.figure(figsize=(10, 6))
    for metric in METRICS:
        fm = frozen_mean[metric]
        fs = frozen_std[metric]
        mm = mutable_mean[metric]
        ms = mutable_std[metric]
        color = COLORS[metric]

        plt.plot(gt_ntotal, fm, label=f"Frozen {metric} (mean)",
                  color=color, linestyle="-", linewidth=2)
        plt.fill_between(gt_ntotal,
                          [v - s for v, s in zip(fm, fs)],
                          [v + s for v, s in zip(fm, fs)],
                          color=color, alpha=0.12, linewidth=0)

        plt.plot(gt_ntotal, mm, label=f"Mutable {metric} (mean)",
                  color=color, linestyle="--", linewidth=2)
        plt.fill_between(gt_ntotal,
                          [v - s for v, s in zip(mm, ms)],
                          [v + s for v, s in zip(mm, ms)],
                          color=color, alpha=0.12, linewidth=0)

    plt.axvline(x=frozen_ntotal[0], color="red", linestyle=":", alpha=0.7,
                label=f"Snapshot (frozen index built at {frozen_ntotal[0]:,})")

    plt.xlabel("Dataset size (gt_ntotal)")
    plt.ylabel("Recall")
    plt.title("Recall: Frozen vs. Mutable Index as Dataset Mutates (IVF1024,Flat  nprobe=64)\n"
               f"Average over {n_runs} runs (shaded band = ±1 std dev across runs)\n"
               "R@X = fraction of true top-X neighbors recovered; solid = frozen, dashed = mutable")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved {png_path}")


def main():
    run_paths = sorted(
        glob.glob("recall_metrics_run*.csv"),
        key=lambda p: int(re.search(r"run(\d+)\.csv$", p).group(1)),
    )

    if not run_paths:
        # Fall back to the legacy single-run filename.
        plot_single_run("recall_metrics.csv", "recall_plot.png", "Single run")
        return

    for path in run_paths:
        run_idx = re.search(r"run(\d+)\.csv$", path).group(1)
        plot_single_run(path, f"recall_plot_run{run_idx}.png", f"Run {run_idx}")
        if has_resource_columns(path):
            plot_resource_metrics(path, f"resource_plot_run{run_idx}.png", f"Run {run_idx}")

    avg_path = "recall_metrics_avg.csv"
    try:
        plot_average(avg_path, "recall_plot_avg.png", len(run_paths))
    except FileNotFoundError:
        print(f"No {avg_path} found, skipping average plot.")


if __name__ == "__main__":
    main()
