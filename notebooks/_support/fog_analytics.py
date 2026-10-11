"""Saved-run analytics: tables, paired changes, per-class AP, density error and plots."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


BALANCED_METRICS = (
    "mAP50_95",
    "AP50",
    "AR100",
    "precision_at_025",
    "recall_at_025",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "serial_fps",
    "density_mean_ms",
)


def add_balanced_bin_rows(results):
    """Add equal-bin macro summaries without changing natural-distribution rows."""
    from fog_research import density_bin

    if "density_bin" not in results:

        def legacy_bin(value):
            if value in ("all", "unknown"):
                return value
            return density_bin(float(value))

        results["density_bin"] = results.density.map(legacy_bin)
    if "distribution" not in results:
        results["distribution"] = "natural"
    candidates = results[
        ~results.density.isin(["all", "unknown", "balanced"])
        & (results.density_bin != "unknown")
    ]
    summaries = []
    for keys, group in candidates.groupby(["model", "dataset", "arm", "seed"]):
        if group.density_bin.nunique() < 2:
            continue
        summary = group.iloc[0].to_dict()
        summary.update(
            density="balanced",
            density_bin="balanced",
            distribution="balanced_bins",
            images=int(round(group.images.mean())),
        )
        for column in BALANCED_METRICS:
            if column in group:
                summary[column] = float(group[column].mean())
        summaries.append(summary)
    if summaries:
        results = pd.concat([results, pd.DataFrame(summaries)], ignore_index=True)
    return results


def export_analytics(output, rows=None):
    output = Path(output)
    if rows is None:
        rows = json.loads((output / "input-manifest.json").read_text(encoding="utf-8"))
    frames, statuses = [], []
    for directory in sorted(output.iterdir()):
        if not directory.is_dir() or not (directory / "status.json").exists():
            continue
        status = json.loads((directory / "status.json").read_text())
        statuses.append(status)
        if status["status"] == "complete" and (directory / "results.csv").exists():
            frames.append(
                pd.read_csv(directory / "results.csv", dtype={"density": str})
            )
    pd.DataFrame(statuses).to_csv(output / "status.csv", index=False)
    if not frames:
        raise RuntimeError(
            f"No completed experiments under {output}. Check status.csv; resume training first."
        )
    results = pd.concat(frames, ignore_index=True)
    settings = {}
    settings_path = output / "settings.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    profile = settings.get(
        "profile", settings.get("config", {}).get("profile_name", "full")
    )
    if isinstance(profile, dict):
        profile = profile.get("name", "full")
    evaluation_split = settings.get("config", {}).get("evaluation_split", "test")
    if "profile" not in results:
        results["profile"] = profile
    if "evaluation_split" not in results:
        results["evaluation_split"] = evaluation_split
    if "planned_epochs" not in results:
        results["planned_epochs"] = settings.get("config", {}).get("epochs")
    if "epochs_trained" not in results:
        results["epochs_trained"] = np.nan
    if "stop_reason" not in results:
        results["stop_reason"] = "unknown"
    results = add_balanced_bin_rows(results)
    results.to_csv(output / "results.csv", index=False)
    keys = ["model", "dataset", "seed", "density"]
    control = results[results.arm == "rgb_control"][
        keys + ["mAP50_95", "mean_ms", "p95_ms"]
    ]
    paired = results[results.arm != "rgb_control"].merge(
        control, on=keys, how="inner", suffixes=("", "_control"), validate="many_to_one"
    )
    paired["delta_mAP"] = paired.mAP50_95 - paired.mAP50_95_control
    paired["delta_latency_ms"] = paired.mean_ms - paired.mean_ms_control
    paired.to_csv(output / "paired-deltas.csv", index=False)
    metric_names = [
        "mAP50_95",
        "AP50",
        "AR100",
        "precision_at_025",
        "recall_at_025",
        "mean_ms",
        "p95_ms",
        "serial_fps",
    ]
    seed_summary = results.groupby(["dataset", "model", "arm", "density"])[
        metric_names
    ].agg(["mean", "std", "count"])
    seed_summary.to_csv(output / "seed-summary.csv")
    by_id = {r["id"]: r for r in rows}
    density_rows = []
    for path in output.glob("density-*/test-predictions.csv"):
        estimates = pd.read_csv(path)
        for dataset, group in estimates.groupby("dataset"):
            error = group.predicted_beta - group.target_beta
            density_rows.append(
                {
                    "experiment": path.parent.name,
                    "dataset": dataset,
                    "images": len(group),
                    "MAE_beta": float(error.abs().mean()),
                    "RMSE_beta": float(np.sqrt((error**2).mean())),
                }
            )
    for path in output.glob("*-predicted_density-*/*-predictions.jsonl"):
        known = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                target = by_id[row["id"]]["beta"]
                if target is not None:
                    known.append((row["beta_input"], target))
        if known:
            error = np.array([p - t for p, t in known])
            density_rows.append(
                {
                    "experiment": path.parent.name,
                    "dataset": path.stem.replace("-predictions", ""),
                    "images": len(known),
                    "MAE_beta": float(np.abs(error).mean()),
                    "RMSE_beta": float(np.sqrt((error**2).mean())),
                }
            )
    pd.DataFrame(
        density_rows,
        columns=["experiment", "dataset", "images", "MAE_beta", "RMSE_beta"],
    ).to_csv(output / "density-quality.csv", index=False)
    make_figures(output, results, paired)
    return results


def make_figures(output, results, paired):
    import matplotlib

    matplotlib.use(
        "Agg", force=True
    )  # saved figures must work without Tcl/Tk or a desktop
    import matplotlib.pyplot as plt
    from fog_research import CLASSES

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120})
    styles = {
        "rgb_control": ("#33658A", "o"),
        "predicted_density": ("#D17A22", "^"),
        "oracle_density": ("#548235", "s"),
    }
    for dataset, frame in results.groupby("dataset"):
        overall = frame[frame.density == "all"]
        if overall.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        labels = []
        for position, ((model, arm), part) in enumerate(
            overall.groupby(["model", "arm"])
        ):
            color, marker = styles[arm]
            label = f"{model}\n{arm}"
            labels.append(label)
            values = part.mAP50_95.to_numpy()
            axes[0].bar(position, np.nanmean(values), color=color, alpha=0.65)
            axes[0].scatter(
                np.full(len(values), position),
                values,
                color="black",
                marker=marker,
                s=20,
            )
            axes[1].scatter(
                part.mean_ms,
                part.mAP50_95,
                color=color,
                marker=marker,
                label=f"{model}: {arm}",
            )
        axes[0].set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axes[0].set(
            ylabel="mAP@[.50:.95]",
            ylim=(0, 1),
            title="Bars: seed mean; points: individual seeds",
        )
        axes[1].set(
            xlabel="End-to-end mean latency (ms/image)",
            ylabel="mAP@[.50:.95]",
            ylim=(0, 1),
            title="Accuracy versus latency",
        )
        axes[1].legend(fontsize=7)
        profiles = ", ".join(sorted(str(value) for value in overall.profile.unique()))
        fig.suptitle(
            f"{dataset} | profile={profiles} | {overall.evaluation_split.iloc[0]} images"
        )
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}-comparison.png", bbox_inches="tight")
        plt.close(fig)
        balanced = frame[frame.density == "balanced"]
        if not balanced.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = []
            for position, ((model, arm), part) in enumerate(
                balanced.groupby(["model", "arm"])
            ):
                labels.append(f"{model}\n{arm}")
                color, marker = styles[arm]
                ax.bar(position, np.nanmean(part.mAP50_95), color=color, alpha=0.65)
                ax.scatter(
                    np.full(len(part), position),
                    part.mAP50_95,
                    color="black",
                    marker=marker,
                    s=20,
                )
            ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
            ax.set(
                ylabel="Equal-beta-bin macro mAP@[.50:.95]",
                ylim=(0, 1),
                title=f"{dataset}: balanced fog-density summary",
            )
            fig.tight_layout()
            fig.savefig(figures / f"{dataset}-balanced.png", bbox_inches="tight")
            plt.close(fig)
        numeric = frame[~frame.density.isin(["all", "unknown", "balanced"])].copy()
        if not numeric.empty:
            numeric["beta"] = numeric.density.astype(float)
            fig, ax = plt.subplots(figsize=(9, 5))
            for (model, arm), part in numeric.groupby(["model", "arm"]):
                mean = part.groupby("beta").mAP50_95.mean()
                ax.plot(
                    mean.index,
                    mean.values,
                    marker=styles[arm][1],
                    label=f"{model}: {arm}",
                )
            ax.set(
                xlabel="Known beta (m^-1)",
                ylabel="Mean mAP@[.50:.95]",
                ylim=(0, 1),
                title=f"{dataset}: accuracy by fog level",
            )
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures / f"{dataset}-density.png")
            plt.close(fig)
        table = overall.groupby(["model", "arm"])[[f"AP_{c}" for c in CLASSES]].mean()
        fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(table))))
        plotted = ax.imshow(
            table.to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis"
        )
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(len(table)), [f"{m}: {a}" for m, a in table.index])
        ax.set_title(f"{dataset}: per-class AP (blank = no ground truth)")
        fig.colorbar(plotted, ax=ax, label="AP@[.50:.95]")
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}-classes.png", bbox_inches="tight")
        plt.close(fig)
        delta = paired[(paired.dataset == dataset) & (paired.density == "all")]
        if not delta.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            names = []
            for index, ((model, arm), part) in enumerate(
                delta.groupby(["model", "arm"])
            ):
                names.append(f"{model}\n{arm}")
                for ax, column in zip(axes, ["delta_mAP", "delta_latency_ms"]):
                    ax.scatter(
                        np.full(len(part), index), part[column], marker=styles[arm][1]
                    )
            for ax in axes:
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
            axes[0].set_ylabel("Paired delta mAP (positive = better)")
            axes[1].set_ylabel("Paired delta latency ms (positive = slower)")
            fig.suptitle(
                f"{dataset}: matched seeds and test images; no significance claim"
            )
            fig.tight_layout()
            fig.savefig(figures / f"{dataset}-deltas.png", bbox_inches="tight")
            plt.close(fig)
    for history_file in output.glob("*/history.csv"):
        history = pd.read_csv(history_file)
        if not {"train_loss", "val_loss"} <= set(history.columns):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history.epoch, history.train_loss, label="Train", marker="o")
        ax.plot(history.epoch, history.val_loss, label="Tuning", marker="^")
        ax.set(xlabel="Epoch", ylabel="Detector loss", title=history_file.parent.name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{history_file.parent.name}-loss.png")
        plt.close(fig)
