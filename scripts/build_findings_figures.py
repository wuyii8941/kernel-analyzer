"""Plot historical findings without pooling incompatible experimental protocols."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose a new repository output directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = {
        "rotary": ROOT / "results/property/result_analysis_v5/rotary_absolute_state_effect.json",
        "training": ROOT / "results/property/result_analysis_v4/iid_training_confirmation/verification.json",
        "structure": ROOT / "results/property/result_analysis_v3/structured_training/analysis/failure_aware_final.json",
    }
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    if data["training"]["status"] != "VERIFIED" or data["training"]["errors"]:
        raise ValueError("training verification failed")
    values = np.array(data["training"]["recomputed_primary"]["paired_values"])
    loo = []
    for i in range(len(values)):
        subset = np.delete(values, i)
        mean = float(subset.mean())
        radius = float(t.ppf(.975, len(subset)-1) * subset.std(ddof=1) / np.sqrt(len(subset)))
        loo.append({"omitted_pair": i, "mean": mean, "interval_95": [mean-radius, mean+radius]})
    output.mkdir(parents=True)
    report = {"data_use": "POST_HOC_DESCRIPTIVE_REANALYSIS_NO_ADDITIONAL_SAMPLES",
              "sources": [str(p.relative_to(ROOT)) for p in paths.values()],
              "leave_one_out": loo,
              "boundary": "Separate protocols in separate figures; no cross-experiment ranking. t intervals retain original distribution assumptions."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    labels = ["warm", "reset"]
    for ax, key, title in zip(axes, ("effect_norm_rms", "reference_norm_rms", "relative_rms"),
                             ("Absolute update difference", "Reference update scale", "Relative update difference")):
        numbers = [data["rotary"]["conditions"][k][key] for k in labels]
        ax.bar(labels, numbers, color=["#377eb8", "#ff7f00"])
        ax.set_title(title, fontsize=10)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    fig.suptitle("RoPE: fixed 16-state confirmation set, one parameter")
    fig.text(.5, .01, "Reset changes both moments and step counter.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.92))
    fig.savefig(output / "state_effect.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.7))
    ax.scatter(np.arange(8), values, color="#377eb8", label="Independent paired stream")
    ax.axhline(.01, color="#e41a1c", linestyle="--", label="Declared mean improvement margin")
    ax.axhline(0, color="gray", linewidth=.7)
    primary = data["training"]["recomputed_primary"]
    lower, upper = primary["interval_95"]
    ax.errorbar([9], [primary["mean"]], yerr=[[primary["mean"]-lower], [upper-primary["mean"]]], fmt="s", color="black", capsize=5, label="Mean and 95% paired t interval")
    ax.set_xticks(list(range(8))+[9], [str(i) for i in range(8)]+["Mean"])
    ax.set_ylabel("Validation loss OFF minus ON")
    ax.set_title("Frozen compensation: new iid streams, 1024 steps")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "training_confirmation.png", dpi=180)
    plt.close(fig)
    outcomes = data["structure"]["condition_outcomes"]
    names = ["OFF", "ON", "KEY_ONLY", "REST_ONLY", "COORDINATE_ROLL", "ONE_EXTRA_STEP_LAG"]
    finite = [outcomes[n].get("COMPLETE_FINITE", 0) for n in names]
    failed = [outcomes[n].get("NUMERICAL_FAILURE", 0) for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, finite, label="Finite completion", color="#377eb8")
    ax.barh(names, failed, left=finite, label="Numerical failure", color="#e41a1c")
    ax.set_xlabel("Observed streams (same-stream intervention, not new confirmation)")
    ax.set_title("Residual interventions: failures retained")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "intervention_outcomes.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
