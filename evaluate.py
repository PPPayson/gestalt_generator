import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utility
# ============================================================

def get_model_name(json_path):
    """
    Example:
        ollama_llama3.2-vision_11b_response.json
        -> llama3.2-vision_11b
    """
    name = json_path.stem

    if name.startswith("ollama_"):
        name = name[len("ollama_"):]

    if name.endswith("_response"):
        name = name[:-len("_response")]

    return name


# ============================================================
# Has-illusion evaluation
# ============================================================

def compute_has_illusion_metrics(json_path):
    """
    Evaluate has_illusion as a binary classification task.

    Ground truth:
        illusory -> True
        control  -> False

    Returns:
        accuracy
        precision
        recall
        TP / FP / TN / FN
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    gt_labels = {
        "illusory": True,
        "control": False,
    }

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for condition, images in data.items():

        if condition not in gt_labels:
            print(
                f"Warning: skipping unknown condition: {condition}"
            )
            continue

        gt = gt_labels[condition]

        for image_name, responses in images.items():

            for repetition, response in responses.items():

                if "has_illusion" not in response:
                    print(
                        f"Warning: missing has_illusion: "
                        f"{condition}/{image_name}/{repetition}"
                    )
                    continue

                prediction = response["has_illusion"]

                if gt and prediction:
                    tp += 1

                elif not gt and prediction:
                    fp += 1

                elif not gt and not prediction:
                    tn += 1

                elif gt and not prediction:
                    fn += 1

    total = tp + fp + tn + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


# ============================================================
# Counting ground truth
# ============================================================

def load_ground_truth_counts(gt_folder):
    """
    Load ground-truth object counts from files like:

        examples/gen100/gt/stim_0a2bb3a026.json

    Ground-truth count is:

        n_inducers + 1

    Returns:
        {
            "stim_0a2bb3a026": count,
            ...
        }
    """

    gt_folder = Path(gt_folder)

    if not gt_folder.exists():
        raise RuntimeError(
            f"Ground-truth folder does not exist: {gt_folder}"
        )

    gt_files = sorted(gt_folder.glob("*.json"))

    if len(gt_files) == 0:
        raise RuntimeError(
            f"No ground-truth JSON files found in {gt_folder}"
        )

    gt_counts = {}

    for gt_path in gt_files:

        with open(gt_path, "r") as f:
            gt_data = json.load(f)

        if "n_inducers" not in gt_data:
            print(
                f"Warning: n_inducers missing in {gt_path}"
            )
            continue

        image_id = gt_path.stem

        gt_count = int(gt_data["n_inducers"]) + 1

        gt_counts[image_id] = gt_count

    return gt_counts


# ============================================================
# Counting evaluation
# ============================================================

def compute_counting_metrics(json_path, gt_counts):
    """
    Evaluate n_objects predictions against ground truth.

    For each image:

        GT count = n_inducers + 1

    Each repetition is treated as an independent prediction.

    Computes:
        - overall exact-match counting accuracy
        - overall MAE
        - illusory exact-match counting accuracy
        - illusory MAE
        - control exact-match counting accuracy
        - control MAE
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    correct = 0
    total = 0
    absolute_error_sum = 0

    missing_gt = 0
    missing_prediction = 0
    invalid_prediction = 0

    # Separate statistics for each class
    class_stats = {
        "illusory": {
            "correct": 0,
            "total": 0,
            "absolute_error_sum": 0,
        },
        "control": {
            "correct": 0,
            "total": 0,
            "absolute_error_sum": 0,
        },
    }

    for condition, images in data.items():

        if condition not in class_stats:
            print(
                f"Warning: unknown condition in counting "
                f"evaluation: {condition}"
            )
            continue

        for image_name, responses in images.items():

            # stim_xxx.jpg -> stim_xxx
            image_id = Path(image_name).stem

            if image_id not in gt_counts:
                print(
                    f"Warning: no counting GT found for "
                    f"{condition}/{image_name}"
                )
                missing_gt += 1
                continue

            gt_count = gt_counts[image_id]

            for repetition, response in responses.items():

                if "n_objects" not in response:
                    print(
                        f"Warning: missing n_objects: "
                        f"{condition}/{image_name}/{repetition}"
                    )
                    missing_prediction += 1
                    continue

                prediction = response["n_objects"]

                try:
                    prediction = int(prediction)

                except (TypeError, ValueError):
                    print(
                        f"Warning: invalid n_objects value: "
                        f"{condition}/{image_name}/{repetition}: "
                        f"{prediction}"
                    )
                    invalid_prediction += 1
                    continue

                error = abs(prediction - gt_count)
                is_correct = prediction == gt_count

                # Overall
                total += 1
                correct += int(is_correct)
                absolute_error_sum += error

                # By class
                class_stats[condition]["total"] += 1
                class_stats[condition]["correct"] += int(
                    is_correct
                )
                class_stats[condition][
                    "absolute_error_sum"
                ] += error

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    mae = (
        absolute_error_sum / total
        if total > 0
        else 0.0
    )

    # Convert accumulated class statistics into metrics
    for condition, stats in class_stats.items():

        class_total = stats["total"]

        stats["accuracy"] = (
            stats["correct"] / class_total
            if class_total > 0
            else 0.0
        )

        stats["mae"] = (
            stats["absolute_error_sum"] / class_total
            if class_total > 0
            else 0.0
        )

    return {
        "accuracy": accuracy,
        "mae": mae,
        "correct": correct,
        "total": total,
        "class_stats": class_stats,
        "missing_gt": missing_gt,
        "missing_prediction": missing_prediction,
        "invalid_prediction": invalid_prediction,
    }


# ============================================================
# Evaluate all models
# ============================================================

def evaluate_folder(folder, gt_folder):

    folder = Path(folder)

    if not folder.exists():
        raise RuntimeError(
            f"Output folder does not exist: {folder}"
        )

    json_files = sorted(folder.glob("*.json"))

    if len(json_files) == 0:
        raise RuntimeError(
            f"No JSON files found in {folder}"
        )

    # Load counting ground truth once
    gt_counts = load_ground_truth_counts(gt_folder)

    print(
        f"Loaded ground-truth counts for "
        f"{len(gt_counts)} images."
    )

    print(
        f"Found {len(json_files)} model result files."
    )

    results = {}

    for json_path in json_files:

        model_name = get_model_name(json_path)

        print()
        print(f"Evaluating: {model_name}")
        print(f"  File: {json_path}")

        illusion_metrics = compute_has_illusion_metrics(
            json_path
        )

        counting_metrics = compute_counting_metrics(
            json_path,
            gt_counts,
        )

        results[model_name] = {
            "illusion": illusion_metrics,
            "counting": counting_metrics,
        }

    return results


# ============================================================
# Print results
# ============================================================

def print_results(results):

    print()
    print("=" * 140)
    print("Model performance")
    print("=" * 140)

    print(
        f"{'Model':35s} "
        f"{'Illus. Acc':>10s} "
        f"{'Precision':>10s} "
        f"{'Recall':>10s} "
        f"{'Count Acc':>10s} "
        f"{'Illus. Count':>12s} "
        f"{'Ctrl Count':>12s} "
        f"{'Count MAE':>10s}"
    )

    print("-" * 140)

    for model, result in results.items():

        illusion = result["illusion"]
        counting = result["counting"]

        illusory_count_acc = (
            counting["class_stats"]["illusory"]["accuracy"]
        )

        control_count_acc = (
            counting["class_stats"]["control"]["accuracy"]
        )

        print(
            f"{model:35s} "
            f"{illusion['accuracy']:10.4f} "
            f"{illusion['precision']:10.4f} "
            f"{illusion['recall']:10.4f} "
            f"{counting['accuracy']:10.4f} "
            f"{illusory_count_acc:12.4f} "
            f"{control_count_acc:12.4f} "
            f"{counting['mae']:10.4f}"
        )

    print("=" * 140)

    # More detailed counts
    print("\nDetailed counts:")

    for model, result in results.items():

        illusion = result["illusion"]
        counting = result["counting"]

        illusory_stats = (
            counting["class_stats"]["illusory"]
        )

        control_stats = (
            counting["class_stats"]["control"]
        )

        print(f"\n{model}")

        print(
            f"  Illusion classification predictions: "
            f"{illusion['total']}"
        )

        print(
            f"    TP={illusion['tp']}, "
            f"FP={illusion['fp']}, "
            f"TN={illusion['tn']}, "
            f"FN={illusion['fn']}"
        )

        print(
            f"  Counting overall: "
            f"{counting['correct']}/"
            f"{counting['total']} correct "
            f"(accuracy={counting['accuracy']:.4f}, "
            f"MAE={counting['mae']:.4f})"
        )

        print(
            f"    Illusory: "
            f"{illusory_stats['correct']}/"
            f"{illusory_stats['total']} correct "
            f"(accuracy={illusory_stats['accuracy']:.4f}, "
            f"MAE={illusory_stats['mae']:.4f})"
        )

        print(
            f"    Control:  "
            f"{control_stats['correct']}/"
            f"{control_stats['total']} correct "
            f"(accuracy={control_stats['accuracy']:.4f}, "
            f"MAE={control_stats['mae']:.4f})"
        )

        if counting["missing_gt"] > 0:
            print(
                f"    Missing GT images: "
                f"{counting['missing_gt']}"
            )

        if counting["missing_prediction"] > 0:
            print(
                f"    Missing n_objects: "
                f"{counting['missing_prediction']}"
            )

        if counting["invalid_prediction"] > 0:
            print(
                f"    Invalid n_objects: "
                f"{counting['invalid_prediction']}"
            )


# ============================================================
# Plot has_illusion metrics
# ============================================================

def plot_illusion_metrics(results, output_path):

    # Sort models by illusion classification accuracy
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["illusion"]["accuracy"],
        reverse=True,
    )

    models = [
        model
        for model, _ in sorted_results
    ]

    accuracy = [
        result["illusion"]["accuracy"]
        for _, result in sorted_results
    ]

    precision = [
        result["illusion"]["precision"]
        for _, result in sorted_results
    ]

    recall = [
        result["illusion"]["recall"]
        for _, result in sorted_results
    ]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(
        figsize=(max(8, len(models) * 1.5), 5)
    )

    bars_accuracy = ax.bar(
        x - width,
        accuracy,
        width,
        label="Accuracy",
    )

    bars_precision = ax.bar(
        x,
        precision,
        width,
        label="Precision",
    )

    bars_recall = ax.bar(
        x + width,
        recall,
        width,
        label="Recall",
    )

    # Numerical labels
    ax.bar_label(
        bars_accuracy,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    ax.bar_label(
        bars_precision,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    ax.bar_label(
        bars_recall,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)

    ax.set_title(
        "Has-Illusion Classification Performance"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        models,
        rotation=45,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved illusion metrics plot to: "
        f"{output_path}"
    )


# ============================================================
# Plot overall counting accuracy
# ============================================================

def plot_counting_accuracy(results, output_path):

    # Sort models by overall counting accuracy
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["counting"]["accuracy"],
        reverse=True,
    )

    models = [
        model
        for model, _ in sorted_results
    ]

    accuracies = [
        result["counting"]["accuracy"]
        for _, result in sorted_results
    ]

    fig, ax = plt.subplots(
        figsize=(max(8, len(models) * 1.2), 5)
    )

    bars = ax.bar(
        models,
        accuracies,
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 1.08)

    ax.set_title(
        "Object Counting Accuracy"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    for label in ax.get_xticklabels():
        label.set_ha("right")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved counting accuracy plot to: "
        f"{output_path}"
    )


# ============================================================
# Plot counting accuracy by class
# ============================================================

def plot_counting_accuracy_by_class(
    results,
    output_path,
):
    """
    Compare exact-match counting accuracy between
    illusory and control images for each model.
    """

    # Sort by overall counting accuracy
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["counting"]["accuracy"],
        reverse=True,
    )

    models = [
        model
        for model, _ in sorted_results
    ]

    illusory_accuracy = [
        result["counting"]["class_stats"][
            "illusory"
        ]["accuracy"]
        for _, result in sorted_results
    ]

    control_accuracy = [
        result["counting"]["class_stats"][
            "control"
        ]["accuracy"]
        for _, result in sorted_results
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(max(8, len(models) * 1.4), 5)
    )

    bars_illusory = ax.bar(
        x - width / 2,
        illusory_accuracy,
        width,
        label="Illusory",
    )

    bars_control = ax.bar(
        x + width / 2,
        control_accuracy,
        width,
        label="Control",
    )

    # Numerical labels
    ax.bar_label(
        bars_illusory,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    ax.bar_label(
        bars_control,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    ax.set_ylabel(
        "Exact-match counting accuracy"
    )

    ax.set_ylim(0, 1.08)

    ax.set_title(
        "Object Counting Accuracy by Image Class"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        models,
        rotation=45,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved counting accuracy by class plot to: "
        f"{output_path}"
    )


# ============================================================
# Plot counting MAE
# ============================================================

def plot_counting_mae(results, output_path):

    # Lower MAE is better
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["counting"]["mae"],
    )

    models = [
        model
        for model, _ in sorted_results
    ]

    maes = [
        result["counting"]["mae"]
        for _, result in sorted_results
    ]

    fig, ax = plt.subplots(
        figsize=(max(8, len(models) * 1.2), 5)
    )

    bars = ax.bar(
        models,
        maes,
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    ax.set_ylabel(
        "Mean Absolute Error"
    )

    ax.set_title(
        "Object Counting Error"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    for label in ax.get_xticklabels():
        label.set_ha("right")

    # Add room for numerical labels
    if len(maes) > 0:

        max_mae = max(maes)

        ax.set_ylim(
            0,
            max_mae * 1.15
            if max_mae > 0
            else 1,
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved counting MAE plot to: "
        f"{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate has_illusion classification and "
            "object counting across model output JSON files."
        )
    )

    parser.add_argument(
        "--folder",
        default="outs",
        help=(
            "Folder containing model response JSON files "
            "(default: outs)"
        ),
    )

    parser.add_argument(
        "--gt_folder",
        default="examples/gen100/gt",
        help=(
            "Folder containing ground-truth JSON files "
            "(default: examples/gen100/gt)"
        ),
    )

    parser.add_argument(
        "--illusion_output",
        default="has_illusion_metrics.png",
        help=(
            "Output path for has_illusion metrics plot"
        ),
    )

    parser.add_argument(
        "--counting_output",
        default="counting_accuracy.png",
        help=(
            "Output path for overall counting accuracy plot"
        ),
    )

    parser.add_argument(
        "--counting_class_output",
        default="counting_accuracy_by_class.png",
        help=(
            "Output path for counting accuracy "
            "by image class plot"
        ),
    )

    parser.add_argument(
        "--counting_mae_output",
        default="counting_mae.png",
        help=(
            "Output path for counting MAE plot"
        ),
    )

    args = parser.parse_args()

    # Evaluate all models
    results = evaluate_folder(
        args.folder,
        args.gt_folder,
    )

    # Print results
    print_results(results)

    # Has-illusion classification
    plot_illusion_metrics(
        results,
        args.illusion_output,
    )

    # Overall counting accuracy
    plot_counting_accuracy(
        results,
        args.counting_output,
    )

    # Counting accuracy separated by
    # illusory vs control images
    plot_counting_accuracy_by_class(
        results,
        args.counting_class_output,
    )

    # Overall counting MAE
    plot_counting_mae(
        results,
        args.counting_mae_output,
    )


if __name__ == "__main__":
    main()