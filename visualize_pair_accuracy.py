"""Plot pair accuracy metrics from a Lightning CSVLogger metrics.csv file."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# this visualization file is designed for stargraph data.

def read_metrics(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as metrics_file:
        reader = csv.DictReader(metrics_file)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {csv_path}")

        required_fields = {"step", "pair_accuracy_next", "pair_accuracy_prev"}
        missing_fields = required_fields.difference(reader.fieldnames)
        if missing_fields:
            raise ValueError(
                f"{csv_path} is missing required metric columns: "
                f"{', '.join(sorted(missing_fields))}"
            )

        steps = []
        aggregate_next = []
        aggregate_prev = []
        pair_columns = {} # pair_dir -> [values]
        pair_metric_columns = {} # pair_dir -> col name
        for column in reader.fieldnames:
            if not column.startswith("pair_accuracy_by_index/"):
                continue
            pair_and_direction = column.removeprefix(
                "pair_accuracy_by_index/"
            ).rsplit("/", 1)
            if len(pair_and_direction) == 2:
                pair, direction = pair_and_direction
                pair = pair.split(',')
                pair = (pair[0], pair[1])
                # GPT still converges without mask, but metric will span from token 0 to prediction seq
                if (int(pair[0]) < 62):
                    continue
                pair_metric_columns[(pair, direction)] = column
                pair_columns[(pair, direction)] = []

        for row in reader:
            if not row.get("step"):
                continue
            steps.append(float(row["step"]))
            aggregate_next.append(
                float(row["pair_accuracy_next"])
                if row.get("pair_accuracy_next")
                else float("nan")
            )
            aggregate_prev.append(
                float(row["pair_accuracy_prev"])
                if row.get("pair_accuracy_prev")
                else float("nan")
            )
            for pair_direction, column in pair_metric_columns.items():
                value = row.get(column)
                pair_columns[pair_direction].append(
                    float(value) if value else float("nan")
                )

    if not steps:
        raise ValueError(f"No metric rows with a training step found in {csv_path}")
    return steps, aggregate_next, aggregate_prev, pair_columns


def visualize(csv_path: Path):
    steps, aggregate_next, aggregate_prev, pair_columns = read_metrics(csv_path)
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    aggregate_axis = fig.add_subplot(2, 2, (1, 2))
    next_axis = fig.add_subplot(2, 2, 3)
    prev_axis = fig.add_subplot(2, 2, 4)

    aggregate_axis.plot(steps, aggregate_next, label="Next")
    aggregate_axis.plot(steps, aggregate_prev, label="Previous")
    aggregate_axis.set_title("Mean pair accuracy")
    aggregate_axis.set_xlabel("Training step")
    aggregate_axis.set_ylabel("Accuracy")
    aggregate_axis.set_ylim(0, 1)
    aggregate_axis.grid(True, alpha=0.3)
    aggregate_axis.legend()

    pairs_sorted_begin = sorted({pair for pair, _ in pair_columns}, key=lambda k: (k[0], k[1]))
    pairs_sorted_end = sorted({pair for pair, _ in pair_columns}, key=lambda k: (k[1], k[0]))
    print(pairs_sorted_begin)
    print(pairs_sorted_end)
    if pairs_sorted_begin:
        tick_positions = list(range(0, len(steps), max(1, len(steps) // 8)))
        if tick_positions[-1] != len(steps) - 1:
            tick_positions.append(len(steps) - 1)

        images = []
        for axis, direction, title, data in (
            (next_axis, "next", "Pair accuracy by index (next)", pairs_sorted_begin),
            (prev_axis, "prev", "Pair accuracy by index (previous)", pairs_sorted_end),
        ):
            matrix = [
                pair_columns.get((pair, direction), [float("nan")] * len(steps))
                for pair in data
            ]
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                vmin=0,
                vmax=1,
            )
            images.append(image)
            axis.set_title(title)
            axis.set_xlabel("Training step")
            axis.set_ylabel("Token index pair")
            axis.set_yticks(range(len(data)), labels=data)
            axis.set_xticks(
                tick_positions,
                labels=[f"{steps[index]:g}" for index in tick_positions],
            )
        fig.colorbar(images[0], ax=[next_axis, prev_axis], label="Accuracy")
    else:
        next_axis.text(
            0.5,
            0.5,
            "No per-index accuracy metrics in this log",
            ha="center",
            va="center",
            transform=next_axis.transAxes,
        )
        next_axis.set_axis_off()
        prev_axis.set_visible(False)

    fig.suptitle(csv_path.name)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pair accuracy metrics from a CSVLogger output."
    )
    parser.add_argument(
        "metrics_file",
        help="Path to metrics.csv, relative to this script's folder (or absolute).",
    )
    args = parser.parse_args()
    metrics_path = Path(args.metrics_file)
    if not metrics_path.is_absolute():
        metrics_path = Path(__file__).resolve().parent / metrics_path
    if not metrics_path.is_file():
        parser.error(f"metrics file not found: {metrics_path}")
    visualize(metrics_path)


if __name__ == "__main__":
    main()
