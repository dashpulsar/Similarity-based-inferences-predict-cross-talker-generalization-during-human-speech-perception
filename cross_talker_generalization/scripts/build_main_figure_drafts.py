from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_COLORS = {
    "Germanic": "#4C78A8",
    "Romance": "#E45756",
    "Slavic": "#72B7B2",
    "Indo-Aryan": "#F2CF5B",
    "Iranian": "#B279A2",
    "Albanian": "#FF9DA6",
    "Sinitic": "#59A14F",
    "Koreanic": "#9C755F",
    "Japonic": "#BAB0AC",
    "Turkic": "#EDC948",
    "Cushitic": "#76B7B2",
    "Unspecified": "#999999",
}

LANGUAGE_FAMILY = {
    "English": "Germanic",
    "German": "Germanic",
    "Dutch": "Germanic",
    "Spanish": "Romance",
    "French": "Romance",
    "Romanian": "Romance",
    "Brazilian Portuguese": "Romance",
    "Russian": "Slavic",
    "Bengali": "Indo-Aryan",
    "Farsi": "Iranian",
    "Albanian": "Albanian",
    "Mandarin": "Sinitic",
    "Chinese": "Sinitic",
    "Korean": "Koreanic",
    "Japanese": "Japonic",
    "Turkish": "Turkic",
    "Somalian": "Cushitic",
    "Indian (reported label)": "Unspecified",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cross_talker_generalization/analysis/model_comparison/reference_checks/figure_drafts"),
    )
    parser.add_argument("--seed", type=int, default=230519)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.04, 1.04, label, transform=axis.transAxes, fontsize=15, fontweight="bold")


def placeholder(axis: plt.Axes, title: str, requirement: str) -> None:
    axis.set_facecolor("#f4f4f4")
    axis.text(0.5, 0.62, title, ha="center", va="center", fontsize=12, fontweight="bold")
    axis.text(0.5, 0.39, requirement, ha="center", va="center", fontsize=8.5, wrap=True)
    axis.text(0.5, 0.16, "SOURCE DATA REQUIRED", ha="center", va="center", fontsize=9, color="#9a3f3f")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#bbbbbb")


def language_inventory(repository: Path) -> pd.DataFrame:
    an19 = pd.read_csv(repository / "data/manifests/AN19-stimulus-manifest.csv")
    x21 = pd.read_csv(repository / "data/manifests/X21-stimulus-manifest.csv")
    rows = []
    for language in sorted(an19["source_l1_language"].dropna().astype(str).unique()):
        clean = "Indian (reported label)" if language == "Indian" else language
        rows.append({"language": clean, "dataset": "AN19"})
    for language in sorted(x21["source_l1_language"].dropna().astype(str).unique()):
        rows.append({"language": language, "dataset": "X21"})
    for language in ("Brazilian Portuguese", "Farsi", "Spanish", "Turkish"):
        rows.append({"language": language, "dataset": "B23"})
    result = pd.DataFrame(rows).drop_duplicates()
    result["family"] = result["language"].map(LANGUAGE_FAMILY).fillna("Unspecified")
    return result.sort_values(["family", "language", "dataset"]).reset_index(drop=True)


def plot_language_families(axis: plt.Axes, inventory: pd.DataFrame) -> None:
    grouped = inventory.groupby(["family", "language"], sort=True)["dataset"].agg(
        lambda values: ", ".join(sorted(set(values)))
    )
    rows = grouped.reset_index()
    y_positions = np.arange(len(rows))[::-1]
    families = list(dict.fromkeys(rows["family"]))
    family_x = {family: 0.16 for family in families}
    for y, row in zip(y_positions, rows.itertuples(index=False)):
        color = FAMILY_COLORS[row.family]
        axis.plot([0.20, 0.43], [y, y], color=color, linewidth=1.2)
        axis.scatter([0.43], [y], s=28, color=color, zorder=3)
        axis.text(0.46, y, row.language, va="center", fontsize=7.5)
        axis.text(0.97, y, row.dataset, va="center", ha="right", fontsize=6.8, color="#555555")
    for family in families:
        ys = y_positions[rows["family"].eq(family).to_numpy()]
        center = float(np.mean(ys))
        axis.plot([family_x[family], 0.20], [center, center], color=FAMILY_COLORS[family], linewidth=2)
        if len(ys) > 1:
            axis.plot([0.20, 0.20], [ys.min(), ys.max()], color=FAMILY_COLORS[family], linewidth=1)
        axis.text(0.15, center, family, ha="right", va="center", fontsize=7.5, fontweight="bold")
    axis.set_xlim(0, 1)
    axis.set_ylim(-1, len(rows))
    axis.axis("off")
    axis.set_title("L1 backgrounds represented in the three studies", fontsize=10)


def build_figure_1(repository: Path, output: Path, inventory: pd.DataFrame) -> list[Path]:
    schematic = repository / "results/figures/X21-base-tr12-representation-schematic-v1/representation_schematic.png"
    image = plt.imread(schematic)
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[1.35, 1], hspace=0.22, wspace=0.18)
    axis_a = fig.add_subplot(grid[0, :])
    axis_a.imshow(image)
    axis_a.axis("off")
    axis_a.set_title("Audio → HuBERT latent sequence → corpus-level 3-D t-SNE trajectory", fontsize=12)
    panel_label(axis_a, "a–b")

    axis_c = fig.add_subplot(grid[1, 0])
    plot_language_families(axis_c, inventory)
    panel_label(axis_c, "c")

    axis_d = fig.add_subplot(grid[1, 1])
    placeholder(
        axis_d,
        "L1 sound-structure similarity to English",
        "Finalize the PHOIBLE-derived segmental/suprasegmental/\nphonotactic metric, language-name mapping, and\nuncertainty rule before calculation.",
    )
    panel_label(axis_d, "d")
    fig.suptitle("Figure 1 draft — computational speech representations and language coverage", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Internal draft: panel a uses an existing source-backed X21 example (base HuBERT, Tr-12); panel d is intentionally not imputed.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    png = output / "figure_01_method_and_language_coverage_draft.png"
    svg = output / "figure_01_method_and_language_coverage_draft.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [schematic, png, svg]


def talker_control_summary(repository: Path, rng: np.random.Generator, n_boot: int) -> tuple[pd.DataFrame, Path]:
    distance_path = repository / "cross_talker_generalization/analysis/reference/tables/talker_distance_tr24.csv"
    distances = pd.read_csv(distance_path)
    summaries = []
    for dataset, control_id in (("AN19", "AN19.Control"), ("X21", "X21.Control")):
        behavior = pd.read_csv(repository / f"data/preprocessed data/{dataset}-behavioral-data.csv")
        behavior = behavior.loc[
            behavior["phase"].eq("test")
            & behavior["exposure_test_condition_id"].astype(str).str.startswith(control_id)
        ].copy()
        test_talkers = sorted(behavior["item_talker"].dropna().astype(str).unique())
        study_distances = distances.loc[
            distances["dataset_id"].eq(dataset)
            & distances["variant"].eq("base")
            & distances["layer"].eq("tr_24")
        ].copy()
        for talker in test_talkers:
            pair = study_distances.loc[
                (
                    study_distances["talker_a_id"].eq(talker)
                    & study_distances["talker_b_id"].astype(str).str.contains(".ENG.", regex=False)
                )
                | (
                    study_distances["talker_b_id"].eq(talker)
                    & study_distances["talker_a_id"].astype(str).str.contains(".ENG.", regex=False)
                )
            ]
            if pair.empty:
                continue
            trials = behavior.loc[behavior["item_talker"].eq(talker)]
            participant = trials.groupby("participant_id", as_index=False).agg(
                correct=("response_correct", "sum"), incorrect=("response_incorrect", "sum")
            )
            point = participant["correct"].sum() / (participant["correct"].sum() + participant["incorrect"].sum())
            boot = np.empty(n_boot)
            for index in range(n_boot):
                sampled = participant.iloc[rng.integers(0, len(participant), size=len(participant))]
                boot[index] = sampled["correct"].sum() / (sampled["correct"].sum() + sampled["incorrect"].sum())
            low, high = np.quantile(boot, [0.025, 0.975])
            summaries.append(
                {
                    "dataset_id": dataset,
                    "talker_id": talker,
                    "mean_l2_l1_dtw": pair["mean_raw_distance"].mean(),
                    "control_accuracy": point,
                    "accuracy_ci_low": low,
                    "accuracy_ci_high": high,
                    "n_control_participants": participant["participant_id"].nunique(),
                    "n_response_units": int((participant["correct"] + participant["incorrect"]).sum()),
                    "n_english_reference_talkers": len(pair),
                }
            )
    result = pd.DataFrame(summaries)
    result["l1_similarity_z"] = result.groupby("dataset_id")["mean_l2_l1_dtw"].transform(
        lambda values: -(values - values.mean()) / values.std(ddof=0)
    )
    return result, distance_path


def plot_control_relationship(axis: plt.Axes, data: pd.DataFrame) -> None:
    colors = {"AN19": "#2878B5", "X21": "#D43F3A"}
    for dataset, frame in data.groupby("dataset_id", sort=True):
        axis.errorbar(
            frame["l1_similarity_z"],
            frame["control_accuracy"],
            yerr=[
                frame["control_accuracy"] - frame["accuracy_ci_low"],
                frame["accuracy_ci_high"] - frame["control_accuracy"],
            ],
            fmt="o",
            color=colors[dataset],
            alpha=0.8,
            capsize=2,
            label=dataset,
        )
        if len(frame) >= 3:
            coefficients = np.polyfit(frame["l1_similarity_z"], frame["control_accuracy"], 1)
            x = np.linspace(frame["l1_similarity_z"].min(), frame["l1_similarity_z"].max(), 100)
            axis.plot(x, np.polyval(coefficients, x), color=colors[dataset], linewidth=1.8)
    axis.set_xlabel("L1-English pronunciation similarity\n(−z mean matched-content DTW, within study)")
    axis.set_ylabel("Control-condition recognition accuracy")
    axis.set_ylim(0.35, 1.02)
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Pronunciation proximity and pre-exposure intelligibility", fontsize=10)


def build_figure_2(repository: Path, output: Path, control: pd.DataFrame) -> list[Path]:
    heatmap = repository / "results/figures/AN19-base-tr24-talker-similarity-heatmap-log-v3/talker_similarity_heatmap.png"
    image = plt.imread(heatmap)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes[0, 0].imshow(image)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("AN19 word-level pronunciation similarity (36 L2 talkers)", fontsize=10)
    panel_label(axes[0, 0], "a")

    placeholder(
        axes[0, 1],
        "Segment-level L2-to-L1 production deviation",
        "Requires a validated segment-instance table linking\nphoneme intervals across L2 and L1 recordings.",
    )
    panel_label(axes[0, 1], "b")

    plot_control_relationship(axes[1, 0], control)
    panel_label(axes[1, 0], "c")

    placeholder(
        axes[1, 1],
        "Segment perception errors × production deviation × lexical constraint",
        "Requires response-to-segment alignment, a declared error\ncoding rule, and a minimal-pair lexicon/count definition.",
    )
    panel_label(axes[1, 1], "d")
    fig.suptitle("Figure 2 draft — production/perception validation of learned representations", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Internal draft: panels a and c are source-backed; panels b and d mark the exact unresolved inputs rather than showing synthetic results.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    png = output / "figure_02_representation_validation_draft.png"
    svg = output / "figure_02_representation_validation_draft.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [heatmap, png, svg]


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    output = (repository / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    inventory = language_inventory(repository)
    inventory_path = output / "figure_01_language_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    figure_1_inputs = build_figure_1(repository, output, inventory)

    control, distance_path = talker_control_summary(repository, rng, args.bootstrap)
    control_path = output / "figure_02c_control_similarity_accuracy.csv"
    control.to_csv(control_path, index=False)
    figure_2_inputs = build_figure_2(repository, output, control)

    readme = """# Main-figure drafts

These are review drafts for the compound figures described after the September meeting.

- Figure 1 combines the existing source-backed HuBERT/t-SNE example with a manifest-derived inventory of L1 backgrounds. Its PHOIBLE panel remains an explicit placeholder because the exact sound-structure similarity metric has not yet been specified.
- Figure 2a reuses the validated AN19 Tr-24 matched-word talker matrix. Figure 2c is newly calculated from control-condition responses and each test talker's mean matched-content DTW distance to the study's L1-English reference talkers. Similarity is standardized within study because the two stimulus domains use different units. The intervals bootstrap participants.
- Figure 2b and 2d remain explicit placeholders. Producing them requires validated segment alignments and response-to-segment/lexical-neighborhood sources that are not currently registered in the project.

No placeholder contains simulated or inferred results. These drafts are suitable for deciding layout and assigning the remaining source-data work, not for publication as complete figures.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    source_paths = [
        repository / "cross_talker_generalization/scripts/build_main_figure_drafts.py",
        inventory_path,
        control_path,
        distance_path,
        *figure_1_inputs,
        *figure_2_inputs,
    ]
    source_paths = sorted({path for path in source_paths if path.is_file()})
    provenance = {
        "status": "draft_with_explicit_missing_panels",
        "seed": args.seed,
        "participant_bootstrap_replicates": args.bootstrap,
        "inputs_and_outputs": {
            str(path.relative_to(repository)): sha256(path) for path in source_paths
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"built main-figure drafts in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
