"""Revise data panels using frozen numerical exports; do not refit models.

Run with the project's Python environment:
    python cross_talker_generalization/scripts/build_september9_data_panels.py

Outputs are new PNG/SVG/PDF panels and inspectable source tables. The language
palette is read from the supplied language_capitals_worldmap.R without executing
it. For the first run, --source-dir identifies the folder containing the five
supplied world-map/heatmap files; later runs use the preserved sources/ copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
PREVIOUS = PROJECT / "analysis_update_2026-09-06/collaborator_report/tables"
OUT = PROJECT / "analysis_update_2026-09-09"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
SOURCES = OUT / "sources"
FAMILY_RENAMES = {"Cushitic": "Afro-Asiatic", "Sinitic": "Sino-Tibetan"}
SOURCE_HASHES: dict[str, str] = {}
GENERATED: list[Path] = []
NOTES: dict = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, **kwargs):
    SOURCE_HASHES[path.relative_to(REPO).as_posix()] = digest(path)
    return pd.read_csv(path, **kwargs)


def write_table(frame, name, index=False, **kwargs):
    path = TABLES / name
    frame.to_csv(path, index=index, **kwargs)
    GENERATED.append(path)


def save(fig, name):
    for extension in ("png", "svg", "pdf"):
        path = FIGURES / f"{name}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        GENERATED.append(path)
    plt.close(fig)
    print(f"Saved {name}", flush=True)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def supplied_sources(source_dir):
    """Preserve supplied files byte-for-byte; never run the plotting R source."""
    SOURCES.mkdir(parents=True, exist_ok=True)
    names = ["language_capitals_worldmap.R", "language_capitals_worldmap.pdf",
             "language_capitals_worldmap.png", "english_phonology_similarity_heatmap.pdf",
             "english_phonology_similarity_heatmap.png"]
    rows = []
    for name in names:
        original = source_dir / name
        preserved = SOURCES / name
        if not original.is_file():
            raise FileNotFoundError(f"Required supplied source: {original}")
        sha = digest(original)
        if original.resolve() != preserved.resolve():
            if preserved.exists() and digest(preserved) != sha:
                raise ValueError(f"Existing preserved source differs: {name}; do not overwrite silently")
            if not preserved.exists():
                shutil.copyfile(original, preserved)
        assert digest(preserved) == sha
        SOURCE_HASHES[preserved.relative_to(REPO).as_posix()] = sha
        rows.append(dict(filename=name, size_bytes=preserved.stat().st_size, sha256=sha,
                         status="Supplied source, unchanged"))
    write_table(pd.DataFrame(rows), "supplied_figure_source_manifest.csv")
    code = (SOURCES / "language_capitals_worldmap.R").read_text(encoding="utf-8-sig")
    match = re.search(r"family_colors\s*<-\s*c\((.*?)\n\)", code, flags=re.S)
    if not match:
        raise ValueError("Cannot identify family_colors in the supplied R source")
    colors = dict(re.findall(r'"([^"\n]+)"\s*=\s*"(#[0-9a-fA-F]{6})"', match.group(1)))
    if len(colors) != 11:
        raise ValueError("Expected exactly eleven language-group colors from the supplied R source")
    NOTES["supplied_sources"] = {
        "world_map": "Supplied R/PDF/PNG preserved unchanged; R not executed",
        "world_map_locations": "Associated national capitals, not participants' or talkers' exact origins",
        "palette_source": "family_colors in supplied language_capitals_worldmap.R",
        "group_name_mappings": FAMILY_RENAMES,
        "heatmap": "Supplied reference illustration, not recomputed or numerically validated",
        "heatmap_numeric_source": "Not supplied; the R script generates only the world map",
        "heatmap_label_issue": "Bottom row label is clipped in the supplied image; the missing text was not inferred or repaired",
    }
    return colors


def coverage_panel(language_map, colors):
    counts = language_map.groupby(["branch", "language", "dataset"]).speaker_id.nunique().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10.4, 7.6), layout="constrained")
    for i, ((branch, language), row) in enumerate(counts.iterrows()):
        for j, dataset in enumerate(("AN19", "X21", "B23")):
            count = int(row.get(dataset, 0))
            if count:
                ax.scatter(j, i, s=240, color=colors[branch])
                ax.text(j, i, str(count), ha="center", va="center", color="white", fontsize=10)
    ax.set_xticks(range(3), ["AN19", "X21", "B23"])
    ax.set_yticks(range(len(counts)), [f"{language} [{branch}]" for branch, language in counts.index])
    ax.set(xlim=(-.5, 2.5), ylim=(len(counts)-.5, -.5), title="Available recording coverage")
    tidy(ax)
    save(fig, "figure_si_language_corpus_coverage")
    write_table(counts.reset_index(), "figure_si_language_corpus_counts.csv")
    write_table(language_map, "figure_si_talker_language_map.csv")
    NOTES["figure_si_language_corpus_coverage"] = {
        "counts": "Unique talkers in each available corpus; includes English reference recordings and recordings not used in exposure",
        "palette": "Exact family_colors from the supplied world-map R source",
        "role": "Supplementary coverage table/plot; supplied geographic world map is retained for Figure 1c",
        "n_language_labels": int(language_map.language.nunique()),
        "n_talkers_by_dataset": language_map.groupby("dataset").speaker_id.nunique().to_dict(),
    }


def matrices(language_map):
    pairs = read(REPO / "results/derived/AN19-talker-validation-base-tr24/talker_pair_summary.csv")
    talkers = language_map.loc[language_map.dataset.eq("AN19")].set_index("speaker_id")
    keys = sorted(talkers.index)
    if len(keys) != 42 or len(pairs) != 861 or not pairs.n_shared_words.eq(138).all():
        raise ValueError("Expected 42 talkers, 861 unordered pairs and 138 shared words")
    distance = pd.DataFrame(np.nan, index=keys, columns=keys)
    similarity = distance.copy()
    for row in pairs.itertuples():
        for matrix, value in ((distance, row.raw_distance), (similarity, row.similarity_exp_k1)):
            matrix.loc[row.talker_a_id, row.talker_b_id] = value
            matrix.loc[row.talker_b_id, row.talker_a_id] = value
    off_diagonal = ~np.eye(len(keys), dtype=bool)
    for matrix in (distance, similarity):
        if not np.isfinite(matrix.to_numpy()[off_diagonal]).all():
            raise ValueError("Talker matrix has an unobserved off-diagonal entry")
        np.testing.assert_allclose(matrix, matrix.T, equal_nan=True)
    english = talkers.index[talkers.language.eq("English")].tolist()
    if len(english) != 6:
        raise ValueError("Expected six English reference talkers")
    english_distance = distance.loc[:, english].mean(axis=1)
    group_distance = english_distance.groupby(talkers.language).mean().to_dict()
    languages = ["English"] + sorted(
        set(talkers.language) - {"English"}, key=lambda name: (group_distance[name], name))
    order = []
    block_rows = []
    for language in languages:
        members = sorted(talkers.index[talkers.language.eq(language)])
        if len(members) > 1:
            sub = distance.loc[members, members].to_numpy().copy()
            np.fill_diagonal(sub, 0)
            # Stable input ordering; average linkage uses the observed DTW distances.
            tree = linkage(squareform(sub), method="average", optimal_ordering=True)
            members = [members[i] for i in leaves_list(tree)]
        block_rows.append(dict(language=language, start=len(order), n_talkers=len(members),
                               mean_distance_to_english=group_distance[language]))
        order.extend(members)
    order_table = talkers.loc[order].reset_index()
    order_table.insert(0, "plot_index", range(len(order)))
    order_table["display_label"] = order_table.language + " " + order_table.speaker_id.str.rsplit(".", n=1).str[-1]
    order_table["mean_distance_to_english"] = order_table.speaker_id.map(english_distance)
    write_table(order_table, "figure_2a_42_talker_order.csv")
    write_table(pd.DataFrame(block_rows), "figure_2a_language_blocks.csv")
    for matrix, quantity in ((distance, "distance"), (similarity, "similarity")):
        matrix = matrix.loc[order, order]
        write_table(matrix, f"figure_2a_42_talker_{quantity}_matrix.csv", index=True, index_label="talker_id")
        fig, ax = plt.subplots(figsize=(12.8, 11.7), layout="constrained")
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        values = matrix.to_numpy()
        im = ax.imshow(values, cmap=cmap, norm=Normalize(vmin=0, vmax=float(np.nanmax(values))),
                       interpolation="nearest")
        labels = order_table.display_label.tolist()
        ax.set_xticks(range(42), labels, rotation=90, fontsize=8, color="black")
        ax.set_yticks(range(42), labels, fontsize=8, color="black")
        ax.tick_params(length=0, pad=4)
        for block in block_rows:
            ax.add_patch(Rectangle((block["start"]-.5, block["start"]-.5),
                                   block["n_talkers"], block["n_talkers"],
                                   fill=False, edgecolor="#909090", linewidth=1.05))
        ax.set_title("AN19", fontsize=16, pad=13)
        label = "Mean similarity across 138 words" if quantity == "similarity" else "Mean DTW distance across 138 words"
        fig.colorbar(im, ax=ax, shrink=.78, pad=.025, label=label)
        save(fig, f"figure_2a_an19_42_talker_{quantity}_linear")
    NOTES["figure_2a"] = {
        "n_talkers": 42, "n_english_talkers": 6, "n_l2_talkers": 36,
        "n_pairs": 861, "n_shared_words": 138,
        "distance": "Mean raw DTW distance across words; physical pairs averaged within each word first",
        "similarity": "Mean of exp(-raw DTW distance) across words, not exp(-mean distance)",
        "dtw_normalization": "mean_sequence_length", "dtw_tau": 2,
        "layer": "tr_24", "representation": "HuBERT base, corpus t-SNE, 3 dimensions",
        "color_scale": "Linear from zero to each matrix's off-diagonal maximum; no logarithm",
        "diagonal": "Masked light gray; no self-pair value estimated",
        "blocks": "Gray outlines denote same-L1 talker groups; labels are black, language first",
        "group_order": "English first; other L1 groups by increasing mean raw DTW distance to six English talkers",
        "within_group_order": "Average-linkage hierarchical clustering, optimal leaf ordering; lexicographically sorted input IDs",
        "shared_order": "Exactly the same raw-distance-derived order in both panels",
    }


def control_panels(language_map, colors):
    binned = read(PREVIOUS / "figure_2c_binned_accuracy.csv")
    curves = read(PREVIOUS / "figure_2c_logistic_curves.csv")
    accent_names = {"KOR": "Korean", "SPA": "Spanish", "CMN": "Mandarin"}
    branches = language_map.drop_duplicates("language").set_index("language").branch
    accent_colors = {accent: colors[branches[name]] for accent, name in accent_names.items()}
    fig = plt.figure(figsize=(13, 6.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(3.8, 1), hspace=.06, wspace=.19,
                          left=.075, right=.985, bottom=.11, top=.91)
    axes = [fig.add_subplot(grid[0, j]) for j in range(2)]
    marginals = [fig.add_subplot(grid[1, j], sharex=axes[j]) for j in range(2)]
    density_rows, targets, audits = [], [], []
    edges = np.linspace(0, 1, 26)
    for dataset, ax, density_ax in zip(("AN19", "X21"), axes, marginals):
        trials = read(PREVIOUS / f"figure_2c_{dataset.lower()}_control_trials.csv")
        pairs = read(PREVIOUS / f"figure_2c_{dataset.lower()}_word_reference_pairs.csv")
        query_keys = pairs[["query_id", "target_key"]].drop_duplicates()
        if query_keys.query_id.duplicated().any():
            raise ValueError("One query maps to multiple physical targets")
        valid = trials.loc[trials.mapping_status.eq("available")].merge(query_keys, on="query_id", validate="many_to_one")
        # AN19 target_key is the recording ID; X21 target_key identifies the cropped
        # keyword interval in its physical sentence recording (verified in prior export).
        unique = valid[["item_accent", "item_talker", "target_key", "similarity"]].drop_duplicates()
        if unique.duplicated(["item_accent", "target_key"]).any():
            raise ValueError("Inconsistent similarity for a physical target")
        unique.insert(0, "dataset", dataset)
        targets.append(unique)
        audits.append(dict(dataset=dataset, total_trials=len(trials), plotted_trials=len(valid),
                           excluded_trials=len(trials)-len(valid), n_physical_targets=len(unique),
                           n_participants=valid.participant_id.nunique()))
        for accent, target_group in unique.groupby("item_accent"):
            color = accent_colors[accent]
            dots = binned.loc[binned.dataset.eq(dataset) & binned.accent.eq(accent)]
            line = curves.loc[curves.dataset.eq(dataset) & curves.accent.eq(accent)]
            yerr = np.stack((dots.accuracy-dots.ci_low, dots.ci_high-dots.accuracy))
            if (yerr < -1e-10).any():
                raise ValueError("Binned interval does not enclose estimate")
            ax.errorbar(dots.mean_similarity, dots.accuracy, yerr=yerr,
                        fmt="o", color=color, alpha=.8, ms=5, capsize=2, zorder=3)
            ax.plot(line.similarity, line.accuracy, color=color, lw=2.3)
            ax.fill_between(line.similarity, line.ci_low, line.ci_high, color=color, alpha=.14)
            counts, _ = np.histogram(target_group.similarity, bins=edges)
            density = counts / (counts.sum() * np.diff(edges))
            if not np.isclose(np.sum(density * np.diff(edges)), 1):
                raise ValueError("Marginal histogram density does not integrate to one")
            density_ax.stairs(density, edges, color=color, lw=1.8)
            density_ax.stairs(density, edges, color=color, alpha=.1, fill=True)
            density_rows.extend(dict(dataset=dataset, accent=accent, left=left, right=right,
                                     count=int(count), density=float(value), n_unique_targets=len(target_group))
                                for left, right, count, value in zip(edges[:-1], edges[1:], counts, density))
        ax.set(title=dataset, ylim=(0, 1), xlim=(0, 1))
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel("Control-condition word accuracy" if dataset == "AN19" else "")
        density_ax.set(xlabel="Similarity to L1-English", ylabel="Density" if dataset == "AN19" else "")
        density_ax.set_ylim(bottom=0)
        density_ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        tidy(ax); tidy(density_ax)
    handles = [Line2D([0], [0], color=accent_colors[accent], marker="o", lw=2, label=name)
               for accent, name in accent_names.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.01),
               ncol=3, frameon=False, title="Test talker L1")
    save(fig, "figure_2c_control_similarity_with_marginals")
    write_table(binned, "figure_2c_binned_accuracy.csv")
    write_table(curves, "figure_2c_logistic_curves.csv")
    write_table(pd.concat(targets, ignore_index=True), "figure_2c_unique_physical_targets.csv")
    write_table(pd.DataFrame(density_rows), "figure_2c_marginal_densities.csv")
    write_table(pd.DataFrame(audits), "figure_2c_mapping_audit.csv")
    NOTES["figure_2c"] = {
        "model_refit": False, "source": "Frozen September 6 point estimates and confidence intervals",
        "model": "Descriptive binomial logistic regressions; no random effects and not cross-validated",
        "confidence_intervals": "95% participant bootstrap, 1,000 replicates in the source analysis",
        "predictor": "exp(-mean English-reference DTW / median distance across unique target words in each dataset)",
        "english_reference_pool": {"AN19": 6, "X21": 5},
        "marginals": "25 common bins on [0,1]; probability density, normalized separately within accent",
        "marginal_weighting": "Each unique physical target recording/word interval counted once, not once per participant response",
        "an19_exclusion": "40 of 1,920 trials remain excluded pending wave/wade recording-identity verification; not asserted missing audio",
        "palette": "Exact family_colors from supplied language_capitals_worldmap.R",
        "counts": audits,
    }


def inventory_panel(colors):
    inventory = read(PREVIOUS / "figure_1d_inventory_similarity.csv")
    inventory["source_branch"] = inventory.branch
    inventory["branch"] = inventory.branch.replace(FAMILY_RENAMES)
    if not inventory.status.eq("computed").all() or len(inventory) != 16:
        raise ValueError("Expected all sixteen L2 inventory summaries")
    inventory = inventory.sort_values(["mean_jaccard", "language"])
    fig, ax = plt.subplots(figsize=(10.8, 7.2), layout="constrained")
    for i, row in enumerate(inventory.itertuples()):
        ax.plot([row.min_jaccard, row.max_jaccard], [i, i], color=colors[row.branch], lw=2)
        ax.scatter(row.mean_jaccard, i, color=colors[row.branch], s=45, zorder=3)
    ax.set_yticks(range(len(inventory)), inventory.language.tolist())
    ax.set(xlabel="L1-L2 phonological inventory similarity with English", xlim=(0, 1),
           title="Phonological inventories")
    tidy(ax)
    save(fig, "figure_1d_phonological_inventory_similarity")
    write_table(inventory, "figure_1d_inventory_similarity.csv")
    NOTES["figure_1d"] = {
        "calculation_changed": False, "n_l2_languages": 16,
        "quantity": "Exact-symbol consonant/vowel inventory Jaccard overlap with English, PHOIBLE 2.0",
        "point": "Mean across all selected L2-by-English inventory pairs",
        "line": "Minimum-to-maximum across inventory sources; NOT a 95% confidence interval",
        "english_inventories": 9, "tones": "Not included; adding tone symbols requires a separate validated comparison",
        "construct": "Segment-inventory overlap only, not phonotactics or a complete phonological-distance measure",
        "source_url": "https://github.com/phoible/dev/tree/v2.0",
        "source_license": "CC BY-SA 3.0; Moran & McCloy (eds.), PHOIBLE 2.0 (2019)",
        "dialect_scope": "Language-level sources; not matched specifically to each speaker's dialect",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCES,
                        help="Folder containing the five supplied world-map/heatmap files; defaults to preserved sources")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
    language_map = read(PREVIOUS / "figure_1c_talker_language_map.csv")
    language_map["source_branch"] = language_map.branch
    language_map["branch"] = language_map.branch.replace(FAMILY_RENAMES)
    colors = supplied_sources(args.source_dir)
    if set(language_map.branch) != set(colors):
        raise ValueError("Corpus language groups and supplied world-map palette do not match")
    write_table(pd.DataFrame([{"branch": key, "color": value} for key, value in colors.items()]),
                "language_branch_palette.csv")
    matrices(language_map)
    control_panels(language_map, colors)
    inventory_panel(colors)
    coverage_panel(language_map, colors)
    SOURCE_HASHES[Path(__file__).relative_to(REPO).as_posix()] = digest(Path(__file__))
    NOTES["source_hashes"] = SOURCE_HASHES
    NOTES["output_hashes"] = {path.relative_to(OUT).as_posix(): digest(path) for path in GENERATED}
    NOTES["software"] = {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__}
    NOTES["palette_source"] = "Exact family_colors from supplied language_capitals_worldmap.R"
    (TABLES / "september9_data_panel_metadata.json").write_text(json.dumps(NOTES, indent=2), encoding="utf-8")
    print("Complete: 5 panels, frozen numerical results retained, supplied world-map palette applied.", flush=True)


if __name__ == "__main__":
    main()
