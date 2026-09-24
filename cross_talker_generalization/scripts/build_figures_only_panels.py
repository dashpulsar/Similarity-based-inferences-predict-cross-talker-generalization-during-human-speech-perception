"""Prepare current figure-only presentation panels from retained numerical tables.

No ASR, feature extraction, DTW calculation, predictor search or GLMM is run.
"""
from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from build_z_value_review import LAYERS, MEASURES, fold_ci, label, measure_label, plot_points, references

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "cross_talker_generalization"
OLD = PROJECT / "analysis_update_2026-09-09"
OUT = PROJECT / "analysis_update_2026-09-11"
FIG = OUT / "figures"
TABLES = OUT / "tables"
SOURCES = {}
INVENTORY = []
SEED = 20260909
N_BOOT = 1000
PHONES = ["ih", "iy", "ae", "eh", "uh", "uw", "th", "dh", "s", "sh", "r", "l"]
COLORS = {"base": "#20639b", "ft": "#b43b35"}


def read(path):
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def save(fig, name, family, **details):
    for ext in ("png", "pdf", "svg"):
        kwargs = {"metadata": {"Author": "", "CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
        fig.savefig(FIG / f"{name}.{ext}", dpi=230, bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    INVENTORY.append(dict(id=name, family=family, path=(FIG / f"{name}.pdf").relative_to(ROOT).as_posix(), **details))


def phone_panels():
    folder = OLD / "an19_phone_alignment/tables"
    instances = read(folder / "figure2b_all_segments_similarity_instances.csv")
    language_means = read(folder / "figure2b_all_segments_similarity_language_means.csv")
    palette = read(OLD / "tables/language_branch_palette.csv")
    # The retained palette table is generated from Florian's world-map R script.
    print("Palette columns:", palette.columns.tolist(), flush=True)
    color_column = "color" if "color" in palette else "hex_color"
    colors = palette.set_index("branch")[color_column].to_dict()
    languages = language_means.language.tolist()
    by_talker = instances.groupby(["canonical_phone", "speaker_id", "language", "branch"]).agg(
        mean_similarity=("mean_similarity", "mean"), n_instances=("interval_id", "size")
    ).reset_index()
    assert by_talker.n_instances.sum() == len(instances) == 5114
    rows = []
    for phone_index, phone in enumerate(sorted(by_talker.canonical_phone.unique())):
        rng = np.random.default_rng(SEED + phone_index)
        for (language, branch), data in by_talker.loc[by_talker.canonical_phone.eq(phone)].groupby(["language", "branch"]):
            values = data.sort_values("speaker_id").mean_similarity.to_numpy()
            low = high = np.nan
            if len(values) > 1:
                draws = values[rng.integers(len(values), size=(N_BOOT, len(values)))].mean(axis=1)
                low, high = np.quantile(draws, [.025, .975])
            rows.append(dict(canonical_phone=phone, language=language, branch=branch,
                             mean_similarity=values.mean(), ci_low=low, ci_high=high,
                             n_talkers=len(values), n_instances=int(data.n_instances.sum())))
    means = pd.DataFrame(rows)
    by_talker.to_csv(TABLES / "phone_instance_weighted_talkers.csv", index=False)
    means.to_csv(TABLES / "phone_instance_weighted_language_means.csv", index=False)
    for i in range(3):
        selected = PHONES[i*4:i*4+4]
        fig, axes = plt.subplots(1, 4, figsize=(13.8, 7), sharey=True)
        fig.subplots_adjust(left=.105, right=.985, bottom=.115, top=.9, wspace=.23)
        for phone, ax in zip(selected, axes):
            stats = means.loc[means.canonical_phone.eq(phone)].set_index("language")
            all_values = []
            for y, language in enumerate(languages):
                points = by_talker.loc[by_talker.canonical_phone.eq(phone) & by_talker.language.eq(language)]
                if points.empty:
                    continue
                point = stats.loc[language]
                color = colors[point.branch]
                values = points.sort_values("speaker_id").mean_similarity.to_numpy()
                all_values.extend(values)
                offsets = np.linspace(-.15, .15, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(values, y+offsets, s=18, color=color, alpha=.6, zorder=3, clip_on=False)
                if point.n_talkers > 1:
                    ax.hlines(y, point.ci_low, point.ci_high, color=color, lw=1.5, zorder=2)
                ax.scatter(point.mean_similarity, y, marker="D", s=42, facecolors="none",
                           edgecolors=color, lw=1.3, zorder=4, clip_on=False)
            ax.set_title(phone.upper(), fontsize=14)
            ax.set_yticks(range(len(languages)), languages, fontsize=10)
            ax.set_ylim(len(languages)-.45, -.55)
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.tick_params(axis="x", labelsize=9)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
            ax.grid(axis="x", color="#eeeeee", linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if not all_values:
                ax.set_xticks([])
                ax.spines["bottom"].set_visible(False)
                ax.text(.5, .5, "Not observed", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        fig.suptitle("AN19", x=.105, ha="left", fontsize=14, y=.98)
        fig.supxlabel("Mean segment similarity to L1-English", fontsize=12, y=.02)
        save(fig, f"an19_phone_similarity_{i+1:02d}", "phone_similarity", phones=selected,
             weighting="equal instances within talker/phone; then equal talkers within L1",
             ci="1000 by-talker percentile bootstrap samples within L1/phone; no CI for n=1")
    return by_talker, means


def hve_panels():
    z = read(PROJECT / "analysis_update_2026-09-06/z_value_review/tables/revised_hve_training_fold_z.csv")
    ll = read(OLD / "tables/hve_retained_fold_loglik.csv")
    for dataset in ("AN19", "X21", "B23"):
        methods = [m for m in MEASURES if m in set(z.loc[z.dataset_id.eq(dataset)].measure)]
        assert set(methods) == set(ll.loc[ll.dataset_id.eq(dataset)].measure)
        for start in range(0, len(methods), 4):
            selected = methods[start:start+4]
            for metric, source in (("z", z), ("loglik", ll)):
                fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), layout="constrained")
                for ax, method in zip(axes.flat, selected):
                    for variant, color, offset in [("base", COLORS["base"], -.1), ("ft", COLORS["ft"], .1)]:
                        variant_col = "model_variant" if metric == "z" else "variant"
                        part = source.loc[source.dataset_id.eq(dataset) & source.measure.eq(method)
                                          & source[variant_col].eq(variant)].copy()
                        assert len(part) == len(LAYERS)*3
                        if metric == "loglik":
                            part["z_value"] = part.held_out_log_likelihood_per_trial
                        plot_points(ax, part, LAYERS, np.arange(len(LAYERS)), color, False, None, offset=offset)
                    ax.set_xticks(range(len(LAYERS)), [label(k) for k in LAYERS], rotation=60, ha="right", fontsize=7)
                    title = measure_label(method)
                    if dataset == "B23":
                        title += "\n(n = 97)" if method == "overall_order_sensitive" else "\n(n = 168)"
                    ax.set_title(title, fontsize=10)
                    ax.set_xlabel("Feature space", fontsize=9)
                    ax.set_ylabel("Training-fold Wald z" if metric == "z" else "Held-out log likelihood per word", fontsize=9)
                    if metric == "z":
                        ax.axhline(0, color="#999999", lw=.65, zorder=0)
                        for sign in (-1, 1):
                            ax.axhline(sign*1.96, color="#b56e00", lw=.8, ls=":", zorder=0)
                    ax.spines[["top", "right"]].set_visible(False)
                for ax in list(axes.flat)[len(selected):]:
                    ax.set_visible(False)
                fig.suptitle(f"{dataset} | HVE", fontsize=13)
                fig.legend(handles=[Line2D([], [], color=COLORS[v], label="Non-ASR-FT" if v=="base" else "ASR-FT")
                                    for v in ("base","ft")],
                           loc="outside lower center", ncol=2, frameon=False)
                save(fig, f"hve_{dataset.lower()}_{metric}_{start//4+1:02d}", f"hve_{metric}",
                     dataset=dataset, methods=selected,
                     scope="two-training-fold GLMM coefficient" if metric=="z" else "frozen predictor-only held-out per-word score")


def x21_comparison():
    source = read(PROJECT / "analysis_update_2026-09-01/tables/crossfitted_lrt_results.csv")
    data = source.loc[source.dataset_id.eq("X21") & source.variant.eq("base")].copy()
    assert len(data) == 4 and data.status.eq("ok").all()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), layout="constrained")
    for ax, comparison, title in zip(
        axes, ["predictor_beyond_condition", "condition_beyond_predictor"],
        ["Predictor beyond condition", "Condition beyond predictor"]
    ):
        part = data.loc[data.comparison_id.eq(comparison)].set_index("family").loc[["SBI", "HVE"]]
        bars = ax.bar(["SBI", "HVE"], part.chisq, color=["#20639b", "#757575"], width=.52)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Likelihood-ratio statistic")
        ax.set_ylim(0, part.chisq.max()*1.3)
        for bar, row in zip(bars, part.itertuples()):
            text = f"p = {row.p_value:.3g}\ndf = {int(row.df)}"
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+part.chisq.max()*.035,
                    text, ha="center", va="bottom", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("X21 | Combined-fold nested comparisons", fontsize=13)
    data.to_csv(TABLES / "x21_nested_comparisons.csv", index=False)
    save(fig, "x21_nested_comparisons", "nested_comparison",
         scope="retained combined-fold selected-model LRT, exploratory; not a held-out prediction score")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype":42, "svg.fonttype":"none", "font.size":11})
    phone_panels()
    hve_panels()
    x21_comparison()
    payload = {"new_model_fits":0, "new_dtw_pairs":0, "source_sha256":SOURCES, "figures":INVENTORY,
               "phone_update":"Per-phone views now use the same pairwise exponential transformation and instance weighting as the overall view; superseded word-type distance plots are not included."}
    (OUT / "panel_sources.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Prepared {len(INVENTORY)} figures.", flush=True)


if __name__ == "__main__":
    main()
