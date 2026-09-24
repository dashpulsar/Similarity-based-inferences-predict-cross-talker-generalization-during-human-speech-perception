"""Plot instance-weighted AN19 segment similarity with talker-bootstrap CI.

Reuse the recorded same-word/same-position DTW pairs. No acoustic model,
feature extraction, DTW computation, or behavioral model is rerun.
"""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UPDATE = ROOT / "cross_talker_generalization/analysis/speech"
SOURCE = UPDATE / "an19_phone_alignment"
TABLES = SOURCE / "tables"
EXPORT = ROOT / "output/figures/an19_phonemes"
PDF = ROOT / "output/pdf/AN19_all_segments_similarity.pdf"
STEM = "figure2b_all_segments_similarity"
SIMILARITY_K = 1.0  # Fixed Figure 2a convention, not optimized or rescaled here.
N_BOOT = 1000
BOOTSTRAP_SEED = 20260909
MATCH = ["comparison_word", "canonical_sequence", "canonical_position", "canonical_phone"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(targets, pairs, language_map):
    """Average similarities over reference recordings, speakers, then instances."""
    selected = targets.loc[targets.six_reference_complete].copy()
    assert selected.interval_id.is_unique
    assert not selected.lexical_identity_unresolved.any()
    assert not selected.no_feature_center.any()
    pairs = pairs.loc[pairs.target_interval_id.isin(selected.interval_id)].copy()
    assert not pairs.duplicated(["target_interval_id", "reference_interval_id"]).any()
    assert np.isfinite(pairs.raw_distance).all() and pairs.raw_distance.ge(0).all()

    # Transform each distance BEFORE averaging: mean(exp(-d)) != exp(-mean(d)).
    pairs["similarity"] = np.exp(-SIMILARITY_K * pairs.raw_distance)
    by_reference_speaker = pairs.groupby(["target_interval_id", "english_speaker_id"]).agg(
        similarity=("similarity", "mean"),
        distance=("raw_distance", "mean"),
        n_recordings=("reference_interval_id", "size"),
    )
    by_instance = by_reference_speaker.groupby("target_interval_id").agg(
        mean_similarity=("similarity", "mean"),
        recomputed_reference_dtw=("distance", "mean"),
        n_reference_speakers=("similarity", "size"),
        n_reference_recordings=("n_recordings", "sum"),
    )
    assert by_instance.n_reference_speakers.eq(6).all()
    assert set(by_instance.index) == set(selected.interval_id)
    instances = selected.merge(
        by_instance, left_on="interval_id", right_index=True, validate="one_to_one"
    )
    assert np.allclose(
        instances.mean_reference_dtw, instances.recomputed_reference_dtw,
        rtol=1e-12, atol=1e-12,
    )
    instances = instances.merge(
        language_map[["speaker_id", "language", "branch"]],
        on="speaker_id", validate="many_to_one", how="left",
    )
    assert instances[["language", "branch"]].notna().all().all()
    instances["instance_weight"] = 1 / instances.groupby("speaker_id").interval_id.transform("size")
    assert np.allclose(instances.groupby("speaker_id").instance_weight.sum(), 1)

    # Every retained interval is one instance, including repeated phone types
    # and different positions in the same word. No intervening word/type mean.
    talkers = instances.groupby(["speaker_id", "language", "branch"]).agg(
        mean_similarity=("mean_similarity", "mean"),
        n_segment_instances=("interval_id", "size"),
        n_segment_types=("canonical_phone", "nunique"),
        n_word_types=("comparison_word", "nunique"),
    ).reset_index()
    groups = talkers.groupby(["language", "branch"]).agg(
        mean_similarity=("mean_similarity", "mean"),
        n_talkers=("speaker_id", "size"),
        n_segment_instances=("n_segment_instances", "sum"),
    ).reset_index().sort_values(
        ["mean_similarity", "language"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    groups.insert(0, "similarity_rank", np.arange(1, len(groups) + 1))
    assert groups.mean_similarity.is_monotonic_decreasing
    assert instances.mean_similarity.between(0, 1).all()
    assert talkers.mean_similarity.between(0, 1).all()
    assert int(groups.n_talkers.sum()) == len(talkers)
    assert int(talkers.n_segment_instances.sum()) == len(instances)

    # Numerical verification of the requested equal-instance weighting.
    weighted = (instances.mean_similarity * instances.instance_weight).groupby(instances.speaker_id).sum()
    actual = talkers.set_index("speaker_id").mean_similarity
    assert np.allclose(weighted.loc[actual.index], actual, rtol=1e-12, atol=1e-15)
    columns = [
        "interval_id", "recording_id", "speaker_id", "language", "branch",
        *MATCH, "mean_similarity", "mean_reference_dtw",
        "n_reference_speakers", "n_reference_recordings", "instance_weight",
    ]
    return instances[columns], talkers, groups, len(pairs)


def add_talker_intervals(talkers, groups):
    """Percentile CI for each L1 mean, conditional on fixed reference/phone means."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for language, data in talkers.groupby("language", sort=True):
        values = data.sort_values("speaker_id").mean_similarity.to_numpy()
        low = high = np.nan
        if len(values) > 1:
            samples = values[rng.integers(len(values), size=(N_BOOT, len(values)))].mean(axis=1)
            low, high = np.quantile(samples, [.025, .975])
        rows.append({"language": language, "ci_low": low, "ci_high": high,
                     "ci_status": "talker_bootstrap_95pct" if len(values) > 1 else "not_estimable_one_talker"})
    result = groups.merge(pd.DataFrame(rows), on="language", validate="one_to_one", sort=False)
    assert result.mean_similarity.is_monotonic_decreasing
    assert result.loc[result.n_talkers.eq(1), ["ci_low", "ci_high"]].isna().all().all()
    multiple = result.loc[result.n_talkers.gt(1)]
    assert multiple[["ci_low", "ci_high"]].notna().all().all()
    assert multiple.ci_low.le(multiple.ci_high).all()
    return result


def draw(talkers, groups, colors):
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 6.8))
    fig.subplots_adjust(left=.235, right=.975, bottom=.12, top=.935)
    for y, group in enumerate(groups.itertuples()):
        data = talkers.loc[talkers.language.eq(group.language)].sort_values("speaker_id")
        # Fixed visual offsets, independent of the bootstrap used for intervals.
        offsets = np.linspace(-.17, .17, len(data)) if len(data) > 1 else np.zeros(1)
        color = colors[group.branch]
        if group.n_talkers > 1:
            ax.hlines(y, group.ci_low, group.ci_high, color=color, linewidth=2, zorder=2)
            ax.vlines([group.ci_low, group.ci_high], y-.075, y+.075,
                      color=color, linewidth=1.5, zorder=2)
        ax.scatter(data.mean_similarity, y + offsets, color=color, s=37, alpha=.68,
                   edgecolors="white", linewidths=.45, zorder=3)
        ax.scatter(group.mean_similarity, y, marker="D", s=95, facecolors="none",
                   edgecolors=color, linewidths=1.8, zorder=4)
    ax.set_yticks(range(len(groups)),
                  [f"{r.language} (n={r.n_talkers})" for r in groups.itertuples()])
    ax.set_ylim(len(groups) - .45, -.55)
    upper = np.ceil(talkers.mean_similarity.max() * 1.12 / .001) * .001
    ax.set_xlim(0, upper)
    ax.xaxis.set_major_locator(MultipleLocator(.001))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("Mean segment similarity to L1-English", labelpad=12)
    ax.grid(axis="x", color="#e7e7e7", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("AN19", loc="left", fontsize=14, pad=10)
    paths = [EXPORT / "AN19_all_segments_similarity.png",
             EXPORT / "AN19_all_segments_similarity.svg", PDF]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"metadata": {"Title": "AN19 segment-instance similarity to English",
                               "Author": "", "CreationDate": None, "ModDate": None}} if path.suffix == ".pdf" else {}
        fig.savefig(path, dpi=250, facecolor="white", **kwargs)
    plt.close(fig)
    return paths


def main():
    metadata_path = TABLES / "figure2b_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pair_path = TABLES / "figure2b_matched_phone_pairs.csv"
    target_path = TABLES / "figure2b_target_phone_estimates.csv"
    reference_path = TABLES / "figure2b_english_reference_tokens.csv"
    language_path = UPDATE / "tables/figure_si_talker_language_map.csv"
    palette_path = UPDATE / "sources/language_capitals_worldmap.R"
    for path in [pair_path, target_path, reference_path]:
        assert sha(path) == metadata["output_hashes"][path.relative_to(SOURCE).as_posix()]
    for path in [language_path, palette_path]:
        assert sha(path) == metadata["source_hashes"][path.relative_to(ROOT).as_posix()]

    targets = pd.read_csv(target_path)
    pairs = pd.read_csv(pair_path)
    references = pd.read_csv(reference_path)
    assert references.interval_id.is_unique
    assert not references.duplicated(["speaker_id", "audio_sha256", *MATCH]).any()
    # Check cached pair identities, rather than rematching or recomputing DTW.
    check = pairs.merge(
        references, left_on="reference_interval_id", right_on="interval_id",
        how="left", validate="many_to_one",
    )
    assert check.speaker_id.eq(check.english_speaker_id).all()
    check = check.merge(
        targets[["interval_id", *MATCH]], left_on="target_interval_id",
        right_on="interval_id", suffixes=("_reference", "_target"),
        how="left", validate="many_to_one",
    )
    assert all(check[f"{key}_reference"].eq(check[f"{key}_target"]).all() for key in MATCH)
    language_map = pd.read_csv(language_path)
    language_map = language_map.loc[language_map.dataset.eq("AN19")]
    instances, talkers, groups, n_pairs = summarize(targets, pairs, language_map)
    groups = add_talker_intervals(talkers, groups)
    assert len(instances) == metadata["six_reference_complete_target_intervals"]
    assert len(talkers) == 36

    palette_text = palette_path.read_text(encoding="utf-8-sig")
    block = re.search(r"family_colors\s*<-\s*c\((.*?)\n\)", palette_text, re.S).group(1)
    colors = dict(re.findall(r'"([^"\n]+)"\s*=\s*"(#[0-9a-fA-F]{6})"', block))
    assert set(groups.branch).issubset(colors)
    outputs = []
    for name, frame in [("instances", instances), ("talkers", talkers), ("language_means", groups)]:
        path = TABLES / f"{STEM}_{name}.csv"
        frame.to_csv(path, index=False)
        outputs.append(path)
    outputs.extend(draw(talkers, groups, colors))
    notes = {
        "statistic": "Mean pairwise exponential similarity, with equal segment-instance weighting within L2 talker",
        "similarity": "exp(-k * raw_distance), before any averaging",
        "k": SIMILARITY_K,
        "k_basis": "Unchanged Figure 2a convention; not fitted, median-normalized or selected using behavior",
        "aggregation": "Mean across distinct reference recordings within each English speaker -> equal mean across six English speakers per target instance -> equal mean across eligible segment instances per L2 talker -> equal mean across L2 talkers per L1",
        "reference_deduplication": "Cached references deduplicated by speaker, audio SHA-256, word, canonical sequence, position and phone",
        "instance_weight": "1 / number of eligible segment instances for that L2 talker; no intermediate word-type or phone-type means",
        "coverage": "Original six-reference-complete sample, no new filtering",
        "eligible_segment_instances": len(instances),
        "matched_pair_rows_used": n_pairs,
        "l2_talkers": len(talkers), "language_groups": len(groups),
        "instances_per_talker": [int(talkers.n_segment_instances.min()), int(talkers.n_segment_instances.max())],
        "language_order": groups.language.tolist(),
        "ordering": "Descending L1-group mean similarity",
        "bootstrap_replicates": N_BOOT, "bootstrap_seed": BOOTSTRAP_SEED,
        "ci": "2.5th and 97.5th percentiles of 1,000 bootstrap means; resample L2 talkers within each L1 with replacement, holding English references and within-talker instance means fixed; no CI for n=1",
        "ci_scope": "Uncertainty in the L1-group mean conditional on the retained sample and automatic annotations, not an individual-talker prediction interval",
        "presentation": "No in-figure legend, methods subtitle or footer; brief AN19 title; methods and symbol definitions in accompanying text",
        "dot": "One L2 talker", "open_diamond": "Equal-talker L1-group mean; for n=1 it coincides with the individual",
        "limitations": ["Automatically aligned intended phones are not manually verified",
                        "Only 5,114 of 16,086 L2 intervals meet the existing six-reference coverage rule",
                        "Available segment and word coverage differs across talkers",
                        "Descriptive similarity, not a listener-accuracy probability or an L1-effect significance test"],
        "alignment_features_dtw_or_glmm_rerun": False,
        "replaces_for_current_request": "AN19_all_segments: old equal-type DTW-distance/CI view, retained but superseded",
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p)
                          for p in [pair_path, target_path, reference_path, language_path, palette_path, metadata_path, Path(__file__).resolve()]},
        "output_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in outputs},
    }
    (TABLES / f"{STEM}_metadata.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(json.dumps({"segment_instances": len(instances), "l2_talkers": len(talkers),
                      "bootstrap_replicates": N_BOOT, "outputs": [str(p) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
