"""Export coverage and filename-case checks, without running recognition again."""
from __future__ import annotations

from collections import Counter
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = HERE / "nygaard_audio"
SOURCE = REPO / "cross_talker_generalization/analysis_update_2026-09-09/an19_phone_alignment/tables/nygaard_audio_source_manifest.csv"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    completed = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    if completed["status"] != "complete":
        raise RuntimeError("The full external-source run has not completed successfully")
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        sources = list(csv.DictReader(handle))
    config = json.loads((REPO / "cross_talker_generalization/configs/an19_phonetic_xeus.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((Path(config["model_dir"]).expanduser() / "src/model/xeusphoneme/resources/ipa_vocab.json").read_text(encoding="utf-8"))
    id_to_token = {index: token for token, index in vocabulary.items()}
    quality = []
    cases = []
    for source in sources:
        record = json.loads((RUN / "recordings" / (source["recording_id"] + ".json")).read_text(encoding="utf-8"))
        if source["sha256"] != record["audio_sha256"]:
            raise RuntimeError("Source audit and recognition refer to different audio bytes")
        if record["audio_source"] != "external_audio_manifest" or record["target_text_used_for_recognition"]:
            raise RuntimeError("Unexpected recognition source or target-conditioned output")
        posteriors = [r["max_posterior"] for r in record["ctc_runs"] if not r["special_token"]]
        samples = int(source["decoded_samples"])
        rate = int(source["sample_rate"])
        expected_resampled = math.ceil(samples * 16000 / rate)
        row = {
            "recording_id": record["recording_id"], "speaker_id": record["speaker_id"],
            "accent": record["accent"], "intended_word_metadata": record["word"],
            "source_filename": source["source_filename"], "external_filename": source["external_filename"],
            "mapping_method": source["mapping_method"], "status": record["status"],
            "processed_transcript": record["processed_transcript"], "predicted_transcript": record["predicted_transcript"],
            "ipa_tokens_json": json.dumps(record["ipa_tokens"], ensure_ascii=False),
            "n_ipa_tokens": record["n_ipa_tokens"], "empty_hypothesis": not record["ipa_tokens"],
            "mean_token_max_posterior": record["mean_token_max_posterior"],
            "emitted_token_posteriors_finite": all(math.isfinite(value) for value in posteriors),
            "emitted_token_posteriors_in_0_1": all(0 <= value <= 1 for value in posteriors),
            "all_logits_finite": record["all_logits_finite"],
            "ipa_tokens_match_utf8_vocabulary": all(run["token"] == id_to_token[run["token_id"]] for run in record["ctc_runs"]),
            "decoded_source_samples": samples, "source_sampling_rate": rate,
            "decoded_duration_seconds": float(source["decoded_duration_seconds"]),
            "recorded_decoded_length_matches_source_audit": samples == record["decoded_source_samples"],
            "model_input_samples_16khz": record["input_samples_16khz"],
            "model_input_duration_seconds": record["input_samples_16khz"] / 16000,
            "resampled_length_matches_decoded_audio": expected_resampled == record["input_samples_16khz"],
            "wav_header_duration_seconds": record["duration_seconds"],
            "audio_sha256": record["audio_sha256"], "source_hash_matches": True,
            "interpretation": "automatic unconstrained IPA hypothesis; not ground truth or automatic relabeling"
        }
        quality.append(row)
        selected = []
        if "hw74" in source["source_filename"].lower():
            selected.append("hw74_wave_wade")
        if "cheif" in source["source_filename"].lower() or record["word"].lower() == "cheif":
            selected.append("cheif_filename_typo")
        if source["source_filename"].lower() != source["external_filename"].lower():
            selected.append("external_filename_differs_hash_matched")
        if selected:
            cases.append(dict(case=";".join(selected), **row))
    if len(quality) != 6261:
        raise RuntimeError("Full-corpus audit must contain 6,261 rows")
    write_csv(RUN / "recognition_quality_audit.csv", quality)
    write_csv(RUN / "lexical_filename_case_audit.csv", cases)
    empty = [row for row in quality if row["empty_hypothesis"]]
    # Retain a header even when there are no empty hypotheses.
    with (RUN / "empty_hypotheses.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality[0]))
        writer.writeheader()
        writer.writerows(empty)
    old = list((HERE / "recordings").glob("*.json"))
    exact_overlap = 0
    for path in old:
        previous = json.loads(path.read_text(encoding="utf-8"))
        current = json.loads((RUN / "recordings" / path.name).read_text(encoding="utf-8"))
        exact_overlap += (previous["audio_sha256"] == current["audio_sha256"]
                          and [r["token_id"] for r in previous["ctc_runs"]] == [r["token_id"] for r in current["ctc_runs"]]
                          and previous["mean_token_max_posterior"] == current["mean_token_max_posterior"])
    summary = {
        "recordings": len(quality), "successful_inference": completed["output_counts"]["ok"],
        "nonempty_hypotheses": len(quality) - len(empty), "empty_hypotheses": len(empty),
        "total_decoded_audio_seconds": sum(row["decoded_duration_seconds"] for row in quality),
        "all_source_hashes_match": all(row["source_hash_matches"] for row in quality),
        "all_model_lengths_match_decoded_audio": all(row["resampled_length_matches_decoded_audio"] for row in quality),
        "all_emitted_posteriors_finite": all(row["emitted_token_posteriors_finite"] for row in quality),
        "all_emitted_posteriors_in_0_1": all(row["emitted_token_posteriors_in_0_1"] for row in quality),
        "all_logits_finite": all(row["all_logits_finite"] for row in quality),
        "all_ipa_tokens_match_utf8_vocabulary": all(row["ipa_tokens_match_utf8_vocabulary"] for row in quality),
        "all_recorded_decoded_lengths_match_source_audit": all(row["recorded_decoded_length_matches_source_audit"] for row in quality),
        "header_decoded_duration_discrepancies": sum(abs(row["wav_header_duration_seconds"] - row["decoded_duration_seconds"]) > 1e-6 for row in quality),
        "lexical_cases": dict(Counter(row["case"] for row in cases)),
        "retained_repository_source_overlap": len(old), "exact_overlap_numeric_token_ids_and_posterior": exact_overlap,
        "overlap_comparison": "Compare numeric CTC token IDs and posterior scores only; archived pre-fix IPA text used the wrong Windows encoding.",
        "interpretation": "Empty outputs are retained and explicitly flagged. Finite model scores and source agreement are computational checks, not phonetic accuracy validation. No filename or lexical label was changed."
    }
    (RUN / "recognition_quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def check_empty_logits(audio_root: Path) -> None:
    """Check that empty greedy decodes do not conceal nonfinite model logits."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    script = REPO / "cross_talker_generalization/scripts/run_an19_phonetic_xeus.py"
    spec = importlib.util.spec_from_file_location("an19_xeus_runner", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    model_dir = runner.resolve_path(config["model_dir"])
    runner.verify_model(config, model_dir)
    model = runner.load_model(config, model_dir, "cuda")
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        sources = {row["recording_id"]: row for row in csv.DictReader(handle)}
    with (RUN / "empty_hypotheses.csv").open(encoding="utf-8-sig", newline="") as handle:
        empty = list(csv.DictReader(handle))
    checked = []
    root = audio_root.resolve(strict=True)
    for item in empty:
        source = sources[item["recording_id"]]
        path = (root / source["external_audio_relpath"]).resolve(strict=True)
        path.relative_to(root)
        if runner.sha256(path) != source["sha256"]:
            raise RuntimeError("Audio hash mismatch in empty-output diagnostic")
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(np.ascontiguousarray(values.mean(axis=1)))
        if rate != 16000:
            waveform = torchaudio.functional.resample(waveform, rate, 16000)
        with torch.inference_mode():
            logits = model(input_values=waveform.unsqueeze(0).cuda()).logits[0]
            finite = bool(torch.isfinite(logits).all())
            probabilities = logits.softmax(-1)
            best = logits.argmax(-1).cpu().tolist()
        blank = model.model.get_blank_id()
        collapsed = [token for index, token in enumerate(best) if token != blank and (index == 0 or token != best[index - 1])]
        tokens = [model.model.token_list[token] for token in collapsed]
        ipa = [token for token in tokens if not (token.startswith("<") and token.endswith(">"))]
        checked.append({"recording_id": item["recording_id"], "all_logits_finite": finite,
                        "greedy_ipa_still_empty": not ipa, "nonblank_argmax_frames": sum(token != blank for token in best),
                        "argmax_token_set": sorted({model.model.token_list[token] for token in best}),
                        "max_nonblank_posterior": float(probabilities[:, [i for i in range(probabilities.shape[1]) if i != blank]].max()),
                        "decoded_audio_rms": float(np.sqrt(np.mean(values ** 2))),
                        "decoded_audio_peak": float(np.max(np.abs(values))),
                        "decoded_duration_seconds": len(values) / rate})
    result = {"recordings_rechecked": len(checked),
              "all_logits_finite": all(row["all_logits_finite"] for row in checked),
              "all_greedy_ipa_still_empty": all(row["greedy_ipa_still_empty"] for row in checked),
              "all_audio_rms_positive": all(row["decoded_audio_rms"] > 0 for row in checked),
              "rows": checked,
              "interpretation": "Post-run diagnostics of empty hypotheses; not phonetic validation or replacement transcriptions."}
    (RUN / "empty_hypothesis_logits_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-empty-logits", action="store_true")
    parser.add_argument("--audio-root", type=Path)
    arguments = parser.parse_args()
    if arguments.check_empty_logits:
        if not arguments.audio_root:
            parser.error("--check-empty-logits requires --audio-root")
        check_empty_logits(arguments.audio_root)
    else:
        main()
