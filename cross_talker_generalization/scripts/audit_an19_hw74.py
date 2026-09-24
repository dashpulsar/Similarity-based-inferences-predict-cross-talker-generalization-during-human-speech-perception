"""Inspect the wave/wade naming conflict without relabeling inputs.

Optional local-only Whisper screening is a diagnostic, not a verified transcript.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "cross_talker_generalization/analysis/model_comparison/reference_checks/collaborator_report/tables"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", action="store_true")
    args = parser.parse_args()
    manifest = pd.read_csv(REPO / "data/manifests/AN19-stimulus-manifest.csv")
    rows = manifest.loc[manifest.source_filename.str.contains("hw74", case=False)].copy()
    common = Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO, text=True).strip())
    model = None
    if args.asr:
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8",
                             cpu_threads=8, local_files_only=True)
    results = []
    with h5py.File(REPO / "data/features/an19_hubert_full_corpus_tsne_3d.h5", "r") as store:
        for row in rows.itertuples():
            path = REPO / row.source_wav_relpath
            pointer = path.read_bytes() if path.stat().st_size < 1024 else b""
            expected_hash = None
            if pointer.startswith(b"version https://git-lfs"):
                expected_hash = next(s.split(":")[1] for s in pointer.decode().splitlines()
                                     if s.startswith("oid sha256:"))
                path = common / "lfs/objects" / expected_hash[:2] / expected_hash[2:4] / expected_hash
            available = path.exists()
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if available else None
            if expected_hash and available:
                assert actual_hash == expected_hash
            key_present = row.recording_id in store["tr_24"][row.speaker_id]
            transcript = None
            # The English references and four actual control-test L2 speakers.
            if model and available and (row.accent == "ENG" or row.speaker_id.rsplit(".", 1)[-1]
                                        in {"km3", "km6", "sm4", "sm5"}):
                segments, _ = model.transcribe(str(path), language="en", beam_size=5,
                                               condition_on_previous_text=False, vad_filter=False)
                transcript = " ".join(s.text.strip() for s in segments)
                print(row.speaker_id, row.word, repr(transcript), flush=True)
            results.append(dict(speaker_id=row.speaker_id, accent=row.accent, file_word=row.word,
                                source_filename=row.source_filename, recording_id=row.recording_id,
                                real_audio_available=available, audio_sha256=actual_hash,
                                tr24_feature_present=key_present, asr_screening=transcript))
    output = pd.DataFrame(results)
    OUT.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT / "an19_hw74_naming_audit.csv", index=False)
    summary = {"status": "unresolved lexical-label conflict; no automatic alias applied",
               "file_labels": output.groupby(["accent", "file_word"]).size().to_dict().__str__(),
               "real_audio_available": int(output.real_audio_available.sum()),
               "features_present": int(output.tr24_feature_present.sum()),
               "asr": "local Whisper large-v3-turbo, English, beam 5, no lexical prompt; not ground truth"
                      if model else "not run"}
    (OUT / "an19_hw74_naming_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
