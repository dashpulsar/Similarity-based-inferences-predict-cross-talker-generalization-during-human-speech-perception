"""Recognize unconstrained IPA hypotheses for every AN19 recording, locally.

This is not forced alignment: the intended word never enters the model. CTC
emission frames are diagnostic indices, not validated phonetic boundaries.
Prepare and CPU-load modes never run audio inference. Default mode runs the
entire manifest, resuming only outputs with matching input/model/config hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import locale
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "cross_talker_generalization/configs/an19_phonetic_xeus.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def verify_model(config: dict, model_dir: Path) -> dict:
    """Fail closed before importing the pinned model's executable Python code."""
    files = {}
    for path in sorted(model_dir.rglob("*")):
        relative = path.relative_to(model_dir)
        if not path.is_file() or any(part in {".cache", "__pycache__"} for part in relative.parts):
            continue
        if path.name == config["weights_filename"]:
            continue
        if path.suffix not in {".py", ".yaml", ".json", ".md"}:
            continue
        files[relative.as_posix()] = sha256(path)
    bundle = hashlib.sha256("".join(k + ":" + v + "\n" for k, v in sorted(files.items())).encode()).hexdigest()
    if len(files) != config["reviewed_source_files"] or bundle != config["source_bundle_sha256"]:
        raise RuntimeError(f"Model code/config differs from the reviewed pinned snapshot: {len(files)} files; {bundle}")
    weights = model_dir / config["weights_filename"]
    if weights.stat().st_size != config["weights_bytes"]:
        raise RuntimeError("Incomplete or unexpected model weights size")
    weight_hash = sha256(weights)
    if weight_hash != config["weights_sha256"]:
        raise RuntimeError("Model safetensors SHA-256 mismatch")
    return {
        "model_id": config["model_id"], "revision": config["model_revision"],
        "license": config["model_license"], "weights_sha256": weight_hash,
        "source_bundle_sha256": bundle, "source_files_sha256": files,
        "code_review": {
            "reviewed_entrypoints": ["modeling_phoneticxeus.py", "configuration_phoneticxeus.py",
                                     "src/model/xeusphoneme/builders.py",
                                     "src/model/xeusphoneme/xeuspr_inference.py",
                                     "src/recipe/phone_recognition/greedy_ctc_strategy.py"],
            "local_source_imports": "Pinned wrapper adds its local model folder to sys.path and imports its bundled src package.",
            "inactive_capabilities": "Upstream supports snapshot downloads and a torch.load pickle checkpoint path. This runner sets offline mode, supplies a local model folder, and loads safetensors only (builder load_ckpt=False, hf_repo=None).",
            "scope": "Static review of executable entrypoints, imports and file/network/process calls; not a security certification."
        }
    }


def get_audio(row: dict, common_git_dir: Path) -> dict:
    """Resolve a checked-out WAV or its existing local Git-LFS object."""
    result = {key: row[key] for key in ("recording_id", "speaker_id", "accent", "word",
                                        "source_filename", "source_wav_relpath")}
    try:
        external = row.get("external_audio_path")
        path = Path(external) if external else REPO / row["source_wav_relpath"]
        if external and not path.is_absolute():
            raise ValueError("external_audio_path must be absolute")
        result["audio_source"] = "external_audio_manifest" if external else "repository_or_local_lfs"
        if external:
            result["external_source_filename"] = path.name
        expected = row.get("external_audio_sha256") or None
        if path.stat().st_size < 1024:
            pointer = path.read_bytes()
            if pointer.startswith(b"version https://git-lfs"):
                if external:
                    raise RuntimeError("Authoritative external audio is a Git-LFS pointer; repository fallback is prohibited")
                expected = next(line.split(":", 1)[1] for line in pointer.decode().splitlines()
                                if line.startswith("oid sha256:"))
                path = common_git_dir / "lfs/objects" / expected[:2] / expected[2:4] / expected
        digest = sha256(path)
        if expected and digest != expected:
            raise RuntimeError("Audio SHA-256 mismatch against the declared source hash")
        import soundfile as sf
        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError("Empty or invalid audio")
        result.update(status="ready", resolved_audio_path=str(path), audio_sha256=digest,
                      source_sampling_rate=info.samplerate, source_channels=info.channels,
                      source_frames=info.frames, duration_seconds=info.duration)
    except Exception as exc:
        result.update(status="input_error", error=f"{type(exc).__name__}: {exc}")
    return result


def load_model(config: dict, model_dir: Path, device: str):
    # Upstream opens its UTF-8 IPA JSON without an explicit encoding. Refuse a
    # locale-dependent decode instead of silently producing corrupted labels.
    if locale.getpreferredencoding(False).lower().replace("-", "") != "utf8":
        raise RuntimeError("The upstream IPA vocabulary requires UTF-8 mode. Run: python -X utf8 run_an19_phonetic_xeus.py ...")
    # Set before importing HF libraries: upstream's optional network path cannot run.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly if desired")
    model_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model_config._name_or_path = str(model_dir)
    model = AutoModel.from_pretrained(str(model_dir), config=model_config,
                                     trust_remote_code=True, local_files_only=True,
                                     use_safetensors=True, torch_dtype=torch.float32)
    vocabulary = json.loads((model_dir / model_config.vocab_file).read_text(encoding="utf-8"))
    id_to_token = {index: token for token, index in vocabulary.items()}
    if model.model.token_list != [id_to_token[index] for index in range(len(id_to_token))]:
        raise RuntimeError("Loaded IPA labels do not match the pinned UTF-8 vocabulary")
    return model.eval().to(device)


def recognize(model, audio: dict, device: str, sampling_rate: int) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    values, original_rate = sf.read(audio["resolved_audio_path"], dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(values.mean(axis=1)))
    if original_rate != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, sampling_rate)
    if not torch.isfinite(waveform).all():
        raise ValueError("Audio contains nonfinite samples")
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_values=waveform.unsqueeze(0).to(device)).logits[0]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Model logits contain nonfinite values")
        probabilities = logits.softmax(dim=-1)
        best_probability, ids = probabilities.max(dim=-1)
        ids = ids.cpu().tolist()
        best_probability = best_probability.cpu().tolist()
    blank = model.model.get_blank_id()
    vocabulary = model.model.token_list
    runs = []
    start = 0
    for stop in range(1, len(ids) + 1):
        if stop == len(ids) or ids[stop] != ids[start]:
            token_id = ids[start]
            if token_id != blank:
                token = vocabulary[token_id]
                runs.append({"token_id": token_id, "token": token,
                             "ctc_start_frame": start, "ctc_stop_frame_exclusive": stop,
                             "max_posterior": max(best_probability[start:stop]),
                             "special_token": token.startswith("<") and token.endswith(">")})
            start = stop
    phones = [run["token"] for run in runs if not run["special_token"]]
    confidence = [run["max_posterior"] for run in runs if not run["special_token"]]
    return {
        "status": "ok", "processed_transcript": "".join(phones).strip(),
        "predicted_transcript": "/".join(run["token"] for run in runs),
        "ipa_tokens": phones, "ctc_runs": runs, "n_ipa_tokens": len(phones),
        "mean_token_max_posterior": float(np.mean(confidence)) if confidence else None,
        "confidence_is_calibrated": False, "ctc_frames_are_phone_boundaries": False,
        "all_logits_finite": True, "ipa_vocabulary_encoding": "utf-8",
        "target_text_used_for_recognition": False, "n_model_frames": len(ids),
        "decoded_source_samples": len(values), "decoded_duration_seconds": len(values) / original_rate,
        "input_samples_16khz": len(waveform), "inference_seconds": time.perf_counter() - started
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audio-root", type=Path,
                        help="Authoritative audio root; requires --source-manifest and never falls back to repository WAVs")
    parser.add_argument("--source-manifest", type=Path,
                        help="CSV with recording_id, external_audio_relpath, sha256 and status for every recording")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--download", action="store_true", help="Download only the pinned public model snapshot before continuing")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true", help="Verify all audio/model files; never load the model or run inference")
    modes.add_argument("--check-load-only", action="store_true", help="Verify inputs and load model on CPU; never run audio inference")
    args = parser.parse_args()
    if not args.prepare_only and not args.check_load_only and not (args.audio_root and args.source_manifest):
        parser.error("Full inference requires --audio-root and --source-manifest. Repository audio is not an inference fallback.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_dir = (args.model_dir or resolve_path(config["model_dir"])).resolve()
    out = (args.output_dir or resolve_path(config["output_dir"])).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(config["model_id"], revision=config["model_revision"], local_dir=str(model_dir),
                          allow_patterns=["*.py", "*.json", "README.md", "src/**", "model.safetensors"], max_workers=4)
    print("Verifying pinned model code and 2.30 GB safetensors...", flush=True)
    integrity = verify_model(config, model_dir)
    json_dump(out / "model_integrity.json", integrity)
    manifest_path = resolve_path(config["manifest"])
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != config["expected_recordings"] or len({row["recording_id"] for row in rows}) != len(rows):
        raise RuntimeError("AN19 manifest count or unique recording IDs differs from the configured full corpus")
    audio_manifest_hash = None
    if bool(args.audio_root) != bool(args.source_manifest):
        raise RuntimeError("--audio-root and --source-manifest must be supplied together")
    if args.source_manifest:
        audio_root = args.audio_root.resolve(strict=True)
        with args.source_manifest.open(encoding="utf-8-sig", newline="") as handle:
            external_rows = list(csv.DictReader(handle))
        external = {row["recording_id"]: row for row in external_rows}
        if len(external) != len(external_rows) or set(external) != {row["recording_id"] for row in rows}:
            raise RuntimeError("External audio manifest must map exactly all AN19 recording IDs once")
        for row in rows:
            mapping = external[row["recording_id"]]
            if mapping.get("status") != "available":
                raise RuntimeError(f"External audio is not marked available for {row['recording_id']}")
            relative = mapping.get("external_audio_relpath", "")
            expected = mapping.get("sha256", "").lower()
            if not relative or Path(relative).is_absolute() or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise RuntimeError(f"Invalid external relative path or SHA-256 for {row['recording_id']}")
            path = (audio_root / relative).resolve(strict=True)
            path.relative_to(audio_root)  # Reject path traversal and links outside the authorized root.
            row["external_audio_path"] = str(path)
            row["external_audio_sha256"] = expected
        audio_manifest_hash = sha256(args.source_manifest)
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO, text=True).strip())
    with ThreadPoolExecutor(max_workers=config["input_workers"]) as executor:
        inputs = list(executor.map(lambda row: get_audio(row, common), rows))
    # Output provenance intentionally omits machine-specific absolute audio/cache paths.
    json_dump(out / "input_audit.json", [{k: v for k, v in item.items() if k != "resolved_audio_path"} for item in inputs])
    runtime = {package: importlib.metadata.version(package) for package in
               ["torch", "torchaudio", "transformers", "huggingface-hub", "numpy", "soundfile"]}
    run_key_payload = {"config": config, "manifest_sha256": sha256(manifest_path),
                       "external_audio_manifest_sha256": audio_manifest_hash,
                       "source_script_sha256": sha256(Path(__file__)),
                       "runtime": runtime, "requested_device": args.device or config["device"]}
    run_key = hashlib.sha256(json.dumps(run_key_payload, sort_keys=True).encode()).hexdigest()
    provenance = dict(run_key_payload, run_key=run_key, python=platform.python_version(),
                      python_utf8_mode=sys.flags.utf8_mode, default_text_encoding=locale.getpreferredencoding(False),
                      annotation_type="unconstrained automatic IPA hypotheses; not human labels",
                      all_recordings=len(inputs), input_status=dict(Counter(item["status"] for item in inputs)),
                      total_audio_seconds=sum(item.get("duration_seconds", 0) for item in inputs),
                      phone_boundaries_provided=False)
    json_dump(out / "run_provenance.json", provenance)
    print(json.dumps({"input_status": provenance["input_status"], "total_audio_seconds": provenance["total_audio_seconds"]}), flush=True)
    if args.prepare_only:
        print("Prepared all manifest recordings; no model or audio inference was run.", flush=True)
        return 0 if all(item["status"] == "ready" for item in inputs) else 2
    device = "cpu" if args.check_load_only else (args.device or config["device"])
    model = load_model(config, model_dir, device)
    if args.check_load_only:
        loaded = {"status": "loaded_on_cpu_without_audio_inference", "parameters": sum(p.numel() for p in model.parameters()),
                  "vocabulary_size": len(model.model.token_list), "blank_id": model.model.get_blank_id(), "runtime": runtime}
        json_dump(out / "runtime_load_check.json", loaded)
        print(json.dumps(loaded), flush=True)
        return 0
    destination = out / "recordings"
    destination.mkdir(exist_ok=True)
    started = time.perf_counter()
    outputs = []
    for index, item in enumerate(inputs, 1):
        target = destination / (item["recording_id"] + ".json")
        result = None
        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            if previous.get("run_key") != run_key or previous.get("audio_sha256") != item.get("audio_sha256"):
                raise RuntimeError(f"Existing output uses different inputs/config: {target}. Choose another --output-dir; nothing was overwritten.")
            if previous.get("status") == "ok":
                result = previous
        if result is None:
            result = {k: v for k, v in item.items() if k != "resolved_audio_path"}
            result["run_key"] = run_key
            if item["status"] == "ready":
                try:
                    result.update(recognize(model, item, device, config["sampling_rate"]))
                except Exception as exc:
                    result.update(status="inference_error", error=f"{type(exc).__name__}: {exc}")
            json_dump(target, result)
        outputs.append(result)
        if index % 100 == 0 or index == len(inputs):
            progress = {"completed": index, "total": len(inputs), "status_counts": dict(Counter(r["status"] for r in outputs)),
                        "elapsed_seconds": time.perf_counter() - started}
            json_dump(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    fields = ["recording_id", "speaker_id", "accent", "word", "source_filename", "status", "processed_transcript",
              "predicted_transcript", "n_ipa_tokens", "mean_token_max_posterior", "duration_seconds", "audio_sha256", "error"]
    with (out / "ipa_hypotheses.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(outputs)
    statuses = dict(Counter(item["status"] for item in outputs))
    json_dump(out / "summary.json", {"status": "complete" if statuses.get("ok", 0) == len(inputs) else "complete_with_errors",
              "manifest_recordings": len(inputs), "output_counts": statuses,
              "empty_ipa_hypotheses": sum(item.get("status") == "ok" and not item.get("ipa_tokens") for item in outputs),
              "elapsed_seconds": time.perf_counter() - started, "run_key": run_key,
              "interpretation": "Machine-generated IPA hypotheses, not ground-truth transcriptions or phone boundaries."})
    return 0 if statuses.get("ok", 0) == len(inputs) else 2


if __name__ == "__main__":
    sys.exit(main())
