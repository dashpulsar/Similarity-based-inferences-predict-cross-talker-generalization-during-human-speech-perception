"""Map the user-selected audio directory to AN19 recording IDs, without aliases.

No original audio, feature file, or canonical manifest is modified. A renamed
file is mapped only by a unique identical SHA-256 within the same speaker;
unmatched/ambiguous recordings never fall back to repository audio.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'cross_talker_generalization/analysis/speech/an19_phone_alignment'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    audio_root = args.audio_root.resolve(strict=True)
    manifest = pd.read_csv(ROOT/'data/manifests/AN19-stimulus-manifest.csv', keep_default_na=False)
    previous = pd.read_csv(OUT/'tables/recording_alignment_status.csv',keep_default_na=False).set_index('recording_id')
    indexed = defaultdict(list)
    files = sorted(p for p in audio_root.rglob('*') if p.is_file() and p.suffix.lower()=='.wav')
    for path in files:
        indexed[path.name.casefold()].append(path)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes=dict(zip(files,pool.map(lambda p:hashlib.sha256(p.read_bytes()).hexdigest(),files)))
    by_hash=defaultdict(list)
    for path,sha in hashes.items():
        by_hash[sha].append(path)

    def inspect(row):
        result = {k:row[k] for k in ['recording_id','speaker_id','source_filename','word','source_l1_language']}
        candidates = indexed[row['source_filename'].casefold()]
        result.update(status='missing',repo_audio_sha256=previous.loc[row['recording_id'],'sha256'],
                      mapping_method='exact_filename_case_insensitive')
        if not candidates:
            candidates=[p for p in by_hash[result['repo_audio_sha256']]
                        if p.name.casefold().startswith(row['source_speaker_code'].casefold())]
            result['mapping_method']='unique_identical_sha256_same_speaker_renamed_file'
        result['candidate_count']=len(candidates)
        if len(candidates)!=1:
            result['status']='ambiguous' if candidates else 'missing'
            result['candidate_paths']=json.dumps([p.relative_to(audio_root).as_posix() for p in candidates])
            return result
        path=candidates[0]
        result['external_audio_relpath']=path.relative_to(audio_root).as_posix()
        result['external_filename']=path.name
        try:
            result['sha256']=hashes[path]
            result['file_bytes']=path.stat().st_size
            audio,rate=sf.read(path,dtype='float32',always_2d=True)
            result.update(status='available',sample_rate=rate,channels=audio.shape[1],decoded_samples=len(audio),
                          decoded_duration_seconds=len(audio)/rate,
                          bytes_equal_to_repo=result['sha256']==result['repo_audio_sha256'])
        except Exception as exc:
            result.update(status='decode_error',error=f'{type(exc).__name__}: {exc}')
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        table=pd.DataFrame(pool.map(inspect,manifest.to_dict('records')))
    output_csv=OUT/'tables/nygaard_audio_source_manifest.csv'
    table.to_csv(output_csv,index=False)
    used=set(table.get('external_audio_relpath',pd.Series(dtype=str)).dropna())
    unused=[{'external_audio_relpath':p.relative_to(audio_root).as_posix(),'source_filename':p.name}
            for p in files if p.relative_to(audio_root).as_posix() not in used]
    pd.DataFrame(unused,columns=['external_audio_relpath','source_filename']).to_csv(OUT/'tables/nygaard_audio_unmatched_files.csv',index=False)
    summary={'source_root_label':'user-selected July/Nygaard_audio',
             'mapping_rule':'unique exact filename (case-insensitive), or unique identical SHA-256 in the same speaker for renamed files; no word aliases or repository fallback',
             'mapping_method_counts':table.mapping_method.value_counts().to_dict(),
             'manifest_recordings':len(table),'directory_wav_files':len(files),
             'matched_available':int(table.status.eq('available').sum()),
             'status_counts':table.status.value_counts().to_dict(),
             'byte_identical_to_preliminary_repo_source':int(table.get('bytes_equal_to_repo',pd.Series(dtype=bool)).eq(True).sum()),
             'unused_directory_wavs':len(unused),
             'decoded_duration_seconds':float(table.get('decoded_duration_seconds',pd.Series(dtype=float)).sum()),
             'mapping_csv_sha256':hashlib.sha256(output_csv.read_bytes()).hexdigest(),
             'original_files_modified':False}
    (OUT/'nygaard_audio_source_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(table.loc[table.status.ne('available') | table.get('bytes_equal_to_repo',False).ne(True)].to_string(index=False))


if __name__=='__main__':
    main()
