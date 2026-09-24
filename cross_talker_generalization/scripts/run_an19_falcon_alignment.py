"""Full AN19 target-conditioned phoneme alignment with pinned FALCON weights.

Retains every manifest recording, annotates locally, and never edits raw audio or
HuBERT feature stores. Uses the official forward/decoder unchanged. Download and
dependency preparation commands are documented with the resulting annotations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'cross_talker_generalization'
BASE_OUT = PROJECT/'analysis/speech/an19_phone_alignment'
OUT = BASE_OUT/'nygaard_audio'
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, str(PROJECT/'artifacts/phone_alignment_runtime'))
sys.path.insert(0, str(PROJECT/'artifacts/vendor/FALCON'))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from next_frame_classifier import NextFrameClassifier
import utils as falcon_utils


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision(path):
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()


def load_dictionary(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith(';;;'):
            continue
        word, *phones = line.split('#')[0].split()
        word = re.sub(r'\(\d+\)$', '', word).lower()
        result.setdefault(word, []).append(phones)
    return result


def decode_recording(row, audio_root):
    try:
        path = (audio_root/row['external_audio_relpath']).resolve(strict=True)
        if not path.is_relative_to(audio_root):
            raise ValueError('Audio path is outside the selected source directory')
        expected = row['expected_audio_sha256']
        actual = digest(path)
        if actual != expected:
            raise ValueError('Selected source audio SHA mismatch; no repository fallback')
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        channels = audio.shape[1]
        mono = audio.mean(axis=1)
        tensor = torch.from_numpy(mono.copy())
        if rate != 16000:
            tensor = torchaudio.functional.resample(tensor, rate, 16000)
        return {'row':row, 'audio':tensor, 'sha256':actual, 'source_rate':rate,
                'source_channels':channels,'decoded_samples':len(mono),
                'duration_seconds':len(mono)/rate,'resampled_samples':len(tensor)}
    except Exception as exc:
        return {'row':row,'error':f'{type(exc).__name__}: {exc}'}


def save_textgrid(path, result):
    duration = result['duration_seconds']
    intervals = result['intervals']
    tiers = [('canonical_phones_automatic', [(p['start_seconds'],p['end_seconds'],p['canonical_phone']) for p in intervals]),
             ('falcon_lh39_targets', [(p['start_seconds'],p['end_seconds'],p['model_phone']) for p in intervals]),
             ('target_word_not_verified_transcript', [(0,duration,result.get('annotation_word',result['word']))])]
    lines = ['File type = "ooTextFile"','Object class = "TextGrid"','',
             'xmin = 0',f'xmax = {duration:.9f}','tiers? <exists>',f'size = {len(tiers)}','item []:']
    for ti,(name,segments) in enumerate(tiers,1):
        lines += [f'    item [{ti}]:','        class = "IntervalTier"',f'        name = "{name}"',
                  '        xmin = 0',f'        xmax = {duration:.9f}',f'        intervals: size = {len(segments)}']
        for si,(start,end,label) in enumerate(segments,1):
            escaped = label.replace('"','""')
            lines += [f'        intervals [{si}]:',f'            xmin = {start:.9f}',
                      f'            xmax = {end:.9f}',f'            text = "{escaped}"']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def align_recording(item, model, lexicon, config, config_sha):
    row = item['row']
    record = {k:row[k] for k in ['recording_id','speaker_id','word','accent','source_l1_language',
                                'source_filename','source_wav_relpath','legacy_key_collision',
                                'external_audio_relpath','external_filename','mapping_method']}
    record.update(config_sha256=config_sha, human_verified=False, status='failed',
                  annotation_word=row['annotation_word'])
    if 'error' in item:
        record['error'] = item['error']
        return record
    record.update({k:v for k,v in item.items() if k not in ['row','audio']})
    choices = lexicon.get(row['annotation_word'].lower())
    if not choices:
        record['error'] = 'No canonical CMU pronunciation; no guessed pronunciation inserted'
        return record
    raw = choices[0]
    canonical = ['sil', *[re.sub(r'\d','',p).lower() for p in raw], 'sil']
    mapped = [p if p in falcon_utils.phoneme_to_idx_MACRO else falcon_utils.timit_to_leehon(p) for p in canonical]
    if any(p is None for p in mapped):
        raise ValueError(f'Unsupported target phone in {canonical}')
    record.update(dictionary_variants=choices, chosen_dictionary_variant=0, target_sequence=canonical,
                  model_target_sequence=mapped, canonical_stressed_pronunciation=raw)
    audio = item['audio'].unsqueeze(0)
    # No padding/batching: the bidirectional LSTM sees each complete utterance.
    with torch.no_grad():
        outputs = model(audio, None, [canonical], None)
        starts = outputs[5][0].detach().cpu().numpy().astype(float)
        probabilities = torch.softmax(outputs[2][0],dim=-1).detach().cpu().numpy()
    duration = record['duration_seconds']
    bounds = np.r_[starts,duration]
    record['predicted_boundaries_seconds'] = bounds.tolist()
    if len(starts) != len(canonical) or not np.isfinite(bounds).all():
        record['error'] = 'Non-finite boundaries or incorrect interval count'
        return record
    if abs(bounds[0])>1e-6 or np.any(np.diff(bounds)<=0) or bounds[-2]>=duration:
        record['error'] = 'Non-positive/non-monotonic/out-of-audio interval; not silently repaired'
        return record
    interval_rows = []
    for i,(phone,model_phone,start,end) in enumerate(zip(canonical,mapped,bounds[:-1],bounds[1:])):
        frame_s = np.arange(len(probabilities))*config['model_timestamp_samples_per_frame']/16000
        inside = (frame_s>=start)&(frame_s<end)
        posterior = float(probabilities[inside, falcon_utils.phoneme_to_idx_MACRO[model_phone]].mean()) if inside.any() else None
        argmax = (falcon_utils.idx_to_phoneme_MACRO[int(probabilities[inside].mean(axis=0).argmax())]
                  if inside.any() else None)
        interval_rows.append({'phone_index':i,'canonical_phone':phone,'model_phone':model_phone,
                              'start_seconds':float(start),'end_seconds':float(end),
                              'duration_seconds':float(end-start),'is_silence':phone=='sil',
                              'mean_target_posterior':posterior,'mean_posterior_argmax_lh39':argmax,
                              'shorter_than_hubert_stride':bool(end-start<config['phone_short_flag_seconds']),
                              'low_target_posterior':posterior is not None and posterior<config['low_mean_posterior_flag']})
    record.update(status='aligned',intervals=interval_rows,model_frames=len(probabilities),
                  lexical_identity_unresolved=bool('hw74' in row['source_filename'].lower()),
                  short_phone_count=sum(p['shorter_than_hubert_stride'] for p in interval_rows if not p['is_silence']),
                  low_posterior_phone_count=sum(p['low_target_posterior'] for p in interval_rows if not p['is_silence']))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--audio-root',type=Path,required=True,
                        help='Authoritative Nygaard_audio directory; never falls back to repository audio')
    parser.add_argument('--source-manifest',type=Path,
                        default=BASE_OUT/'tables/nygaard_audio_source_manifest.csv')
    args = parser.parse_args()
    audio_root = args.audio_root.resolve(strict=True)
    source_path = args.source_manifest.resolve(strict=True)
    source_sha = digest(source_path)
    config_path = PROJECT/'configs/an19_falcon_alignment.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config_sha = digest(config_path)
    vendor = PROJECT/'artifacts/vendor'
    assert git_revision(vendor/'FALCON') == config['code_revision']
    assert git_revision(vendor/'cmudict') == config['dictionary_revision']
    for child in ['tables','recordings','textgrids']:
        (OUT/child).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest_path = ROOT/'data/manifests/AN19-stimulus-manifest.csv'
    manifest = pd.read_csv(manifest_path,keep_default_na=False)
    assert len(manifest)==6261 and manifest.recording_id.is_unique
    sources = pd.read_csv(source_path,keep_default_na=False)
    assert len(sources)==6261 and sources.recording_id.is_unique
    assert sources.status.eq('available').all()
    assert set(sources.recording_id)==set(manifest.recording_id)
    manifest = manifest.merge(sources[['recording_id','external_audio_relpath','external_filename','mapping_method','sha256']]
                              .rename(columns={'sha256':'expected_audio_sha256'}),
                              on='recording_id',how='left',validate='one_to_one',sort=False)
    override_path = PROJECT/'configs/an19_annotation_transcript_overrides.json'
    overrides = json.loads(override_path.read_text(encoding='utf-8'))['overrides']
    manifest['annotation_word'] = manifest['word']
    for recording_id, override in overrides.items():
        matched = manifest.recording_id.eq(recording_id)
        assert matched.sum() == 1
        assert manifest.loc[matched,'word'].iloc[0] == override['original_word']
        manifest.loc[matched,'annotation_word'] = override['annotation_word']
    dictionary_path = vendor/'cmudict/cmudict.dict'
    lexicon = load_dictionary(dictionary_path)
    missing_words = sorted(set(manifest.annotation_word.str.lower())-set(lexicon))
    print(f'Full batch: {len(manifest)} recordings; {manifest.speaker_id.nunique()} talkers; dictionary gaps={missing_words}',flush=True)
    checkpoint = hf_hub_download(config['weights_repository'],config['weights_filename'],revision=config['weights_revision'])
    assert digest(checkpoint)==config['weights_sha256']
    # Pinned official checkpoint includes argparse.Namespace; never deserialize
    # arbitrary user checkpoints or unknown URLs through this trusted-source path.
    ckpt = torch.load(checkpoint,map_location='cpu',weights_only=False)
    model = NextFrameClassifier(ckpt['hparams'])
    model.load_state_dict({k.replace('NFC.',''):v for k,v in ckpt['state_dict'].items()},strict=True)
    model = model.to(args.device).eval()
    del ckpt
    falcon_utils.tqdm = lambda iterable,*args,**kwargs: iterable
    started=time.time()
    records=[]
    phone_rows=[]
    reused_records=0
    with (OUT/'inference.log').open('a',encoding='utf-8') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        items=pool.map(lambda row:decode_recording(row,audio_root),manifest.to_dict('records'))
        for index,item in enumerate(items,1):
            row=item['row']
            json_path=OUT/'recordings'/f"{row['recording_id']}.json"
            existing=json.loads(json_path.read_text(encoding='utf-8')) if json_path.exists() else None
            if (existing and existing.get('status')=='aligned'
                    and existing.get('config_sha256')==config_sha
                    and existing.get('source_manifest_sha256')==source_sha
                    and existing.get('sha256')==item.get('sha256')
                    and existing.get('annotation_word',existing['word'])==row['annotation_word']):
                result=existing
                result.setdefault('annotation_word',row['annotation_word'])
                reused_records+=1
            else:
                try:
                    with redirect_stdout(log),redirect_stderr(log):
                        result=align_recording(item,model,lexicon,config,config_sha)
                except Exception as exc:
                    result={k:row[k] for k in ['recording_id','speaker_id','word','annotation_word','accent','source_filename']}
                    result.update(status='failed',error=f'{type(exc).__name__}: {exc}',config_sha256=config_sha)
                result['source_manifest_sha256']=source_sha
                result['audio_source']='user-selected July/Nygaard_audio'
                json_path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
            if result['status']=='aligned':
                save_textgrid(OUT/'textgrids'/f"{row['recording_id']}.TextGrid",result)
                for p in result['intervals']:
                    phone_rows.append({k:result[k] for k in ['recording_id','speaker_id','word','annotation_word','accent','source_l1_language']}|p)
            records.append({k:v for k,v in result.items() if not isinstance(v,(list,dict))})
            if index%100==0 or index==len(manifest):
                successful=sum(r['status']=='aligned' for r in records)
                progress={'processed':index,'total':len(manifest),'aligned':successful,'failed':index-successful,'elapsed_seconds':round(time.time()-started,1)}
                (OUT/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                print(json.dumps(progress),flush=True)
    recording_table=pd.DataFrame(records)
    recording_table.to_csv(OUT/'tables/recording_alignment_status.csv',index=False)
    pd.DataFrame(phone_rows).to_csv(OUT/'tables/phone_intervals.csv',index=False)
    recording_table.loc[recording_table.status.ne('aligned')].to_csv(OUT/'tables/failed_recordings.csv',index=False)
    lex_rows=[]
    for word, annotation_word in manifest[['word','annotation_word']].drop_duplicates().sort_values('word').itertuples(index=False,name=None):
        variants=lexicon.get(annotation_word.lower(),[])
        lex_rows.append({'word':word,'annotation_word':annotation_word,'selected_variant':0 if variants else None,
                         'stressed_phones':' '.join(variants[0]) if variants else '',
                         'all_variants':json.dumps(variants)})
    pd.DataFrame(lex_rows).to_csv(OUT/'tables/target_pronunciations.csv',index=False)
    summary={'config':config,'config_sha256':config_sha,'manifest_sha256':digest(manifest_path),
             'runner_sha256':digest(__file__),'transcript_overrides':overrides,
             'audio_source':'user-selected July/Nygaard_audio',
             'source_manifest_sha256':source_sha,'repository_audio_fallback':False,
             'transcript_overrides_sha256':digest(override_path),
             'dictionary_sha256':digest(dictionary_path),'processed_recordings':len(records),
             'aligned_recordings':int(recording_table.status.eq('aligned').sum()),
             'failed_recordings':int(recording_table.status.ne('aligned').sum()),
             'phone_intervals_including_silence':len(phone_rows),
             'speech_phone_intervals':sum(not p['is_silence'] for p in phone_rows),
             'elapsed_seconds_this_invocation':time.time()-started,
             'cached_aligned_records_reused':reused_records,
             'records_inferred_this_invocation':len(records)-reused_records,
             'runtime':{'python':sys.version.split()[0],
             'torch':torch.__version__,'torchaudio':torchaudio.__version__,'numpy':np.__version__,
             'device':torch.cuda.get_device_name() if args.device.startswith('cuda') else args.device,
             'loader_workers':args.workers,'cpu_threads':args.cpu_threads},
             'source_audio_modified':False,'hubert_features_recomputed':False,
             'all_output_automatically_generated':True,'human_verified':False}
    (OUT/'alignment_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f"Complete: {summary['aligned_recordings']}/{len(records)} aligned; {len(phone_rows)} intervals. {OUT}",flush=True)


if __name__=='__main__':
    main()
