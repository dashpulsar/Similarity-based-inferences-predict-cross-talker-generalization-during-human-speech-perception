"""Refit the saved X21 tau/k grid after excluding Talker-specific responses."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from ctg.provenance import atomic_write_csv, atomic_write_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rscript', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--budget-seconds', type=float, default=1800)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a fresh output directory')
    meta = json.loads((source / 'manifest.json').read_text())
    assert meta['status'] == 'complete' and meta['dataset'] == 'X21'
    assert json.loads((source / 'validation.json').read_text())['status'] == 'passed'
    out.mkdir(parents=True)
    (out / 'preparation').mkdir()
    start = time.perf_counter()
    deadline = start + args.budget_seconds
    excluded = 'X21.Talker_specific'
    timing = pd.read_csv(source / 'preparation/dtw_timings.csv')
    tasks, records, hashes = [], [], {}
    reference = None
    for i, row in enumerate(timing.itertuples()):
        old_path = source / 'preparation' / Path(row.training_file).name
        original = pd.read_csv(old_path)
        frame = original.loc[original.condition_id.ne(excluded)].copy().reset_index(drop=True)
        # This column belongs to an old k and must not be presented as a current similarity.
        frame = frame.drop(columns=['similarity_exp_k'], errors='ignore')
        assert frame.fold.ne(0).all() and frame.raw_distance.gt(0).all()
        assert set(frame.condition_id) == {'X21.Control', 'X21.Multi_talker', 'X21.Single_talker'}
        keys = ['participant_id', 'fold', 'condition_id', 'behavior_item_id', 'response_expected']
        if reference is None:
            reference = frame[keys]
            counts = frame.groupby('condition_id', as_index=False).agg(
                responses=('participant_id', 'size'), participants=('participant_id', 'nunique'))
            atomic_write_csv(out / 'retained_conditions.csv', counts)
        else:
            pd.testing.assert_frame_equal(reference, frame[keys])
        dest = out / 'preparation' / f'tau_{i:04d}_training.csv'
        atomic_write_csv(dest, frame)
        hashes[str(old_path)] = sha256_file(old_path)
        hashes[str(dest)] = sha256_file(dest)
        tasks.append((i, row.tau, dest))
        records.append(dict(tau=row.tau, seconds=0, training_file=dest.name,
                            distance_origin='unchanged cached distances; filtered responses'))
    atomic_write_csv(out / 'preparation/dtw_timings.csv', pd.DataFrame(records))
    atomic_write_csv(out / 'k_grid.csv', pd.read_csv(source / 'k_grid.csv'))
    hashes[str(source / 'manifest.json')] = sha256_file(source / 'manifest.json')
    r_source = PROJECT / 'R/fit_sbi_landscape.R'
    meta.update(status='running', source_landscape=str(source), excluded_conditions=[excluded],
                plot_label='X21 Tr-24 (Talker-specific excluded)',
                scope='X21 Tr-24; Control, Single-talker, Multi-talker; training folds 1+2 only',
                responses=len(frame), participants=int(frame.participant_id.nunique()),
                excluded_responses=int(original.condition_id.eq(excluded).sum()),
                jobs=args.jobs, budget_seconds=args.budget_seconds, sources=hashes,
                run_source_path=str(Path(__file__).resolve()),
                run_source_sha256=sha256_file(Path(__file__)), r_source_sha256=sha256_file(r_source),
                training_scaling='Recomputed separately for each candidate using retained training responses only',
                distance_recomputed=False, stale_similarity_column_removed=True)
    for key in ['elapsed_seconds', 'rows_saved', 'successful_fits', 'preparation_manifest', 'workers']:
        meta.pop(key, None)
    atomic_write_json(out / 'manifest.json', meta)
    print(f'Retained {len(frame)} responses / {meta["participants"]} participants; '
          f'excluded {meta["excluded_responses"]}; {args.jobs} R workers', flush=True)
    def fit(task):
        idx, tau, path = task
        dest = out / f'tau_{idx:03d}_fits.csv'
        remain = deadline - time.perf_counter()
        if remain <= 0:
            return dict(tau=tau, status='not_started_budget', file=dest.name)
        t0 = time.perf_counter()
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            result = subprocess.run([str(args.rscript), str(r_source), str(path), str(out/'k_grid.csv'), str(dest)],
                                    capture_output=True, text=True, env=env, timeout=min(900, remain))
            (out/f'tau_{idx:03d}.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
            status = 'complete' if result.returncode == 0 else 'process_failed'
        except subprocess.TimeoutExpired as error:
            status = 'timeout'
            (out/f'tau_{idx:03d}.log').write_text(str(error), encoding='utf-8')
        return dict(tau=tau, status=status, file=dest.name, wall_seconds=time.perf_counter()-t0)
    completed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(fit, task) for task in tasks]):
            completed.append(future.result())
            atomic_write_csv(out/'batch_status.csv', pd.DataFrame(completed))
            print(f'{len(completed)}/{len(tasks)} tau batches: {completed[-1]}', flush=True)
    tables = []
    for task in completed:
        path = out/task['file']
        if path.exists():
            table = pd.read_csv(path)
            table['tau'] = task['tau']
            tables.append(table)
    values = pd.concat(tables, ignore_index=True).sort_values(['tau', 'k']) if tables else pd.DataFrame()
    atomic_write_csv(out/'landscape.csv', values)
    complete = len(values) == meta['total_fits'] and all(t['status']=='complete' for t in completed)
    meta.update(status='complete' if complete else 'partial', rows_saved=len(values),
                successful_fits=int(values.status.eq('ok').sum()) if len(values) else 0,
                elapsed_seconds=time.perf_counter()-start)
    atomic_write_json(out/'manifest.json', meta)
    print(json.dumps({key:meta[key] for key in ['status','rows_saved','successful_fits','elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
