"""Expanded AN19 training-only tau/k grid, with bounded parallel R processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from probe_sbi_joint_optimization import Probe,PROJECT
from ctg.provenance import atomic_write_csv,atomic_write_json,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rscript',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--budget-seconds',type=float,default=1800)
    args=p.parse_args()
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True)
    start=time.perf_counter();deadline=start+args.budget_seconds
    taus=np.round(np.arange(.5,4.0001,.125),6)
    ks=np.sort(np.unique(np.r_[np.geomspace(1e-6,2.,41),.001,.05]))
    atomic_write_csv(out/'k_grid.csv',pd.DataFrame({'k':np.r_[ks,0.]}))
    probe=Probe(out/'preparation',args.rscript,args.budget_seconds,4)
    tasks=[]
    for index,tau in enumerate(taus):
        path,_=probe.distances(float(tau))
        tasks.append((index,float(tau),path))
    manifest=dict(probe.manifest)
    manifest.update(scope='Expanded AN19 base Tr-24 training landscape; fold 0 excluded',
        tau_bounds=[.5,4.],k_bounds=[1e-6,2.],tau_values=taus.tolist(),k_values=ks.tolist(),
        grid_fits=int(len(taus)*len(ks)),linear_limit_fits=len(taus),total_fits=int(len(taus)*(len(ks)+1)),
        status='running',jobs=args.jobs,budget_seconds=args.budget_seconds,
        representation_of_similarity='standardized expm1(-k*d), affine-equivalent to standardized exp(-k*d)',
        linear_limit='standardized negative distance; not literal constant exp(0)',
        tau_below_one='non-metric sensitivity: triangle inequality does not generally hold',
        preparation_manifest='preparation/manifest.json',
        run_source_sha256=sha256_file(Path(__file__)),r_source_sha256=sha256_file(PROJECT/'R/fit_sbi_landscape.R'))
    atomic_write_json(out/'manifest.json',manifest)
    def fit(task):
        idx,tau,path=task
        destination=out/f'tau_{idx:03d}_fits.csv'
        remain=deadline-time.perf_counter()
        if remain<=0:return dict(tau=tau,status='not_started_budget',file=destination.name)
        begin=time.perf_counter()
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        try:
            r=subprocess.run([str(args.rscript),str(PROJECT/'R/fit_sbi_landscape.R'),str(path),str(out/'k_grid.csv'),str(destination)],
                capture_output=True,text=True,env=env,timeout=min(600,remain))
            (out/f'tau_{idx:03d}.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
            status='complete' if r.returncode==0 else 'process_failed'
        except subprocess.TimeoutExpired as error:
            status='timeout'
            (out/f'tau_{idx:03d}.log').write_text(str(error),encoding='utf-8')
        return dict(tau=tau,status=status,file=destination.name,wall_seconds=time.perf_counter()-begin)
    completed=[]
    with ThreadPoolExecutor(max_workers=max(1,args.jobs)) as pool:
        futures=[pool.submit(fit,t) for t in tasks]
        for future in as_completed(futures):
            result=future.result();completed.append(result)
            atomic_write_csv(out/'batch_status.csv',pd.DataFrame(completed))
            print(f"{len(completed)}/{len(tasks)} tau batches: {result}",flush=True)
    tables=[]
    for batch in completed:
        path=out/batch['file']
        if path.exists():
            frame=pd.read_csv(path);frame['tau']=batch['tau'];tables.append(frame)
    values=pd.concat(tables,ignore_index=True).sort_values(['tau','k']) if tables else pd.DataFrame()
    atomic_write_csv(out/'landscape.csv',values)
    complete=len(values)==manifest['total_fits'] and all(x['status']=='complete' for x in completed)
    manifest.update(status='complete' if complete else 'partial',rows_saved=len(values),
        elapsed_seconds=time.perf_counter()-start,successful_fits=int(values.status.eq('ok').sum()) if len(values) else 0)
    atomic_write_json(out/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','rows_saved','successful_fits','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
