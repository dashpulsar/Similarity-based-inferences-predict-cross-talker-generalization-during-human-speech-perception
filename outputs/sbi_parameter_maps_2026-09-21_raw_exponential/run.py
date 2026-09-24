"""Re-evaluate every original positive-k grid point, with raw exp(-k*d)."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import subprocess, os, time, json, hashlib, random, csv
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'cross_talker_generalization/analysis_update_2026-09-18/x21_parameter_landscape'
R=Path('C:/Program Files/R/R-4.4.1/bin/x64/Rscript.exe')
JOBS=16; POINT_SECONDS=90; BUDGET_SECONDS=1800

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic_text(path,text):
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(text,encoding='utf-8'); temp.replace(path)
def dump(path,value): atomic_text(path,json.dumps(value,indent=2))

def main():
    if (HERE/'manifest.json').exists(): raise FileExistsError('Preserve this run; use a new directory to repeat.')
    points=HERE/'points'; points.mkdir()
    source=json.loads((SOURCE/'manifest.json').read_text())
    assert source['status']=='complete' and source['grid_fits']==1247
    tasks=[(ti,ki,float(tau),float(k)) for ti,tau in enumerate(source['tau_values']) for ki,k in enumerate(source['k_values'])]
    assert len(tasks)==1247 and all(t[3]>0 for t in tasks)
    # Fixed order independent of scores, covering the domain even if a time limit is reached.
    random.Random(20260921).shuffle(tasks)
    inputs={ti:SOURCE/'preparation'/f'tau_{ti:04d}_training.csv' for ti in range(len(source['tau_values']))}
    manifest=dict(status='running',scope='X21 all conditions, HuBERT-base Tr-24, 3-D t-SNE, training folds 1+2',
      started_utc=datetime.now(timezone.utc).isoformat(),tau_values=source['tau_values'],k_values=source['k_values'],
      grid_fits=1247,jobs=JOBS,per_point_timeout_seconds=POINT_SECONDS,total_budget_seconds=BUDGET_SECONDS,
      transformation='similarity_raw = exp(-k * raw_distance); no exponential shift, expm1, or z-score',
      distance_normalization='mean_sequence_length, unchanged; existing distances reused',
      model='cbind(correct,incorrect) ~ similarity_raw + test_talker_id + (1|participant_id) + (1|analysis_item_id)',
      glmm=dict(optimizer='bobyqa',maxfun=20000,nAGQ=1,calc_derivs=True,family='binomial-logit'),
      random_effect_structure_fallback=False,outer_test_used=False,k_zero='Excluded: original HTML also displays positive-k grid only, not the separately fitted standardized linear limits',
      data_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs.values()},
      scripts_sha256={p.name:sha(p) for p in [HERE/'fit_raw_point.R',Path(__file__)]},
      reference_manifest_sha256=sha(SOURCE/'manifest.json'),task_order_seed=20260921)
    dump(HERE/'manifest.json',manifest)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic(); deadline=start+BUDGET_SECONDS
    def fit(t):
        ti,ki,tau,k=t; stem=f't{ti:02d}_k{ki:02d}'
        dest=points/(stem+'.csv'); log=points/(stem+'.log')
        row=dict(tau_index=ti,k_index=ki,tau=tau,k=k,status='not_started_budget',z=None,log_likelihood=None)
        remaining=deadline-time.monotonic()
        if remaining<=0:return row
        begin=time.monotonic()
        with log.open('w',encoding='utf-8') as stream:
            command=[str(R),str(HERE/'fit_raw_point.R'),str(inputs[ti]),format(k,'.17g'),str(dest)]
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:
                code=proc.wait(timeout=min(POINT_SECONDS,remaining)); process_status='returned' if code==0 else 'process_failed'
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(); process_status='timeout'
                stream.write('\nWALL-TIME LIMIT: no valid completed fit was available.\n')
        if dest.exists():
            try:
                with dest.open(newline='',encoding='utf-8') as f: saved=next(csv.DictReader(f))
                row.update(saved)
            except (StopIteration,csv.Error): row['warnings']='Incomplete per-point CSV'
        if process_status!='returned' or row['status']=='running':
            row.update(status=process_status if process_status!='returned' else 'incomplete_result',
                       z=None,log_likelihood=None,coefficient=None,std_error=None)
        row.update(tau_index=ti,k_index=ki,tau=tau,k=k,wall_seconds=time.monotonic()-begin)
        return row
    rows=[]; last=0
    def checkpoint(final=False):
        frame=pd.DataFrame(rows).sort_values(['tau_index','k_index'])
        temp=HERE/'landscape.csv.tmp'; frame.to_csv(temp,index=False); temp.replace(HERE/'landscape.csv')
        progress=dict(total=1247,recorded=len(rows),elapsed_seconds=time.monotonic()-start,
                      status_counts=frame.status.value_counts().to_dict())
        dump(HERE/'progress.json',progress)
        print(json.dumps(progress),flush=True)
        if final:
            manifest.update(progress,status='all_points_processed' if not frame.status.eq('not_started_budget').any() else 'budget_limited',
                            finished_utc=datetime.now(timezone.utc).isoformat())
            dump(HERE/'manifest.json',manifest)
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        for f in as_completed([pool.submit(fit,t) for t in tasks]):
            rows.append(f.result())
            if time.monotonic()-last>25 or len(rows)==1247:
                checkpoint(); last=time.monotonic()
    checkpoint(final=True)
    subprocess.run([os.sys.executable,str(HERE/'plot.py')],check=True,env=env)

if __name__=='__main__': main()
