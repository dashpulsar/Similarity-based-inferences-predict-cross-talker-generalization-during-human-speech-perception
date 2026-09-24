"""Validate and plot a completed, tiny SBI optimizer feasibility run."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from ctg.provenance import atomic_write_csv,atomic_write_json,sha256_file


def summarize(out):
    meta=json.loads((out/'manifest.json').read_text())
    if meta['status']!='tiny_benchmark_complete':
        raise ValueError('Only a completed tiny benchmark can be summarized as complete')
    values=pd.read_csv(out/'evaluations.csv')
    trace=pd.read_csv(out/'search_trace.csv')
    timing=pd.read_csv(out/'dtw_timings.csv')
    assert not values.evaluation.duplicated().any()
    assert len(trace)==72
    assert trace.groupby(['method','objective']).size().eq(12).all()
    assert trace.seed.eq(42).all()
    valid=values.loc[values.status.eq('ok')]
    np.testing.assert_allclose(valid.z,valid.coefficient/valid.std_error,rtol=1e-10,atol=1e-10)
    assert valid.n_rows.eq(meta['responses']).all()
    assert valid.participants.eq(meta['participants']).all()
    base_rows=None
    sources=[]
    for row in timing.itertuples():
        path=out/row.training_file
        d=pd.read_csv(path)
        assert d.fold.ne(0).all() and len(d)==meta['responses']
        cols=['participant_id','fold','condition_id','behavior_item_id','response_expected','response_correct','response_incorrect']
        if base_rows is None: base_rows=d[cols]
        else: pd.testing.assert_frame_equal(base_rows,d[cols])
        for fit in valid.loc[valid.tau.eq(row.tau)].itertuples():
            s=np.exp(-fit.k*d.raw_distance.to_numpy())
            np.testing.assert_allclose([s.mean(),s.std(ddof=1)],[fit.similarity_mean,fit.similarity_sd],rtol=1e-9,atol=1e-15)
        sources.append(dict(path=path.name,sha256=sha256_file(path)))
    summary=[]
    for (method,objective),group in trace.groupby(['method','objective']):
        selected=group.loc[group.valid].sort_values(['score','trial'],ascending=[False,True]).iloc[0]
        fit=values.loc[values.evaluation.eq(selected.evaluation)].iloc[0]
        best=np.maximum.accumulate(group.score.fillna(-np.inf).to_numpy())
        np.testing.assert_allclose(best,group.best_score)
        for r in group.itertuples():
            value=values.loc[values.evaluation.eq(r.evaluation)].iloc[0]
            if r.valid: np.testing.assert_allclose(r.score,value[objective])
        summary.append(dict(method=method,objective=objective,seed=42,proposals=len(group),
            new_fits=int((~group.full_evaluation_cached).sum()),best_tau=selected.tau,best_k=selected.k,
            signed_z=fit.z,log_likelihood=fit.log_likelihood,elapsed_seconds=group.elapsed_seconds.max(),
            singular_fits=int(values.loc[values.evaluation.isin(group.evaluation),'singular'].sum()),
            failed_proposals=int((~group.valid).sum()),
            boundary_tau=bool(np.isclose(selected.tau,meta['tau_bounds']).any()),
            boundary_k=bool(np.isclose(selected.k,meta['k_bounds'],rtol=1e-8,atol=1e-12).any())))
    result=pd.DataFrame(summary)
    atomic_write_csv(out/'method_summary.csv',result)
    atomic_write_json(out/'validation.json',dict(status='passed',unique_fits=len(values),
        training_responses=meta['responses'],test_fold_rows_used=0,
        successful_fits=len(valid),failed_fits=len(values)-len(valid),
        singular_fits=int(valid.singular.sum()),same_response_rows_across_tau=True,
        coefficient_standard_error_z_consistency=True,exponential_similarity_scaling_consistency=True,
        search_score_and_running_best_consistency=True,training_files=sources,
        caveat='One dataset/layer/fold/seed, twelve proposals. No general optimizer ranking or independent prediction evaluation.'))
    colors={'grid':'#222222','tpe':'#377eb8','differential_evolution':'#d95f02'}
    labels={'grid':'Grid (3 x 4)','tpe':'Optuna-TPE','differential_evolution':'Differential evolution'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,objective,ylabel in zip(axes,['z','log_likelihood'],['Best signed training z','Best training GLMM log-likelihood']):
        for method in colors:
            d=trace.loc[trace.method.eq(method)&trace.objective.eq(objective)]
            ax.plot(d.trial,d.best_score,marker='o',ms=3,color=colors[method],label=labels[method])
        ax.set(xlabel='Candidate evaluations',ylabel=ylabel,xticks=[1,4,8,12])
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('AN19 Tr-24: preliminary joint tau/k search (one training split, one seed)',fontsize=11)
    fig.savefig(out/'optimizer_progress.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    probe=values.iloc[:9]
    for ax,objective,title in zip(axes,['z','log_likelihood'],['Signed training z','Training GLMM log-likelihood']):
        table=probe.pivot(index='tau',columns='k',values=objective).sort_index().sort_index(axis=1)
        im=ax.imshow(table.to_numpy(),aspect='auto',origin='lower',cmap='viridis')
        ax.set(xticks=range(3),xticklabels=[f'{v:g}' for v in table.columns],
            yticks=range(3),yticklabels=[f'{v:g}' for v in table.index],xlabel='k',ylabel='tau',title=title)
        for i in range(3):
            for j in range(3): ax.text(j,i,f'{table.iloc[i,j]:.2f}',ha='center',va='center',color='white',fontsize=10,
                bbox=dict(facecolor='black',alpha=.3,edgecolor='none',pad=2))
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Nine evaluated parameter pairs (no interpolation)',fontsize=11)
    fig.savefig(out/'parameter_probe.png',dpi=180)
    plt.close(fig)
    print(result.to_string(index=False))
    print('Validation passed; no held-out evaluation performed.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    summarize(p.parse_args().input)
