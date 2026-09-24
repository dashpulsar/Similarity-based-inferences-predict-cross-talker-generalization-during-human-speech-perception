"""Check completed parameter-grid fits and draw explicitly sampled landscapes."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from ctg.provenance import atomic_write_csv,atomic_write_json,sha256_file


def main(out):
    meta=json.loads((out/'manifest.json').read_text())
    if meta['status']!='complete':raise ValueError('Grid must finish before final plotting')
    assert sha256_file(PROJECT/'R/fit_sbi_landscape.R')==meta['r_source_sha256']
    assert sha256_file(Path(meta.get('run_source_path',PROJECT/'scripts/run_sbi_landscape.py')))==meta['run_source_sha256']
    plot_label=meta.get('plot_label',f"{meta['dataset']} Tr-24")
    for path,digest in meta['sources'].items():
        assert sha256_file(path)==digest, f'Source changed: {path}'
    data=pd.read_csv(out/'landscape.csv')
    grid=data.loc[data.kind.eq('grid')].copy()
    limit=data.loc[data.kind.eq('linear_limit')].copy()
    taus=np.array(meta['tau_values']);ks=np.array(meta['k_values'])
    assert len(grid)==len(taus)*len(ks) and len(limit)==len(taus)
    assert not data.duplicated(['tau','k']).any()
    assert grid.groupby('tau').size().eq(len(ks)).all()
    for tau,g in grid.groupby('tau'):
        np.testing.assert_allclose(sorted(g.k),ks,rtol=1e-12)
    data['eligible']=data.status.eq('ok') & data.singular.eq(False) & np.isfinite(data.z) & np.isfinite(data.log_likelihood)
    grid=data.loc[data.kind.eq('grid')].copy();limit=data.loc[data.kind.eq('linear_limit')].copy()
    good=data.loc[data.status.eq('ok')]
    np.testing.assert_allclose(good.z,good.coefficient/good.std_error,rtol=1e-9,atol=1e-9)
    assert data.n_rows.eq(meta['responses']).all() and data.n_word_responses.eq(meta['responses']).all()
    # Independently check interval-based row identity and numerical scaling.
    dt=pd.read_csv(out/'preparation/dtw_timings.csv')
    reference=None;input_checks=[];distance_scales=[]
    for row in dt.itertuples():
        d=pd.read_csv(out/'preparation'/Path(row.training_file).name)
        assert d.fold.ne(0).all() and len(d)==meta['responses']
        assert not d.condition_id.isin(meta.get('excluded_conditions',[])).any()
        keys=['participant_id','fold','condition_id','behavior_item_id','response_expected','response_correct','response_incorrect']
        if reference is None:reference=d[keys]
        else:pd.testing.assert_frame_equal(reference,d[keys])
        for fit in data.loc[data.tau.eq(row.tau)].itertuples():
            a=-fit.k*(d.raw_distance.to_numpy()-d.raw_distance.min())
            s=-d.raw_distance.to_numpy() if fit.k==0 else np.expm1(a) if np.max(np.abs(a))<.1 else np.exp(a)
            np.testing.assert_allclose([s.mean(),s.std(ddof=1)],[fit.predictor_shifted_mean,fit.predictor_sd],rtol=1e-9,atol=1e-13)
        input_checks.append(dict(tau=row.tau,rows=len(d),test_fold_rows=0))
        distance_scales.append(dict(tau=row.tau,mean_distance=d.raw_distance.mean(),sd_distance=d.raw_distance.std(),
            zero_distance_rows=int(d.raw_distance.eq(0).sum())))
        if row.tau==2.:
            condition_summary=d.assign(zero_distance=d.raw_distance.eq(0)).groupby('condition_id',as_index=False).agg(
                n_rows=('raw_distance','size'),zero_distance_rows=('zero_distance','sum'),mean_distance=('raw_distance','mean'))
            atomic_write_csv(out/'tau2_distance_by_condition.csv',condition_summary)
    winners=[];profiles=[];regions=[]
    for scope in ['expanded','metric_tau_ge_1']:
        eligible=grid.loc[grid.eligible & (grid.tau.ge(1) if scope=='metric_tau_ge_1' else True)]
        for objective,tolerance in [('z',.01),('log_likelihood',.1)]:
            winner=eligible.loc[eligible[objective].idxmax()]
            record=winner.to_dict();record.update(scope=scope,objective=objective,
                tau_at_search_boundary=bool(np.isclose(winner.tau,[eligible.tau.min(),eligible.tau.max()]).any()),
                k_at_search_boundary=bool(np.isclose(winner.k,[ks.min(),ks.max()],rtol=1e-10,atol=1e-15).any()))
            winners.append(record)
            near=eligible.loc[eligible[objective]>=winner[objective]-tolerance]
            regions.append(dict(scope=scope,objective=objective,descriptive_tolerance=tolerance,
                points=len(near),tau_min=near.tau.min(),tau_max=near.tau.max(),k_min=near.k.min(),k_max=near.k.max()))
            if scope=='expanded':
                for tau,g in eligible.groupby('tau'):
                    r=g.loc[g[objective].idxmax()]
                    ref=limit.loc[limit.tau.eq(tau)&limit.eligible]
                    profiles.append(dict(tau=tau,objective=objective,best_k=r.k,best_score=r[objective],
                        linear_limit_score=float(ref.iloc[0][objective]) if len(ref) else np.nan,
                        improvement_over_limit=float(r[objective]-ref.iloc[0][objective]) if len(ref) else np.nan))
    winners=pd.DataFrame(winners);profiles=pd.DataFrame(profiles)
    profiles=profiles.merge(pd.DataFrame(distance_scales),on='tau',validate='many_to_one')
    profiles['best_k_times_mean_distance']=profiles.best_k*profiles.mean_distance
    atomic_write_csv(out/'best_grid_points.csv',winners)
    atomic_write_csv(out/'tau_profiles.csv',profiles)
    atomic_write_csv(out/'near_best_regions.csv',pd.DataFrame(regions))
    atomic_write_csv(out/'flagged_fits.csv',data.loc[~data.eligible])
    atomic_write_json(out/'validation.json',dict(status='passed',grid_fits=len(grid),linear_limit_fits=len(limit),
        eligible=int(data.eligible.sum()),failed=int(data.status.eq('failed').sum()),
        nonconverged=int(data.status.eq('nonconverged').sum()),singular=int(data.singular.eq(True).sum()),
        fixed_training_rows=meta['responses'],held_out_rows_used=0,
        numerical_scaling_checked=True,z_arithmetic_checked=True,grid_coverage_checked=True,source_hashes_checked=True,
        input_checks=input_checks,interpretation='Training surface, conditional on one dataset/layer/split and specified bounds.'))
    figure_dir=out/'figures';figure_dir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(figure_dir/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.25)
        fig.savefig(figure_dir/f'{name}.svg',bbox_inches='tight',pad_inches=.25)
        plt.close(fig)
    xx,yy=np.meshgrid(taus,np.log10(ks))
    label={'z':'Signed training z','log_likelihood':'Training GLMM log-likelihood'}
    surfaces={}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,obj in zip(axes,['z','log_likelihood']):
        table=grid.assign(value=grid[obj].where(grid.eligible)).pivot(index='k',columns='tau',values='value').sort_index().sort_index(axis=1)
        zz=table.to_numpy();surfaces[obj]=zz
        im=ax.pcolormesh(xx,yy,np.ma.masked_invalid(zz),shading='nearest',cmap='viridis',rasterized=True)
        ax.contour(xx,yy,np.ma.masked_invalid(zz),levels=7,colors='white',linewidths=.55,alpha=.6)
        winner=winners.loc[winners.scope.eq('expanded')&winners.objective.eq(obj)].iloc[0]
        ax.plot(winner.tau,np.log10(winner.k),'*',ms=14,color='red',mec='white',mew=.7)
        ax.axvline(1,color='black',ls='--',lw=1)
        ax.set(xlabel='tau',ylabel='log10(k)',title=label[obj])
        fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle(f"{plot_label}: expanded parameter landscape\n29 x 43 evaluated grid; contours interpolate between grid points",fontsize=12)
    save(fig,'parameter_landscape_2d')
    fig=plt.figure(figsize=(13,5.2),layout='constrained')
    for index,obj in enumerate(['z','log_likelihood']):
        ax=fig.add_subplot(1,2,index+1,projection='3d')
        ax.plot_surface(xx,yy,surfaces[obj],cmap='viridis',rstride=1,cstride=1,linewidth=.1,alpha=.92)
        w=winners.loc[winners.scope.eq('expanded')&winners.objective.eq(obj)].iloc[0]
        ax.scatter(w.tau,np.log10(w.k),w[obj],color='red',s=45,depthshade=False)
        ax.set(xlabel='tau',ylabel='log10(k)',zlabel=label[obj],title=label[obj])
        ax.set_box_aspect((1,1,.85),zoom=.80)
        ax.tick_params(labelsize=9)
        ax.zaxis.labelpad=20
        ax.zaxis.label.set_fontsize(10)
        ax.view_init(elev=28,azim=-135)
    fig.suptitle(f"{plot_label} — surfaces connect evaluated grid points",fontsize=12)
    save(fig,'parameter_landscape_3d')
    for obj in ['z','log_likelihood']:
        fig=plt.figure(figsize=(9,6.5),layout='constrained')
        ax=fig.add_subplot(111,projection='3d')
        surface=ax.plot_surface(xx,yy,surfaces[obj],cmap='viridis',rstride=1,cstride=1,linewidth=.1,alpha=.95)
        w=winners.loc[winners.scope.eq('expanded')&winners.objective.eq(obj)].iloc[0]
        ax.scatter(w.tau,np.log10(w.k),w[obj],color='red',s=60,depthshade=False)
        ax.set(xlabel='tau',ylabel='log10(k)',zlabel=label[obj],title=f"{plot_label}: {label[obj]}")
        ax.set_box_aspect((1,1,.85),zoom=.85)
        ax.zaxis.labelpad=20
        ax.view_init(elev=28,azim=-135)
        save(fig,f'landscape_3d_{obj}')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,obj in enumerate(['z','log_likelihood']):
        for tau in [.5,1.,2.,4.]:
            d=grid.loc[grid.tau.eq(tau)].sort_values('k')
            axes[0,col].semilogx(d.k,d[obj].where(d.eligible),label=f'tau={tau:g}')
        axes[0,col].set(xlabel='k (log scale)',ylabel=label[obj],title='Profiles at selected tau values')
        d=profiles.loc[profiles.objective.eq(obj)]
        axes[1,col].plot(d.tau,d.best_score,label='Best evaluated k',color='black')
        axes[1,col].plot(d.tau,d.linear_limit_score,label='Small-k standardized limit',color='gray',ls='--')
        axes[1,col].axvline(1,color='.5',lw=.8,ls=':')
        axes[1,col].set(xlabel='tau',ylabel=label[obj],title='Profile optimum and linear-limit reference')
        axes[0,col].legend(frameon=False,fontsize=9);axes[1,col].legend(frameon=False,fontsize=9)
        for ax in axes[:,col]:ax.spines[['top','right']].set_visible(False)
    save(fig,'parameter_profiles')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for obj in ['z','log_likelihood']:
        d=profiles.loc[profiles.objective.eq(obj)]
        axes[0].semilogy(d.tau,d.best_k,marker='o',ms=3,label=label[obj])
        axes[1].plot(d.tau,d.best_k_times_mean_distance,marker='o',ms=3,label=label[obj])
    axes[0].set(xlabel='tau',ylabel='Best evaluated k (log scale)',title='Ridge of preferred parameter pairs')
    axes[1].set(xlabel='tau',ylabel='Best k x mean training distance',title='Distance-scale compensation diagnostic')
    for ax in axes:ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
    save(fig,'parameter_ridge')
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    ok=grid.loc[grid.eligible]
    scatter=ax.scatter(ok.z,ok.log_likelihood,c=ok.tau,cmap='viridis',s=9,alpha=.7)
    ax.set(xlabel='Signed training z',ylabel='Training GLMM log-likelihood',title='Both objectives from the same candidate fits')
    fig.colorbar(scatter,ax=ax,label='tau')
    save(fig,'objective_agreement')
    print(winners[['scope','objective','tau','k','z','log_likelihood','tau_at_search_boundary','k_at_search_boundary']].to_string(index=False))
    print(pd.DataFrame(regions).to_string(index=False))
    print('Elapsed:',meta['elapsed_seconds'],'seconds; eligible',data.eligible.sum(),'/',len(data))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    main(p.parse_args().input)
