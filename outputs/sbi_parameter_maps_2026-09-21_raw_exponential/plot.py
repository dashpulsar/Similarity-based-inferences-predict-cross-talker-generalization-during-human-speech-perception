"""Plot actual reported raw-exponential scores, including flagged fits, without imputation."""
from pathlib import Path
import json, html, subprocess, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
SOURCE=ROOT/'cross_talker_generalization/analysis/sbi/parameter_maps/x21_all_conditions'

def main():
    meta=json.loads((HERE/'manifest.json').read_text())
    d=pd.read_csv(HERE/'landscape.csv')
    for col in ['z','log_likelihood','coefficient','std_error','predictor_sd','wall_seconds']:
        if col not in d: d[col]=np.nan
        d[col]=pd.to_numeric(d[col],errors='coerce')
    if 'warnings' not in d: d['warnings']=''
    if 'singular' not in d: d['singular']=False
    d['warnings']=d.warnings.fillna('')
    singular=d.singular.fillna(False).astype(str).str.lower().eq('true')
    d['flagged']=d.status.ne('ok') | singular | d.warnings.ne('')
    taus=np.array(meta['tau_values']); ks=np.array(meta['k_values'])
    xx,kk=np.meshgrid(taus,ks); yy=np.log10(kk)
    fig=make_subplots(rows=1,cols=2,specs=[[{'type':'surface'},{'type':'surface'}]],horizontal_spacing=.12,
                      subplot_titles=['Raw exponential: signed training Wald z','Raw exponential: fitted GLMM log-likelihood'])
    for col,score in enumerate(['z','log_likelihood'],1):
        surf=d.pivot(index='k_index',columns='tau_index',values=score).reindex(index=range(len(ks)),columns=range(len(taus))).to_numpy()
        fig.add_trace(go.Surface(x=xx,y=yy,z=surf,colorscale='Viridis',connectgaps=False,showscale=False,
                      hoverinfo='skip',name=score,showlegend=False),row=1,col=col)
        for flagged,label,color,symbol in [(False,'No recorded warning','#333333','circle'),(True,'Warning / nonconvergence / singular','#c5342d','x')]:
            q=d[np.isfinite(d[score]) & d.flagged.eq(flagged)].copy()
            custom=q[['tau','k','z','log_likelihood','status','warnings','predictor_sd']].to_numpy()
            fig.add_trace(go.Scatter3d(x=q.tau,y=np.log10(q.k),z=q[score],customdata=custom,
                 mode='markers',name=label,legendgroup=label,showlegend=(col==1),
                 marker=dict(size=3 if flagged else 2,color=color,symbol=symbol,opacity=.8 if flagged else .4),
                 hovertemplate='tau: %{customdata[0]:.3f}<br>k: %{customdata[1]:.8g}<br>z: %{customdata[2]:.7g}<br>logLik: %{customdata[3]:.7g}<br>status: %{customdata[4]}<br>predictor SD: %{customdata[6]:.5g}<br>%{customdata[5]}<extra></extra>'),row=1,col=col)
    camera=dict(eye=dict(x=-1.45,y=-1.7,z=1.1))
    for scene,ztitle in [('scene','Signed training Wald z'),('scene2','Fitted log-likelihood')]:
        fig.update_layout(**{scene:dict(xaxis=dict(title='tau'),yaxis=dict(title='log10(k)'),zaxis=dict(title=ztitle),
                        camera=camera,dragmode='orbit',aspectmode='manual',aspectratio=dict(x=1,y=1,z=.85))})
    # Presentation toggle only; reported scores and hover values always remain unchanged.
    linear=[fig.data[i].z for i in [0,1,2]]
    signedlog=[np.sign(np.asarray(v,dtype=float))*np.log10(1+np.abs(np.asarray(v,dtype=float))) for v in linear]
    fig.update_layout(template='plotly_white',height=700,margin=dict(l=10,r=10,t=110,b=45),
        title=dict(text='X21 Tr-24: no exponential stabilization or z-scoring',x=.5),
        font=dict(family='Arial',size=13),legend=dict(orientation='h',x=.5,xanchor='center',y=-.02),
        updatemenus=[dict(type='buttons',direction='right',x=0,y=1.16,buttons=[
            dict(label='Linear z axis',method='update',args=[{'z':linear},{'scene.zaxis.title.text':'Signed training Wald z'},[0,1,2]]),
            dict(label='Signed-log z display',method='update',args=[{'z':signedlog},{'scene.zaxis.title.text':'sign(z) log10(1 + |z|)'},[0,1,2]])])])
    sync="""
    const c=document.getElementById('raw-landscape');let sync=false;
    c.on('plotly_relayout',e=>{if(sync)return;const a=e['scene.camera']?'scene':e['scene2.camera']?'scene2':null;if(!a)return;
    sync=true;Plotly.relayout(c,{[(a==='scene'?'scene2':'scene')+'.camera']:e[a+'.camera']}).finally(()=>sync=false);});
    """
    plot_html=fig.to_html(full_html=False,include_plotlyjs=True,div_id='raw-landscape',post_script=sync,
                        config=dict(responsive=True,scrollZoom=True,displaylogo=False))
    finite=np.isfinite(d.z) & np.isfinite(d.log_likelihood)
    clean=finite & ~d.flagged
    summary=dict(run_status=meta['status'],expected_points=1247,recorded_points=len(d),
        status_counts=d.status.value_counts().to_dict(),finite_score_pairs=int(finite.sum()),
        clean_finite_pairs=int(clean.sum()),finite_flagged_pairs=int((finite & d.flagged).sum()),
        fits_with_warnings=int(d.warnings.ne('').sum()),max_abs_z=float(d.z.abs().max()) if d.z.notna().any() else None,
        elapsed_seconds=meta.get('elapsed_seconds'),per_point_timeout_seconds=meta['per_point_timeout_seconds'])
    ref=pd.read_csv(SOURCE/'landscape.csv'); ref=ref[ref.kind.eq('grid')]
    # Decimal serialization can differ by the last bit: pair through original grid indexes instead.
    ref['tau_index']=ref.tau.apply(lambda t:int(np.argmin(abs(taus-t))))
    ref['k_index']=ref.k.apply(lambda k:int(np.argmin(abs(ks-k))))
    matched=d.merge(ref[['tau_index','k_index','z','log_likelihood']],on=['tau_index','k_index'],suffixes=('','_stable'),validate='one_to_one')
    matched['z_difference']=matched.z-matched.z_stable
    matched['log_likelihood_difference']=matched.log_likelihood-matched.log_likelihood_stable
    matched.to_csv(HERE/'comparison_with_stable.csv',index=False)
    summary['maximum_absolute_z_difference']=float(matched.z_difference.abs().max()) if matched.z.notna().any() else None
    summary['reported_abs_z_above_100']=int(d.z.abs().gt(100).sum())
    # Matrix of statuses has no interpolated cells; missing cells remain explicitly identified.
    d['category']=np.where(~finite,2,np.where(d.flagged,1,0))
    status=d.pivot(index='k_index',columns='tau_index',values='category').reindex(index=range(len(ks)),columns=range(len(taus))).fillna(3).to_numpy()
    figstat=go.Figure(go.Heatmap(x=taus,y=np.log10(ks),z=status,zmin=0,zmax=3,
        colorscale=[[0,'#43956a'],[.1666,'#43956a'],[.1667,'#d77047'],[.5,'#d77047'],[.5001,'#777777'],[.8333,'#777777'],[.8334,'#dddddd'],[1,'#dddddd']],
        colorbar=dict(tickvals=[0,1,2,3],ticktext=['No recorded warning','Flagged finite result','No finite score pair','Not recorded yet']),
        hovertemplate='tau: %{x}<br>log10(k): %{y}<br>status category: %{z}<extra></extra>'))
    figstat.update_layout(title='Grid coverage and fit warnings',template='plotly_white',height=480,
                          xaxis_title='tau',yaxis_title='log10(k)',margin=dict(l=65,r=220,t=65,b=60))
    status_html=figstat.to_html(full_html=False,include_plotlyjs=False,div_id='raw-status',config=dict(responsive=True,displaylogo=False))
    details=html.escape(json.dumps(summary,indent=2))
    note=f"{len(d):,} / 1,247 grid points recorded; {int(finite.sum()):,} have two finite scores; {int((finite & d.flagged).sum()):,} of those are flagged. Run status: {meta['status']}."
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><title>X21 raw exponential parameter landscape</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:15px/1.5 Arial,sans-serif;margin:20px;color:#172b40}p{max-width:1200px}code,pre{background:#f1f4f7;padding:4px}pre{white-space:pre-wrap}h1{font-size:22px}</style></head><body>'
    page+='<h1>X21 parameter map: direct exp(-k * distance)</h1><p>'+note+'</p>'
    page+='<p>The positive-k grid, cached DTW distances, training responses and GLMM settings match the original experiment. Only exponential stabilization and z-scoring were removed. DTW mean-sequence-length normalization is unchanged. All finite reported scores are displayed, including warnings and nonconverged fits; these marked peaks are not valid evidence of better prediction. Gaps are not filled. Surface facets only connect evaluated points.</p>'
    page+='<p>Drag to rotate; hover over a point for the exact score and warnings. The default z axis is linear. The optional signed-log display changes only the display scale, not the predictor or fit. Tau below one is the original non-metric sensitivity range.</p>'+plot_html+status_html
    page+='<details><summary>Run summary</summary><pre>'+details+'</pre></details></body></html>'
    (HERE/'x21_raw_exponential.html').write_text(page,encoding='utf-8')
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Static scientific preview, showing the same finite values on linear score axes.
    f=plt.figure(figsize=(14,6))
    for i,score in enumerate(['z','log_likelihood'],1):
        ax=f.add_subplot(1,2,i,projection='3d')
        zz=d.pivot(index='k_index',columns='tau_index',values=score).reindex(index=range(len(ks)),columns=range(len(taus))).to_numpy()
        ax.plot_surface(xx,yy,np.ma.masked_invalid(zz),cmap='viridis',rstride=1,cstride=1,alpha=.85,linewidth=0)
        q=d[np.isfinite(d[score]) & d.flagged]
        ax.scatter(q.tau,np.log10(q.k),q[score],s=6,color='#be352f',alpha=.65)
        ax.set_xlabel('tau');ax.set_ylabel('log10(k)');ax.set_zlabel('Wald z' if score=='z' else 'GLMM log-likelihood')
        ax.set_title('Signed training z' if score=='z' else 'Fitted training log-likelihood')
        ax.view_init(elev=25,azim=-125)
    f.suptitle('X21 Tr-24: raw exponential, no standardization',fontsize=16)
    f.text(.5,.025,'Red points carry warnings / convergence flags. Missing fits are not imputed. Training only.',ha='center')
    f.tight_layout(rect=[0,.08,1,.93]); f.savefig(HERE/'raw_landscape_preview.png',dpi=170);plt.close(f)
    print(json.dumps(summary,indent=2),flush=True)
    if meta['status']!='running':
        subprocess.run([sys.executable,str(HERE/'verify.py')],check=True)
        node=Path('C:/Users/Alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
        subprocess.run([str(node),str(HERE/'check_html.cjs')],check=True)

if __name__=='__main__': main()
