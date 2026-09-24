"""Export the validated X21 grid as an offline, rotatable Plotly figure."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main(out):
    meta = json.loads((out / 'manifest.json').read_text())
    validation = json.loads((out / 'validation.json').read_text())
    assert meta['status'] == 'complete' and validation['status'] == 'passed'
    data = pd.read_csv(out / 'landscape.csv')
    data = data.loc[data.kind.eq('grid')].copy()
    assert len(data) == meta['grid_fits']
    eligible = (data.status.eq('ok') & data.singular.eq(False)
                & np.isfinite(data.z) & np.isfinite(data.log_likelihood))
    taus = np.array(meta['tau_values'])
    ks = np.array(meta['k_values'])
    xx, kk = np.meshgrid(taus, ks)
    yy = np.log10(kk)
    surfaces = {}
    for score in ['z', 'log_likelihood']:
        surfaces[score] = data.assign(value=data[score].where(eligible)).pivot(
            index='k', columns='tau', values='value').sort_index().sort_index(axis=1).to_numpy()
    assert surfaces['z'].shape == xx.shape == (len(ks), len(taus))
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                        horizontal_spacing=.12,
                        subplot_titles=['Signed training z', 'Training GLMM log-likelihood'])
    hover = ('tau: %{customdata[0]:.3f}<br>k: %{customdata[1]:.7g}'
             '<br>Signed z: %{customdata[2]:.6f}'
             '<br>Log-likelihood: %{customdata[3]:.6f}<extra></extra>')
    for col, score in enumerate(['z', 'log_likelihood'], 1):
        fig.add_trace(go.Surface(x=xx, y=yy, z=surfaces[score], hoverinfo='skip',
                                colorscale='Viridis', showscale=False,
                                connectgaps=False, name=score), row=1, col=col)
        good = data.loc[eligible]
        # Surface hover can interpolate coordinates; use exact fitted points for readout.
        fig.add_trace(go.Scatter3d(x=good.tau, y=np.log10(good.k), z=good[score],
                                  customdata=good[['tau', 'k', 'z', 'log_likelihood']].to_numpy(),
                                  mode='markers', marker=dict(size=2, opacity=.25, color='#444'),
                                  hovertemplate=hover, showlegend=False), row=1, col=col)
        best = good.loc[good[score].idxmax()]
        fig.add_trace(go.Scatter3d(x=[best.tau], y=[np.log10(best.k)], z=[best[score]],
                                  customdata=[[best.tau, best.k, best.z, best.log_likelihood]],
                                  mode='markers', marker=dict(size=6, color='#c52828',
                                                             line=dict(color='white', width=1)),
                                  hovertemplate='Best evaluated pair<br>' + hover,
                                  showlegend=False), row=1, col=col)
    camera = dict(eye=dict(x=-1.45, y=-1.7, z=1.1))
    for scene, ztitle in [('scene', 'Signed training z'), ('scene2', 'Log-likelihood')]:
        fig.update_layout(**{scene: dict(
            xaxis=dict(title='tau'),
            yaxis=dict(title='log10(k)'),
            zaxis=dict(title=ztitle, tickformat='.2f', separatethousands=False),
            camera=camera, dragmode='orbit', aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=.85))})
    fig.update_layout(template='plotly_white', height=720, autosize=True,
                      title=dict(text=meta.get('plot_label','X21 Tr-24')+': tau / k parameter landscape', x=.5),
                      font=dict(family='Arial, sans-serif', size=13),
                      margin=dict(l=12, r=12, t=85, b=75), uirevision='fixed-grid')
    fig.add_annotation(x=.5, y=-.08, xref='paper', yref='paper', showarrow=False,
                       text='Training only · 29 × 43 evaluated pairs · tau < 1: non-metric sensitivity',
                       font=dict(size=12, color='#555'))
    # Keep both views at the same angle for direct objective comparison.
    script = """
    const chart = document.getElementById('sbi-landscape');
    let syncing = false;
    chart.on('plotly_relayout', function(event) {
      if (syncing) return;
      const from = event['scene.camera'] ? 'scene' : event['scene2.camera'] ? 'scene2' : null;
      if (!from) return;
      const to = from === 'scene' ? 'scene2' : 'scene';
      syncing = true;
      Plotly.relayout(chart, {[to + '.camera']: event[from + '.camera']})
        .then(() => { syncing = false; }).catch(() => { syncing = false; });
    });
    """
    dest = out / 'figures' / 'parameter_landscape_interactive.html'
    fig.write_html(dest, include_plotlyjs=True, full_html=True, div_id='sbi-landscape',
                   post_script=script, config=dict(responsive=True, scrollZoom=True,
                   displaylogo=False, toImageButtonOptions=dict(format='png', scale=2,
                   filename='x21_parameter_landscape')))
    print(f'{dest.resolve()} ({dest.stat().st_size:,} bytes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    main(parser.parse_args().input)
