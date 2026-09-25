"""Curate the reviewed updates; preserve the complete 49-page collection.

Run --prepare-sbi in the scientific environment, then --render in the PDF
environment. This is a display revision, with no new model fits or searches.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'cross_talker_generalization/analysis/presentation'
OLD = ROOT / 'cross_talker_generalization/analysis/speech'
PDF_DIR = ROOT / 'output/pdf'
FULL_PDF = PDF_DIR / 'cross_talker_figures_only.pdf'
FIGURE_PDF = PDF_DIR / 'cross_talker_figures_concise.pdf'
SBI_PDF = OUT / 'figures/concise_sbi_z_summary.pdf'

# Old page numbers are deliberate: this is an editorial selection, not ranking
# panels by the strength, significance or direction of their findings.
PLAN = [
    ('Speech to latent trajectories', [1], 'single'),
    ('Language backgrounds', [2], 'single'),
    ('Phonological inventory and feature comparisons', [3, 4], 'vertical'),
    ('AN19 distance and similarity matrices', [5, 6], 'horizontal'),
    ('AN19 all-segment similarity', [7], 'single'),
    ('Similarity to English and control accuracy', [11], 'single'),
    ('SBI across datasets and model variants', [12, 13, 14, 20, 21, 22], 'sbi'),
    ('X21 HVE: four retained definitions, z', [15], 'single'),
    ('X21 HVE: the same definitions, held-out likelihood', [16], 'single'),
    ('X21 condition-specific curves', [17], 'single'),
    ('X21 pooled curves', [18], 'single'),
    ('X21 nested-model comparisons', [19], 'single'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_sbi():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd
    from build_z_value_review import LAYERS, fold_ci, label, plot_points, references

    paths = [OLD / 'tables/sbi_retained_fold_z.csv', OLD / 'tables/shared_historical_ceiling_z.csv']
    frame, ceilings = [pd.read_csv(p) for p in paths]
    assert len(frame) == 342
    assert frame.scope.eq('historical_test_fold_refit').all()
    sizes = frame.groupby(['dataset_id', 'source', 'layer']).fold.agg(['size', 'nunique'])
    assert sizes.eq(3).all().all()
    keys = ['mfcc39', 'strf24_legacy', *LAYERS]
    positions = [0, 1, *range(3, len(LAYERS) + 3)]
    colors = {'base': '#20639b', 'ft': '#b43b35'}
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.0), sharex=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.115, top=.95, hspace=.27)
    rows, ceiling_rows = [], []
    for dataset, ax in zip(['AN19', 'X21', 'B23'], axes):
        group = frame.loc[frame.dataset_id.eq(dataset)].copy()
        ceiling = ceilings.loc[ceilings.dataset_id.eq(dataset)].sort_values('fold').z_value.to_numpy()
        mu, lo, hi = fold_ci(ceiling)
        assert np.allclose(group.ceiling_mean_z, mu)
        assert np.allclose(group.percent_ceiling, 100 * group.z_value / mu)
        references(ax, ceiling, True)
        baseline = group.loc[group.source.isin(keys[:2])]
        plot_points(ax, baseline, keys[:2], positions[:2], 'black', True, ceiling, connect=False)
        for variant, offset in [('base', -.14), ('ft', .14)]:
            subset = group.loc[group.source.eq(variant)]
            plot_points(ax, subset, LAYERS, positions[2:], colors[variant], True, ceiling, offset=offset)
        # A common axis preserves the negative B23 estimates and the full band.
        ax.set_ylim(-48, 128)
        ax.set_yticks([-40, 0, 50, 100])
        ax.set_ylabel('')
        ax.set_title(dataset, loc='left', fontsize=12, pad=4)
        ax.set_xlim(-.6, positions[-1]+.6)
        ax.tick_params(axis='y', labelsize=9)
        for (source, layer), part in group.groupby(['source', 'layer']):
            mean, low, high = fold_ci(part.sort_values('fold').percent_ceiling.to_numpy())
            rows.append(dict(dataset_id=dataset, source=source, layer=layer,
                             mean_percent=mean, ci_low=low, ci_high=high))
        ceiling_rows.append(dict(dataset_id=dataset, mean_z=mu, mean_percent=100,
                                 ci_low_percent=100*lo/mu, ci_high_percent=100*hi/mu,
                                 significance_percent=100*1.96/mu))
    handles = [
        Line2D([], [], color=colors['base'], label='Non-ASR-FT'),
        Line2D([], [], color=colors['ft'], label='ASR-FT'),
        Line2D([], [], color='#777777', marker='o', linestyle='none', markersize=3, label='Individual folds'),
        Patch(facecolor='#bbbbbb', alpha=.4, label='Ceiling'),
        Line2D([], [], color='#b56e00', linestyle=':', label='Significance threshold (z = +/- 1.96)'),
    ]
    axes[0].legend(handles=handles, loc='center', bbox_to_anchor=(.5, 79),
                   bbox_transform=axes[0].get_yaxis_transform(), ncol=5,
                   fontsize=8, frameon=False, columnspacing=1.1, handlelength=1.8)
    axes[-1].set_xticks(positions, [label(k) for k in keys], rotation=45, ha='right', fontsize=9)
    axes[-1].set_xlabel('Feature space', fontsize=11)
    fig.supylabel('Predictor z / mean ceiling z (%)', fontsize=12, x=.018)
    for ext in ['pdf', 'png', 'svg']:
        kw = {'metadata': {'Author': '', 'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {}
        fig.savefig(SBI_PDF.with_suffix('.'+ext), dpi=220, bbox_inches='tight', facecolor='white', **kw)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / 'tables/concise_sbi_plot_summary.csv', index=False)
    pd.DataFrame(ceiling_rows).to_csv(OUT / 'tables/concise_sbi_ceiling_summary.csv', index=False)
    (OUT / 'concise_sbi_sources.json').write_text(json.dumps({
        'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        'n_fold_values': len(frame), 'bootstrap': '27 ordered resamples of three folds',
        'scope': 'historical_test_fold_refit', 'new_fits': 0,
        'change': 'Remove raw-z duplication; overlay variants; show each acoustic baseline once per dataset.'
    }, indent=2), encoding='utf-8')
    print(SBI_PDF)








def assemble(render):
    import fitz
    from build_figures_only_package import previews

    full = fitz.open(FULL_PDF)
    assert len(full) == 49
    output = fitz.open()
    manifest = []
    for number, (title, previous, kind) in enumerate(PLAN, 1):
        if kind == 'single':
            page = full[previous[0]-1]
            target = output.new_page(width=page.rect.width, height=page.rect.height)
            target.show_pdf_page(target.rect, full, previous[0]-1)

        elif kind in ('horizontal', 'vertical'):
            sources = [full[i-1] for i in previous]
            gap, margin = 18, 12
            if kind == 'horizontal':
                height = max(p.rect.height for p in sources)
                widths = [p.rect.width * height / p.rect.height for p in sources]
                target = output.new_page(width=sum(widths)+gap+margin*2, height=height+margin*2)
                cursor = margin
                rects = []
                for width in widths:
                    rects.append(fitz.Rect(cursor, margin, cursor+width, margin+height))
                    cursor += width+gap
            else:
                # Inventory above the supplied feature examples, as requested.
                width = max(p.rect.width for p in sources)
                heights = [p.rect.height for p in sources]
                target = output.new_page(width=width+margin*2, height=sum(heights)+gap+margin*2)
                cursor = margin
                rects = []
                for source, height in zip(sources, heights):
                    left = (target.rect.width-source.rect.width)/2
                    rects.append(fitz.Rect(left, cursor, left+source.rect.width, cursor+height))
                    cursor += height+gap
            for index, rect in zip(previous, rects):
                target.show_pdf_page(rect, full, index-1)

        else:
            with fitz.open(SBI_PDF) as source:
                target = output.new_page(width=source[0].rect.width, height=source[0].rect.height)
                target.show_pdf_page(target.rect, source, 0)

        record = dict(page=number, title=title, previous_pages=previous, layout=kind)
        manifest.append(record)
    included = sorted(i for m in manifest for i in m['previous_pages'])
    excluded = sorted(set(range(1,50))-set(included))
    assert included == list(range(1,8))+list(range(11,23))
    assert excluded == list(range(8,11))+list(range(23,50))
    assert len(output) == 12
    output.set_toc([[1, f'{e["page"]:02d}. {e["title"]}', e['page']] for e in manifest])
    output.set_metadata({'title': 'Cross-talker generalization: concise figure update',
                         'author': '', 'creationDate': '', 'modDate': ''})
    output.save(FIGURE_PDF, garbage=4, deflate=True)
    output.close()
    full.close()

    audit = dict(original_pages=49, final_pages=12, removed_pages=excluded, new_model_fits=0,
                 figures=manifest, source_pdf_sha256=sha(FULL_PDF),
                 sbi_figure_sha256=sha(SBI_PDF), rationale='User-requested editorial selection; not significance-based filtering')
    (OUT / 'concise_figure_manifest.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if render:
        previews(FIGURE_PDF, ROOT/'tmp/pdfs/concise_update/figures', 1.15)
    print(FIGURE_PDF)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-sbi', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.prepare_sbi:
        prepare_sbi()
    else:
        assemble(args.render)
