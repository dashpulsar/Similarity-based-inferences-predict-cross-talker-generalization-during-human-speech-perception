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
OUT = ROOT / 'cross_talker_generalization/analysis_update_2026-09-11'
OLD = ROOT / 'cross_talker_generalization/analysis_update_2026-09-09'
PDF_DIR = ROOT / 'output/pdf'
FULL_PDF = PDF_DIR / 'cross_talker_figures_only.pdf'
FIGURE_PDF = PDF_DIR / 'cross_talker_figures_concise.pdf'
NOTES_PDF = PDF_DIR / 'cross_talker_speaker_notes_concise_bilingual.pdf'
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


SBI_EN = (
    'This single figure brings together the three datasets. The horizontal axis starts with MFCC and STRF, '
    'followed by the retained HuBERT layers. Blue is the non-ASR-fine-tuned model and red is the ASR-fine-tuned '
    'model; each acoustic baseline is shown once in black. The vertical axis is predictor z divided by that '
    'dataset\'s mean ceiling z, multiplied by 100. I use the same ceiling for both model variants within a '
    'dataset, not a common ceiling across datasets. Gray dots are the three fold estimates; colored points '
    'and error bars show their mean and 95 percent fold-bootstrap interval. The dashed line is 100 percent, '
    'with a gray interval for the ceiling; orange lines are the rescaled nominal z thresholds of plus or '
    'minus 1.96. I removed the raw-z panels because they plotted the same values on another scale. These '
    'are retained notebook test-fold refits, not frozen held-out predictions. In X21, averaging k across '
    'folds also introduces parameter-selection leakage. For AN19 and B23 the stored ceilings use a broader '
    'participant sample, so they are historical references, not matched predictive upper bounds. The '
    'percentages are not accuracy or explained variance, and three folds give only a coarse uncertainty summary.'
)
SBI_CN = (
    '这一页把三个数据集放在一起。横轴先是 MFCC 和 STRF，再是已计算的 HuBERT 各层。蓝色为未做 ASR 微调，'
    '红色为 ASR 微调；每个声学 baseline 只用黑色画一次。纵轴是 predictor 的 z 除以该数据集平均 ceiling z，'
    '再乘以 100。同一个数据集的两个模型版本共用同一 ceiling，不同数据集不共用一个 ceiling。灰色点是三个折，'
    '彩色点和误差棒是均值及 95% 折间 bootstrap 区间。虚线是 100%，灰色带是 ceiling 的区间；橙色线是换算到'
    '这个坐标上的正负 1.96。原始 z 的面板只是换一个尺度重复这些数值，所以不再展示。这些仍是 notebook '
    '里在 test fold 重新拟合得到的 z，不是冻结模型后的留出预测。X21 还存在跨折平均 k 带来的参数选择信息泄漏；'
    'AN19 和 B23 的旧 ceiling 使用更广的参与者样本，因此只能当作历史参考，不能称为样本匹配的预测上限。'
    '这个百分比不是正确率或解释方差，三个折也只能粗略地表达不确定性。'
)


def special_note(kind):
    if kind == 'vertical':
        return (
            'These two panels describe language similarity at different levels. At the top, each row is a language '
            'and moving right means greater phonological inventory overlap with English. I calculate Jaccard overlap: '
            'shared phoneme symbols divided by the union of symbols. The point averages the selected inventory pairs '
            'against nine English inventories; the bar is their minimum-to-maximum range, not a confidence interval. '
            'This measure does not include tone or phonotactics. Below it is the feature comparison Florian supplied. '
            'Columns are languages, rows are segmental, phonotactic and prosodic properties, and greener cells indicate '
            'greater similarity to English. I have preserved those supplied scores, not recalculated them. Their score '
            'table and generation code are still needed before I can explain the exact assignment of each value.',
            '这两个面板从不同层面描述语言相似性。上面每行是一种语言，越靠右表示音位库存与英语越重合。'
            '我使用 Jaccard：共有音位符号的数量除以两种库存符号的并集大小。点是所选库存与九个英语库存组合的'
            '平均值；横线是这些组合的最小到最大范围，不是置信区间。这个指标没有包含声调和音系配列。下面是 '
            'Florian 提供的具体音系特征图，列是语言，行是音段、音系配列和韵律特征，越绿表示与英语越相似。'
            '这些分数沿用原图，不是我重新计算的。原始评分表和生成代码仍然缺失，因此不能逐格说明赋值方式。'
        )
    if kind == 'horizontal':
        return (
            'Here I put the distance and similarity versions side by side. Both axes contain the same 42 AN19 '
            'talkers, including six English reference talkers, in the same order. For each talker pair I match the '
            'same 138 words and calculate DTW between their Tr-24, three-dimensional trajectories. The local distance '
            'is Euclidean, and the accumulated cost is divided by the mean number of frames in the two sequences. '
            'On the left, I average these distances within word and then across words. On the right, I first '
            'convert each recording-pair distance to exp(-distance), with k fixed at one, and then perform the '
            'same averaging. It is not the exponential of the final average distance. Both color scales are linear: '
            'larger values mean less similar on the left but more similar on the right. English comes first, other '
            'language groups are ordered by average distance to English, and speakers are clustered within each '
            'language. Gray boxes mark language groups; the masked diagonal is not an observed zero. These are '
            'representation-based comparisons, not correlations or probabilities of a correct human response.',
            '这里把距离和相似度两个版本并排展示。横纵轴都是相同顺序的 42 位 AN19 说话人，其中六位是英语参考。'
            '每一对说话人匹配同样的 138 个词，用 Tr-24 三维轨迹计算 DTW。局部距离是欧氏距离，累积代价除以'
            '两条序列帧数的平均值。左图在词内汇总，再跨词平均距离。右图先把每一对录音的距离转换成 '
            'exp(-distance)，固定 k=1，再进行相同平均，而不是对最终平均距离取指数。两边色标都是线性的，'
            '左边数值越大越不相似，右边越大越相似。英语组在最前，其他语言组按与英语的平均距离排序，组内再聚类。'
            '灰色框标记语言组，对角线是屏蔽的自我比较，不是测得的零。这些是表示空间的比较，不是相关系数，'
            '也不是人类回答正确的概率。'
        )
    if kind == 'sbi':
        return SBI_EN, SBI_CN
    raise ValueError(kind)


def revise_single(old_page, record):
    en, cn = record['english'], record['chinese']
    if old_page == 2:
        en = en.replace('The map tells us which backgrounds we cover, while the separate coverage plot tells us how many speakers we have for each dataset.', '')
        cn = cn.replace('这张图负责说明覆盖了哪些语言，具体每个数据集有多少位说话人则在后面的覆盖图中展示。', '')
    if old_page == 15:
        en = ('For HVE I retain this X21 example, rather than the complete method-by-layer catalogue. '
              'The four panels show frame dispersion around the overall pool mean, adjacent-frame changes '
              'across the actual exposure sequence, average frame dispersion within each sentence recording, '
              'and dispersion of sentence-instance mean vectors within each repeated sentence type. '
              'The global transition measure includes token boundaries; it is the one that needs exposure order. '
              'The horizontal axis is HuBERT layer, and the vertical axis is the HVE coefficient divided by its '
              'standard error: the Wald z. Blue and red distinguish the two model variants. Each gray dot comes '
              'from fitting the predictor-only GLMM to two training folds; colored points and bars summarize the '
              'three fits with a 95 percent fold-bootstrap interval. Positive and negative values indicate the '
              'direction of the fitted association, not prediction accuracy. Orange lines mark nominal plus or '
              'minus 1.96. I do not divide these training-fit z values by the older test-refit ceiling. Non-DTW '
              'dispersion here uses squared deviations at tau = 2 without a final root, and the transition '
              'measure averages squared frame changes. None of these four panels uses DTW. This is a retained '
              'example, not a claim that only four HVE definitions exist.')
        cn = ('HVE 这里保留 X21 的一个示例，不再逐页展示所有方法和层。四个面板依次是整体帧离散度、'
              '真实 exposure 顺序中的相邻帧变化、句子实例内部帧离散度，以及同一句子类型不同实例的平均向量之间的离散度。'
              '整体顺序指标包含 token 边界，所以它需要真实 exposure 顺序。横轴是 HuBERT 层，纵轴是 HVE '
              '系数除以标准误，也就是 Wald z。蓝红区分两个模型版本。每个灰点来自在两个训练折上拟合 '
              'predictor-only GLMM；彩色点和棒汇总三个拟合的均值及 95% 折间 bootstrap 区间。正负表示'
              '拟合关系的方向，不是预测正确率。橙线是名义上的正负 1.96。我没有用旧 test-refit ceiling '
              '归一化这些训练拟合的 z。这里 dispersion 在 tau=2 时使用平方偏差且不取最终根号，顺序指标则'
              '平均相邻帧的平方变化。这四个面板均不使用 DTW。这只是保留的示例，不意味着 HVE 只有四种定义。')
    if old_page == 16:
        en = ('This page shows the same four HVE definitions in the same order, but now asks how well the model '
              'predicts held-out responses. The x-axis is layer and the y-axis is log likelihood per word; higher, '
              'or less negative, is better. For each split I standardize the predictor using training data, fit '
              'the predictor-only GLMM on the two training folds, and score the untouched third fold without '
              'refitting. Predictions set random-effect contributions to zero. Each score is the sum of the '
              'observed responses\' log probabilities divided by the number of word trials. Gray dots are the '
              'three fold scores; colored means and 95 percent intervals use the same fold-bootstrap summary. '
              'This is distinct from the preceding coefficient z plot: a large coefficient z need not mean '
              'better held-out prediction. I keep this one likelihood page for that distinction, without the '
              'exhaustive appendix or an incompatible ceiling. It evaluates the retained candidates; it does '
              'not provide an independent outer-fold evaluation of choosing the best layer from these scores.')
        cn = ('这一页是相同顺序的四种 HVE 方法，但问题变成模型能否预测没有参与拟合的反应。横轴是层，'
              '纵轴是每个词的 log likelihood，越高、越接近零越好。每次划分用训练数据标准化 predictor，'
              '在两个训练折上拟合 predictor-only GLMM，然后不重新拟合，直接给第三折评分。预测时随机效应'
              '贡献设为零。把实际反应的预测概率取对数、求和，再除以词试次数，就得到每词得分。灰点是三个折'
              '的得分，彩色均值和 95% 区间沿用折间 bootstrap。这与上一页的系数 z 不同：系数 z 大，不一定'
              '意味着留出预测好。因此保留这一页来说明区别，不再附上所有逐层结果，也不加定义不匹配的 ceiling。'
              '这里评价的是已有候选项，并不是再用独立外层折验证“从这些分数中选出最好层”的整个过程。')
    return en.strip(), cn.strip()


def assemble(render):
    import fitz
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from build_figures_only_package import previews

    full = fitz.open(FULL_PDF)
    old_notes = json.loads((OUT / 'speaker_notes.json').read_text(encoding='utf-8'))
    assert len(full) == len(old_notes) == 49
    output = fitz.open()
    manifest, notes = [], []
    for number, (title, previous, kind) in enumerate(PLAN, 1):
        if kind == 'single':
            page = full[previous[0]-1]
            target = output.new_page(width=page.rect.width, height=page.rect.height)
            target.show_pdf_page(target.rect, full, previous[0]-1)
            en, cn = revise_single(previous[0], old_notes[previous[0]-1])
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
            en, cn = special_note(kind)
        else:
            with fitz.open(SBI_PDF) as source:
                target = output.new_page(width=source[0].rect.width, height=source[0].rect.height)
                target.show_pdf_page(target.rect, source, 0)
            en, cn = special_note(kind)
        record = dict(page=number, title=title, previous_pages=previous, layout=kind)
        manifest.append(record)
        notes.append(dict(**record, english=en, chinese=cn))
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

    intro_en = ('These notes follow the 12-page concise figure PDF. English is for speaking; Chinese is for preparation. '
                'This edition removes the detailed phone panels and the long appendix, combines related comparisons, '
                'and shows SBI once on the normalized-z scale. It does not add new model fits. The complete collection '
                'is retained separately; omitted pages are not evidence that those analyses are invalid or unwanted forever.')
    intro_cn = ('讲稿对应精简图集的 12 页，英文用于口头讲，中文用于准备。这个版本移除了逐音素细图和长篇附录，'
                '合并相关对照；SBI 只保留归一化 z 的一次展示，没有新增模型拟合。完整版另外保留；移出本次汇报'
                '不表示这些分析无效，或今后都不需要。')
    lines = ['# Concise figure update: speaking notes / 精简图集讲稿', '', intro_en, '', intro_cn, '',
             '[Figure PDF](../../output/pdf/cross_talker_figures_concise.pdf) | '
             '[Notes PDF](../../output/pdf/cross_talker_speaker_notes_concise_bilingual.pdf)', '']
    pdfmetrics.registerFont(TTFont('ConciseEnglish', 'C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('ConciseChinese', 'C:/Windows/Fonts/simhei.ttf'))
    title_style = ParagraphStyle('title', fontName='ConciseEnglish', fontSize=14, leading=18, spaceAfter=13)
    en_style = ParagraphStyle('en', fontName='ConciseEnglish', fontSize=10.3, leading=13.5, spaceAfter=12)
    cn_style = ParagraphStyle('cn', fontName='ConciseChinese', fontSize=10.3, leading=14.5, spaceAfter=12, wordWrap='CJK')
    label_style = ParagraphStyle('label', fontName='ConciseEnglish', fontSize=9, leading=11, spaceAfter=7)
    story = [Paragraph('Cross-talker generalization: concise speaking notes', title_style),
             Paragraph(html.escape(intro_en), en_style), Paragraph(html.escape(intro_cn), cn_style), PageBreak()]
    for i, e in enumerate(notes):
        heading = f'Figure PDF page {e["page"]:02d} | {e["title"]}'
        lines += ['## '+heading, '', '**English**', '', e['english'], '', '**中文对照**', '', e['chinese'], '']
        story += [Paragraph(html.escape(heading), title_style), Paragraph('English', label_style),
                  Paragraph(html.escape(e['english']), en_style), Paragraph('中文对照', cn_style),
                  Paragraph(html.escape(e['chinese']), cn_style)]
        if i != len(notes)-1:
            story.append(PageBreak())
    lines += ['## Scope', '',
              'Figure 2d remains unavailable. The current annotation estimates intended-phone intervals, not listener '
              'errors. Historical SBI z/ceiling limitations are explained on page 7; HVE training z and held-out '
              'likelihood are distinct quantities. The supplied feature heatmap still lacks its numerical scoring '
              'table/code. Removing the long appendix does not resolve these scientific gaps.', '']
    (OUT / 'SPEAKER_NOTES_CONCISE_BILINGUAL.md').write_text('\n'.join(lines), encoding='utf-8')
    SimpleDocTemplate(str(NOTES_PDF), pagesize=A4, leftMargin=40, rightMargin=40,
                      topMargin=32, bottomMargin=32, title='Cross-talker generalization: concise bilingual notes',
                      author='').build(story)
    audit = dict(original_pages=49, final_pages=12, removed_pages=excluded, new_model_fits=0,
                 figures=manifest, source_pdf_sha256=sha(FULL_PDF), source_notes_sha256=sha(OUT/'speaker_notes.json'),
                 sbi_figure_sha256=sha(SBI_PDF), rationale='User-requested editorial selection; not significance-based filtering')
    (OUT / 'concise_figure_manifest.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (OUT / 'concise_speaker_notes.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')
    write_index(manifest)
    if render:
        previews(FIGURE_PDF, ROOT/'tmp/pdfs/concise_update/figures', 1.15)
        previews(NOTES_PDF, ROOT/'tmp/pdfs/concise_update/notes', 1.1)
    print(FIGURE_PDF)
    print(NOTES_PDF)


def write_index(manifest):
    lines = ['# Concise figure update', '',
             'Use this 12-page edition for the current discussion: [figures](../../output/pdf/cross_talker_figures_concise.pdf), '
             '[bilingual notes PDF](../../output/pdf/cross_talker_speaker_notes_concise_bilingual.pdf), '
             'and [editable notes](SPEAKER_NOTES_CONCISE_BILINGUAL.md). The 49-page collection remains unchanged for reference.', '',
             '## What changed', '',
             '- Removed original pages 8-10 and 23-49 from this presentation, following the latest user request.',
             '- Replaced the six SBI pages with one three-dataset figure: overlay non-ASR-FT/ASR-FT, show MFCC/STRF once, '
             'and retain normalized z only. Raw z was the same information on another scale. The three folds, 95% intervals, '
             '100% ceiling band and nominal significance references remain.',
             '- Put inventory overlap above the supplied feature heatmap; put the requested distance/similarity matrices side by side.',
             '- Keep the overall segment summary, not individual-phone details; keep both condition-specific and pooled curves because '
             'they answer different questions. Keep the two original X21 HVE examples, not the entire method catalogue.', '',
             'This selection is an editorial decision, not a claim that Florian rejected every omitted analysis. In particular, '
             'his PDF explicitly requested likelihood plots; the exhaustive appendix is omitted here under the newer user instruction. '
             'No data, old PDFs, model results, or source figures were deleted. No model was rerun.', '',
             'The prior review of 23 Slack messages and 20 PDF annotations remains the source context. Historical statistical '
             'limitations remain in the spoken notes. The source feature heatmap and its unknown score construction are not relabeled '
             'as a newly computed result.', '',
             '## Page correspondence', '', '| New page | Topic | Original pages |', '|---:|---|---|']
    lines += [f'| {e["page"]} | {e["title"]} | '+', '.join(str(x) for x in e['previous_pages'])+' |' for e in manifest]
    lines += ['', '## Rebuild', '',
              'Use the scientific environment for the first command and the PDF environment for the second.', '',
              '    python cross_talker_generalization/scripts/build_concise_figure_package.py --prepare-sbi',
              '    python cross_talker_generalization/scripts/build_concise_figure_package.py --render', '']
    (OUT / 'CONCISE_EDITION.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-sbi', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.prepare_sbi:
        prepare_sbi()
    else:
        assemble(args.render)
