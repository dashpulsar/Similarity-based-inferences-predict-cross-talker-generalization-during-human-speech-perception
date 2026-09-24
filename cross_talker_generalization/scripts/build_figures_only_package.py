"""Create a figures-only PDF and aligned bilingual oral notes; no model fits."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
import fitz
from PIL import Image, ImageDraw
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/"cross_talker_generalization"
OLD=PROJECT/"analysis_update_2026-09-09"
OUT=PROJECT/"analysis_update_2026-09-11"
PDF_DIR=ROOT/"output/pdf"
FIGURE_PDF=PDF_DIR/"cross_talker_figures_only.pdf"
NOTES_PDF=PDF_DIR/"cross_talker_speaker_notes_bilingual.pdf"
PREVIEW=ROOT/"tmp/pdfs/figures_only"


def sections(path):
    return {p.split("\n",1)[0]:p.split("\n",1)[1]
            for p in re.split(r"^## ",path.read_text(encoding="utf-8"),flags=re.M)[1:]}


def paragraphs(section):
    return tuple(re.search(pattern,section,re.S).group(1).strip() for pattern in
                 [r"\*\*English\.\*\* (.*?)(?=\n\n)",r"\*\*中文。\*\* (.*?)(?=\n\n)"])


def manifest():
    generated={e["id"]:e for e in json.loads((OUT/"panel_sources.json").read_text(encoding="utf-8"))["figures"]}
    pages=[]
    def add(key,title,path,family,**extra):
        pages.append(dict(id=key,title=title,source=path.relative_to(ROOT).as_posix(),family=family,**extra))
    def old(key,title,family,**extra):
        add(key,title,OLD/"figures"/f"{key}.pdf",family,**extra)
    def new(key,title):
        item=generated[key]
        add(key,title,ROOT/item["path"],item["family"],**{k:v for k,v in item.items() if k not in ("id","family","path")})
    old("figure_1ab_method","Speech to latent trajectories","production",note_key="Figure 1a–b — From speech to a trajectory")
    add("language_capitals_worldmap","Language backgrounds",OLD/"sources/language_capitals_worldmap.pdf","production",note_key="Figure 1c — Language backgrounds")
    old("figure_1d_phonological_inventory_similarity","Inventory overlap with English","production",note_key="Figure 1d — Inventory overlap with English")
    add("english_phonology_similarity_heatmap","Supplied phonological-feature comparison",OLD/"sources/english_phonology_similarity_heatmap.pdf","production",note_key="Figure 1d companion — Supplied phonological-feature heatmap")
    old("figure_2a_an19_42_talker_distance_linear","AN19 talker distance matrix","production",note_key="Figure 2a — Talker-by-talker pronunciation distance")
    old("figure_2a_an19_42_talker_similarity_linear","AN19 talker similarity matrix","production",note_key="Figure 2a companion — Talker-by-talker pronunciation similarity")
    add("AN19_all_segments_similarity","AN19 all-segment similarity",PDF_DIR/"AN19_all_segments_similarity.pdf","production",note_key="Figure 2b overview — All eligible segment instances")
    for i,title in enumerate(["IH, IY, AE and EH","UH, UW, TH and DH","S, SH, R and L"],1):
        new(f"an19_phone_similarity_{i:02d}",f"AN19 intended phones: {title}")
        pages[-1]["note_key"]=f"Figure 2b detail {i} — {title}"
    old("figure_2c_control_similarity_with_marginals","English-reference similarity and control accuracy","production",note_key="Figure 2c — Similarity to English and control-test intelligibility")
    for dataset in ["an19","x21","b23"]:
        old(f"sbi_{dataset}_base_z",f"{dataset.upper()} SBI: historical non-ASR-FT z","sbi_z",dataset=dataset.upper(),variant="base")
    new("hve_x21_z_01","X21 HVE: training-fold z, definitions 1-4")
    new("hve_x21_loglik_01","X21 HVE: held-out likelihood, definitions 1-4")
    old("x21_s_curves_by_condition","X21 condition-specific curves","condition_curves")
    old("x21_s_curves_pooled","X21 pooled curve","pooled_curve")
    new("x21_nested_comparisons","X21 selected-model nested comparisons")
    for dataset in ["an19","x21","b23"]:
        old(f"sbi_{dataset}_ft_z",f"{dataset.upper()} SBI: historical ASR-FT z","sbi_z",dataset=dataset.upper(),variant="ft")
    for dataset in ["an19","x21","b23"]:
        for variant in ["base","ft"]:
            old(f"sbi_{dataset}_{variant}_loglik",f"{dataset.upper()} SBI: {variant} held-out likelihood","sbi_loglik",dataset=dataset.upper(),variant=variant)
    for metric in ["z","loglik"]:
        for dataset,n in [("an19",2),("x21",5),("b23",4)]:
            for i in range(1,n+1):
                key=f"hve_{dataset}_{metric}_{i:02d}"
                if key not in {p["id"] for p in pages}: new(key,f"{dataset.upper()} HVE: {metric}, definition group {i}")
    old("figure_si_language_corpus_coverage","Supplement: corpus talker coverage","production",note_key="Supplement — Corpus coverage")
    for i,e in enumerate(pages,1):
        e["page"]=i
        path=ROOT/e["source"]
        assert path.is_file(),path
        e["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(pages)==49
    return pages


def compile_figures(pages):
    doc=fitz.open()
    audit=[]
    for e in pages:
        source=fitz.open(ROOT/e["source"])
        assert len(source)==1
        page=source[0]
        bottom_extension=0.0
        if e["id"]=="english_phonology_similarity_heatmap":
            # The supplied PDF contains the complete last row label below its
            # original page box. Expose the existing text; do not reconstruct it.
            blocks=page.get_text("blocks",clip=fitz.INFINITE_RECT())
            label_blocks=[b for b in blocks if "Sentence intonation" in b[4]]
            assert len(label_blocks)==1
            bottom_extension=max(0.0,label_blocks[0][3]+12-page.rect.height)
            box=page.mediabox
            page.set_mediabox(fitz.Rect(box.x0,box.y0-bottom_extension,box.x1,box.y1))
            assert "Sentence intonation" in page.get_text()
        remove=[]
        if e["id"]=="figure_1ab_method":
            remove=["Raw activations; first and last shown","Lines join adjacent frames",
                    "Same coordinates; zoomed to the word","L1-English, male, talker 055",
                    "1,024 latent dimensions","One 3-D vector per frame"]
        if e["family"]=="sbi_loglik":
            remove=["Predictor-only model | Higher is better | Mean and 95% three-fold bootstrap interval"]
        for phrase in remove:
            boxes=page.search_for(phrase)
            if not boxes: raise ValueError(f"Expected explanatory text not found: {phrase}")
            for rect in boxes: page.add_redact_annot(rect,fill=(1,1,1))
        if remove: page.apply_redactions(images=0,graphics=0,text=0)
        output=doc.new_page(width=page.rect.width,height=page.rect.height)
        output.show_pdf_page(output.rect,source,0)
        audit.append(dict(page=e["page"],id=e["id"],removed_annotations=remove,
                          recovered_bottom_margin_points=bottom_extension,
                          dimensions_points=[page.rect.width,page.rect.height]))
        source.close()
    doc.set_metadata({"title":"Cross-talker generalization: figures","author":"","creationDate":"","modDate":""})
    # Bookmarks provide navigation without adding any text page or footer.
    doc.set_toc([[1,f'{e["page"]:02d}. {e["title"]}',e["page"]] for e in pages])
    doc.save(FIGURE_PDF,garbage=4,deflate=True)
    doc.close()
    (OUT/"figure_assembly_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")


def hve_note(e,stats):
    is_z=e["family"]=="hve_z"
    base_en,base_cn=paragraphs(stats["Revised HVE z: shared narration" if is_z else "Revised HVE held-out likelihood: shared narration"])
    # This package uses non-ASR-FT terminology in the legend.
    base_en=base_en.replace("Blue is HuBERT base","Blue is the non-ASR-fine-tuned model")
    rows={}
    for line in stats["Revised HVE held-out likelihood: shared narration"].splitlines():
        if line.startswith("| ") and chr(96) in line:
            parts=[p.strip() for p in line.strip("|").split("|")]
            if len(parts)==3:
                key=parts[0].replace(chr(96),"").replace("_*","")
                rows[key]=(parts[1],parts[2])
    method_en=[];method_cn=[]
    for method in e["methods"]:
        key=method if method in rows else method.rsplit("_",1)[0]
        if key not in rows: raise KeyError(method)
        unit="" if method in rows else method.rsplit("_",1)[1]
        a,b=rows[key]
        title=method.replace("_"," ")
        method_en.append(f"{title}: {a}")
        method_cn.append(f"{title}：{b}")
    en=f"This is {e['dataset']}. "+base_en+" In panel order: "+" ".join(method_en)
    cn=f"这一页是 {e['dataset']}。"+base_cn+" 按面板顺序："+" ".join(method_cn)
    en+=" Here non-DTW dispersion uses squared deviations at tau = 2, without a final root; within-type DTW uses Euclidean local distance."
    cn+=" 这里非 DTW dispersion 在 tau=2 时使用平方偏差、不取最终根号；within-type DTW 则使用欧氏局部距离。"
    if e["dataset"]=="AN19":
        en+=" These seven retained AN19 definitions have not yet incorporated the new phone annotations."
        cn+=" 这批 AN19 的七种已存结果尚未接入新生成的音素标注。"
    return en,cn


def narration(e,production,stats):
    if e.get("note_key"):
        en,cn=paragraphs(production[e["note_key"]])
        if e["family"]=="phone_similarity":
            en+=" Each phone panel has its own x-axis range and scientific-notation multiplier, so I read the tick values rather than compare horizontal positions across panels."
            cn+=" 每个音素面板使用各自的横轴范围和科学计数倍率，因此要比较刻度值，不能直接跨面板比较点的横向位置。"
        return en,cn
    if e["family"].startswith("hve_"): return hve_note(e,stats)
    if e["family"] in ("sbi_z","sbi_loglik"):
        key="SBI layerwise z: shared narration" if e["family"]=="sbi_z" else "SBI held-out likelihood: shared narration"
        en,cn=paragraphs(stats[key])
        variant="non-ASR-fine-tuned" if e["variant"]=="base" else "ASR-fine-tuned"
        en=f"This is {e['dataset']}, using the {variant} representation. "+en
        cn=f"这一页是 {e['dataset']} 的"+("未做 ASR 微调" if e["variant"]=="base" else "ASR 微调")+"表示。"+cn
        if e["family"]=="sbi_z":
            if e["dataset"]=="X21":
                en+=" The source also reports that k was averaged across folds, allowing parameter-selection leakage; this is not unbiased out-of-sample validation."
                cn+=" 另外，来源记录说明 k 在折间取平均，存在参数选择的信息泄漏，因此不能把它作为无偏的样本外验证。"
            else:
                en+=" The stored ceiling uses a broader participant sample than these SBI fits, so it is a historical reference, not a matched predictive upper bound."
                cn+=" 旧 ceiling 的参与者范围比这些 SBI 拟合更广，因此它只是历史参考，不是样本匹配的预测上限。"
        elif e["dataset"]=="AN19":
            en+=" In AN19, talker random-effect fallbacks differ across layer fits, so the model structure is not fully controlled across layers."
            cn+=" AN19 部分层发生了说话人随机效应结构回退，因此层间模型结构尚未完全一致。"
        return en,cn
    if e["family"]=="condition_curves": return paragraphs(stats["X21 condition-specific curves"])
    if e["family"]=="pooled_curve": return paragraphs(stats["X21 pooled curves"])
    if e["family"]=="nested_comparison":
        return (
        "Here I test whether SBI or HVE and experimental condition each add information beyond the other. These are the selected non-ASR-fine-tuned X21 configurations: SBI at Tr-14 and HVE at CNN-6 using within-word transitions. The x-axis distinguishes SBI and HVE. The y-axis is the likelihood-ratio statistic, twice the improvement in fitted log likelihood; it is not z or an out-of-fold gain. On the left, I compare condition-only with the joint model. On the right, I compare predictor-only with that same joint model. For each fold, predictor values were standardized using the mean and standard deviation from the other two folds. Those values were combined, and all three GLMMs were fitted on the same response rows. I do not reselect the predictor for each comparison. The labels give the chi-square reference degrees of freedom and p-values. For SBI, the two p-values are .0107 and .000105; for HVE they are .0532 and .0117. So condition does add to HVE. The selected HVE's predictor-only coefficient is negative, which does not support the expected positive variability effect. These tests are exploratory because predictor selection used this study. They are not an independent confirmation, and the bars do not represent three folds.",
        "这里检验 SBI 或 HVE 与实验 condition 是否各自包含对方之外的信息。展示的是 X21 获选的未做 ASR 微调配置：SBI 为 Tr-14，HVE 为 CNN-6 的单词内部 transitions。横轴区分 SBI 和 HVE，纵轴是 likelihood-ratio 统计量，即拟合 log likelihood 改善量的两倍，不是 z，也不是 OOF gain。左边比较 condition-only 与 joint model；右边比较 predictor-only 与同一个 joint model。对于每个当前折，先用另外两折的均值和标准差标准化 predictor 值，再合并这些值，在相同反应行上拟合三个 GLMM，没有为每次比较重新挑选 predictor。柱上的标签给出卡方参考自由度和 p 值。SBI 的两项 p 值是 .0107 和 .000105，HVE 是 .0532 和 .0117，因此 condition 对 HVE 确实有额外贡献。获选 HVE 在 predictor-only 模型中的系数为负，并不支持 variability 的预期正效应。由于 predictor 也在本研究中选择，这些检验是探索性的，不是独立确认，柱也不代表三个折。")
    raise ValueError(e)


INTRO_EN=("This guide follows the figure PDF page by page. English is for speaking and Chinese is for preparation. "
"The main sequence is pages 1-19; the rest contains complete layer/method profiles and corpus coverage. "
"For a shorter recap, use pages 1, 2, 5, 7, 11, 13, 15, 16, 17 and 19, and keep other pages for questions. "
"This version incorporates the recent discussion, the annotated report, and the HLP direct messages from September 6 onward, including all three reply threads. "
"This is a current figure compilation, not a claim that all manuscript analyses are complete.")
INTRO_CN=("这份稿子按图集页码逐页对应，英文可以口头讲，中文用于准备；这些文字不放进图集 PDF。"
"主线是第 1-19 页，后面是完整逐层/逐方法结果和语料覆盖。简短 recap 可以讲第 1、2、5、7、11、13、15、16、17、19 页，其他页用于回答问题。"
"本版已综合最近的讨论、批注报告，以及 9 月 6 日以来 HLP 私聊中的全部三个回复线程。"
"这是现有图的整合，不代表全部论文分析已经完成。")


def build_notes(pages):
    production=sections(OUT/"production_notes_draft.md")
    stats=sections(OUT/"statistical_notes_draft.md")
    records=[]
    for e in pages:
        en,cn=narration(e,production,stats)
        en=en.replace("\u2013","-").replace("\u2014","-")
        records.append(dict(**e,english=en,chinese=cn))
    lines=["# Figure-by-figure speaker notes / 逐图讲解稿","",INTRO_EN,"",INTRO_CN,"",
           "[Figure PDF](../../output/pdf/cross_talker_figures_only.pdf) | [Bilingual notes PDF](../../output/pdf/cross_talker_speaker_notes_bilingual.pdf)",""]
    for e in records:
        lines += [f"## Figure PDF page {e['page']:02d} | {e['title']}","","**English**","",e["english"],"",
                  "**中文对照**","",e["chinese"],"",f"Figure source: [{Path(e['source']).name}](../../{e['source']})",""]
    lines += ["## Remaining gaps / 未完成部分","",
              "Figure 2d (listener segment errors and lexical constraint), matched current-definition ceilings, controlled acoustic-component ablations, and the full z-versus-likelihood optimization comparison remain unfinished. The supplied phonological-feature heatmap has no recovered numerical scoring table/code. The automatic phoneme results quantify intended-phone intervals, not verified actual pronunciations. Old word-type-weighted phoneme distance panels and the known unmatched acoustic-component rankings are not included.","",
              "Figure 2d 的听者音段错误和词汇约束、当前定义下样本匹配的 ceiling、受控的声学成分消融，以及完整的 z/likelihood 优化目标对比仍未完成。Florian 提供的音系特征热图尚缺原始评分表和代码。自动音素结果来自预期音素区间，不等于人工验证的实际发音。旧版按词类型加权的音素距离图，以及已知预处理不匹配的声学成分排名，没有放入这个图集。",""]
    (OUT/"SPEAKER_NOTES_BILINGUAL.md").write_text("\n".join(lines),encoding="utf-8")
    (OUT/"speaker_notes.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    pdfmetrics.registerFont(TTFont("NoteEnglish","C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("NoteChinese","C:/Windows/Fonts/simhei.ttf"))
    title=ParagraphStyle("title",fontName="NoteEnglish",fontSize=14,leading=18,spaceAfter=13)
    en_style=ParagraphStyle("en",fontName="NoteEnglish",fontSize=10.3,leading=13.5,spaceAfter=12)
    cn_style=ParagraphStyle("cn",fontName="NoteChinese",fontSize=10.3,leading=14.5,spaceAfter=12,wordWrap="CJK")
    label_style=ParagraphStyle("label",fontName="NoteEnglish",fontSize=9,leading=11,spaceAfter=7,textColor="#555555")
    story=[Paragraph("Cross-talker generalization: speaking notes",title),
           Paragraph(html.escape(INTRO_EN),en_style),Paragraph(html.escape(INTRO_CN),cn_style),PageBreak()]
    for i,e in enumerate(records):
        story += [Paragraph(html.escape(f"Figure PDF page {e['page']:02d} | {e['title']}"),title),
                  Paragraph("English",label_style),Paragraph(html.escape(e["english"]),en_style),
                  Paragraph("中文对照",cn_style),Paragraph(html.escape(e["chinese"]),cn_style)]
        if i<len(records)-1: story.append(PageBreak())
    SimpleDocTemplate(str(NOTES_PDF),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=32,bottomMargin=32,
                      title="Cross-talker generalization: bilingual speaking notes",author="").build(story)
    return records


def previews(path,folder,scale):
    folder.mkdir(parents=True,exist_ok=True)
    doc=fitz.open(path); thumbs=[]
    for i,page in enumerate(doc):
        png=folder/f"page_{i+1:02d}.png"
        page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(png)
        with Image.open(png) as image:
            image=image.convert("RGB"); image.thumbnail((420,290))
            card=Image.new("RGB",(440,320),"white")
            card.paste(image,((440-image.width)//2,10))
        ImageDraw.Draw(card).text((10,304),str(i+1),fill="black")
        thumbs.append(card)
    for start in range(0,len(thumbs),12):
        chunk=thumbs[start:start+12]
        board=Image.new("RGB",(1320,320*((len(chunk)+2)//3)),"#dddddd")
        for j,im in enumerate(chunk): board.paste(im,((j%3)*440,(j//3)*320))
        board.save(folder/f"contact_{start//12+1:02d}.png")
    return len(doc)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--render",action="store_true")
    args=parser.parse_args()
    PDF_DIR.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    pages=manifest(); compile_figures(pages); build_notes(pages)
    (OUT/"figure_manifest.json").write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding="utf-8")
    readme=("# Figure updates\n\n"
    "Start with the **12-page concise update**: [figure PDF](../../output/pdf/cross_talker_figures_concise.pdf), "
    "[bilingual speaker notes](SPEAKER_NOTES_CONCISE_BILINGUAL.md) "
    "([PDF](../../output/pdf/cross_talker_speaker_notes_concise_bilingual.pdf)), and "
    "[selection/page correspondence](CONCISE_EDITION.md). This is the current presentation; the complete reference collection below is preserved separately.\n\n"
    "## Complete reference collection\n\n"
    "Use [the figure PDF](../../output/pdf/cross_talker_figures_only.pdf) and [the bilingual speaker notes](SPEAKER_NOTES_BILINGUAL.md) ([PDF](../../output/pdf/cross_talker_speaker_notes_bilingual.pdf)). "
    "The figure PDF contains no cover, explanatory prose, external captions, names or dates; axes, panel labels and necessary legends remain. Bookmarks match the numbered speaking notes.\n\n"
    "The 49-page collection has a 19-page main sequence, followed by the remaining layer/method panels. HVE z and likelihood pages now use identical method order. "
    "AN19 phone-specific pages use the same instance weighting and pairwise exponential similarity as the overall segment summary, with talker-bootstrap CIs; they replace the old word-type-weighted distance view.\n\n"
    "## Evidence coverage\n\n"
    "The HLP Slack connection was successfully verified. The available direct-message history from September 6 onward was read, including all three complete reply threads: 23 messages in total, nine from Florian. "
    "A two-page DM search was exhausted to check for replies outside the newly started threads. The latest returned reply was on September 10 (Asia/Shanghai). "
    "All 20 annotations in the locally supplied report_Sep_4-tfj.pdf were also checked. No raw private-message export or credentials are included in this package.\n\n"
    "## Review decisions applied\n\n"
    "1. Method illustration: first/last latent dimensions with an intervening ellipsis, white background, panels a/b, and a clearly identified L1-English example without a start legend (page 1). The example is explicitly changed to The wife helped her husband, not a silent a/the correction. [Discussion](https://hlplab.slack.com/archives/D0BHN5WAR24/p1788705101469909).\n"
    "2. Language context: supplied world map and feature heatmap, consistent language groups/colors, and inventory overlap (pages 2-4). The clipped heatmap row label is recovered from existing PDF text by extending the page box; no scores or wording are invented.\n"
    "3. Pronunciation matrices: English reference talkers included, language blocks, common talker order, and separate linear distance/similarity colors (pages 5-6).\n"
    "4. AN19 segments: automatic intended-phone alignment; same word/context comparison; equal segment instances within each L2 talker; descending mean similarity; talker-bootstrap CIs and simplified labels (pages 7-10). [Averaging clarification](https://hlplab.slack.com/archives/D0BHN5WAR24/p1788974045909979), [CI/layout clarification](https://hlplab.slack.com/archives/D0BHN5WAR24/p1788978312362729).\n"
    "5. Control responses: marginal similarity distributions, shared language colors, and explicit test/control interpretation (page 11). HW74 remains excluded from these matched-content comparisons because of the recorded wave/wade mismatch; no automatic relabeling. [Follow-up](https://hlplab.slack.com/archives/D0BHN5WAR24/p1788976621464199).\n"
    "6. Statistical displays: SBI normalized-z profiles retained separately from genuine held-out likelihood; complete supported HVE method profiles in matched panel order. Historical z and ceiling limitations are explained in the notes, not hidden behind a predictive-performance claim.\n"
    "7. Behavioral curves and comparison: condition-specific/pooled X21 curves, explicit nested-model comparison directions, and corpus coverage as supplementary material (pages 17-19 and 49). No unsupported result is added to fill an unavailable panel.\n\n"
    "This is a presentation package, not certification that all analyses are final. Figure 2d remains unavailable; historical z/ceiling and current held-out/model-comparison results have distinct scopes, explained in the notes. "
    "Known-mismatched acoustic ablation rankings are excluded. No ASR, t-SNE, DTW, GLMM or predictor search was rerun.\n\n"
    "## Regenerate\n\n"
    "Use the scientific Python environment for panel preparation, then Python with PyMuPDF, Pillow and ReportLab for assembly. "
    "The bilingual PDF uses the installed Windows Arial and SimHei fonts.\n\n"
    "    python cross_talker_generalization/scripts/build_figures_only_panels.py\n"
    "    python cross_talker_generalization/scripts/build_figures_only_package.py --render\n\n"
    "## Page index\n\n| PDF page | Figure |\n|---:|---|\n")
    (OUT/"README.md").write_text(readme+"\n".join(f"| {e['page']} | {e['title']} |" for e in pages)+"\n",encoding="utf-8")
    if args.render:
        a=previews(FIGURE_PDF,PREVIEW/"figures",1.05)
        b=previews(NOTES_PDF,PREVIEW/"notes",1)
        print(f"Rendered {a} figure pages and {b} note pages.",flush=True)
    print(FIGURE_PDF); print(NOTES_PDF)


if __name__=="__main__": main()
