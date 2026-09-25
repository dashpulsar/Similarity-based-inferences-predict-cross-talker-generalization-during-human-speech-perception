"""Create a figures-only PDF; no model fits."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
import fitz
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/"cross_talker_generalization"
OLD=PROJECT/"analysis/speech"
OUT=PROJECT/"analysis/presentation"
PDF_DIR=ROOT/"output/pdf"
FIGURE_PDF=PDF_DIR/"cross_talker_figures_only.pdf"
PREVIEW=ROOT/"tmp/pdfs/figures_only"






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
    parser=argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args=parser.parse_args()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pages=manifest()
    compile_figures(pages)
    (OUT/"figure_manifest.json").write_text(json.dumps(pages, indent=2), encoding="utf-8")
    if args.render:
        previews(FIGURE_PDF, PREVIEW/"figures", 1.05)
    print(FIGURE_PDF)


if __name__=="__main__": main()
