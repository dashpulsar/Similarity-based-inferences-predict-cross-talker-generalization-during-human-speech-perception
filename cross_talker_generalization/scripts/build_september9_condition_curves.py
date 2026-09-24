"""Relabel the retained X21 condition and pooled S-curves without refitting.

The original SVG's numerical geometry is preserved. This is a presentation
revision, not a new statistical analysis. Run from the project environment:

    python cross_talker_generalization/scripts/build_september9_condition_curves.py

Requires Matplotlib, PyMuPDF, and Playwright with an installed Chromium browser.
The default is the installed Microsoft Edge channel; use --browser-channel
chromium for Playwright's bundled Chromium. No model features are recomputed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath, TextToPath
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/figures/X21-base-tr24-notebook-s-curves-legacy-axis-z-v5"
OUTPUT = ROOT / "cross_talker_generalization/analysis_update_2026-09-09"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
TITLES = {
    "035 Mandarin Chinese (Cmn), Male (M)": "Mandarin Chinese (CMN), Male (M) 035",
    "032 CMN, M": "CMN-M-032",
    "043 CMN, M": "CMN-M-043",
    "037 CMN, M": "CMN-M-037",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_geometry(root):
    """All data marks, uncertainty bands, and axes must remain byte-identical."""
    prefixes = ("line2d_", "PolyCollection_", "PathCollection_", "LineCollection_", "matplotlib.axis_")
    return {
        element.get("id"): ET.tostring(element)
        for element in root.iter()
        if element.get("id", "").startswith(prefixes)
    }


def text_as_svg_path(text: str, size: float) -> str:
    path = TextPath((0, 0), text, size=size, prop=FontProperties(weight="bold"))
    codes = {
        MplPath.MOVETO: "M", MplPath.LINETO: "L", MplPath.CURVE3: "Q",
        MplPath.CURVE4: "C", MplPath.CLOSEPOLY: "Z",
    }
    commands = []
    for vertices, code in path.iter_segments(curves=True, simplify=False):
        commands.append(codes[code] + (" " + " ".join(f"{v:.8f}" for v in vertices) if code != MplPath.CLOSEPOLY else ""))
    return " ".join(commands)


def relabel(source: Path, destination: Path, dpi: int, browser):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(source.read_bytes(), parser=parser)
    before = numerical_geometry(root)
    all_defs_before = {node.get("id"): node.attrib.copy() for node in root.iter(f"{{{SVG}}}path") if node.get("id")}
    font = FontProperties(weight="bold")
    font_measurer = TextToPath()
    global_defs = root.find(f"{{{SVG}}}defs")
    replacements = []
    for group in list(root.iter(f"{{{SVG}}}g")):
        comment = next((node for node in group if node.tag is ET.Comment), None)
        old = comment.text.strip() if comment is not None else ""
        if old not in TITLES:
            continue
        old_group = group.find(f"{{{SVG}}}g")
        match = re.fullmatch(r"translate\(([-.\d]+) ([-.\d]+)\) scale\(([-.\d]+) ([-.\d]+)\)", old_group.get("transform", ""))
        if not match:
            raise ValueError(f"Unexpected title transform: {old}")
        x, y, scale_x, scale_y = map(float, match.groups())
        if abs(scale_x + scale_y) > 1e-8:
            raise ValueError("Unexpected anisotropic title scale")
        size = 100 * scale_x
        font.set_size(size)
        old_width = font_measurer.get_text_width_height_descent(old, font, False)[0]
        new_width = font_measurer.get_text_width_height_descent(TITLES[old], font, False)[0]
        new_x = x + (old_width - new_width) / 2
        # Other axis labels can refer to glyphs first declared in the title.
        for defs in old_group.iter(f"{{{SVG}}}defs"):
            for glyph in defs:
                global_defs.append(copy.deepcopy(glyph))
        group_id = group.get("id")
        group.clear()
        group.set("id", group_id)
        group.append(ET.Comment(" " + TITLES[old] + " "))
        ET.SubElement(group, f"{{{SVG}}}path", {
            "d": text_as_svg_path(TITLES[old], size),
            "transform": f"translate({new_x:.8f} {y:.8f}) scale(1 -1)",
            "style": "fill: #000000",
            "aria-label": TITLES[old],
        })
        replacements.append({"before": old, "after": TITLES[old]})
    if len(replacements) != 4:
        raise ValueError(f"Expected four title changes, found {len(replacements)}")
    assert before == numerical_geometry(root), "A data or uncertainty mark changed"
    all_defs_after = {node.get("id"): node.attrib.copy() for node in root.iter(f"{{{SVG}}}path") if node.get("id")}
    assert all_defs_before == all_defs_after, "A referenced glyph definition changed"
    # The visible figure has no date or author; do not propagate the old timestamp.
    for metadata in root.findall(f"{{{SVG}}}metadata"):
        root.remove(metadata)
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    svg_path = destination.with_suffix(".svg")
    svg_path.write_bytes(payload)
    # Chromium preserves SVG clipping and alpha. Direct MuPDF SVG conversion
    # loses these in this source, making uncertainty bands opaque and unclipped.
    _, _, width, height = map(float, root.get("viewBox").split())
    page = browser.new_page()
    page.set_content(
        f'<html><head><style>@page {{size:{width}pt {height}pt; margin:0}}'
        'html,body {margin:0;padding:0} svg {display:block}</style></head><body>'
        + payload.decode("utf-8").split("?>", 1)[-1] + '</body></html>',
        wait_until="load",
    )
    pdf_bytes = page.pdf(width=f"{width / 72}in", height=f"{height / 72}in", print_background=True,
                         margin={side: "0" for side in ("top", "bottom", "left", "right")},
                         prefer_css_page_size=True)
    page.close()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        pdf.set_metadata({"title": destination.stem.replace("_", " "), "author": "", "creationDate": "", "modDate": ""})
        pdf.save(destination.with_suffix(".pdf"))
        pdf[0].get_pixmap(dpi=dpi, alpha=False).save(destination.with_suffix(".png"))
    return {"replacements": replacements, "unchanged_geometry_groups": len(before)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--browser-channel", default="msedge", help="Installed Chromium channel, or chromium for Playwright's bundled browser")
    args = parser.parse_args()
    figures = OUTPUT / "figures"
    tables = OUTPUT / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    old_metadata = json.loads((SOURCE / "provenance.json").read_text(encoding="utf-8"))
    trial_source = SOURCE / "s_curve_trials.csv"
    assert sha256(trial_source) == old_metadata["outputs_sha256"][trial_source.name]
    with trial_source.open(encoding="utf-8", newline="") as handle:
        trials = list(csv.DictReader(handle))
    assert len(trials) == 16477
    assert len({row["participant_id"] for row in trials}) == 320
    assert {row["layer"] for row in trials} == {"tr_24"}
    summary = []
    for talker in old_metadata["talker_order"]:
        for condition in old_metadata["condition_order"] + ["All conditions (pooled)"]:
            rows = [row for row in trials if row["item_talker"] == talker and (condition == "All conditions (pooled)" or row["condition"] == condition)]
            summary.append({
                "test_talker": talker, "condition": condition, "trial_rows": len(rows),
                "participants": len({row["participant_id"] for row in rows}),
                "accuracy": sum(float(row["listener_accuracy"]) for row in rows) / len(rows),
            })
    checks = {}
    with sync_playwright() as playwright:
        launch_args = {} if args.browser_channel == "chromium" else {"channel": args.browser_channel}
        browser = playwright.chromium.launch(headless=True, **launch_args)
        for stem in ("x21_s_curves_by_condition", "x21_s_curves_pooled"):
            source = SOURCE / f"{stem}.svg"
            assert sha256(source) == old_metadata["outputs_sha256"][source.name]
            checks[stem] = relabel(source, figures / stem, args.dpi, browser)
        browser.close()
    metadata = {
        "revision": "X21 Tr-24 base display revision; titles changed, all scientific geometry unchanged",
        "source_directory": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": {name: sha256(SOURCE / name) for name in ("provenance.json", "s_curve_trials.csv", "x21_s_curves_by_condition.svg", "x21_s_curves_pooled.svg")},
        "representation": "HuBERT base, 3-D t-SNE, Transformer layer 24",
        "coordinate_scaling": old_metadata["coordinate_scaling"],
        "similarity_definition": old_metadata["similarity_definition"],
        "similarity_parameter_origin": old_metadata["k_origin"],
        "curve_definition": old_metadata["curve_definition"],
        "point_definition": old_metadata["point_definition"],
        "pooling_definition": "All trial rows from all four exposure conditions within each test talker. Trial-weighted, not an equally weighted mean of four condition curves.",
        "interval_definition": "Existing bands and point intervals retained exactly from the historical descriptive SVG. Source metadata does not establish the bootstrap resampling unit or replicate count; no participant-cluster or three-fold uncertainty claim is made.",
        "not_a_new_glmm_fit": True,
        "not_cross_validated_prediction": True,
        "geometry_checks": checks,
        "trial_summary": summary,
        "output_sha256": {path.name: sha256(path) for path in sorted(figures.glob("x21_s_curves_*"))},
    }
    target = tables / "x21_s_curve_revision_provenance.json"
    target.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved both X21 curve versions with exact original scientific geometry to {figures}")
    print(f"Source verification: {len(trials)} trials, 320 participants; four corrected titles per figure")


if __name__ == "__main__":
    main()
