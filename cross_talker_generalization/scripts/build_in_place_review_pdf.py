"""Revise the annotated report's pages while retaining and replying to its comments.

No model fitting or source-file overwrite. Original comment text, authors, IDs,
timestamps and annotation types are preserved; anchors are explicitly repositioned.
Requires PyMuPDF. The unmodified input is embedded for comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import fitz

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / 'cross_talker_generalization/analysis/speech'

# First ten pages retain the reviewed report's topic order. Added material follows.
# Source pages are one-based pages in the already visually verified clean report.
PAGE_MAP = [1, 2, 4, 5, 7, 9, 10, 13, 15, 18, 3, 6, 8, 11, 12, 14, 16, 17]
NAVIGATION = {
    1: [(17, 'Ceiling and sample details'), (18, 'Acoustic scaling')],
    3: [(11, 'Language world map')],
    4: [(12, 'Specific phonological features')],
    5: [(13, 'Similarity matrix')],
    7: [(14, 'SBI likelihood'), (15, 'HVE likelihood')],
    8: [(16, 'Pooled black curves')],
    9: [(17, 'Sample and model details')],
    11: [(3, 'Back to coverage')], 12: [(4, 'Back to inventory')],
    13: [(5, 'Back to distance')], 14: [(7, 'Back to SBI z')],
    15: [(7, 'Back to SBI z')], 16: [(8, 'Back to condition curves')],
    17: [(9, 'Back to model comparisons')], 18: [(1, 'Back to summary')],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_array(numbers):
    return '[' + ' '.join(f'{x:.6f}' for x in numbers) + ']'


def pdf_rect(rect, page):
    return pdf_array(tuple(rect * ~page.transformation_matrix))


def move_annotation(doc, page, annot, entry):
    """Keep original annotation identity and wording; adjust geometry only."""
    xref = annot.xref
    previous = list(annot.rect)
    kind = annot.type[0]
    if kind == fitz.PDF_ANNOT_HIGHLIGHT:
        matches = page.search_for(entry['anchor'], quads=True)
        if not matches:
            raise ValueError(f"Missing highlight target: {entry['id']} / {entry['anchor']}")
        quad = matches[0]
        points = [fitz.Point(p) * ~page.transformation_matrix for p in quad]
        doc.xref_set_key(xref, 'QuadPoints', pdf_array([v for p in points for v in p]))
        rect = quad.rect + (-2, -1, 2, 1)
        doc.xref_set_key(xref, 'Rect', pdf_rect(rect, page))
        # Reload the annotation object before rebuilding its appearance.
        annot = page.load_annot(xref)
        annot.update()
    elif kind == fitz.PDF_ANNOT_INK:
        # Preserve the original mark's shape and color in the right review margin.
        # Scaling it there keeps the original stroke from obscuring revised text.
        old_rect = annot.rect
        new_width = 16.0
        factor = new_width / old_rect.width
        new_height = old_rect.height * factor
        top = 75 if entry['id'] == 'P17' else entry['reply_y']
        new_rect = fitz.Rect(805, top, 805 + new_width, top + new_height)
        lists = []
        for stroke in annot.vertices:
            flat = []
            for x, y in stroke:
                point = fitz.Point(805 + (x-old_rect.x0)*factor,
                                   top + (y-old_rect.y0)*factor)
                point = point * ~page.transformation_matrix
                flat.extend(point)
            lists.append(pdf_array(flat))
        doc.xref_set_key(xref, 'InkList', '['+' '.join(lists)+']')
        doc.xref_set_key(xref, 'Rect', pdf_rect(new_rect, page))
        # Keep the original appearance stream; its BBox is mapped to the new Rect.
        # Match stroke metadata to the resized appearance for later viewer edits.
        width = annot.border.get('width', 1) * factor
        doc.xref_set_key(xref, 'BS', f'<</W {width:.6f} /S /S>>')
    elif kind == fitz.PDF_ANNOT_TEXT:
        # Stagger formerly coincident notes beside the affected caption/figure.
        annot.set_rect(fitz.Rect(804, entry['reply_y'], 820, entry['reply_y']+16))
    else:
        raise ValueError(f'Unsupported original annotation type {annot.type}')
    annot = page.load_annot(xref)
    return page, annot, previous


def replace_page_contents(doc, page, revised, source_index):
    """Replace page content streams, not merely cover old text with white boxes."""
    for link in page.get_links():
        page.delete_link(link)
    empty = doc.get_new_xref()
    doc.update_object(empty, '<<>>')
    doc.update_stream(empty, b'')
    page.set_contents(empty)
    # Exclude the clean report's footer/page number, then draw this edition's own.
    content = fitz.Rect(0, 0, page.rect.width, page.rect.height-38)
    page.show_pdf_page(content, revised, source_index, clip=content)
    for link in revised[source_index].get_links():
        if link['kind'] == fitz.LINK_URI:
            page.insert_link({'kind': fitz.LINK_URI, 'from': link['from'], 'uri': link['uri']})


def add_navigation(doc, page, number):
    shade = (0.37, 0.44, 0.48)
    blue = (0.14, 0.36, 0.55)
    page.insert_text((40, 17),
                     'REVISED IN PLACE | Original review comments retained; anchors updated'
                     if number <= 10 else 'SUPPLEMENT | Added in response to the review',
                     fontsize=6.7, color=shade)
    page.insert_text((40, page.rect.height-22), 'Cross-talker generalization | Revision with replies',
                     fontsize=8, color=shade)
    page.insert_text((page.rect.width-52, page.rect.height-22), str(number), fontsize=8, color=shade)
    x = 40
    for destination, title in NAVIGATION.get(number, []):
        text = f'{title} (p. {destination})'
        width = fitz.get_text_length(text, fontsize=8)
        rect = fitz.Rect(x, page.rect.height-50, x+width+10, page.rect.height-36)
        page.insert_text((x, page.rect.height-40), text, fontsize=8, color=blue)
        page.insert_link({'kind': fitz.LINK_GOTO, 'from': rect,
                          'page': destination-1, 'to': fitz.Point(0, 0)})
        x += width+34


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--revised', type=Path,
                        help='Optional clean page template; otherwise render the current report source to tmp')
    parser.add_argument('--output', type=Path, default=REPO/'output/pdf/report_Sep_4-tfj_revised_with_replies.pdf')
    args = parser.parse_args()
    original = args.original.resolve()
    output = args.output.resolve()
    if args.revised is None:
        args.revised = REPO/'tmp/pdfs/florian_in_place/revised_page_template.pdf'
        subprocess.run([
            sys.executable, str(Path(__file__).with_name('build_collaborator_report_pdf.py')),
            '--source', str(REPORT/'REPORT_FOR_FLORIAN.md'),
            '--output', str(args.revised), '--image-max-height', '350',
        ], check=True)
    if output in [original, args.revised.resolve()]:
        raise ValueError('The output must not overwrite either input PDF.')
    source_hash = sha(original)
    entries = json.loads((REPORT/'tables/pdf_review_replies.json').read_text(encoding='utf-8'))
    expected = {entry['original_id']: entry for entry in entries}
    doc = fitz.open(original)
    revised = fitz.open(args.revised)
    if len(doc) != 10 or len(revised) != 18:
        raise ValueError('This review mapping requires the original 10-page annotated PDF and its retained 18-page clean template. Pass --revised output/pdf/cross_talker_analysis_report_for_florian_reviewed.pdf. The newer 22-page phoneme report has a separate page layout and must not silently reuse these anchors.')
    originals = {}
    for pno, page in enumerate(doc):
        for annot in page.annots() or []:
            originals[annot.info['id']] = {'xref': annot.xref, 'info': annot.info,
                                           'type': list(annot.type), 'page': pno+1}
    assert set(originals) == set(expected), 'Unexpected original comment inventory'
    for _ in range(8):
        doc.new_page(width=doc[0].rect.width, height=doc[0].rect.height)
    titles = []
    for index, source_page in enumerate(PAGE_MAP):
        page = doc[index]
        replace_page_contents(doc, page, revised, source_page-1)
        text = revised[source_page-1].get_text().splitlines()
        titles.append(text[2])
        add_navigation(doc, page, index+1)
    intro = ('I have revised the original ten pages and kept your comments, repositioned beside the revised content. '
             'My responses are attached as replies under each comment (author: Response). Pages 11-18 contain the '
             'additional figures and explanations; the page links below navigate to them. The unmodified reviewed '
             'PDF is embedded as an attachment for reference. No new GLMM fits were run for these revisions.')
    assert doc[0].insert_textbox(fitz.Rect(46, 365, 790, 440), intro,
                                fontsize=10, lineheight=1.45, color=(.22,.29,.33)) >= 0
    audit = []
    for entry in entries:
        record = originals[entry['original_id']]
        assert record['page'] == entry['page']
        page = doc[entry['page']-1]
        annot = page.load_annot(record['xref'])
        page, annot, old_rect = move_annotation(doc, page, annot, entry)
        # Some appearance refreshes can update metadata; restore original info.
        annot.set_info(record['info'])
        new_rect = list(annot.rect)
        reply = page.add_text_annot(fitz.Point(824, entry['reply_y']), entry['reply'], icon='Comment')
        reply.set_rect(fitz.Rect(824, entry['reply_y'], 838, entry['reply_y']+14))
        reply.set_info(title='Response', subject=f"Reply to {entry['id']}")
        reply.set_colors(stroke=(.15,.48,.72))
        reply.set_irt_xref(annot.xref)
        doc.xref_set_key(reply.xref, 'RT', '/R')
        reply.set_open(False)
        reply.update()
        audit.append({'id':entry['id'],'page':entry['page'],'original_id':entry['original_id'],
                      'original_rect':old_rect,'revised_rect':new_rect,
                      'reply_id':reply.info['id'],'reply':entry['reply'],
                      'anchor':entry['anchor'],'related_pages':entry['related_pages']})
    doc.set_toc([[1, f'{i+1}. {title}', i+1] for i, title in enumerate(titles)])
    doc.embfile_add('Original reviewed PDF', original.read_bytes(), filename=original.name,
                    ufilename=original.name, desc='Unmodified annotated source; revised anchors are in the main document.')
    metadata = dict(doc.metadata)
    metadata.update(title='Cross-talker generalization: revised report with comment replies', author='')
    doc.set_metadata(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output, garbage=3, deflate=True)
    doc.close()
    revised.close()
    assert sha(original) == source_hash, 'Original source unexpectedly changed'
    # Reopen the final file: verify identities, reply threading and appendix links.
    final = fitz.open(output)
    found = {}
    replies = []
    for page in final:
        for annot in page.annots() or []:
            if annot.info['id'] in originals:
                previous = originals[annot.info['id']]
                assert annot.info == previous['info'], annot.info['id']
                assert list(annot.type) == previous['type']
                found[annot.info['id']] = annot.xref
            else:
                replies.append((annot.xref, final.xref_get_key(annot.xref,'IRT'), annot.info))
    assert len(found) == 20 and len(replies) == 20
    assert {int(irt[1].split()[0]) for _,irt,_ in replies} == set(found.values())
    assert final.embfile_get(0) == original.read_bytes()
    previews = REPO/'tmp/pdfs/florian_in_place'
    previews.mkdir(parents=True,exist_ok=True)
    for i,page in enumerate(final):
        page.get_pixmap(matrix=fitz.Matrix(1.3,1.3),annots=True).save(previews/f'page_{i+1:02d}.png')
    (REPORT/'tables/pdf_in_place_revision_audit.json').write_text(json.dumps({
        'original_sha256':source_hash,'clean_revision_sha256':sha(args.revised),
        'output_sha256':sha(output),'pages':len(final),'original_comments_preserved':len(found),
        'native_threaded_replies':len(replies),'input_unchanged':True,
        'original_file_embedded_unchanged':True,'new_model_fits':0,
        'page_mapping_from_clean_report':PAGE_MAP,'annotations':audit,
    },indent=2),encoding='utf-8')
    print(f'{output}\n{len(final)} pages; 20 preserved original annotations; 20 native linked replies.')


if __name__ == '__main__':
    main()
