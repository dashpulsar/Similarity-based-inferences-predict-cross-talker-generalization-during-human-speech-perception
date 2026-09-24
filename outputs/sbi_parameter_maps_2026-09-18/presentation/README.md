# X21 parameter maps

Open `X21_parameter_maps.pptx`. Slide 1 contains all four conditions. Slide 2 excludes Talker-specific. Click the figure or **Open interactive map** to open the corresponding HTML in a browser.

Keep these files in the same folder:

- `X21_parameter_maps.pptx`
- `x21_all_conditions.html`
- `x21_without_talker_specific.html`

Extract the complete ZIP before opening the deck. In PowerPoint editing view a hyperlink may require Ctrl+click; in Slide Show view click it. PowerPoint or your organization may restrict local HTML links. If a link is blocked, open the corresponding HTML directly in your browser. Both HTML files include their plotting library and aggregate data, so they can work offline. The curves are static previews inside PowerPoint; rotation takes place in the browser.

Both analyses use X21 non-ASR-fine-tuned HuBERT Tr-24, existing 3-D t-SNE, and participant training folds 1+2. Each map contains 1,247 fitted tau/k pairs. Additional small-k reference fits are outside the plotted positive-k grid. These are training landscapes. Raw z and total fitted log likelihood across the two samples do not provide a direct performance comparison.

Source reports are in `cross_talker_generalization/analysis_update_2026-09-18/x21_parameter_landscape/` and `x21_parameter_landscape_no_talker_specific/` in the project. No new analysis was run to make this deck.
