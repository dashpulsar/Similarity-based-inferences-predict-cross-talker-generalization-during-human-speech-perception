![AN19 intended-phone deviations: IH, IY, AE and EH](figures/figure2b_automatic_intended_phone_deviation_01.png)

AN19 intended-phone distances from six L1-English talkers in HuBERT Tr-24 3-D t-SNE space. Comparisons match word, canonical phone sequence and position; DTW uses tau = 2 and mean-sequence-length normalization. FALCON supplies automatic, unverified boundaries. Small points show L2 talkers; larger points show L1 means with 95% intervals from 1,000 talker-bootstrap samples. Diamonds denote single-talker groups without intervals. Word coverage varies by talker and phone; these are descriptive estimates, not balanced-word tests of language differences.

---

![AN19 intended-phone deviations: UH, UW, TH and DH](figures/figure2b_automatic_intended_phone_deviation_02.png)

The same method, weighting and uncertainty display apply throughout. Panel labels use ARPAbet, and horizontal scales differ across panels. The twelve displayed phone types were fixed before calculating deviations. DH has no intended tokens in this corpus's target lexicon, so its panel is explicitly empty. Some talker/phone means use only one or two word types. These automatic target-conditioned comparisons do not establish actual phoneme substitutions or listeners' perception errors.

---

![AN19 intended-phone deviations: S, SH, R and L](figures/figure2b_automatic_intended_phone_deviation_03.png)

Across all phone types, 5,114 of 16,086 L2 intervals have usable HuBERT frames and all six English references; 1,641 of those intervals belong to the eleven observed phone types shown here. HW74 wave/wade is excluded. Requiring each target/reference interval to be at least 20 ms with mean target posterior at least .2 leaves only 197 intervals across all phone types. That posterior is not calibrated accuracy; the reduced coverage prevents claiming robustness to annotation quality. No missing trajectory was interpolated.
