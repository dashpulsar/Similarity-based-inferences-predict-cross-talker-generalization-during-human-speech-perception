# Statistical figure speaker notes: English and Chinese

These notes refer to existing figure stems rather than final PDF page numbers. They were checked against the scripts, source metadata, fold tables and combined-data model outputs. No models were fitted for this document. The current Slack connection was not used as evidence for any additional comments.

## SBI layerwise z: shared narration

Applies to `sbi_an19_base_z`, `sbi_an19_ft_z`, `sbi_x21_base_z`, `sbi_x21_ft_z`, `sbi_b23_base_z`, and `sbi_b23_ft_z` in `analysis/speech/figures`.

**English.** Here I am showing how the association between similarity and human responses changes across the feature spaces. On the horizontal axis, MFCC and STRF come first, followed by the 18 HuBERT layers we analyzed. These are CNN-2 through CNN-6 and every second transformer layer from Tr-0 to Tr-24, not all 32 extraction outputs. In the lower panel, the vertical axis is the predictor's Wald z: its fitted coefficient divided by its standard error. In the upper panel, I divide each z by the mean behavioral-reference z and multiply by 100. So the dashed line at 100 is the mean reference, and the gray band shows its fold-bootstrap interval. Small gray points show the three fold results, and the black point and bars summarize their mean and 95% fold-bootstrap interval. The orange lines are the nominal plus or minus 1.96 reference. The important distinction is that these are retained notebook results in which the response model was refitted on each test partition. They describe the association in those partitions; they are not frozen-model predictions of unseen participants. The percentage is a rescaling of z, not percentage accuracy or percentage variance explained.

**中文。** 这张图展示 similarity 与人类反应的关联如何随着特征空间变化。横轴最左边是 MFCC 和 STRF，后面是这次实际分析的 18 个 HuBERT 层：CNN-2 到 CNN-6，以及 Tr-0 到 Tr-24 每隔一层取一个，并不是全部 32 个提取输出。下面一幅的纵轴是预测变量的 Wald z，也就是回归系数除以标准误；上面一幅把这个 z 除以行为参考值的三折平均 z，再乘以 100。因此，100% 虚线是参考值的平均水平，灰色带是参考值的折间 bootstrap 区间。灰色小点是三折各自的结果，黑点和误差棒是均值及其 95% 折间 bootstrap 区间；橙色线对应正负 1.96 的名义参考阈值。这里需要区分的是，这些是保留的 notebook 结果，每个测试分区上重新拟合了反应模型，展示的是关联，而不是冻结模型对新参与者的预测。这个百分比也不是正确率或解释方差的百分比。

**If asked about the ceiling.** I estimate item accuracy from the other participant folds, convert it to log odds, attach it to the target fold, and fit a GLMM there. The stored AN19 and X21 reference keys pool over condition; B23 includes condition. This historical reference is useful for reproducing the existing display, but it is not yet the matched item-by-condition reference needed for the revised predictive analysis. The AN19 and B23 reference populations are also broader than their SBI populations. I therefore would not call it a strict, matched upper bound on these models.

**如果被问到 ceiling。** 我先用其他参与者折估计每个项目的正确率，把它转成 log odds，再匹配到目标折并在那里拟合 GLMM。保留的 AN19 和 X21 参考值在分组时合并了 condition，B23 则包含 condition；而且 AN19 和 B23 的参考样本范围比 SBI 样本更大。因此，这个参考值可以用来复现旧图，但还不是新预测分析所需的、样本和方法完全匹配的 item-by-condition 上限。

### SBI z: per-figure facts and optional transitions

| Figure stem | Optional English sentence | 中文对应 |
|---|---|---|
| `sbi_an19_base_z` | In AN19, the strongest mean neural z is at Tr-12, about 13.69; STRF is about 13.94 and MFCC about 11.72. The acoustic baselines are genuinely strong in these stored association results. | AN19 中神经网络的平均 z 最高出现在 Tr-12，约为 13.69；STRF 约为 13.94，MFCC 约为 11.72。声学基线在这批保留的关联结果中确实很强。 |
| `sbi_an19_ft_z` | With ASR fine-tuning, the largest mean z is about 10.46 at Tr-20, below the two acoustic baselines in this display. That does not by itself establish a predictive ranking. | ASR 微调版本的平均 z 最高约为 10.46，出现在 Tr-20，在这张图里低于两个声学基线，但这本身不能确定样本外预测的排名。 |
| `sbi_x21_base_z` | In X21, the largest mean z is about 8.07 at Tr-22, compared with about 5.74 for MFCC and 5.72 for STRF. This is a historical descriptive result, with an additional parameter-selection caveat. | X21 的平均 z 最高约为 8.07，出现在 Tr-22；MFCC 约为 5.74，STRF 约为 5.72。这是历史性的描述结果，而且还有参数选择方面的限制。 |
| `sbi_x21_ft_z` | The fine-tuned X21 profile also peaks at Tr-22, at about 7.97. The two profiles are close at their peaks; I would not interpret that small difference as evidence that one model is better. | X21 微调版本也在 Tr-22 达到峰值，约为 7.97。两者的峰值很接近，不能仅凭这一小段差距就说某个模型更好。 |
| `sbi_b23_base_z` | The B23 associations are much weaker in this display: the largest neural mean z is about 1.85 at Tr-20. MFCC and STRF are about 0.51 and 0.28. | 这张图中 B23 的关联弱得多：神经网络平均 z 最高约为 1.85，出现在 Tr-20；MFCC 和 STRF 分别约为 0.51 和 0.28。 |
| `sbi_b23_ft_z` | In fine-tuned B23, the largest mean z is about 1.45 at Tr-24. I retain the sign and the fold spread rather than treating these small values as zero. | B23 微调版本的平均 z 最高约为 1.45，出现在 Tr-24。我保留了正负方向和折间变化，并没有把较小的值当成零。 |

**Source-specific warning, not optional for interpretation:** X21's plotted `corrected` rows use the mean of the three optimized k values. Their own run metadata explicitly records cross-fold k leakage: some observations influence k through other folds' training sets. They must not be described as an unbiased held-out test, even before considering the test-partition refit. AN19 and B23 k selection used validation-z-based criteria, rather than the current predictor-only held-out-likelihood selection criterion. These figures preserve those historical analyses, not the completed implementation of the newer selection plan.

**与来源有关的重要限制：** X21 图中的 `corrected` 结果使用三个优化后 k 的平均值，来源元数据明确记录了跨折 k 信息泄漏。因此，即使不考虑测试分区上的重新拟合，也不能把它叫作无偏的测试结果。AN19 和 B23 的历史 k 选择依据验证集 z，而不是目前讨论的 predictor-only held-out likelihood 标准。这些图保留的是历史分析，不代表新的选择方案已经全部实现。

## SBI held-out likelihood: shared narration

Applies to the six `sbi_{dataset}_{variant}_loglik` figures in `analysis/speech/figures`.

**English.** This companion figure asks a different question: how well does a model fitted to two participant folds predict responses in the third? The horizontal axis contains the same acoustic baselines and HuBERT layers. The vertical axis is held-out log likelihood per word, so higher, or less negative, is better. For each held-out response, I use the probability from the training model and score the observed correct or incorrect outcome. I then add those scores and divide by the number of words. The model contains the theoretical predictor but not experimental condition, with the registered participant and item effects and the dataset's talker structure. Predictions use the fitted fixed effects with all random-effect contributions set to zero; the model is not refitted on the test participants. Gray points are the three held-out folds, and the line and bars give their mean and 95% fold-bootstrap interval. These particular SBI runs used standardized negative DTW, not the exponential similarity used in the old z figure, so the two figures are not a controlled comparison of reporting metrics. A higher score here also does not, on its own, answer whether SBI adds beyond condition; that requires the joint-versus-condition comparison.

**中文。** 这张配套图回答的是另一个问题：用其中两折参与者拟合的模型，能否预测第三折参与者的反应？横轴仍然是声学基线和 HuBERT 各层，纵轴是每个单词的 held-out log likelihood，越高、也就是越接近零越好。我用训练模型给出测试反应的概率，根据实际正确或错误结果计算 log score，然后加总并除以单词数。模型包括理论预测变量，但不包括实验 condition，同时保留注册的参与者、项目和相应的说话人结构。预测时随机效应贡献统一设为零，并没有在测试参与者上重新拟合。灰点是三折结果，连线和误差棒是均值及 95% 折间 bootstrap 区间。这批 SBI 模型使用标准化后的负 DTW，而旧 z 图使用指数 similarity，所以两张图不是只更换评价指标的严格对应实验。这里分数更高也不等于已经证明 SBI 超出 condition 仍有贡献，后者需要比较 joint model 和 condition-only model。

**Technical formula, if asked:** for correct and incorrect counts c and e, the score is `[c log(p) + e log(1-p)] / (c+e)`, summed before normalization across rows within a fold. This is a Bernoulli per-word predictive score and omits the binomial coefficient constant. The R model is binomial-logit `glmer`; `predict(..., re.form=NA, allow.new.levels=TRUE)` supplies test probabilities. The display averages the three fold-normalized values equally. Candidate selection instead sums all held-out losses and divides by all held-out trials. With unequal fold sizes, the display's mean and the selection score need not be exactly identical.

**计算细节：** 对正确数 c、错误数 e 和预测概率 p，先累计 `c log(p) + e log(1-p)`，再除以累计单词数。这里省略了二项式组合常数。图中的中心值对三个折的单词平均分数等权平均；真正选择候选模型时则先加总所有折的 loss，再除以所有折的单词数。折的大小不完全相同时，这两个平均值不一定完全相同。

| Figure stem | Selected HuBERT layer by pooled held-out predictor-only loss | Pooled log likelihood per word | Page-specific spoken caution |
|---|---|---:|---|
| `sbi_an19_base_loglik` | Tr-10 | -0.619245 | AN19 uses 120 participants and 5,685 word responses. Talker random-effect fallbacks differ across layer fits, so this is not yet a fixed-structure layer comparison. / AN19 有 120 名参与者和 5,685 个单词反应；部分层发生说话人随机效应结构回退，因此还不是固定结构的层间比较。 |
| `sbi_an19_ft_loglik` | Tr-10 | -0.636033 | The same caveat applies; the retained base winner scores higher than the fine-tuned winner. / 同样需要注意结构回退；这批保留结果里 base 的获选模型分数高于微调版本。 |
| `sbi_x21_base_loglik` | Tr-14 | -0.482788 | X21 uses 320 participants and 16,477 word responses, including control-test rows, with a fixed test-talker block. / X21 包括 320 名参与者和 16,477 个单词反应，也包括 control 的测试反应，并使用说话人固定区组。 |
| `sbi_x21_ft_loglik` | Tr-22 | -0.483093 | The selected score is close to the base score; selection used the same study, not an independent validation sample. / 获选分数接近 base；模型选择来自同一个研究，而非独立验证样本。 |
| `sbi_b23_base_loglik` | Tr-24 | -0.568780 | B23 has 168 participants, 10,080 count-binomial rows, and 54,096 scored words. / B23 有 168 名参与者、10,080 行二项计数反应，累计评分单词数为 54,096。 |
| `sbi_b23_ft_loglik` | Tr-20 | -0.569008 | Differences among B23 layers are very small on this scale; the y-axis is still per word, not per sentence. / B23 各层在这个尺度上的差异很小；纵轴仍然是每个单词，而不是每个句子。 |

The missing likelihood ceiling was not silently replaced by the notebook z reference. A shared item-by-condition prediction reference with matching response rows and procedure remains to be completed.

## Revised HVE z: shared narration

Applies to all 11 `hve_revised_{dataset}_{nn}` figures in `analysis/model_comparison/reference_checks/z_value_review/figures`.

**English.** Here I switch from similarity to variability in the exposure speech itself. Each small panel is a different definition of HVE, and the horizontal axis is the same set of 18 HuBERT layers. The vertical axis is the signed Wald z of the variability coefficient: above zero means that more variability is associated with better recognition, and below zero means the reverse. Blue is HuBERT base and red is ASR-fine-tuned. The small gray points are the three training fits, each using two participant folds; the colored point and interval summarize their mean and 95% fold-bootstrap interval. These are predictor-only fits, without condition. Unlike the historical SBI figure, these are training-fold coefficients from the revised exposure analysis. I have therefore not divided them by the old test-refit ceiling or added a 100% band. The dotted lines are only nominal plus or minus 1.96 references; they are not corrected significance tests across all these definitions and layers. The sign matters: a large negative z is not evidence for the predicted positive effect of variability.

**中文。** 这里从 similarity 转到 exposure 语音本身的 variability，也就是 HVE。每个小面板是一种 HVE 定义，横轴仍然是 18 个 HuBERT 层，纵轴是 variability 回归系数带正负号的 Wald z。正值表示 variability 越大，识别越好；负值表示相反。蓝色是 base，红色是 ASR 微调版本。灰色小点来自三次训练拟合，每次使用其中两折参与者；彩色点和误差棒表示均值及 95% 折间 bootstrap 区间。这里拟合的是不包含 condition 的 predictor-only 模型。与历史 SBI 图不同，这些是修订后 exposure 分析的训练折系数，因此我没有拿旧的测试重拟合 ceiling 来归一化，也没有加 100% 灰带。正负 1.96 的虚线只是名义参考，并不是对全部层和全部定义进行校正后的显著性判断。尤其要保留正负方向：很大的负 z 并不支持 variability 的预期正效应。

## Revised HVE held-out likelihood: shared narration

Applies to all 11 `hve_{dataset}_loglik_{nn}` figures in `analysis/speech/figures`.

**English.** These panels use the same revised HVE definitions but show their held-out prediction scores instead of their coefficient z values. Again, the horizontal axis is HuBERT layer, and blue and red distinguish the two model versions. The vertical axis is log likelihood per word from the predictor-only model, with higher values indicating better held-out prediction. I fit on two participant folds, keep the model fixed, and score the third fold. The gray points show the three held-out folds, and the error bars summarize their 95% fold-bootstrap interval. Unlike z, this score does not tell us whether the variability association is positive or negative, so I use the z panels to explain its direction. A favorable score can reflect a negative variability association. I also keep B23's global-order sample separate: it contains 97 participants, compared with 168 for the other definitions. The scores should not be ranked across those different samples.

**中文。** 这些面板使用同一批修订后的 HVE 定义，但是展示 held-out prediction 分数，而不是回归系数的 z。横轴是 HuBERT 层，蓝红两色区分模型版本；纵轴是 predictor-only 模型的每单词 log likelihood，越高表示测试预测越好。我在两折参与者上拟合后冻结模型，再评价第三折。灰点是三个测试折，误差棒是它们的 95% 折间 bootstrap 区间。与 z 不同，这个分数不反映关联是正还是负，所以方向需要结合 z 图解释：较好的分数也可能来自负向的 variability 关联。此外，B23 的 global-order 分析只有 97 人，其他定义有 168 人，不能跨这两个不同样本直接排名。

### HVE measure explanations: reusable oral sentences

At tau = 2, the current non-DTW dispersion functions use squared Euclidean deviations without a final square root. This describes dispersion, not mean Euclidean distance. Within-type DTW, by contrast, uses rooted Euclidean local distances and the original mean-sequence-length normalization. These operations should not be conflated in speech or captions.

在 tau = 2 时，目前非 DTW 的 dispersion 函数使用平方欧氏偏差，最终不取平方根。因此它是 dispersion，而不是平均欧氏距离。相反，within-type DTW 的局部距离是取过根号的欧氏距离，并使用原来的平均序列长度归一化。这两个计算不能混为一谈。

| Measure family | Natural English explanation | 中文对应 |
|---|---|---|
| `overall` | I pool all frames from the exposure recordings, find their common center, and average their squared deviations from that center. Longer recordings contribute more frames. | 我合并 exposure 录音的全部 frame，求共同中心，再平均每个 frame 到中心的平方偏差。较长的录音包含更多 frame，因此权重也更高。 |
| `overall_order_sensitive` | I join complete exposure recordings in their actual presentation order, including the transition from the end of one recording to the beginning of the next, and average squared adjacent-frame changes. I do not insert an unobserved inter-trial pause. | 我按真实 exposure 顺序连接完整录音，包括前一段末尾到下一段开头的过渡，再平均相邻 frame 的平方变化；没有额外插入未观测的试次间停顿。 |
| `within_token_*` | For each sentence, word, or phone instance, I measure how its frames spread around its own center, then average across instances. | 对每个句子、单词或音素实例，先计算内部 frame 围绕自身中心的分散程度，再对实例等权平均。 |
| `within_type_*` | I first turn each instance into its mean feature vector, measure how those vectors vary among instances of the same type, and then average across types that have at least two instances. | 我先把每个实例变成平均特征向量，计算同类型不同实例之间的变化，再对至少有两个实例的类型求平均。 |
| `between_type_*` | I find the mean vector for each type and measure the spread among those type means, giving each type equal weight. | 我先求每个类型的平均向量，再计算这些类型均值之间的分散程度，每个类型权重相同。 |
| `order_*` | I average squared adjacent-frame changes inside each instance and then average over instances with at least two frames. This does not include transitions between instances and therefore does not require their trial order. | 我在每个实例内部平均相邻 frame 的平方变化，再对至少包含两个 frame 的实例求平均。这不包括实例之间的边界，因此不需要试次顺序。 |
| `mean_dissimilarity_*` | Within each type, I compare all pairs of instances with DTW, average their distances, and then average across types that have a pair. Here I compare whole trajectories rather than their centers. | 对同一类型内部的所有实例对计算 DTW，先在类型内平均，再对有配对的类型平均。这里比较完整轨迹，而不是只比较中心。 |

### HVE dataset context

**AN19, English.** The current revised HVE results cover seven definitions: the two overall definitions and five word-level definitions. They use 120 exposure participants and 5,760 test-word responses. The later automatic phone alignment has not been fed back into these HVE fits, so its availability does not mean the AN19 phone-level HVE panels have already been calculated. There are no sentence tokens in this isolated-word corpus.

**AN19，中文。** 目前修订后的 HVE 包括七种定义：两种 overall 定义和五种 word-level 定义，使用 120 名 exposure 参与者及 5,760 个测试单词反应。后来生成的自动音素对齐还没有重新接入这批 HVE 拟合，所以有了音素边界不代表 AN19 音素级 HVE 已经算完。这个语料是孤立单词，也没有句子 token。

**X21, English.** X21 has all 17 definitions, including global exposure order, and uses the same 320 participants and 16,477 responses as the revised SBI analysis. This includes control-test responses; the table's inherited label “all exposure participants” should not be read as excluding those controls. The selected CNN-6 within-word transition predictor has a negative training-fold association, with mean z about minus 4.44. Its good likelihood score therefore does not show that more variability helped.

**X21，中文。** X21 有全部 17 种定义，包括 global exposure order，样本与修订后的 SBI 一致，为 320 人、16,477 个反应，也包括 control 的测试反应。来源表沿用的 “all exposure participants” 不能理解成排除了 control。选中的 CNN-6 单词内部 transition 指标平均训练折 z 约为负 4.44，因此它的 likelihood 较好不代表 variability 越大越有帮助。

**B23, English.** B23 has 14 order-independent definitions for all 168 exposed participants and one global-order definition for the 97 participants whose order can be recovered. Within-sentence-type dispersion and within-sentence-type DTW are missing because these exposure pools do not have repeated instances of a sentence type. These are missing estimates, not zeros. The full sample contains 10,080 response rows and 54,096 scored words; the ordered subset has 5,820 rows and 31,234 scored words.

**B23，中文。** B23 对全部 168 名 exposure 参与者有 14 种不依赖试次顺序的定义，另对能恢复顺序的 97 人计算 global-order。句子类型内部 dispersion 和 DTW 缺失，是因为这些 exposure pools 没有同一句子类型的重复实例；它们是缺失估计，而不是零。全部样本有 10,080 行反应、54,096 个评分单词；顺序子样本有 5,820 行、31,234 个单词。

### Exact HVE page contents

Panel order is top left, top right, bottom left, bottom right; fewer than four methods leaves the remaining panels unused. Importantly, the existing z and likelihood scripts use different ordering, so equal page suffixes are **not** matched method sets. Use the method titles, not the page suffix, to pair them.

| z figure stem | Methods in panel order |
|---|---|
| `hve_revised_an19_01` | `overall`; `overall_order_sensitive`; `within_token_word`; `within_type_word` |
| `hve_revised_an19_02` | `between_type_word`; `order_word`; `mean_dissimilarity_word` |
| `hve_revised_x21_01` | `overall`; `overall_order_sensitive`; `within_token_sentence`; `within_type_sentence` |
| `hve_revised_x21_02` | `between_type_sentence`; `order_sentence`; `mean_dissimilarity_sentence`; `within_token_word` |
| `hve_revised_x21_03` | `within_type_word`; `between_type_word`; `order_word`; `mean_dissimilarity_word` |
| `hve_revised_x21_04` | `within_token_phoneme`; `within_type_phoneme`; `between_type_phoneme`; `order_phoneme` |
| `hve_revised_x21_05` | `mean_dissimilarity_phoneme` |
| `hve_revised_b23_01` | `overall`; `overall_order_sensitive`; `within_token_sentence`; `between_type_sentence` |
| `hve_revised_b23_02` | `order_sentence`; `within_token_word`; `within_type_word`; `between_type_word` |
| `hve_revised_b23_03` | `order_word`; `mean_dissimilarity_word`; `within_token_phoneme`; `within_type_phoneme` |
| `hve_revised_b23_04` | `between_type_phoneme`; `order_phoneme`; `mean_dissimilarity_phoneme` |

| Likelihood figure stem | Methods in panel order |
|---|---|
| `hve_an19_loglik_01` | `between_type_word`; `mean_dissimilarity_word`; `order_word`; `overall` |
| `hve_an19_loglik_02` | `overall_order_sensitive`; `within_token_word`; `within_type_word` |
| `hve_x21_loglik_01` | `between_type_phoneme`; `between_type_sentence`; `between_type_word`; `mean_dissimilarity_phoneme` |
| `hve_x21_loglik_02` | `mean_dissimilarity_sentence`; `mean_dissimilarity_word`; `order_phoneme`; `order_sentence` |
| `hve_x21_loglik_03` | `order_word`; `overall`; `overall_order_sensitive`; `within_token_phoneme` |
| `hve_x21_loglik_04` | `within_token_sentence`; `within_token_word`; `within_type_phoneme`; `within_type_sentence` |
| `hve_x21_loglik_05` | `within_type_word` |
| `hve_b23_loglik_01` | `between_type_phoneme`; `between_type_sentence`; `between_type_word`; `mean_dissimilarity_phoneme` |
| `hve_b23_loglik_02` | `mean_dissimilarity_word`; `order_phoneme`; `order_sentence`; `order_word` |
| `hve_b23_loglik_03` | `overall`; `overall_order_sensitive`; `within_token_phoneme`; `within_token_sentence` |
| `hve_b23_loglik_04` | `within_token_word`; `within_type_phoneme`; `within_type_word` |

## Combined-data nested-model comparison

Available figure: `analysis/model_comparison/pooled_lrt/figures/figure_01_crossfitted_nested_lrt`. Retained September 9 report uses the exact X21 table in place of this dense figure; the figure is optional rather than mandatory. Do not confuse it with an OOF-gain figure or a z profile.

**English.** This figure asks whether the theoretical predictor and experimental condition each contribute information that the other does not capture. The left column compares the condition-only model with the joint model, so it tests the predictor beyond condition. The right column compares the predictor-only model with the same joint model, so it tests condition beyond the predictor. Each point is one selected SBI or HVE specification, not a fold. Blue marks SBI, green marks HVE, circles are base, and squares are ASR-fine-tuned. The horizontal axis is minus log ten of the likelihood-ratio-test p-value, so farther right means a smaller p-value; the dashed line marks p equals .05. I first standardize each held-out partition using moments from the other two folds. I then combine the partitions, fit the three GLMMs once, and compare their likelihoods. This implements the combined-data option, not an average of fitted model parameters. The test statistic is twice the improvement in log likelihood, evaluated against a chi-square distribution. Because I selected the layer or HVE definition using the same study, these are exploratory, selection-conditional p-values, not an independent confirmation or a held-out prediction score.

**中文。** 这张图检验理论预测变量和实验 condition 是否各自包含对方没有解释的信息。左列比较 condition-only 和 joint model，因此问的是 predictor 在 condition 之外还有没有贡献；右列比较 predictor-only 和同一个 joint model，因此问的是 condition 在 predictor 之外还有没有贡献。每个点是一种获选 SBI 或 HVE 配置，而不是一折。蓝色为 SBI，绿色为 HVE；圆形表示 base，方形表示 ASR 微调。横轴是 likelihood-ratio-test p 值的负十进制对数，越往右 p 越小，虚线表示 p=.05。我先用另外两折的均值和标准差标准化当前折，然后合并三个分区，重新拟合三个 GLMM 并比较 likelihood。这是 combined-data 方案，不是把模型参数直接平均。统计量是 log likelihood 改善量的两倍，并用卡方分布评价。由于层和 HVE 定义也在同一研究内选择，这些 p 值是探索性、以选择结果为条件的检验，不是独立确认，也不是 held-out prediction 分数。

**X21 result to say aloud.** In X21 the response rows and model structures match, which makes the comparison easier to interpret. For the selected base SBI at Tr-14, the predictor adds beyond condition at p=.01068, and condition adds beyond the predictor at p=.000105. For CNN-6 within-word transition HVE, the corresponding p-values are .05318 and .01168. So it would be wrong to say that condition does not matter for HVE. Its predictor-only coefficient is strongly negative, z=-5.5038, and becomes -1.9516 after condition is included. HVE and SBI need not show equal incremental tests, because their predictor-only models explain different parts of the data. The selected HVE base and fine-tuned inputs and results here are identical, not two independent replications.

**可以口头说的 X21 结果。** X21 的反应行和模型结构匹配，因此更适合直接解释。获选的 base Tr-14 SBI 在 condition 之外的贡献 p=.01068，condition 在 SBI 之外的贡献 p=.000105；CNN-6 单词内部 transition HVE 对应的两个 p 值分别为 .05318 和 .01168。因此，不能说 condition 对 HVE 没有额外作用。HVE 在 predictor-only 模型中的系数明显为负，z=-5.5038；加入 condition 后为 -1.9516。两个 predictor-only 模型原本解释的数据部分不同，所以它们的增量检验不需要相等。这里获选 HVE 的 base 与微调输入及结果完全相同，也不是两次独立复现。

**Other-dataset cautions.** In AN19 the selected SBI models use 5,685 responses and omit the talker random intercept after fitting fallback; HVE uses 5,760 responses and retains it. Each within-family nested test uses matched models, but raw SBI-versus-HVE likelihoods are not directly comparable across those samples and structures. In B23, the full and ordered HVE strata remain separate. The combined test also contains an outer-fold fixed block; this is absent from the ordinary predictor-only fold-profile fits. The full comparison has 28 LRTs from 42 converged, non-singular GLMMs. These are nominal p-values with no correction for the predictor search or the set of comparisons.

### Selected configurations and exact combined-data p-values

The columns below are `predictor beyond condition` and `condition beyond predictor`, respectively. They come from the September 1 combined-data LRT table, not the August 27 ordinary full-data LRTs or the frozen-model gain tables.

| Dataset and family | Variant | Selected feature | Predictor beyond condition p | Condition beyond predictor p |
|---|---|---|---:|---:|
| AN19 SBI | base | Tr-10 | .00003383 | .002321 |
| AN19 SBI | FT | Tr-10 | .002199 | .005265 |
| AN19 HVE | base and FT, identical result | CNN-3 global exposure order | .15100 | .002017 |
| X21 SBI | base | Tr-14 | .010681 | .0001051 |
| X21 SBI | FT | Tr-22 | .028371 | .000004119 |
| X21 HVE | base and FT, identical result | CNN-6 within-word adjacent frames | .053179 | .011680 |
| B23 SBI | base | Tr-24 | .068712 | .444829 |
| B23 SBI | FT | Tr-20 | .067085 | .441966 |
| B23 HVE, 168 participants | base | Tr-12 between sentence types | .007884 | .319414 |
| B23 HVE, 168 participants | FT | Tr-4 between sentence types | .013505 | .481413 |
| B23 HVE, 97 participants | base | Tr-6 global exposure order | .013269 | .175761 |
| B23 HVE, 97 participants | FT | Tr-20 global exposure order | .042836 | .200212 |

## X21 condition-specific curves

Figure: `x21_s_curves_by_condition` in `analysis/speech/figures`.

**English.** This figure brings the model values back to listener accuracy. Each panel is one Mandarin-English test talker, with talker 035 in the large panel and 032, 043 and 037 in the smaller panels. The horizontal axis is exposure-to-test similarity from HuBERT base Tr-24 after 3-D t-SNE; the vertical axis is the proportion of correctly recognized words. Gray is control, green is multi-talker, blue is single-talker, and red is talker-specific exposure. I summarize the trial responses in ten similarity bins per condition and fit a separate ordinary binomial logistic curve for each condition and test talker. The similarity values are exp of minus k times the retained DTW distance, with k about .358 and the historical coordinate scaling. These are descriptive curves rather than mixed-model held-out predictions. The plot retains its original intervals, but their source record does not establish the resampling unit, so I cannot call them participant-bootstrap or three-fold intervals. A vertical gap between curves is descriptive; the nested models provide the test of condition beyond similarity.

**中文。** 这张图把模型值和听者正确率直接放在一起。每个面板是一位说普通话背景英语的测试说话人，035 是大图，032、043 和 037 是旁边的小图。横轴是 HuBERT base Tr-24 经三维 t-SNE 后得到的 exposure-to-test similarity，纵轴是单词识别正确率。灰色是 control，绿色是 multi-talker，蓝色是 single-talker，红色是 talker-specific。每个 condition 用十个 similarity 分箱汇总反应，并在每个 condition 和测试说话人内单独拟合普通二项 logistic 曲线。similarity 是 exp(-k×DTW)，这里 k 约为 .358，使用历史坐标缩放。这些是描述曲线，不是混合模型的 held-out prediction。图中保留了原区间，但来源记录没有说明重采样单位，所以不能称为参与者 bootstrap 或三折区间。曲线之间的高度差是描述，condition 是否在 similarity 之外有贡献应由嵌套模型检验。

**Useful exact context:** there are 16,477 trial rows from 320 participants, 20 participants in each test-talker-by-condition cell. Talker 035 has 4,117 rows; each other talker has 4,120. The four mean accuracies for 035 are .8043 control, .8621 multi-talker, .8680 single-talker and .8709 talker-specific. The transformation is `exp(-0.35784438264009538 * raw_distance)` with `legacy_per_dimension_z` feature-coordinate scaling. It uses a cross-fold mean k from the legacy X21 run and is not free of its parameter-selection dependence.

## X21 pooled curves

Figure: `x21_s_curves_pooled` in `analysis/speech/figures`.

**English.** This is the same X21 data, but now I combine the conditions within each test talker. The axes and similarity calculation are unchanged. Each black point summarizes trial accuracy in one of twenty similarity bins, and the black line is one logistic regression fitted to the pooled trial rows. I did not average the four colored curves from the previous figure. The overall accuracy is about .851 for talker 035, .818 for 032, .838 for 043 and .841 for 037. This gives a compact view of the overall pattern, but pooling does not control for condition and can mix within-condition and between-condition differences. It is therefore a descriptive summary, not a substitute for the conditional analysis or the GLMM comparison.

**中文。** 这还是同一批 X21 数据，但我现在在每位测试说话人内部合并所有 condition。横纵轴和 similarity 的算法不变。每个黑点代表二十个 similarity 分箱中一个箱的反应正确率，黑线是对合并后的全部 trial rows 拟合的一条 logistic 曲线，并不是把前一张的四条彩色曲线平均。四位说话人的整体正确率分别约为 .851、.818、.838 和 .841。这能简洁展示总体趋势，但合并并不等于控制 condition，会混合 condition 内部与 condition 之间的差异，因此不能替代分条件分析或 GLMM 比较。

## Evidence map for assembly and verification

| Question | Local source |
|---|---|
| Figure points, 27-resample fold intervals, z normalization, HVE z source scopes | `scripts/build_z_value_review.py` |
| September 9 likelihood and SBI z redraws | `scripts/build_september9_statistical_panels.py` |
| Retained SBI z values | `analysis/speech/tables/sbi_retained_fold_z.csv` |
| Historical ceiling values | `analysis/speech/tables/shared_historical_ceiling_z.csv` |
| Revised HVE signed z, 7/17/15 definitions, fit diagnostics | `analysis/model_comparison/reference_checks/z_value_review/tables/revised_hve_training_fold_z.csv` and `revised_hve_coverage.csv` |
| Frozen predictor-only likelihood values | `analysis/speech/tables/sbi_retained_fold_loglik.csv` and `hve_retained_fold_loglik.csv` |
| Revised HVE formulas and root convention | `src/ctg/metrics.py`, `src/ctg/exposure.py` |
| Random-effect structure, training-only predictor scaling, `re.form=NA` scoring | `R/fit_confirmatory.R` |
| Historical SBI structures, k fitting, X21 leakage, sample/ceiling mismatch | `results/statistics/{dataset}-{variant}-original-structure-z-*/run_metadata.json` |
| Historical ceiling construction and refitting | `src/ctg/ceiling.py`, `R/fit_ceiling_compatibility.R` |
| Selected candidate identity and loss criterion | `analysis/model_comparison/selection/tables/sbi_predictor_only_selected_layers.csv`, `hve_predictor_only_selected_methods.csv` |
| Combined-data test implementation and exact results | `R/fit_crossfitted_lrt.R`, `analysis/model_comparison/pooled_lrt/tables/crossfitted_lrt_results.csv` and `crossfitted_model_diagnostics.csv` |
| X21 curve formula, k, populations, CI provenance limitation | `analysis/speech/tables/x21_s_curve_revision_provenance.json` |

No figure-specific bootstrap here estimates uncertainty in the HuBERT representation, the t-SNE fit, the layer selection, or the historical k selection. The three-fold intervals summarize only the supplied fold statistics; because the training sets overlap and there are only three folds, they should be read as descriptive fold variation rather than a comprehensive uncertainty estimate for the study.
