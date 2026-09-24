# Cross-talker generalization: bilingual speaking script

Aligned with the 14 slides in [re_florian_fig_update.pptx](re_florian_fig_update.pptx). Read the English passages for the presentation; use the Chinese passages as a matching reference. Allow approximately 15 minutes, including brief pauses for the figures.

对应 PPT 的 14 页。英文部分可直接用于讲述，中文逐页对照。加上指图和短暂停顿，预计约 15 分钟。

## 01 | Cross-talker generalization

### English

Hi everyone. I'd like to walk through the updated figures and explain how the different analyses connect. The main question is how the speech people hear during exposure relates to their understanding of a new talker. I'll start with the speech representations, then show the pronunciation comparisons, and finish with the links to listener responses. Along the way, I'll explain what each point represents and how the values are calculated.

### 中文对照

大家好，今天我想结合更新后的图，说明这些分析是怎样联系起来的。我们的主要问题是：听者在 exposure 阶段接触到的语音，与他们之后理解新说话人的表现有什么关系。我会先介绍语音表示，再展示发音之间的比较，最后讲它们与听者反应的关系。过程中也会说明每个点代表什么，以及图上的数值是怎样算出来的。

## 02 | Speech as a latent trajectory

### English

This example shows an English speaker saying, “The wife helped her husband.” On the left, the waveform shows amplitude over time. I pass the recording through HuBERT and extract its frame-by-frame representation. Here we're looking at transformer layer 24, which gives 1,024 values for each frame. The two traces show the first and last dimensions, with the dots indicating the dimensions in between. I then use the existing three-dimensional t-SNE representation. Its x, y and z axes describe positions in that reduced space, and connecting consecutive frames gives the speech trajectory. On the right, “wife” is enlarged, with its three phone intervals shown in different colors. The enlarged view uses the same coordinates as the sentence view.

### 中文对照

这个例子是一位英语母语说话人在说“The wife helped her husband”。左边的波形展示振幅随时间的变化。我把录音输入 HuBERT，提取逐帧的潜在表示。这里使用第 24 个 Transformer 层，每一帧有 1,024 个数值。中间的两条线分别展示第一个和最后一个维度，省略号表示中间的其他维度。接着，我使用已有的三维 t-SNE 表示。x、y、z 轴表示降维空间中的三个坐标，把相邻帧连接起来，就得到语音的轨迹。右边放大了“wife”这个词，并用不同颜色标记三个音素区间。放大图沿用整句图中的同一套坐标。

## 03 | Language backgrounds in the data

### English

This map introduces the language backgrounds represented in our speech corpora. The labels identify the speakers' first languages, and the shared colors follow the language groups in Florian's map. The geographical markers use associated capital cities. I've carried the same language colors into the later figures so that we can follow each group across analyses. This gives us the background for comparing pronunciation patterns across different first languages.

### 中文对照

这张地图介绍语料中包含的母语背景。标签表示说话人的第一语言，相同颜色沿用 Florian 地图里的语言分组。地理标记使用相应国家的首都位置。我在后面的图中也沿用这套语言配色，方便大家在不同分析之间识别同一个语言组。接下来，我们就在这些母语背景下比较英语发音的模式。

## 04 | How similar are the sound inventories?

### English

Here I use PHOIBLE to compare consonant and vowel inventories with English. Each row is a language, and moving right means greater overlap. The calculation is Jaccard similarity: the number of shared phoneme symbols divided by the number of distinct symbols across the two inventories. A language can have several inventory sources, so I compare the selected inventories against nine English inventories. Each point is the average across those comparisons. The horizontal line shows the minimum-to-maximum range across inventory sources. This gives us a language-level reference based on consonant and vowel symbols.

### 中文对照

这里我用 PHOIBLE 比较各语言的辅音和元音库存与英语有多相似。每一行是一种语言，越靠右表示重合程度越高。计算使用 Jaccard similarity，也就是两套库存共有的音位符号数，除以它们合起来包含的不同符号总数。同一种语言可能有多个库存来源，所以我把所选库存与九个英语库存逐一比较。点表示这些比较的平均值，横线表示不同库存来源之间的最小值到最大值。这提供了一个基于辅音和元音符号的语言层面参考。

## 05 | Which phonological properties differ?

### English

This is the feature comparison Florian shared. The columns are languages, and the rows describe particular properties of their sound systems, including segments, phonotactics, stress and intonation. Greener cells indicate greater similarity to English, while redder cells indicate lower similarity. The figure gives a more detailed view of the properties that may contribute to language differences. I'm using the supplied scores here; their exact numerical construction remains a point to document with the original score table and code.

### 中文对照

这是 Florian 分享的具体音系特征对比图。每一列是一种语言，每一行描述它们声音系统中的某个属性，包括音段、音系配列、重音和语调。越绿表示与英语越相似，越红表示相似程度越低。这张图更具体地展示了哪些属性可能构成语言之间的差异。这里沿用他提供的分数，具体的数值构造方式还需要结合原始评分表和代码补充记录。

## 06 | Comparing the same words across talkers

### English

Now we move to the recordings. Both matrices contain the same 42 AN19 talkers, including six English reference talkers. Each cell compares a pair of speakers using the same 138 words. For each matched word, I calculate DTW between their three-dimensional HuBERT trajectories. DTW aligns corresponding parts across different speaking rates. I use Euclidean local distance and divide the accumulated cost by the mean frame count of the two sequences. The left matrix averages the resulting distances. For the right matrix, I first convert each recording-pair distance to exp of minus distance, then average within and across words. Both color scales are linear. English comes first, the other language groups follow their average distance to English, and the gray boxes show the language groups.

### 中文对照

接下来比较具体录音。两个矩阵都包含同一批 42 位 AN19 说话人，其中六位是英语母语参考。每个格子比较两位说话人在相同 138 个词上的发音。对于每个匹配的词，我计算两条三维 HuBERT 轨迹之间的 DTW。DTW 可以对齐语速有所差异的对应部分。这里的局部距离是欧氏距离，累积代价除以两条序列平均的帧数。左图汇总这些距离。右图先把每一对录音的距离转换成 exp(-distance)，再在词内和词间取平均。两个色标都是线性的。英语组排在最前，其他语言组按与英语的平均距离排列，灰色框标记同一语言组。

## 07 | Average similarity across segment instances

### English

This is the revised segment summary. Each small dot represents one L2 talker, each open diamond is a language-group mean, and the rows run from higher to lower mean similarity. To calculate a dot, I use FALCON forced alignment to locate the intended phones in each word, then extract their frames from the existing HuBERT representation. Each segment is matched to the same word and phone position from six English speakers. I convert the pairwise DTW distances to similarity, average the recordings within each English reference speaker, and then average across the six references. Finally, every retained segment instance contributes equally to that L2 talker's mean. This gives 5,114 usable instances across 36 L2 talkers. The confidence intervals come from 1,000 talker-level bootstrap samples within each language group. Korean and Spanish have multiple talkers and group intervals; the other groups contribute individual-talker estimates. The analysis describes automatically aligned intended-phone intervals, with coverage determined by the available frames and reference matches.

### 中文对照

这是更新后的音段汇总图。每个小点代表一位二语说话人，空心菱形是语言组均值，语言组按平均相似度从高到低排列。计算一个点时，我先用 FALCON 强制对齐定位每个词中预期音素的时间区间，再从已有的 HuBERT 表示中取出相应的帧。每个音段都匹配六位英语说话人在同一个词、同一个音素位置的实例。我先把成对的 DTW 距离转换成相似度，在每位英语参考说话人内部平均录音，再平均六位参考说话人。最后，这位二语说话人的每个有效音段实例都等权进入他的平均值。最终保留了 36 位二语说话人的 5,114 个有效实例。置信区间来自各语言组内的 1,000 次说话人层面 bootstrap。韩语组和西班牙语组有多位说话人，因此可以估计组均值区间；其他语言组展示单个说话人的估计值。这个分析描述自动对齐得到的预期音素区间，覆盖范围取决于可用的帧和参考匹配。

## 08 | Similarity to English and listener accuracy

### English

Here we connect similarity to human responses. The horizontal axis is a test word's similarity to English reference productions, and the vertical axis is listener accuracy in the control-condition test. AN19 is on the left and X21 is on the right. Each point summarizes a similarity quantile bin, and the curves are descriptive logistic fits. Their uncertainty intervals come from resampling listeners. The small histograms show the distribution of test-word similarities, counting each target once. For these plots, I average the English-reference distances, scale by a dataset-specific median distance, and apply the exponential transformation. Interpretation therefore uses each plot's own similarity scale. The upward curves suggest that words closer to the English reference are generally easier for these listeners to understand.

### 中文对照

这里把相似度与人类反应联系起来。横轴是测试词与英语参考发音的相似度，纵轴是控制条件测试中的听者正确率。左边是 AN19，右边是 X21。每个点汇总一个相似度分位区间，曲线来自描述性的 logistic 拟合，区间通过对听者重抽样得到。下面的小直方图展示测试词的相似度分布，每个测试目标只计一次。这两张图的计算先平均英语参考距离，再用数据集内部的中位距离缩放，最后进行指数转换，因此解释时使用各图自身的相似度尺度。曲线整体向上，说明在这些数据中，更接近英语参考发音的词通常更容易被听懂。

## 09 | SBI across layers and datasets

### English

We now turn to similarity-based inference, or SBI, which relates exposure-to-test similarity to behavior. Each panel is a dataset. MFCC and STRF appear first on the horizontal axis, followed by the HuBERT layers. Blue is pretrained HuBERT, and red is the ASR-fine-tuned model. The vertical axis divides the predictor's z-value by the dataset's mean ceiling z and expresses that ratio as a percentage. Each dataset has its own ceiling, shared by the two model versions. Gray dots are the three fold estimates, and the colored bars give the 95 percent fold-bootstrap intervals. The dashed line marks 100 percent, the gray band shows ceiling uncertainty, and the orange lines mark the rescaled nominal z thresholds of plus or minus 1.96. AN19 shows a stronger layer pattern, X21 is relatively stable, and B23 has greater uncertainty. These values come from historical test-fold GLMM refits. X21 also uses k averaged across folds, so parameter selection draws on information across the folds. AN19 and B23 use ceilings estimated from broader participant samples. I therefore use this figure as a descriptive layer comparison, with matched-sample ceilings remaining a follow-up task.

### 中文对照

接下来是 similarity-based inference，也就是 SBI，它把 exposure 与 test 之间的语音相似度同听者行为联系起来。每个面板对应一个数据集。横轴先展示 MFCC 和 STRF，再展示 HuBERT 各层。蓝色是预训练 HuBERT，红色是经过 ASR 微调的模型。纵轴用 predictor 的 z-value 除以该数据集平均的 ceiling z，再转换成百分比。每个数据集有自己的 ceiling，两个模型版本在这个数据集内部共用它。灰点是三个折的估计，彩色误差棒是 95% 折间 bootstrap 区间。虚线表示 100%，灰带展示 ceiling 的不确定性，橙线表示换算后的名义 z 阈值，也就是正负 1.96。AN19 的层间变化更明显，X21 相对平稳，B23 的不确定性更大。这些数值来自历史版本在 test fold 上重新拟合的 GLMM。X21 还使用了跨折平均的 k，所以参数选择用到了各折的信息。AN19 和 B23 的 ceiling 来自更广的参与者样本。因此，我把这张图用于描述层间模式，样本匹配的 ceiling 仍是后续工作之一。

## 10 | What does exposure variability capture?

### English

HVE looks at variability within the exposure speech itself. Here I show four definitions in X21. The first measures how all frames spread around the exposure pool's mean. The second measures adjacent-frame changes across the actual exposure sequence, including transitions between recordings. The third averages the frame dispersion within each sentence recording. The fourth measures dispersion among the mean vectors of repeated instances of the same sentence type. These measures use squared deviations or squared changes at tau equal to two. On the horizontal axis we have HuBERT layers. The vertical axis is the HVE coefficient divided by its standard error, giving a Wald z-value. Each fold estimate comes from a predictor-only GLMM fitted on two training folds. Positive and negative values show the direction of the association. The changing profiles show why the choice of variability definition and representation layer matters.

### 中文对照

HVE 关注 exposure 语音集合内部的变异程度。这里用 X21 展示四种定义。第一种计算所有帧围绕 exposure 集合均值的分散程度。第二种计算真实 exposure 顺序中的相邻帧变化，也包含不同录音之间的过渡。第三种先计算每个句子录音内部的帧离散度，再取平均。第四种计算同一句子类型的重复实例中，各实例平均向量之间的离散度。这些指标在 tau 等于 2 时使用平方偏差或平方变化。横轴是 HuBERT 层，纵轴是 HVE 系数除以标准误得到的 Wald z-value。每个折的估计来自在两个训练折上拟合的 predictor-only GLMM。正负值表示拟合关系的方向。各条曲线随层变化的模式，也说明 variability 的定义和表示层都会影响结果。

## 11 | HVE prediction on held-out listeners

### English

This page evaluates the same four definitions on held-out listeners. For each split, I standardize the predictor using the training data, fit the predictor-only GLMM on two folds, and use that fitted model to score the third fold. Predictions use the fixed effects, with random-effect contributions set to zero. The score sums the log probabilities assigned to the observed responses and divides by the number of word trials. Higher values on the vertical axis mean better held-out prediction. The dots and intervals summarize the three folds. Here the mean profiles are close and the intervals overlap substantially, so differences between the displayed candidates are small relative to fold variation. These scores evaluate individual candidates; evaluating the complete layer-selection procedure would require an additional independent evaluation step.

### 中文对照

这一页用留出听者评价相同的四种定义。每次划分先用训练数据标准化 predictor，在两个折上拟合 predictor-only GLMM，再用拟合好的模型给第三折评分。预测使用固定效应，随机效应贡献设为零。评分时，把模型赋予实际反应的概率取对数、求和，再除以词试次数。纵轴数值越高，表示留出预测越好。点和区间汇总三个折的结果。这里几条均值曲线很接近，区间也有较大重叠，说明这些候选项之间的差异相对于折间变化较小。这些分数评价各个候选项；要评价完整的选层过程，还需要一个额外的独立评价步骤。

## 12 | Similarity within experimental conditions

### English

These curves make the X21 relationship easier to see. Each panel is one test talker, with talker 035 in the large panel. The horizontal axis is exposure-to-test similarity from HuBERT Tr-24, and the vertical axis is word recognition accuracy. Gray indicates control, green multi-talker exposure, blue single-talker exposure, and red talker-specific exposure. I calculate similarity as exp of minus k times DTW, with k around 0.358 for these retained curves. Within each talker and condition, I summarize responses in ten bins and fit a logistic curve. This lets us see how the similarity-accuracy relationship varies across conditions. The shaded regions retain the original uncertainty estimates; their resampling details remain to be verified. The later model comparison formally evaluates the additional contribution of condition.

### 中文对照

这些曲线更直观地展示 X21 中的关系。每个面板是一位测试说话人，035 放在大图中。横轴是 HuBERT Tr-24 得到的 exposure-to-test similarity，纵轴是单词识别正确率。灰色表示 control，绿色表示 multi-talker，蓝色表示 single-talker，红色表示 talker-specific。这些已有曲线的 similarity 使用 exp(-k×DTW)，其中 k 大约是 0.358。在每位说话人和每个 condition 内，我用十个分箱汇总反应，再拟合 logistic 曲线。这样可以看到相似度与正确率的关系如何随条件变化。阴影沿用原图的不确定性估计，重采样细节仍待核对。后面的模型比较会正式评价 condition 的额外贡献。

## 13 | Similarity when conditions are pooled

### English

Here I combine the conditions within each test talker. Each black point summarizes accuracy in one of twenty similarity bins, and the black line is a logistic curve fitted to the pooled trial responses. The horizontal and vertical axes keep the same meanings as before. This gives a compact view of the overall upward relationship. Pooling brings together both within-condition and between-condition variation. We can now use the joint models to examine the separate contributions of similarity and condition.

### 中文对照

这里在每位测试说话人内部合并所有 condition。每个黑点汇总二十个 similarity 分箱中一个箱的正确率，黑线则是对合并后的试次反应拟合的一条 logistic 曲线。横纵轴的含义与前一页相同。这张图简洁地展示了总体向上的关系。合并后的结果同时包含 condition 内部和 condition 之间的变化。接下来，我们用 joint model 进一步考察 similarity 和 condition 各自的贡献。

## 14 | Do predictor and condition add information?

### English

Finally, these X21 comparisons ask what the theoretical predictor and condition each contribute. The selected configurations are SBI at Tr-14 and HVE at CNN-6, using within-word transitions. I standardize each fold's predictor using the other two folds, combine those values, and fit the models on the same response rows. The left panel adds the predictor to condition; the right panel adds condition to the predictor. The vertical axis is twice the improvement in fitted log likelihood, and the labels give the p-values. For SBI, adding the predictor gives p around 0.011, and adding condition gives p around 0.0001. For HVE, the corresponding values are about 0.053 and 0.012. The selected HVE coefficient is negative, so that direction is important when discussing exposure variability. These are exploratory, selection-conditional comparisons. Taken together, the figures show a relationship between speech similarity and listener responses, alongside contributions from experimental condition. The next step is to align the model comparisons and ceiling estimates under a common analysis specification.

### 中文对照

最后，这些 X21 模型比较考察理论 predictor 和 condition 各自贡献了什么。选中的配置是 Tr-14 的 SBI，以及 CNN-6 的单词内部 transitions HVE。我用另外两个折的数据标准化当前折的 predictor，再合并这些值，在相同的反应行上拟合模型。左图是在 condition 的基础上加入 predictor，右图是在 predictor 的基础上加入 condition。纵轴是拟合 log likelihood 改善量的两倍，标签给出 p 值。在 condition 模型中加入 SBI 后，p 大约是 0.011；在 SBI 模型中加入 condition 后，p 大约是 0.0001。HVE 对应的两个 p 值大约是 0.053 和 0.012。获选 HVE 的系数为负，所以讨论 exposure variability 时需要结合这个方向。这些是以当前 predictor 选择结果为前提的探索性比较。综合来看，这批图展示了语音相似度与听者反应之间的关系，也展示了实验 condition 的贡献。接下来需要在统一的分析设定下对齐模型比较和 ceiling 估计。
