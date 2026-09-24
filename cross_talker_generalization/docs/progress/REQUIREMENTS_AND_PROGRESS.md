# 工作清单：9 月 11 日会议及会后邮件

只整理 **9 月 11 日会议、会后诊断邮件、9 月 15 日往返邮件**。不包括今天的新邮件。

## 已经完成

| 内容 | 去哪里看 |
|---|---|
| 已向 Florian 说明：当前是三折 train-test，没有独立调参集。 | [邮件记录](<../../analysis/diagnostics/CORRESPONDENCE_2026-09-15.md>) |
| 六张训练／测试诊断图已做好，用来观察模型在测试集上的表现是否变差。 | [六张图及解释](<../../analysis/diagnostics/sbi_review/README.md>) |
| X21 的两套参数地图已做好，可以旋转查看。 | [完整条件地图](<../../../outputs/sbi_parameter_maps_2026-09-18/presentation/x21_all_conditions.html>)；[去掉 Talker-specific 的地图](<../../analysis/sbi/parameter_maps/x21_without_talker_specific/figures/parameter_landscape_interactive.html>) |

注意：六张诊断图对应当前固定距离模型；参数地图只覆盖一个层、一个训练划分。它们各自已经完成，不代表全部优化实验完成。

## 还要做

| 顺序 | 要回答的问题／要完成的事 | 目前做到哪里 |
|---|---|---|
| 1 | **按 z 选参数，和按 likelihood 选参数，结论是否不同？** 两种选法都展示 z 和 likelihood。 | 已有单层地图，完整跨折／跨层对照没做完。进一步计算此前暂停，恢复时用简单、说明清楚且检查过稳定性的搜索即可。 |
| 2 | **为什么 AN19 的 MFCC／STRF 比 HuBERT 还强，而 X21 没有这个现象？** | 已有排查和解释线索，还没有完全确定原因。[现有解释](<../../analysis/acoustic_baselines/AN19_BASELINE_EXPLANATION.md>) |
| 3 | **不同层的表现差异，是否有统计证据？** | 还没做检验。先确定比较哪些层、采用什么检验。 |
| 4 | **把会上指出的图形问题收尾。** | 方法图：修正时间轴与布局；地图／音系热图：核对来源和取值；talker 矩阵：调整色标并查看含／不含英语组；逐层图：统一坐标轴、图例与 ceiling 表达。已有图，修改还没全部完成。 |
| 5 | **整理后发给合作者。** 先讲 Tr-24，再讲各层；附上图、代码和简短方法说明。 | 已有[汇报 PPT](<../../../outputs/sbi_parameter_maps_2026-09-18/presentation/X21_parameter_maps.pptx>)。最终整理及哪些材料已经发出，仍需确认。 |

**会议最强调的是前两项：两种优化目标的比较，以及 AN19 高基线的原因。**

## 暂时不做

- **HVE 后续复核：**按你的安排继续暂停。
- **复杂优化器竞赛：**不作为必要任务；保留简单搜索及必要的稳定性检查。
- **新增数据集的音素图、语言组间检验、exposure 实际理解建模：**属于讨论方向，暂不自动开展。
- **今天新邮件的要求：**不在这份清单中。

---

<details>
<summary>需要时展开：逐项依据、详细文件地址和历史记录</summary>

## 9 月 11 日会议与会后诊断邮件：合并待办与完成情况

更新：2026 年 9 月 21 日。

**本页范围：9 月 11 日会议、会后训练／测试诊断邮件，以及 9 月 15 日往返邮件。按 Zhengyang 最新要求，不纳入今天新收到的方法邮件。** 会上重申的早期任务仍列在正文；更早且此次未重申的项目留在文末历史附录。按问题合并重复要求，不按首次提出时间删去本轮仍需跟进的事项。

当前仅更新这份清单，没有修改实验代码、运行分析、发送消息或发布文件。任务列出不代表已获准恢复此前暂停的计算。

## 一、先看当前进度与边界

- **已经完成的具体内容：**9 月 15 日说明当前 train-test 流程；固定 τ=2、负标准化距离模型的六张 SBI 训练／测试诊断图；X21 两套代表性参数地图及 PPT。图已准备不等于已外发。
- **会议明确的两个重点仍未全部解决：**AN19 的高声学基线原因；分别按 z 和 likelihood 优化并交叉展示两种结果。它们虽在早期已提出，仍属于本轮跟进。
- **仍未完成：**完整跨折／跨层优化目标对照、层间差异统计检验、若干图示及资料来源核查、最终图形样式和分享。
- **HVE 后续复核继续暂停。**完整跨折／跨层参数研究此前也暂停，本次整理不自动恢复。暂停来自 Zhengyang 的安排。
- **不把高级优化器竞赛列为必要任务。**需要交代搜索算法和合理性、检查起点或网格精度、边界与拟合稳定性；不要求证明 Optuna 或其他复杂方法优于所有替代方法。
- **不新增今天邮件提出的实验。**本页不加入其方法分支比较、正则化流程或 offset 方案；也不据此改写当前计算。

## 二、来源与状态说明

| 标记 | 来源 | 使用边界 |
|---|---|---|
| M | [9 月 11 日会议自动转录](<../../../outputs/meeting_2026-09-11/TRANSCRIPT_EN.md>) | 时间戳相对录音开头；无可靠逐句说话人分离。以下是讨论要求的归纳，层间检验名称未可靠识别。本次重读记录，未重新转录视频。 |
| E1 | [会后训练／测试诊断邮件（清单末尾保留原文）](<../../../outputs/meeting_2026-09-11/TODO_CN.md>) | 原邮件日期未提供；要求按观测数归一化的 likelihood 诊断，明确不做 z 比值。 |
| E2 | [9 月 15 日往返邮件](<../../analysis/diagnostics/CORRESPONDENCE_2026-09-15.md>) | 日期由 Zhengyang 确认；Florian 接受 test/train mean log loss。这次回复没有回答独立调参问题。 |
| U | Zhengyang 对工作的安排 | 保留有符号 z；HVE 暂停；不把高级优化器比较作为必要研究任务。这些是用户决定，不归到 Florian 名下。 |

状态含义：**已完成（限定范围）**、**部分完成**、**未完成／待明确方案**、**暂停**、**建议／待讨论**。准备好材料与外发材料分开记录。所有文件链接均为本地完整地址。

## 三、核心分析与优化核查

T 编号是本页合并后的编号；“原表”指之前的 21 项表格。

| 编号 | 合并后的待办与来源 | 当前状态、已做和仍需完成 | 已有文件／证据 |
|---|---|---|---|
| **T01** | **分别按 z 和 GLMM log-likelihood 优化理论参数，并交叉展示两种指标。** M 18:51–22:07、23:10–23:36、30:22–32:04；原表 5。 | **局部完成；完整对照未完成、此前暂停。**AN19 小测试和 X21 单层／单训练划分地图已有。每个参数候选必须先正常进行 GLMM 似然拟合，再提取有符号 z／log-likelihood；已有地图符合这一点。完整跨折／跨层结果尚未形成。 | [小规模测试](<../../analysis/sbi/optimizer_probe/README.md>)；[X21 两目标结果](<../../analysis/sbi/parameter_maps/x21_all_conditions/README.md>)；[目标对应图](<../../analysis/sbi/parameter_maps/x21_all_conditions/figures/objective_agreement.png>) |
| **T02** | **说明具体搜索方法及理由，检查边界和搜索／拟合稳定性。** M 24:23–28:26、29:13–29:42、31:07–35:33；原表 6。 | **部分完成。**历史 TPE／bounded search、范围、惩罚和当前算法已查清，地图已检查边界与近常数相似性。Florian 要求的是方法合理且稳定，没有要求证明高级优化器优越。若用确定性网格，无随机起点；网格加密、边界和必要的局部复核可作为我们的替代验证建议。跨折变化另行记录，不能冒充同一数据上的起点检查。 | [优化器核查](<../../analysis/sbi/OPTIMIZER_AUDIT.md>)；[地图数值检查](<../../analysis/sbi/parameter_maps/x21_all_conditions/README.md>) |
| **T03** | **分享旧参数曲面，用更新代码重画代表曲面并观察噪声是否减少。** M 35:39–36:15；原表 7。 | **新版图已完成；旧新严格对照及外发待确认。**X21 完整条件和剔除 Talker-specific 两版各完成 1,247 个网格点及 29 个极限参考拟合，均为 Tr-24 的一个训练划分。当前平滑曲面不证明所有层／划分的全局光滑性，也没有确定旧图不规则的具体原因。 | [完整条件互动图](<../../../outputs/sbi_parameter_maps_2026-09-18/presentation/x21_all_conditions.html>)；[剔除 Talker-specific 互动图](<../../analysis/sbi/parameter_maps/x21_without_talker_specific/figures/parameter_landscape_interactive.html>)；[地图说明](<../../analysis/sbi/parameter_maps/x21_all_conditions/README.md>) |
| **T04** | **解释 AN19 的 MFCC／STRF 为什么接近或高于 HuBERT，而 X21 没有同样现象。** M 28:50–29:12；早期任务本次列为重点；原表 10。 | **部分完成。**已核对声学组件标准化、复现完整基线、检查条件内／共同随机结构，并完成单词均值分析及 PPT。单词间差异提供解释线索；尚未完全解释历史 z 排名或确定时长、音素组成、录音因素等机制。 | [综合解释](<../../analysis/acoustic_baselines/AN19_BASELINE_EXPLANATION.md>)；[两页 PPT](<../../../outputs/an19_baseline_explanation_2026-09-18/presentation/AN19_acoustic_baselines.pptx>)；[组件诊断图](<../../analysis/acoustic_baselines/components/matched_acoustic_components.pdf>) |
| **T05** | **在确认优化正确后，用同一套核实后的方法复核 HVE，再判断其解释力。** M 40:56–41:29；原表 9。 | **暂停，尚未完成会议要求的完整复核。**已有 HVE 定义、exposure 和结果不等于这一复核已完成。暂停由 Zhengyang 决定，不能记为 Florian 同意停止，也不能把当前弱结果直接写成 HVE 已被否定。 | [已有 HVE 结果范围](<../../analysis/diagnostics/PROGRESS_REPORT.md>)；[批次验证](<../../analysis/diagnostics/source_validation/final_validation.json>) |

## 四、训练／测试诊断、层间检验和结果展示

| 编号 | 合并后的待办与来源 | 当前状态、已做和仍需完成 | 已有文件／证据 |
|---|---|---|---|
| **T06** | **说明当前使用 train-test 还是 train-tune-test。** E1、E2；原表 3。 | **现状说明已完成。**你已于 9 月 15 日回复：参与者级三折，两折训练、一折测试，没有独立 tuning split；用同一批 CV 分数选择的最佳配置没有额外独立测试。就本页纳入的邮件而言，Florian 认可了诊断比值，没有回答另设调参划分的问题。因此记录这个待确认问题，不把“必须实现 nested CV”添加为本轮确定任务。 | [已发送回复与确认范围](<../../analysis/diagnostics/CORRESPONDENCE_2026-09-15.md>)；[当前划分及评分代码](<../../R/fit_confirmatory.R>) |
| **T07** | **完成 SI 逐层训练／测试拟合诊断。** E1、E2；原表 4。 | **当前固定距离模型范围内已完成。**采用每词级响应 mean test log loss / mean training log loss，同一训练模型／标准化／随机效应置零规则给两边评分；六张 SBI 图已核验并加入 PPT。不是 z 比值，也不是基线 gain 比值。未来若完成新优化模型，应提供其对应诊断；现图不能替代。 | [六张图、计算及解释](<../../analysis/diagnostics/sbi_review/README.md>)；[逐折分数](<../../analysis/diagnostics/si_diagnostics/tables/paired_fold_ratios.csv>)；[核验结果](<../../analysis/diagnostics/sbi_review/verification.json>)；[PPT 第 5–10 页](<../../../outputs/sbi_parameter_maps_2026-09-18/presentation/X21_parameter_maps.pptx>) |
| **T08** | **对层间差异提供统计证据。** M 42:12–43:45；原表 12。 | **未完成。**会议提到保留同一折结果的配对关系，接受差异证据有限的结果。比较哪些层、具体检验及适用性待明确；转录未恢复检验名称，没有决定改成五折，也没有支持把重叠训练折当作完全独立样本。 | [讨论与待办](<../../../outputs/meeting_2026-09-11/TODO_CN.md>)；[42 分钟附近转录](<../../../outputs/meeting_2026-09-11/TRANSCRIPT_EN.md>)；没有已完成的检验结果文件 |
| **T09** | **先固定 Tr-24 讲结果，再展示逐层变化。** M 41:29–42:12；早期任务重申；原表 11。 | **部分完成。**固定层与逐层图都有，当前论文／汇报顺序尚未核对。旧固定层 gain 图与 z 图统计量不同；新分析确定后需对应更新，不能仅改标题。 | [固定 Tr-24 既有汇总图](<../../analysis/model_comparison/reference_checks/figures/figure_01_fixed_tr24_sbi_predictor_gain.png>)；[X21 逐层 z 示例](<../../analysis/speech/figures/sbi_x21_base_z.pdf>) |
| **T10** | **完善 z／ceiling 图样式，统一并分享给合作者。** M 36:22–38:55；原表 13；早期样式要求继续保留。 | **部分完成。**已有三折点、95% 区间、mean ceiling=100% 灰带和显著性参考线；会议要求减少多余负轴留白、简化图例，把说明移至图注。最后的轴范围／跨合作者统一及外发未核实。E1 的 likelihood 诊断要求本身不意味着必须重算 z ceiling；若更换正式 z 的拟合方式，才需核对匹配的 ceiling。 | [既有图形修订记录](<../../analysis/speech/REVIEW_RESPONSE.md>)；[逐层 z 示例](<../../analysis/speech/figures/sbi_x21_base_z.pdf>)；[图形规范](<../Z_VALUE_FIGURE_SPEC.md>) |

## 五、Figure 1／验证图的修改

这些项目在早期已启动，9 月 11 日再次提出具体修改，仍属于本轮跟进。

| 编号 | 合并后的待办与来源 | 当前状态、已做和仍需完成 | 已有文件／证据 |
|---|---|---|---|
| **T11** | **完成 Figure 1a/b：语音→DNN→三维轨迹。** M 00:00–02:22、07:15–07:33；原表 15。 | **部分完成。**同一句话中的词／音段须对应；统一音段颜色，白底、简化轨迹，图注写真实帧步长。已有真实示例，并已确认 Tr-24 为 20 ms 步长、25 ms 前端感受野；图中时间位置仍需由 duration/T 近似改为核实后的时间映射，布局和图注未收尾。 | [现有 Figure 1a/b](<../../analysis/speech/figures/figure_1ab_method.pdf>)；[时间步和输入核查](<../../analysis/diagnostics/source_validation/FIGURE_INPUT_AUDIT.md>)；[绘图代码](<../../scripts/build_september9_method_panel.py>) |
| **T12** | **核对语言地图的范围、分组、地理标点和说明。** M 02:32–04:06；原表 16。 | **部分完成。**已保存 Florian 的图和代码并采用其配色。最终语言覆盖、标点所代表的地理含义仍需核对，不能把示意标点当作真实说话者所在地。 | [世界地图](<../../analysis/speech/sources/language_capitals_worldmap.pdf>)；[地图 R 代码](<../../analysis/speech/sources/language_capitals_worldmap.R>)；[采用的分组配色](<../../analysis/speech/tables/language_branch_palette.csv>) |
| **T13** | **核查传统音系热图与 inventory 汇总的来源、选择和计算。** M 04:02–06:02、07:33–10:27；原表 17。 | **部分完成。**原图及范围更窄的 inventory-overlap 候选图已有。须核实数据库／具体特征值、评分、tone 和权重、汇总定义及裁切标签；不能用库存重叠结果替代整个音系配列／超音段热图的来源证明。 | [原音系特征热图](<../../analysis/speech/sources/english_phonology_similarity_heatmap.pdf>)；[库存重叠候选图](<../../analysis/speech/figures/figure_1d_phonological_inventory_similarity.pdf>) |
| **T14** | **改善 talker 距离／相似性矩阵可读性，比较含／不含英语组的版本。** M 13:22–17:35；原表 18。 | **部分完成。**AN19 42 位 talker、138 个共同词、861 对的矩阵已有并核验。还需处理距离与相似性颜色方向的混淆，检查英语组对色标范围的影响，确定最终展示。不能采用转录中的未确认词数 178。 | [距离矩阵](<../../analysis/speech/figures/figure_2a_an19_42_talker_distance_linear.pdf>)；[相似性矩阵](<../../analysis/speech/figures/figure_2a_an19_42_talker_similarity_linear.pdf>)；[词数与配对核查](<../../analysis/diagnostics/source_validation/FIGURE_INPUT_AUDIT.md>) |

## 六、候选扩展与展示取舍

这些需要保留，但不自动变成已批准的新实验。既有 Figure 2b 覆盖问题与 Figure 2d 听者错误分析仍在历史附录，不宣称已完成。

| 编号 | 合并后的待办与来源 | 当前状态、已做和仍需完成 | 已有文件／证据 |
|---|---|---|---|
| **T15** | **在音素／segment 相似性汇总中考虑更多数据集和 talker。** M 11:41–13:14。 | **有条件建议；未实施新跨数据集图。**先确认各数据集有可比的英语参考及同词同语境录音。现有 AN19 图可用，不把它当作已完成跨数据集扩展。 | [AN19 当前汇总图](<../../../output/pdf/AN19_all_segments_similarity.pdf>)；[自动标注与覆盖](<../../analysis/speech/an19_phone_alignment/README.md>) |
| **T16** | **考虑检验同 L1 内与不同 L1 间的相似性差异。** M 17:53–18:30。 | **候选统计扩展；未完成。**这是矩阵之外的建议，与 T08 层间检验不同。抽样单位、依赖关系及模型尚未确定。 | [已有矩阵数值](<../../analysis/speech/tables/figure_2a_42_talker_similarity_matrix.csv>)；[talker 语言对应](<../../analysis/speech/tables/figure_si_talker_language_map.csv>) |
| **T17** | **确定传统语言参考面板保留多少、如何与模型验证衔接。** M 04:27–04:58、06:32–07:33、10:03–10:27。 | **待讨论。**明确传统语言级描述与录音／talker 级结果的关系和图中位置；没有确定必须新增一个传统评分与模型评分的相关检验。 | [已有图文报告](<../../analysis/speech/REPORT_FOR_FLORIAN.md>)；[传统参考候选图](<../../analysis/speech/figures/figure_1d_phonological_inventory_similarity.pdf>) |
| **T18** | **利用 exposure 中实际理解／转写的音和词，更具体地预测 test。** M 39:06–40:31。 | **长期方向；未实施。**会议邀请提出方案，尚无简单确定实现；需要区分具体内容迁移与一般听者能力。不能用总体 exposure/test 正确率相关或“最初十次 exposure”曲线替代。 | [原会议讨论](<../../../outputs/meeting_2026-09-11/TRANSCRIPT_EN.md>)；[既有设计待办](<../../../TODO.md>)；没有完成的新模型结果 |

## 七、文档、分享与沟通

### T19｜说明实际方法，并分享图／代码／算法依据【部分完成；外发待核实】

来源：M 26:17–26:23、33:13–33:54、35:04–36:15、38:34–38:40。

- [x] 已核查历史优化器与实际参数地图方法，已有说明链接；本页整合本次会议及 E1/E2，保留已有结果的统计范围。
- [x] 已有参数地图、诊断图、AN19 解释及 PPT。
- [ ] 用简洁文字向合作者解释外层参数搜索与内部 GLMM 拟合的区别、所用算法及选择理由；不能只回答“使用默认设置”。
- [ ] 分享需要的旧／新曲面、逐层图样式、代码与算法说明链接，并核实哪些已由 Zhengyang 发出。材料在本地存在不等于已交付。
- [ ] 最终论文／Overleaf 面板顺序与跨合作者图形格式待核对；本次不自动提交 Git、上传 Overleaf 或发送邮件。

材料入口：[当前十页 PPT](<../../../outputs/sbi_parameter_maps_2026-09-18/presentation/X21_parameter_maps.pptx>)（1–2 页地图，3–4 页 AN19 基线，5–10 页诊断）；[优化器说明](<../../analysis/sbi/OPTIMIZER_AUDIT.md>)。

**会议安排记录（不计入科学任务）：**M 43:56–45:01 最后讨论 10:30，并有已放日历的发言；时区未明确。以实际邀请为准，本次不推定已经外发、不创建或修改日历。

## 八、建议跟进顺序与暂停事项

这只是整理后的建议，不是新一轮计算授权：

1. **先汇报已完成内容：**T06 流程说明、T07 六张诊断图、T03 新版参数地图；同时标明固定距离／指数相似性、训练拟合／冻结预测等各自统计定义。
2. **继续保留会议两个核心问题：**T01 两种优化目标对照、T04 AN19 高基线。完整参数分析此前暂停，恢复执行需另行安排；复杂优化器竞赛不作为前置条件。
3. **推进未完成的检验和展示：**T08 层间差异检验、T09–T10 结果顺序与图形统一、T11–T14 方法图／来源／矩阵。具体统计检验仍需明确。
4. **完成 T19 的说明和资料共享。**T15–T18 的候选扩展单独讨论。
5. **T05 HVE 继续暂停。**本次没有恢复 HVE 或参数搜索，也没有增加今天邮件的任务。

## 九、与之前 21 项表格的对应关系

- 原 3 → T06；原 4 → T07；原 5 → T01；原 6 → T02；原 7 → T03；原 9 → T05。
- 原 10 → T04；原 11 → T09；原 12 → T08；原 13 → T10。
- 原 15 → T11；原 16 → T12；原 17 → T13；原 18 → T14。
- 原 1、2、8、14、19、20、21 留在历史附录；不自动重跑，也不因此视为取消。需要引用既有方法／结果时保留其统计范围。
- T15–T18 单列会议的建议和长期方向，T19 汇总方法说明与资料分享要求。没有增加今天新邮件的任务。

---

## 历史附录：此前要求与材料，仅供追溯

**以下内容不计入上面的本轮会议／会后邮件清单，也不作为当前新增优先事项。** 保留旧编号 Rxx 以便对应以前的记录；其中有些邮件日期未明确，不将它们归为 9 月 11 日后新增要求。

### 附录 A｜更早或日期未确认的来源

- **M1｜9 月 1 日会议：**本次依据先前带时间戳的 [会议要求核对记录](<../FLORIAN_REQUIREMENTS_REVIEW.md>)。本轮未重新找到并完整重读当日原始自动转录；下文时间点沿用该记录，不声称重新听完了这段视频。
- **E1｜8 月 28 日邮件：**要求确认三折、提供跨折 bootstrap CI、行为参考线、灰线及 HVE 定义说明，并统一图形风格。复现问题原本主要写给 Wei-Kai；统一格式等要求也适用于共同项目。
- **E2｜9 月 2 日邮件链及其开头的五点后续说明：**涉及优化与报告的区别、统一消融图、Figures 1/2、AN19 高基线和具体 exposure 理解的扩展。开头五点正文没有独立日期头，不能给它指定未经确认的发送日期。E1/E2 的原始附件为 [pasted-text.txt（本机邮件附件，位于项目外）](<C:/Users/Alex/.codex/attachments/2de57af5-34ae-4d37-b127-b4ed07af9bbc/pasted-text.txt>)；本报告不再复制私人邮件头。
- **E3｜早期 predictor-only／模型比较澄清及嵌套 GLMM 邮件：**对话没有提供确切发送日期。[8 月 27 日实现记录](<../../analysis/model_comparison/selection/README.md>) 和 [9 月 1 日 LRT 报告](<../../analysis/model_comparison/pooled_lrt/README.md>)说明了相应工作；实现日期不能当作邮件日期。
- **P/S｜原 PDF 批注与粘贴的 Slack 对话：**索引见 [9 月 9 日逐条回复](<../../analysis/speech/REVIEW_RESPONSE.md>)。它们是补充来源，不混写为视频／邮件要求。本次没有声称读取了未提供的完整 Slack 私聊或最新 Overleaf。

### 附录 B｜早期分析、复现和论文面板任务

这些条目保持既有完成范围和文件证据；不会因为移到附录而变成已完成或被取消。

#### R01｜按 predictor-only 选择，再进行两种模型比较【已完成】

来源：E3；M1 16:46–20:17。（历史记录；不计入本轮新增要求。）

要求：在不加入 condition 的模型中选择理论 predictor，然后比较 joint 与 condition-only、joint 与 predictor-only；两次比较不各自重新优化理论 predictor。

已做／现状：8 月 27 日版本已实现 predictor-only 的留出评分选择和两种后续比较。所选层／方法使用同一批 CV 分数，结果仍属于探索性选择；这项完成不等于连续参数 τ、k 的优化已经完成。

尚缺／解释范围：所选最佳配置仍没有额外独立的最终测试；正文 T01 另行跟踪参数优化。

对应文件与地址：

- 方法与结果说明：[cross_talker_generalization/analysis/model_comparison/selection/README.md](<../../analysis/model_comparison/selection/README.md>)
- SBI 层选择结果表：[cross_talker_generalization/analysis/model_comparison/selection/tables/sbi_predictor_only_selected_layers.csv](<../../analysis/model_comparison/selection/tables/sbi_predictor_only_selected_layers.csv>)
- 两种后续比较图：[cross_talker_generalization/analysis/model_comparison/selection/figures/figure_02_sbi_selected_downstream_comparisons.png](<../../analysis/model_comparison/selection/figures/figure_02_sbi_selected_downstream_comparisons.png>)
- 候选选择与报告代码：[cross_talker_generalization/src/ctg/report_core.py](<../../src/ctg/report_core.py>)

#### R02｜补充合并三个测试折的嵌套 GLMM likelihood-ratio test【已完成】

来源：E3。（历史记录；不计入本轮新增要求。）

要求：在 CV 预测评价之外，增加合并数据的嵌套模型似然比检验。

已做／现状：9 月 1 日完成 42 个 GLMM 拟合和 28 个比较，均成功。实际采用方案 B：合并按折标准化的 predictor 后拟合；没有执行方案 C 的 τ／k 参数平均。当前 τ 和变换固定，层／HVE 方法为类别选择，不能直接平均。

尚缺／解释范围：这些 p 值以已选 predictor 为条件，尚未校正同一批数据上的选择；AN19／B23 样本和模型结构的比较范围仍需保留说明。

对应文件与地址：

- 方法、方案 B 与解释范围：[cross_talker_generalization/analysis/model_comparison/pooled_lrt/README.md](<../../analysis/model_comparison/pooled_lrt/README.md>)
- 28 项 LRT 结果：[cross_talker_generalization/analysis/model_comparison/pooled_lrt/tables/crossfitted_lrt_results.csv](<../../analysis/model_comparison/pooled_lrt/tables/crossfitted_lrt_results.csv>)
- 模型诊断表：[cross_talker_generalization/analysis/model_comparison/pooled_lrt/tables/crossfitted_model_diagnostics.csv](<../../analysis/model_comparison/pooled_lrt/tables/crossfitted_model_diagnostics.csv>)
- LRT 汇总图：[cross_talker_generalization/analysis/model_comparison/pooled_lrt/figures/figure_01_crossfitted_nested_lrt.png](<../../analysis/model_comparison/pooled_lrt/figures/figure_01_crossfitted_nested_lrt.png>)

#### R08｜统一 HVE 定义，并恢复实际 exposure【部分完成】

来源：早期 E3／P/S。（历史记录；不计入本轮新增要求。）

要求：明确局部与全局顺序、重复实例的要求、取 root 与不取 root 的含义，以及参与者实际听过的 exposure。

已做／现状：AN19／X21 的顺序和 B23 的录音分配已整合。局部 token 内 transitions 与全局跨 token transitions 已区分；B23 有 168 名可用无序 exposure 参与者，其中 97 名可恢复完整顺序。跨 token 边界的具体定义来自 Zhengyang 的选择。

尚缺／解释范围：当前 dispersion 仍不取 1/τ root；root 问题的最终约定和同条件对照尚未确立。对照是待讨论的验证方式，不能写成 Florian 已明确要求的实验。HVE 后续工作暂停。

对应文件与地址：

- HVE 公式、顺序和重复实例规范：[cross_talker_generalization/docs/SCIENTIFIC_SPEC.md](<../SCIENTIFIC_SPEC.md>)
- HVE 计算代码：[cross_talker_generalization/src/ctg/metrics.py](<../../src/ctg/metrics.py>)
- 实际 exposure 整合代码：[cross_talker_generalization/src/ctg/exposure.py](<../../src/ctg/exposure.py>)
- 实际 exposure 修订后的结果：[cross_talker_generalization/analysis/model_comparison/selection/README.md](<../../analysis/model_comparison/selection/README.md>)

#### R14｜提供分组正确、比较范围一致的行为 ceiling【部分完成】

来源：E1；M1 05:56–08:27；P01。（历史记录；不计入本轮新增要求。）

要求：用训练中的行为反应预测测试反应作为参考，保留必要的条件／语境分组和不确定性；可比分析采用共同定义。

已做／现状：9 月 18 日已修正 AN19／X21 的 condition 分组，并区分 X21 重复词的句子语境；三个数据集的 likelihood 参考已重算。

尚缺／解释范围：将修订参考值应用到完全匹配的理论模型样本尚未全部闭合。该 likelihood 参考不产生 Wald z；历史 z ceiling 是另一套保留的计算。

对应文件与地址：

- 修正内容及三个数据集的结果说明：[cross_talker_generalization/analysis/sbi/README.md](<../../analysis/sbi/README.md>)
- 原值与修正值比较图：[cross_talker_generalization/analysis/model_comparison/conditional_ceiling/conditional_ceiling_comparison.pdf](<../../analysis/model_comparison/conditional_ceiling/conditional_ceiling_comparison.pdf>)
- 对应比较表：[cross_talker_generalization/analysis/model_comparison/conditional_ceiling/comparison.csv](<../../analysis/model_comparison/conditional_ceiling/comparison.csv>)
- 行为参考值计算代码：[cross_talker_generalization/src/ctg/ceiling_cv.py](<../../src/ctg/ceiling_cv.py>)

#### R19｜完成 Figure 2b 的音素级 L2-to-English 差异图【部分完成】

来源：E2 图注；S05。（历史记录；不计入本轮新增要求。）

要求：展示音素级发音差异；缺少标注时可考虑 ASR／自动对齐或其他数据集。

已做／现状：已自动对齐 AN19 全部 6,261 条录音并生成候选音素图。16,086 个 L2 音素区间中，有 5,114 个具备六个可用英语参考。

尚缺／解释范围：不同词、音素和 talker 的覆盖不均，仍需评估影响。自动对齐的目标音素不能等同于人工确认的实际发音，也不能代替听者错误标签。

对应文件与地址：

- 自动标注方法、来源与覆盖记录：[cross_talker_generalization/analysis/speech/an19_phone_alignment/README.md](<../../analysis/speech/an19_phone_alignment/README.md>)
- Figure 2b 候选图：[cross_talker_generalization/analysis/speech/an19_phone_alignment/figures/figure2b_automatic_intended_phone_deviation.png](<../../analysis/speech/an19_phone_alignment/figures/figure2b_automatic_intended_phone_deviation.png>)
- 按语言／音素汇总的结果表：[cross_talker_generalization/analysis/speech/an19_phone_alignment/tables/figure2b_language_phone_estimates.csv](<../../analysis/speech/an19_phone_alignment/tables/figure2b_language_phone_estimates.csv>)
- 自动对齐运行代码：[cross_talker_generalization/scripts/run_an19_falcon_alignment.py](<../../scripts/run_an19_falcon_alignment.py>)
- 正式音频来源的对齐完成记录：[cross_talker_generalization/analysis/speech/an19_phone_alignment/nygaard_audio/alignment_summary.json](<../../analysis/speech/an19_phone_alignment/nygaard_audio/alignment_summary.json>)
- Figure 2b 的参考覆盖表：[cross_talker_generalization/analysis/speech/an19_phone_alignment/tables/figure2b_reference_coverage.csv](<../../analysis/speech/an19_phone_alignment/tables/figure2b_reference_coverage.csv>)

#### R20｜展示 control 条件的 intelligibility 与相似性关系【已完成】

来源：E2 Figure 2c；相关 P09–P11、P17–P18。（历史记录；不计入本轮新增要求。）

要求：展示 AN19／X21 control 测试反应与英语参考相似性的关系，并改善条件／talker 曲线表达。

已做／现状：已生成同词匹配的 control 曲线、边缘分布和共享图例；X21 分条件曲线及黑色汇总曲线也已完成。

尚缺／解释范围：这些是描述性关联。HW74 的暂时排除仍是独立待解决事项；现图不是最初十次 exposure 的分析。

对应文件与地址：

- AN19／X21 control 曲线与边缘分布：[cross_talker_generalization/analysis/speech/figures/figure_2c_control_similarity_with_marginals.pdf](<../../analysis/speech/figures/figure_2c_control_similarity_with_marginals.pdf>)
- X21 分条件曲线：[cross_talker_generalization/analysis/speech/figures/x21_s_curves_by_condition.pdf](<../../analysis/speech/figures/x21_s_curves_by_condition.pdf>)
- X21 黑色汇总曲线：[cross_talker_generalization/analysis/speech/figures/x21_s_curves_pooled.pdf](<../../analysis/speech/figures/x21_s_curves_pooled.pdf>)
- 条件曲线作图代码：[cross_talker_generalization/scripts/build_september9_condition_curves.py](<../../scripts/build_september9_condition_curves.py>)

#### R21｜完成 Figure 2d 的听者音素错误分析【未完成】

来源：E2 图注。（历史记录；不计入本轮新增要求。）

要求：分析听者 segment 错误与 L2-to-L1 差异、词汇约束之间的关系。

已做／现状：已有图注和自动声学音素区间，可作为准备材料。

尚缺／解释范围：听者回答到音素的对齐、词汇邻居定义和行为模型仍缺少；目前没有完成的 Figure 2d 结果图。

对应文件与地址：

- Figure 1/2 面板要求：[cross_talker_generalization/docs/MAIN_FIGURE_SPEC.md](<../MAIN_FIGURE_SPEC.md>)
- 已有声学标注及其使用边界：[cross_talker_generalization/analysis/speech/an19_phone_alignment/README.md](<../../analysis/speech/an19_phone_alignment/README.md>)

#### R22｜说明 HDF5 的生成、seed 与数值复现【部分完成】

来源：8 月复现邮件及随后提供的 seed 问题。（历史记录；不计入本轮新增要求。）

要求：区分下游 seed 与特征提取／t-SNE 设置，说明哪些步骤可以复现、有哪些输入尚需提供。

已做／现状：已有输入注册、下游配置和说明；Figure 1 核查把归档 extractor/helper 与 HDF5 记录的哈希对应起来。

尚缺／解释范围：尚不能确认完整的音频→HDF5 再生成包、完整上游 seed／环境记录以及合作者数值完全一致的复现都已完成。某份旧代码有 seed 不足以证明所有最终 HDF5 的来源。

对应文件与地址：

- 项目技术文档：输入及复现范围：[TECHNICAL_DOCUMENTATION.md](<../../../TECHNICAL_DOCUMENTATION.md>)
- HDF5 输入注册：[cross_talker_generalization/configs/project.json](<../../configs/project.json>)
- 提取器／HDF5 的已核对来源：[cross_talker_generalization/analysis/diagnostics/source_validation/FIGURE_INPUT_AUDIT.md](<../../analysis/diagnostics/source_validation/FIGURE_INPUT_AUDIT.md>)
- 当前复现运行说明：[cross_talker_generalization/docs/RUNBOOK.md](<../RUNBOOK.md>)

### 附录 C｜早期 PDF／Slack 与文档补充事项

这些较早的批注和材料保留供追溯，不计入 9 月 11 日会议及会后邮件的本轮工作。

#### S01｜按全部 segment instances 汇总 similarity、排序、保留 CI 并简化标注【已完成】

对同词、同语境的音素实例，先汇总英语参考，再对每个 L2 talker 的有效实例等权平均，最后展示语言组均值并按 similarity 排序。当前图已恢复韩语／西班牙语组的 talker-bootstrap CI，单 talker 语言组没有估计 CI；顶部图例已删除，x 轴标签已简化。旧的等类型加权版本和无 CI 版本不再作为当前图。


对应文件与地址：

- 当前可直接分享的 PDF：[output/pdf/AN19_all_segments_similarity.pdf](<../../../output/pdf/AN19_all_segments_similarity.pdf>)
- 实例平均与 CI 设置记录：[cross_talker_generalization/analysis/speech/an19_phone_alignment/tables/figure2b_all_segments_similarity_metadata.json](<../../analysis/speech/an19_phone_alignment/tables/figure2b_all_segments_similarity_metadata.json>)
- 逐 talker 相似性结果：[cross_talker_generalization/analysis/speech/an19_phone_alignment/tables/figure2b_all_segments_similarity_talkers.csv](<../../analysis/speech/an19_phone_alignment/tables/figure2b_all_segments_similarity_talkers.csv>)
- 图的计算与绘制代码：[cross_talker_generalization/scripts/build_an19_all_segment_panel.py](<../../scripts/build_an19_all_segment_panel.py>)

#### S02｜区分 control-test 与最初十次 exposure 的表现【未完成】

现有曲线已明确标记为 control 测试结果；最初十次 exposure 的窗口定义及新分析尚未完成，不能用原曲线改名代替。现有图与批注记录：


对应文件与地址：

- 已完成的 control-test 图：[cross_talker_generalization/analysis/speech/figures/figure_2c_control_similarity_with_marginals.pdf](<../../analysis/speech/figures/figure_2c_control_similarity_with_marginals.pdf>)
- P13 的问题及处理说明：[cross_talker_generalization/analysis/speech/REVIEW_RESPONSE.md](<../../analysis/speech/REVIEW_RESPONSE.md>)

#### S03｜核实 AN19 HW74 的 wave／wade 内容【部分完成】

已记录命名／文本差异和受影响数据；尚无确定的录音内容与标签协调结论。当前 Figure 2c 仍排除 40 条相关 control 响应，没有自动改词。源码／表格定位完成不能作为真实发音已经核实的证明。


对应文件与地址：

- 42 个录音的 HW74 命名核查表：[cross_talker_generalization/analysis/model_comparison/reference_checks/collaborator_report/tables/an19_hw74_naming_audit.csv](<../../analysis/model_comparison/reference_checks/collaborator_report/tables/an19_hw74_naming_audit.csv>)
- 核查解释与暂时排除依据：[cross_talker_generalization/analysis/model_comparison/reference_checks/collaborator_report/SOURCE_REVIEW.md](<../../analysis/model_comparison/reference_checks/collaborator_report/SOURCE_REVIEW.md>)
- HW74 核查代码：[cross_talker_generalization/scripts/audit_an19_hw74.py](<../../scripts/audit_an19_hw74.py>)

#### S04｜文档去重、保留 Florian 的修改、以 TODO 记录缺口【部分完成】

早期针对性修订与 TODO 已建立，也有保留 Florian 文稿的修改记录；后来各版本又积累了过时或矛盾的状态，完整去重尚未结束。本次只更新本核对报告，没有重写旧记录。


对应文件与地址：

- 保留 Florian 文稿及针对性修改记录：[DOCUMENTATION_CHANGES.md](<../../../DOCUMENTATION_CHANGES.md>)
- 现有项目待办：[TODO.md](<../../../TODO.md>)
- 9 月 11 日会议待办：[outputs/meeting_2026-09-11/TODO_CN.md](<../../../outputs/meeting_2026-09-11/TODO_CN.md>)

#### S05｜澄清 Whisper 的训练状态及 HVE 定义【已有说明；整体部分完成】

提供的邮件链中已有 Zhengyang 对 Whisper 与 HuBERT 训练方式的说明。HVE 的顺序／exposure 实现已存在，最终定义和优化问题仍见历史 R08 及正文 T05。这些文件不能证明合作者代码完全一致或数值精确复现。

对应文件与地址：

- 已提供的 Whisper 邮件说明：[pasted-text.txt（项目外的原始邮件附件）](<C:/Users/Alex/.codex/attachments/2de57af5-34ae-4d37-b127-b4ed07af9bbc/pasted-text.txt>)。
- HVE 定义与公式：[cross_talker_generalization/docs/SCIENTIFIC_SPEC.md](<../SCIENTIFIC_SPEC.md>)。
- 实际 exposure 修订及结果：[cross_talker_generalization/analysis/model_comparison/selection/README.md](<../../analysis/model_comparison/selection/README.md>)。

### 附录 D｜其他旧待办与记录中的过时表述

- SBI/HVE/声学特征联合后向选择、全层多变量模型、全维表征及 DTW path-length 敏感性等仍在 [旧项目 TODO](<../../../TODO.md>) 中。本轮不把它们扩大为新的会议要求，也不改变已选定的历史 DTW normalization。
- [9 月 9 日批注回复](<../../analysis/speech/REVIEW_RESPONSE.md>)仍把声学组件标准化和 condition-key ceiling 修正列为待做；这两项计算已于 9 月 17–18 日完成，见正文 T04 和历史 R14。
- 旧清单中的参数地图、六张 SBI 诊断图及其解释已更新：新版地图和诊断已生成，优化稳健性／严格旧新对照仍未结束。
- [9 月 18 日旧任务记录](<../../analysis/sbi/NEXT_FLORIAN_TASKS.md>)存在“保留历史 z 图”与早期“必须重算新 z ceiling”段落并存的问题。本次会议／邮件清单以最新范围为准：保留原图，不能从诊断邮件推导出必须重算 z。
- 旧文档的“两页 PPT”描述已经过时，当前是十页；准备文件与实际外发分别记录。
- 9 月 17 日的 282 项 HVE 诊断不等于该批次覆盖全部方法的全部层；其覆盖范围见 [批次说明](<../../analysis/diagnostics/PROGRESS_REPORT.md>)。更早候选搜索的范围另见 [8 月 27 日记录](<../../analysis/model_comparison/selection/README.md>)。
- 更早的完整 Figures 1/2／消融清单、HDF5 精确再生成、音素图和文档去重问题均作为背景保留；本次不重新启动这些历史工作。

此前报告的其他配套文件也保留在这里，供历史查阅：

- 旧版图文报告：[cross_talker_generalization/analysis/speech/REPORT_FOR_FLORIAN.md](<../../analysis/speech/REPORT_FOR_FLORIAN.md>)
- 历史 X21 base likelihood 配套图：[cross_talker_generalization/analysis/speech/figures/sbi_x21_base_loglik.pdf](<../../analysis/speech/figures/sbi_x21_base_loglik.pdf>)
- talker 与语言对应表：[cross_talker_generalization/analysis/speech/tables/figure_si_talker_language_map.csv](<../../analysis/speech/tables/figure_si_talker_language_map.csv>)
- 库存相似性候选数值：[cross_talker_generalization/analysis/speech/tables/figure_1d_inventory_similarity.csv](<../../analysis/speech/tables/figure_1d_inventory_similarity.csv>)
- AN19 矩阵数值：[cross_talker_generalization/analysis/speech/tables/figure_2a_42_talker_similarity_matrix.csv](<../../analysis/speech/tables/figure_2a_42_talker_similarity_matrix.csv>)

本次仅更新这份核对报告：合并 9 月 11 日会议、会后诊断邮件与 9 月 15 日往返邮件，排除今天新收到的方法邮件，保留早期材料作为历史附录。旧记录中的执行顺序不覆盖正文的最新范围。没有运行实验、修改代码或结果、发送消息或发布文件。

</details>
