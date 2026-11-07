# RelP → MoE：SVA neuron / expert 实验协议

2026-11-07。配合 [数学实现 guide](relp_moe_implementation_guide.md)。目标是识别关键单元并测量干预效果，不要求提取完整边图。模型权重不训练；下文 discovery/train 指估计归因排名的数据。

## 1. 可以复用什么，能怎样简化？

复用原论文的 SVA 文本、答案对、四种语法任务和 logit-margin 指标。MoE 不要求另造一套语言任务；需要适配的是 tokenizer、q=g h 坐标、expert 汇总和路由条件下的干预定义。这里以 `allenai/OLMoE-1B-7B-0125` 为首个模型，不能直接把结果称为 Llama-3.1-8B 原结果的完全复现。

| 子任务 | 输入例子 | 目标 / foil | 需要判断哪个主语 |
|---|---|---|---|
| simple | The parents | are / is | 唯一主语 parents |
| nounpp | The secretaries near the car | have / has | 主句主语 secretaries；介词短语内 car 是干扰 |
| rc | The athlete that the managers like | does / do | 主句主语 athlete；跳过中间关系从句 |
| within_rc | The athlete that the managers | like / likes | 从句主语 managers，而不是开头 athlete |

例子说明语法关系，不替代下载的数据。实际 row 使用自己的 `clean_answer / patch_answer`；不能全局固定为 are/is。

推荐简化**样本量和运行预算**，保留原任务的语法结构：先用 `smoke` 的 simple + nounpp 检查直接一致和干扰名词两种情况，每项 20 discovery / 8 calibration / 8 validation，不使用正式 test。随后扩展到四个子任务。只跑 simple 可以证明流程能运行，不能支持跨干扰结构的结论。

论文依据：[§4 数据与指标](https://arxiv.org/html/2601.22594v2#S4)、[§5.2 RelP](https://arxiv.org/html/2601.22594v2#S5.SS2)。论文主设置每项 300 对训练、40 对 held-out；下述 calibration/validation 是本项目增加的开发划分。

## 2. 来源固定与已准备好的文件

- 数据仓库：`TransluceAI/circuits`，commit `2d215d4ba016ba8602b69d71fc0f1ad139a427b7`。
- 官方路径：`data/feature_circuits/{simple,rc,within_rc,nounpp}_{train,test}.json`。这些文件实际是 **JSONL**。
- 模型/tokenizer revision：`9b0c1aa87e34a20052389dce1f0cf01da783f654`。
- [sources.lock.json](../data/relp_sva/sources.lock.json) 保存 Git blob SHA-1、长度、model revision 与 tokenizer/config SHA-256；[download_receipt.json](../data/relp_sva/download_receipt.json) 保存实际下载校验。
- 原始数据和 tokenizer 位于 `data/relp_sva/raw/`、`tokenizer/`，可下载再生且不入 Git；没有下载模型权重。

三份现成的、与 OLMoE tokenizer 对齐的 manifest：

| 清单目录 | 用途 |
|---|---|
| [olmoe_smoke_v1](../data/relp_sva/olmoe_smoke_v1/manifest.json) | simple + nounpp，小规模开发，不动正式 test |
| [olmoe_paper_v1](../data/relp_sva/olmoe_paper_v1/manifest.json) | 四任务，保持论文的 discovery/test 数量和原文件划分 |
| [olmoe_grouped_v1](../data/relp_sva/olmoe_grouped_v1/manifest.json) | 四任务，按前缀句子对分组、去重后的泛化补充 |

每个子目录包含 `discovery.jsonl / calibration.jsonl / validation.jsonl / test.jsonl / selection.json`。每行保存源文件行号 `row_id`、原文、两个答案、两端实际 IDs/tokens、target/foil IDs、token→word 映射、变化位置及 group/content ID。`selection.json` 记录筛选计数、签名和重复/重叠；顶层 manifest 保存文件哈希，runner 读取时校验。

## 3. 数据分配及原数据的重复问题

`paper` 每个任务：官方 train 筛选后、固定 seed 12 洗牌，前 300 行 discovery，随后 40 calibration、随后 40 validation；官方 test 独立同 seed 洗牌后取 40。额外 80 行不从 discovery 的 300 行扣除。各行 ID 不重叠，但**内容可能重复**。

| 子任务 | 官方 train/test 原始行数 | OLMoE 对齐后 train/test 行数 | discovery/calibration/validation/test |
|---|---:|---:|---|
| simple | 480 / 120 | 413 / 97 | 300 / 40 / 40 / 40 |
| rc | 105600 / 26400 | 73463 / 18337 | 300 / 40 / 40 / 40 |
| within_rc | 9600 / 2400 | 7777 / 1943 | 300 / 40 / 40 / 40 |
| nounpp | 1920 / 480 | 1546 / 396 | 300 / 40 / 40 / 40 |

不要把该清单称为“作者发布的精确 300/40 row IDs”：作者代码存在 tokenizer 过滤与采样实现细节，本项目采用下节明示的更严格对齐规则并保存自己的实际清单。复现的是官方数据、任务和规模；此偏离必须记录。

**官方数据中有重复与跨文件重叠。** 本次 paper 清单 discovery/test 的完全相同有向内容分别重叠 27、2、2、0 个（按 simple/rc/within_rc/nounpp 顺序）；无视单复数方向及答案、只比较前缀句子对时分别重叠 17、3、20、11 组。不能把“不同文件/不同 row ID”误报为“未见句子泛化”。paper profile 用于对照基准，不用于声称严格的内容独立性。

为此同时提供 `grouped`：合并官方文件，按 `(clean_prefix, patch_prefix, clean_answer, patch_answer)` 去重；将两个前缀排序后哈希为 group，让反向输入对、同前缀的不同动词候选留在同一组。按组固定 seed 洗牌，60%/15%/10%/15% 分给 discovery/calibration/validation/test，再分别取不超过 300/40/40/40 条。所有划分间 group 重叠为零；这是项目新增协议，不是论文原划分。

| grouped 实际数量 | discovery | calibration | validation | test |
|---|---:|---:|---:|---:|
| simple | 76 | 16 | 15 | 24 |
| rc | 300 | 40 | 40 | 40 |
| within_rc | 300 | 40 | 40 | 40 |
| nounpp | 300 | 40 | 40 | 40 |

simple 的独特句子对少，不能靠复制记录凑成 300 个独立例子。grouped 只保证完整前缀对不交叉，不保证全部词汇不交叉，也不是新语法模板的外推。

Discovery 只生成排名；calibration 用于前向验收和可选均值估计；validation 用于调试/选方案；test 只在方案锁定后运行。paper profile 的内容重叠仍存在，严格泛化的结论采用 grouped。不得按模型是否回答正确来重新筛选主清单；报告所有保留样本的基础 margin/正确率，需要条件分析时另列。

## 4. OLMoE 的确定筛选和 token 对齐规则

1. 用固定 revision 的 fast tokenizer、raw prefix、`add_special_tokens=True`；不加 chat template，不人为追加 BOS，不添加 padding。该 OLMoE tokenizer 在这些前缀上自然不加 BOS。
2. A=`clean_prefix`，B=`patch_prefix`。检查两端长度相等且 IDs 不完全相同。
3. 分别对 A/B 追加两个答案，检查前缀 IDs 完整保持、后缀恰为一个 token，同一个答案在 A/B 后的 token ID 相同；两个候选 ID 必须不同。
4. 对每个非特殊 token 用 offset 判断所属的空格分词位置；一个 token 必须唯一对应一个 word。A/B 的 token→word 签名必须一致，不能仅靠长度相同判断语义位置一致。
5. 检查模板 word 数：simple=2，rc=6，within_rc/nounpp=5。主语 word 下标 simple/rc/nounpp=1，within_rc=4（均零起算）。
6. 在 train 合格行中选 token 数最短的签名；同长时选覆盖最多的签名，再按签名字典序打破平局。将该签名固定应用到所有划分，不根据 test 表现选签名。
7. 当前四任务选中的签名分别 `[0,1]`、`[0,1,2,3,4,5]`、`[0,1,2,3,4]`、`[0,1,2,3,4]`：恰好每个 word 一个 token。未达条件的记录有明确 rejection/other-signature 计数。
8. 之后才按 profile/seed 选样，保存实际 IDs 和 SHA-256。换 tokenizer/模型必须重建，不能复用 Llama token IDs，也不能通过截断句子或填充来凑齐。

没有按激活、归因或模型 margin 过滤。默认候选池为全部非特殊 token 的所有层 neuron；`final` 是另一个分析范围，不能替代四任务的全位置主结果。

## 5. 排名已实现的确定定义

对每个子任务单独计算 clean endpoint paired RelP：`R_i=(q_A−q_B)*beta_A`。模型参数固定，两端自然路由，反传冻结 router。两端同一 row 的目标均是 `logit(clean_answer)−logit(patch_answer)`。

完整分数在截断前按样本有符号平均：`Rbar=mean_i(R_i)`，按 `abs(Rbar)` 排名，保留正负。未活跃节点的零值也参与样本计数。逐样本 top-K 不能用于重建这个均值；入口流式累积全量均值，不必保存所有样本张量。

- neuron 坐标：`(layer, position, expert, neuron)`。
- expert-position：对全部 neuron 有符号求和得到 `(layer,position,expert)`，再独立排序。
- expert-all-positions：再对候选范围内位置有符号求和得到 `(layer,expert)`，另表导出；用 `--unit expert_all_positions` 在指定位置池内联合干预整个 expert。不能把它的数量与 expert-position 数量混报。
- 无路由但另一端有路由的节点保留；expert IDs 不按 top-k 槽位编号。

`score` 输出：全量 `aggregate.pt`、三张独立 ranking JSON、逐样本 `cases.jsonl/parquet`、锁定后的 `selection_plan.json` 和 run metadata。主模式 paired；可选 unpaired 仍解释 target−foil，应标明它不同于官方公开 nopair 分支使用单个正确答案 logit 的实现。

## 6. 验收、干预和随机对照

`verify` 在 calibration 的短输入对上比较：HF/functional 每层 q、MoE 输出、expert ID 集合、权重、最终 residual、统一 fp32 readout 的 margin；RelP on/off 前向；自然 A 路由的有限 beta；完整 neuron patch 与 output patch，以及对应 HF output hook 干预的结果。

默认数值阈值 `atol=1e-4, rtol=1e-3, max_relative_l2=1e-3`，路由集合要求相同。这是严格起点，不是已证明所有 7B bf16 环境都能通过。若失败，查看定位、统一计算精度/顺序；不能为了继续实验随意放宽直到通过。若采用不同容差，预先说明依据、保存失败与新检查记录。旧的 18.7% 误差仍不能作通过证据。

`score/evaluate` 必须使用通过的 `verification.json`，绑定当前 checkpoint 安装、dtype、软件、核心代码、task 和 manifest。文件大小/时间用于绑定本地权重安装，revision 和 tokenizer/config 哈希用于来源记录；它不是对整个 7B 权重文件逐字节重新哈希的证明。迁移机器或更换环境要重新验证。

主干预 baseline 选择 **zero q**，其意义是移除有效写入。对当前位置的 expert，置零其全部 q；其他 expert 权重不重新归一化，不禁止 router 选它，也不寻找替代 expert。下游路由、attention、norm 与激活自然重算。分数解释 router-weighted 内容路径，不是 router 决策归因。

同一固定排名在每个 held-out 输入上做：

- `remove`：只消融集合 C。
- `keep`：保留 C、消融候选池其余单元；其他未纳入池的计算保留。
- `noise`：在 A 上把 C 的 q 换成 B 的 q；A 端 RelP 对应的变化方向为 −R_A。
- `denoise`：在 B 上把 C 的 q 换成 A 的 q；这里使用同一个 A 端发现的集合测恢复，不能把 R_A 数值直接当成 B 端系数预测。

可选 `mean_q_routed`：用 calibration A 的 `E[q | expert selected]`，按 layer-position-expert 分别估计；route mask 从真实 top-k 读取，不能以 activation 非零代替。干预只替换当前被路由选中的专家，未见校准专家的均值取零、计数保存在 artifact 中。这是明确的 MoE 条件均值扩展，不是原论文 dense mean ablation 的原样复现。主 zero 和此 mean 版本分别建 score/run 目录。

随机对照默认 seeds 101/102/103：每个预算匹配 RelP 集合的 layer/position 数量，并从 discovery 曾活跃的候选节点中抽取；whole-expert 单位则匹配每层 expert 数量。所有测试案例用同一个随机集合。仍不保证精确匹配每个节点的激活频率，报告这一限制。不将一个 expert 与一个 neuron 当作同等规模。

锁定预算：neuron `1,5,10,50,100,200,500,1000`；expert-position `1,2,5,10,20,50`。smoke 可在 score 时缩成 neuron `1,5`、expert `1,2` 和一个随机 seed。evaluate 读取已锁定 plan，不在 test 上临时选择另一个预算/基线。所有阈值扫描结果都保留，不只报告最漂亮的 N。

报告原始 margin、正确率及干预变化，归一化采用**均值之比**：

\[
\mathrm{keep}=\frac{\mathbb E[F_{\mathrm{keep}}-F_\varnothing]}{\mathbb E[F_A-F_\varnothing]},\quad
\mathrm{remove}=\frac{\mathbb E[F_{\mathrm{remove}}-F_\varnothing]}{\mathbb E[F_A-F_\varnothing]},\quad
\mathrm{denoise}=\frac{\mathbb E[F_{B\leftarrow A}-F_B]}{\mathbb E[F_A-F_B]}.
\]

keep 对应 faithfulness，remove 是 remaining/completeness（理想 0），noise 报损失的 paired 差异比例。分母绝对值≤1e-6 返回 null 并标记，不把越界比例裁剪到 [0,1]。输出每条样本/预算/对照的 JSONL 和 Parquet，另记录所有位置的 top-k 集合变化及路由权重 L1 变化。汇总包含描述性均值；正式统计推断应对 group 做 bootstrap，而不是把重复行当作独立观测。

## 7. 可直接执行的交接命令

在 repository 根目录，使用隔离环境。依赖见 [requirements-relp-sva.txt](../requirements-relp-sva.txt)，已在 CPU 测试；服务器仍需匹配 CUDA 的 torch wheel。不要覆盖现有实验环境。

```bash
python scripts/relp_sva_prepare.py fetch
# 已提供的实际 manifest 可直接用；要重建，请指定一个不存在的新目录。
python scripts/relp_sva_prepare.py prepare --profile paper \
  --output-dir data/relp_sva/olmoe_paper_rebuilt
```

`fetch` 只下载官方数据和固定 tokenizer/config，不下载权重。GPU runner 的 `--model-dir` 必须指向完整、同 revision 的本地 snapshot，并通过 tokenizer/config 哈希检查。

```bash
export RELP_MODEL_DIR=/absolute/path/to/OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654
export RELP_SVA_MANIFEST=data/relp_sva/olmoe_smoke_v1
export RELP_SVA_TASK=simple
export RELP_SVA_RUN=results/relp_sva_smoke_simple_001

bash scripts/gpu_queue.sh sva-verify -- python scripts/relp_sva_run.py verify \
  --model-dir "$RELP_MODEL_DIR" --manifest-dir "$RELP_SVA_MANIFEST" --task "$RELP_SVA_TASK" \
  --output-dir "$RELP_SVA_RUN/verify"

bash scripts/gpu_queue.sh sva-score -- python scripts/relp_sva_run.py score \
  --model-dir "$RELP_MODEL_DIR" --manifest-dir "$RELP_SVA_MANIFEST" --task "$RELP_SVA_TASK" \
  --verification "$RELP_SVA_RUN/verify/verification.json" --output-dir "$RELP_SVA_RUN/score" \
  --mode paired --pool non_special --baseline zero \
  --neuron-budgets 1 5 --expert-budgets 1 2 --random-seeds 101

bash scripts/gpu_queue.sh sva-eval-neurons -- python scripts/relp_sva_run.py evaluate \
  --model-dir "$RELP_MODEL_DIR" --manifest-dir "$RELP_SVA_MANIFEST" --task "$RELP_SVA_TASK" \
  --verification "$RELP_SVA_RUN/verify/verification.json" --score-dir "$RELP_SVA_RUN/score" \
  --unit neuron --split validation --output-dir "$RELP_SVA_RUN/validation_neuron"

bash scripts/gpu_queue.sh sva-eval-experts -- python scripts/relp_sva_run.py evaluate \
  --model-dir "$RELP_MODEL_DIR" --manifest-dir "$RELP_SVA_MANIFEST" --task "$RELP_SVA_TASK" \
  --verification "$RELP_SVA_RUN/verify/verification.json" --score-dir "$RELP_SVA_RUN/score" \
  --unit expert_position --split validation --output-dir "$RELP_SVA_RUN/validation_expert"
```

若要测量跨位置的完整 expert，另运行 evaluate，把 `--unit` 改为 `expert_all_positions`，使用新的输出目录。它仍受 score 时的位置池约束；例如 final 池只干预最终位置。

每一步须等前一步成功；所有输出目录必须不存在。然后在 nounpp 重复 smoke。正式实验换用 paper/grouped manifest、新输出目录，四任务分别验收/评分；使用正式预算，完成 validation 后锁定计划，再以 `--split test` 输出到新的 test 目录。不要把前面 smoke 的小预算带到正式结果后称为完整预算扫描。

## 8. 已验证与待运行的界限

已下载并校验 8 份数据与固定 tokenizer；已生成三套实际 manifest。CPU 使用随机初始化的真实 HF OLMoE 检查 native/functional 对齐、patch 恒等式、neuron→expert 分数、符号汇总与随机对照，并运行 `verify → score → evaluate` 的完整 CLI 流程；还检查错误验证记录会阻止评分。11 项测试通过，记录及文件哈希见 [validation_report.json](../data/relp_sva/validation_report.json)。三套清单重建后的 JSONL 与交付文件逐字节一致。

未下载 7B 权重，未运行正式 GPU 实验，没有新排名/效果结论。全专家 FModel 的成本仍高于原生稀疏 forward；先在短样本上检查显存。完整 checkpoint 的验收失败时应修实现或解释数值原因，不能用 CPU 小模型的通过记录代替。

Agent 最终交付：数据 manifest/选择审计、当前环境验收记录、neuron 与两种 expert 排名、完整 score 均值、锁定计划、逐例干预数据、四任务分开及宏平均的结果表。标明 paper/grouped、zero/mean、neuron/expert-position，区分恢复差异、必要性与保留集合后的 faithfulness。
