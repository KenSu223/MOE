# RelP → MoE implementation guide

更新：2026-11-07。本文把已保存的讨论、已有 pilot 代码和最小执行入口连接起来。

**当前研究目标：用 RelP 找到对指定目标重要的 neurons 和 experts，并分别给出它们的归因分数与排名。** 本阶段以关键单元识别为交付，不要求提取完整的节点间连接或完整 circuit。

**当前可执行范围：OLMoE 的神经元归因，以及 Mistral dense 对照。** 归因坐标是路由加权后的专家中间激活。这里不是一个支持任意 MoE 的通用库，也尚未完成 router 决策归因。

**SVA 执行补充（2026-11-07）：** 要在论文 SVA 任务上识别 neuron/expert，请结合 [SVA 实验协议](relp_sva_experiment_protocol.md) 使用。已补充固定数据版本、OLMoE token 对齐后的实际样本清单、跨样本有符号汇总、独立 expert 排名、原生一致性验收和 q-level 干预入口。下面第 5 节仍介绍原来的单例 demo；正式实验使用新入口 `scripts/relp_sva_run.py`。

## 1. 之前的内容保存在哪里？

| 材料 | 内容和状态 |
|---|---|
| [完整数学教程](../../papers/2601.22594_math_moe/tutorial.md)；[HTML 阅读版](../../papers/2601.22594_math_moe/tutorial.html) | 第 11–17 节详细解释 MoE 前向、冻结 router、half rule、数值例子、路由/内容分解、实验流程和代码对应。已有记录。 |
| [数学例子检查](../../papers/2601.22594_math_moe/verify_math.py) | 22 项 NumPy 小例子；不加载模型，不能代替真实模型验证。 |
| [FModel 核心实现](../moetrace/ext14_circuit_relp.py) | 已有 OLMoE/Mistral functional forward、RelP 反向规则、节点 patch。 |
| [历史 pilot driver](../scripts/ext14ci_relp_run.py) | 已有多种归因比较和 intervention 验证；依赖原服务器路径、数据与前序实验。 |
| [历史 forward 验证](../scripts/ext14ci_relp_verify.py)；[分析脚本](../scripts/ext14ci_relp_analyze.py) | 已有；验证脚本输出误差，不会根据误差阈值自动判失败。 |
| [已有实验报告](../results/sections/ext14_circuit.md) | C2 部分是此前的小规模实验记录，不是本次新跑的结果。 |
| [最小打分入口](../scripts/relp_score_pair.py) | 本次补充；直接接收 checkpoint 路径和 token IDs，不依赖历史数据集或 `/home/ubuntu/MOE`。本次只完成静态及输入检查，未运行模型。 |

建议执行顺序：先读第 2–4 节确认归因定义，按第 5 节跑一个输入对，再按第 6 节验证实现，最后决定是否复现第 7 节的历史 pilot。若要做新架构，先完成第 9 节的适配。

## 2. 先固定我们究竟要算什么

### 2.1 一个 forward 里有两个候选 token 的 logits

设主输入为 A，参考输入为 B。A 可以是 clean prompt，B 是经过最小替换的 counterfactual prompt。一个 prompt 的最后位置会输出整个词表的 logits；我们从**同一次 forward 的同一个位置**取两个数：

$$
F(A)=z_{\mathrm{target}}(A)-z_{\mathrm{foil}}(A).
$$

例如 SVA 的 target 是 ` are`，foil 是 ` is`。这是两个词表坐标，不是分别把两个词追加到 prompt 后再运行。A、B 两端必须使用同一组 target/foil，不能在 B 端交换正负定义。多 token 答案需要另行定义指标；本入口只接受单个 next-token ID。

### 2.2 MoE 节点使用 q = 路由权重 × 原始神经元激活

对 layer ℓ、position t、expert e、neuron j：

$$
h_{\ell t e}=\operatorname{SiLU}(W_{g,e}x_{\ell t})\odot W_{u,e}x_{\ell t},\qquad
q_{\ell t e j}=g_{\ell t e}h_{\ell t e j}.
$$

这里 g 已包含原模型的 top-k mask，以及原模型本来采用的归一化；未选中的专家取 g=0。模块输出为

$$
o_{\ell t}=\sum_e W_{d,e}q_{\ell t e}.
$$

所有样本共用 `(layer, position, expert, neuron)` 坐标。**不能把 clean 的 top-k 第 1 个槽位与 reference 的 top-k 第 1 个槽位当成同一个专家。** 要按 expert ID 对齐。

| 对象 | MoE shape，单输入 B=1 |
|---|---|
| `input_ids` | `[1, T]` |
| 每层 `topi`, `topw` | `[T, k]` |
| scatter 后的 g | `[T, E]` |
| 原始专家激活 h | `[T, E, I]` |
| `q[layer]` | `[1, T, E, I]` |
| 输出投影权重 | `[E, H, I]` |
| `beta[layer]` | 与 q 相同 |

I 是每个专家的中间维度，H 是 residual 维度。dense 对照的 q 为 `[1,T,I]`，最小入口输出时把 expert 坐标记为 0。

### 2.3 paired 与 unpaired 是两种不同的问题

记 β 为按照 RelP 规则传播得到的系数，也就是代码里的 modified/pseudo-gradient：

$$
\beta^A_n=\widetilde{\frac{\partial F(A)}{\partial q_n}},\qquad
R^{\mathrm{paired},A}_n=(q^A_n-q^B_n)\beta^A_n,\qquad
R^{\mathrm{unpaired},A}_n=q^A_n\beta^A_n.
$$

- paired 问：相对于这一个参考输入，哪些有效激活的变化值得关注？
- unpaired 问：在这一次运行中，相对于节点零值基线，哪些有效激活贡献较大？这里没有实际运行一个“零输入句子”。

paired 默认推荐把 A 设为 clean，并明确报告系数取自 clean 端。若要与 corrupted→clean 的 rescue 更直接对应，可以另外报告 `(q_clean-q_corrupt) * beta_corrupt`，标注 endpoint。两者不应混为同一分数。

**不能再额外乘一次 g。** 因为 q 已经含 g。使用 `q * dF/dh` 会重复计算路由权重；这里应该用 `q * dF/dq`。

未路由的专家在某一端 q=0，但在扩展的 q 坐标上 β 可以非零：如果 λ 是输出 o 收到的下游系数，则

$$
\beta_{e}=W_{d,e}^{\mathsf T}\lambda.
$$

因此，paired 归因要保留两端活跃专家的并集。只记录 A 端活跃专家，会丢失 B 有、A 没有的贡献。unpaired 的这些 A 端不活跃节点分数则为零。

### 2.4 从 neuron 分数得到 expert 分数

先固定同一层 ℓ、同一 token 位置 t 和同一个系数端点 A。下面省略 ℓ、t；不同层的同号 expert ID 表示不同专家。令 $d_{e,j}$ 为 $W_{d,e}$ 的第 j 列，$c_e=W_{d,e}h_e$ 为专家未加权输出，$\lambda^A$ 为从最终目标传回该 MoE 输出的 RelP 系数。于是

$$
\beta_{ej}^A=d_{e,j}^{\mathsf T}\lambda^A.
$$

专家 e 的 unpaired 分数就是它内部所有神经元分数的有符号和：

$$
\begin{aligned}
R_e^{\mathrm{unpaired},A}
&=\sum_j q_{ej}^A\beta_{ej}^A\\
&=\left(\sum_jq_{ej}^Ad_{e,j}\right)^{\mathsf T}\lambda^A\\
&=(W_{d,e}q_e^A)^{\mathsf T}\lambda^A
=(g_e^Ac_e^A)^{\mathsf T}\lambda^A.
\end{aligned}
$$

paired 采用同样的加和方式：

$$
\begin{aligned}
R_e^{\mathrm{paired},A}
&=\sum_j(q_{ej}^A-q_{ej}^B)\beta_{ej}^A\\
&=\bigl[W_{d,e}(q_e^A-q_e^B)\bigr]^{\mathsf T}\lambda^A\\
&=(g_e^Ac_e^A-g_e^Bc_e^B)^{\mathsf T}\lambda^A.
\end{aligned}
$$

为什么能加起来？因为 down projection 将各神经元的写入向量线性相加，而它们使用同一个下游方向 $\lambda^A$；对每个写入分别做点积再相加，等于先合成专家写入再做点积。这里不需要再跑一次专家级 backward。实现时用 q 的表达式即可；某端 g=0 时，该端专家的有效写入为零，不必为了这两个等式额外计算未执行专家的原始内容。

**先完整求和，再按专家排名。** 可以按 $|R_e|$ 找净归因幅度最大的专家，同时保留 $R_e$ 的正负号。$|\sum_jR_{ej}|$ 与 $\sum_j|R_{ej}|$ 不同：例如两个神经元分别为 +5、−5，专家净分数为 0，绝对分数之和为 10。后者可以单独报告为内部绝对归因总量，不能替代专家净分数。

当前这些分数对应 `(layer, position, expert)`。若最终需要每个 `(layer, expert)` 的排名，可以预先确定 token 候选范围，再将该范围内各位置的专家有符号分数相加，并注明位置范围与聚合方式。这是在汇总该专家不同位置的归因，不是把不同层节点相加来分解整个模型输出。跨样本排名也需要明确是逐输入排名，还是按固定的数据聚合规则形成共享列表。

这些是 RelP 分数的代数汇总，**不保证等于真实移除整个专家的输出变化**。移除会改变后续非线性计算和路由；“关键”在归因阶段指相对于指定目标、基线和端点的高分候选。

**当前代码状态：** 最小入口计算了完整神经元分数，但仅导出 neuron top-K，尚未导出 expert 排名。要得到 expert 分数，应在每层神经元 top-K 截断前，沿 neuron 维度执行 `expert_score = neuron_score.sum(dim=-1)`，再独立进行专家排名与导出。只把 `scores_top.json` 里入选神经元相加，会遗漏同一专家中未导出的正、负项，不能视为完整专家分数。

## 3. RelP backward 在代码中怎么实现？

### 3.1 前向仍取当前运行的数值，反向换传播规则

现有 `FModel` 在每一层做这些处理：

| 部件 | 前向值 | RelP backward |
|---|---|---|
| RMSNorm | 使用当前输入计算的分母 | 分母 detach，包括 OLMoE 的 Q/K norm |
| Attention | 当前输入的 attention pattern | pattern detach，保留 V 路径 |
| SiLU | `silu(a)` | sigmoid gate detach |
| SwiGLU 乘法 | 两分支乘积 | 两个分支各分一半 relevance |
| Router | 当前输入自己的 top-k 和权重 | 选择与权重 detach |

乘法的实际实现为：

```python
sg = torch.sigmoid(gate_pre).detach()
s = gate_pre * sg
s = s + (torch.nn.functional.silu(gate_pre) - s).detach()
act = 0.5 * (s * up.detach() + s.detach() * up)
q = router_weight.detach()[..., None] * act
```

前向的两个乘积相同，其平均仍是原乘积；反向一个分支只承担一半。这就是 half rule。冻结的 router weight 是常数，不是第三条需要分 relevance 的可变乘法分支，因此不是除以 3。数值精度下的前向一致性仍须实测。

### 3.2 两端的路由不需要相同

默认 `route=None`：A 用 A 的自然路由，B 用 B 的自然路由。`detach()` 只表示不沿 router 反传，不表示把两个 forward 的路由锁成一样。

只有显式传入 `route=reference_routes` 才是在另一输入上强制采用指定的专家和权重。那已经改变了问题，最小打分入口不这么做。

### 3.3 从 logit 差怎样传到 q？

若最后归一化的 residual 为 r，则

$$
F=(W_U[\mathrm{target}]-W_U[\mathrm{foil}])^{\mathsf T}r.
$$

所以反向的初始方向就是两个 unembedding 向量之差。随后沿计算图按上述规则传播，直至得到每层的 β。代码使用

```python
main = fm.forward(ids, [target_id], [foil_id], relp=True, grad=True)
beta = torch.autograd.grad(main["delta"].sum(), main["q"])
```

`grad=True` 会让 embedding 输出需要梯度；虽然所有模型参数被冻结，激活仍然可以反传。`delta.sum()` 是为了得到一个标量，本入口 batch size 是 1。

**这里的 β 不是原模型普通导数。** `q*β` 不自动等于普通 `activation × gradient`；更不意味着 paired 分数精确积分了两个输入之间所有非线性和路由切换。不要用原模型有限差分去要求 RelP 的 pseudo-gradient 与真实导数相等。

### 3.4 完整边界是什么？上游分数怎样考虑后续层？

先看一个 MoE 输出，固定一次运行并省略端点标记。由第 2.4 节的线性加和，有

$$
\sum_eR_e=\sum_{e,j}q_{ej}\beta_{ej}=o^{\mathsf T}\lambda.
$$

右侧是**这个 MoE 输出分支**获得的归因，不一定是最终指标 F。若该分支进入残差加法 $r_{\mathrm{next}}=r+o$，加法输出收到系数 $\lambda$，则两个输入分支获得的份额为

$$
r^{\mathsf T}\lambda+o^{\mathsf T}\lambda
=(r+o)^{\mathsf T}\lambda
=r_{\mathrm{next}}^{\mathsf T}\lambda.
$$

这里 $r^{\mathsf T}\lambda$ 只表示 **r 直接进入加法的旁路份额**。r 还可能通过 norm 和 MoE 产生 o；r 节点收到的总系数会包含这条额外路径。不能拿 r 的总归因再与 o 的归因相加，否则会重复计算经过 MoE 的份额。

**完整边界是所分析的 RelP 传播图上的一个完整切面，不必等于某一层。** 在这个切面上，每条通向目标的贡献路径被计入一次，既不遗漏旁路，也不同时计入同一路径上的上游和下游。切面可以由节点或明确的分支边组成。某个 block 之后、所有 token 的完整 residual 状态是自然的例子；一层的 MoE 神经元集合通常遗漏 residual 旁路，单个 token 的状态也可能遗漏经后续 attention 从其他位置传来的贡献。

如果每个局部运算都按指定规则守恒，且所有相关路径和偏置等来源均已计入，则零参考的 unpaired 归因在完整边界上的总和可以保持为最终目标 F。这里只对规定的 RelP 传播规则成立，不能据此把任意选出的 neurons/experts 分数之和解释成整个输出。

上游分数与下游的联系来自同一次反向传播。将所有 token 的 residual 拼成向量，记第 ℓ 个 block 为 $r^{\ell+1}=f_\ell(r^\ell)$，其局部修改后反向算子为 $\widetilde J_\ell^{\mathsf T}$，则

$$
\lambda^\ell=\widetilde J_\ell^{\mathsf T}\lambda^{\ell+1}.
$$

从最终 logit 差开始，逐层使用这个递推，得到每一层的系数。因而早期神经元的 $\beta$ 已包含它通过后续 attention V 路径、MLP/MoE 和残差路径对目标的归因关系，不是仅凭当前层激活大小打分。不同 token 间的联系也包含在 attention 的传播中。被冻结的 QK pattern 和 router 决策路径不在该反向解释范围内。若直接用最终 unembedding 方向代替中间层的下游系数，通常得到的是 DLA，不能一般地代替 RelP。

**paired 分数另有一个限制：完整边界也不能保证单端点 paired RelP 等于真实输出差。** 在同一 MoE 模块上，第 2.4 节只保证

$$
\sum_{e,j}(q_{ej}^A-q_{ej}^B)\beta_{ej}^A
=(o^A-o^B)^{\mathsf T}\lambda^A.
$$

右侧使用 A 端的一个系数，不一般等于 $F(A)-F(B)$。最小反例是 $F(x)=x^2$，取 A=2、B=1：half rule 在 A 的系数为 2，paired 分数为 $(2-1)\times2=2$，真实输出差却是 $4-1=3$。

对于本阶段寻找关键 neurons 和 experts 的目标，可以在不同层、不同位置比较分数并选出高分单元，不需要先构造一个完整切面。完整边界约束的是“何时可以把分数总和解释成输出总量”；跨层排名本身不要求总分等于 F。

## 4. 到底需要几个 forward？

| 工作 | forward 数 | backward 数 | 说明 |
|---|---:|---:|---|
| 最小 unpaired | 1 | 1 | A 的 forward 同时保留反向图 |
| 最小 paired，系数取 A | 2 | 1 | B 缓存 q；A forward + backward |
| 在 paired 上同时输出 unpaired | 仍为 2 | 仍为 1 | 复用 qA 和 βA，只增加逐元素乘法 |
| 两端各取一次系数 | 2 | 2 | 需分别保留/建立两端的图；额外实现或缓存策略可能增加 forward |
| 历史 pilot 的每样本核心打分 | 5 | 3 | 缓存两端，再分别调用 clean RelP、corrupt RelP、corrupt AtP |

历史 pilot 还单独运行均值收集和大量 patch/ablation forward，因此总成本远超过表中的核心打分。

`FModel.node_grads()` 内部会自己再做一次 forward。因此如果你已经拿到了 `main`，应直接调用 `autograd.grad`，不要再调用 `node_grads()`。新入口已按最小流程实现。

### 4.1 “base”与实际执行顺序不要混淆

当前入口的实际顺序是：**先跑 B=`reference_ids` 并缓存 qB；再跑 A=`input_ids` 并保留计算图；只对 A 的 F 做一次 RelP backward。** 最终是 `(qA−qB)*betaA`。如果把原始、被解释的输入 A 叫作 base，那么系数确实取自这个 base；但它在当前代码中是第二个执行的 forward。如果把差值参照 B 叫作 baseline，那么当前系数并不取自这个 baseline。

两次 forward 使用相同的模型参数 θ，没有训练或更新权重。不同的是输入触发的激活、路由权重、attention pattern、RMSNorm 分母和 sigmoid gate；本入口的 backward 使用 A 端记录的这些量及 RelP 传播规则。B 端贡献参考激活 qB，不把 B 端的冻结量用于 A 端反传。

在确定性 eval 模式下，也可以先跑 A 并保留它的完整计算图，再以 no-grad 跑 B，最后对 A 反传，公式与结果不变。决定数学含义的是“在哪一端取系数”，不是墙钟时间上谁先执行。若固定差值方向 `qA−qB`、改成取 betaB，得到的是另一个 endpoint 的评分，需要显式修改实现；直接交换输入字段还会把差值符号一起反转。

## 5. 可直接照着执行的最小实验

### Step 1：确定目录、环境和 checkpoint

当前 Mac 项目代码在：

```bash
cd /Users/kensu/Downloads/transferred_downloads/MOE/MOE
python3 scripts/relp_score_pair.py --help
```

实际模型运行需要已安装 `torch`、`transformers`、`accelerate`、`numpy` 的环境，以及完整本地 checkpoint。后续 SVA 补充提供了在 CPU 小型 HF OLMoE 上验证的 [依赖版本](../requirements-relp-sva.txt)，建议另建隔离环境；服务器还需匹配 CUDA 的 torch wheel。不同 transformers 版本的 OLMoE expert 布局可能不同，不能以“安装成功”代替兼容性验证。

OLMoE 历史使用的 checkpoint 是 `allenai/OLMoE-1B-7B-0125`。**填写具体 snapshot 目录**，包括 `config.json`、权重和 tokenizer 文件；不要填写 Hugging Face 仓库名。记录 snapshot revision，避免缓存目录更新后悄悄换模型。

```bash
export RELP_MODEL_DIR=/absolute/path/to/OLMoE-1B-7B-0125/snapshots/EXACT_REVISION
```

原单例入口最初只做了语法和数据契约预检查。后续 SVA 补充已在本机隔离环境安装 torch/transformers，并用随机初始化的小型 HF OLMoE 完成 11 项测试；未下载或测试完整 7B checkpoint。7B 级别实际运行安排在 GPU 环境，小型模型测试不能代替该 checkpoint 的验收。

### Step 2：准备一个输入对

下面使用真实 tokenizer 生成 IDs，不手填示意数字。这个 SVA 例子是入口演示，不是论文实验数据集复现。

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["RELP_MODEL_DIR"], local_files_only=True)
clean_text = "The parents"
reference_text = "The parent"
clean = tokenizer.encode(clean_text, add_special_tokens=True)
reference = tokenizer.encode(reference_text, add_special_tokens=True)
assert len(clean) == len(reference), "Choose an equal-length, position-aligned pair"

def next_id(prompt, prefix_ids, continuation):
    combined = tokenizer.encode(prompt + continuation, add_special_tokens=True)
    assert combined[:len(prefix_ids)] == prefix_ids, "Continuation changed prefix tokenization"
    suffix = combined[len(prefix_ids):]
    assert len(suffix) == 1, "This entry point requires a single next-token candidate"
    return suffix[0]

target = next_id(clean_text, clean, " are")
foil = next_id(clean_text, clean, " is")
assert target == next_id(reference_text, reference, " are")
assert foil == next_id(reference_text, reference, " is")
payload = {
    "input_ids": clean, "reference_ids": reference,
    "target_id": target, "foil_id": foil,
    "input_text": clean_text, "reference_text": reference_text,
}
with Path("relp_pair.json").open("x") as output:
    output.write(json.dumps(payload, indent=2) + "\n")
print(tokenizer.convert_ids_to_tokens(clean))
print(tokenizer.convert_ids_to_tokens(reference))
PY
```

如果 assertion 失败，需要换一组满足 tokenizer 约束的输入，不能截断或插入 padding 来凑长度。本实现没有 attention mask/padding 支持；也不会自动识别语义位置对齐。相同 token 数只是必要条件，仍要检查打印出的 token 序列。

### Step 3：不加载权重，先验证输入契约

```bash
python3 scripts/relp_score_pair.py \
  --model-dir "$RELP_MODEL_DIR" \
  --input-json relp_pair.json \
  --validate-only
```

这一步只需 Python 标准库，检查 token ID 范围、两个候选不同、两端长度以及基本架构限制。它不验证 tokenizer 对齐、权重布局、模型输出或 GPU。

### Step 4：在 GPU 环境运行一次 paired + unpaired

原实验服务器的 GPU 任务通过项目队列运行；队列会使用项目 `.venv`，因此 checkpoint 和输入文件必须在该服务器可读。

```bash
bash scripts/gpu_queue.sh relp-pair-demo -- \
  python scripts/relp_score_pair.py \
  --model-dir "$RELP_MODEL_DIR" \
  --input-json relp_pair.json \
  --output-dir results/relp_pair_demo_001 \
  --device cuda --dtype bfloat16 --top-k 100
```

`--output-dir` 必须是新目录，入口拒绝覆盖旧运行。使用 `--dtype float32` 可做小模型/短输入的精度对照，但增加内存需求。此脚本仍使用计算所有专家的旧 FModel，不保证大模型或长序列在任意显卡上装得下。

要只做 unpaired，在输入 JSON 中**删除** `reference_ids` 字段，并换一个输出目录。不要把它设为空列表或 null。此时只运行 1 个 forward + 1 个 backward。

### Step 5：检查输出的含义

| 文件 | 怎么读 |
|---|---|
| `run_meta.json` | checkpoint 路径、config/代码哈希、软件版本、两端 F、forward/backward 数、归因 endpoint、top-k 变化数量 |
| `input.json` | 实际输入 IDs、target/foil 和自带元数据 |
| `scores_top.json` | paired/unpaired 各自的前 K 个节点，按绝对分数排序，保留分数正负号和具体坐标 |
| `routes.json` | 两端每层每个 token 的专家 ID 和权重；dense 是空列表 |

一个节点记录包含 `layer, position, expert, neuron, score, q_input, q_reference, beta_input`，坐标从 0 开始。正分数表示在这个 endpoint 和基线定义下，方向上支持 `target−foil`；负分数方向相反。它不是概率，也不是已实测的 rescue 大小。

入口默认从所有位置、所有层的神经元里选前 K。若限定 final position、替换词之后或排除 BOS，候选池过滤必须发生在每层及全局 top-K 截断之前。当前 CLI 没有候选池参数，也不保存完整分数；不能只过滤已输出的 `scores_top.json`，就把结果当作受限候选池的真正 top-K。需要增加候选 mask 后重新运行，或者增加完整分数导出后再排序。不同层节点可能位于同一传播链，**不能把跨层所有分数相加称作输出 F 或其变化的分解**；具体原因见第 3.4 节。

`changed_topk_sets_across_layer_positions` 仅统计集合改变，不包括集合相同但权重改变；权重变化要查看完整的 `routes.json`。这个入口只做打分，没有 ablation、patching 或 circuit faithfulness 评估。

## 6. 在科学结论之前，必须完成哪些验证？

按下列顺序检查。前两项不通过时，先修 forward，不开展大规模归因比较。

| 检查 | 具体操作 | 判定 |
|---|---|---|
| 原生 HF vs functional | 同样 IDs、dtype、eval 模式；逐层比较 MoE 输出、两端 top-k 与权重、最终 residual 和指标 | 用逐层绝对/相对误差定位问题；路由差异单列，不只看最终 F |
| RelP on/off 的 forward | 同样 IDs，比较 `relp=False` 与 `relp=True` 的 q、模块输出和 F | 应在预先设定的数值容差内一致；不能把额外差异解释为 attribution 效果 |
| 反向规则 | 用小型可解析乘法验证 half rule 和固定路由局部守恒 | 两个乘法分支各分一半；g 不重复乘 |
| q 节点覆盖 | 用 A/B 选择不同专家的例子；检查不活跃节点 q=0、并集节点保留 | 不能因 A 未路由就删掉该专家的 paired 项 |
| 完整节点 patch 恒等式 | 同一层同一位置，把所有 q 替换成来源值；与替换整个 MoE 输出比较 | 两次 intervention 的输出应在容差内相同 |
| 实际因果效果 | 对若干节点和 top-N 集合运行明确规定的 patch/ablation | 比较预测分数与真实 ΔF；另外报告路由变化 |

设定容差时固定 dtype、计算顺序和读出方式。现有 FModel 用 fp32 对两个 unembedding 向量之差读出；直接从 HF 的 bf16 全词表 logits 相减，可能包含不同的舍入误差。需要逐层核对后才能归因误差来源，不能一概称作 bf16 噪声。

历史验证文件在 [OLMoE](../results/p6ci_relp_verify_olmoe.json) 和 [Mistral](../results/p6ci_relp_verify_mistral7b.json)。例如 OLMoE 存档报告 `max_abs_delta_f_vs_hf≈0.165`、`max_mlp_out_rel≈0.187`，同时 `max_abs_delta_relp_vs_plain=0`。这说明“切换 RelP 没改变该实现的前向值”和“该实现严格匹配原生模型”是两个独立问题。**这些存档不能直接作为当前依赖环境的通过证明；约 18.7% 的最大局部相对误差也需要定位后解释。**

原验证脚本只输出统计，退出码 0 不等于所有数值都通过。其最后的 finite-gradient 检查还把 clean 路由传给 corrupt 输入，检验的是一个指定路由配置下能否反传；这不能替代两端自然路由的正确性测试。

用于更大实验时，若需要均值消融，用独立的 calibration 数据估计均值。若目标是跨样本共享的关键 neuron/expert 列表，在 discovery 数据上选择或聚合单元，在 validation 数据上确定预算与选择规则，锁定后再用 held-out test 评估。不能在最终 test 上一边调整列表、一边用同一批结果声称共享列表的泛化。若目标是逐输入寻找关键单元，可以在每个测试输入上按事先固定的算法选点，但应报告为逐输入排名或集合。这里要求数据用途和最终测试独立性清楚，不强制使用四个物理数据文件。

## 7. 如何复现已有的历史 pilot？

这条路线适用于原 Linux 布局 `/home/ubuntu/MOE`，并且 checkpoint 缓存、`.venv` 和前序数据都已准备好。**它不是把当前 Mac 目录 cd 进去就能运行的脚本。** `moetrace/models.py`、`ext14_circuit.py` 等仍有服务器绝对路径；单改 driver 顶部一行 `sys.path` 不够。

先在单独的实验副本中执行，保留原结果。旧 driver 会写固定输出目录，旧 analyze 会更新汇总文件，没有新入口的防覆盖保护。

以 OLMoE/CounterFact 为例，需要已有：

- `results/p6ci_c3_data/case_sets_cf_mix_olmoe.json`
- `results/p6ci_c3_data/cf_mix_olmoe.parquet`
- `HF_HOME` 下的 OLMoE snapshot，或原项目既有缓存配置。

先验证并阅读误差报告：

```bash
cd /home/ubuntu/MOE
bash scripts/gpu_queue.sh relp-verify-olmoe -- \
  python scripts/ext14ci_relp_verify.py olmoe cf
cat results/p6ci_relp_verify_olmoe.json
```

确认第 6 节要求后，再运行 3-case smoke：

```bash
bash scripts/gpu_queue.sh relp-smoke-olmoe -- \
  python scripts/ext14ci_relp_run.py olmoe cf --smoke
.venv/bin/python scripts/ext14ci_relp_analyze.py --smoke
```

检查 smoke 文件中的非有限值、归因与 patch 定义、候选池和指标分母；之后才运行 48-case pilot：

```bash
bash scripts/gpu_queue.sh relp-pilot-olmoe -- \
  python scripts/ext14ci_relp_run.py olmoe cf --n-cases 48
.venv/bin/python scripts/ext14ci_relp_analyze.py
```

长任务按 [PLAYBOOK](../PLAYBOOK.md) 放入持久会话/脱离终端的任务链。分析脚本会扫描匹配的所有历史运行，不一定只分析刚完成的一次。

当前组合支持 OLMoE 的 `cf/wino/ioi` 和 Mistral 的 `cf/wino`，但各自需要不同的已有数据和前序结果。Mistral loader 还依赖 C1 的筛选记录，不是任意输入文本的接口。

输出目录是 `results/p6ci_relp_<model>_<task>[_smoke]/`，包含 `relp_cases/grid/focal/multi/circuits/nodes_top.parquet`、`run_meta.json` 和完成标记。汇总在 `results/tables/ext14ci_relp_*.md` 等文件中。

### 7.1 读 pilot 时必须知道的协议差异

| pilot 行为 | 对结论的限制 |
|---|---|
| 同时算 clean/corrupt RelP 与 corrupt AtP | 不是最小 2-forward 流程 |
| `relp=False` 也 detach router | 这里的 AtP 对照仍然冻结 router backward，不是包含 router 全路径的普通梯度 |
| 均值来自同一组所选案例的 clean+corrupt，排除位置 0 | 不是独立 calibration 数据估计的均值 |
| 均值统计的是已加权 q，并用激活非零作为 routed proxy | 不等价于先估原始 h 的条件均值、再乘当前 g；严格实现应从 top-k mask 计数 |
| paired circuit 用 clean/corrupt 的 q 直接互换 | 属于有效贡献 patch，可能混合两端的路由权重 |
| mean ablation 限制在当前 run 被选中的专家上 | 避免把未路由专家直接注入均值，但仍是 q 空间消融 |
| intervention forward 默认重算后续路由 | backward 冻结路由并不阻止 intervention 引起路由切换 |
| 旧 `n_switch` 只看后续层 final token 的 top-k 集合 | 不覆盖所有位置，也不覆盖仅权重改变 |
| recovery 与 mean-ablation faithfulness 各有基线 | 不能用小 circuit 的高 recovery 宣称相同大小的高 faithfulness |

## 8. 关于权重和大于 1：打分与验证要分开

**最小归因阶段没有把 A 的专家贡献塞进 B。** 两个 forward 各自按原模型计算，所以不会因为归因计算本身制造混合权重；`qA−qB` 只是在 forward 完成之后比较两个记录。

还要检查模型本来的归一化定义：某些架构对 top-k 权重重新归一化，某些架构保留全专家 softmax 后选中的权重，后者的和可能小于 1。适配时保留原定义，不能统一强制 sum=1。现有 FModel 遵循 `norm_topk_prob`。

但如果之后验证时只替换某个专家的已加权贡献：

$$
o_B^{\mathrm{patch}}=o_B-g^B_e u^B_e+g^A_e u^A_e,
$$

在两端各自权重和为 1 的前提下，混合后的形式系数和会是 `1−gB_e+gA_e`，可能大于或小于 1。这类实验可以定义成“专家输出贡献的替换”，却不能描述成一个重新运行 router 得到的自然混合。自动重新归一化会同时改变其他专家，变成另一个 intervention。

若研究问题是“保持 B 的路由，只替换该专家算出的内容”，应定义：

$$
o_B^{\mathrm{content\ patch}}
=o_B+g^B_e W_{d,e}(h^A_e-h^B_e).
$$

该内容干预对应的逐神经元 RelP 候选分数是 `gB_e * (hA_e−hB_e) * betaB_e`，沿 neuron 维度求和得到专家分数；它与完整的 `(qA−qB)*betaB` 评分不同。β 是修改后的反向系数，不能把该分数直接称为原模型真实的一阶 Taylor 近似。若要计算真实的一阶项，必须在相同干预定义下使用普通导数，并要求当前局部可微。B 端未选中该专家时 gB=0，这个 intervention 不会激活它。要这样验证，需要缓存/计算原始 h；**不能用 qA/gA 去恢复 A 端没有选中专家的 h**。当前最小入口不实现这种验证，历史 pilot 的 q-patch 也不是它。

若进一步做路由/内容的四组合实验，需明确固定的 layer/position、下游上下文，以及如何获得两端所有所需专家的 h。数学教程第 15 节已有推导，但本指南没有将这部分标记成已完成代码。

## 9. 扩到其他 MoE 或大模型，要改什么？

### 9.1 架构适配

不能只把 `FModel("olmoe")` 换成其他模型名。当前代码依赖 OLMoE fused expert weights `[E,2I,H] / [E,H,I]`、特定 Q/K norm 和无 bias 的投影。Qwen、Mixtral、含 shared experts 的模型不在此入口支持范围内。

适配清单：

1. 对齐原生 attention、RoPE、norm、mask、activation、bias 和 shared-expert 分支，先跑通逐层前向一致性。
2. 对齐 router logits 精度、top-k、权重归一化与可选缩放。显式记录 native routing；不要从不完整配置猜默认值。
3. 明确 q 坐标以及输出投影方向，验证 `o = sum_e Wdown_e q_e`。
4. 仅对已推导的运算应用 RelP 规则。额外 clipping、激活或乘法不能无审查地套用 half rule。
5. 检查脚本的 module/layout guards，并完成第 6 节后才开放新的 `model_type`。

### 9.2 从研究原型变成稀疏实现

现有 FModel 显式计算所有 E 个专家。激活和梯度的保存量随 `L×T×E×I` 增长，失去了稀疏 MoE 的一部分效率优势；一次 forward 的成本也不能直接等同原模型稀疏 forward。

优化方向是：每端只计算其被选中的专家，保存两端活跃集合的并集；在每层 MoE 输出取得下游系数 λ 后，用 `beta_e = Wdown_e.T @ lambda` 为需要的 q 坐标计算系数。A 未路由而 B 路由的节点仍能得到 β。不要因为想保持稀疏就直接删除这类 paired 节点。

这项优化尚未实现。它应与当前显式全专家版在小模型上逐节点比较，再测速度和显存。

## 10. 当前完成边界与交接清单

**已经有的：** 完整数学笔记；OLMoE/Mistral RelP prototype；历史小规模打分与干预报告；本次新增的 portable 单例入口和本指南。

**SVA 补充已实现：** 通过 `scripts/relp_sva_prepare.py` 锁定来源并生成样本清单；通过 `scripts/relp_sva_run.py` 在 top-K 前汇总全量分数、导出 neuron/expert 排名、显式选择位置池、验收原生前向并评估干预。已保存校准、验证和测试数据用途及重叠审计，详见 [SVA 协议](relp_sva_experiment_protocol.md)。原单例 `relp_score_pair.py` 的功能边界保持不变。

**仍待实际运行：** 在完整 OLMoE checkpoint / GPU 上通过当前环境的一致性验收，然后产出正式任务的排名和干预结果。已实现 runner 不等于真实模型已经通过验证；CPU 小型模型检查也不能代替它。

**本阶段不要求完成：** 完整 edge circuit 提取、router 决策归因和纯 route/content 四组合实验。其他 MoE 架构的兼容及稀疏高效实现按后续需要扩展；当前文档没有将它们标记为已实现。

原单例文档/入口验证记录：Python 语法和 `--help` 检查；paired/unpaired 输入契约及错误输入检查；教程 22 项数学小例子的重跑。后续 SVA 补充已在隔离 CPU 环境安装依赖、下载固定 tokenizer 和数据，使用随机初始化的小型 HF OLMoE 完成集成检查；没有下载 7B 权重或启动 GPU 正式实验。

给下一位实现者的执行顺序：**一个短输入对 → 检查实际 token → native/functional 逐层对齐 → 2-forward/1-backward 打分 → 验证少量节点 → 明确定义消融与路由策略 → 扩展数据集。** 每个 run 记录 checkpoint revision、环境版本、归因 endpoint、候选节点池、baseline、干预对象和指标定义。
