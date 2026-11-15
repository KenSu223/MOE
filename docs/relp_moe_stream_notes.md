# RelP → Qwen3-30B-A3B / Mixtral-8x7B：流式实现说明与验收记录

2026-11-07。配合 [实现 guide](relp_moe_implementation_guide.md) 与 [SVA 协议](relp_sva_experiment_protocol.md)。数学定义、
数据规则、排名、干预和汇总**全部沿用**这两份文档；本文只记录为了在 24 GB A10G 上跑 30B/47B 模型而改变的**执行方式**，
以及实际验收证据和必须报告的偏离。

## 1. 为什么不能直接用 FModel

FModel 把整个模型放在 GPU 上并显式计算全部专家；Qwen3-30B-A3B（61 GB bf16）和 Mixtral-8x7B（93 GB）都放不下。
guide §9 也写明 Qwen/Mixtral 不在原入口支持范围内。新实现：

| 文件 | 作用 |
|---|---|
| `moetrace/relp_moe_stream.py` | `StreamRelP`：逐层流式加载权重；前向逐算子复现 transformers 5.22.2 的 eager 模块；RelP 反向 = 反向逐层流式 |
| `scripts/relp_sva_moe_run.py` | 与 `relp_sva_run.py` 相同的 `verify / score / evaluate` 命令、门禁、输出文件；另有 `evaluate-multi`、`--reuse-scores` |
| `scripts/relp_moe_chain.sh` | 正式实验链（smoke → paper → grouped，两模型；全部经 `gpu_queue.sh`） |
| `scripts/relp_moe_crosscheck_olmoe.py` | 在真实 OLMoE 上与 FModel（guide 参考实现）逐节点比较 |
| `scripts/relp_moe_weightcheck.py` | 真实 checkpoint 的权重加载与 HF `from_pretrained` 逐张量比较 |
| `tests/test_relp_moe_stream.py` | 22 项 GPU 测试（三种 MoE 架构的随机小模型） |
| `scripts/relp_sva_prepare.py --model qwen3|mixtral` | 按各自 tokenizer 重建清单；OLMoE 输出逐字节不变 |

## 2. 实现要点（数学不变）

- **q = (SiLU(g)·u)·w**，与 `relp_sva.native_capture` 的定义相同（Mixtral 的路由权重是 fp32，乘积显式转回激活精度）。
- **前向与原生逐位一致**：注意力、RoPE、RMSNorm、路由（Qwen3/OLMoE 把权重转回 bf16，Mixtral 保持 fp32）、eager 专家循环
  （输出 = (W_down·act)·w，bf16 `index_add_`，专家编号升序累加）。原生参照固定为 `attn_implementation="eager"`、
  `experts_implementation="eager"`（transformers 5.22.2 在 GPU 上默认 `grouped_mm`，本项目不用它作参照）。
- **RelP 反向**：前向保存每层入口残差；反向流式重新载入第 ℓ 层，用保存的残差按 RelP 规则重算该层（前向数值完全相同），
  一次 autograd 把 λ_{ℓ+1} 传到 λ_ℓ（含 attention V 路径的跨位置项）。MoE 输出直接加入残差，所以其系数就是 λ_{ℓ+1}，
  对任意专家 β_e = W_down,eᵀ λ_{ℓ+1}（guide §2.3/§9.2）。β 在 A 或 B 路由过的专家并集上计算；其余节点两端 q=0，分数为 0。
  RelP 规则与 FModel 相同：RMSNorm 分母（含 Qwen3 逐头 q/k norm）、attention pattern、SiLU 的 sigmoid、路由选择与权重均 detach，
  SwiGLU 乘法用 half rule。
- **干预**：完全被来源 run 替换的 (位置, 专家) 直接取来源 run 的贡献向量（因此整位置 q-patch 与输出 patch 逐位相同）；
  完全置零的贡献为 0；部分替换的用 W_down·q_final；下游路由自然重算。

## 3. 验收证据

| 检查 | 结果 |
|---|---|
| 随机小模型（Qwen3-MoE / Mixtral / OLMoE，bf16 与 fp32） | 流式前向、原生 harness 与完整 HF 模型逐位相同（每层输出、top-k、权重、指标） |
| 同上，RelP 分数 | paired/unpaired 分数与 support 等于显式全专家整图 autograd 参照（FModel 数学） |
| 同上，干预 | zero / mean_q_routed / source、部分与完整选择，与稠密 `where(S,V,q)` 参照一致；整位置 q-patch = 输出 patch = 原生 hook（逐位） |
| 真实 OLMoE，fp32，对 FModel（guide 参考实现） | support 完全相同（1,187,840 节点），分数相对 L2 9e-7，top-1000 神经元与 top-50 expert-position 完全相同；192 条干预 margin 差 ≤ 7e-6，路由变化计数 192/192 相同 |
| 真实权重加载（Qwen3、Mixtral） | layer 0 全部参数及 embed/norm/lm_head 与 HF `from_pretrained`（其 WeightConverter）逐位相同 |
| 真实 checkpoint `verify`（协议默认阈值 atol 1e-4、rtol 1e-3、相对 L2 1e-3） | Qwen3 2148/2148、Mixtral 1444/1444 通过（开发运行，`results/relp_moe_dev/`；正式运行会逐任务重新验收） |
| 权重分片完整性 | 41 个 safetensors 分片的 SHA-256 与 HF 固定 revision 的 LFS 哈希一致 |

## 4. 必须报告的偏离与事实

1. **bf16 批组成效应（原生 HF 自身的性质）**：同一输入单独跑和放进 batch 跑，GEMM 内核不同，bf16 舍入不同。开发验收中原生
   HF 的 margin 差最大 0.26（Qwen3，|margin| 均值 3.3）/ 0.21（Mixtral，4.5），top-k 集合变化 399/1920（Qwen3 第 8 名专家常近乎并列）
   / 13（Mixtral）。实现对原生在**相同批组成**下逐位一致；评分和干预按批处理行（行之间独立）。该效应记录在每个 `verify` 的
   `diagnostics` 中，不进入门禁。
2. **Mixtral 的 BOS**：按协议第 4 节规则 1（`add_special_tokens=True`，不人为追加）保留 tokenizer 默认的 `<s>`；它是特殊 token，
   `non_special` 池自动排除该位置。（与早期 Lu et al. 复现用 no-BOS 不同，两者是不同协议。）
3. **Mixtral paper/simple**：Mixtral tokenizer 下只剩 316 个对齐的 train 行，不够 300+40+40。用显式开关 `--short-discovery`：
   calibration/validation/test 仍各 40，discovery 取余下的 236 行，不复制记录；记录在 manifest。其他任务均为 300。
4. **原生 q 捕获**：`relp_sva.native_capture` 的写法在 Mixtral（fp32 路由权重）上会报 dtype 错，这里加了显式转换，定义不变。
5. **routing_weight_l1**（描述性诊断）：向量化后在 fp32 权重（Mixtral）上与逐行实现只差求和顺序（相对 2e-8）；top-k 变化计数完全一致。
6. **效率改动不改变结果**：`evaluate-multi` 在一次流式前向中完成同一 split 的多个 (unit, baseline) 评估，每个仍写独立目录；
   `score_mean` 用 `--reuse-scores` 复制 zero 基线的 discovery 分数（分数与基线无关，逐位相同），只新算 calibration 均值。

## 5. 正式运行计划

`bash scripts/relp_moe_chain.sh`：smoke（simple、nounpp；预算 neuron 1/5、expert 1/2、seed 101；无 test）→ paper（四任务）→
grouped（四任务），每个模型、每个任务：verify → score_zero → score_mean → validation（neuron / expert_position /
expert_all_positions × zero / mean_q_routed）；该 profile 全部任务的 validation 完成后再跑 test。预算、基线、位置池在 score 时锁定
（协议默认），test 不做任何选择。输出 `results/relp_sva_moe/<model>/<profile>/<task>/`。

实测单步耗时（GPU 墙钟，NVMe 约 1.4 GB/s 为下限）：verify 100 s / 185 s，score（300 对）77 s / 151 s，score_mean 54 s / 85 s，
evaluate-multi（40 行 × 6 个评估）129 s / 145 s（Qwen3 / Mixtral）。
