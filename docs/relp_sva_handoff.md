# RelP → MoE SVA agent 交接

本地准备完成于 2026-11-07；尚未运行完整 checkpoint 或 GPU 实验。提交状态以仓库 Git 记录为准。

先读 [SVA 实验协议](relp_sva_experiment_protocol.md)，再查 [数学实现 guide](relp_moe_implementation_guide.md)。数据版本、筛选、token 对齐、样本用途、干预和逐步命令都在协议中。

## 执行顺序

1. 使用现有 `KenSu223/MOE` 仓库（本地基准 commit `b538f5978a0b0314cc9e86911c0dd071d3e3853b`）。交接 ZIP 是该仓库的增量文件包，不是独立项目；将 `payload/` 中的相对路径放入仓库前，检查目标文件差异并保留其他工作。
2. 建隔离 Python 环境，参考 `requirements-relp-sva.txt`。CUDA torch wheel 按服务器环境安装，实际环境由 runner 记录。可先运行 `python -m pytest tests/test_relp_sva.py -q`。
3. `python scripts/relp_sva_prepare.py fetch` 下载并校验官方数据与 tokenizer。ZIP 不包含可再生的 raw/tokenizer 缓存，也不包含权重。
4. 准备固定 revision 的完整 OLMoE checkpoint；按协议使用已有 GPU queue，先对 `olmoe_smoke_v1` 的 simple/nounpp 做 `verify → score → evaluate`。
5. 原生一致性验收不通过时，先定位实现/精度差异。不得把 CPU 小模型记录当作完整 checkpoint 的通过证书，不得绕过 verification gate。
6. smoke 通过后，四个子任务分别运行正式 paper profile；独立内容泛化采用 grouped profile。先 validation 锁定方案，再运行 test。评分保存完整有符号均值，expert 排名从全部 neuron 汇总，不能从 neuron top-K 回收。

## 已交付与边界

- 三套 tokenizer 对齐后的实际清单、源版本和哈希、逐行来源、重复与重叠审计。
- 数据准备、评分与验收 CLI；neuron、expert-position、跨位置 expert 排名与干预；匹配随机对照；逐例 JSONL/Parquet 输出。
- [验证记录](../data/relp_sva/validation_report.json)：11 项 CPU tiny-HF-OLMoE 测试通过；三套 JSONL 重建一致。
- 没有完整 7B 的验收、正式排名或干预效果。不要把当前交付解释为已得到论文级实验结论。
- paper profile 保持官方文件划分与 300/40 discovery/test 规模，但不是作者精确 row IDs；其内容有重叠。grouped 是项目新增协议，simple 数量少于 300，不能复制记录补数。

ZIP 内 `PACKAGE_SHA256.json` 校验交付文件；可选数学例子的部分链接指向本工作区的 `papers/`，不包含在增量包中，不是 SVA runner 的运行依赖。
