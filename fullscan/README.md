# fullscan — 全序列单点扫描与可审计报告

对整条序列做 masked-marginal 扫描，输出排序后的突变候选、完整 LLR 矩阵，以及一份把**运行参数、权重校验和、健全性检查**全部写在正文里的 PDF 报告。

与仓库根目录的 `mutation_prediction.py` 是同一套打分方法（逐位点 `<mask>`、全长上下文、对 20 种氨基酸取 softmax）。这里补的是三件根目录脚本没有的事：

| | 根目录 `mutation_prediction.py` | `fullscan/scan.py` |
| :--- | :--- | :--- |
| 设备 | MPS 或 CPU | **CUDA** / MPS / CPU |
| 排序依据 | `probability`，即 p(mut) | 同时给出 **LLR = log p(mut) − log p(wt)** 两种排序 |
| 候选字母表 | 词表里所有单字符 token（含 `X B U Z O . -`） | 严格限定 20 种标准氨基酸 |
| 产物 | CSV + 热图 | CSV + 20×L 完整 LLR 矩阵 + 热图 + PDF 报告 |
| 自检 | 无 | `verify.py`：token 对齐、复制任务、野生型回收率 |

<a id="why-llr"></a>
## 为什么默认看 LLR，而不是 p(mut)

`p(mut)` 回答的是「模型最笃定这里该换成别的残基的位置在哪」。它完全不看野生型残基原本有多可能，因此排名会被「模型认为本来就标错了」的位点占据。

`LLR = log p(mut) − log p(wt)` 回答的是「相对**当前这个**残基，模型有多偏好替换」。这是 ESM 系列论文和 ProteinGym 中与实测适应度相关的量。

两份表都会输出，默认建议用 LLR。

> [!IMPORTANT]
> LLR 高只说明该替换在 UniRef 训练分布下「看起来自然」，**不等于**活性、稳定性或表达量会提升。催化与结合残基通常高度保守，模型会正确地拒绝替换它们；但它同样会给出大量合理却无用的替换。输出是待测清单，不是预测。

## 用法

```bash
cd fullscan
python -m pip install torch transformers pandas numpy matplotlib pillow reportlab
```

权重默认从 HuggingFace 读取本地目录。若直连慢，`fetch_weights.sh` 用 16 路分块并行下载并校验字节数（单连接 ~54 kB/s 的网络下可达 ~2.5 MB/s）：

```bash
bash fetch_weights.sh
```

把序列放进 `query.fasta`（标准 FASTA，取第一条），然后：

```bash
python scan.py --fasta query.fasta --batch-size 8
```

```bash
python report.py
```

跑完建议顺手自检一次——它会独立重算，不读 `scan.py` 的输出：

```bash
python verify.py
```

### 主要参数

| 参数 | 默认 | 说明 |
| :--- | :--- | :--- |
| `--fasta` | `query.fasta` | 输入序列 |
| `--model-dir` | `hf/esm2_t33_650M_UR50D` | 本地权重目录 |
| `--top-k` | `5` | 每个位点保留的非野生型候选数 |
| `--top-n` | `100` | 最终排名长度 |
| `--batch-size` | `8` | 并行遮蔽的位点数；显存/内存不足时调小 |
| `--threads` | `0` | CPU 线程数，0 为默认 |

### 产物

| 文件 | 内容 |
| :--- | :--- |
| `results/esm2_top100_by_llr.csv` | 前 100，按 LLR 排序 |
| `results/esm2_top100_by_probability.csv` | 前 100，按 p(mut) 排序 |
| `results/esm2_all_mutations.csv` | 全部 L×top_k 候选 |
| `results/esm2_llr_matrix.csv` | 20×L 完整 LLR 矩阵 |
| `results/esm2_heatmap.png` | 全位点替换景观 |
| `results/run_meta.json` | 运行参数、权重 SHA-256、自检结果、最保守位点 |
| `results/ESM2_top100_report.pdf` | 以上内容的报告版 |

## 自检做了什么

`verify.py` 检查三件会让排名整体失真、但不会报错的事：

1. **token 偏移**——分词后第 `p` 个位置必须正好是 `sequence[p-1]`。差一位不会报错，但整张表会系统性错位。
2. **未遮蔽复制任务**——不加遮蔽时模型应能复现输入序列（预期 > 0.95）。偏低说明权重没加载对。
3. **遮蔽野生型回收率**——逐位点遮蔽后 top-1 命中野生型的比例。天然蛋白上 ESM-2 650M 通常在 0.45–0.65。偏低说明模型、分词或权重有问题。

`check_nonstandard.py` 针对候选字母表这一项单独验证，见下。

## 候选字母表的问题

根目录脚本这样构造候选集：

```python
index_to_aa = {i: a for i, a in enumerate(self.alphabet.all_toks)
               if a not in ('<cls>', '<pad>', '<eos>', '<unk>', '<mask>')}
...
if len(aa) == 1:
    aa_probs[aa] = probs[token_idx]
```

`len(aa) == 1` 能滤掉多字符的特殊 token，但 ESM-2 词表里 `X`、`B`、`U`、`Z`、`O`、`.`、`-` **都是单字符**，会被当作合法替换进入 top-k，产出「突变成 X」或「突变成 -」这类没有意义的条目。

`check_nonstandard.py` 用原脚本的字母表重跑一遍，统计这种情况实际发生的频率。`scan.py` 直接按 20 种标准氨基酸取索引，不依赖长度过滤。

## 示例

[`example/`](example/) 是一条 318 aa 序列（`WP_018752751.1`）的完整产物，包含两份前 100 和 PDF 报告。该序列的具体解读见 [`example/README.md`](example/README.md)。

## 边界

- 扫描是**单点**的，不考虑上位效应；多点组合请接 `combo_mutation_design.py`。
- masked-marginal 每个位点一次前向，共 L 次。318 aa 在 6 核 CPU 上约 10 分钟，GPU 上约 1–2 分钟。
- 报告中的一切数字都由 `scan.py` 写入 `run_meta.json`，`report.py` 只做排版、不重算，因此 PDF 不可能与 CSV 不一致。
