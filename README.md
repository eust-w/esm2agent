<div align="center">
  <img src="docs/readme/hero.svg" alt="ESM2Agent — workflow overview" width="100%">
  <h1>ESM2Agent</h1>
  <p><strong>从蛋白质序列评分，到可检查的突变候选。</strong></p>
  <p><img alt="Model: ESM-2" src="https://img.shields.io/badge/Model-ESM--2-329469?style=flat-square">
  <img alt="Runtime: PyTorch" src="https://img.shields.io/badge/Runtime-PyTorch-d76b42?style=flat-square">
  <img alt="Code license: Apache-2.0" src="https://img.shields.io/badge/Code%20license-Apache--2.0-d99536?style=flat-square"></p>
  <p>
    <a href="#overview">项目亮点</a> ·
    <a href="#quick-start">快速开始</a> ·
    <a href="#workflow">工作流</a> ·
    <a href="#validation">验证与边界</a> ·
    <a href="#documentation">文档</a>
  </p>
  <p><a href="README.en.md">English</a> · <a href="README.md">简体中文</a></p>
</div>

---

<a id="overview"></a>
## 把序列模型接到候选探索流程

**ESM2Agent** 是围绕 ESM-2 的蛋白质突变探索工具集：对单个位点评分，组合候选突变，并可引入结构与功能域信息辅助排序，输出 CSV、热图和交互式结构视图。

| 位点评分 | 组合设计 | 结果可视化 |
| :--- | :--- | :--- |
| 指定位点、Top-k 替换与模型概率 | 乘法 / 加法 / 几何评分及多样性选择 | 突变热图、候选排名、PDB 位点展示 |

> [!NOTE]
> 模型概率与组合分数用于筛选候选。它们不等于实验验证的活性、稳定性或协同效应，也不构成可直接用于实验的性能承诺。

<a id="quick-start"></a>
## 快速开始

推荐从仓库的轻量 Conda 环境开始，再补齐脚本直接使用的 Python 依赖。

```bash
git clone https://github.com/eust-w/esm2agent.git
cd esm2agent
conda env create -f simple_environment.yml
conda activate esm_env
python -m pip install fair-esm numpy pandas matplotlib seaborn \
  biopandas networkx scipy py3Dmol

python mutation_prediction.py --help
```

首次预测会下载 ESM-2 权重。当前预测器默认加载 `esm2_t33_650M_UR50D`，自动选择 **Apple MPS 或 CPU**，尚未自动选择 CUDA。

```bash
python mutation_prediction.py \
  --sequence "$(cat /path/to/your_sequence.txt)" \
  --positions 10,20,30 --top_k 5 \
  --output mutations.csv --visualize --vis_output mutation_heatmap.png
```

输入是单字母氨基酸序列文本，不含 FASTA 标题；位点为 **1-based**，需在序列长度以内。请使用有权处理的输入序列。

<a id="workflow"></a>
## 一条可以逐步检查的分析链

```text
蛋白质序列 → ESM-2 位点评分 → mutations.csv
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
               候选组合与多样性             热图与排名
                    │
             可选 PDB / 功能域
                    ▼
             结构辅助评分与 HTML 视图
```

### 1. 单点结果 → 组合候选

```bash
python combo_mutation_design.py \
  --mutations_csv mutations.csv --output_prefix candidates \
  --top_positions 20 --mutations_per_position 2 --combo_size 3 --diversity
```

### 2. 可选结构信息

```bash
python advanced_combo_mutation_design.py \
  --mutations_csv mutations.csv \
  --pdb_file /path/to/structure.pdb \
  --domains_file /path/to/domains.csv \
  --output_prefix structured_candidates \
  --combo_size 3 --max_combinations 10000 --diversity --visualize_structure
```

功能域 CSV 使用 `domain_name,start_position,end_position` 三列。PDB 残基编号应与输入序列位置对应；结构与功能域参数可省略。

### 3. 单独生成热图

```bash
python generate_heatmap.py \
  --csv_file mutations.csv --output mutation_heatmap.png \
  --top_n 100 --max_per_chart 30
```

热图脚本默认分图保存，实际图片名可能带分片后缀。各脚本的可用选项以 `--help` 为准。

| 产物 | 用途 |
| :--- | :--- |
| 位点 CSV | `position`、`original`、`mutation`、`mutated_to`、`probability`、`log_probability` |
| 组合 CSV / PNG | 候选组合、排序分数与排名图 |
| Diverse CSV / PNG | 多样性选择后的候选及可视化 |
| Structure HTML | 有 PDB 输入时的突变位点视图 |

<a id="validation"></a>
## 验证与边界

```bash
python combo_mutation_design.py --help
python advanced_combo_mutation_design.py --help
python generate_heatmap.py --help
```

- 先对少量有效位点运行，检查 CSV 和输出图，再扩大范围。
- 组合数量随候选位点与组合大小增长；高级脚本可用 `--max_combinations` 限制。
- 结构权重、功能域权重和“协同”评分是启发式设计，需独立验证其科学适用性。
- 仓库没有在 README 中提供已验证的湿实验结果或准确率基准。

<details>
<summary><strong>现有示例脚本与环境说明</strong></summary>

`run_fhase_mutation.py` 固定读取 `fhase_sequence.txt` 并使用固定输出文件名，**没有命令行参数解析**。处理自定义输入应使用上面的 `mutation_prediction.py`。

`environment.yml` 是名为 `esmfold` 的较重环境，并含旧版 CUDA / OpenMM 固定依赖；`simple_environment.yml` 的环境名为 `esm_env`。两者不应混用环境名称。

</details>

<a id="documentation"></a>
## 代码导航

| 文件 | 作用 |
| :--- | :--- |
| [mutation_prediction.py](mutation_prediction.py) | ESM-2 预测与单点可视化 |
| [combo_mutation_design.py](combo_mutation_design.py) | 组合评分与多样性选择 |
| [advanced_combo_mutation_design.py](advanced_combo_mutation_design.py) | 结构 / 功能域辅助排序 |
| [generate_heatmap.py](generate_heatmap.py) | CSV 到热图 |
| [visualize_structure.py](visualize_structure.py) | 结构展示辅助脚本 |

## 许可证与来源

代码采用 [Apache-2.0](LICENSE)，以仓库 LICENSE 为准。ESM-2 及其他依赖遵循各自许可；`reference/` 中的文章和配图、输入序列及结构数据的权利需单独核对，不应推定由代码许可证覆盖。
