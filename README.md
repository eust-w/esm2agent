# ESM2Agent - 蛋白质突变预测与设计工具

ESM2Agent是一个基于ESM-2语言模型的蛋白质突变预测和设计工具集，用于辅助蛋白质工程中的突变设计。该工具集包括单点突变预测、组合突变设计以及整合蛋白质结构信息的高级组合突变设计功能。

## 功能特点

- **单点突变预测**：利用ESM-2语言模型预测单点突变的概率
- **组合突变设计**：基于单点突变预测结果，设计可能产生协同效应的突变组合
- **高级组合突变设计**：整合蛋白质结构信息和功能域信息，设计更精确的突变组合
- **可视化**：生成突变热图和组合突变评分可视化
- **结构可视化**：在蛋白质三维结构上展示突变位点

## 环境配置

### 依赖项

项目依赖以下Python库：

```
numpy
pandas
matplotlib
seaborn
biopandas
networkx
scipy
py3Dmol
```

### 安装

1. 克隆仓库：

```bash
git clone https://github.com/eust-w/esm2agent.git
cd esm2agent
```

2. 创建并激活conda环境：

```bash
conda env create -f environment.yml
conda activate esm_env
```

或者使用简化版环境：

```bash
conda env create -f simple_environment.yml
conda activate esm_env
```

## 使用指南

### 1. 单点突变预测

`mutation_prediction.py`脚本用于预测蛋白质序列中的单点突变概率。

#### 基本用法

```bash
python run_fhase_mutation.py --sequence_file fhase_sequence.txt --output_csv fhase_mutations_full.csv
```

#### 参数说明

- `--sequence_file`：包含蛋白质序列的文本文件
- `--output_csv`：输出的CSV文件路径
- `--batch_size`：批处理大小（默认：8）
- `--top_k`：每个位点保留的突变数量（默认：5）

#### 输出

脚本将生成一个CSV文件，包含以下列：
- `position`：突变位点
- `original`：原始氨基酸
- `mutation`：突变（格式：原始氨基酸+位置+突变氨基酸）
- `mutated_to`：突变后的氨基酸
- `probability`：突变概率
- `log_probability`：突变概率的对数值

### 2. 基础组合突变设计

`combo_mutation_design.py`脚本用于设计组合突变，基于单点突变预测结果。

#### 基本用法

```bash
python combo_mutation_design.py --mutations_csv fhase_mutations_full.csv --output_prefix fhase_combo_mutations --combo_size 3 --diversity
```

#### 参数说明

- `--mutations_csv`：单点突变预测结果文件
- `--output_prefix`：输出文件前缀
- `--top_positions`：考虑的顶部位点数量（默认：30）
- `--mutations_per_position`：每个位点考虑的突变数量（默认：2）
- `--combo_size`：突变组合的大小（默认：3）
- `--synergy_model`：协同效应计算模型（选项：multiplicative, additive, geometric，默认：multiplicative）
- `--diversity`：是否生成多样化的组合突变
- `--diversity_weight`：多样性权重（默认：0.5）

#### 输出

脚本将生成以下文件：
- `{output_prefix}.csv`：所有组合突变的评分结果
- `{output_prefix}.png`：顶部组合突变的可视化
- `{output_prefix}_diverse.csv`：多样化的组合突变结果（如果使用--diversity选项）
- `{output_prefix}_diverse.png`：多样化组合突变的可视化（如果使用--diversity选项）

### 3. 高级组合突变设计

`advanced_combo_mutation_design.py`脚本整合了蛋白质结构信息和功能域信息，设计更精确的组合突变。

#### 基本用法

```bash
python advanced_combo_mutation_design.py --mutations_csv fhase_mutations_full.csv --pdb_file protein_structure.pdb --domains_file protein_domains.csv --output_prefix fhase_advanced_mutations --combo_size 4 --diversity --visualize_structure
```

#### 参数说明

- `--mutations_csv`：单点突变预测结果文件（必需）
- `--pdb_file`：蛋白质结构PDB文件（可选）
- `--domains_file`：功能域信息文件（可选）
- `--output_prefix`：输出文件前缀（默认：advanced_combo_mutations）
- `--top_positions`：考虑的顶部位点数量（默认：30）
- `--mutations_per_position`：每个位点考虑的突变数量（默认：2）
- `--combo_size`：突变组合的大小（默认：3）
- `--max_combinations`：生成的最大组合数量（默认：100000）
- `--synergy_model`：协同效应计算模型（选项：multiplicative, additive, geometric，默认：multiplicative）
- `--structure_weight`：结构信息在评分中的权重（默认：0.3）
- `--domain_weight`：功能域信息在评分中的权重（默认：0.2）
- `--domain_focus`：专注于特定功能域（可选）
- `--diversity`：生成多样化的组合突变
- `--diversity_weight`：多样性权重（默认：0.5）
- `--visualize_structure`：在结构上可视化突变

#### 功能域文件格式

功能域信息文件应为CSV格式，包含以下列：
```
domain_name,start_position,end_position
catalytic_domain,100,200
binding_domain,201,300
...
```

#### 输出

脚本将生成以下文件：
- `{output_prefix}.csv`：所有组合突变的评分结果
- `{output_prefix}.png`：顶部组合突变的可视化
- `{output_prefix}_diverse.csv`：多样化的组合突变结果（如果使用--diversity选项）
- `{output_prefix}_diverse.png`：多样化组合突变的可视化（如果使用--diversity选项）
- `{output_prefix}_structure.html`：在结构上可视化突变的交互式HTML文件（如果使用--visualize_structure选项）

### 4. 可视化突变热图

`generate_heatmap.py`脚本用于生成突变概率的热图可视化。

#### 基本用法

```bash
python generate_heatmap.py --mutations_csv fhase_mutations_full.csv --output_prefix fhase_mutations_heatmap
```

#### 参数说明

- `--mutations_csv`：单点突变预测结果文件
- `--output_prefix`：输出文件前缀
- `--top_k`：每个位点显示的顶部突变数量（默认：5）
- `--positions_per_plot`：每个热图显示的位点数量（默认：25）

## 示例

### 单点突变预测

```bash
python run_fhase_mutation.py --sequence_file fhase_sequence.txt --output_csv fhase_mutations_full.csv
```

### 生成突变热图

```bash
python generate_heatmap.py --mutations_csv fhase_mutations_full.csv --output_prefix fhase_mutations_heatmap
```

### 基础组合突变设计

```bash
python combo_mutation_design.py --mutations_csv fhase_mutations_full.csv --output_prefix fhase_combo_mutations --top_positions 30 --mutations_per_position 2 --combo_size 3 --diversity
```

### 高级组合突变设计

```bash
python advanced_combo_mutation_design.py --mutations_csv fhase_mutations_full.csv --output_prefix fhase_advanced_mutations --combo_size 4 --top_positions 20 --mutations_per_position 2 --diversity
```

### 带结构信息的高级组合突变设计

```bash
python advanced_combo_mutation_design.py --mutations_csv fhase_mutations_full.csv --pdb_file protein_structure.pdb --output_prefix fhase_advanced_mutations_structure --combo_size 3 --structure_weight 0.5 --visualize_structure
```

## 注意事项

1. 对于大型蛋白质序列，单点突变预测可能需要较长时间
2. 组合突变数量随着`combo_size`、`top_positions`和`mutations_per_position`的增加而指数增长，请根据计算资源适当设置参数
3. 使用结构信息时，确保PDB文件中的残基编号与序列中的位置一致

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
