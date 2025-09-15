import os
from mutation_prediction import ESMMutationPredictor

# 读取FHase序列
with open('fhase_sequence.txt', 'r') as f:
    sequence = f.read().strip()

print(f"序列长度: {len(sequence)}个氨基酸")

# 初始化突变预测器
predictor = ESMMutationPredictor()

# 预测所有位点的突变
# 分析整个序列的所有氨基酸位点
positions = list(range(1, len(sequence) + 1))  # 预测所有位点
print(f"正在预测全部 {len(sequence)} 个位点的突变...")

results = predictor.predict_mutations(
    sequence=sequence,
    positions=positions,
    top_k=5  # 每个位点返回概率最高的5种突变
)

# 保存结果
results.to_csv('fhase_mutations_full.csv', index=False)
print(f"突变预测结果已保存至 fhase_mutations_full.csv")

# 生成可视化
print("生成突变预测可视化...")
predictor.visualize_mutations(
    df=results,
    output_file='fhase_mutations_full_heatmap.png',
    top_n_positions=100  # 展示前100个最有潜力的位点
)
print("可视化已保存至 fhase_mutations_full_heatmap.png")

# 输出前100个最有潜力的突变到文件中
print("\n将前100个最有潜力的突变输出到文件中...")
top_mutations = results.sort_values('probability', ascending=False).head(100)

# 将结果保存到CSV文件
top_mutations.to_csv('fhase_top100_mutations.csv', index=False)

# 同时也保存为更易读的文本格式
with open('fhase_top100_mutations.txt', 'w') as f:
    f.write("排名,突变,原始氨基酸,位置,突变氨基酸,概率\n")
    for i, (_, row) in enumerate(top_mutations.iterrows(), 1):
        f.write(f"{i},{row['mutation']},{row['original']},{row['position']},{row['mutated_to']},{row['probability']:.4f}\n")

print(f"前100个最有潜力的突变已保存至 fhase_top100_mutations.csv 和 fhase_top100_mutations.txt")

# 同时也打印到控制台
print("\n前10个最有潜力的突变:")
for i, (_, row) in enumerate(top_mutations.head(10).iterrows(), 1):
    print(f"{i}. {row['mutation']}: 概率 = {row['probability']:.4f}")
