import torch
import esm

# 加载ESM-2模型（这里使用650M参数版本）
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # 禁用dropout，确保结果确定性

# 准备蛋白质序列数据（示例序列）
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# 提取每残基表示（无梯度计算）
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# 生成每序列平均表示（可选，用于下游任务）
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

print("每残基表示形状:", token_representations.shape)  # 示例输出: (batch_size, seq_len, hidden_dim)
print("序列平均表示:", sequence_representations[0][:10])  # 前10维示例
