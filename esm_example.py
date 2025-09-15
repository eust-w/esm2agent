import torch
import esm

# 加载ESMFold模型，该模型集成了ESM-2用于端到端结构预测
model = esm.pretrained.esmfold_v1()
model = model.eval()

# 检查是否有可用的GPU
if torch.cuda.is_available():
    model = model.cuda()
    print("Moving model to GPU for better performance.")
else:
    print("No GPU found, using CPU. This might be slow.")

# 准备蛋白质序列数据（示例序列）
sequence = "MIRDEGMLQQLLSTIRDFVKNELIPREHEVAEKDCIPEDIIQQMRELGLFGLTIPEEYGGLGITMEEEVNVAFELGQTSPAFRSLIGTNNGIGSSGLIIDGTEEQKQKYLPRYASGEIIGSFCLTEPEAGSDAASLKTTAVKDGDFYILNGTKRFITNAPHAATFTVMARTNPAIKGAGGISAFLVEANTPGITLGKIDQKMGQKGSHTCDVIFENCRVPASALIGGVEGVGFKTAMKVLDKGRLHIGAYSVGVAERMLNDALHYAVERKQFGQPIANFQLIQAMLADSKAEIYAAKCMVLDAARRRDEGQNISTEASCAKMFATEMCGRVADRCVQIHGGAGYISEYSIERFYRDVRLFRLYEGTTQVQQIIIAKNMIKEVTS"

# 使用ESMFold进行结构预测
with torch.no_grad():
    output = model.infer_pdb(sequence)

# 将预测的结构保存为PDB文件
with open("result.pdb", "w") as f:
    f.write(output)

print("Structure prediction complete. The result is saved as result.pdb")
