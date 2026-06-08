import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
# 1. 读取数据集
try:
    df = pd.read_csv('./data/smi_pmf.csv')
    print(f"成功读取数据，样本数: {len(df)}")
except FileNotFoundError:
    print("错误: 请检查 smi_pmf.csv 路径")

# 2. 计算 Morgan 分子指纹
def smiles_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 返回 numpy 数组格式的位向量
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)

print("正在计算分子指纹...")
fps = np.array([smiles_to_fp(s) for s in df['SMILES']])

# 3. 利用 t-SNE 进行降维
# 针对小样本 (N=75)，建议设置较低的 perplexity
print("正在执行 t-SNE 降维...")
# 先用 PCA 预降维到 30 维以减少噪声(可选,但推荐)
pca = PCA(n_components=min(30, len(df)), random_state=42)
fps_pca = pca.fit_transform(fps)

tsne = TSNE(
    n_components=2, 
    perplexity=30,      # 小样本关键参数:建议 N/5 左右
    early_exaggeration=40,
    random_state=42,    # 固定随机种子以便复现
    init='pca',         # 使用 PCA 初始化使结果更稳健
    learning_rate='auto',
    n_iter=1000,
    verbose=1           # 显示进度信息
)
tsne_results = tsne.fit_transform(fps_pca)

# 4. 合并数据并保存
df['t-SNE Dimension 1'] = tsne_results[:, 0]
df['t-SNE Dimension 2'] = tsne_results[:, 1]

df.to_csv('./data/chem_space.csv', index=False)
print("处理完成，结果已保存至 chem_space.csv")

# 绘图-
# --- 1. 环境设置 (字体与样式) ---
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # 优先使用Arial
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 18

# 模拟或读取数据 (假设 df 已经包含了 t-SNE 结果)
# df = pd.read_csv('./data/chem_space.csv') 

# --- 2. 绘图准备 ---
plt.figure(figsize=(10, 8))
sns.set_style("ticks")

pp_raw = df['PMF']
# 归一化尺寸范围 [50, 450]，PMF 值越大，点越大
pp_sizes = ((pp_raw - pp_raw.min()) / (pp_raw.max() - pp_raw.min())) * 350 + 100

# --- 3. 核心散点图绘制 ---
scatter = plt.scatter(
    df['t-SNE Dimension 1'], 
    df['t-SNE Dimension 2'], 
    s=pp_sizes,
    c='black',
    edgecolor='w', 
    linewidth=0.8,
    alpha=0.8
)

# --- 4. 横向尺寸图例设置 ---
# 生成5个代表性数值
legend_values = np.linspace(pp_raw.min(), pp_raw.max(), 5)

# 创建虚假的散点用于生成图例
legend_elements = []
for val in legend_values:
    # 重新计算对应的显示尺寸
    s = ((val - pp_raw.min()) / (pp_raw.max() - pp_raw.min())) * 400 + 50
    legend_elements.append(plt.scatter([], [], c='gray', alpha=0.5, s=s, edgecolor='k'))

# 将图例放置在顶部中心 (原本标题的位置)
# ncol=5 实现横向排列,frameon=False 去掉边框
plt.legend(
    legend_elements, 
    [f"{v:.0f}" for v in legend_values],
    title="Adsorption energy [kcal/mol]",
    loc='lower center', 
    bbox_to_anchor=(0.5, 1.01), # 坐标(0.5, 1.02) 表示在图上方中心
    ncol=5, 
    frameon=False,
    columnspacing=1.2,
    handletextpad=0.5,
    title_fontsize=18
)

# --- 5. 细节修饰 ---
plt.xlabel('t-SNE Dimension 1', fontsize=18, fontweight='normal', family='DejaVu Sans')
plt.ylabel('t-SNE Dimension 2', fontsize=18, fontweight='normal', family='DejaVu Sans')
plt.grid(True, linestyle=':', alpha=0.4)

# 调整布局以适应顶部的图例
plt.tight_layout()

plt.savefig('./figure/chemical_space.png', dpi=800, bbox_inches='tight')