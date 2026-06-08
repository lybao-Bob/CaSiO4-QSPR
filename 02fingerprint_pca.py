import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import joblib  # 新增：用于保存模型对象
import os

warnings.filterwarnings("ignore")

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    """将SMILES转换为Morgan指纹向量"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    # === Step 1: 读取数据与配置路径 ===
    input_file = './data/smi_pmf.csv'
    output_file = './data/fingerprint_PCA.csv'
    # 确保保存路径与你后续训练脚本中的目录结构一致
    model_save_dir = './pretrained_model/fingerprint/PCA'
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"🔹 Reading dataset: {input_file}")
    data = pd.read_csv(input_file)

    # === Step 2: SMILES -> Morgan指纹 ===
    print("🔹 Computing Morgan fingerprints (ECFP, radius=2, nBits=2048)...")
    fps = []
    for s in data['SMILES']:
        fp = smiles_to_morgan(s)
        fps.append(fp)

    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    fps = [fps[i] for i in valid_idx]
    data = data.iloc[valid_idx].reset_index(drop=True)

    fps_array = np.array(fps)
    print(f"✅ Successfully generated {fps_array.shape[0]} fingerprints")

    # === Step 3: 标准化指纹特征并保存 Scaler ===
    print("🔹 Standardizing fingerprint features...")
    scaler = StandardScaler()
    fps_scaled = scaler.fit_transform(fps_array)
    # 保存特征标准化器，用于预测时对 Morgan 指纹进行同样的缩放
    joblib.dump(scaler, os.path.join(model_save_dir, 'fingerprint_scaler.pkl'))

    # === Step 4: PCA降维并保存 PCA 对象 ===
    print("🔹 Performing PCA dimensionality reduction (90% variance retained)...")
    pca = PCA(n_components=0.90, random_state=42)
    fps_pca = pca.fit_transform(fps_scaled)
    # 保存 PCA 对象，用于预测时将高维指纹投影到相同的主成分空间
    joblib.dump(pca, os.path.join(model_save_dir, 'pca_processor.pkl'))
    print(f"✅ PCA object and Scaler saved to: {model_save_dir}")

    # === Step 5: 保留两位小数并生成 DataFrame ===
    fps_pca_df = pd.DataFrame(fps_pca, columns=[f'PC{i+1}' for i in range(fps_pca.shape[1])])
    fps_pca_df = fps_pca_df.round(2)

    # === Step 6: 合并结果并保存 CSV ===
    result = pd.concat(
        [data[['SMILES', 'PMF']].reset_index(drop=True),
         fps_pca_df.reset_index(drop=True)],
        axis=1
    )
    result.to_csv(output_file, index=False)

    print(f"✅ Done! PCA features saved to: {output_file}")
    print(f"🧩 PCA components retained: {fps_pca_df.shape[1]}")

if __name__ == '__main__':
    main()