import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import mordred
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

def compute_mordred_descriptors_2D(mols):
    """计算 2D Mordred 描述符，避免3D嵌入问题"""
    calc = Calculator(descriptors, ignore_3D=True)  # 只计算2D描述符
    df = calc.pandas(mols)
    # 将 Mordred 错误值替换为 NaN
    df = df.applymap(lambda x: np.nan if isinstance(x, (mordred.error.Missing, mordred.error.Error)) else x)
    return df

def clean_descriptors(df, corr_threshold=0.99):
    """清洗描述符数据：去除NaN、零方差、高相关"""
    # 1. 去除全为NaN的列
    df = df.dropna(axis=1, how='all')
    
    # 2. 去除NaN比例超过50%的列
    nan_ratio = df.isna().sum() / len(df)
    df = df.loc[:, nan_ratio <= 0.01]
    
    # 3. 用中位数填充剩余NaN
    df = df.fillna(df.median())
    
    # 4. 去除零方差特征
    df = df.loc[:, df.std() != 0]
    
    # 5. 去除高度相关的特征
    if len(df.columns) > 1:
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        df = df.drop(columns=to_drop)
    
    return df

def save_configuration(scaler, pca, feature_names, output_dir='./data'):
    """保存预处理配置"""
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存特征名称
    feature_df = pd.DataFrame({'feature_name': feature_names})
    feature_df.to_csv(f'./pretrained_model/des/PCA/des_PCA_name.csv', index=False)
    
    # 保存配置
    config = {
        'scaler': scaler,
        'pca': pca,
        'feature_names': feature_names,
        'n_components': pca.n_components_,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }
    
    with open(f'./pretrained_model/des/PCA/des_PCA_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print(f"✅ 配置已保存:")
    print(f"   - 特征名称: ./pretrained_model/des/PCA/des_PCA_name.csv")
    print(f"   - 预处理配置:./pretrained_model/des/PCA/des_PCA_config.pkl")

def main():
    # === Step 1: 读取数据 ===
    input_file = './data/smi_pmf.csv'
    output_file = './data/des_PCA.csv'
    data = pd.read_csv(input_file)
    
    print(f"原始数据样本数: {len(data)}")

    # === Step 2: SMILES -> RDKit Mol ===
    print("\n转换为RDKit分子对象...")
    data['mol'] = data['SMILES'].map(lambda x: Chem.MolFromSmiles(x))
    
    # 统计无效SMILES
    invalid_count = data['mol'].isna().sum()
    if invalid_count > 0:
        print(f"⚠️  发现 {invalid_count} 个无效SMILES，将被移除")
    
    data = data.dropna(subset=['mol'])
    print(f"有效分子数: {len(data)}")

    # === Step 3: 计算 2D Mordred 描述符 ===
    print("\n计算2D Mordred描述符...")
    des_df = compute_mordred_descriptors_2D(data['mol'])
    print(f"原始描述符数量: {des_df.shape[1]}")

    # === Step 4: 清洗特征 ===
    print("\n清洗描述符数据...")
    des_clean = clean_descriptors(des_df, corr_threshold=0.90)
    print(f"清洗后描述符数量: {des_clean.shape[1]}")

    # === Step 5: 标准化描述符 ===
    print("\n标准化描述符...")
    scaler = StandardScaler()
    des_scaled = scaler.fit_transform(des_clean)

    # === Step 6: PCA 降维（保留95%方差） ===
    print("\n执行PCA降维...")
    pca = PCA(n_components=0.95, random_state=42)
    des_pca = pca.fit_transform(des_scaled)
    des_pca_df = pd.DataFrame(des_pca, columns=[f'PC{i+1}' for i in range(des_pca.shape[1])])
    
    print(f"PCA保留的方差比例: {np.sum(pca.explained_variance_ratio_):.3f}")
    print(f"主成分数量: {des_pca.shape[1]}")

    # === Step 7: 保存配置 ===
    print("\n保存预处理配置...")
    save_configuration(scaler, pca, des_clean.columns.tolist())

    # === Step 8: 保留两位小数 ===
    des_pca_df = des_pca_df.round(2)

    # === Step 9: 合并结果 ===
    result = pd.concat([data[['SMILES', 'PMF']].reset_index(drop=True),
                        des_pca_df.reset_index(drop=True)], axis=1)

    # === Step 10: 保存 ===
    result.to_csv(output_file, index=False)
    
    print(f"\n✅ PCA降维后的数据已保存: {output_file}")
    print(f"最终数据维度: {result.shape}")
    print(f"主成分数量: {des_pca_df.shape[1]}")

if __name__ == '__main__':
    main()