import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors, error
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial.distance import pdist, squareform
import scipy.stats
from minepy import MINE
import warnings
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 忽略不必要的库警告
warnings.filterwarnings("ignore")

# ============================================================
# 全局绘图配置 
# ============================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
FONT_SIZE = 18  
FIG_SIZE = (8, 6) 
LINE_WIDTH = 1.5  
DOT_SIZE = 80   

def write_log(msg, logfile="descriptor_screening_log.txt"):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)

def distcorr(X, Y):
    """计算距离相关系数"""
    X = np.atleast_2d(X).T if np.ndim(X) == 1 else np.atleast_2d(X)
    Y = np.atleast_2d(Y).T if np.ndim(Y) == 1 else np.atleast_2d(Y)
    n = X.shape[0]
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy) + 1e-9)

def get_independent_subset(X, shap_values, feature_names, n_target=6, corr_threshold=0.8):
    """基于重要性排序并剔除冗余特征的工具函数"""
    mean_shap = np.abs(shap_values).mean(0)
    sorted_indices = np.argsort(mean_shap)[::-1]
    
    selected_indices = []
    for idx in sorted_indices:
        if len(selected_indices) >= n_target:
            break
        current_fea = feature_names[idx]
        is_redundant = False
        for s_idx in selected_indices:
            prev_fea = feature_names[s_idx]
            corr, _ = scipy.stats.pearsonr(X[current_fea], X[prev_fea])
            if abs(corr) > corr_threshold:
                is_redundant = True
                break
        if not is_redundant:
            selected_indices.append(idx)
    return [feature_names[i] for i in selected_indices]

# ============================================================
# 核心绘图函数 (使用预计算的 Top-6)
# ============================================================
def plot_final_custom(X_full, y, top6_features, logfile="descriptor_screening_log.txt"):
    """
    使用预计算的 Top-6 特征进行可视化
    
    参数:
        X_full: 完整的特征数据框（10个特征）
        y: 目标变量
        top6_features: Top-6 特征名称列表
        logfile: 日志文件路径
    """
    # 确保文件夹存在
    if not os.path.exists('./figure/shap'): 
        os.makedirs('./figure/shap', exist_ok=True)
    
    write_log(f"开始绘制 Top-6 特征可视化图: {top6_features}", logfile)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_full), columns=X_full.columns)
    
    # 训练模型评估 SHAP
    rfg = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=1)
    rfg.fit(X_scaled, y.values.ravel())
    
    explainer = shap.TreeExplainer(rfg)
    shap_values = explainer.shap_values(X_scaled)
    
    # 提取 Top-6 对应的数据和 SHAP 值
    top6_indices = [X_full.columns.get_loc(f) for f in top6_features]
    X_top6 = X_full[top6_features]
    X_top6_scaled = X_scaled[top6_features]
    shap_values_top6 = shap_values[:, top6_indices]

    # 1. 特征重要性条形图 (Top-6)
    importance_df = pd.DataFrame({
        'feature': top6_features, 
        'importance': np.abs(shap_values_top6).mean(0)
    }).sort_values(by='importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance_df['feature'], importance_df['importance'], color='#2F365F', edgecolor='black')
    ax.set_xlabel('Mean |SHAP Value|', fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig('./figure/top6_importance_bar.png', dpi=600)
    plt.close()
    write_log("已保存: top6_importance_bar.png", logfile)

    # 2. SHAP Summary Plot (Top-6)
    plt.figure(figsize=FIG_SIZE)
    shap.summary_plot(shap_values_top6, X_top6_scaled, show=False)
    plt.xticks(fontsize=FONT_SIZE); plt.yticks(fontsize=FONT_SIZE)
    plt.xlabel('SHAP value', fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig('./figure/top6_shap_summary.png', dpi=600)
    plt.close()
    write_log("已保存: top6_shap_summary.png", logfile)

    # 3. 依赖图 (Top-6)
    for i, fea in enumerate(top6_features):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        shap.dependence_plot(fea, shap_values_top6, X_top6_scaled, display_features=X_top6, 
                             show=False, ax=ax, dot_size=DOT_SIZE, cmap='coolwarm')
        ax.xaxis.label.set_size(FONT_SIZE); ax.yaxis.label.set_size(FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)
        plt.tight_layout()
        plt.savefig(f'./figure/shap/top6_dep_{fea}.png', dpi=600)
        plt.close()
    write_log(f"已保存: {len(top6_features)} 个依赖图", logfile)

    # 4. 热图 (Pearson) - 仅限 Top-6
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    corr_matrix = X_top6.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True,
                annot_kws={"size": FONT_SIZE}, cbar_kws={'label': 'Pearson Coefficient'})
    ax.tick_params(labelsize=FONT_SIZE)
    plt.tight_layout()

    plt.xticks(rotation=20, ha='center')
    plt.yticks(rotation=45)

    plt.tight_layout()

    plt.savefig('./figure/top6_corr_heatmap.png', dpi=600)
    plt.close()
    write_log("已保存: top6_corr_heatmap.png", logfile)

# ============================================================
# 主计算流程
# ============================================================
def main():
    logfile = "descriptor_screening_log.txt"
    if os.path.exists(logfile): os.remove(logfile)
    write_log("===== 启动特征筛选流程 =====", logfile)

    input_file = './data/smi_pmf.csv' 
    output_file = './data/des_RFE.csv'

    # 初始化文件夹
    for path in ['./data', './figure/shap']:
        if not os.path.exists(path): os.makedirs(path)

    data = pd.read_csv(input_file).drop_duplicates(subset=['SMILES']).reset_index(drop=True)
    data['mol'] = data['SMILES'].map(Chem.MolFromSmiles)
    data = data.dropna(subset=['mol']).reset_index(drop=True)
    y = data['PMF']

    # 1. 描述符计算与方差/基本共线性初筛
    write_log("步骤 1: 计算 Mordred 描述符", logfile)
    calc = Calculator(descriptors, ignore_3D=True)
    des_df = calc.pandas(data['mol'], nproc=os.cpu_count() or 4)
    des_df = des_df.applymap(lambda x: np.nan if isinstance(x, (error.Missing, error.Error)) else x).astype(float).dropna(axis=1)
    des_df = des_df.iloc[:, VarianceThreshold(0).fit(des_df).get_support()]
    write_log(f"方差筛选后特征数: {des_df.shape[1]}", logfile)
    
    # 初步去重 (0.95) 以减小计算量
    corr = des_df.corr().abs()
    to_drop = [col for col in corr.columns if any((corr[col] > 0.95) & (corr.index != col))]
    des_df = des_df.drop(columns=to_drop)
    write_log(f"初步去重后特征数: {des_df.shape[1]}", logfile)

    # 2. 统计相关性筛选
    write_log("步骤 2: 统计相关性筛选", logfile)
    mine = MINE(alpha=0.5, c=10)
    Thresholds = {'pearson': 0.4, 'spearman': 0.35, 'distance': 0.35, 'mic': 0.35}
    selected_mask = []
    for col in des_df.columns:
        xi = des_df[col]
        mine.compute_score(xi, y)
        score = (int(abs(scipy.stats.pearsonr(xi, y)[0]) >= Thresholds['pearson']) +
                 int(abs(scipy.stats.spearmanr(xi, y)[0]) >= Thresholds['spearman']) +
                 int(distcorr(xi, y) >= Thresholds['distance']) +
                 int(mine.mic() >= Thresholds['mic']))
        selected_mask.append(score >= 2)
    X_stat = des_df.loc[:, selected_mask]
    write_log(f"统计筛选后特征数: {X_stat.shape[1]}", logfile)

    # 3. RFECV 自动选择
    write_log(f"步骤 3: RFECV 递归特征消除 (输入特征数: {X_stat.shape[1]})", logfile)
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    selector = RFECV(estimator=rf, step=1, cv=5, scoring='neg_mean_squared_error', min_features_to_select=6)
    selector.fit(X_stat, y)
    rfecv_features = X_stat.columns[selector.support_].tolist()
    write_log(f"RFECV 选出特征数: {len(rfecv_features)}", logfile)

    # 4. RRE 精炼：分别保存 10 个特征（用于建模）和 6 个特征（用于可视化）
    write_log("步骤 4: RRE 精炼 - 基于 SHAP 去除冗余特征", logfile)
    X_rfecv = X_stat[rfecv_features]
    temp_rf = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42).fit(X_rfecv, y)
    shap_vals = shap.TreeExplainer(temp_rf).shap_values(X_rfecv)
    
    # 保存 10 个独立特征到 CSV（用于后续建模）
    final_10_features = get_independent_subset(X_rfecv, shap_vals, rfecv_features, n_target=10, corr_threshold=0.8)
    write_log(f"保存至 CSV 的特征数（用于建模）: {len(final_10_features)}", logfile)
    write_log(f"10 个特征列表: {final_10_features}", logfile)
    
    # 提取 Top-6 特征（用于可视化）
    top6_features = get_independent_subset(X_rfecv, shap_vals, rfecv_features, n_target=6, corr_threshold=0.8)
    write_log(f"用于可视化的 Top-6 特征数: {len(top6_features)}", logfile)
    write_log(f"Top-6 特征列表: {top6_features}", logfile)

    # 5. 保存数据（10 个特征）
    write_log("步骤 5: 保存特征数据到 CSV", logfile)
    final_df = pd.concat([data[['SMILES', 'PMF']], des_df[final_10_features]], axis=1)
    final_df.to_csv(output_file, index=False)
    write_log(f"已保存至: {output_file}", logfile)
    
    # 6. 执行绘图（使用 Top-6 特征）
    write_log("步骤 6: 生成可视化图表", logfile)
    X_for_plot = des_df[final_10_features]
    plot_final_custom(X_for_plot, y, top6_features, logfile)
    
    write_log("===== 任务完成 =====", logfile)

if __name__ == '__main__':
    main()