# train_models_on_pca_stdY.py
import os
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
import xgboost as xgb

# ------- CONFIG -------
DATA_FILES = {
    "des": "./data/des_PCA.csv"
}
TEST_SIZE = 0.2
RANDOM_STATE = 1
CV_FOLDS = 10
OUT_FIG_DIR = "./figure"
OUT_MODEL_DIR = "./pretrained_model"
BO_INIT_POINTS = 10
BO_N_ITERS = 50 # 50

os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_MODEL_DIR, exist_ok=True)

# ------- 绘图函数 -------
def plot_pred(y_train, y_train_pred, y_test, y_test_pred, title_x, title_y, outpath):
    plt.rc('font', family='DejaVu Sans')
    font1 = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 18}
    plt.figure(figsize=(6,5.5))
    plt.plot(y_train, y_train_pred, 'o', color='#233142', label='Training data', markersize=8, alpha=0.9)
    plt.plot(y_test, y_test_pred, 'o', color='#f95959', label='Test data', markersize=8, alpha=0.9)
    plt.legend(loc="lower right", fontsize=18, frameon=False)
    plt.xlabel(title_x, font1)
    plt.ylabel(title_y, font1)
    plt.xlim(15, 80)
    plt.ylim(15, 80)
    plt.plot([15, 80], [15, 80], 'k--', lw=1.5)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

# ------- Cross-Validation R² -------
def cv_score_r2(model, X, y, cv=CV_FOLDS):
    n_jobs=-1 # 可以加速交叉验证
    scores = cross_val_score(model, X, y.ravel(), cv=cv, scoring='r2', n_jobs=-1)
    
    mu = np.mean(scores)
    sigma = np.std(scores)
    
    # 2.0 是惩罚系数。如果模型在某次交叉验证中表现极差，这个值会大幅下降。
    return mu - 2.5 * sigma

# ------- MAIN LOOP -------
for ds_name, ds_file in DATA_FILES.items():
    print("\n" + "="*40)
    print(f"Processing dataset: {ds_name} <- {ds_file}")
    print("="*40)

    if not os.path.exists(ds_file):
        print(f"File not found: {ds_file}")
        continue

    df = pd.read_csv(ds_file)
    if 'PMF' not in df.columns:
        raise ValueError(f"PMF not found in {ds_file}")

    feat_cols = [c for c in df.columns if c not in ['SMILES', 'PMF']]
    X = df[feat_cols].values
    y = df[['PMF']].values

    # === 数据划分 ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # === 标准化 ===
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = X_scaler.fit_transform(X_train)
    X_test_s = X_scaler.transform(X_test)
    y_train_s = y_scaler.fit_transform(y_train).ravel()
    y_test_s = y_scaler.transform(y_test).ravel()

    fig_dir = os.path.join(OUT_FIG_DIR, ds_name, "PCA")
    model_dir = os.path.join(OUT_MODEL_DIR, ds_name, "PCA")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ======== XGBoost ========
    def xgb_cv_obj(learning_rate, n_estimators, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
        model = xgb.XGBRegressor(
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_child_weight=int(min_child_weight),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            gamma=float(gamma),
            reg_alpha=float(reg_alpha),
            reg_lambda=float(reg_lambda),
            random_state=RANDOM_STATE, verbosity=0
        )
        return cv_score_r2(model, X_train_s, y_train_s, cv=CV_FOLDS)

    print(">>> Bayesian optimization for XGBoost ...")
    xgb_optimizer = BayesianOptimization(
        f=xgb_cv_obj,
        pbounds={
            "learning_rate": (0.01, 0.2),
            "n_estimators": (200, 800),
            "max_depth": (3, 10),
            "min_child_weight": (1, 10),
            "subsample": (0.6, 0.9),
            "colsample_bytree": (0.6, 0.9),
            "gamma": (0, 1.0),
            "reg_alpha": (0, 1.0),
            "reg_lambda": (0.1, 10.0)
        },
        random_state=4
    )
    xgb_optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITERS)
    best = xgb_optimizer.max['params']
    model_xgb = xgb.XGBRegressor(
        learning_rate=best['learning_rate'], n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']), min_child_weight=int(best['min_child_weight']),
        subsample=best['subsample'], colsample_bytree=best['colsample_bytree'],
        gamma=best['gamma'], reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'],
        random_state=RANDOM_STATE, verbosity=0
    )
    model_xgb.fit(X_train_s, y_train_s)
    y_train_pred = y_scaler.inverse_transform(model_xgb.predict(X_train_s).reshape(-1,1))
    y_test_pred = y_scaler.inverse_transform(model_xgb.predict(X_test_s).reshape(-1,1))
    print("*** XGBoost ***")
    print("Train R2:", r2_score(y_train, y_train_pred))
    print("Test R2:", r2_score(y_test, y_test_pred))
    joblib.dump(model_xgb, os.path.join(model_dir, "XGboost_Opt.model"))
    plot_pred(y_train, y_train_pred, y_test, y_test_pred,
              'Calculated adsorption energy', 'XGBoost Predicted adsorption energy',
              os.path.join(fig_dir, "XGboost.png"))

    # ======== Random Forest ========
    def rf_cv_obj(n_estimator, max_depths, min_samples_split, min_samples_leaf):
        model = RandomForestRegressor(
            n_estimators=int(n_estimator), max_depth=int(max_depths),
            min_samples_split=int(min_samples_split), min_samples_leaf=int(min_samples_leaf),
            random_state=RANDOM_STATE, n_jobs=-1
        )
        return cv_score_r2(model, X_train_s, y_train_s, cv=CV_FOLDS)

    print(">>> Bayesian optimization for RandomForest ...")
    rf_optimizer = BayesianOptimization(
        f=rf_cv_obj,
        pbounds={"n_estimator": (100, 1000), "max_depths": (5, 12), # <<<<<<<<<<<<<<<<<<<<
                 "min_samples_split": (10, 30), "min_samples_leaf": (5, 15)},
        random_state=2
    )
    rf_optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITERS)
    best = rf_optimizer.max['params']
    model_rf = RandomForestRegressor(
        n_estimators=int(best['n_estimator']), max_depth=int(best['max_depths']),
        min_samples_split=int(best['min_samples_split']), min_samples_leaf=int(best['min_samples_leaf']),
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model_rf.fit(X_train_s, y_train_s)
    y_train_pred = y_scaler.inverse_transform(model_rf.predict(X_train_s).reshape(-1,1))
    y_test_pred = y_scaler.inverse_transform(model_rf.predict(X_test_s).reshape(-1,1))
    print("*** RF ***")
    print("Train R2:", r2_score(y_train, y_train_pred))
    print("Test R2:", r2_score(y_test, y_test_pred))
    joblib.dump(model_rf, os.path.join(model_dir, "RF_Opt.model"))
    plot_pred(y_train, y_train_pred, y_test, y_test_pred,
              'Calculated adsorption energy', 'RF Predicted adsorption energy',
              os.path.join(fig_dir, "RF.png"))

    # ======== Kernel Ridge ========
    def krr_cv_obj(kernel_chose, alpha, gamma, degree, coef0):
        kernels = ["rbf", "laplacian", "polynomial", "sigmoid"] # <<<<<<<<<<<<<<<<<<<
        kernel = kernels[int(kernel_chose) % len(kernels)]
        model = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma, degree=int(degree), coef0=coef0)
        return cv_score_r2(model, X_train_s, y_train_s, cv=CV_FOLDS)

    print(">>> Bayesian optimization for KRR ...")
    krr_optimizer = BayesianOptimization( # <<<<<<<<<<<<<<<<<<<
        f=krr_cv_obj,
        pbounds={
            'kernel_chose': (0, 3.99), 
            'alpha': (1e-4, 10.0), 
            'gamma': (1e-5, 0.1),
            'degree': (2, 4), 
            'coef0': (0, 10) 
        },
        random_state=2
    )
    krr_optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITERS)
    best = krr_optimizer.max['params']
    kernel = ["rbf", "laplacian", "polynomial", "sigmoid"][int(best['kernel_chose'])%4]
    model_krr = KernelRidge(kernel=kernel, alpha=best['alpha'], gamma=best['gamma'],
                            degree=int(best['degree']), coef0=best['coef0'])
    model_krr.fit(X_train_s, y_train_s)
    y_train_pred = y_scaler.inverse_transform(model_krr.predict(X_train_s).reshape(-1,1))
    y_test_pred = y_scaler.inverse_transform(model_krr.predict(X_test_s).reshape(-1,1))
    print("*** KRR ***")
    print("Train R2:", r2_score(y_train, y_train_pred))
    print("Test R2:", r2_score(y_test, y_test_pred))
    joblib.dump(model_krr, os.path.join(model_dir, "KRR_Opt.model"))
    plot_pred(y_train, y_train_pred, y_test, y_test_pred,
              'Calculated adsorption energy', 'KRR Predicted adsorption energy',
              os.path.join(fig_dir, "KRR.png"))

    # ======== MLP ========
    def mlp_cv_obj(alphas, node, nlayer, solver_chose):
        solver = ['adam', 'lbfgs', 'sgd'][int(solver_chose)%3]
        layer_sizes = tuple([int(node)] * int(max(1, min(5, nlayer))))
        model = MLPRegressor(solver=solver, alpha=alphas, hidden_layer_sizes=layer_sizes,
                             max_iter=5000, random_state=RANDOM_STATE)
        return cv_score_r2(model, X_train_s, y_train_s, cv=CV_FOLDS)

    print(">>> Bayesian optimization for MLP ...")
    mlp_optimizer = BayesianOptimization(
        f=mlp_cv_obj,
        pbounds={'alphas': (0.1, 100.0), 'node': (10, 100), 'nlayer': (1, 2), 'solver_chose': (0, 2)}, # <<<<<<<<<<<<<<<<<<<
        random_state=1
    )
    mlp_optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITERS)
    best = mlp_optimizer.max['params']
    solver = ['adam', 'lbfgs', 'sgd'][int(best['solver_chose'])%3]
    layers = tuple([int(best['node'])] * int(max(1, min(5, best['nlayer']))))
    model_mlp = MLPRegressor(solver=solver, alpha=best['alphas'],
                             hidden_layer_sizes=layers, max_iter=5000, random_state=RANDOM_STATE)
    model_mlp.fit(X_train_s, y_train_s)
    y_train_pred = y_scaler.inverse_transform(model_mlp.predict(X_train_s).reshape(-1,1))
    y_test_pred = y_scaler.inverse_transform(model_mlp.predict(X_test_s).reshape(-1,1))
    print("*** MLP ***")
    print("Train R2:", r2_score(y_train, y_train_pred))
    print("Test R2:", r2_score(y_test, y_test_pred))
    joblib.dump(model_mlp, os.path.join(model_dir, "MLP_Opt.model"))
    plot_pred(y_train, y_train_pred, y_test, y_test_pred,
              'Calculated adsorption energy', 'MLP Predicted adsorption energy',
              os.path.join(fig_dir, "MLP.png"))

    # ======== 保存 Scaler ========
    joblib.dump(X_scaler, os.path.join(model_dir, "X_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(model_dir, "y_scaler.pkl"))
    print(f"✅ Models and scalers saved under {model_dir}")

print("\nAll datasets processed successfully.")
