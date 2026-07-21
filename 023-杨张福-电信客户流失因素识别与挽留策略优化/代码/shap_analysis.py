import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from style_config import *

# ============================================================
#  1. 数据读入与清洗（同 model.py）
# ============================================================

df = pd.read_csv("Telco_customer_churn.csv")
df.drop(["CustomerID", "Count", "Country"], axis=1, inplace=True)
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"], errors="coerce"
).fillna(0)

print("=== 数据清洗完成 ===")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")

# ============================================================
#  2. 特征工程（同 model.py）
# ============================================================

drop_cols = [
    "Churn Label", "Churn Score", "Churn Reason",
    "Lat Long", "Latitude", "Longitude",
    "State", "City", "Zip Code",
]

X = df.drop(columns=drop_cols + ["Churn Value"])
y = df["Churn Value"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print(f"编码后特征数: {X_encoded.shape[1]}")

# ---- 切分 + 标准化 ----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

num_cols_in_train = [
    c for c in num_cols if c in X_encoded.columns
]
scaler = StandardScaler()
X_train[num_cols_in_train] = scaler.fit_transform(
    X_train[num_cols_in_train]
)
X_test[num_cols_in_train] = scaler.transform(
    X_test[num_cols_in_train]
)

print("标准化完成")

# ============================================================
#  3. 训练逻辑回归（为 SHAP 提供模型）
# ============================================================

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)
lr.fit(X_train, y_train)

print("逻辑回归训练完成")

# ============================================================
#  4. SHAP 可解释性分析
# ============================================================

explainer = shap.LinearExplainer(lr, X_train)
shap_values = explainer.shap_values(X_test)

X_test_df = X_test.copy()
try:
    X_test_df.columns = [
        c.replace("_", " ") for c in X_test_df.columns
    ]
except Exception:
    pass

# ---- 4.1 SHAP Summary Bar（Top 10 特征重要性） ----
fig, ax = plt.subplots(figsize=(7, 5))
shap.summary_plot(
    shap_values, X_test_df, plot_type="bar",
    max_display=10, show=False,
)
ax.set_title("SHAP 特征重要性（Top 10）", fontsize=13,
             fontweight="bold")
if SAVE:
    plt.savefig("figures/shap_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- 4.2 SHAP 特征贡献（水平柱状图） ----
mean_shap_all = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_shap_all)[-10:]

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(
    range(10),
    mean_shap_all[top_idx],
    color="#5A7B9A", alpha=0.85,
)
ax.set_yticks(range(10))
ax.set_yticklabels(
    [X_test_df.columns[i] for i in top_idx],
    fontsize=9,
)
ax.set_xlabel("平均 |SHAP| 值", fontsize=11)
ax.set_title("SHAP 特征重要性（Top 10）", fontsize=13,
             fontweight="bold")
ax.invert_yaxis()
if SAVE:
    plt.savefig("figures/shap_beeswarm.png", dpi=300,
                bbox_inches="tight")
plt.close()

# ---- 4.3 SHAP Dependence（Top 2 特征依赖图，纯 matplotlib） ----
mean_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_shap)[-2:]

for idx in reversed(top_idx):
    feat_name = X_test_df.columns[idx]
    shap_col = shap_values[:, idx]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(
        X_test_df.iloc[:, idx],
        shap_col,
        c=shap_col, cmap="RdBu_r",
        alpha=0.5, s=10,
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feat_name, fontsize=11)
    ax.set_ylabel("SHAP 值", fontsize=11)
    ax.set_title(f"SHAP 依赖图 — {feat_name}", fontsize=13,
                 fontweight="bold")
    if SAVE:
        fname = f"figures/shap_{feat_name[:15].replace(' ', '_')}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

# ---- 4.4 打印 Top 5 关键因子 ----
mean_shap_df = pd.DataFrame({
    "特征": X_test_df.columns,
    "平均 |SHAP|": mean_shap,
})
mean_shap_df.sort_values(
    "平均 |SHAP|", ascending=False, inplace=True
)

print(f"\n{'='*50}")
print("Top 5 关键流失因子")
print(f"{'='*50}")
for _, row in mean_shap_df.head(5).iterrows():
    print(f"  {row['特征']:30s}  "
          f"SHAP = {row['平均 |SHAP|']:.4f}")

print("\nSHAP 分析完成，图片已保存至 figures/ 文件夹")
