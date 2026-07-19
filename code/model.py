import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 12,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SAVE = True

# ============================================================
#  1. 数据读入与清洗
# ============================================================

df = pd.read_csv("Telco_customer_churn.csv")
df.drop(["CustomerID", "Count", "Country"], axis=1, inplace=True)
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"], errors="coerce"
).fillna(0)

print("=== 数据清洗完成 ===")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")

# ============================================================
#  2. 特征选择：剔除无信息/泄露列
# ============================================================

drop_cols = [
    "Churn Label",       # 与 Churn Value 等价
    "Churn Score",       # 数据泄露
    "Churn Reason",      # 仅流失客户有值
    "Lat Long",          # 经纬度冗余列
    "Latitude",          # EDA 确认相关系数 < 0.01
    "Longitude",         # 同上
    "State",             # 高基数分类
    "City",              # 高基数分类
    "Zip Code",          # 高基数分类
]

X = df.drop(columns=drop_cols + ["Churn Value"])
y = df["Churn Value"]

print(f"\n候选特征列数: {X.shape[1]}")
print(f"目标变量分布: 0={sum(y==0)}, 1={sum(y==1)}")

# ============================================================
#  3. 类别特征编码（One-Hot）
# ============================================================

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n分类特征 ({len(cat_cols)}): {cat_cols}")
print(f"数值特征 ({len(num_cols)}): {num_cols}")

X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print(f"编码后特征数: {X_encoded.shape[1]}")

# ============================================================
#  4. 训练集 / 测试集划分
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print(f"\n训练集: {X_train.shape[0]} 条")
print(f"测试集: {X_test.shape[0]} 条")
print(f"训练集流失率: {y_train.mean():.2%}")
print(f"测试集流失率: {y_test.mean():.2%}")

# ============================================================
#  5. 数值特征标准化
# ============================================================

from sklearn.preprocessing import StandardScaler

num_cols_in_train = [
    c for c in num_cols
    if c in X_encoded.columns
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
#  6. 逻辑回归建模
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

# ============================================================
#  7. 模型评估
# ============================================================

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*40}")
print("逻辑回归评估结果")
print(f"{'='*40}")
print(f"Accuracy:  {acc:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred))

# ---- 7.1 混淆矩阵 ----
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["未流失", "流失"]).plot(
    ax=ax, cmap="Blues", values_format="d",
)
ax.set_title("逻辑回归混淆矩阵", fontsize=13, fontweight="bold")
if SAVE:
    plt.savefig("figures/confusion_matrix.png", dpi=300)
plt.show()

# ---- 7.2 ROC 曲线 ----
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#DD8452", linewidth=2,
        label=f"逻辑回归 (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
ax.set_xlabel("假正率 (FPR)", fontsize=11)
ax.set_ylabel("真正率 (TPR)", fontsize=11)
ax.set_title("ROC 曲线", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
if SAVE:
    plt.savefig("figures/roc_curve.png", dpi=300)
plt.show()

# ============================================================
#  8. 特征重要性（逻辑回归系数）
# ============================================================

coef_df = pd.DataFrame({
    "特征": X_encoded.columns,
    "系数": lr.coef_[0],
})
coef_df["|系数|"] = coef_df["系数"].abs()
coef_df.sort_values("|系数|", ascending=False, inplace=True)

print(f"\n{'='*50}")
print("特征系数排序（绝对值降序）")
print(f"{'='*50}")
for _, row in coef_df.head(15).iterrows():
    direction = "↑ 正相关" if row["系数"] > 0 else "↓ 负相关"
    print(f"  {row['特征']:30s}  {row['系数']:+.4f}  {direction}")

# ---- 8.1 特征重要性图（前 10） ----
top10 = coef_df.head(10)
colors = ["#DD8452" if v > 0 else "#4C72B0"
          for v in top10["系数"]]
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(range(len(top10)), top10["系数"], color=colors, alpha=0.8)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10["特征"], fontsize=9)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("逻辑回归系数", fontsize=11)
ax.set_title("Top 10 特征系数", fontsize=13, fontweight="bold")
if SAVE:
    plt.savefig("figures/feature_importance.png", dpi=300)
plt.show()

# ============================================================
#  9. 对比模型：随机森林（可选）
# ============================================================

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    class_weight="balanced",
    n_estimators=100,
    random_state=42,
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

auc_rf = roc_auc_score(y_test, y_prob_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"\n{'='*40}")
print("对比模型评估")
print(f"{'='*40}")
print(f"随机森林 - Accuracy: {acc_rf:.4f}, AUC: {auc_rf:.4f}")

# ---- 9.1 两模型 ROC 曲线对比 ----
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#DD8452", linewidth=2,
        label=f"逻辑回归 (AUC = {auc:.3f})")
ax.plot(fpr_rf, tpr_rf, color="#4C72B0", linewidth=2,
        label=f"随机森林 (AUC = {auc_rf:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
ax.set_xlabel("假正率 (FPR)", fontsize=11)
ax.set_ylabel("真正率 (TPR)", fontsize=11)
ax.set_title("模型 ROC 曲线对比", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
if SAVE:
    plt.savefig("figures/roc_compare.png", dpi=300)
plt.show()

print("\n所有模型图片已保存至 figures/ 文件夹")
