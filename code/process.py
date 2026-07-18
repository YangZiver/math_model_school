import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  1. 数据读入与清洗
# ============================================================

# 将xlsx文件转换成csv文件，运行一次后注释
# df = pd.read_excel("Telco_customer_churn.xlsx")
# df.to_csv("Telco_customer_churn.csv", index=False, encoding="utf-8-sig")

df = pd.read_csv("Telco_customer_churn.csv")
df.drop(["CustomerID", "Count", "Country"], axis=1, inplace=True)
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)

print("=== 数据清洗完成 ===")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
print()

# ============================================================
#  2. EDA —— 数据探索与可视化
# ============================================================

# 中文字体设置（适配 Windows）
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

SAVE = True  # 是否保存图片到 figures/

# -------------------- 2.1 整体流失率 --------------------

churn_rate = df["Churn Value"].mean()
print(f"整体流失率: {churn_rate:.2%}")
print(f"流失客户数: {df['Churn Value'].sum():.0f}")
print(f"未流失客户数: {(1 - df['Churn Value']).sum():.0f}")
print()

# -------------------- 2.2 数值特征分布对比 --------------------

num_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for i, col in enumerate(num_cols):
    # 直方图
    for val, label in [(0, "未流失"), (1, "流失")]:
        subset = df[df["Churn Value"] == val][col]
        axes[0, i].hist(subset, bins=40, alpha=0.6, label=label)
    axes[0, i].set_xlabel(col)
    axes[0, i].set_ylabel("频数")
    axes[0, i].legend()
    axes[0, i].set_title(f"{col} 分布对比")

    # 箱线图
    bp_data = [df[df["Churn Value"] == v][col].dropna() for v in [0, 1]]
    axes[1, i].boxplot(bp_data)
    axes[1, i].set_xticklabels(["未流失", "流失"])
    axes[1, i].set_ylabel(col)
    axes[1, i].set_title(f"{col} 箱线图")

plt.tight_layout()
if SAVE:
    plt.savefig("figures/数值特征分布对比.png", dpi=150, bbox_inches="tight")
plt.show()

# -------------------- 2.3 分类特征流失率对比 --------------------

cat_cols = [
    "Contract", "Internet Service", "Payment Method",
    "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Paperless Billing",
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    rate = df.groupby(col)["Churn Value"].mean().sort_values(ascending=False)
    count = df[col].value_counts()
    # 柱状图显示流失率
    axes[i].bar(rate.index, rate.values, color="coral", alpha=0.8)
    # 在柱子上标注样本数
    for j, (k, v) in enumerate(rate.items()):
        axes[i].text(j, v + 0.01, f"n={count[k]}", ha="center", fontsize=8)
    axes[i].set_title(f"不同 {col} 的流失率")
    axes[i].set_ylabel("流失率")
    axes[i].tick_params(axis="x", rotation=15)

plt.tight_layout()
if SAVE:
    plt.savefig("figures/分类特征流失率对比.png", dpi=150, bbox_inches="tight")
plt.show()

# -------------------- 2.4 相关性热力图 --------------------

num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
churn_corr = corr["Churn Value"].sort_values(ascending=False)

print("=== 各数值特征与流失率的相关系数 ===")
print(churn_corr)
print()

# 热力图
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        val = corr.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="black")

ax.set_title("数值特征相关系数热力图")
plt.tight_layout()
if SAVE:
    plt.savefig("figures/相关性热力图.png", dpi=150, bbox_inches="tight")
plt.show()

print("=== EDA 完成，所有图片已保存到 figures/ 文件夹 ===")
