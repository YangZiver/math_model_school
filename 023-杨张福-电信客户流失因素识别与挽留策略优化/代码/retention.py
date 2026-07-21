import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from style_config import *

# ============================================================
#  1. 数据准备（复用 model.py 的处理）
# ============================================================

df = pd.read_csv("Telco_customer_churn.csv")
df.drop(["CustomerID", "Count", "Country"], axis=1, inplace=True)
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)

drop_cols = [
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "Lat Long",
    "Latitude",
    "Longitude",
    "State",
    "City",
    "Zip Code",
]
X = df.drop(columns=drop_cols + ["Churn Value"])
y = df["Churn Value"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

num_cols_in_train = [c for c in num_cols if c in X_encoded.columns]
scaler = StandardScaler()
X_train[num_cols_in_train] = scaler.fit_transform(X_train[num_cols_in_train])
X_test[num_cols_in_train] = scaler.transform(X_test[num_cols_in_train])

# ============================================================
#  2. 训练逻辑回归 + 预测全量流失概率
# ============================================================

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)
lr.fit(X_train, y_train)

# 对训练集和测试集分别预测，合并
y_prob_train = lr.predict_proba(X_train)[:, 1]
y_prob_test = lr.predict_proba(X_test)[:, 1]
all_probs = np.concatenate([y_prob_train, y_prob_test])

# 合并完整的原始数据和预测概率
df_full = df.copy()
df_full["Churn_Prob"] = np.nan
df_full.loc[X_train.index, "Churn_Prob"] = y_prob_train
df_full.loc[X_test.index, "Churn_Prob"] = y_prob_test

print("=== 预测概率统计 ===")
print(f"均值: {all_probs.mean():.3f}  标准差: {all_probs.std():.3f}")
print(f"最小值: {all_probs.min():.3f}  最大值: {all_probs.max():.3f}")

# ============================================================
#  3. 分层：用分位数划分风险级别
# ============================================================

p50 = np.percentile(all_probs, 50)
p80 = np.percentile(all_probs, 80)


def assign_tier(prob):
    if prob > p80:
        return "高风险"
    elif prob > p50:
        return "中风险"
    else:
        return "低风险"


df_full["Risk_Tier"] = df_full["Churn_Prob"].apply(assign_tier)

tier_counts = df_full["Risk_Tier"].value_counts()
tier_order = ["高风险", "中风险", "低风险"]

print(f"\n=== 分层阈值 ===")
print(f"低风险 ≤ {p50:.3f}  < 中风险 ≤ {p80:.3f}  < 高风险")
print(f"\n=== 各层级人数 ===")
total = len(df_full)
for t in tier_order:
    cnt = tier_counts[t]
    print(f"  {t}: {cnt:5d} 人 ({cnt / total * 100:.1f}%)")

# ---- 3.1 分层饼图 ----
fig, ax = plt.subplots(figsize=(5, 4))
ax.pie(
    [tier_counts[t] for t in tier_order],
    labels=[f"{t}\n({tier_counts[t]}人)" for t in tier_order],
    autopct="%1.1f%%",
    startangle=90,
    colors=[C_HIGH, C_MID, C_LOW],
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
)
ax.set_title("客户流失风险分层", fontsize=13, fontweight="bold")
if SAVE:
    plt.savefig("figures/risk_tiers.png", dpi=300)
plt.show()

# ============================================================
#  4. 各层级特征对比分析
# ============================================================

key_metrics = {
    "Tenure Months": "在网时长（月）",
    "Monthly Charges": "月消费金额（美元）",
    "Total Charges": "总消费金额（美元）",
    "CLTV": "生命周期价值",
}

tier_stats = []
for t in tier_order:
    subset = df_full[df_full["Risk_Tier"] == t]
    row = {"层级": t, "人数": len(subset)}
    for col, _ in key_metrics.items():
        row[col] = subset[col].mean()
    # 关键分类特征的比例
    row["光纤比例"] = (subset["Internet Service"] == "Fiber optic").mean()
    row["月付合同比例"] = (subset["Contract"] == "Month-to-month").mean()
    row["有家属比例"] = (subset["Dependents"] == "Yes").mean()
    tier_stats.append(row)

tier_df = pd.DataFrame(tier_stats)

print(f"\n=== 各层级特征对比 ===")
print(tier_df.to_string(index=False))

# ---- 4.1 特征对比雷达图/柱状图 ----
compare_cols = [
    "Tenure Months",
    "Monthly Charges",
    "光纤比例",
    "月付合同比例",
    "有家属比例",
]
fig, axes = plt.subplots(1, 3, figsize=(9, 4))
colors = [C_HIGH, C_MID, C_LOW]

for i, col in enumerate(compare_cols):
    vals = [tier_df[tier_df["层级"] == t][col].values[0] for t in tier_order]
    axes[i % 3].bar(
        tier_order,
        vals,
        color=colors,
        alpha=0.8,
        width=0.5,
    )
    axes[i % 3].set_title(col, fontsize=11, fontweight="bold")
    axes[i % 3].tick_params(labelsize=8)

plt.tight_layout()
if SAVE:
    plt.savefig("figures/tier_comparison.png", dpi=300)
plt.show()

# ============================================================
#  5. 投入产出估算
# ============================================================

high_risk_count = tier_counts["高风险"]
mid_risk_count = tier_counts["中风险"]

reach_rate = 0.60  # 触达率（假设）
retain_rate = 0.30  # 挽留成功率（假设）
cost_per_customer = 50  # 人均优惠成本（美元）
avg_cltv = df_full["CLTV"].mean()

reached = high_risk_count * reach_rate
saved = reached * retain_rate
revenue_saved = saved * avg_cltv
total_cost = reached * cost_per_customer
roi = (revenue_saved - total_cost) / total_cost

print("\n=== 投入产出估算 ===")
print(f"高风险客户数: {high_risk_count}")
print(f"触达率: {reach_rate:.0%} → 触达 {reached:.0f} 人")
print(f"挽回成功率: {retain_rate:.0%} → 挽回 {saved:.0f} 人")
print(f"人均 CLTV: ${avg_cltv:.0f}")
print(f"预期挽回收益: ${revenue_saved:,.0f}")
print(f"挽留总成本: ${total_cost:,.0f}")
print(f"投资回报率 (ROI): {roi:.2f} 倍")
print(f"净收益: ${revenue_saved - total_cost:,.0f}")

print("\n所有分层图片已保存至 figures/ 文件夹")
