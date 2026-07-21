import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from style_config import *

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
#  2. EDA 可视化
# ============================================================

# ------------ 2.1 整体流失率（环形图）------------

def plot_churn_rate():
    churn_count = df["Churn Value"].sum()
    not_count = len(df) - churn_count
    churn_pct = churn_count / len(df) * 100

    fig, ax = plt.subplots(figsize=(5, 4.5))
    colors_pie = [C_NOT_CHURN, C_CHURN]

    wedges, texts = ax.pie(
        [not_count, churn_count],
        labels=["未流失\n{:,d} 人".format(not_count),
                "流失\n{:,d} 人".format(churn_count)],
        startangle=90,
        colors=colors_pie,
        wedgeprops={
            "width": 0.45,
            "edgecolor": "white",
            "linewidth": 2,
        },
        textprops={"fontsize": 10},
    )

    # 标注百分比
    for w, pct in zip(wedges, [100-churn_pct, churn_pct]):
        ang = (w.theta2 - w.theta1) / 2 + w.theta1
        x = np.cos(np.deg2rad(ang)) * 0.72
        y = np.sin(np.deg2rad(ang)) * 0.72
        ax.text(x, y, f"{pct:.1f}%",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white")

    ax.set_title("客户流失分布", fontsize=14, fontweight="bold", pad=15)
    if SAVE:
        plt.savefig("figures/整体流失率.png", dpi=300)
    plt.show()

# ------------ 2.2 数值特征分布对比 ------------

def plot_numerical_features():
    num_cols = [
        ("Tenure Months", "在网时长（月）"),
        ("Monthly Charges", "月消费金额（美元）"),
        ("Total Charges", "总消费金额（美元）"),
    ]
    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    for i, (col, xlabel) in enumerate(num_cols):
        ax_hist = fig.add_subplot(gs[0, i])
        for val, clr, lbl in [(0, C_NOT_CHURN, "未流失"),
                               (1, C_CHURN, "流失")]:
            d = df[df["Churn Value"] == val][col].dropna()
            ax_hist.hist(d, bins=40, alpha=0.55, color=clr,
                         label=lbl, edgecolor="white", linewidth=0.3)
        ax_hist.set_xlabel(xlabel, fontsize=10)
        ax_hist.set_ylabel("频数", fontsize=10)
        ax_hist.tick_params(labelsize=8)
        if i == 0:
            ax_hist.legend(fontsize=8, loc="upper right",
                           frameon=False)

        ax_box = fig.add_subplot(gs[1, i])
        bp_data = [df[df["Churn Value"] == v][col].dropna()
                   for v in [0, 1]]
        bp = ax_box.boxplot(
            bp_data, patch_artist=True, widths=0.45,
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8, "linestyle": "--"},
            medianprops={"linewidth": 1.2, "color": "#222222"},
            flierprops={"markersize": 3, "markerfacecolor": "#999999",
                        "alpha": 0.6},
        )
        for patch, clr in zip(bp["boxes"],
                              [C_NOT_CHURN, C_CHURN]):
            patch.set_facecolor(clr)
            patch.set_alpha(0.45)
        ax_box.set_xticklabels(["未流失", "流失"], fontsize=9)
        ax_box.set_ylabel(xlabel, fontsize=10)
        ax_box.tick_params(labelsize=8)

    fig.suptitle("数值特征分布对比", fontsize=14,
                 fontweight="bold", y=1.01)
    if SAVE:
        plt.savefig("figures/数值特征分布对比.png", dpi=300)
    plt.show()

# ------------ 2.3 分类特征流失率对比 ------------

CAT_ORDER = [
    "Contract", "Internet Service", "Payment Method",
    "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Paperless Billing",
]

CAT_TITLES = {
    "Contract": "合同类型",
    "Internet Service": "互联网服务",
    "Payment Method": "支付方式",
    "Senior Citizen": "是否老年人",
    "Partner": "是否有配偶",
    "Dependents": "是否有家属",
    "Phone Service": "电话服务",
    "Paperless Billing": "无纸化账单",
}

def plot_categorical_features():
    fig = plt.figure(figsize=(12, 6.5))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    for i, col in enumerate(CAT_ORDER):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        churn_rates = (df.groupby(col)["Churn Value"]
                       .mean().sort_values(ascending=False))
        counts = df[col].value_counts()
        n = len(churn_rates)

        # 渐变色条
        from matplotlib.colors import to_rgb
        base = to_rgb(BLUE)
        shades = [tuple(base[k] * (0.45 + 0.5 * j / max(n-1, 1))
                        + 1.0 * (0.55 - 0.5 * j / max(n-1, 1))
                        for k in range(3))
                  for j in range(n)]

        bars = ax.bar(range(n), churn_rates.values,
                      color=shades, width=0.55,
                      edgecolor="white", linewidth=0.5)
        for j, (k, v) in enumerate(churn_rates.items()):
            ax.text(j, v + 0.02, f"n={counts[k]}",
                    ha="center", fontsize=7, color="#555555")

        ax.set_xticks(range(n))
        ax.set_xticklabels(churn_rates.index,
                           fontsize=7, rotation=20, ha="right")
        ax.set_ylabel("流失率", fontsize=9)
        ax.set_title(CAT_TITLES.get(col, col),
                     fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)
        ymax = churn_rates.values.max()
        ax.set_ylim(0, min(1.0, ymax * 1.3))

    fig.suptitle("分类特征流失率对比", fontsize=14,
                 fontweight="bold", y=1.02)
    if SAVE:
        plt.savefig("figures/分类特征流失率对比.png", dpi=300)
    plt.show()

# ------------ 2.4 相关性热力图 ------------

def plot_corr_heatmap():
    num_df = df.select_dtypes(include=[np.number])
    # 按 Churn Value 降序排列
    corr_order = num_df.corr()["Churn Value"].sort_values(
        ascending=False).index.tolist()
    corr = num_df[corr_order].corr()

    labels = {
        "Churn Value": "流失\n标签",
        "Churn Score": "流失\n评分",
        "CLTV": "CLTV",
        "Total Charges": "总消费",
        "Monthly Charges": "月消费",
        "Tenure Months": "在网\n时长",
        "Longitude": "经度",
        "Latitude": "纬度",
        "Zip Code": "邮编",
    }
    col_labels = [labels.get(c, c) for c in corr_order]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = sns.diverging_palette(250, 15, s=75, l=40, center="light",
                                  as_cmap=True)
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1,
                   aspect="equal")

    ax.set_xticks(range(len(corr_order)))
    ax.set_yticks(range(len(corr_order)))
    ax.set_xticklabels(col_labels, fontsize=9,
                       rotation=30, ha="right")
    ax.set_yticklabels(col_labels, fontsize=9)

    for i in range(len(corr_order)):
        for j in range(len(corr_order)):
            val = corr.iloc[i, j]
            clr = "white" if abs(val) > 0.55 else "#333333"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=clr, weight="bold"
                    if abs(val) > 0.55 else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Pearson r", fontsize=10)
    ax.set_title("数值特征相关系数矩阵", fontsize=13,
                 fontweight="bold", pad=12)

    if SAVE:
        plt.savefig("figures/相关性热力图.png", dpi=300)
    plt.show()

    churn_corr = corr["Churn Value"].sort_values(ascending=False)
    print("\n=== 数值特征与流失率的相关系数 ===")
    print(churn_corr)

# ============================================================
#  3. 主程序入口
# ============================================================

if __name__ == "__main__":
    churn_rate = df["Churn Value"].mean()
    print(f"整体流失率: {churn_rate:.2%}")
    print(f"流失: {df['Churn Value'].sum():.0f} / "
          f"未流失: {(1 - df['Churn Value']).sum():.0f}\n")

    plot_churn_rate()
    plot_numerical_features()
    plot_categorical_features()
    plot_corr_heatmap()

    print("\n所有图片已保存至 figures/ 文件夹")
