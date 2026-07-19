import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
import warnings
warnings.filterwarnings("ignore")

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
#  2. 可视化全局配置
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_NOT = "#7EA7C4"
COLOR_CHURN = "#DD8452"
COLOR_BAR = "#5A7B9A"
ALPHA = 0.6

CAT_PALETTE = [
    "#4477AA", "#66CCEE", "#228833", "#CCBB44",
    "#EE6677", "#AA3377", "#44BB99", "#000000",
]

SAVE = True

# ============================================================
#  3. EDA 可视化
# ============================================================

# -------------------- 3.1 整体流失率（环形图）----------------

def plot_churn_rate():
    churn_rate = df["Churn Value"].mean()
    churn_count = df["Churn Value"].sum()
    not_count = len(df) - churn_count

    fig, ax = plt.subplots(figsize=(5, 4.5))

    wedges, texts, autotexts = ax.pie(
        [not_count, churn_count],
        labels=["未流失", "流失"],
        autopct="%1.1f%%",
        startangle=90,
        colors=[COLOR_NOT, COLOR_CHURN],
        wedgeprops={
            "width": 0.4, "edgecolor": "white",
            "linewidth": 1.5,
        },
        textprops={"fontsize": 10},
        pctdistance=0.75,
        labeldistance=1.1,
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")

    ax.text(
        0, 0, f"{churn_rate:.1%}",
        ha="center", va="center", fontsize=22,
        fontweight="bold", color="#444444",
    )

    ax.set_title("整体流失率", fontsize=14, fontweight="bold", pad=12)
    if SAVE:
        plt.savefig("figures/整体流失率.png", dpi=300)
    plt.show()

# ------------ 3.2 数值特征分布对比（GridSpec 2x3）-----------

def plot_numerical_features():
    num_cols = [
        ("Tenure Months", "在网时长（月）"),
        ("Monthly Charges", "月消费金额（美元）"),
        ("Total Charges", "总消费金额（美元）"),
    ]
    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    for i, (col, xlabel) in enumerate(num_cols):
        # 直方图
        ax_hist = fig.add_subplot(gs[0, i])
        for val, clr, lbl in [
            (0, COLOR_NOT, "未流失"), (1, COLOR_CHURN, "流失")
        ]:
            data = df[df["Churn Value"] == val][col].dropna()
            ax_hist.hist(
                data, bins=40, alpha=ALPHA,
                color=clr, label=lbl, edgecolor="white", linewidth=0.3,
            )
        ax_hist.set_xlabel(xlabel, fontsize=11)
        ax_hist.set_ylabel("频数", fontsize=11)
        ax_hist.tick_params(labelsize=9)
        if i == 0:
            ax_hist.legend(fontsize=9, loc="upper right")

        # 箱线图
        ax_box = fig.add_subplot(gs[1, i])
        bp_data = [
            df[df["Churn Value"] == v][col].dropna()
            for v in [0, 1]
        ]
        bp = ax_box.boxplot(
            bp_data, patch_artist=True, widths=0.4,
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            medianprops={"linewidth": 1.2, "color": "#CC0000"},
        )
        for patch, clr in zip(bp["boxes"], [COLOR_NOT, COLOR_CHURN]):
            patch.set_facecolor(clr)
            patch.set_alpha(ALPHA)
        ax_box.set_xticklabels(["未流失", "流失"], fontsize=9)
        ax_box.set_ylabel(xlabel, fontsize=11)
        ax_box.tick_params(labelsize=9)

    fig.suptitle(
        "数值特征分布对比", fontsize=14, fontweight="bold", y=1.01
    )
    if SAVE:
        plt.savefig("figures/数值特征分布对比.png", dpi=300)
    plt.show()

# ------------ 3.3 分类特征流失率对比（GridSpec 2x4）-----------

def _make_gradient(base_rgb, n, reverse=True):
    """从 base_rgb 到白色的渐变色列表"""
    factor = 0.85
    if reverse:
        weights = [factor - (factor - 0.35) * j / max(n - 1, 1)
                   for j in range(n)]
    else:
        weights = [0.35 + (factor - 0.35) * j / max(n - 1, 1)
                   for j in range(n)]
    return [
        tuple(base_rgb[k] * w + 1.0 * (1 - w) for k in range(3))
        for w in weights
    ]

def plot_categorical_features():
    cat_cols = [
        "Contract", "Internet Service", "Payment Method",
        "Senior Citizen", "Partner", "Dependents",
        "Phone Service", "Paperless Billing",
    ]
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3)

    for i, col in enumerate(cat_cols):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        rate = (
            df.groupby(col)["Churn Value"]
            .mean()
            .sort_values(ascending=False)
        )
        count = df[col].value_counts()
        x_pos = range(len(rate))
        n_bars = len(rate)

        base_rgb = to_rgb(CAT_PALETTE[i])
        bar_colors = _make_gradient(base_rgb, n_bars, reverse=True)

        bars = ax.bar(
            x_pos, rate.values,
            color=bar_colors, width=0.6,
            edgecolor="white", linewidth=0.6,
        )
        for j, (k, v) in enumerate(rate.items()):
            ax.text(
                j, v + 0.015, f"n={count[k]}",
                ha="center", fontsize=8,
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            rate.index, fontsize=7, rotation=20, ha="right",
        )
        ax.set_ylabel("流失率", fontsize=10)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)

    fig.suptitle(
        "分类特征流失率对比", fontsize=14, fontweight="bold", y=1.02
    )
    if SAVE:
        plt.savefig("figures/分类特征流失率对比.png", dpi=300)
    plt.show()

# ----------------- 3.4 相关性热力图 -----------------

def plot_corr_heatmap():
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    col_labels = [
        "Zip Code", "Latitude", "Longitude",
        "Tenure", "Monthly\nCharges", "Total\nCharges",
        "Churn\nValue", "Churn\nScore", "CLTV",
    ]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(col_labels, fontsize=9)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            clr = "white" if abs(val) > 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=7,
                color=clr,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Pearson 相关系数", fontsize=10)
    ax.set_title("数值特征相关系数热力图", fontsize=13, fontweight="bold")

    if SAVE:
        plt.savefig("figures/相关性热力图.png", dpi=300)
    plt.show()

    churn_corr = corr["Churn Value"].sort_values(ascending=False)
    print("\n=== 数值特征与流失率的相关系数 ===")
    print(churn_corr)

# ============================================================
#  4. 主程序入口
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
