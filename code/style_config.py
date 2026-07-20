"""统一绘图风格 —— 对标顶刊出版质量"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# -----------------------------------------------------------
#  全局参数
# -----------------------------------------------------------
plt.rcParams.update({
    # 字体层级
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # 线条与标记
    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    # 刻度
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # 输出
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # 坐标轴框架
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
})

# -----------------------------------------------------------
#  seaborn 底层样式
# -----------------------------------------------------------
sns.set_style("white")

# -----------------------------------------------------------
#  中文字体（必须在 seaborn 之后设置，否则被覆盖）
# -----------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
})

# -----------------------------------------------------------
#  统一配色方案（colorblind-friendly，偏冷稳重色调）
# -----------------------------------------------------------
BLUE   = "#3B7DD8"   # 未流失 / 低风险
RED    = "#D8413A"   # 流失 / 高风险
GREEN  = "#2E8B57"   # XGBoost
GRAY   = "#6C7A89"   # 中风险

# 三模型配色
C_LR = RED              # 逻辑回归
C_RF = BLUE             # 随机森林
C_XGB = GREEN           # XGBoost

# 三风险层级配色
C_HIGH  = RED           # 高风险
C_MID   = GRAY          # 中风险
C_LOW   = BLUE          # 低风险

# 两分类配色
C_NOT_CHURN = BLUE
C_CHURN     = RED

SAVE = True
