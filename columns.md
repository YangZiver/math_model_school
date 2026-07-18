# C题 电信客户流失数据 — 列名中英对照表

数据文件：`Telco_customer_churn.xlsx`（7043 条记录，33 列）

| # | 英文列名 | 含义 | 类型 | 说明 |
|---|----------|------|------|------|
| 1 | `CustomerID` | 客户ID | 文本 | 唯一标识，建模时丢弃 |
| 2 | `Count` | 计数 | 数值 | 恒为 1，建模时丢弃 |
| 3 | `Country` | 国家 | 文本 | 均为 United States，建模时丢弃 |
| 4 | `State` | 州/省 | 文本 | 分类特征 |
| 5 | `City` | 城市 | 文本 | 分类特征 |
| 6 | `Zip Code` | 邮编 | 文本 | 分类特征 |
| 7 | `Lat Long` | 经纬度 | 文本 | 地理聚合字符串 |
| 8 | `Latitude` | 纬度 | 数值 | 地理特征 |
| 9 | `Longitude` | 经度 | 数值 | 地理特征 |
| 10 | `Gender` | 性别 | 二分类 | Male / Female |
| 11 | `Senior Citizen` | 是否老年人 | 二分类 | Yes / No |
| 12 | `Partner` | 是否有配偶/伴侣 | 二分类 | Yes / No |
| 13 | `Dependents` | 是否有家属/子女 | 二分类 | Yes / No |
| 14 | `Tenure Months` | 在网时长（月数） | 数值 | 核心特征 |
| 15 | `Phone Service` | 是否开通电话服务 | 二分类 | Yes / No |
| 16 | `Multiple Lines` | 是否多条电话线 | 多分类 | Yes / No / No phone service |
| 17 | `Internet Service` | 互联网服务类型 | 多分类 | DSL / Fiber optic / No |
| 18 | `Online Security` | 是否开通在线安全服务 | 多分类 | Yes / No / No internet service |
| 19 | `Online Backup` | 是否开通在线备份服务 | 多分类 | Yes / No / No internet service |
| 20 | `Device Protection` | 是否开通设备保护服务 | 多分类 | Yes / No / No internet service |
| 21 | `Tech Support` | 是否开通技术支持服务 | 多分类 | Yes / No / No internet service |
| 22 | `Streaming TV` | 是否开通流媒体电视 | 多分类 | Yes / No / No internet service |
| 23 | `Streaming Movies` | 是否开通流媒体电影 | 多分类 | Yes / No / No internet service |
| 24 | `Contract` | 合同类型 | 多分类 | Month-to-month / One year / Two year（核心特征）|
| 25 | `Paperless Billing` | 是否无纸化账单 | 二分类 | Yes / No |
| 26 | `Payment Method` | 支付方式 | 多分类 | Mailed check / Electronic check / Credit card (automatic) / Bank transfer (automatic) |
| 27 | `Monthly Charges` | 月消费金额 | 数值 | 连续值，核心特征 |
| 28 | `Total Charges` | 总消费金额 | 数值 | 连续值 |
| 29 | `Churn Label` | 是否流失（标签） | 二分类 | Yes / No（可读展示用） |
| 30 | `Churn Value` | 是否流失（数值） | 二分类 | 1 = 流失, 0 = 未流失（**目标变量**）|
| 31 | `Churn Score` | 流失风险评分 | 数值 | 已有流失评分（分析可对比） |
| 32 | `CLTV` | 客户生命周期价值 | 数值 | Customer Lifetime Value |
| 33 | `Churn Reason` | 流失原因 | 文本 | 仅流失客户有值，EDA/报告用 |

---

### 特征分组

| 分组 | 包含列 |
|------|--------|
| **人口统计** | Gender, Senior Citizen, Partner, Dependents |
| **账户/合同信息** | Tenure Months, Contract, Paperless Billing, Payment Method |
| **费用信息** | Monthly Charges, Total Charges, CLTV |
| **服务订阅** | Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies |
| **地理信息** | Country, State, City, Zip Code, Lat Long, Latitude, Longitude |
| **目标变量** | Churn Label, Churn Value |
| **额外信息** | CustomerID, Count, Churn Score, Churn Reason |
