import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import io
import gc
warnings.filterwarnings('ignore')

# -------------------------- Streamlit页面配置 --------------------------
st.set_page_config(
    page_title="比特币已实现波动率分析",
    page_icon="📈",
    layout="wide"  # 宽屏展示，适配多子图
)

# -------------------------- 全局绘图样式配置 --------------------------
plt.rcParams['font.size'] = 12
plt.style.use('seaborn-v0_8')
# 解决matplotlib中文显示问题（部署后也生效）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 页面主标题与说明 --------------------------
st.title("📈 比特币价格数据清洗与已实现波动率(Realized Volatility)分析")
st.divider()
st.info("本应用实现比特币行情数据清洗、对数收益率计算、已实现波动率建模及平稳性检验，数据来源：data.csv", icon="ℹ️")

# -------------------------- 步骤1：加载并查看原始数据 --------------------------
def preprocess_data_locally():
    """本地预处理：将369MB CSV转为Parquet并优化数据类型（仅本地执行1次）"""
    # 本地读取全量CSV
    df = pd.read_csv('data.csv')
    # 1. 只保留核心列
    usecols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[usecols]
    # 2. 优化数据类型（内存减半）
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')  # 时间戳转datetime
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype('float32')  # float64→float32，精度足够
    # 3. 保存为Parquet（压缩比高，读取快）
    df.to_parquet('data.parquet', index=False, compression='snappy')
    print("✅ 本地预处理完成，生成data.parquet文件")

# -------------------------- 第二步：缓存加载压缩后的Parquet文件（Streamlit运行） --------------------------
@st.cache_data(persist="disk", ttl=None, show_spinner="正在加载全量数据（首次加载较慢，请稍等）...")
def load_full_data():
    """加载 data.parquet 文件（替代原CSV）"""
    # ========== 核心修改：读取parquet而非csv ==========
    try:
        # 本地测试：指定parquet文件路径（你的实际路径）
        # parquet_path = r"C:\Users\lenovo\Desktop\data\data.parquet"
        
        # Streamlit部署：相对路径（parquet和app.py同目录）
        parquet_path = "data.parquet"
        
        # 读取parquet文件（核心修改行）
        df = pd.read_parquet(parquet_path)
        
        # 释放内存
        gc.collect()
        return df
    except FileNotFoundError:
        raise FileNotFoundError("未找到data.parquet文件！请确认文件和app.py同目录")
    except MemoryError:
        raise MemoryError("内存不足！Parquet已优化，若仍报错请检查服务器配置")


# -------------------------- 步骤2：数据预处理与清洗 --------------------------
st.header("2. 数据预处理与清洗")
st.subheader("时间戳转换与索引设置")
# 时间戳转换（秒级）
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Timestamp', inplace=True)
# 确保索引为DatetimeIndex
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)
st.success("✅ 时间戳转换完成，已将Timestamp设为行索引", icon="✔️")

st.subheader("异常值与无效数据清洗")
original_len = len(df)
# 删除核心价格列空值
df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
# 修正High/Low逻辑矛盾
df['High'] = np.maximum(df['High'], df['Low'])
df['Low'] = np.minimum(df['Low'], df['High'])
# 限制Close在[Low, High]范围内
df['Close'] = np.clip(df['Close'], df['Low'], df['High'])

# 过滤常量价格+零成交量的无效段
price_change = df['Close'].diff().abs()
constant_mask = (price_change < 1e-6) & (df['Volume'] == 0)
is_constant_error = constant_mask.rolling(window=10, min_periods=1).sum() >= 10

# 最终有效数据过滤条件
valid_mask = (
    (df['Close'] >= 10) & 
    (df['Close'] <= 100000) & 
    (~is_constant_error) & 
    (df['Volume'] >= 0)
)
df_clean = df[valid_mask].copy()

# 展示清洗结果
clean_rate = (1 - len(df_clean)/original_len)*100
st.metric(
    label="数据清洗结果",
    value=f"剩余有效数据 {len(df_clean)} 行",
    delta=f"剔除无效数据 {clean_rate:.1f}%"
)

# -------------------------- 步骤3：对数收益率计算与特征分析 --------------------------
st.header("3. 对数收益率计算与微观结构特征分析")
# 计算对数收益率
returns = np.log(df_clean['Close'] / df_clean['Close'].shift(1))
# 滚动标准差（60窗口）
rolling_std = returns.rolling(window=60).std()
# 极端跳变点检测
extreme_jump = abs(returns) > (5 * rolling_std)
# 买卖价差反弹（微观结构噪声）
bid_ask_bounce = (returns * returns.shift(1)) < 0

# 展示检测结果
col1, col2 = st.columns(2)
with col1:
    st.metric(label="极端价格跳变点数量", value=extreme_jump.sum())
with col2:
    st.metric(label="微观结构噪声点数量", value=bid_ask_bounce.sum())

# 对数收益率分布与近期走势绘图
st.subheader("对数收益率分布与近期走势（最后1000个数据）")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
# 收益率分布直方图
sns.histplot(returns.dropna(), bins=100, kde=True, ax=ax1)
ax1.set_title('Log Returns Distribution', fontsize=14)
ax1.set_xlabel('Log Returns')
ax1.set_ylabel('Frequency')
# 近期收益率走势
ax2.plot(returns.index[-1000:], returns.values[-1000:], linewidth=0.5, color='#2E86AB')
ax2.set_title('Recent Log Returns (Last 1000)', fontsize=14)
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('Log Returns')
plt.tight_layout()
st.pyplot(fig)  # Streamlit展示绘图
st.divider()

# -------------------------- 步骤4：重采样与已实现波动率计算 --------------------------
st.header("4. 数据重采样与已实现波动率(RV)计算")
# 1分钟重采样（可自行修改sampling_freq，如'5T'=5分钟，'1H'=1小时）
sampling_freq = '1T'
st.write(f"当前重采样频率：**{sampling_freq}** (1分钟)",)
# 重采样取最后一个值，删除空值
df_resampled = df_clean.resample(sampling_freq).last().dropna()
# 计算重采样后的对数收益率
resampled_returns = np.log(df_resampled['Close'] / df_resampled['Close'].shift(1)).dropna()

# 计算日度已实现波动率（收益率平方和）
rv_daily = resampled_returns.groupby(resampled_returns.index.date).apply(lambda x: np.sum(x**2))
rv_daily.index = pd.to_datetime(rv_daily.index) 

# 年化因子计算（252个交易日，1440分钟/天）
minutes_per_sample = int(sampling_freq[:-1]) if sampling_freq[:-1].isdigit() else 1
annualization_factor = 252 * (1440 // minutes_per_sample)
# 年化已实现波动率（开方）
rv_annualized = np.sqrt(rv_daily * annualization_factor)

st.metric(label="日度已实现波动率观测数量", value=len(rv_daily))
st.success(f"✅ 年化因子计算完成：{annualization_factor}（基于{minutes_per_sample}分钟重采样）", icon="✔️")

# -------------------------- 步骤5：已实现波动率平稳性检验 --------------------------
st.header("5. 已实现波动率平稳性检验（ADF检验）")
# ADF单位根检验
adf_result = adfuller(rv_daily.dropna())
# 展示ADF检验结果
adf_data = {
    "ADF统计量": f"{adf_result[0]:.4f}",
    "p值": f"{adf_result[1]:.20f}",
    "1%临界值": f"{adf_result[4]['1%']:.4f}",
    "5%临界值": f"{adf_result[4]['5%']:.4f}",
    "10%临界值": f"{adf_result[4]['10%']:.4f}",
    "检验结果": "平稳（无单位根）" if adf_result[1] < 0.05 else "非平稳（存在单位根）"
}
st.dataframe(pd.DataFrame(adf_data, index=['结果']), use_container_width=True)

# ACF/PACF绘图（检验长记忆性）
st.subheader("ACF/PACF图（滞后30期）- 检验长记忆性")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(rv_daily.dropna(), lags=30, ax=ax1, color='#2E86AB')
plot_pacf(rv_daily.dropna(), lags=30, ax=ax2, color='#E63946')
ax1.set_title('ACF Plot (Long Memory Check)', fontsize=14)
ax2.set_title('PACF Plot', fontsize=14)
plt.tight_layout()
st.pyplot(fig)
st.divider()

# -------------------------- 步骤6：年化已实现波动率走势展示 --------------------------
st.header("6. 比特币年化已实现波动率走势")
fig, ax = plt.subplots(1, 1, figsize=(18, 6))
ax.plot(rv_annualized, color='#2E86AB', linewidth=1.2, alpha=0.9)
ax.set_title('Bitcoin Annualized Realized Volatility', fontsize=18, pad=20)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Annualized Volatility', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# 页面底部说明
st.divider()

st.caption("📊 本应用基于Streamlit开发 | 数据清洗+波动率分析全流程 | 2026")





