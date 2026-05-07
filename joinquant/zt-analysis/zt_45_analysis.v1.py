# -*- coding: utf-8 -*-
"""
zt_45_analysis.v1.py - 涨停板后45日数据分析程序 (v1: 修复缺失因子/IC/VIF/流水线问题 + 股票评分系统)
运行环境: JoinQuant Research Notebook (https://www.joinquant.com/research)

功能概述:
1. 构建近45交易日涨停股票池（排除ST/上市不足45天/一字板）
2. 多维度因子计算（90+因子，含涨停特征因子和Alpha因子）
3. IC衰减监控 & 因子共线性检测（基础设施优先）
4. T+N (N=1~5) 涨跌统计（编码: 0=下跌, 1=上涨<9.9%, 2=上涨≥9.9%）
5. 因子-收益分桶相关性矩阵
6. 股票池评分 — 基于因子-收益相关性打分，强势/弱势/淘汰分类，Top5交易信号
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import datetime as _dt
import time as _time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# JoinQuant 环境导入（在 research notebook 中已预装）
# ============================================================
try:
    from jqdata import *
    JQ_AVAILABLE = True
    print('[INFO] JoinQuant 环境已加载')
except ImportError:
    JQ_AVAILABLE = False
    print('[WARNING] JoinQuant 环境未检测到，部分功能将不可用')

# display() 兼容：Jupyter / IPython / 纯终端
try:
    from IPython.display import display as _ip_display
    _display = _ip_display
except ImportError:
    def _display(obj, **kwargs):
        """纯终端 fallback：直接 print"""
        print(obj)

# ============================================================
# PART 0: 配置常量
# ============================================================

LOOKBACK_DAYS = 15          # 回溯交易日数
TRACK_DAYS = 5              # T+N 追踪天数 (N=1~5)
HISTORY_DAYS = 120          # 因子计算所需历史天数（MA60等需要足够前置数据）
BATCH_SIZE = 300            # API批量查询大小
USE_MINUTE_DATA = False     # 分钟数据（封板速度等）— 默认关闭以加速，可设True开启
SKIP_SLOW_FACTORS = False   # 跳过慢速因子（资金流get_money_flow）— 设True可大幅加速
COLLINEARITY_THRESHOLD = 0.7  # 共线性相关系数阈值
VIF_THRESHOLD = 5.0         # VIF阈值
MIN_IC_SAMPLES = 20         # IC计算最小样本数
MISSING_THRESHOLD = 0.90    # 因子缺失率超过此值则跳过该因子

# T+5 收益分桶定义 (Section 4.1)
RETURN_BUCKETS = {
    'strong_rise':   (0.05, np.inf),     # >5%
    'moderate_rise': (0.0, 0.05),        # 0%~5%
    'moderate_fall': (-0.05, 0.0),       # -5%~0%
    'strong_fall':   (-np.inf, -0.05),   # <-5%
}

# 因子分类定义 — 涨停板专属因子最高优先级
FACTOR_CATEGORIES = {
    # ---- 涨停板专属因子（最高优先级） ----
    'limit_hit': [
        'seal_speed', 'open_count', 'limit_type', 'seal_time',
        'limit_duration', 'continuous_limit_count', 'first_limit_flag',
        'zt_volume_ratio', 'zt_turnover_rate', 'seal_amount_ratio',
        'zt_strength', 'limit_up_timeband',
        'zt_momentum', 'zt_gap_type', 'zt_volume_surge', 'zt_seal_strength_approx',
    ],
    # ---- 行情因子 ----
    'price_quote': [
        'pct_change', 'amplitude', 'upper_shadow', 'lower_shadow',
        'open_pct', 'body_ratio', 'open_close_ratio',
        'high_low_ratio', 'gap_pct',
        'close_to_high', 'close_to_low', 'intraday_range',
    ],
    # ---- 成交量因子 ----
    'volume': [
        'volume', 'money', 'vwap', 'volume_ma5_ratio', 'volume_ma10_ratio',
        'volume_ma20_ratio', 'money_ma5_ratio', 'volume_std5_ratio',
        'volume_skew', 'obv_slope',
    ],
    # ---- 行业因子 ----
    'industry': [
        'industry_zt_count', 'industry_zt_ratio', 'industry_zt_rank',
    ],
    # ---- 趋势因子 ----
    'trend': [
        'ma5_dist', 'ma10_dist', 'ma20_dist', 'ma60_dist',
        'above_ma5', 'above_ma10', 'above_ma20', 'above_ma60',
        'macd', 'macd_signal', 'macd_hist',
        'rsi_6', 'rsi_12', 'rsi_24',
        'kdj_k', 'kdj_d', 'kdj_j',
        'boll_position', 'boll_width',
        'ma5_slope', 'ma10_slope', 'ma20_slope', 'trend_strength',
    ],
    # ---- 换手率因子 ----
    'turnover': [
        'turnover_rate', 'turnover_ma5_ratio', 'turnover_ma10_ratio',
        'turnover_std5',
    ],
    # ---- 资金流因子 ----
    'capital_flow': [
        'net_mf_amount', 'net_mf_vol', 'net_mf_ratio',
    ],
    # ---- 估值因子 ----
    'valuation': [
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio',
        'market_cap', 'circulating_market_cap',
    ],
    # ---- 规模因子 ----
    'scale': [
        'market_cap_log', 'circulating_cap_log',
    ],
    # ---- 基本面因子 ----
    'fundamental': [
        'roe', 'roa', 'net_profit_margin', 'gross_profit_margin',
        'revenue_growth', 'profit_growth', 'debt_asset_ratio',
    ],
    # ---- 前日因子 ----
    'prev_day': [
        'prev_pct_change', 'prev_volume_ratio', 'prev_amplitude',
        'prev_upper_shadow', 'prev_lower_shadow', 'prev_turnover',
    ],
    # ---- 集合竞价因子 ----
    'auction': [
        'auction_pct', 'auction_volume_ratio',
    ],
    # ---- 波动率因子 ----
    'volatility': [
        'hist_vol_5', 'hist_vol_10', 'hist_vol_20', 'atr_14',
        'vol_ratio_5_20', 'max_drawdown_20',
    ],
    # ---- 动量因子 ----
    'momentum': [
        'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
    ],
    # ---- 技术形态因子 ----
    'technical_shape': [
        'doji_flag', 'hammer_flag', 'engulfing_flag',
    ],
    # ---- Alpha因子（量价背离优先 + 涨停特征Alpha） ----
    'alpha': [
        'alpha_6', 'alpha_12', 'alpha_33', 'alpha_54',
        'alpha_41', 'alpha_53', 'alpha_101',
        'alpha_zt_seal', 'alpha_zt_open', 'alpha_zt_type',
        'alpha_zt_vol', 'alpha_zt_turnover',
        'alpha_5', 'alpha_15', 'alpha_20',
    ],
}

ALL_FACTORS = []
for _f in FACTOR_CATEGORIES.values():
    ALL_FACTORS.extend(_f)

# ★ v1新增: 分钟数据依赖因子列表 — USE_MINUTE_DATA=False时自动排除
MINUTE_DEPENDENT_FACTORS = [
    'seal_speed', 'open_count', 'seal_time', 'limit_duration',
    'zt_strength', 'limit_up_timeband',
    'auction_pct', 'auction_volume_ratio',
]

# ★ v1.2新增: 因子中英文名称映射 — 输出报告时同时显示中英文
FACTOR_NAME_CN = {
    # ---- 涨停板专属因子 ----
    'seal_speed': '封板速度', 'open_count': '开板次数', 'limit_type': '涨停类型',
    'seal_time': '封板时长', 'limit_duration': '涨停持续时间', 'continuous_limit_count': '连板数',
    'first_limit_flag': '首板标志', 'zt_volume_ratio': '涨停量比', 'zt_turnover_rate': '涨停换手率',
    'seal_amount_ratio': '封单金额比', 'zt_strength': '涨停强度', 'limit_up_timeband': '涨停时段',
    'zt_momentum': '涨停动量', 'zt_gap_type': '缺口类型', 'zt_volume_surge': '涨停放量',
    'zt_seal_strength_approx': '封单强度近似',
    # ---- 行情因子 ----
    'pct_change': '涨跌幅', 'amplitude': '振幅', 'upper_shadow': '上影线',
    'lower_shadow': '下影线', 'open_pct': '开盘涨跌', 'body_ratio': '实体比',
    'open_close_ratio': '开收比', 'high_low_ratio': '高低比', 'gap_pct': '跳空比例',
    'close_to_high': '收盘/最高', 'close_to_low': '收盘/最低', 'intraday_range': '日内波幅',
    # ---- 成交量因子 ----
    'volume': '成交量', 'money': '成交额', 'vwap': 'VWAP', 'volume_ma5_ratio': '量/MA5',
    'volume_ma10_ratio': '量/MA10', 'volume_ma20_ratio': '量/MA20', 'money_ma5_ratio': '额/MA5',
    'volume_std5_ratio': '量标准差/MA5', 'volume_skew': '量偏度', 'obv_slope': 'OBV斜率',
    # ---- 行业因子 ----
    'industry_zt_count': '行业涨停数', 'industry_zt_ratio': '行业涨停占比', 'industry_zt_rank': '行业涨停排名',
    # ---- 趋势因子 ----
    'ma5_dist': 'MA5偏离', 'ma10_dist': 'MA10偏离', 'ma20_dist': 'MA20偏离', 'ma60_dist': 'MA60偏离',
    'above_ma5': 'MA5上方', 'above_ma10': 'MA10上方', 'above_ma20': 'MA20上方', 'above_ma60': 'MA60上方',
    'macd': 'MACD', 'macd_signal': 'MACD信号', 'macd_hist': 'MACD柱',
    'rsi_6': 'RSI6', 'rsi_12': 'RSI12', 'rsi_24': 'RSI24',
    'kdj_k': 'KDJ_K', 'kdj_d': 'KDJ_D', 'kdj_j': 'KDJ_J',
    'boll_position': '布林位置', 'boll_width': '布林宽度',
    'ma5_slope': 'MA5斜率', 'ma10_slope': 'MA10斜率', 'ma20_slope': 'MA20斜率', 'trend_strength': '趋势强度',
    # ---- 换手率因子 ----
    'turnover_rate': '换手率', 'turnover_ma5_ratio': '换手/MA5', 'turnover_ma10_ratio': '换手/MA10',
    'turnover_std5': '换手标准差5',
    # ---- 资金流因子 ----
    'net_mf_amount': '主力净流入额', 'net_mf_vol': '主力净流入量', 'net_mf_ratio': '主力净流入比',
    # ---- 估值因子 ----
    'pe_ratio': 'PE', 'pb_ratio': 'PB', 'ps_ratio': 'PS', 'pcf_ratio': 'PCF',
    'market_cap': '总市值', 'circulating_market_cap': '流通市值',
    # ---- 规模因子 ----
    'market_cap_log': '总市值对数', 'circulating_cap_log': '流通市值对数',
    # ---- 基本面因子 ----
    'roe': 'ROE', 'roa': 'ROA', 'net_profit_margin': '净利率', 'gross_profit_margin': '毛利率',
    'revenue_growth': '营收增速', 'profit_growth': '利润增速', 'debt_asset_ratio': '资产负债率',
    # ---- 前日因子 ----
    'prev_pct_change': '前日涨跌', 'prev_volume_ratio': '前日量比', 'prev_amplitude': '前日振幅',
    'prev_upper_shadow': '前日上影线', 'prev_lower_shadow': '前日下影线', 'prev_turnover': '前日换手',
    # ---- 集合竞价因子 ----
    'auction_pct': '竞价涨跌', 'auction_volume_ratio': '竞价量比',
    # ---- 波动率因子 ----
    'hist_vol_5': '历史波动5日', 'hist_vol_10': '历史波动10日', 'hist_vol_20': '历史波动20日',
    'atr_14': 'ATR14', 'vol_ratio_5_20': '波动比5/20', 'max_drawdown_20': '最大回撤20日',
    # ---- 动量因子 ----
    'ret_1d': '1日收益', 'ret_3d': '3日收益', 'ret_5d': '5日收益', 'ret_10d': '10日收益', 'ret_20d': '20日收益',
    # ---- 技术形态因子 ----
    'doji_flag': '十字星', 'hammer_flag': '锤子线', 'engulfing_flag': '吞没形态',
    # ---- Alpha因子 ----
    'alpha_6': 'Alpha6', 'alpha_12': 'Alpha12', 'alpha_33': 'Alpha33', 'alpha_54': 'Alpha54',
    'alpha_41': 'Alpha41', 'alpha_53': 'Alpha53', 'alpha_101': 'Alpha101',
    'alpha_zt_seal': 'Alpha涨停封单', 'alpha_zt_open': 'Alpha涨停开板', 'alpha_zt_type': 'Alpha涨停类型',
    'alpha_zt_vol': 'Alpha涨停量', 'alpha_zt_turnover': 'Alpha涨停换手',
    'alpha_5': 'Alpha5', 'alpha_15': 'Alpha15', 'alpha_20': 'Alpha20',
}

print("=" * 70)
print("涨停板后45日数据分析程序 zt_45_analysis.v1")
print("=" * 70)
print(f"回溯: {LOOKBACK_DAYS}日 | T+N: {TRACK_DAYS}天 | 因子: {len(ALL_FACTORS)}个")
print(f"分类: {list(FACTOR_CATEGORIES.keys())}")
print(f"分钟数据: {'开启' if USE_MINUTE_DATA else '关闭(加速)'}")
print()


# ============================================================
# PART 1: 工具函数
# ============================================================

def safe_divide(a, b, fillna=0.0):
    """安全除法，避免除零"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(b) > 1e-10, a / b, np.nan)
    if isinstance(result, np.ndarray):
        result = np.nan_to_num(result, nan=fillna)
    elif np.isnan(result):
        result = fillna
    return result


def encode_return(ret):
    """T+N涨跌编码: 0=下跌, 1=上涨<9.9%, 2=上涨≥9.9%"""
    if pd.isna(ret):
        return np.nan
    if ret < 0:
        return 0
    elif ret < 0.099:
        return 1
    else:
        return 2


def classify_return_bucket(ret):
    """将T+5收益率分入桶"""
    if pd.isna(ret):
        return np.nan
    for name, (lo, hi) in RETURN_BUCKETS.items():
        if lo <= ret < hi:
            return name
    return 'other'


def compute_high_limit(stock_code, pre_close):
    """根据股票代码和前收盘价计算涨停价"""
    if pd.isna(pre_close) or pre_close <= 0:
        return np.nan
    code_num = stock_code[:6]
    # 科创板 688xxx / 创业板 300xxx,301xxx: 20%涨跌幅
    if code_num.startswith('688') or code_num.startswith('300') or code_num.startswith('301'):
        return round(pre_close * 1.2, 2)
    # 北交所 8xxxxx / 4xxxxx: 30%
    elif code_num.startswith('8') or code_num.startswith('4'):
        return round(pre_close * 1.3, 2)
    # 主板: 10%
    else:
        return round(pre_close * 1.1, 2)


def compute_macd(close_s, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    if len(close_s) < slow + signal:
        return np.nan, np.nan, np.nan
    ema_f = close_s.ewm(span=fast, adjust=False).mean()
    ema_s = close_s.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)
    return dif.iloc[-1], dea.iloc[-1], hist.iloc[-1]


def compute_rsi(close_s, window):
    """计算RSI指标"""
    if len(close_s) < window + 1:
        return np.nan
    delta = close_s.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else np.nan


def compute_kdj(high_s, low_s, close_s, n=9, m1=3, m2=3):
    """计算KDJ指标"""
    if len(high_s) < n:
        return np.nan, np.nan, np.nan
    low_n = low_s.rolling(window=n, min_periods=n).min()
    high_n = high_s.rolling(window=n, min_periods=n).max()
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = (close_s - low_n) / denom * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k.iloc[-1], d.iloc[-1], j.iloc[-1]


def compute_boll(close_s, window=20, num_std=2):
    """计算布林带位置和宽度"""
    if len(close_s) < window:
        return np.nan, np.nan
    ma = close_s.rolling(window=window).mean()
    std = close_s.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    c = close_s.iloc[-1]
    u = upper.iloc[-1]
    l = lower.iloc[-1]
    m = ma.iloc[-1]
    width = (u - l) / m if m > 0 else np.nan
    pos = (c - l) / (u - l) if (u - l) > 0 else 0.5
    return pos, width


def _to_timestamp(d):
    """统一转换为 pd.Timestamp，避免 numpy.datetime64 / Timestamp 哈希不匹配"""
    if d is None:
        return None
    return pd.Timestamp(d)


# ============================================================
# PART 2: 涨停股票池构建
# ============================================================

def get_zt_stocks_on_date(date):
    """
    获取指定日期的涨停股票（批量方式）
    返回: list of stock codes

    策略: 使用fq=None获取未复权数据，用涨幅百分比阈值判断涨停
    不依赖high_limit字段，也不依赖前复权pre_close
    """
    all_stocks = get_all_securities(types=['stock'], date=date)
    stock_list = all_stocks.index.tolist()

    zt_stocks = []
    _batch_size = 80
    _error_count = 0

    for i in range(0, len(stock_list), _batch_size):
        batch = stock_list[i:i + _batch_size]
        try:
            # 关键: fq=None 获取未复权数据，pre_close才是真实前收盘价
            # panel=False: 返回DataFrame而非Panel（JQ Python3.6默认返回Panel）
            # skip_paused=False: 批量查询时JQ要求对齐日期，不能跳过停牌
            # fill_paused=False: 停牌股数据为NaN而非前值填充
            price_df = get_price(batch, end_date=date, count=1,
                                 frequency='daily',
                                 fields=['open', 'close', 'high', 'low',
                                         'volume', 'money', 'pre_close'],
                                 skip_paused=False, fq=None,
                                 panel=False, fill_paused=False)
            if price_df is None or price_df.empty:
                continue

            # panel=False 返回的DataFrame自带 'code' 列
            # 兼容处理: 确保有code列
            code_col = None
            for col in ('code', 'security', 'symbol'):
                if col in price_df.columns:
                    code_col = col
                    break
            if code_col is None:
                # 尝试从索引重置
                if isinstance(price_df.index, pd.MultiIndex):
                    price_df = price_df.reset_index()
                    for col in price_df.columns:
                        if col in ('code', 'security', 'symbol'):
                            code_col = col
                            break
            if code_col is None:
                continue

            # 逐行检查涨停 — 用涨幅百分比判断（比涨停价计算更稳健）
            for _, row in price_df.iterrows():
                code = str(row[code_col])
                close_val = row.get('close', np.nan)
                pre_close = row.get('pre_close', np.nan)

                # 跳过停牌/无效数据（fill_paused=False时停牌股为NaN）
                if pd.isna(close_val) or pd.isna(pre_close) or pre_close <= 0:
                    continue

                pct = close_val / pre_close - 1
                code_num = code[:6]
                if code_num.startswith('688') or code_num.startswith('300') or code_num.startswith('301'):
                    threshold = 0.195   # 20%板 (创业板/科创板)
                elif code_num.startswith('8') or code_num.startswith('4'):
                    threshold = 0.295   # 30%板 (北交所)
                else:
                    threshold = 0.095   # 10%板 (主板)

                if pct >= threshold:
                    zt_stocks.append(code)

        except Exception as e:
            _error_count += 1
            if _error_count <= 2:
                print(f"  [WARNING] get_zt_stocks_on_date 批次错误: {e}")

    if _error_count > 0:
        print(f"  [WARNING] get_zt_stocks_on_date 共 {_error_count} 个批次出错")

    return zt_stocks


def _check_and_add_zt(code, row, zt_list):
    """检查是否涨停并添加到列表 — 用涨幅百分比判断"""
    close_val = row.get('close', np.nan)
    pre_close = row.get('pre_close', np.nan)

    if pd.isna(close_val) or pd.isna(pre_close) or pre_close <= 0:
        return

    # 用涨幅百分比判断涨停（比涨停价计算更稳健，不受复权影响）
    pct = close_val / pre_close - 1
    code_num = code[:6]
    if code_num.startswith('688') or code_num.startswith('300') or code_num.startswith('301'):
        threshold = 0.195   # 20%板
    elif code_num.startswith('8') or code_num.startswith('4'):
        threshold = 0.295   # 30%板
    else:
        threshold = 0.095   # 10%板

    if pct >= threshold:
        zt_list.append(code)


def filter_zt_pool(zt_list, date):
    """
    过滤涨停股票池:
    1. 排除ST
    2. 排除上市不足45天
    3. 排除一字板
    """
    if not zt_list:
        return []

    # 1. 排除ST
    st_stocks = set()
    try:
        st_df = get_extras('is_st', zt_list, start_date=date, end_date=date)
        if st_df is not None and not st_df.empty:
            # get_extras('is_st') 返回: 行=日期, 列=股票代码, 值=True/False
            if isinstance(st_df, pd.DataFrame) and st_df.shape[0] > 0:
                st_stocks = set(st_df.columns[st_df.iloc[0] == True].tolist())
    except:
        pass

    # 2. 排除上市不足45天
    young_stocks = set()
    try:
        all_sec = get_all_securities(types=['stock'], date=date)
        for stock in zt_list:
            if stock in all_sec.index:
                start_d = all_sec.loc[stock, 'start_date']
                if isinstance(start_d, str):
                    start_d = pd.Timestamp(start_d)
                list_days = len(get_trade_days(start_date=start_d, end_date=date))
                if list_days < 45:
                    young_stocks.add(stock)
    except:
        pass

    # 3. 排除一字板
    yiziban_stocks = set()
    try:
        price_df = get_price(zt_list, end_date=date, count=1,
                             frequency='daily',
                             fields=['open', 'close', 'high', 'low'],
                             skip_paused=False, fq=None,
                             panel=False, fill_paused=False)
        if price_df is not None and not price_df.empty:
            # panel=False 返回的DataFrame自带 'code' 列
            code_col = None
            for col in ('code', 'security', 'symbol'):
                if col in price_df.columns:
                    code_col = col
                    break
            if code_col is None:
                if isinstance(price_df.index, pd.MultiIndex):
                    price_df = price_df.reset_index()
                    for col in price_df.columns:
                        if col in ('code', 'security', 'symbol'):
                            code_col = col
                            break

            if code_col is not None:
                for _, row in price_df.iterrows():
                    if _is_yiziban(row):
                        yiziban_stocks.add(str(row[code_col]))
    except:
        pass

    exclude = st_stocks | young_stocks | yiziban_stocks
    filtered = [s for s in zt_list if s not in exclude]

    return filtered


def _is_yiziban(row):
    """判断是否一字板: open==close==high==low"""
    o, c, h, l = row.get('open', 0), row.get('close', 0), row.get('high', 0), row.get('low', 0)
    if pd.isna(o) or pd.isna(c):
        return False
    return abs(o - c) < 0.01 and abs(c - h) < 0.01 and abs(h - l) < 0.01


# ============================================================
# PART 3: 股票池管理器
# ============================================================

class StockPoolManager:
    """涨停股票池管理器 - 5天过期，重新涨停可续期"""

    def __init__(self):
        self.pool = {}          # {code: {'entry_date':, 'remaining':, 'limit_count':}}
        self.all_entries = []   # 所有入池记录
        self.daily_log = []     # 每日日志

    def add_stocks(self, stock_list, date):
        for stock in stock_list:
            if stock in self.pool:
                self.pool[stock]['remaining'] = TRACK_DAYS
                self.pool[stock]['limit_count'] += 1
            else:
                self.pool[stock] = {
                    'entry_date': date,
                    'remaining': TRACK_DAYS,
                    'limit_count': 1,
                }
            self.all_entries.append({
                'stock': stock, 'date': date,
                'limit_count': self.pool[stock]['limit_count'],
            })

    def tick(self, date):
        """每日更新：减少剩余天数，移除过期股票"""
        expired = []
        for stock in self.pool:
            self.pool[stock]['remaining'] -= 1
            if self.pool[stock]['remaining'] <= 0:
                expired.append(stock)
        for stock in expired:
            del self.pool[stock]
        self.daily_log.append({
            'date': date,
            'pool_size': len(self.pool),
        })


# ============================================================
# PART 4: 因子计算引擎
# ============================================================

class FactorEngine:
    """因子计算引擎 - 批量+缓存优化"""

    def __init__(self):
        self.factor_records = []   # 所有因子记录
        self._price_cache = {}     # 历史价格缓存
        self._fund_cache = {}      # 基本面缓存
        self._industry_cache = {}  # 行业缓存（跨日复用）
        self._industry_zt_count_cache = {}  # 每日行业涨停数缓存 {date: {ind_code: count}}
        self._day_price_cache = {} # 每日批量价格缓存 {date: {stock: price_df}}

    # ---------- 缓存层 ----------

    def _get_hist_price(self, stock, end_date, days=HISTORY_DAYS):
        key = f"{stock}_{end_date}"
        if key not in self._price_cache:
            # 优先从每日批量缓存获取
            day_key = str(end_date)
            if day_key in self._day_price_cache and stock in self._day_price_cache[day_key]:
                self._price_cache[key] = self._day_price_cache[day_key][stock]
            else:
                try:
                    start = pd.Timestamp(end_date) - pd.Timedelta(days=int(days * 1.8))
                    df = get_price(stock, start_date=start, end_date=end_date,
                                   frequency='daily',
                                   fields=['open', 'close', 'high', 'low',
                                           'volume', 'money', 'pre_close'],
                                   skip_paused=True, fq='pre')
                    self._price_cache[key] = df if df is not None else pd.DataFrame()
                except:
                    self._price_cache[key] = pd.DataFrame()
        return self._price_cache[key]

    def preload_day_prices(self, stock_list, date):
        """批量预加载当日所有涨停股的历史价格 — 减少逐股API调用"""
        day_key = str(date)
        if day_key not in self._day_price_cache:
            self._day_price_cache[day_key] = {}

        uncached = [s for s in stock_list
                    if f"{s}_{date}" not in self._price_cache
                    and s not in self._day_price_cache[day_key]]
        if not uncached:
            return

        # 批量获取: 使用 get_price panel=False 一次取多只股票
        try:
            start = pd.Timestamp(date) - pd.Timedelta(days=int(HISTORY_DAYS * 1.8))
            batch = uncached[:BATCH_SIZE]
            df = get_price(batch, start_date=start, end_date=date,
                           frequency='daily',
                           fields=['open', 'close', 'high', 'low',
                                   'volume', 'money', 'pre_close'],
                           skip_paused=True, fq='pre', panel=False)
            if df is not None and not df.empty:
                for code in batch:
                    sub = df[df['code'] == code] if 'code' in df.columns else pd.DataFrame()
                    if sub.empty:
                        sub = pd.DataFrame()
                    self._day_price_cache[day_key][code] = sub
                    self._price_cache[f"{code}_{date}"] = sub
        except:
            pass

        # 未批量获取到的，逐个获取
        for s in uncached:
            key = f"{s}_{date}"
            if key not in self._price_cache:
                try:
                    start = pd.Timestamp(date) - pd.Timedelta(days=int(HISTORY_DAYS * 1.8))
                    df = get_price(s, start_date=start, end_date=date,
                                   frequency='daily',
                                   fields=['open', 'close', 'high', 'low',
                                           'volume', 'money', 'pre_close'],
                                   skip_paused=True, fq='pre')
                    self._price_cache[key] = df if df is not None else pd.DataFrame()
                    self._day_price_cache[day_key][s] = self._price_cache[key]
                except:
                    self._price_cache[key] = pd.DataFrame()
                    self._day_price_cache[day_key][s] = pd.DataFrame()

    def precompute_day_industry_zt(self, zt_stocks_on_date, date):
        """预计算当日各行业涨停数 — 消除O(N²)循环"""
        day_key = str(date)
        if day_key in self._industry_zt_count_cache:
            return self._industry_zt_count_cache[day_key]

        ind_count = {}
        for s in zt_stocks_on_date:
            ind = self._get_industry(s, date)
            ind_count[ind] = ind_count.get(ind, 0) + 1

        self._industry_zt_count_cache[day_key] = ind_count
        return ind_count

    def _get_fundamentals_batch(self, stock_list, date):
        """批量获取基本面数据 — 修正JQ字段名"""
        key = str(date)
        if key not in self._fund_cache:
            self._fund_cache[key] = {}
        if not stock_list:
            return self._fund_cache[key]

        # 查询未缓存的股票
        uncached = [s for s in stock_list if s not in self._fund_cache[key]]
        if uncached:
            # ★ v1.1: 拆分为两次查询 — 避免单个错误字段名导致全部基本面数据丢失
            # 查询1: valuation表 + indicator核心字段(已验证可用)
            try:
                q1 = query(
                    valuation.code,
                    valuation.pe_ratio, valuation.pb_ratio,
                    valuation.ps_ratio, valuation.pcf_ratio,
                    valuation.market_cap, valuation.circulating_market_cap,
                    valuation.capitalization, valuation.circulating_cap,
                    valuation.turnover_ratio,
                    indicator.roe, indicator.roa,
                    indicator.net_profit_margin, indicator.gross_profit_margin,
                ).filter(valuation.code.in_(uncached))
                df1 = get_fundamentals(q1, date=date)
                if df1 is not None and not df1.empty:
                    for _, row in df1.iterrows():
                        self._fund_cache[key][row['code']] = row.to_dict()
                    if len(self._fund_cache[key]) <= len(uncached):  # 仅首次时打印
                        sample_code = uncached[0] if uncached else ''
                        sample = self._fund_cache[key].get(sample_code, {})
                        filled = sum(1 for v in sample.values() if not pd.isna(v))
                        #print(f"    [基本面] 查询1成功: {len(df1)}只股票, 样本{sample_code}: {filled}个非空字段")
                else:
                    print(f"    [基本面⚠] 查询1返回空, 日期={date}, 股票数={len(uncached)}")
            except Exception as e:
                print(f"    [基本面✗] 查询1异常(valuation+核心indicator): {e}")

            # 查询2: indicator增长字段 — 正确字段名为 inc_revenue_year_on_year / inc_net_profit_year_on_year
            try:
                q2 = query(
                    valuation.code,
                    indicator.inc_revenue_year_on_year,
                    indicator.inc_net_profit_year_on_year,
                ).filter(valuation.code.in_(uncached))
                df2 = get_fundamentals(q2, date=date)
                if df2 is not None and not df2.empty:
                    for _, row in df2.iterrows():
                        code = row['code']
                        if code not in self._fund_cache[key]:
                            self._fund_cache[key][code] = {}
                        self._fund_cache[key][code]['inc_revenue_year_on_year'] = row.get('inc_revenue_year_on_year', np.nan)
                        self._fund_cache[key][code]['inc_net_profit_year_on_year'] = row.get('inc_net_profit_year_on_year', np.nan)
                    #print(f"    [基本面] 查询2成功: {len(df2)}只股票增长字段已获取")
                else:
                    print(f"    [基本面⚠] 查询2返回空(增长字段), 日期={date}")
            except Exception as e:
                print(f"    [基本面✗] 查询2异常(增长字段): {e}")

            # 查询3: 资产负债率 — 尝试多种JQ字段名
            # 方案A: indicator.debt_to_asset_ratio (JQ可能的正确字段名)
            # 方案B: balance表计算 total_liability / total_assets
            debt_ratio_ok = False
            try:
                q3a = query(
                    valuation.code,
                    indicator.debt_to_asset_ratio,
                ).filter(valuation.code.in_(uncached))
                df3a = get_fundamentals(q3a, date=date)
                if df3a is not None and not df3a.empty:
                    filled = 0
                    for _, row in df3a.iterrows():
                        code = row['code']
                        if code not in self._fund_cache[key]:
                            self._fund_cache[key][code] = {}
                        val = row.get('debt_to_asset_ratio', np.nan)
                        if not pd.isna(val):
                            self._fund_cache[key][code]['debt_asset_ratio'] = float(val)
                            filled += 1
                        else:
                            self._fund_cache[key][code]['debt_asset_ratio'] = np.nan
                    if filled > 0:
                        debt_ratio_ok = True
                        # 仅首次打印
                        if len(self._fund_cache[key]) <= len(uncached) + 5:
                            print(f"    [基本面] 查询3a成功(indicator.debt_to_asset_ratio): {filled}/{len(df3a)}只")
            except Exception as e:
                pass  # 字段名可能不存在，静默跳过

            if not debt_ratio_ok:
                try:
                    q3b = query(
                        valuation.code,
                        balance.total_liability,
                        balance.total_assets,
                    ).filter(valuation.code.in_(uncached))
                    df3b = get_fundamentals(q3b, date=date)
                    if df3b is not None and not df3b.empty:
                        filled = 0
                        for _, row in df3b.iterrows():
                            code = row['code']
                            if code not in self._fund_cache[key]:
                                self._fund_cache[key][code] = {}
                            tl = row.get('total_liability', np.nan)
                            ta = row.get('total_assets', np.nan)
                            if not pd.isna(tl) and not pd.isna(ta) and ta != 0:
                                self._fund_cache[key][code]['debt_asset_ratio'] = float(tl) / float(ta)
                                filled += 1
                            else:
                                self._fund_cache[key][code]['debt_asset_ratio'] = np.nan
                        if filled > 0:
                            debt_ratio_ok = True
                            if len(self._fund_cache[key]) <= len(uncached) + 5:
                                print(f"    [基本面] 查询3b成功(balance表计算): {filled}/{len(df3b)}只")
                    # else: balance数据可能季度更新，某些日期无数据属正常
                except Exception as e:
                    if len(self._fund_cache[key]) <= len(uncached) + 5:
                        print(f"    [基本面⚠] 查询3异常(资产负债率): {e}")

            # 未查到的标记为空
            for s in uncached:
                if s not in self._fund_cache[key]:
                    self._fund_cache[key][s] = {}
        return self._fund_cache[key]

    def _get_industry(self, stock, date):
        """获取行业代码 — 带缓存"""
        if stock not in self._industry_cache:
            try:
                ind = get_industry(stock, date)
                if isinstance(ind, dict):
                    self._industry_cache[stock] = ind.get(stock, {}).get('行业代码', 'Unknown')
                else:
                    self._industry_cache[stock] = 'Unknown'
            except:
                self._industry_cache[stock] = 'Unknown'
        return self._industry_cache[stock]

    # ---------- 行情因子 ----------

    def _price_quote_factors(self, price_data):
        f = {}
        if price_data is None or len(price_data) < 1:
            return {k: np.nan for k in FACTOR_CATEGORIES['price_quote']}
        row = price_data.iloc[-1]
        pc = row.get('pre_close', np.nan)
        if pd.isna(pc) and len(price_data) > 1:
            pc = price_data.iloc[-2]['close']
        if pd.isna(pc) or pc <= 0:
            return {k: np.nan for k in FACTOR_CATEGORIES['price_quote']}

        f['pct_change'] = (row['close'] - pc) / pc
        f['amplitude'] = (row['high'] - row['low']) / pc
        f['upper_shadow'] = (row['high'] - max(row['open'], row['close'])) / pc
        f['lower_shadow'] = (min(row['open'], row['close']) - row['low']) / pc
        f['open_pct'] = (row['open'] - pc) / pc
        hl = row['high'] - row['low']
        f['body_ratio'] = abs(row['close'] - row['open']) / hl if hl > 0 else 0
        f['open_close_ratio'] = (row['open'] - row['close']) / pc
        f['high_low_ratio'] = row['high'] / row['low'] if row['low'] > 0 else np.nan
        # 跳空幅度: (今日开盘 - 昨日收盘) / 昨日收盘
        f['gap_pct'] = (row['open'] - pc) / pc
        # 新增行情因子
        f['close_to_high'] = (row['high'] - row['close']) / row['high'] if row['high'] > 0 else np.nan
        f['close_to_low'] = (row['close'] - row['low']) / row['low'] if row['low'] > 0 else np.nan
        f['intraday_range'] = hl / pc if pc > 0 else np.nan
        return f

    # ---------- 量价因子 ----------

    def _volume_factors(self, price_data):
        f = {}
        if price_data is None or len(price_data) < 2:
            return {k: np.nan for k in FACTOR_CATEGORIES['volume']}
        row = price_data.iloc[-1]
        f['volume'] = row['volume']
        f['money'] = row['money']
        f['vwap'] = row['money'] / row['volume'] if row['volume'] > 0 else row['close']

        for w, key in [(5, 'volume_ma5_ratio'), (10, 'volume_ma10_ratio'),
                        (20, 'volume_ma20_ratio')]:
            if len(price_data) > w:
                ma = price_data['volume'].iloc[-(w + 1):-1].mean()
                f[key] = row['volume'] / ma if ma > 0 else 1.0
            else:
                f[key] = np.nan

        if len(price_data) > 5:
            ma5 = price_data['money'].iloc[-6:-1].mean()
            f['money_ma5_ratio'] = row['money'] / ma5 if ma5 > 0 else 1.0
        else:
            f['money_ma5_ratio'] = np.nan

        # 5日量标准差比
        if len(price_data) > 6:
            std5 = price_data['volume'].iloc[-6:-1].std()
            f['volume_std5_ratio'] = row['volume'] / std5 if std5 > 0 else np.nan
        else:
            f['volume_std5_ratio'] = np.nan

        # 新增: 成交量偏度 (近20日)
        if len(price_data) > 20:
            vol20 = price_data['volume'].iloc[-20:]
            f['volume_skew'] = vol20.skew()
        else:
            f['volume_skew'] = np.nan

        # 新增: OBV斜率 (近10日)
        if len(price_data) > 10:
            close_s = price_data['close'].iloc[-10:]
            vol_s = price_data['volume'].iloc[-10:]
            direction = np.sign(close_s.diff().fillna(0))
            obv = (direction * vol_s).cumsum()
            if len(obv) >= 3:
                x = np.arange(len(obv))
                try:
                    slope = np.polyfit(x, obv.values, 1)[0]
                    f['obv_slope'] = slope / (obv.iloc[-1] + 1e-10)  # 归一化
                except:
                    f['obv_slope'] = np.nan
            else:
                f['obv_slope'] = np.nan
        else:
            f['obv_slope'] = np.nan

        return f

    # ---------- 行业因子（优化: 缓存行业查询） ----------

    def _industry_factors(self, stock, date, zt_stocks_on_date):
        f = {}
        ind_code = self._get_industry(stock, date)

        # 使用预计算的行业涨停数缓存 — O(1)查找
        day_key = str(date)
        ind_count_map = self._industry_zt_count_cache.get(day_key, {})
        zt_count = ind_count_map.get(ind_code, 0)

        # 如果缓存未命中，回退到逐个查询
        if day_key not in self._industry_zt_count_cache:
            zt_count = 0
            for s in zt_stocks_on_date:
                s_ind = self._get_industry(s, date)
                if s_ind == ind_code:
                    zt_count += 1

        f['industry_zt_count'] = zt_count
        f['industry_zt_ratio'] = zt_count / len(zt_stocks_on_date) if len(zt_stocks_on_date) > 0 else np.nan
        # 行业涨停排名: 该行业涨停数在所有行业中的百分位排名
        if ind_count_map:
            sorted_counts = sorted(ind_count_map.values(), reverse=True)
            rank_pos = sorted_counts.index(zt_count) if zt_count in sorted_counts else len(sorted_counts)
            f['industry_zt_rank'] = rank_pos / len(sorted_counts) if len(sorted_counts) > 1 else 0.0
        else:
            f['industry_zt_rank'] = np.nan
        return f

    # ---------- 趋势因子 ----------

    def _trend_factors(self, price_data):
        f = {}
        nan_map = {k: np.nan for k in FACTOR_CATEGORIES['trend']}
        if price_data is None or len(price_data) < 10:
            return nan_map

        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        cur = close.iloc[-1]

        # MA距离 & 是否在MA上方
        for w, key in [(5, 'ma5'), (10, 'ma10'), (20, 'ma20'), (60, 'ma60')]:
            if len(close) >= w:
                ma = close.iloc[-w:].mean()
                f[f'{key}_dist'] = (cur - ma) / ma if ma > 0 else 0
                f[f'above_{key}'] = 1 if cur > ma else 0
            else:
                f[f'{key}_dist'] = np.nan
                f[f'above_{key}'] = np.nan

        # MACD
        try:
            macd_val, signal_val, hist_val = compute_macd(close)
            f['macd'] = macd_val
            f['macd_signal'] = signal_val
            f['macd_hist'] = hist_val
        except:
            f['macd'] = f['macd_signal'] = f['macd_hist'] = np.nan

        # RSI
        for w, key in [(6, 'rsi_6'), (12, 'rsi_12'), (24, 'rsi_24')]:
            try:
                f[key] = compute_rsi(close, w)
            except:
                f[key] = np.nan

        # KDJ
        try:
            k, d, j = compute_kdj(high, low, close)
            f['kdj_k'] = k
            f['kdj_d'] = d
            f['kdj_j'] = j
        except:
            f['kdj_k'] = f['kdj_d'] = f['kdj_j'] = np.nan

        # BOLL
        try:
            pos, width = compute_boll(close)
            f['boll_position'] = pos
            f['boll_width'] = width
        except:
            f['boll_position'] = f['boll_width'] = np.nan

        # 新增: MA斜率 (近5日线性回归斜率)
        for w, key in [(5, 'ma5_slope'), (10, 'ma10_slope'), (20, 'ma20_slope')]:
            try:
                if len(close) >= w:
                    seg = close.iloc[-w:]
                    x = np.arange(w)
                    slope = np.polyfit(x, seg.values, 1)[0]
                    f[key] = slope / (seg.iloc[-1] + 1e-10)  # 归一化斜率
                else:
                    f[key] = np.nan
            except:
                f[key] = np.nan

        # 新增: 趋势强度 (多头排列得分: MA5>MA10>MA20>MA60)
        try:
            ma5 = close.iloc[-5:].mean() if len(close) >= 5 else np.nan
            ma10 = close.iloc[-10:].mean() if len(close) >= 10 else np.nan
            ma20 = close.iloc[-20:].mean() if len(close) >= 20 else np.nan
            ma60 = close.iloc[-60:].mean() if len(close) >= 60 else np.nan
            score = 0
            if not pd.isna(ma5) and not pd.isna(ma10):
                score += 1 if ma5 > ma10 else -1
            if not pd.isna(ma10) and not pd.isna(ma20):
                score += 1 if ma10 > ma20 else -1
            if not pd.isna(ma20) and not pd.isna(ma60):
                score += 1 if ma20 > ma60 else -1
            f['trend_strength'] = score / 3.0  # 归一化到[-1, 1]
        except:
            f['trend_strength'] = np.nan

        return f

    # ---------- 换手率因子（使用基本面数据，避免额外API调用） ----------

    def _turnover_factors(self, stock, date, fund_data):
        f = {}
        # turnover_ratio 来自 valuation 表（已在 _get_fundamentals_batch 中查询）
        tr = fund_data.get('turnover_ratio', np.nan)
        f['turnover_rate'] = tr

        # 换手率均值比值 — 使用历史基本面数据
        # ★ v1修复: 主动预加载历史换手率，而非依赖缓存中可能不存在的历史数据
        try:
            tr_list = []
            # 获取近12个交易日的换手率 — 使用get_fundamentals逐日查询
            for offset in range(1, 12):
                try:
                    offset_date = pd.Timestamp(date) - pd.Timedelta(days=offset * 2)
                    fund_key = str(offset_date.date())
                    # 如果缓存中已有该日数据，直接使用
                    if fund_key in self._fund_cache and stock in self._fund_cache[fund_key]:
                        tr_val = self._fund_cache[fund_key][stock].get('turnover_ratio', np.nan)
                        if not pd.isna(tr_val):
                            tr_list.append((offset, tr_val))
                    else:
                        # ★ v1: 缓存未命中时，主动查询该日基本面数据
                        if JQ_AVAILABLE:
                            try:
                                q = query(valuation.code, valuation.turnover_ratio).filter(valuation.code == stock)
                                df_off = get_fundamentals(q, date=offset_date.date())
                                if df_off is not None and not df_off.empty:
                                    tr_val = df_off.iloc[0]['turnover_ratio']
                                    # 写入缓存供后续使用
                                    if fund_key not in self._fund_cache:
                                        self._fund_cache[fund_key] = {}
                                    if stock not in self._fund_cache[fund_key]:
                                        self._fund_cache[fund_key][stock] = {}
                                    self._fund_cache[fund_key][stock]['turnover_ratio'] = tr_val
                                    if not pd.isna(tr_val):
                                        tr_list.append((offset, tr_val))
                            except Exception:
                                pass  # 个别日期查询失败不影响整体
                except:
                    pass

            if len(tr_list) >= 5:
                ma5 = np.mean([v for _, v in tr_list[:5]])
                f['turnover_ma5_ratio'] = tr / ma5 if not pd.isna(tr) and ma5 > 0 else np.nan
            else:
                f['turnover_ma5_ratio'] = np.nan

            if len(tr_list) >= 10:
                ma10 = np.mean([v for _, v in tr_list[:10]])
                f['turnover_ma10_ratio'] = tr / ma10 if not pd.isna(tr) and ma10 > 0 else np.nan
            else:
                f['turnover_ma10_ratio'] = np.nan
        except:
            f['turnover_ma5_ratio'] = f['turnover_ma10_ratio'] = np.nan

        # 新增: 换手率5日标准差
        if len(tr_list) >= 5:
            f['turnover_std5'] = np.std([v for _, v in tr_list[:5]])
        else:
            f['turnover_std5'] = np.nan
        return f

    # ---------- 资金流因子 ----------

    def _capital_flow_factors(self, stock, date):
        f = {k: np.nan for k in FACTOR_CATEGORIES['capital_flow']}
        if SKIP_SLOW_FACTORS:
            return f
        try:
            mf = get_money_flow(stock, start_date=date, end_date=date)
            if mf is not None and not mf.empty:
                row = mf.iloc[0]
                f['net_mf_amount'] = row.get('net_mf_amount', np.nan)
                f['net_mf_vol'] = row.get('net_mf_vol', np.nan)
                buy = abs(row.get('mf_buy_amount', 0))
                sell = abs(row.get('mf_sell_amount', 0))
                total = buy + sell
                f['net_mf_ratio'] = row.get('net_mf_amount', np.nan) / total if total > 0 else np.nan
            else:
                # ★ v1: 诊断日志
                if not SKIP_SLOW_FACTORS:
                    pass  # get_money_flow返回空是常见情况，不打印
        except Exception as e:
            # ★ v1: 仅首次异常时打印
            if not hasattr(self, '_mf_error_logged'):
                print(f"    [资金流⚠] get_money_flow异常({stock}): {e}")
                self._mf_error_logged = True
        return f

    # ---------- 估值因子 ----------

    def _valuation_factors(self, fund_data):
        f = {}
        for key in ['pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio',
                     'market_cap', 'circulating_market_cap']:
            f[key] = fund_data.get(key, np.nan)
        return f

    # ---------- 规模因子 ----------

    def _scale_factors(self, fund_data):
        f = {}
        mc = fund_data.get('market_cap', np.nan)
        cmc = fund_data.get('circulating_market_cap', np.nan)
        f['market_cap_log'] = np.log(mc) if not pd.isna(mc) and mc > 0 else np.nan
        f['circulating_cap_log'] = np.log(cmc) if not pd.isna(cmc) and cmc > 0 else np.nan
        return f

    # ---------- 基本面因子（修正字段名） ----------

    def _fundamental_factors(self, fund_data):
        f = {}
        mapping = {
            'roe': 'roe', 'roa': 'roa',
            'net_profit_margin': 'net_profit_margin',
            'gross_profit_margin': 'gross_profit_margin',
            'revenue_growth': 'inc_revenue_year_on_year',    # v1.1: JQ正确字段名
            'profit_growth': 'inc_net_profit_year_on_year',  # v1.1: JQ正确字段名
            'debt_asset_ratio': 'debt_asset_ratio',  # v1.1: 从balance表计算
        }
        for fkey, dkey in mapping.items():
            f[fkey] = fund_data.get(dkey, np.nan)
        return f

    # ---------- 前日因子 ----------

    def _prev_day_factors(self, price_data, fund_data):
        f = {k: np.nan for k in FACTOR_CATEGORIES['prev_day']}
        if price_data is None or len(price_data) < 2:
            return f
        prev = price_data.iloc[-2]
        ppc = price_data.iloc[-3]['close'] if len(price_data) >= 3 else prev.get('pre_close', np.nan)
        if pd.isna(ppc) or ppc <= 0:
            return f

        f['prev_pct_change'] = (prev['close'] - ppc) / ppc
        f['prev_amplitude'] = (prev['high'] - prev['low']) / ppc
        f['prev_upper_shadow'] = (prev['high'] - max(prev['open'], prev['close'])) / ppc
        f['prev_lower_shadow'] = (min(prev['open'], prev['close']) - prev['low']) / ppc

        if len(price_data) >= 7:
            vma5 = price_data['volume'].iloc[-7:-2].mean()
            f['prev_volume_ratio'] = prev['volume'] / vma5 if vma5 > 0 else np.nan

        # 前日换手率 — 从基本面数据获取
        f['prev_turnover'] = fund_data.get('turnover_ratio', np.nan)
        return f

    # ---------- 涨停特征因子（最高优先级） ----------

    def _limit_hit_factors(self, stock, date, price_data, pool_mgr):
        f = {k: np.nan for k in FACTOR_CATEGORIES['limit_hit']}

        # 连板数 & 首板标志
        info = pool_mgr.pool.get(stock, {})
        lc = info.get('limit_count', 1)
        f['continuous_limit_count'] = lc
        f['first_limit_flag'] = 1 if lc == 1 else 0
        f['limit_type'] = lc  # 1=首板, 2=2连板, ...

        # 涨停量比: 涨停日成交量 / 5日均量
        if price_data is not None and len(price_data) >= 6:
            vol_today = price_data.iloc[-1]['volume']
            vol_ma5 = price_data['volume'].iloc[-6:-1].mean()
            f['zt_volume_ratio'] = vol_today / vol_ma5 if vol_ma5 > 0 else np.nan
        else:
            f['zt_volume_ratio'] = np.nan

        # 涨停换手率: 使用基本面换手率
        # (将在 compute_all 中补充)

        # 封单额/流通市值: 需要封单额数据，JQ不直接提供，用成交额近似
        if price_data is not None and len(price_data) >= 1:
            row = price_data.iloc[-1]
            money_val = row.get('money', 0)
            # 流通市值从基本面获取（在compute_all中补充）
            f['seal_amount_ratio'] = np.nan  # 占位，后续补充

        # 涨停板强度: 综合指标 (连板数 * 封板速度权重)
        f['zt_strength'] = np.nan  # 需要分钟数据，占位

        # 涨停时段: 需要分钟数据
        f['limit_up_timeband'] = np.nan  # 占位

        # ---- 新增: 涨停板专属因子（日频数据可计算） ----

        # zt_momentum: 涨停前5日累计涨幅 — 涨停前动量
        if price_data is not None and len(price_data) >= 6:
            try:
                close_5d_ago = price_data.iloc[-6]['close']
                close_yesterday = price_data.iloc[-2]['close'] if len(price_data) >= 2 else np.nan
                if not pd.isna(close_5d_ago) and close_5d_ago > 0 and not pd.isna(close_yesterday):
                    f['zt_momentum'] = (close_yesterday - close_5d_ago) / close_5d_ago
                else:
                    f['zt_momentum'] = np.nan
            except:
                f['zt_momentum'] = np.nan
        else:
            f['zt_momentum'] = np.nan

        # zt_gap_type: 跳空类型 — 0=无跳空, 1=小跳空(<3%), 2=大跳空(≥3%)
        if price_data is not None and len(price_data) >= 2:
            try:
                prev_close = price_data.iloc[-2]['close']
                today_open = price_data.iloc[-1]['open']
                if not pd.isna(prev_close) and prev_close > 0:
                    gap_pct = (today_open - prev_close) / prev_close
                    if gap_pct < 0.005:
                        f['zt_gap_type'] = 0
                    elif gap_pct < 0.03:
                        f['zt_gap_type'] = 1
                    else:
                        f['zt_gap_type'] = 2
                else:
                    f['zt_gap_type'] = np.nan
            except:
                f['zt_gap_type'] = np.nan
        else:
            f['zt_gap_type'] = np.nan

        # zt_volume_surge: 涨停日量比 vs 20日均量
        if price_data is not None and len(price_data) >= 21:
            try:
                vol_today = price_data.iloc[-1]['volume']
                vol_ma20 = price_data['volume'].iloc[-21:-1].mean()
                f['zt_volume_surge'] = vol_today / vol_ma20 if vol_ma20 > 0 else np.nan
            except:
                f['zt_volume_surge'] = np.nan
        else:
            f['zt_volume_surge'] = np.nan

        # zt_seal_strength_approx: 近似封板强度 — (收盘价-开盘价)/(涨停价-开盘价+0.001)
        # 无分钟数据时，用日K近似: 收盘越接近涨停价，封板越强
        if price_data is not None and len(price_data) >= 1:
            try:
                row = price_data.iloc[-1]
                open_val = row['open']
                close_val = row['close']
                pre_close_val = row.get('pre_close', np.nan)
                if pd.isna(pre_close_val) and len(price_data) >= 2:
                    pre_close_val = price_data.iloc[-2]['close']
                if not pd.isna(pre_close_val) and pre_close_val > 0:
                    high_limit_p = compute_high_limit(stock, pre_close_val)
                    denom = high_limit_p - open_val
                    if denom > 0:
                        f['zt_seal_strength_approx'] = (close_val - open_val) / denom
                    else:
                        f['zt_seal_strength_approx'] = np.nan
                else:
                    f['zt_seal_strength_approx'] = np.nan
            except:
                f['zt_seal_strength_approx'] = np.nan
        else:
            f['zt_seal_strength_approx'] = np.nan

        # 分钟级数据: 封板速度、开板次数、封板时长
        if not USE_MINUTE_DATA:
            return f

        try:
            min_df = get_price(stock, start_date=date, end_date=date,
                               frequency='minute',
                               fields=['open', 'close', 'high', 'low', 'volume', 'money'],
                               skip_paused=False, fq='pre')
            if min_df is None or min_df.empty:
                return f

            high_limit_price = price_data.iloc[-1]['close']  # 涨停日收盘≈涨停价

            # 封板速度: 第一次触及涨停的时间点
            seal_idx = None
            for i, row in min_df.iterrows():
                if row['high'] >= high_limit_price * 0.995:
                    seal_idx = min_df.index.get_loc(i)
                    break

            total_min = len(min_df)
            if seal_idx is not None and total_min > 0:
                f['seal_speed'] = 1.0 - (seal_idx / total_min)
                f['seal_time'] = seal_idx
                # 涨停时段: 1=早盘(9:30-10:30), 2=午盘(10:30-13:00), 3=尾盘(13:00-15:00)
                if seal_idx < total_min * 0.125:  # 前1/8 ≈ 前30分钟
                    f['limit_up_timeband'] = 1
                elif seal_idx < total_min * 0.5:
                    f['limit_up_timeband'] = 2
                else:
                    f['limit_up_timeband'] = 3
            else:
                f['seal_speed'] = 0.0
                f['seal_time'] = np.nan
                f['limit_up_timeband'] = 4  # 未封板/尾盘封板

            # 开板次数: 涨停后跌破涨停价
            open_count = 0
            at_limit = False
            for _, row in min_df.iterrows():
                if row['high'] >= high_limit_price * 0.995:
                    at_limit = True
                elif at_limit and row['close'] < high_limit_price * 0.995:
                    open_count += 1
                    at_limit = False
            f['open_count'] = open_count

            # 封板时长: 在涨停价的分钟占比
            at_limit_min = sum(1 for _, r in min_df.iterrows()
                               if r['close'] >= high_limit_price * 0.995)
            f['limit_duration'] = at_limit_min / total_min if total_min > 0 else 0

            # 涨停板强度: 封板速度 * (1 - 开板次数/10) * 封板时长
            if not pd.isna(f['seal_speed']):
                f['zt_strength'] = f['seal_speed'] * (1 - open_count / 10) * f.get('limit_duration', 0.5)

        except Exception:
            pass

        return f

    # ---------- 集合竞价因子 ----------

    def _auction_factors(self, stock, date, price_data):
        f = {k: np.nan for k in FACTOR_CATEGORIES['auction']}

        if not USE_MINUTE_DATA:
            return f

        try:
            min_df = get_price(stock, start_date=date, end_date=date,
                               frequency='minute',
                               fields=['open', 'close', 'high', 'low', 'volume', 'money'],
                               skip_paused=False, fq='pre')
            if min_df is None or min_df.empty:
                return f
            first = min_df.iloc[0]
            pc = price_data.iloc[-2]['close'] if price_data is not None and len(price_data) >= 2 else np.nan
            if not pd.isna(pc) and pc > 0:
                f['auction_pct'] = (first['open'] - pc) / pc
            if len(min_df) > 5:
                avg5 = min_df.iloc[:5]['volume'].mean()
                f['auction_volume_ratio'] = first['volume'] / avg5 if avg5 > 0 else np.nan
        except:
            pass
        return f

    # ---------- 波动率因子 ----------

    def _volatility_factors(self, price_data):
        f = {k: np.nan for k in FACTOR_CATEGORIES['volatility']}
        if price_data is None or len(price_data) < 6:
            return f

        close = price_data['close']
        high = price_data['high']
        low = price_data['low']

        # 历史波动率
        rets = close.pct_change().dropna()
        for w, key in [(5, 'hist_vol_5'), (10, 'hist_vol_10'), (20, 'hist_vol_20')]:
            if len(rets) >= w:
                f[key] = rets.iloc[-w:].std()
            else:
                f[key] = np.nan

        # ATR(14)
        if len(price_data) >= 15:
            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1)),
            }).max(axis=1)
            f['atr_14'] = tr.rolling(window=14).mean().iloc[-1]
        else:
            f['atr_14'] = np.nan

        # 新增: 波动率比 (5日/20日)
        if len(rets) >= 20:
            vol5 = rets.iloc[-5:].std()
            vol20 = rets.iloc[-20:].std()
            f['vol_ratio_5_20'] = vol5 / vol20 if vol20 > 0 else np.nan
        else:
            f['vol_ratio_5_20'] = np.nan

        # 新增: 20日最大回撤
        if len(close) >= 20:
            seg = close.iloc[-20:]
            cummax = seg.cummax()
            drawdown = (seg - cummax) / cummax
            f['max_drawdown_20'] = drawdown.min()
        else:
            f['max_drawdown_20'] = np.nan

        return f

    # ---------- Alpha因子（价格量背离优先 + 涨停特征Alpha） ----------

    def _alpha_factors(self, price_data, limit_factors, volume_factors, turnover_rate):
        f = {}
        nan_map = {k: np.nan for k in FACTOR_CATEGORIES['alpha']}
        if price_data is None or len(price_data) < 15:
            return nan_map

        close = price_data['close']
        open_ = price_data['open']
        high = price_data['high']
        low = price_data['low']
        vol = price_data['volume']

        # Alpha#6: -1 * correlation(open, volume, 10) — 价格量背离
        try:
            if len(price_data) >= 10:
                c = open_.iloc[-10:].corr(vol.iloc[-10:])
                f['alpha_6'] = -1 * c if not pd.isna(c) else 0
            else:
                f['alpha_6'] = np.nan
        except:
            f['alpha_6'] = np.nan

        # Alpha#12: sign(delta(volume,1)) * (-1 * delta(close,1)) — 量价反转
        try:
            if len(price_data) >= 2:
                dv = vol.iloc[-1] - vol.iloc[-2]
                dc = close.iloc[-1] - close.iloc[-2]
                f['alpha_12'] = np.sign(dv) * (-1 * dc)
            else:
                f['alpha_12'] = np.nan
        except:
            f['alpha_12'] = np.nan

        # Alpha#33: rank(-1 * (1 - (open / close))) — 日内收益模式
        try:
            if close.iloc[-1] > 0:
                f['alpha_33'] = -1 * (1 - (open_.iloc[-1] / close.iloc[-1]))
            else:
                f['alpha_33'] = np.nan
        except:
            f['alpha_33'] = np.nan

        # Alpha#54: -1 * ((low-close)*(open^5)) / ((low-high)*(close^5))
        try:
            c5 = close.iloc[-1] ** 5
            o5 = open_.iloc[-1] ** 5
            denom = (low.iloc[-1] - high.iloc[-1]) * c5
            f['alpha_54'] = -1 * ((low.iloc[-1] - close.iloc[-1]) * o5) / denom if abs(denom) > 1e-10 else np.nan
        except:
            f['alpha_54'] = np.nan

        # Alpha#41: 量价背离 — rank(correlation(close, volume, 7)) * rank(delta(close, 7))
        try:
            if len(price_data) >= 8:
                corr_cv = close.iloc[-7:].corr(vol.iloc[-7:])
                delta_c7 = close.iloc[-1] - close.iloc[-8]
                f['alpha_41'] = -1 * corr_cv * np.sign(delta_c7) if not pd.isna(corr_cv) else np.nan
            else:
                f['alpha_41'] = np.nan
        except:
            f['alpha_41'] = np.nan

        # Alpha#53: 买卖力道变化 — -1 * delta(((close-low)-(high-close))/(close-low), 9)
        try:
            if len(price_data) >= 10 and close.iloc[-1] > 0:
                cl = close - low
                hc = high - close
                cl_shift = close.shift(1) - low.shift(1)
                hc_shift = high.shift(1) - close.shift(1)
                cur_val = ((cl - hc) / cl).iloc[-1]
                prev_val = ((cl_shift - hc_shift) / cl_shift).iloc[-10] if len(price_data) >= 11 else 0
                f['alpha_53'] = -1 * (cur_val - prev_val) if not pd.isna(cur_val) and not pd.isna(prev_val) else np.nan
            else:
                f['alpha_53'] = np.nan
        except:
            f['alpha_53'] = np.nan

        # Alpha#101: 日内收益占比 — (close - open) / ((high - low) + 0.001)
        try:
            hl_range = high.iloc[-1] - low.iloc[-1]
            f['alpha_101'] = (close.iloc[-1] - open_.iloc[-1]) / (hl_range + 0.001)
        except:
            f['alpha_101'] = np.nan

        # 涨停特征Alpha（最高优先级）
        f['alpha_zt_seal'] = limit_factors.get('seal_speed', np.nan)
        f['alpha_zt_open'] = -limit_factors.get('open_count', np.nan)  # 负号: 开板越少值越大
        f['alpha_zt_type'] = limit_factors.get('limit_type', np.nan)
        f['alpha_zt_vol'] = volume_factors.get('volume_ma5_ratio', np.nan)
        f['alpha_zt_turnover'] = turnover_rate

        # Alpha#5: -1 * ts_max(close, 5) 的排名变化 — 短期极值反转
        try:
            if len(price_data) >= 10:
                ts_max_5 = close.rolling(5).max()
                delta_tsmax = ts_max_5.iloc[-1] - ts_max_5.iloc[-6] if len(ts_max_5) >= 6 else np.nan
                f['alpha_5'] = -1 * delta_tsmax if not pd.isna(delta_tsmax) else np.nan
            else:
                f['alpha_5'] = np.nan
        except:
            f['alpha_5'] = np.nan

        # Alpha#15: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3) — 高点量价背离
        try:
            if len(price_data) >= 6:
                rank_h = high.iloc[-6:].rank()
                rank_v = vol.iloc[-6:].rank()
                corr_3 = rank_h.rolling(3).corr(rank_v)
                f['alpha_15'] = -1 * corr_3.iloc[-3:].sum() if len(corr_3.dropna()) >= 1 else np.nan
            else:
                f['alpha_15'] = np.nan
        except:
            f['alpha_15'] = np.nan

        # Alpha#20: -1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))
        try:
            if len(price_data) >= 2:
                d_high = high.shift(1).iloc[-1]
                d_close = close.shift(1).iloc[-1]
                d_low = low.shift(1).iloc[-1]
                cur_open = open_.iloc[-1]
                if not pd.isna(d_high) and not pd.isna(d_close) and not pd.isna(d_low):
                    r1 = cur_open - d_high
                    r2 = cur_open - d_close
                    r3 = cur_open - d_low
                    f['alpha_20'] = -1 * np.sign(r1) * np.sign(r2) * np.sign(r3) * abs(r1 * r2 * r3) ** (1/3)
                else:
                    f['alpha_20'] = np.nan
            else:
                f['alpha_20'] = np.nan
        except:
            f['alpha_20'] = np.nan

        return f

    # ---------- 动量因子 ----------

    def _momentum_factors(self, price_data):
        """多周期动量因子 — ret_1d, ret_3d, ret_5d, ret_10d, ret_20d"""
        f = {k: np.nan for k in FACTOR_CATEGORIES['momentum']}
        if price_data is None or len(price_data) < 2:
            return f

        close = price_data['close']
        for window, key in [(1, 'ret_1d'), (3, 'ret_3d'), (5, 'ret_5d'),
                            (10, 'ret_10d'), (20, 'ret_20d')]:
            try:
                if len(close) > window:
                    prev = close.iloc[-(window + 1)]
                    cur = close.iloc[-1]
                    if not pd.isna(prev) and prev > 0:
                        f[key] = (cur - prev) / prev
            except:
                pass
        return f

    # ---------- 技术形态因子 ----------

    def _technical_shape_factors(self, price_data):
        """K线形态识别 — doji_flag, hammer_flag, engulfing_flag"""
        f = {k: np.nan for k in FACTOR_CATEGORIES['technical_shape']}
        if price_data is None or len(price_data) < 2:
            return f

        try:
            cur = price_data.iloc[-1]
            prev = price_data.iloc[-2]
            c_o = cur['close'] - cur['open']
            h_l = cur['high'] - cur['low']

            # 十字星: 实体 < 全幅的10%
            if h_l > 0:
                f['doji_flag'] = 1 if abs(c_o) / h_l < 0.1 else 0
            else:
                f['doji_flag'] = np.nan

            # 锤子线: 下影线 > 实体2倍, 上影线 < 实体
            body = abs(c_o)
            upper_shadow = cur['high'] - max(cur['open'], cur['close'])
            lower_shadow = min(cur['open'], cur['close']) - cur['low']
            if body > 0:
                f['hammer_flag'] = 1 if lower_shadow > 2 * body and upper_shadow < body else 0
            else:
                f['hammer_flag'] = np.nan

            # 吞没形态: 今日实体完全包含前日实体, 且方向相反
            prev_body = prev['close'] - prev['open']
            cur_body = cur['close'] - cur['open']
            if prev_body != 0:
                engulfed = (min(cur['open'], cur['close']) < min(prev['open'], prev['close']) and
                           max(cur['open'], cur['close']) > max(prev['open'], prev['close']))
                opposite_sign = (prev_body > 0 and cur_body < 0) or (prev_body < 0 and cur_body > 0)
                f['engulfing_flag'] = 1 if engulfed and opposite_sign else 0
            else:
                f['engulfing_flag'] = np.nan
        except:
            pass

        return f

    # ---------- 总入口 ----------

    def compute_all(self, stock, date, zt_stocks_on_date, pool_mgr):
        """计算单只股票的全部因子"""
        price_data = self._get_hist_price(stock, date)
        fund_batch = self._get_fundamentals_batch([stock], date)
        fund_data = fund_batch.get(stock, {})

        # ★ 关键修复: 提取close价格，用于T+N收益计算
        close_val = np.nan
        if price_data is not None and len(price_data) > 0:
            close_val = price_data.iloc[-1]['close']

        all_f = {'code': stock, 'date': date, 'close': close_val}

        fq = self._price_quote_factors(price_data)
        vf = self._volume_factors(price_data)
        tf = self._trend_factors(price_data)
        pf = self._prev_day_factors(price_data, fund_data)
        lf = self._limit_hit_factors(stock, date, price_data, pool_mgr)
        af = self._auction_factors(stock, date, price_data)
        vf_vol = self._volatility_factors(price_data)
        mf = self._momentum_factors(price_data)
        sf = self._technical_shape_factors(price_data)
        tr_rate = fund_data.get('turnover_ratio', np.nan)

        all_f.update(fq)
        all_f.update(vf)
        all_f.update(self._industry_factors(stock, date, zt_stocks_on_date))
        all_f.update(tf)
        all_f.update(self._turnover_factors(stock, date, fund_data))
        all_f.update(self._capital_flow_factors(stock, date))
        all_f.update(self._valuation_factors(fund_data))
        all_f.update(self._scale_factors(fund_data))
        all_f.update(self._fundamental_factors(fund_data))
        all_f.update(pf)
        all_f.update(lf)
        all_f.update(af)
        all_f.update(vf_vol)
        all_f.update(mf)
        all_f.update(sf)
        all_f.update(self._alpha_factors(price_data, lf, vf, tr_rate))

        # 补充涨停换手率
        all_f['zt_turnover_rate'] = tr_rate

        # 补充封单额/流通市值
        cmc = fund_data.get('circulating_market_cap', np.nan)
        money_val = price_data.iloc[-1].get('money', np.nan) if price_data is not None and len(price_data) > 0 else np.nan
        if not pd.isna(cmc) and cmc > 0 and not pd.isna(money_val):
            all_f['seal_amount_ratio'] = money_val / (cmc * 1e8)  # market_cap单位是亿

        self.factor_records.append(all_f)
        return all_f


# ============================================================
# PART 5: IC衰减监控器（基础设施优先）
# ============================================================

class ICDdecayMonitor:
    """IC衰减监控 — 评估因子预测能力随时间衰减"""

    def __init__(self):
        self.ic_data = {}  # {factor: {tn_N: ic_value}}

    def compute_all_ic(self, factor_df, return_df, factor_names=None):
        if factor_names is None:
            factor_names = [c for c in factor_df.columns if c in ALL_FACTORS]

        self.ic_data = {}
        for f in factor_names:
            if f not in factor_df.columns:
                continue
            self.ic_data[f] = {}
            for n in range(1, TRACK_DAYS + 1):
                rn = f'tn_{n}'
                if rn not in return_df.columns:
                    continue
                ic = self._spearman_ic(factor_df[f], return_df[rn])
                self.ic_data[f][rn] = ic

    @staticmethod
    def _spearman_ic(x, y):
        """计算Spearman IC"""
        try:
            valid = ~(x.isna() | y.isna())
            if valid.sum() < MIN_IC_SAMPLES:
                return np.nan
            c, _ = spearmanr(x[valid], y[valid])
            return c
        except:
            return np.nan

    def get_top_factors(self, period='tn_5', top_n=20, min_abs_ic=0.02):
        """获取指定T+N周期IC最高的因子"""
        results = {}
        for f, ics in self.ic_data.items():
            ic = ics.get(period, np.nan)
            if not pd.isna(ic) and abs(ic) >= min_abs_ic:
                results[f] = ic
        return sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    def print_report(self):
        print(f"\n{'='*70}")
        print("IC衰减监控报告")
        print(f"{'='*70}")

        if not self.ic_data:
            print("  无IC数据")
            return

        # IC最高的因子
        for period in [f'tn_{n}' for n in range(1, TRACK_DAYS + 1)]:
            top = self.get_top_factors(period=period, top_n=10, min_abs_ic=0.01)
            if top:
                print(f"\n  {period} IC Top 10:")
                for f, ic in top:
                    print(f"    {f:30s}: IC = {ic:.4f}")

        # IC衰减分析
        print(f"\n  IC衰减分析 (|IC|随T+N变化):")
        decay_examples = list(self.ic_data.keys())[:10]
        for f in decay_examples:
            ics = self.ic_data[f]
            vals = [ics.get(f'tn_{n}', np.nan) for n in range(1, TRACK_DAYS + 1)]
            vals_str = " → ".join(f"{v:.3f}" if not pd.isna(v) else "NaN" for v in vals)
            print(f"    {f:30s}: {vals_str}")


# ============================================================
# PART 6: 共线性检测器（基础设施优先）
# ============================================================

class CollinearityDetector:
    """因子共线性检测 — 识别冗余因子"""

    def __init__(self):
        self.corr_matrix = None
        self.high_corr_pairs = []
        self.vif_dict = {}

    def compute_corr(self, factor_df, factor_names=None):
        if factor_names is None:
            factor_names = [c for c in factor_df.columns if c in ALL_FACTORS]
        avail = [f for f in factor_names if f in factor_df.columns]
        df = factor_df[avail].dropna(how='all')
        if df.empty:
            return pd.DataFrame()
        self.corr_matrix = df.corr(method='spearman')
        return self.corr_matrix

    def detect_high_corr(self, threshold=COLLINEARITY_THRESHOLD):
        if self.corr_matrix is None:
            return []
        self.high_corr_pairs = []
        cols = self.corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = self.corr_matrix.iloc[i, j]
                if abs(v) >= threshold:
                    self.high_corr_pairs.append((cols[i], cols[j], v))
        self.high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return self.high_corr_pairs

    def compute_vif(self, factor_df, factor_names=None):
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from statsmodels.tools.tools import add_constant
        except ImportError:
            print("  [跳过VIF] statsmodels未安装")
            return {}

        if factor_names is None:
            factor_names = [c for c in factor_df.columns if c in ALL_FACTORS]
        avail = [f for f in factor_names if f in factor_df.columns]

        # ★ v1.3修复: VIF=999的根因与迭代消除
        # v1.2已修复: 高缺失率/近零方差/缺少截距 → 但仍全部999
        # v1.3新增:
        #   1) 更严格缺失率阈值(30%而非50%) — 中位数填充后仍造成共线性
        #   2) 高相关预过滤 — |ρ|>0.95的因子对移除缺失率较高者
        #   3) 迭代VIF消除 — 每轮移除最高VIF因子，直到所有VIF<阈值

        df = factor_df[avail].copy()

        # 步骤1: 过滤高缺失率列(>30%缺失 → 中位数填充后仍造成严重共线性)
        n_rows = len(df)
        low_missing = []
        high_missing = []
        for col in df.columns:
            missing_rate = df[col].isna().mean()
            if missing_rate > 0.30:
                high_missing.append(col)
            else:
                low_missing.append(col)
        if high_missing:
            print(f"    [VIF] 跳过{len(high_missing)}个高缺失率因子(>30%): {high_missing[:5]}{'...' if len(high_missing)>5 else ''}")
        df = df[low_missing]

        # 步骤2: 中位数填充剩余NaN
        for col in df.columns:
            if df[col].isna().any():
                med = df[col].median()
                if pd.isna(med):
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(med)

        # 步骤3: 过滤近零方差列(方差<1e-10 → 填充后为常数)
        near_zero_var = []
        good_var = []
        for col in df.columns:
            if df[col].var() < 1e-10:
                near_zero_var.append(col)
            else:
                good_var.append(col)
        if near_zero_var:
            print(f"    [VIF] 跳过{len(near_zero_var)}个近零方差因子: {near_zero_var[:5]}{'...' if len(near_zero_var)>5 else ''}")
        df = df[good_var]

        # 步骤3.5: 高相关预过滤 — |ρ|>0.95的因子对，移除缺失率较高者
        if len(df.columns) > 1:
            corr_m = df.corr(method='spearman').abs()
            removed_corr = set()
            cols = list(df.columns)
            for i in range(len(cols)):
                if cols[i] in removed_corr:
                    continue
                for j in range(i + 1, len(cols)):
                    if cols[j] in removed_corr:
                        continue
                    if corr_m.loc[cols[i], cols[j]] > 0.95:
                        # 移除缺失率较高的那个
                        mi = factor_df[cols[i]].isna().mean() if cols[i] in factor_df.columns else 1.0
                        mj = factor_df[cols[j]].isna().mean() if cols[j] in factor_df.columns else 1.0
                        drop = cols[j] if mj >= mi else cols[i]
                        removed_corr.add(drop)
            if removed_corr:
                print(f"    [VIF] 预过滤{len(removed_corr)}个高相关因子(|ρ|>0.95): {list(removed_corr)[:5]}")
                df = df[[c for c in df.columns if c not in removed_corr]]

        # 样本数检查: VIF需要样本数 > 因子数 + 1
        if df.empty or len(df) < len(df.columns) + 1:
            print(f"    [VIF⚠] 样本数({len(df)})不足，需要至少{len(df.columns)+1}条")
            return {}

        # 步骤4: 迭代VIF消除 — 每轮移除最高VIF因子，直到所有VIF<阈值
        MAX_VIF_ITER = 50
        removed_by_vif = []
        remaining_cols = list(df.columns)

        for iteration in range(MAX_VIF_ITER):
            if len(remaining_cols) < 2:
                break
            sub_df = df[remaining_cols]
            X = add_constant(sub_df.values)
            vif_results = {}
            max_vif = 0.0
            max_vif_col = None
            for i, f in enumerate(remaining_cols):
                try:
                    vif = variance_inflation_factor(X, i + 1)
                    vif_results[f] = vif
                    if vif > max_vif:
                        max_vif = vif
                        max_vif_col = f
                except Exception:
                    vif_results[f] = np.nan

            if max_vif_col is None or max_vif <= VIF_THRESHOLD * 2:
                # 所有VIF都在可接受范围内(阈值×2作为宽松上限)
                break

            # 移除最高VIF因子
            removed_by_vif.append((max_vif_col, max_vif))
            remaining_cols.remove(max_vif_col)

        if removed_by_vif:
            print(f"    [VIF] 迭代消除{len(removed_by_vif)}个高VIF因子:")
            for fname, fvif in removed_by_vif[:10]:
                cn = FACTOR_NAME_CN.get(fname, '')
                print(f"      {fname}({cn}): VIF={fvif:.1f}")

        # 最终VIF计算(对剩余因子)
        if len(remaining_cols) < 2:
            print(f"    [VIF⚠] 迭代消除后剩余因子不足2个，无法计算VIF")
            self.vif_dict = {}
            return self.vif_dict

        final_df = df[remaining_cols]
        X = add_constant(final_df.values)
        self.vif_dict = {}
        for i, f in enumerate(final_df.columns):
            try:
                vif = variance_inflation_factor(X, i + 1)
                self.vif_dict[f] = min(vif, 999)
            except Exception:
                self.vif_dict[f] = np.nan
        return self.vif_dict

    def suggest_removal(self):
        """建议移除的冗余因子"""
        remove = set()
        for f1, f2, corr in self.high_corr_pairs:
            if f1 not in remove and f2 not in remove:
                # 保留VIF较低的（或名称较短的）
                v1 = self.vif_dict.get(f1, 999)
                v2 = self.vif_dict.get(f2, 999)
                remove.add(f2 if v2 >= v1 else f1)
        return sorted(remove)

    def print_report(self):
        print(f"\n{'='*70}")
        print("因子共线性检测报告")
        print(f"{'='*70}")

        if self.corr_matrix is not None:
            print(f"\n高相关因子对 (|ρ| >= {COLLINEARITY_THRESHOLD}):")
            if self.high_corr_pairs:
                for f1, f2, v in self.high_corr_pairs[:30]:
                    cn1 = FACTOR_NAME_CN.get(f1, '')
                    cn2 = FACTOR_NAME_CN.get(f2, '')
                    label1 = f"{f1}({cn1})" if cn1 else f1
                    label2 = f"{f2}({cn2})" if cn2 else f2
                    print(f"  {label1:35s} <-> {label2:35s}  ρ={v:.4f}")
            else:
                print("  未发现高相关因子对")

        if self.vif_dict:
            print(f"\nVIF分析 (阈值={VIF_THRESHOLD}):")
            sorted_vif = sorted(self.vif_dict.items(),
                                key=lambda x: x[1] if not pd.isna(x[1]) else 0,
                                reverse=True)
            for f, v in sorted_vif[:20]:
                cn = FACTOR_NAME_CN.get(f, '')
                label = f"{f}({cn})" if cn else f
                flag = " ⚠️" if v > VIF_THRESHOLD else ""
                print(f"  {label:40s} VIF={v:.2f}{flag}")

        removal = self.suggest_removal()
        if removal:
            print(f"\n建议移除的冗余因子 ({len(removal)}个):")
            for r in removal:
                cn = FACTOR_NAME_CN.get(r, '')
                label = f"{r}({cn})" if cn else r
                print(f"  {label}")


# ============================================================
# PART 7: T+N 追踪器
# ============================================================

class TNTracker:
    """T+N涨跌追踪 — 个股与汇总统计"""

    def __init__(self):
        self.records = []  # [{stock, entry_date, entry_close, tn_1..tn_5, tn_1_enc..tn_5_enc}]

    def compute_tn_returns(self, zt_events, trade_days, lookback_start):
        """
        计算所有涨停事件的T+N收益
        zt_events: [{stock, date, close}, ...]
        trade_days: 全部交易日列表
        lookback_start: 分析期起始日
        """
        # 建立交易日索引 — 统一转换为 pd.Timestamp 避免类型不匹配
        td_list = [_to_timestamp(d) for d in trade_days]
        td_idx = {d: i for i, d in enumerate(td_list)}

        # 按股票分组，预取价格
        stock_events = {}
        for ev in zt_events:
            stock_events.setdefault(ev['stock'], []).append(ev)

        print(f"  预取T+1~T+{TRACK_DAYS}收益数据: {len(stock_events)}只股票...")
        count = 0
        for stock, events in stock_events.items():
            count += 1
            if count % 50 == 0:
                print(f"    进度: {count}/{len(stock_events)}")

            # 获取该股票在分析期+追踪期的价格
            try:
                earliest = _to_timestamp(min(e['date'] for e in events))
                latest_idx = max(td_idx.get(_to_timestamp(e['date']), 0) for e in events)
                end_idx = min(latest_idx + TRACK_DAYS + 1, len(td_list) - 1)
                end_date = td_list[end_idx]

                # ★ 修复: 使用 fq='pre' 前复权，与因子引擎的 close 一致
                price = get_price(stock, start_date=earliest, end_date=end_date,
                                  frequency='daily', fields=['close'],
                                  skip_paused=True, fq='pre')
                if price is None or price.empty:
                    continue

                # 构建日期->收盘价映射 — key 统一为 pd.Timestamp
                price_map = {}
                for idx, row in price.iterrows():
                    ts = _to_timestamp(idx)
                    price_map[ts] = row['close']

            except:
                continue

            # 计算每个事件的T+N收益
            for ev in events:
                entry_date = _to_timestamp(ev['date'])
                entry_close = ev['close']
                if pd.isna(entry_close) or entry_close <= 0:
                    continue

                di = td_idx.get(entry_date)
                if di is None:
                    continue

                rec = {
                    'stock': stock,
                    'entry_date': entry_date,
                    'entry_close': entry_close,
                }

                for n in range(1, TRACK_DAYS + 1):
                    target_idx = di + n
                    if target_idx < len(td_list):
                        target_date = td_list[target_idx]
                        target_close = price_map.get(target_date, np.nan)
                        if not pd.isna(target_close) and entry_close > 0:
                            ret = (target_close - entry_close) / entry_close
                            rec[f'tn_{n}'] = ret
                            rec[f'tn_{n}_encoded'] = encode_return(ret)
                        else:
                            rec[f'tn_{n}'] = np.nan
                            rec[f'tn_{n}_encoded'] = np.nan
                    else:
                        rec[f'tn_{n}'] = np.nan
                        rec[f'tn_{n}_encoded'] = np.nan

                self.records.append(rec)

    def get_results_df(self):
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)

    def get_aggregated_stats(self):
        """汇总T+N统计"""
        df = self.get_results_df()
        if df.empty:
            return pd.DataFrame()

        stats = {}
        for n in range(1, TRACK_DAYS + 1):
            key = f'tn_{n}'
            enc_key = f'{key}_encoded'
            if key not in df.columns:
                continue
            rets = df[key].dropna()
            encs = df[enc_key].dropna() if enc_key in df.columns else pd.Series(dtype=float)
            stats[key] = {
                '样本数': len(rets),
                '平均收益': rets.mean(),
                '中位收益': rets.median(),
                '收益标准差': rets.std(),
                '胜率': (rets > 0).mean() if len(rets) > 0 else np.nan,
                '下跌(0)': (encs == 0).sum() if len(encs) > 0 else 0,
                '上涨<9.9%(1)': (encs == 1).sum() if len(encs) > 0 else 0,
                '上涨≥9.9%(2)': (encs == 2).sum() if len(encs) > 0 else 0,
            }
        return pd.DataFrame(stats).T

    def get_encoded_distribution(self):
        """T+N编码分布"""
        df = self.get_results_df()
        if df.empty:
            return pd.DataFrame()
        dist = {}
        for n in range(1, TRACK_DAYS + 1):
            enc_key = f'tn_{n}_encoded'
            if enc_key not in df.columns:
                continue
            enc = df[enc_key].dropna()
            total = len(enc)
            dist[f'T+{n}'] = {
                '下跌(0)': (enc == 0).sum(),
                '上涨<9.9%(1)': (enc == 1).sum(),
                '上涨≥9.9%(2)': (enc == 2).sum(),
                '合计': total,
                '下跌占比': (enc == 0).mean() if total > 0 else np.nan,
                '上涨<9.9%占比': (enc == 1).mean() if total > 0 else np.nan,
                '上涨≥9.9%占比': (enc == 2).mean() if total > 0 else np.nan,
            }
        return pd.DataFrame(dist).T


# ============================================================
# PART 8: 因子-收益相关性分析
# ============================================================

def build_bucket_correlation_matrix(factor_df, return_df):
    """
    构建因子-收益分桶相关性矩阵 (Section 4.1)
    对T+5收益分桶，计算每个因子与各桶的二值Spearman相关
    """
    if 'tn_5' not in return_df.columns:
        print("  [跳过] 缺少T+5收益数据")
        return pd.DataFrame()

    ret5 = return_df['tn_5']
    factor_cols = [c for c in factor_df.columns if c in ALL_FACTORS]

    bucket_corr = {}
    for bname, (lo, hi) in RETURN_BUCKETS.items():
        in_bucket = ((ret5 >= lo) & (ret5 < hi)).astype(float)
        corrs = {}
        for fc in factor_cols:
            if fc not in factor_df.columns:
                continue
            valid = ~(factor_df[fc].isna() | in_bucket.isna())
            if valid.sum() < MIN_IC_SAMPLES:
                corrs[fc] = np.nan
                continue
            try:
                c, _ = spearmanr(factor_df[fc][valid], in_bucket[valid])
                corrs[fc] = c
            except:
                corrs[fc] = np.nan
        bucket_corr[bname] = corrs

    result = pd.DataFrame(bucket_corr)
    if 'strong_rise' in result.columns:
        result = result.reindex(result['strong_rise'].abs().sort_values(ascending=False).index)
    return result


def identify_positive_factors(bucket_corr_df, min_corr=0.03):
    """识别与各收益分桶正相关的因子 (Section 3)"""
    results = {}
    for bucket in bucket_corr_df.columns:
        s = bucket_corr_df[bucket]
        pos = s[s > min_corr].sort_values(ascending=False)
        results[bucket] = pos
    return results


# ============================================================
# PART 8.1: 股票池评分系统
# ============================================================

class StockScorer:
    """
    股票池评分系统 — 基于因子-收益相关性分析

    评分逻辑:
    1. 使用IC值确定因子预测方向和权重
    2. 使用分桶相关性验证因子与涨跌方向的关系
    3. 对股票池中每只股票计算综合得分 (z-score加权求和)
    4. 分类: 强势 / 弱势 / 淘汰
    5. 输出Top5及交易信号(买入价/止损价/目标价)
    """

    # 分类阈值 (百分位)
    STRONG_PERCENTILE = 70    # 得分 >= P70 → 强势
    WEAK_PERCENTILE = 30      # 得分 <= P30 → 淘汰, P30~P70 → 弱势

    # 交易信号参数
    MIN_STOP_LOSS_PCT = 0.03   # 最小止损比例 3%
    MAX_STOP_LOSS_PCT = 0.08   # 最大止损比例 8%
    DEFAULT_STOP_LOSS_PCT = 0.05  # 默认止损比例 5%
    TARGET_REWARD_RATIO = 2.0  # 默认盈亏比

    def __init__(self, ic_monitor, bucket_corr_df, positive_factors, available_factors):
        self.ic_monitor = ic_monitor
        self.bucket_corr_df = bucket_corr_df
        self.positive_factors = positive_factors
        self.available_factors = available_factors
        self.scores_df = pd.DataFrame()
        self.factor_weights = {}
        self._latest_factors = None   # 保存最新因子数据用于贡献分析
        self._zscore_data = None     # 保存z-score数据
        self._compute_factor_weights()

    def _compute_factor_weights(self):
        """
        计算因子权重 — 基于IC和分桶相关性

        权重公式:
        - IC与分桶方向一致: w = sign(IC) * |IC|^0.5 * (|bucket_net|^0.5 + 0.5)
        - IC与分桶方向矛盾: w = sign(IC) * |IC|^0.5 * 0.3  (大幅削弱)
        - 无分桶数据:       w = sign(IC) * |IC|^0.5           (纯IC权重)
        """
        ic_data = self.ic_monitor.ic_data

        for f in self.available_factors:
            if f not in ic_data:
                continue

            # T+1 IC (最相关的预测周期)
            ic_tn1 = ic_data[f].get('tn_1', np.nan)
            if pd.isna(ic_tn1) or abs(ic_tn1) < 0.01:
                continue

            # 分桶相关性 — 净方向信号
            bucket_weight = 0.0
            if not self.bucket_corr_df.empty and f in self.bucket_corr_df.index:
                sr = self.bucket_corr_df.loc[f, 'strong_rise'] if 'strong_rise' in self.bucket_corr_df.columns else 0.0
                mr = self.bucket_corr_df.loc[f, 'moderate_rise'] if 'moderate_rise' in self.bucket_corr_df.columns else 0.0
                sf = self.bucket_corr_df.loc[f, 'strong_fall'] if 'strong_fall' in self.bucket_corr_df.columns else 0.0
                mf = self.bucket_corr_df.loc[f, 'moderate_fall'] if 'moderate_fall' in self.bucket_corr_df.columns else 0.0
                bucket_weight = (
                    ((sr if not pd.isna(sr) else 0) + (mr if not pd.isna(mr) else 0)) -
                    ((sf if not pd.isna(sf) else 0) + (mf if not pd.isna(mf) else 0))
                )

            # IC方向与分桶方向一致性检验
            if bucket_weight == 0.0:
                # 无分桶数据，纯IC权重
                self.factor_weights[f] = np.sign(ic_tn1) * (abs(ic_tn1) ** 0.5)
            elif np.sign(ic_tn1) * np.sign(bucket_weight) > 0:
                # 方向一致，增强权重
                self.factor_weights[f] = np.sign(ic_tn1) * (abs(ic_tn1) ** 0.5) * (abs(bucket_weight) ** 0.5 + 0.5)
            else:
                # 方向矛盾，大幅削弱
                self.factor_weights[f] = np.sign(ic_tn1) * (abs(ic_tn1) ** 0.5) * 0.3

    def score_stocks(self, factor_df, pool_mgr=None):
        """
        对股票池中的股票进行评分

        Parameters
        ----------
        factor_df : DataFrame — 因子数据 (含code, date, close, 各因子列)
        pool_mgr : StockPoolManager — 股票池管理器 (可选，用于标记活跃池股票)

        Returns
        -------
        DataFrame — 评分结果 (code, date, score, classification, buy_price, stop_loss, target_price, ...)
        """
        if factor_df.empty or not self.factor_weights:
            print("  [跳过评分] 无因子数据或无有效权重")
            return pd.DataFrame()

        factor_df = factor_df.copy()

        # 每只股票取最新日期的因子记录
        # ★ v1.3修复: date列可能为字符串类型，idxmax()在pandas groupby中对字符串列会报TypeError
        #   解决方案: 先转为datetime再取idxmax
        if factor_df['date'].dtype == object or str(factor_df['date'].dtype) == 'string':
            factor_df['_date_dt'] = pd.to_datetime(factor_df['date'], errors='coerce')
            latest_idx = factor_df.groupby('code')['_date_dt'].idxmax()
            factor_df = factor_df.drop(columns=['_date_dt'])
        else:
            latest_idx = factor_df.groupby('code')['date'].idxmax()
        latest_factors = factor_df.loc[latest_idx].copy()

        # 标记活跃池股票 (最近5天内涨停且未过期)
        if pool_mgr is not None and hasattr(pool_mgr, 'pool') and len(pool_mgr.pool) > 0:
            pool_codes = set(pool_mgr.pool.keys())
            latest_factors['in_active_pool'] = latest_factors['code'].isin(pool_codes)
        else:
            latest_factors['in_active_pool'] = False

        if len(latest_factors) == 0:
            print("  [跳过评分] 无可评分股票")
            return pd.DataFrame()

        # ── 向量化评分: z-score标准化 + 加权求和 ──
        weight_factors = list(self.factor_weights.keys())
        score_components = []
        valid_counts = []
        zscore_data = {}

        for f in weight_factors:
            if f not in latest_factors.columns:
                continue
            vals = latest_factors[f].astype(float)
            mean_val = vals.mean()
            std_val = vals.std()
            if pd.isna(std_val) or std_val < 1e-10:
                continue
            z_series = (vals - mean_val) / std_val
            zscore_data[f] = z_series
            is_valid = vals.notna().astype(float)
            z_filled = z_series.fillna(0.0)
            weight = self.factor_weights[f]
            score_components.append(weight * z_filled)
            valid_counts.append(is_valid)

        if not score_components:
            print("  [跳过评分] 无有效因子z-score数据")
            return pd.DataFrame()

        # ★ v1.3修复: Python内置sum()从0开始累加，0+Series在旧版pandas可能返回标量
        #   改用显式Series累加，避免类型坍缩
        score_arr = score_components[0]
        for sc in score_components[1:]:
            score_arr = score_arr.add(sc, fill_value=0)
        contributing_arr = valid_counts[0]
        for vc in valid_counts[1:]:
            contributing_arr = contributing_arr.add(vc, fill_value=0)
        # 归一化: 除以贡献因子数的平方根，避免因子多的股票得分偏高
        score_arr = score_arr / np.maximum(contributing_arr ** 0.5, 1.0)

        # 保存用于贡献分析
        self._latest_factors = latest_factors
        self._zscore_data = zscore_data

        # 构建结果DataFrame
        result_cols = ['code', 'date', 'close']
        avail_result_cols = [c for c in result_cols if c in latest_factors.columns]
        result = latest_factors[avail_result_cols].copy()
        if 'atr_14' in latest_factors.columns:
            result['atr_14'] = latest_factors['atr_14'].values
        result['in_active_pool'] = latest_factors['in_active_pool'].values
        result['raw_score'] = score_arr.values
        result['contributing_factors'] = contributing_arr.astype(int).values

        # ★ v1.3: 添加中文股票名称列
        try:
            code_name_map = {}
            for code in result['code'].unique():
                try:
                    info = get_security_info(code)
                    code_name_map[code] = info.display_name if info else ''
                except Exception:
                    code_name_map[code] = ''
            result['stock_name'] = result['code'].map(code_name_map).fillna('')
        except Exception:
            result['stock_name'] = ''

        # 标准化得分到 [0, 100]
        # ★ v1.3优化: 改用百分位排名评分，避免大量股票聚集在100分
        #   旧方法: z-score标准化+clip → 多只股票超出3σ被截断为100
        #   新方法: rank(pct=True)*100 → 均匀分布在[0,100]，无聚集
        if result['raw_score'].std() > 1e-10:
            result['score'] = result['raw_score'].rank(pct=True) * 100
        else:
            result['score'] = 50.0

        # ── 分类: 强势 / 弱势 / 淘汰 ──
        n_stocks = len(result)
        if n_stocks >= 10:
            strong_th = np.percentile(result['score'], self.STRONG_PERCENTILE)
            weak_th = np.percentile(result['score'], self.WEAK_PERCENTILE)
        elif n_stocks >= 5:
            strong_th = 65.0
            weak_th = 35.0
        else:
            strong_th = 60.0
            weak_th = 40.0

        def classify(score):
            if score >= strong_th:
                return '强势'
            elif score <= weak_th:
                return '淘汰'
            else:
                return '弱势'

        result['classification'] = result['score'].apply(classify)

        # ── 计算交易信号 ──
        result = self._compute_trading_signals(result)

        self.scores_df = result
        return result

    def _compute_trading_signals(self, scores_df):
        """
        计算买入价、止损价、目标价

        止损逻辑: ATR优先 → 固定比例兜底
        目标逻辑: 盈亏比 (强势3:1, 弱势2:1, 淘汰1.5:1)
        """
        buy_prices = []
        stop_losses = []
        targets = []

        for idx, row in scores_df.iterrows():
            close = row.get('close', np.nan)
            atr = row.get('atr_14', np.nan)
            classification = row.get('classification', '弱势')

            if pd.isna(close) or close <= 0:
                buy_prices.append(np.nan)
                stop_losses.append(np.nan)
                targets.append(np.nan)
                continue

            buy_price = close  # 买入价 = 最新收盘价

            # 止损: ATR优先，否则固定比例
            if not pd.isna(atr) and atr > 0:
                atr_pct = atr / close
                stop_pct = max(self.MIN_STOP_LOSS_PCT, min(self.MAX_STOP_LOSS_PCT, atr_pct))
            else:
                if classification == '强势':
                    stop_pct = 0.06
                elif classification == '弱势':
                    stop_pct = self.DEFAULT_STOP_LOSS_PCT
                else:
                    stop_pct = self.MIN_STOP_LOSS_PCT

            stop_loss = buy_price * (1 - stop_pct)

            # 目标价: 盈亏比
            risk = buy_price - stop_loss
            if classification == '强势':
                rr = 3.0
            elif classification == '弱势':
                rr = self.TARGET_REWARD_RATIO
            else:
                rr = 1.5

            target = buy_price + risk * rr

            buy_prices.append(round(buy_price, 2))
            stop_losses.append(round(stop_loss, 2))
            targets.append(round(target, 2))

        scores_df['buy_price'] = buy_prices
        scores_df['stop_loss'] = stop_losses
        scores_df['target_price'] = targets

        return scores_df

    def get_top_n(self, n=5):
        """获取得分最高的N只股票"""
        if self.scores_df.empty:
            return pd.DataFrame()
        return self.scores_df.nlargest(n, 'score')

    def get_top_factor_contributors(self, code, top_n=3):
        """获取指定股票得分贡献最大的因子"""
        if self._latest_factors is None or self._zscore_data is None:
            return []

        row_idx = self._latest_factors[self._latest_factors['code'] == code].index
        if len(row_idx) == 0:
            return []
        idx = row_idx[0]

        contributions = []
        for f, z_series in self._zscore_data.items():
            if idx not in z_series.index:
                continue
            z_val = z_series.loc[idx]
            if pd.isna(z_val):
                continue
            weight = self.factor_weights.get(f, 0)
            contrib = weight * z_val
            contributions.append((f, contrib, z_val, weight))

        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:top_n]

    def get_classification_summary(self):
        """获取分类汇总统计"""
        if self.scores_df.empty:
            return pd.DataFrame()
        # ★ v1.3修复: 命名聚合语法agg(count=...)需要pandas>=0.25，旧版不支持
        #   改用字典聚合+重命名列
        summary = self.scores_df.groupby('classification').agg({
            'code': 'count',
            'score': 'mean',
            'close': 'mean'
        })
        summary.columns = ['count', 'avg_score', 'avg_close']
        return summary.round(2)

    def print_report(self, top_n=5):
        """打印评分报告"""
        if self.scores_df.empty:
            print("  无评分数据")
            return

        print(f"\n{'='*70}")
        print("股票池评分报告 — 基于因子-收益相关性分析 (下个交易日预测)")
        print(f"{'='*70}")

        # ── 因子权重信息 ──
        print(f"\n  有效因子权重数: {len(self.factor_weights)}")
        if self.factor_weights:
            sorted_weights = sorted(self.factor_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            print(f"  权重Top 10因子 (weight = IC方向 × |IC|^0.5 × 分桶一致性):")
            for f, w in sorted_weights[:10]:
                ic_val = self.ic_monitor.ic_data.get(f, {}).get('tn_1', np.nan)
                ic_str = f"{ic_val:.4f}" if not pd.isna(ic_val) else "N/A"
                print(f"    {f:30s}: weight={w:+.4f}  IC(tn_1)={ic_str}")

        # ── 评分统计 ──
        print(f"\n  评分统计:")
        print(f"    评分股票数: {len(self.scores_df)}")
        print(f"    得分范围: [{self.scores_df['score'].min():.1f}, {self.scores_df['score'].max():.1f}]")
        print(f"    得分均值: {self.scores_df['score'].mean():.1f}")
        active = self.scores_df[self.scores_df['in_active_pool']] if 'in_active_pool' in self.scores_df.columns else pd.DataFrame()
        print(f"    活跃池中股票数: {len(active)}")

        # ── 分类汇总 ──
        summary = self.get_classification_summary()
        if not summary.empty:
            print(f"\n  分类汇总:")
            _display(summary)

        # ── Top N 强势股 ──
        top = self.get_top_n(top_n)
        if not top.empty:
            display_cols = ['code', 'stock_name', 'date', 'score', 'classification', 'close', 'buy_price', 'stop_loss', 'target_price', 'contributing_factors']
            avail_cols = [c for c in display_cols if c in top.columns]
            print(f"\n  ★ Top {top_n} 强势股 (下个交易日推荐关注):")
            _display(top[avail_cols].round(2))

            # 详细交易信号 + 因子贡献
            print(f"\n  交易信号详情:")
            for _, row in top.iterrows():
                code = row['code']
                stock_name = row.get('stock_name', '')
                name_label = f"{stock_name}" if stock_name else code
                score_val = row.get('score', 0)
                cls = row.get('classification', '')
                buy = row.get('buy_price', np.nan)
                sl = row.get('stop_loss', np.nan)
                tgt = row.get('target_price', np.nan)

                if not pd.isna(buy) and not pd.isna(sl) and not pd.isna(tgt):
                    risk_pct = (buy - sl) / buy * 100 if buy > 0 else 0
                    reward_pct = (tgt - buy) / buy * 100 if buy > 0 else 0
                    rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                    print(f"    {name_label}({code}): 得分={score_val:.1f} [{cls}]")
                    print(f"      买入={buy:.2f}  止损={sl:.2f}(-{risk_pct:.1f}%)  目标={tgt:.2f}(+{reward_pct:.1f}%)  盈亏比={rr_ratio:.1f}:1")

                # Top 3 因子贡献 (含中文名)
                contribs = self.get_top_factor_contributors(code, top_n=3)
                if contribs:
                    contrib_str = " | ".join(
                        f"{f}({FACTOR_NAME_CN.get(f, '')})({c:+.3f})" if FACTOR_NAME_CN.get(f, '') else f"{f}({c:+.3f})"
                        for f, c, z, w in contribs
                    )
                    print(f"      关键因子: {contrib_str}")

        # ── 全部分类列表 ──
        strong = self.scores_df[self.scores_df['classification'] == '强势'].sort_values('score', ascending=False)
        weak = self.scores_df[self.scores_df['classification'] == '弱势'].sort_values('score', ascending=False)
        eliminated = self.scores_df[self.scores_df['classification'] == '淘汰'].sort_values('score', ascending=False)

        if not strong.empty:
            print(f"\n  强势股列表 ({len(strong)}只):")
            cols = ['code', 'stock_name', 'date', 'score', 'close', 'buy_price', 'stop_loss', 'target_price']
            avail = [c for c in cols if c in strong.columns]
            _display(strong[avail].round(2))

        if not weak.empty:
            print(f"\n  弱势股列表 ({len(weak)}只):")
            cols = ['code', 'stock_name', 'date', 'score', 'close']
            avail = [c for c in cols if c in weak.columns]
            _display(weak[avail].round(2))

        if not eliminated.empty:
            print(f"\n  淘汰股列表 ({len(eliminated)}只):")
            cols = ['code', 'stock_name', 'date', 'score', 'close']
            avail = [c for c in cols if c in eliminated.columns]
            _display(eliminated[avail].round(2))


# ============================================================
# PART 9: 输出与报告
# ============================================================

def print_section_header(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def print_daily_summary(date, zt_count, pool_size, new_count, total_factors):
    """打印每日扫描摘要"""
    print(f"  [{date}] 涨停:{zt_count}只 | 池中:{pool_size}只 | 新增:{new_count}只 | 因子记录:{total_factors}条")


def print_final_reports(factor_df, ic_monitor, collinearity_det, tn_tracker,
                        bucket_corr_df, positive_factors, missing_report,
                        stock_scorer=None):
    """打印最终分析报告"""
    print_section_header('PART A: 因子缺失率报告')
    if missing_report is not None and not missing_report.empty:
        high_missing = missing_report[missing_report['缺失率'] > MISSING_THRESHOLD]
        if len(high_missing) > 0:
            print(f"  以下 {len(high_missing)} 个因子缺失率超过 {MISSING_THRESHOLD:.0%}，建议关注:")
            _display(high_missing.sort_values('缺失率', ascending=False).head(30))
        else:
            print(f"  所有因子缺失率均低于 {MISSING_THRESHOLD:.0%}")
        low_missing = missing_report[missing_report['缺失率'] <= MISSING_THRESHOLD]
        print(f"  可用因子: {len(low_missing)}/{len(missing_report)} (缺失率≤{MISSING_THRESHOLD:.0%})")
    else:
        print("  无缺失率数据")

    print_section_header('PART B: 因子IC衰减报告')
    ic_monitor.print_report()

    print_section_header('PART C: 因子共线性报告')
    collinearity_det.print_report()

    print_section_header('PART D: T+N涨跌统计')
    agg_stats = tn_tracker.get_aggregated_stats()
    if not agg_stats.empty:
        print("  T+N收益汇总:")
        _display(agg_stats.round(4))
    else:
        print("  无T+N收益数据")

    enc_dist = tn_tracker.get_encoded_distribution()
    if not enc_dist.empty:
        print("\n  T+N编码分布 (0=下跌, 1=上涨<9.9%, 2=上涨≥9.9%):")
        _display(enc_dist)
    else:
        print("  无T+N编码分布数据")

    # T+N详细记录 (前20条)
    tn_df = tn_tracker.get_results_df()
    if not tn_df.empty:
        # 统计NaN数量
        tn_cols = [c for c in tn_df.columns if c.startswith('tn_') and not c.endswith('_encoded')]
        total_cells = len(tn_df) * len(tn_cols)
        nan_cells = tn_df[tn_cols].isna().sum().sum() if tn_cols else 0
        nan_pct = nan_cells / total_cells * 100 if total_cells > 0 else 0
        print(f"\n  T+N详细记录 (共{len(tn_df)}条，展示前20条):")
        if nan_cells > 0:
            print(f"  ⚠ T+N收益中存在NaN值: {nan_cells}/{total_cells} ({nan_pct:.1f}%) — 原因: 入场日期接近分析期末，未来交易日数据尚不可用")
        _display(tn_df.head(20).round(4))
    else:
        print("  无T+N详细记录")

    print_section_header('PART E: 因子-收益分桶相关性矩阵')
    if not bucket_corr_df.empty:
        print("  因子与T+5收益分桶的Spearman相关系数:")
        # ★ v1.3: 将因子英文名替换为"英文名(中文名)"格式，直接在表格中显示
        display_df = bucket_corr_df.round(4).head(30).copy()
        new_index = []
        for f in display_df.index:
            cn = FACTOR_NAME_CN.get(f, '')
            new_index.append(f"{f}({cn})" if cn else f)
        display_df.index = new_index
        _display(display_df)
    else:
        print("  无分桶相关性数据")

    print_section_header('PART F: 正相关因子识别')
    if positive_factors:
        for bucket, factors in positive_factors.items():
            if len(factors) > 0:
                print(f"\n  与 [{bucket}] 正相关的因子 (corr>0.03):")
                for fname, fval in factors.items():
                    cn = FACTOR_NAME_CN.get(fname, '')
                    label = f"{fname}({cn})" if cn else fname
                    print(f"    {label}: {fval:.4f}")
    else:
        print("  无正相关因子数据")

    print_section_header('PART G: 因子数据概览')
    if not factor_df.empty:
        print(f"  总记录数: {len(factor_df)}")
        print(f"  股票数: {factor_df['code'].nunique() if 'code' in factor_df.columns else 'N/A'}")
        print(f"  日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}" if 'date' in factor_df.columns else "")

        # ★ v1.2: 添加中文股票名称
        if 'code' in factor_df.columns:
            try:
                unique_codes = factor_df['code'].unique()[:30]  # 限制查询数量
                code_name_map = {}
                for code in unique_codes:
                    try:
                        info = get_security_info(code)
                        code_name_map[code] = info.display_name if info else ''
                    except Exception:
                        code_name_map[code] = ''
                # 在展示数据中添加股票名称列
                factor_df_display = factor_df.copy()
                factor_df_display['stock_name'] = factor_df_display['code'].map(code_name_map)
                # 将NaN名称填充为空字符串
                factor_df_display['stock_name'] = factor_df_display['stock_name'].fillna('')
            except Exception:
                factor_df_display = factor_df.copy()
                factor_df_display['stock_name'] = ''
        else:
            factor_df_display = factor_df.copy()

        # ★ v1.2: 标注分钟数据依赖因子的NaN原因
        nan_factors = []
        for col in MINUTE_DEPENDENT_FACTORS:
            if col in factor_df.columns:
                nan_rate = factor_df[col].isna().mean()
                if nan_rate > 0.5:
                    cn = FACTOR_NAME_CN.get(col, '')
                    nan_factors.append(f"{col}({cn})" if cn else col)
        if nan_factors and not USE_MINUTE_DATA:
            print(f"\n  ⚠ 以下因子因USE_MINUTE_DATA=False而全为NaN (需分钟数据):")
            for nf in nan_factors:
                print(f"    {nf}")

        # 展示前10条因子记录(含股票名称)
        display_cols = ['code', 'stock_name', 'date', 'close'] + [c for c in ALL_FACTORS[:10] if c in factor_df_display.columns]
        avail_cols = [c for c in display_cols if c in factor_df_display.columns]
        if avail_cols:
            print(f"\n  因子数据样例 (前10条, {len(avail_cols)}列):")
            _display(factor_df_display[avail_cols].head(10).round(4))
    else:
        print("  无因子数据")

    # ── PART H: 股票池评分报告 ──
    if stock_scorer is not None and not stock_scorer.scores_df.empty:
        print_section_header('PART H: 股票池评分 & 交易信号')
        stock_scorer.print_report(top_n=5)
    else:
        print_section_header('PART H: 股票池评分 & 交易信号')
        print("  无评分数据 (StockScorer未运行或股票池为空)")

    print('\n' + '=' * 70)
    print('分析完成！本程序仅供研究参考，不构成投资建议。')
    print('=' * 70)


def main(reference_date=None):
    """
    主函数 — 涨停板后45日数据分析
    reference_date: 参考日期T (默认今天), 分析区间 = [T-45, T-1]
    """
    t0 = _time.time()

    # ── Step 0: 确定分析日期范围 ──
    if reference_date is None:
        reference_date = _dt.date.today()
    elif isinstance(reference_date, str):
        reference_date = pd.Timestamp(reference_date).date()

    all_trade_days = get_trade_days(end_date=reference_date, count=LOOKBACK_DAYS + 10)
    # ★ 修复: get_trade_days 返回 numpy array，用 len() 而非 truth value 判断
    if len(all_trade_days) == 0:
        print("[ERROR] 无法获取交易日列表")
        return None

    # 分析区间: 最近45个交易日 (不含T日本身)
    trade_days = all_trade_days[-LOOKBACK_DAYS:]
    lookback_start = trade_days[0]
    lookback_end = trade_days[-1]

    print_section_header('Step 0: 分析参数')
    print(f"  参考日期T: {reference_date}")
    print(f"  分析区间: {lookback_start} ~ {lookback_end} ({len(trade_days)}个交易日)")
    print(f"  因子总数: {len(ALL_FACTORS)}个")
    print(f"  T+N追踪: {TRACK_DAYS}天")

    # ── Step 1: 初始化组件 ──
    pool_mgr = StockPoolManager()
    engine = FactorEngine()
    ic_monitor = ICDdecayMonitor()
    collinearity_det = CollinearityDetector()
    tn_tracker = TNTracker()

    # ── Step 2: 逐日扫描涨停股票 ──
    print_section_header('Step 2: 逐日扫描涨停股票 (45天)')
    all_zt_events = []  # 用于T+N追踪: [{stock, date, close}, ...]
    daily_zt_map = {}   # {date: [stock_codes]}

    for i, date in enumerate(trade_days):
        day_start = _time.time()

        # 2a. 获取当日涨停股票
        zt_stocks = get_zt_stocks_on_date(date)

        # 2b. 过滤: 排除ST/上市不足45天/一字板
        filtered = filter_zt_pool(zt_stocks, date)
        daily_zt_map[date] = filtered

        # 2c. 更新股票池
        pool_mgr.add_stocks(filtered, date)

        # ★ 性能优化: 批量预加载当日涨停股的价格数据 + 行业涨停计数
        if len(filtered) > 0:
            engine.preload_day_prices(filtered, date)
            engine.precompute_day_industry_zt(filtered, date)

        # 2d. 为当日涨停股计算因子
        new_count = 0
        for stock in filtered:
            factors = engine.compute_all(stock, date, filtered, pool_mgr)
            if factors is not None:
                new_count += 1
                # 收集T+N事件
                close_val = factors.get('close', np.nan)
                if not pd.isna(close_val) and close_val > 0:
                    all_zt_events.append({
                        'stock': stock,
                        'date': date,
                        'close': close_val,
                    })

        pool_mgr.tick(date)

        day_elapsed = _time.time() - day_start
        print_daily_summary(date, len(zt_stocks), len(pool_mgr.pool),
                            new_count, len(engine.factor_records))

    step2_time = _time.time() - t0
    print(f"\n  Step 2 完成, 耗时: {step2_time:.1f}s, 累计因子记录: {len(engine.factor_records)}条")

    # ── Step 3: 构建因子DataFrame ──
    print_section_header('Step 3: 构建因子DataFrame')
    if len(engine.factor_records) == 0:
        print("[ERROR] 无因子记录，分析终止")
        return None

    factor_df = pd.DataFrame(engine.factor_records)

    # 计算因子缺失率
    missing_report_data = []
    for col in ALL_FACTORS:
        if col in factor_df.columns:
            missing_rate = factor_df[col].isna().mean()
            missing_report_data.append({'因子': col, '缺失率': missing_rate, '有效数': factor_df[col].notna().sum()})
        else:
            missing_report_data.append({'因子': col, '缺失率': 1.0, '有效数': 0})
    missing_report = pd.DataFrame(missing_report_data)

    # 识别可用因子 (缺失率低于阈值)
    # ★ v1修复: 排除分钟数据依赖因子（当USE_MINUTE_DATA=False时）
    available_factors = missing_report[missing_report['缺失率'] <= MISSING_THRESHOLD]['因子'].tolist()
    if not USE_MINUTE_DATA:
        minute_excluded = [f for f in available_factors if f in MINUTE_DEPENDENT_FACTORS]
        available_factors = [f for f in available_factors if f not in MINUTE_DEPENDENT_FACTORS]
        if minute_excluded:
            print(f"  [v1] 排除分钟数据依赖因子: {minute_excluded}")
    print(f"  总因子数: {len(ALL_FACTORS)}")
    print(f"  可用因子数 (缺失率≤{MISSING_THRESHOLD:.0%}): {len(available_factors)}")
    print(f"  高缺失因子数 (缺失率>{MISSING_THRESHOLD:.0%}): {len(ALL_FACTORS) - len(available_factors)}")

    # ── Step 4: T+N收益计算 ──
    # ★ v1关键修复: 将T+N计算移到IC衰减之前！
    # v0的Bug: Step 4(IC)在Step 6(T+N)之前，导致IC计算时factor_df中没有T+N列
    print_section_header('Step 4: T+N收益计算')
    if len(all_zt_events) > 0:
        # 获取完整交易日列表 (含追踪期)
        extended_end = pd.Timestamp(reference_date) + pd.Timedelta(days=30)
        full_trade_days = get_trade_days(end_date=extended_end, count=LOOKBACK_DAYS + 50)
        tn_tracker.compute_tn_returns(all_zt_events, full_trade_days, lookback_start)
        print(f"  T+N事件数: {len(all_zt_events)}, 计算记录: {len(tn_tracker.records)}")
    else:
        print("  [跳过] 无涨停事件")

    # ── Step 5: 合并T+N收益 + IC衰减监控 ──
    # ★ v1: 先合并T+N到factor_df，再计算IC（确保IC有收益数据可用）
    print_section_header('Step 5: 合并T+N收益 & IC衰减监控')
    bucket_corr_df = pd.DataFrame()
    positive_factors = {}

    # 合并T+N收益到factor_df
    if len(tn_tracker.records) > 0:
        tn_df = tn_tracker.get_results_df()
        if not tn_df.empty and not factor_df.empty:
            # 将T+N收益合并到因子数据
            for n in range(1, TRACK_DAYS + 1):
                col = f'tn_{n}'
                if col in tn_df.columns:
                    merge_cols = ['stock', 'entry_date', col]
                    avail_merge = [c for c in merge_cols if c in tn_df.columns]
                    if len(avail_merge) >= 3:
                        tn_merge = tn_df[avail_merge].copy()
                        tn_merge = tn_merge.rename(columns={'stock': 'code', 'entry_date': 'date'})
                        tn_merge = tn_merge.drop_duplicates(subset=['code', 'date'], keep='first')
                        tn_merge['date'] = tn_merge['date'].astype(str)
                        factor_df['date'] = factor_df['date'].astype(str)
                        factor_df = factor_df.merge(tn_merge, on=['code', 'date'], how='left', suffixes=('', '_tn'))
            print(f"  T+N收益已合并到factor_df, 列数: {len(factor_df.columns)}")

    # IC衰减监控 — 现在factor_df中已有T+N列
    return_cols = {}
    for n in range(1, TRACK_DAYS + 1):
        col = f'tn_{n}'
        if col in factor_df.columns:
            return_cols[col] = factor_df[col]

    if return_cols:
        return_df = factor_df[['code', 'date'] + list(return_cols.keys())].copy()
    else:
        return_df = pd.DataFrame()

    if len(available_factors) > 0 and not return_df.empty:
        avail_factor_df = factor_df[available_factors].copy()
        ic_monitor.compute_all_ic(avail_factor_df, return_df, factor_names=available_factors)
        print(f"  IC计算完成: {len(available_factors)}个因子 x {len(return_df)}条记录")
    else:
        print("  [跳过IC] 缺少因子或收益数据")

    # ── Step 6: 共线性检测 ──
    print_section_header('Step 6: 共线性检测')
    if len(available_factors) > 1:
        avail_factor_df = factor_df[available_factors].copy()
        collinearity_det.compute_corr(avail_factor_df, factor_names=available_factors)
        # VIF计算 (可能较慢，限制因子数)
        vif_factors = available_factors[:50]  # 最多50个因子计算VIF
        collinearity_det.compute_vif(avail_factor_df, factor_names=vif_factors)
        print(f"  共线性检测完成: {len(available_factors)}个因子")
    else:
        print("  [跳过] 可用因子不足")

    # ── Step 7: 因子-收益分桶相关性 ──
    print_section_header('Step 7: 因子-收益分桶相关性')
    if len(tn_tracker.records) > 0 and not factor_df.empty:
        # 构建收益DataFrame (T+N已在Step 5合并)
        return_df_merged = factor_df[['code', 'date'] + [f'tn_{n}' for n in range(1, TRACK_DAYS + 1)
                                                          if f'tn_{n}' in factor_df.columns]].copy()

        # 分桶相关性
        avail_factor_cols = [c for c in available_factors if c in factor_df.columns]
        if avail_factor_cols and 'tn_5' in return_df_merged.columns:
            bucket_corr_df = build_bucket_correlation_matrix(
                factor_df[avail_factor_cols], return_df_merged)
            if not bucket_corr_df.empty:
                positive_factors = identify_positive_factors(bucket_corr_df)
                print(f"  分桶相关性计算完成: {len(avail_factor_cols)}个因子")
        else:
            print("  [跳过] 缺少T+5收益数据或可用因子")
    else:
        print("  [跳过] 无T+N记录或因子数据")

    # ── Step 8: 股票池评分 ──
    print_section_header('Step 8: 股票池评分 & 分类')
    stock_scorer = None
    if len(available_factors) > 0 and not factor_df.empty:
        try:
            # ★ v1.3: bucket_corr_df为空时仍可评分(使用纯IC权重)
            if bucket_corr_df.empty:
                print("  [提示] 分桶相关性为空，将使用纯IC权重进行评分")
            stock_scorer = StockScorer(ic_monitor, bucket_corr_df, positive_factors, available_factors)
            if not stock_scorer.factor_weights:
                print("  [警告] 无有效因子权重 — 尝试降低IC阈值...")
                # ★ v1.3: 降低IC阈值重试
                stock_scorer_low = StockScorer.__new__(StockScorer)
                stock_scorer_low.ic_monitor = ic_monitor
                stock_scorer_low.bucket_corr_df = bucket_corr_df
                stock_scorer_low.positive_factors = positive_factors
                stock_scorer_low.available_factors = available_factors
                stock_scorer_low.scores_df = pd.DataFrame()
                stock_scorer_low._latest_factors = None
                stock_scorer_low._zscore_data = None
                stock_scorer_low.factor_weights = {}
                # 使用更低的IC阈值
                ic_data = ic_monitor.ic_data
                for f in available_factors:
                    if f not in ic_data:
                        continue
                    ic_tn1 = ic_data[f].get('tn_1', np.nan)
                    if pd.isna(ic_tn1) or abs(ic_tn1) < 0.001:
                        continue
                    # 纯IC权重(无分桶验证)
                    stock_scorer_low.factor_weights[f] = np.sign(ic_tn1) * (abs(ic_tn1) ** 0.5)
                if stock_scorer_low.factor_weights:
                    print(f"  [降级] 使用IC阈值0.001, 获得{len(stock_scorer_low.factor_weights)}个因子权重")
                    stock_scorer = stock_scorer_low
                else:
                    print("  [跳过] 即使降低IC阈值仍无有效因子权重")
                    stock_scorer = None

            if stock_scorer is not None:
                stock_scorer.score_stocks(factor_df, pool_mgr)
                if not stock_scorer.scores_df.empty:
                    print(f"  评分完成: {len(stock_scorer.scores_df)}只股票")
                    summary = stock_scorer.get_classification_summary()
                    if not summary.empty:
                        print("  分类汇总:")
                        _display(summary)
                else:
                    print("  [警告] 评分结果为空 (可能股票池中无有效因子数据)")
                    stock_scorer = None
        except Exception as e:
            import traceback as _tb
            print(f"  [ERROR] StockScorer执行失败: {e}")
            _tb.print_exc()
            stock_scorer = None
    else:
        print("  [跳过] 缺少可用因子或因子数据")

    # ── Step 9: 输出报告 ──
    print_section_header('Step 9: 输出报告')
    print_final_reports(factor_df, ic_monitor, collinearity_det, tn_tracker,
                        bucket_corr_df, positive_factors, missing_report,
                        stock_scorer=stock_scorer)

    total_time = _time.time() - t0
    print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

    return {
        'factor_df': factor_df,
        'ic_monitor': ic_monitor,
        'collinearity_det': collinearity_det,
        'tn_tracker': tn_tracker,
        'bucket_corr_df': bucket_corr_df,
        'positive_factors': positive_factors,
        'missing_report': missing_report,
        'available_factors': available_factors,
        'stock_scorer': stock_scorer,
    }


# ============================================================
# 执行入口
# ============================================================
print("\n[READY] 执行 results = main() 开始分析")
print("  可选: main(reference_date='2025-01-15') 指定参考日期")
print()

results = main()
