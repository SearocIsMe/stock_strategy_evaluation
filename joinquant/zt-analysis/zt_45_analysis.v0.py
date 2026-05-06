# -*- coding: utf-8 -*-
"""
zt_45_analysis.v0.py - 涨停板后45日数据分析程序
运行环境: JoinQuant Research Notebook (https://www.joinquant.com/research)

功能概述:
1. 构建近45交易日涨停股票池（排除ST/上市不足45天/一字板）
2. 多维度因子计算（90+因子，含涨停特征因子和Alpha因子）
3. IC衰减监控 & 因子共线性检测（基础设施优先）
4. T+N (N=1~5) 涨跌统计（编码: 0=下跌, 1=上涨<9.9%, 2=上涨≥9.9%）
5. 因子-收益分桶相关性矩阵
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

LOOKBACK_DAYS = 45          # 回溯交易日数
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

print("=" * 70)
print("涨停板后45日数据分析程序 zt_45_analysis.v0")
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
            try:
                # 注意: turnover_ratio 在 valuation 表，不在 indicator 表
                # debt_asset_ratio 是 JQ 正确字段名，不是 debt_ratio
                q = query(
                    valuation.code,
                    valuation.pe_ratio, valuation.pb_ratio,
                    valuation.ps_ratio, valuation.pcf_ratio,
                    valuation.market_cap, valuation.circulating_market_cap,
                    valuation.capitalization, valuation.circulating_cap,
                    valuation.turnover_ratio,
                    indicator.roe, indicator.roa,
                    indicator.net_profit_margin, indicator.gross_profit_margin,
                    indicator.inc_revenue_year, indicator.inc_net_profit_year,
                    indicator.debt_asset_ratio,
                ).filter(valuation.code.in_(uncached))
                df = get_fundamentals(q, date=date)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        self._fund_cache[key][row['code']] = row.to_dict()
            except:
                pass
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
        try:
            # 尝试从缓存获取近5日/10日的换手率
            tr_list = []
            for offset in range(1, 12):
                try:
                    offset_date = pd.Timestamp(date) - pd.Timedelta(days=offset * 2)
                    fund_key = str(offset_date.date())
                    if fund_key in self._fund_cache and stock in self._fund_cache[fund_key]:
                        tr_val = self._fund_cache[fund_key][stock].get('turnover_ratio', np.nan)
                        if not pd.isna(tr_val):
                            tr_list.append((offset, tr_val))
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
        except:
            pass
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
            'revenue_growth': 'inc_revenue_year',
            'profit_growth': 'inc_net_profit_year',
            'debt_asset_ratio': 'debt_asset_ratio',  # JQ正确字段名
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
        except ImportError:
            print("  [跳过VIF] statsmodels未安装")
            return {}

        if factor_names is None:
            factor_names = [c for c in factor_df.columns if c in ALL_FACTORS]
        avail = [f for f in factor_names if f in factor_df.columns]
        df = factor_df[avail].dropna()
        if df.empty or len(df) < len(avail) + 1:
            return {}

        self.vif_dict = {}
        for i, f in enumerate(avail):
            try:
                vif = variance_inflation_factor(df.values, i)
                self.vif_dict[f] = min(vif, 999)
            except:
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
                    print(f"  {f1:30s} <-> {f2:30s}  ρ={v:.4f}")
            else:
                print("  未发现高相关因子对")

        if self.vif_dict:
            print(f"\nVIF分析 (阈值={VIF_THRESHOLD}):")
            sorted_vif = sorted(self.vif_dict.items(),
                                key=lambda x: x[1] if not pd.isna(x[1]) else 0,
                                reverse=True)
            for f, v in sorted_vif[:20]:
                flag = " ⚠️" if v > VIF_THRESHOLD else ""
                print(f"  {f:30s} VIF={v:.2f}{flag}")

        removal = self.suggest_removal()
        if removal:
            print(f"\n建议移除的冗余因子 ({len(removal)}个): {removal}")


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

    result = pd.DataFrame(bucket_corr, index=factor_cols)
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
                        bucket_corr_df, positive_factors, missing_report):
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
        print(f"\n  T+N详细记录 (共{len(tn_df)}条，展示前20条):")
        _display(tn_df.head(20).round(4))
    else:
        print("  无T+N详细记录")

    print_section_header('PART E: 因子-收益分桶相关性矩阵')
    if not bucket_corr_df.empty:
        print("  因子与T+5收益分桶的Spearman相关系数:")
        _display(bucket_corr_df.round(4).head(30))
    else:
        print("  无分桶相关性数据")

    print_section_header('PART F: 正相关因子识别')
    if positive_factors:
        for bucket, factors in positive_factors.items():
            if len(factors) > 0:
                print(f"\n  与 [{bucket}] 正相关的因子 (corr>0.03):")
                for fname, fval in factors.items():
                    print(f"    {fname}: {fval:.4f}")
    else:
        print("  无正相关因子数据")

    print_section_header('PART G: 因子数据概览')
    if not factor_df.empty:
        print(f"  总记录数: {len(factor_df)}")
        print(f"  股票数: {factor_df['code'].nunique() if 'code' in factor_df.columns else 'N/A'}")
        print(f"  日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}" if 'date' in factor_df.columns else "")
        # 展示前10条因子记录
        display_cols = ['code', 'date', 'close'] + [c for c in ALL_FACTORS[:10] if c in factor_df.columns]
        avail_cols = [c for c in display_cols if c in factor_df.columns]
        if avail_cols:
            print(f"\n  因子数据样例 (前10条, {len(avail_cols)}列):")
            _display(factor_df[avail_cols].head(10).round(4))
    else:
        print("  无因子数据")

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
    available_factors = missing_report[missing_report['缺失率'] <= MISSING_THRESHOLD]['因子'].tolist()
    print(f"  总因子数: {len(ALL_FACTORS)}")
    print(f"  可用因子数 (缺失率≤{MISSING_THRESHOLD:.0%}): {len(available_factors)}")
    print(f"  高缺失因子数 (缺失率>{MISSING_THRESHOLD:.0%}): {len(ALL_FACTORS) - len(available_factors)}")

    # ── Step 4: IC衰减监控 ──
    print_section_header('Step 4: IC衰减监控')
    # 构建收益DataFrame (T+N)
    # 先从因子记录中提取close和date，用于IC计算
    return_cols = {}
    for n in range(1, TRACK_DAYS + 1):
        col = f'tn_{n}'
        if col in factor_df.columns:
            return_cols[col] = factor_df[col]

    # 如果factor_df中没有T+N列，用空DataFrame
    if return_cols:
        return_df = factor_df[['code', 'date'] + list(return_cols.keys())].copy()
    else:
        return_df = pd.DataFrame()

    # 只对可用因子计算IC
    if len(available_factors) > 0 and not return_df.empty:
        avail_factor_df = factor_df[available_factors].copy()
        ic_monitor.compute_all_ic(avail_factor_df, return_df, factor_names=available_factors)
        print(f"  IC计算完成: {len(available_factors)}个因子 x {len(return_df)}条记录")
    else:
        print("  [跳过] 缺少因子或收益数据")

    # ── Step 5: 共线性检测 ──
    print_section_header('Step 5: 共线性检测')
    if len(available_factors) > 1:
        avail_factor_df = factor_df[available_factors].copy()
        collinearity_det.compute_corr(avail_factor_df, factor_names=available_factors)
        # VIF计算 (可能较慢，限制因子数)
        vif_factors = available_factors[:50]  # 最多50个因子计算VIF
        collinearity_det.compute_vif(avail_factor_df, factor_names=vif_factors)
        print(f"  共线性检测完成: {len(available_factors)}个因子")
    else:
        print("  [跳过] 可用因子不足")

    # ── Step 6: T+N收益计算 ──
    print_section_header('Step 6: T+N收益计算')
    if len(all_zt_events) > 0:
        # 获取完整交易日列表 (含追踪期)
        extended_end = pd.Timestamp(reference_date) + pd.Timedelta(days=30)
        full_trade_days = get_trade_days(end_date=extended_end, count=LOOKBACK_DAYS + 50)
        tn_tracker.compute_tn_returns(all_zt_events, full_trade_days, lookback_start)
        print(f"  T+N事件数: {len(all_zt_events)}, 计算记录: {len(tn_tracker.records)}")
    else:
        print("  [跳过] 无涨停事件")

    # ── Step 7: 因子-收益分桶相关性 ──
    print_section_header('Step 7: 因子-收益分桶相关性')
    bucket_corr_df = pd.DataFrame()
    positive_factors = {}

    # 合并T+N收益到factor_df
    if len(tn_tracker.records) > 0:
        tn_df = tn_tracker.get_results_df()
        if not tn_df.empty and not factor_df.empty:
            # 将T+N收益合并到因子数据
            # 按股票+日期合并
            for n in range(1, TRACK_DAYS + 1):
                col = f'tn_{n}'
                if col in tn_df.columns:
                    merge_cols = ['stock', 'entry_date', col]
                    avail_merge = [c for c in merge_cols if c in tn_df.columns]
                    if len(avail_merge) >= 3:
                        tn_merge = tn_df[avail_merge].copy()
                        tn_merge = tn_merge.rename(columns={'stock': 'code', 'entry_date': 'date'})
                        # 取每个(股票,日期)的第一条记录
                        tn_merge = tn_merge.drop_duplicates(subset=['code', 'date'], keep='first')
                        factor_df = factor_df.merge(tn_merge, on=['code', 'date'], how='left', suffixes=('', '_tn'))

            # 构建收益DataFrame
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
            print("  [跳过] 缺少T+N收益或因子数据")
    else:
        print("  [跳过] 无T+N记录")

    # ── Step 8: 输出报告 ──
    print_section_header('Step 8: 输出报告')
    print_final_reports(factor_df, ic_monitor, collinearity_det, tn_tracker,
                        bucket_corr_df, positive_factors, missing_report)

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
    }


# ============================================================
# 执行入口
# ============================================================
print("\n[READY] 执行 results = main() 开始分析")
print("  可选: main(reference_date='2025-01-15') 指定参考日期")
print()

results = main()
