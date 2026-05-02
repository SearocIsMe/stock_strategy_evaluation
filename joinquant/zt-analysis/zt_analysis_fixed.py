#!/usr/bin/env python
# coding: utf-8

# # 涨停板股票后 N 日走势分析
# 
# ## JoinQuant Research Notebook 研究程序
# 
# ---
# 
# ### 核心逻辑
# 1. 从用户上传的 CSV/Excel 文件读取备选股票列表
# 2. 以"自选时间"作为涨停日，"自选价格"作为涨停日收盘价
# 3. 调用 JoinQuant API 获取涨停日后 N=1~5 日的行情数据
# 4. 计算多维度因子，构建评分模型
# 5. 分类为强势/平稳/弱势，输出 Top3 候选
# 
# ### 字段映射
# 
# | 文件字段 | 程序内部含义 | 说明 |
# |---|---|---|
# | 自选时间 | 涨停日 | 定义为该股涨停的日期 |
# | 自选价格 | 涨停日收盘价 | 涨停当日的收盘价 |
# | 自选收益 | 相对涨停日收益 | (现价-涨停价)/涨停价 |
# | 连涨天数 | 连板/趋势强度 | 连续涨停天数参考 |
# | 昨日涨幅% | 是否涨停判断 | >=9.8% 视为涨停 |
# 
# ### 评分模型 (0~100)
# 
# | 因子类别 | 权重 | 细项 |
# |---|---|---|
# | 价格强度 | 30 | N日收益、最大涨幅、防守线满足比例 |
# | 趋势结构 | 20 | MA5位置、乖离率、连涨天数、3/5日涨幅 |
# | 成交量 | 20 | 量比、换手率、内外盘比、量价配合 |
# | 资金 | 15 | 主力净流入、主力净比、3日主力净流入 |
# | 基本面 | 10 | 市盈率、ROE、净利润同比、毛利率 |
# | 风险扣分 | 5 | 最大回撤、开板次数、振幅 |
# 
# ### 分类规则
# - **强势股**：score ≥ 70，价格多数时间 > 涨停价，未跌破-3%
# - **平稳股**：40 ≤ score < 70，价格围绕涨停价波动
# - **弱势股**：score < 40，快速跌破涨停价，跌破-3%
# 
# ---

# ## 1. 环境导入

# In[3]:


import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# 在普通 Python 脚本环境中没有 IPython.display.display；这里做兼容，避免脚本运行时报 NameError。
try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

# ============================================================================
# JoinQuant 环境导入（在 research notebook 中已预装）
# ============================================================================
try:
    from jqdata import *
    JQ_AVAILABLE = True
    print('[INFO] JoinQuant 环境已加载')
except ImportError:
    JQ_AVAILABLE = False
    print('[WARNING] JoinQuant 环境未检测到，部分功能将使用模拟数据')

print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')


# ## 2. 全局配置

# In[4]:


# ============================================================================
# 全局配置 — 可根据需要修改
# ============================================================================
CONFIG = {
    'file_path': 'stock_data.csv',       # 用户数据文件路径
    'max_days_from_zt': 6,               # 涨停日距当前最大天数
    'defense_line': -0.03,               # 防守线 -3%
    'zt_threshold': 9.8,                 # 涨停判断阈值（%）
    'N_days': [1, 2, 3, 4, 5],           # 分析的N日范围
    'top_k': 3,                          # 输出 Top K 股票
    'score_weights': {
        'price_strength': 30,
        'trend_structure': 20,
        'volume': 20,
        'capital_flow': 15,
        'fundamental': 10,
        'risk_deduction': 5,
    }
}

print('全局配置已加载')
print(f'  防守线: {CONFIG["defense_line"]:.1%}')
print(f'  涨停阈值: {CONFIG["zt_threshold"]}%')
print(f'  N日范围: {CONFIG["N_days"]}')
print(f'  Top K: {CONFIG["top_k"]}')


# ## 3. 字段映射定义

# In[5]:


# ============================================================================
# 字段映射：文件字段 → 标准化字段名
# ============================================================================
FIELD_MAP = {
    # 基本信息
    '代码': 'code',
    '名称': 'name',
    '自选时间': 'zt_date',           # → 涨停日
    '自选价格': 'zt_close',          # → 涨停日收盘价
    '自选收益': 'zt_return',         # → 相对涨停日收益
    '连涨天数': 'consecutive_up',    # → 连板/趋势强度
    '昨日涨幅%': 'prev_day_pct',     # → 是否涨停判断

    # 价格/行情
    '最新': 'latest_price',
    '涨幅%': 'pct_change',
    '涨跌': 'price_change',
    '最高': 'high',
    '最低': 'low',
    '开盘': 'open',
    '昨收': 'pre_close',
    '振幅%': 'amplitude',
    '均价': 'avg_price',

    # 成交
    '总量': 'volume',
    '现量': 'current_vol',
    '金额': 'amount',
    '量比': 'vol_ratio',
    '换手%': 'turnover_rate',
    '内盘': 'inner_vol',
    '外盘': 'outer_vol',
    '内外比': 'inner_outer_ratio',

    # 趋势
    '3日涨幅%': 'pct_3d',
    '6日涨幅%': 'pct_6d',
    '5日涨幅%': 'pct_5d',
    '本月涨幅%': 'pct_month',
    '今年涨幅%': 'pct_year',
    '近一月涨幅%': 'pct_1m',
    '近一年涨幅%': 'pct_1y',

    # 换手趋势
    '3日换手%': 'turnover_3d',
    '6日换手%': 'turnover_6d',
    '5日换手率%': 'turnover_5d',
    '10日换手率%': 'turnover_10d',

    # 资金
    '主力净流入': 'main_net_inflow',
    '主力净比': 'main_net_pct',
    '3日主力净流入': 'main_net_inflow_3d',

    # 估值
    '市盈率': 'pe_ratio',
    '市盈率(动)': 'pe_dynamic',
    '市盈率(TTM)': 'pe_ttm',
    '市净率': 'pb_ratio',
    '市销率': 'ps_ratio',
    '股息率TTM%': 'dividend_yield_ttm',

    # 规模
    '总股本': 'total_shares',
    '总市值': 'market_cap',
    '流通股本': 'float_shares',
    '流通市值': 'float_market_cap',
    '人均持股数': 'shares_per_person',

    # 基本面
    '每股收益': 'eps',
    'ROE': 'roe',
    'ROA': 'roa',
    '营业收入同比%': 'revenue_yoy',
    '净利润同比%': 'profit_yoy',
    '扣非净利润同比%': 'deducted_profit_yoy',
    '销售毛利率%': 'gross_margin',
    '资产负债率': 'debt_ratio',

    # 涨停特征
    '首次涨停时间': 'first_zt_time',
    '最终涨停时间': 'last_zt_time',
    '封单额': 'seal_amount',
    '封单量': 'seal_volume',
    '封成比%': 'seal_volume_ratio',
    '封流比%': 'seal_float_ratio',
    '涨停开板次数': 'zt_open_count',
    '今年累计涨停天数': 'zt_days_ytd',
    '几天几板': 'days_boards',

    # 竞价
    '竞价涨幅%': 'auction_pct',
    '竞价换手率%': 'auction_turnover',
    '竞价实际换手率%': 'auction_real_turnover',
    '竞价量': 'auction_volume',
    '竞价金额': 'auction_amount',
    '未匹配量': 'unmatched_volume',
    '未匹配金额': 'unmatched_amount',
}

print(f'字段映射已定义，共 {len(FIELD_MAP)} 个字段')


# ## 4. 模块 1: `read_data()` — 读取用户文件

# In[6]:


def read_data(file_path=None):
    """
    读取用户提供的 CSV / TXT / Excel 文件，返回原始 DataFrame。

    Parameters
    ----------
    file_path : str
        文件路径，支持 .csv / .txt / .xlsx / .xls

    Returns
    -------
    pd.DataFrame
        原始数据（列名尚未标准化）
    """
    if file_path is None:
        file_path = CONFIG['file_path']

    print(f"[read_data] 正在读取文件: {file_path}")

    # 根据扩展名选择读取方式
    ext = file_path.lower().split('.')[-1]

    if ext in ('csv', 'txt'):
        # 尝试常见编码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
    elif ext in ('xlsx', 'xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请使用 CSV/TXT/Excel")

    print(f"[read_data] 读取完成，共 {len(df)} 行, {len(df.columns)} 列")
    print(f"[read_data] 列名: {list(df.columns)[:10]}...")

    return df


print('✅ read_data() 已定义')


# ## 5. 模块 2: `clean_data()` — 数据清洗与标准化

# In[7]:


def _clean_numeric_series(series):
    """清理常见行情软件导出的数字字符串，如逗号、百分号、空值占位符。"""
    if series.dtype == object:
        series = (
            series.astype(str)
            .str.strip()
            .str.replace('%', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace('，', '', regex=False)
            .str.replace('--', '', regex=False)
            .str.replace('None', '', regex=False)
            .str.replace('nan', '', regex=False)
        )
    return pd.to_numeric(series, errors='coerce')


def _convert_pct_to_float(series):
    """将百分比字符串（如 '9.8%'）转为 float（如 9.8）"""
    return _clean_numeric_series(series)


def _normalize_jq_code(code_str):
    """
    将各种格式的股票代码标准化为 JoinQuant 格式。
    例: '000001' → '000001.XSHE', '600000' → '600000.XSHG'
    """
    code_str = str(code_str).strip().upper()

    # 如果已经是 JQ 格式
    if '.XSHE' in code_str or '.XSHG' in code_str:
        return code_str

    # 去掉可能的前缀/后缀，兼容 CSV 把 000001 读成 1 或 1.0 的情况
    code_str = code_str.replace('SH', '').replace('SZ', '').replace('BJ', '')
    if code_str.endswith('.0'):
        code_str = code_str[:-2]
    code_str = ''.join(ch for ch in code_str if ch.isdigit())

    # 补齐6位
    code_str = code_str.zfill(6)

    # 判断交易所
    if code_str.startswith(('6', '9', '5')):
        return f"{code_str}.XSHG"
    elif code_str.startswith(('0', '1', '2', '3')):
        return f"{code_str}.XSHE"
    elif code_str.startswith(('4', '8')):
        return f"{code_str}.XSHE"  # 北交所暂归XSHE
    else:
        return f"{code_str}.XSHE"


def clean_data(raw_df, filter_st=True, filter_yizhi=True):
    """
    数据清洗：字段标准化、类型转换、过滤无效数据。

    Parameters
    ----------
    raw_df : pd.DataFrame
        read_data() 输出的原始数据
    filter_st : bool
        是否剔除 ST 股票
    filter_yizhi : bool
        是否剔除一字板（无法交易）

    Returns
    -------
    pd.DataFrame
        清洗后的标准化数据
    """
    df = raw_df.copy()

    print(f"[clean_data] 开始清洗，原始数据 {len(df)} 行")

    # ---- 2.1 字段名标准化 ----
    rename_dict = {}
    for col in df.columns:
        col_stripped = str(col).strip()
        if col_stripped in FIELD_MAP:
            rename_dict[col] = FIELD_MAP[col_stripped]

    df = df.rename(columns=rename_dict)
    print(f"[clean_data] 字段标准化完成，映射了 {len(rename_dict)} 个字段")

    # ---- 2.2 日期格式统一 ----
    if 'zt_date' in df.columns:
        df['zt_date'] = pd.to_datetime(df['zt_date'], errors='coerce')
        # 去除无效日期
        df = df.dropna(subset=['zt_date'])

    # ---- 2.3 百分比字段转 float ----
    pct_fields = [
        'pct_change', 'amplitude', 'turnover_rate',
        'pct_3d', 'pct_6d', 'pct_5d', 'pct_month', 'pct_year', 'pct_1m', 'pct_1y',
        'turnover_3d', 'turnover_6d', 'turnover_5d', 'turnover_10d',
        'main_net_pct', 'dividend_yield_ttm',
        'revenue_yoy', 'profit_yoy', 'deducted_profit_yoy', 'gross_margin',
        'seal_volume_ratio', 'seal_float_ratio',
        'auction_pct', 'auction_turnover', 'auction_real_turnover',
        'prev_day_pct', 'zt_return',
    ]

    for field in pct_fields:
        if field in df.columns:
            df[field] = _convert_pct_to_float(df[field])

    # ---- 2.4 数值字段转 float ----
    numeric_fields = [
        'zt_close', 'latest_price', 'price_change', 'high', 'low', 'open',
        'pre_close', 'avg_price', 'volume', 'current_vol', 'amount',
        'vol_ratio', 'inner_vol', 'outer_vol', 'inner_outer_ratio',
        'main_net_inflow', 'main_net_inflow_3d',
        'pe_ratio', 'pe_dynamic', 'pe_ttm', 'pb_ratio', 'ps_ratio',
        'total_shares', 'market_cap', 'float_shares', 'float_market_cap',
        'shares_per_person', 'eps', 'roe', 'roa', 'debt_ratio',
        'seal_amount', 'seal_volume', 'zt_open_count', 'zt_days_ytd',
        'consecutive_up',
        'auction_volume', 'auction_amount', 'unmatched_volume', 'unmatched_amount',
    ]

    for field in numeric_fields:
        if field in df.columns:
            df[field] = _clean_numeric_series(df[field])

    # ---- 2.5 股票代码标准化为 JQ 格式 ----
    if 'code' in df.columns:
        df['code'] = df['code'].astype(str).str.strip()
        df['jq_code'] = df['code'].apply(_normalize_jq_code)

    # ---- 2.6 过滤：涨停日距当前 > 6 天 ----
    if 'zt_date' in df.columns:
        today = dt.datetime.now()
        max_date = today - dt.timedelta(days=CONFIG['max_days_from_zt'] + 3)  # 加缓冲
        # 获取交易日列表（如果 JQ 可用）
        if JQ_AVAILABLE:
            try:
                trade_days = get_trade_days(end_date=today.strftime('%Y-%m-%d'),
                                            count=CONFIG['max_days_from_zt'] + 5)
                min_zt_date = pd.to_datetime(trade_days[0])
                df = df[df['zt_date'] >= min_zt_date]
            except Exception:
                df = df[df['zt_date'] >= max_date]
        else:
            df = df[df['zt_date'] >= max_date]

    # ---- 2.7 过滤：停牌 ----
    if JQ_AVAILABLE and 'jq_code' in df.columns:
        try:
            code_list = df['jq_code'].dropna().unique().tolist()
            if code_list:
                current_data = get_current_data()
                paused_codes = [c for c in code_list if current_data[c].paused]
                df = df[~df['jq_code'].isin(paused_codes)]
        except Exception as e:
            print(f"[clean_data] 停牌过滤异常（跳过）: {e}")

    # ---- 2.8 过滤：ST（可选） ----
    if filter_st and JQ_AVAILABLE and 'jq_code' in df.columns:
        try:
            code_list = df['jq_code'].dropna().unique().tolist()
            if code_list:
                current_data = get_current_data()
                st_codes = [c for c in code_list if current_data[c].is_st]
                df = df[~df['jq_code'].isin(st_codes)]
        except Exception as e:
            print(f"[clean_data] ST过滤异常（跳过）: {e}")

    # ---- 2.9 过滤：一字板（无法交易） ----
    if filter_yizhi:
        # 一字板判断：开盘价 == 涨停价 且 最低价 == 涨停价
        if all(col in df.columns for col in ['open', 'low', 'zt_close']):
            df = df[~((df['open'] == df['zt_close']) & (df['low'] == df['zt_close']))]

    # 重置索引
    df = df.reset_index(drop=True)

    print(f"[clean_data] 清洗完成，剩余 {len(df)} 行")

    return df


print('✅ clean_data() 及辅助函数已定义')


# ## 6. 模块 3: `get_price_data()` — 获取 JQ 行情数据

# In[8]:


def _simulate_price_data(df, n_days):
    """当 JQ 不可用时，生成模拟数据用于测试"""
    print("[get_price_data] 生成模拟行情数据...")

    for n in CONFIG['N_days']:
        base_return = df.get('pct_change', pd.Series([0] * len(df)))
        df[f'return_n{n}'] = base_return * np.random.uniform(0.5, 1.5, len(df)) * (n / 3.0)
        df[f'max_return_n{n}'] = df[f'return_n{n}'] * np.random.uniform(1.0, 2.0, len(df))
        df[f'max_drawdown_n{n}'] = -abs(df[f'return_n{n}']) * np.random.uniform(0.3, 1.5, len(df))
        df[f'close_n{n}'] = df.get('zt_close', 10) * (1 + df[f'return_n{n}'] / 100)
        df[f'high_n{n}'] = df.get('zt_close', 10) * (1 + df[f'max_return_n{n}'] / 100)
        df[f'low_n{n}'] = df.get('zt_close', 10) * (1 + df[f'max_drawdown_n{n}'] / 100)
        df[f'volume_n{n}'] = df.get('volume', 1e6) * np.random.uniform(0.5, 1.5, len(df))

    df['ma5'] = df.get('zt_close', 10) * np.random.uniform(0.95, 1.05, len(df))
    df['ma10'] = df.get('zt_close', 10) * np.random.uniform(0.90, 1.05, len(df))
    df['bias_ma5'] = np.random.uniform(-0.05, 0.05, len(df))
    df['overall_max_drawdown'] = np.random.uniform(-0.08, -0.01, len(df))
    df['overall_max_return'] = np.random.uniform(0.01, 0.15, len(df))
    df['consecutive_up_post'] = np.random.randint(0, 4, len(df))
    df['days_above_zt'] = np.random.randint(0, n_days + 1, len(df))
    df['ratio_above_zt'] = df['days_above_zt'] / n_days
    df['days_above_defense'] = np.random.randint(0, n_days + 1, len(df))
    df['ratio_above_defense'] = df['days_above_defense'] / n_days

    return df


def get_price_data(cleaned_df, n_days=5):
    """
    调用 JoinQuant API 获取涨停日后 N 天的行情数据。

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        clean_data() 输出的标准化数据
    n_days : int
        最大回看天数

    Returns
    -------
    pd.DataFrame
        增加了行情数据的 DataFrame
    """
    df = cleaned_df.copy()

    print(f"[get_price_data] 开始获取行情数据，共 {len(df)} 只股票")

    if not JQ_AVAILABLE:
        print("[get_price_data] JQ 不可用，使用模拟数据")
        df = _simulate_price_data(df, n_days)
        return df

    # 存储每只股票的 N 日数据
    price_data_list = []

    for idx, row in df.iterrows():
        jq_code = row.get('jq_code', '')
        zt_date = row.get('zt_date', None)
        zt_close = row.get('zt_close', np.nan)

        if pd.isna(zt_date) or not jq_code:
            price_data_list.append({})
            continue

        zt_date_str = zt_date.strftime('%Y-%m-%d') if isinstance(zt_date, pd.Timestamp) else str(zt_date)

        try:
            # 获取涨停日及后 N 天的数据
            price_df = get_price(
                jq_code,
                end_date=zt_date_str,
                frequency='daily',
                fields=['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'high_limit', 'low_limit'],
                count=n_days + 10,
                panel=False,
                fill_paused=False,
                skip_paused=True
            )

            if price_df is None or price_df.empty:
                price_data_list.append({})
                continue

            # 找到涨停日所在行
            zt_date_pd = pd.to_datetime(zt_date_str)
            price_df['time'] = pd.to_datetime(price_df['time'])

            # 涨停日及之后的行
            mask = price_df['time'] >= zt_date_pd
            after_zt = price_df[mask].head(n_days + 1)

            if len(after_zt) == 0:
                price_data_list.append({})
                continue

            # 涨停日收盘价（如果文件中没有，用行情数据）
            if pd.isna(zt_close) and len(after_zt) > 0:
                zt_close = after_zt.iloc[0]['close']

            # 计算 N 日数据
            stock_price_data = {}
            stock_price_data['zt_close_jq'] = zt_close

            for n in CONFIG['N_days']:
                if len(after_zt) > n:
                    day_n = after_zt.iloc[n]
                    stock_price_data[f'close_n{n}'] = day_n['close']
                    stock_price_data[f'high_n{n}'] = day_n['high']
                    stock_price_data[f'low_n{n}'] = day_n['low']
                    stock_price_data[f'volume_n{n}'] = day_n['volume']
                    stock_price_data[f'amount_n{n}'] = day_n['money']
                    stock_price_data[f'return_n{n}'] = (day_n['close'] - zt_close) / zt_close if zt_close > 0 else np.nan
                    stock_price_data[f'max_return_n{n}'] = (day_n['high'] - zt_close) / zt_close if zt_close > 0 else np.nan
                    stock_price_data[f'max_drawdown_n{n}'] = (day_n['low'] - zt_close) / zt_close if zt_close > 0 else np.nan
                else:
                    for key in [f'close_n{n}', f'high_n{n}', f'low_n{n}', f'volume_n{n}', f'amount_n{n}', f'return_n{n}', f'max_return_n{n}', f'max_drawdown_n{n}']:
                        stock_price_data[key] = np.nan

            # 计算多日综合指标
            if len(after_zt) > 1:
                post_zt = after_zt.iloc[1:]  # 涨停日之后的数据

                # MA5 / MA10
                try:
                    ma_df = get_price(
                        jq_code,
                        end_date=zt_date_str,
                        frequency='daily',
                        fields=['close'],
                        count=15,
                        panel=False,
                        skip_paused=True
                    )
                    if ma_df is not None and len(ma_df) >= 5:
                        stock_price_data['ma5'] = ma_df['close'].tail(5).mean()
                    if ma_df is not None and len(ma_df) >= 10:
                        stock_price_data['ma10'] = ma_df['close'].tail(10).mean()
                except Exception:
                    stock_price_data['ma5'] = np.nan
                    stock_price_data['ma10'] = np.nan

                # 5日均线乖离率
                if 'ma5' in stock_price_data and not pd.isna(stock_price_data.get('ma5')):
                    latest_close = after_zt.iloc[-1]['close'] if len(after_zt) > 0 else np.nan
                    stock_price_data['bias_ma5'] = (latest_close - stock_price_data['ma5']) / stock_price_data['ma5'] if stock_price_data['ma5'] > 0 else np.nan
                else:
                    stock_price_data['bias_ma5'] = np.nan

                # N日最大回撤/涨幅（跨所有N日）
                if len(post_zt) > 0 and zt_close > 0:
                    stock_price_data['overall_max_drawdown'] = (post_zt['low'].min() - zt_close) / zt_close
                    stock_price_data['overall_max_return'] = (post_zt['high'].max() - zt_close) / zt_close
                else:
                    stock_price_data['overall_max_drawdown'] = np.nan
                    stock_price_data['overall_max_return'] = np.nan

                # 连续上涨天数
                consecutive_up = 0
                for _, day_row in post_zt.iterrows():
                    if day_row['close'] > day_row['pre_close']:
                        consecutive_up += 1
                    else:
                        break
                stock_price_data['consecutive_up_post'] = consecutive_up

                # 收盘价 > 涨停价的天数
                stock_price_data['days_above_zt'] = int((post_zt['close'] > zt_close).sum())
                stock_price_data['ratio_above_zt'] = (post_zt['close'] > zt_close).sum() / len(post_zt) if len(post_zt) > 0 else 0

                # 收盘价 > 涨停价*0.97 的天数（-3%防守线）
                stock_price_data['days_above_defense'] = int((post_zt['close'] > zt_close * 0.97).sum())
                stock_price_data['ratio_above_defense'] = (post_zt['close'] > zt_close * 0.97).sum() / len(post_zt) if len(post_zt) > 0 else 0

            price_data_list.append(stock_price_data)

        except Exception as e:
            print(f"[get_price_data] 获取 {jq_code} 行情失败: {e}")
            price_data_list.append({})

    # 合并行情数据
    price_df_result = pd.DataFrame(price_data_list)
    df = pd.concat([df, price_df_result], axis=1)

    print(f"[get_price_data] 行情数据获取完成")

    return df


print('✅ get_price_data() 及辅助函数已定义')


# ## 7. 模块 4: `calc_factors()` — 因子计算

# In[9]:


def calc_factors(price_df):
    """
    计算多维度因子，为评分模型提供输入。

    Parameters
    ----------
    price_df : pd.DataFrame
        get_price_data() 输出的含行情数据

    Returns
    -------
    pd.DataFrame
        增加了因子列的 DataFrame
    """
    df = price_df.copy()

    print(f"[calc_factors] 开始计算因子，共 {len(df)} 只股票")

    # ---- 4.1 价格强度因子 ----
    df['factor_return_3d'] = df.get('return_n3', np.nan)
    df['factor_return_5d'] = df.get('return_n5', np.nan)
    df['factor_max_return'] = df.get('overall_max_return', np.nan)
    df['factor_defense_ratio'] = df.get('ratio_above_defense', np.nan)
    df['factor_above_zt_ratio'] = df.get('ratio_above_zt', np.nan)

    # ---- 4.2 趋势结构因子 ----
    if 'close_n3' in df.columns and 'ma5' in df.columns:
        df['factor_ma5_position'] = np.where(
            df['ma5'] > 0,
            df['close_n3'] / df['ma5'] - 1,
            np.nan
        )
    else:
        df['factor_ma5_position'] = np.nan

    df['factor_bias_ma5'] = df.get('bias_ma5', np.nan)

    if 'consecutive_up_post' in df.columns:
        df['factor_consecutive_up'] = df.get('consecutive_up', 0) + df['consecutive_up_post']
    else:
        df['factor_consecutive_up'] = df.get('consecutive_up', 0)

    df['factor_pct_3d'] = df.get('pct_3d', np.nan)
    df['factor_pct_5d'] = df.get('pct_5d', np.nan)

    # ---- 4.3 成交量因子 ----
    df['factor_vol_ratio'] = df.get('vol_ratio', np.nan)
    df['factor_turnover'] = df.get('turnover_rate', np.nan)
    df['factor_inner_outer'] = df.get('inner_outer_ratio', np.nan)

    if 'return_n1' in df.columns and 'volume_n1' in df.columns and 'volume' in df.columns:
        vol_change = df['volume_n1'] / df['volume'].replace(0, np.nan)
        df['factor_vol_price'] = np.where(
            df['return_n1'] > 0,
            vol_change,
            -vol_change
        )
    else:
        df['factor_vol_price'] = np.nan

    # ---- 4.4 资金因子 ----
    df['factor_main_net_inflow'] = df.get('main_net_inflow', np.nan)
    df['factor_main_net_pct'] = df.get('main_net_pct', np.nan)
    df['factor_main_net_3d'] = df.get('main_net_inflow_3d', np.nan)

    # ---- 4.5 基本面因子 ----
    df['factor_pe'] = df.get('pe_ttm', df.get('pe_ratio', np.nan))
    df['factor_roe'] = df.get('roe', np.nan)
    df['factor_profit_yoy'] = df.get('profit_yoy', np.nan)
    df['factor_gross_margin'] = df.get('gross_margin', np.nan)

    # ---- 4.6 风险因子（扣分项） ----
    df['factor_max_drawdown'] = df.get('overall_max_drawdown', np.nan)
    df['factor_zt_open_count'] = df.get('zt_open_count', np.nan)
    df['factor_amplitude'] = df.get('amplitude', np.nan)

    # ---- 4.7 涨停特征因子（辅助） ----
    df['factor_seal_amount'] = df.get('seal_amount', np.nan)
    df['factor_seal_ratio'] = df.get('seal_volume_ratio', np.nan)
    df['factor_days_boards'] = df.get('days_boards', np.nan)
    df['factor_first_zt_time'] = df.get('first_zt_time', np.nan)

    print(f"[calc_factors] 因子计算完成")

    return df


print('✅ calc_factors() 已定义')


# ## 8. 模块 5: `analyze_N_day()` — N日统计分析

# In[10]:


def analyze_N_day(factor_df):
    """
    对 N=1~5 日进行统计分析。

    Parameters
    ----------
    factor_df : pd.DataFrame
        calc_factors() 输出的含因子数据

    Returns
    -------
    pd.DataFrame
        N日统计汇总表
    """
    print(f"[analyze_N_day] 开始 N 日统计分析")

    results = []

    for n in CONFIG['N_days']:
        return_col = f'return_n{n}'

        if return_col not in factor_df.columns:
            continue

        valid = factor_df.dropna(subset=[return_col])
        if len(valid) == 0:
            continue

        zt_close_col = 'zt_close'
        if zt_close_col not in valid.columns:
            continue

        # 1. 收盘价 > 涨停日收盘价的股票数
        above_zt_count = int((valid[return_col] > 0).sum())
        above_zt_ratio = above_zt_count / len(valid) if len(valid) > 0 else 0

        # 2. 连续 N 天 > 涨停价的股票
        consecutive_above = 0
        if all(f'return_n{i}' in factor_df.columns for i in range(1, n + 1)):
            consecutive_above_list = []
            for idx, row in valid.iterrows():
                all_above = True
                for i in range(1, n + 1):
                    if pd.isna(row.get(f'return_n{i}', np.nan)) or row[f'return_n{i}'] <= 0:
                        all_above = False
                        break
                if all_above:
                    consecutive_above_list.append(idx)
            consecutive_above = len(consecutive_above_list)

        # 3. 收盘价 > 涨停价 * 0.97 的股票
        defense_line = CONFIG['defense_line']
        above_defense_count = int((valid[return_col] > defense_line).sum())
        above_defense_ratio = above_defense_count / len(valid) if len(valid) > 0 else 0

        # 4. 连续满足 -3% 防守线的股票
        consecutive_defense = 0
        if all(f'return_n{i}' in factor_df.columns for i in range(1, n + 1)):
            consecutive_defense_list = []
            for idx, row in valid.iterrows():
                all_above_d = True
                for i in range(1, n + 1):
                    if pd.isna(row.get(f'return_n{i}', np.nan)) or row[f'return_n{i}'] <= defense_line:
                        all_above_d = False
                        break
                if all_above_d:
                    consecutive_defense_list.append(idx)
            consecutive_defense = len(consecutive_defense_list)

        # 5. 最大涨幅
        max_return_col = f'max_return_n{n}'
        avg_max_return = valid[max_return_col].mean() if max_return_col in valid.columns else np.nan

        # 6. 最大回撤
        max_dd_col = f'max_drawdown_n{n}'
        avg_max_drawdown = valid[max_dd_col].mean() if max_dd_col in valid.columns else np.nan

        # 7. 平均收益
        avg_return = valid[return_col].mean()

        result = {
            'N日': n,
            '股票数': len(valid),
            '收盘>涨停价_数量': above_zt_count,
            '收盘>涨停价_比例': f'{above_zt_ratio:.1%}',
            '连续N日>涨停价_数量': consecutive_above,
            '收盘>防守线_数量': above_defense_count,
            '收盘>防守线_比例': f'{above_defense_ratio:.1%}',
            '连续满足防守线_数量': consecutive_defense,
            '平均最大涨幅': f'{avg_max_return:.2%}' if not pd.isna(avg_max_return) else 'N/A',
            '平均最大回撤': f'{avg_max_drawdown:.2%}' if not pd.isna(avg_max_drawdown) else 'N/A',
            '平均收益': f'{avg_return:.2%}' if not pd.isna(avg_return) else 'N/A',
        }
        results.append(result)

    result_df = pd.DataFrame(results)
    print(f"[analyze_N_day] 分析完成，共 {len(results)} 个 N 日统计")

    return result_df


print('✅ analyze_N_day() 已定义')


# ## 9. 模块 6: `classify_stock()` — 股票分类（强/平/弱）

# In[11]:


def classify_stock(factor_df):
    """
    根据多维度条件将股票分为强势/平稳/弱势。

    Parameters
    ----------
    factor_df : pd.DataFrame
        calc_factors() 输出的含因子数据

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (强势股, 平稳股, 弱势股)
    """
    print(f"[classify_stock] 开始股票分类")

    df = factor_df.copy()
    df['classification'] = '平稳'  # 默认平稳

    for idx, row in df.iterrows():
        # ---- 强势股条件 ----
        ratio_above_zt = row.get('ratio_above_zt', 0)
        if pd.isna(ratio_above_zt):
            ratio_above_zt = 0

        max_dd = row.get('overall_max_drawdown', 0)
        if pd.isna(max_dd):
            max_dd = 0
        not_break_defense = max_dd > CONFIG['defense_line']

        vol_ratio = row.get('vol_ratio', 1)
        if pd.isna(vol_ratio):
            vol_ratio = 1
        vol_stable = vol_ratio >= 0.8

        main_net = row.get('main_net_inflow', 0)
        if pd.isna(main_net):
            main_net = 0
        main_positive = main_net >= 0

        turnover = row.get('turnover_rate', 0)
        if pd.isna(turnover):
            turnover = 0
        turnover_active = turnover >= 3

        consec_up = row.get('consecutive_up', 0)
        if pd.isna(consec_up):
            consec_up = 0
        high_consec = consec_up >= 2

        # 强势评分
        strong_score = 0
        if ratio_above_zt > 0.6:
            strong_score += 2
        elif ratio_above_zt > 0.4:
            strong_score += 1
        if not_break_defense:
            strong_score += 2
        if vol_stable:
            strong_score += 1
        if main_positive:
            strong_score += 1
        if turnover_active:
            strong_score += 1
        if high_consec:
            strong_score += 1

        # ---- 弱势股条件 ----
        weak_score = 0
        if ratio_above_zt < 0.3:
            weak_score += 2
        if not not_break_defense:
            weak_score += 2

        return_n1 = row.get('return_n1', 0)
        if pd.isna(return_n1):
            return_n1 = 0
        if return_n1 < 0 and vol_ratio > 1.5:
            weak_score += 2
        if main_net < 0:
            weak_score += 1

        # ---- 分类 ----
        if strong_score >= 5:
            df.at[idx, 'classification'] = '强势'
        elif weak_score >= 4:
            df.at[idx, 'classification'] = '弱势'
        else:
            df.at[idx, 'classification'] = '平稳'

    strong_df = df[df['classification'] == '强势'].copy()
    neutral_df = df[df['classification'] == '平稳'].copy()
    weak_df = df[df['classification'] == '弱势'].copy()

    print(f"[classify_stock] 分类完成: 强势 {len(strong_df)}, 平稳 {len(neutral_df)}, 弱势 {len(weak_df)}")

    return strong_df, neutral_df, weak_df


print('✅ classify_stock() 已定义')


# ## 10. 模块 7: `score_stock()` — 评分模型

# In[12]:


def score_stock(factor_df):
    """
    构建评分模型，对每只股票打分（0~100）。

    评分维度：
      1. 价格强度（30分）
      2. 趋势结构（20分）
      3. 成交量（20分）
      4. 资金（15分）
      5. 基本面（10分）
      6. 风险扣分（5分）
    """
    print(f"[score_stock] 开始评分")

    df = factor_df.copy()
    scores = []

    for idx, row in df.iterrows():
        # ======== 1. 价格强度（30分）========
        price_score = 0

        # 1a. N日收益（15分）
        return_3d = row.get('factor_return_3d', np.nan)
        if not pd.isna(return_3d):
            if return_3d > 0.05:
                price_score += 15
            elif return_3d > 0.02:
                price_score += 12
            elif return_3d > 0:
                price_score += 8
            elif return_3d > -0.03:
                price_score += 4
            else:
                price_score += 0

        # 1b. 最大涨幅（5分）
        max_ret = row.get('factor_max_return', np.nan)
        if not pd.isna(max_ret):
            if max_ret > 0.10:
                price_score += 5
            elif max_ret > 0.05:
                price_score += 4
            elif max_ret > 0:
                price_score += 2

        # 1c. 防守线满足比例（5分）
        defense_r = row.get('factor_defense_ratio', np.nan)
        if not pd.isna(defense_r):
            if defense_r > 0.8:
                price_score += 5
            elif defense_r > 0.6:
                price_score += 4
            elif defense_r > 0.4:
                price_score += 2

        # 1d. 收盘>涨停价比例（5分）
        above_zt_r = row.get('factor_above_zt_ratio', np.nan)
        if not pd.isna(above_zt_r):
            if above_zt_r > 0.7:
                price_score += 5
            elif above_zt_r > 0.5:
                price_score += 3
            elif above_zt_r > 0.3:
                price_score += 1

        # ======== 2. 趋势结构（20分）========
        trend_score = 0

        # 2a. MA5 位置（5分）
        ma5_pos = row.get('factor_ma5_position', np.nan)
        if not pd.isna(ma5_pos):
            if ma5_pos > 0.03:
                trend_score += 5
            elif ma5_pos > 0:
                trend_score += 4
            elif ma5_pos > -0.03:
                trend_score += 2

        # 2b. 乖离率（5分）
        bias = row.get('factor_bias_ma5', np.nan)
        if not pd.isna(bias):
            if 0 < bias < 0.05:
                trend_score += 5
            elif -0.02 < bias <= 0:
                trend_score += 4
            elif 0.05 <= bias < 0.10:
                trend_score += 3
            elif bias >= 0.10:
                trend_score += 1
            else:
                trend_score += 1

        # 2c. 连涨天数（5分）
        consec = row.get('factor_consecutive_up', 0)
        if pd.isna(consec):
            consec = 0
        if consec >= 3:
            trend_score += 5
        elif consec >= 2:
            trend_score += 4
        elif consec >= 1:
            trend_score += 2

        # 2d. 短期涨幅趋势（5分）
        pct_3d = row.get('factor_pct_3d', np.nan)
        if not pd.isna(pct_3d):
            if pct_3d > 10:
                trend_score += 5
            elif pct_3d > 5:
                trend_score += 4
            elif pct_3d > 0:
                trend_score += 2

        # ======== 3. 成交量（20分）========
        vol_score = 0

        # 3a. 量比（8分）
        vr = row.get('factor_vol_ratio', np.nan)
        if not pd.isna(vr):
            if 1.0 <= vr <= 2.0:
                vol_score += 8
            elif 0.8 <= vr < 1.0:
                vol_score += 6
            elif 2.0 < vr <= 3.0:
                vol_score += 5
            elif vr > 3.0:
                vol_score += 3
            else:
                vol_score += 2

        # 3b. 换手率（6分）
        tr = row.get('factor_turnover', np.nan)
        if not pd.isna(tr):
            if 5 <= tr <= 15:
                vol_score += 6
            elif 3 <= tr < 5 or 15 < tr <= 20:
                vol_score += 4
            elif 1 <= tr < 3 or tr > 20:
                vol_score += 2
            else:
                vol_score += 1

        # 3c. 内外盘比（3分）
        io = row.get('factor_inner_outer', np.nan)
        if not pd.isna(io):
            if io < 0.8:
                vol_score += 3
            elif io < 1.0:
                vol_score += 2
            else:
                vol_score += 1

        # 3d. 量价配合（3分）
        vp = row.get('factor_vol_price', np.nan)
        if not pd.isna(vp):
            if vp > 1.2:
                vol_score += 3
            elif vp > 0.8:
                vol_score += 2
            else:
                vol_score += 1

        # ======== 4. 资金（15分）========
        capital_score = 0

        # 4a. 主力净流入（5分）
        mni = row.get('factor_main_net_inflow', np.nan)
        if not pd.isna(mni):
            if mni > 0:
                capital_score += 5
            elif mni > -1e7:
                capital_score += 3
            else:
                capital_score += 0

        # 4b. 主力净比（5分）
        mnp = row.get('factor_main_net_pct', np.nan)
        if not pd.isna(mnp):
            if mnp > 5:
                capital_score += 5
            elif mnp > 0:
                capital_score += 4
            elif mnp > -5:
                capital_score += 2
            else:
                capital_score += 0

        # 4c. 3日主力净流入（5分）
        mni3 = row.get('factor_main_net_3d', np.nan)
        if not pd.isna(mni3):
            if mni3 > 0:
                capital_score += 5
            elif mni3 > -5e7:
                capital_score += 3
            else:
                capital_score += 0

        # ======== 5. 基本面（10分）========
        fund_score = 0

        # 5a. 市盈率（3分）
        pe = row.get('factor_pe', np.nan)
        if not pd.isna(pe):
            if 10 <= pe <= 30:
                fund_score += 3
            elif 0 < pe < 10 or 30 < pe <= 50:
                fund_score += 2
            elif pe > 50:
                fund_score += 1

        # 5b. ROE（3分）
        roe = row.get('factor_roe', np.nan)
        if not pd.isna(roe):
            if roe > 15:
                fund_score += 3
            elif roe > 8:
                fund_score += 2
            elif roe > 0:
                fund_score += 1

        # 5c. 净利润同比（2分）
        profit_y = row.get('factor_profit_yoy', np.nan)
        if not pd.isna(profit_y):
            if profit_y > 20:
                fund_score += 2
            elif profit_y > 0:
                fund_score += 1

        # 5d. 毛利率（2分）
        gm = row.get('factor_gross_margin', np.nan)
        if not pd.isna(gm):
            if gm > 30:
                fund_score += 2
            elif gm > 15:
                fund_score += 1

        # ======== 6. 风险扣分（5分）========
        risk_deduction = 0

        # 6a. 最大回撤（2分）
        max_dd = row.get('factor_max_drawdown', 0)
        if pd.isna(max_dd):
            max_dd = 0
        if max_dd < -0.05:
            risk_deduction += 2
        elif max_dd < -0.03:
            risk_deduction += 1

        # 6b. 开板次数（2分）
        zt_open = row.get('factor_zt_open_count', 0)
        if pd.isna(zt_open):
            zt_open = 0
        if zt_open >= 3:
            risk_deduction += 2
        elif zt_open >= 2:
            risk_deduction += 1

        # 6c. 振幅（1分）
        amp = row.get('factor_amplitude', 0)
        if pd.isna(amp):
            amp = 0
        if amp > 10:
            risk_deduction += 1

        # ======== 总分 ========
        total_score = (price_score + trend_score + vol_score +
                       capital_score + fund_score - risk_deduction)
        total_score = max(0, min(100, total_score))

        scores.append({
            'price_score': price_score,
            'trend_score': trend_score,
            'vol_score': vol_score,
            'capital_score': capital_score,
            'fund_score': fund_score,
            'risk_deduction': risk_deduction,
            'total_score': total_score,
        })

    scores_df = pd.DataFrame(scores, index=df.index)
    df = pd.concat([df, scores_df], axis=1)

    # 按 total_score 降序排列
    df = df.sort_values('total_score', ascending=False).reset_index(drop=True)

    print(f"[score_stock] 评分完成，分数范围: {df['total_score'].min():.0f} ~ {df['total_score'].max():.0f}")

    return df


print('✅ score_stock() 已定义')


# ## 11. 模块 8: `correlation_analysis()` — 因子相关性分析

# In[13]:


def correlation_analysis(scored_df):
    """
    分析各因子与 N 日收益之间的相关性。

    Parameters
    ----------
    scored_df : pd.DataFrame
        score_stock() 输出的含评分数据

    Returns
    -------
    pd.DataFrame
        因子相关性矩阵
    """
    print(f"[correlation_analysis] 开始因子相关性分析")

    # 选择因子列
    factor_cols = [
        'factor_return_3d', 'factor_return_5d', 'factor_max_return',
        'factor_defense_ratio', 'factor_above_zt_ratio',
        'factor_ma5_position', 'factor_bias_ma5', 'factor_consecutive_up',
        'factor_pct_3d', 'factor_pct_5d',
        'factor_vol_ratio', 'factor_turnover', 'factor_inner_outer', 'factor_vol_price',
        'factor_main_net_inflow', 'factor_main_net_pct', 'factor_main_net_3d',
        'factor_pe', 'factor_roe', 'factor_profit_yoy', 'factor_gross_margin',
        'factor_max_drawdown', 'factor_zt_open_count', 'factor_amplitude',
        'factor_seal_amount', 'factor_seal_ratio', 'factor_days_boards',
    ]

    # 目标变量
    target_cols = ['return_n1', 'return_n2', 'return_n3', 'return_n4', 'return_n5', 'total_score']

    # 筛选存在的列
    available_factors = [c for c in factor_cols if c in scored_df.columns]
    available_targets = [c for c in target_cols if c in scored_df.columns]

    if not available_factors or not available_targets:
        print("[correlation_analysis] 可用因子或目标变量不足，跳过")
        return pd.DataFrame()

    # 计算相关性
    all_cols = available_factors + available_targets
    corr_data = scored_df[all_cols].select_dtypes(include=[np.number])
    corr_matrix = corr_data.corr()

    # 只返回因子与目标之间的相关性。
    # 注意：部分因子如“几天几板”可能是字符串列，select_dtypes 会自动剔除；
    # 因此这里需要按 corr_matrix 实际存在的列再做一次交集，避免 KeyError。
    numeric_factors = [c for c in available_factors if c in corr_matrix.index]
    numeric_targets = [c for c in available_targets if c in corr_matrix.columns]

    if not numeric_factors or not numeric_targets:
        print("[correlation_analysis] 数值型因子或目标变量不足，跳过")
        return pd.DataFrame()

    factor_target_corr = corr_matrix.loc[numeric_factors, numeric_targets]

    print(f"[correlation_analysis] 分析完成，{len(numeric_factors)} 个因子 × {len(numeric_targets)} 个目标")

    return factor_target_corr


print('✅ correlation_analysis() 已定义')


# ## 12. 模块 9: `output_results()` — 输出结果

# In[14]:


def _generate_trade_signal(candidates_df):
    """
    生成简单交易信号。

    买入条件：强势 or 平稳 + 未跌破-3% + 量比>1 + 主力不为负
    卖出条件：跌破-3% + 放量长阴 + 主力流出
    """
    df = candidates_df.copy()
    df['trade_signal'] = '观望'

    for idx, row in df.iterrows():
        classification = row.get('classification', '')
        max_dd = row.get('overall_max_drawdown', 0)
        if pd.isna(max_dd):
            max_dd = 0
        vol_ratio = row.get('vol_ratio', 0)
        if pd.isna(vol_ratio):
            vol_ratio = 0
        main_net = row.get('main_net_inflow', 0)
        if pd.isna(main_net):
            main_net = 0
        return_n1 = row.get('return_n1', 0)
        if pd.isna(return_n1):
            return_n1 = 0

        # 卖出判断
        break_defense = max_dd < CONFIG['defense_line']
        heavy_drop = return_n1 < -0.03 and vol_ratio > 1.5
        main_outflow = main_net < 0 and abs(main_net) > 1e7

        if break_defense or heavy_drop or main_outflow:
            df.at[idx, 'trade_signal'] = '⚠️ 卖出'
        elif classification in ('强势', '平稳') and not break_defense and vol_ratio > 1 and main_net >= 0:
            df.at[idx, 'trade_signal'] = '✅ 买入'
        elif classification == '强势' and not break_defense:
            df.at[idx, 'trade_signal'] = '✅ 买入(强势)'
        else:
            df.at[idx, 'trade_signal'] = '👀 观望'

    return df


def output_results(cleaned_df, n_day_df, strong_df, neutral_df, weak_df,
                   scored_df, corr_df):
    """
    输出5个表 + 交易候选排序。

    表1: 清洗后股票列表
    表2: N日表现表
    表3: 强势股列表（重点）
    表4: 因子相关性表
    表5: 每日交易候选排序表
    """
    print('\n' + '=' * 80)
    print('涨停板股票后 N 日走势分析 — 结果输出')
    print('=' * 80)

    # ---- 表1: 清洗后股票列表 ----
    print('\n' + '─' * 60)
    print('📊 表1: 清洗后股票列表')
    print('─' * 60)

    display_cols_1 = ['code', 'name', 'zt_date', 'zt_close', 'pct_change',
                      'turnover_rate', 'vol_ratio', 'main_net_inflow', 'consecutive_up']
    available_cols_1 = [c for c in display_cols_1 if c in cleaned_df.columns]
    if available_cols_1:
        display(cleaned_df[available_cols_1].head(20))
    else:
        display(cleaned_df.head(20))

    # ---- 表2: N日表现表 ----
    print('\n' + '─' * 60)
    print('📊 表2: N日表现统计')
    print('─' * 60)
    if not n_day_df.empty:
        display(n_day_df)
    else:
        print('无 N 日统计数据')

    # ---- 表3: 强势股列表（重点）----
    print('\n' + '─' * 60)
    print('📊 表3: 强势股列表（⭐ 重点）')
    print('─' * 60)

    if not strong_df.empty:
        display_cols_3 = ['code', 'name', 'zt_date', 'zt_close', 'total_score',
                          'return_n1', 'return_n3', 'return_n5',
                          'ratio_above_zt', 'ratio_above_defense',
                          'main_net_inflow', 'turnover_rate', 'consecutive_up']
        available_cols_3 = [c for c in display_cols_3 if c in strong_df.columns]
        display(strong_df[available_cols_3])
    else:
        print('无强势股')

    # ---- 表4: 因子相关性表 ----
    print('\n' + '─' * 60)
    print('📊 表4: 因子相关性表（因子 vs N日收益/总分）')
    print('─' * 60)

    if not corr_df.empty:
        display(corr_df.round(3))
    else:
        print('无相关性数据')

    # ---- 表5: 每日交易候选排序表 ----
    print('\n' + '─' * 60)
    print('📊 表5: 每日交易候选排序表')
    print('─' * 60)

    if not scored_df.empty:
        candidates = scored_df[scored_df['classification'].isin(['强势', '平稳'])].copy()
        candidates = _generate_trade_signal(candidates)
        candidates = candidates.sort_values('total_score', ascending=False)

        display_cols_5 = ['code', 'name', 'zt_date', 'zt_close', 'total_score',
                          'classification', 'trade_signal',
                          'return_n1', 'return_n3', 'return_n5',
                          'main_net_inflow', 'vol_ratio']
        available_cols_5 = [c for c in display_cols_5 if c in candidates.columns]

        top_k = CONFIG['top_k']
        if len(candidates) > top_k:
            print(f'\n🏆 主选股票 (Top {top_k}):')
            display(candidates[available_cols_5].head(top_k))
            print(f'\n📋 候补股票:')
            display(candidates[available_cols_5].iloc[top_k:])
        else:
            display(candidates[available_cols_5])
    else:
        print('无候选股票')

    # ---- 汇总统计 ----
    print('\n' + '=' * 80)
    print('📈 汇总统计')
    print('=' * 80)
    print(f'  清洗后股票数: {len(cleaned_df)}')
    print(f'  强势股数: {len(strong_df)}')
    print(f'  平稳股数: {len(neutral_df)}')
    print(f'  弱势股数: {len(weak_df)}')
    if not scored_df.empty:
        print(f'  平均评分: {scored_df["total_score"].mean():.1f}')
        print(f'  最高评分: {scored_df["total_score"].max():.1f}')
        if 'code' in scored_df.columns and 'name' in scored_df.columns:
            top1 = scored_df.iloc[0]
            print(f'  ⭐ 最优标的: {top1.get("name", top1.get("code", "N/A"))} (评分: {top1["total_score"]:.0f})')

    print('\n' + '=' * 80)
    print('分析完成！本程序仅供研究参考，不构成投资建议。')
    print('=' * 80)


print('✅ output_results() 及辅助函数已定义')


# ## 13. 主函数: `run_analysis()` — 一键运行

# In[15]:


def run_analysis(file_path=None, filter_st=True, filter_yizhi=True):
    """
    一键运行涨停板后 N 日走势分析的全部流程。

    Parameters
    ----------
    file_path : str, optional
        数据文件路径
    filter_st : bool
        是否过滤 ST 股票
    filter_yizhi : bool
        是否过滤一字板

    Returns
    -------
    dict
        包含所有分析结果的字典
    """
    print('🚀 涨停板股票后 N 日走势分析 — 开始运行')
    print('=' * 80)

    # Step 1: 读取数据
    raw_df = read_data(file_path)

    # Step 2: 数据清洗
    cleaned_df = clean_data(raw_df, filter_st=filter_st, filter_yizhi=filter_yizhi)

    if len(cleaned_df) == 0:
        print('⚠️ 清洗后无有效数据，请检查文件内容和格式')
        return {}

    # Step 3: 获取行情数据
    price_df = get_price_data(cleaned_df)

    # Step 4: 计算因子
    factor_df = calc_factors(price_df)

    # Step 5: N日统计分析
    n_day_df = analyze_N_day(factor_df)

    # Step 6: 股票分类
    strong_df, neutral_df, weak_df = classify_stock(factor_df)

    # 将分类结果合并后再评分，确保 scored_df 保留 classification 列。
    classified_df = pd.concat([strong_df, neutral_df, weak_df], axis=0).sort_index()

    # Step 7: 评分
    scored_df = score_stock(classified_df)

    # 评分后重新生成三类股票表，这样 strong_df/neutral_df/weak_df 也带 total_score。
    strong_df = scored_df[scored_df['classification'] == '强势'].copy()
    neutral_df = scored_df[scored_df['classification'] == '平稳'].copy()
    weak_df = scored_df[scored_df['classification'] == '弱势'].copy()

    # Step 8: 因子相关性分析
    corr_df = correlation_analysis(scored_df)

    # Step 9: 输出结果
    output_results(cleaned_df, n_day_df, strong_df, neutral_df, weak_df,
                   scored_df, corr_df)

    # 返回所有结果
    results = {
        'cleaned_df': cleaned_df,
        'n_day_df': n_day_df,
        'strong_df': strong_df,
        'neutral_df': neutral_df,
        'weak_df': weak_df,
        'scored_df': scored_df,
        'corr_df': corr_df,
    }

    return results


print('✅ run_analysis() 已定义')


# ---
# 


# ============================================================================
# 脚本入口：直接运行 python zt_analysis_fixed.py 时使用
# JoinQuant Notebook 中也可以直接调用：results = run_analysis(file_path='Table_1429.csv')
# ============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='涨停板股票后 N 日走势分析')
    parser.add_argument('--file', default=CONFIG['file_path'], help='CSV/TXT/Excel 文件路径')
    parser.add_argument('--keep-st', action='store_true', help='不剔除 ST 股票')
    parser.add_argument('--keep-yizhi', action='store_true', help='不剔除一字板')
    args = parser.parse_args()

    results = run_analysis(
        file_path=args.file,
        filter_st=not args.keep_st,
        filter_yizhi=not args.keep_yizhi,
    )
