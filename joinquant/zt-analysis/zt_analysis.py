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
from datetime import datetime, timedelta
import warnings
import re
warnings.filterwarnings('ignore')

# ============================================================================
# display() 兼容：Jupyter / IPython / 纯终端
# ============================================================================
try:
    from IPython.display import display as _ip_display
    _display = _ip_display
except ImportError:
    def _display(obj, **kwargs):
        """纯终端 fallback：直接 print"""
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
    '涨速%': 'price_speed',
    '实体涨幅%': 'real_body_pct',

    # 成交
    '总量': 'volume',
    '现量': 'current_vol',
    '金额': 'amount',
    '量比': 'vol_ratio',
    '换手%': 'turnover_rate',
    '内盘': 'inner_vol',
    '外盘': 'outer_vol',
    '内外比': 'inner_outer_ratio',
    '买入价': 'bid_price',
    '卖出价': 'ask_price',
    '委比%': 'bid_ask_ratio',
    '委差': 'bid_ask_diff',
    '买一量': 'bid1_volume',
    '卖一量': 'ask1_volume',

    # 行业
    '所属行业': 'industry',

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
    '人均持股数(最新公告)': 'shares_per_person',  # CSV 实际列名

    # 基本面
    '每股收益': 'eps',
    '每股收益(期末股本)': 'eps',              # CSV 实际列名
    '每股收益(TTM)': 'eps_ttm',
    'ROE': 'roe',
    '加权净资产收益率%': 'roe',               # CSV 实际列名
    'ROA': 'roa',
    '总资产收益率%': 'roa',                   # CSV 实际列名
    '营业收入同比%': 'revenue_yoy',
    '营业总收入同比%': 'revenue_yoy',         # CSV 实际列名
    '净利润同比%': 'profit_yoy',
    '归属净利润同比%': 'profit_yoy',          # CSV 实际列名
    '扣非净利润同比%': 'deducted_profit_yoy',
    '销售毛利率%': 'gross_margin',
    '资产负债率': 'debt_ratio',
    '资产负债比率%': 'debt_ratio',            # CSV 实际列名
    '现金流量比例%': 'cash_flow_ratio',
    '总负债': 'total_debt',
    '股利支付率%': 'dividend_payout_ratio',

    # 昨日数据
    '昨成交量': 'prev_volume',
    '昨成交额': 'prev_amount',
    '昨量比': 'prev_vol_ratio',
    '昨换手%': 'prev_turnover_rate',

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
    '昨封单额': 'prev_seal_amount',
    '昨封单量': 'prev_seal_volume',
    '昨封成比%': 'prev_seal_volume_ratio',

    # 竞价
    '竞价涨幅%': 'auction_pct',
    '竞价换手率%': 'auction_turnover',
    '竞价实际换手率%': 'auction_real_turnover',
    '竞价量': 'auction_volume',
    '竞价金额': 'auction_amount',
    '未匹配量': 'unmatched_volume',
    '未匹配金额': 'unmatched_amount',
    '竞昨量比': 'auction_prev_vol_ratio',
    '竞昨成交比%': 'auction_prev_turnover_ratio',

    # pandas 对重复列名自动添加的后缀（CSV 中有重复列名）
    '封单额.1': 'final_seal_amount',       # 最终涨停时间对应的封单额
    '本月涨幅%.1': 'pct_month_dup',         # 重复列，后续丢弃
    '近一月涨幅%.1': 'pct_1m_dup',          # 重复列，后续丢弃
    '连涨天数.1': 'consecutive_up_dup',     # 重复列，后续丢弃
    '总市值.1': 'market_cap_dup',           # 重复列，后续丢弃
    '流通市值.1': 'float_market_cap_dup',   # 重复列，后续丢弃
    '市净率.1': 'pb_ratio_dup',             # 重复列，后续丢弃
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
        na_vals = ['—', '—', '-', 'NA', 'N/A', 'nan', 'None', '']
        for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding, na_values=na_vals,
                                 keep_default_na=True)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace',
                             na_values=na_vals, keep_default_na=True)
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


def _convert_pct_to_float(series):
    """将百分比字符串（如 '9.8%'）转为 float（如 9.8）"""
    if series.dtype == object:
        series = series.astype(str).str.replace('%', '', regex=False)
        series = pd.to_numeric(series, errors='coerce')
    return series


def _convert_chinese_number(val):
    """
    将含中文数量词的字符串转为 float。
    例: '39.3万' → 393000.0, '7.51亿' → 751000000.0, '1.74万' → 17400.0
    纯数字字符串直接转 float，无法转换返回 NaN。
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ('—', '—', '-', '', 'NA', 'N/A', 'nan', 'None'):
        return np.nan
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    # 处理中文数量词
    try:
        if '亿' in s:
            return float(s.replace('亿', '')) * 1e8
        elif '万' in s:
            return float(s.replace('万', '')) * 1e4
    except (ValueError, TypeError):
        pass
    return np.nan


def _convert_chinese_number_series(series):
    """对整列应用 _convert_chinese_number"""
    return series.apply(_convert_chinese_number)


def _parse_days_boards(val):
    """
    解析'几天几板'字符串，提取板数（涨停次数）。
    例: '3天3板' → 3, '昨日首板' → 1, '1天1板' → 1
    无法解析返回 NaN。
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ('—', '—', '-', '', 'NA', 'N/A'):
        return np.nan
    # 匹配 "X天X板" 或 "X板"
    m = re.search(r'(\d+)板', s)
    if m:
        return int(m.group(1))
    # 匹配 "首板" → 1
    if '首板' in s:
        return 1
    return np.nan


def _normalize_jq_code(code_str):
    """
    将各种格式的股票代码标准化为 JoinQuant 格式。
    例: '000001' → '000001.XSHE', '600000' → '600000.XSHG'
    """
    code_str = str(code_str).strip().upper()

    # 如果已经是 JQ 格式
    if '.XSHE' in code_str or '.XSHG' in code_str:
        return code_str

    # 去掉可能的前缀
    code_str = code_str.replace('SH', '').replace('SZ', '').replace('BJ', '')
    code_str = code_str.replace('.', '')

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
        col_stripped = col.strip()
        if col_stripped in FIELD_MAP:
            rename_dict[col] = FIELD_MAP[col_stripped]

    df = df.rename(columns=rename_dict)
    print(f"[clean_data] 字段标准化完成，映射了 {len(rename_dict)} 个字段")

    # ---- 2.2 日期格式统一 ----
    if 'zt_date' in df.columns:
        df['zt_date'] = pd.to_datetime(df['zt_date'], errors='coerce')
        # 去除无效日期
        df = df.dropna(subset=['zt_date'])

    # ---- 2.3 删除重复列（pandas 对重复列名自动添加 .1 后缀） ----
    dup_cols = [c for c in df.columns if c.endswith('_dup')]
    if dup_cols:
        df = df.drop(columns=dup_cols)
        print(f"[clean_data] 删除 {len(dup_cols)} 个重复列: {dup_cols}")

    # ---- 2.4 百分比字段转 float ----
    pct_fields = [
        'pct_change', 'amplitude', 'turnover_rate', 'price_speed', 'real_body_pct',
        'pct_3d', 'pct_6d', 'pct_5d', 'pct_month', 'pct_year', 'pct_1m', 'pct_1y',
        'turnover_3d', 'turnover_6d', 'turnover_5d', 'turnover_10d',
        'main_net_pct', 'dividend_yield_ttm',
        'revenue_yoy', 'profit_yoy', 'deducted_profit_yoy', 'gross_margin',
        'seal_volume_ratio', 'seal_float_ratio',
        'auction_pct', 'auction_turnover', 'auction_real_turnover',
        'prev_day_pct', 'zt_return', 'bid_ask_ratio',
        'roe', 'roa', 'debt_ratio', 'cash_flow_ratio', 'dividend_payout_ratio',
        'prev_turnover_rate', 'prev_seal_volume_ratio',
        'auction_prev_turnover_ratio',
    ]

    for field in pct_fields:
        if field in df.columns:
            df[field] = _convert_pct_to_float(df[field])

    # ---- 2.5 数值字段转 float（含中文数量词处理） ----
    # 需要中文数量词转换的字段（可能含 万/亿）
    chinese_num_fields = [
        'volume', 'current_vol', 'amount',
        'inner_vol', 'outer_vol',
        'main_net_inflow', 'main_net_inflow_3d',
        'total_shares', 'market_cap', 'float_shares', 'float_market_cap',
        'shares_per_person',
        'seal_amount', 'seal_volume',
        'auction_volume', 'auction_amount', 'unmatched_volume', 'unmatched_amount',
        'final_seal_amount',
        'prev_volume', 'prev_amount',
        'prev_seal_amount', 'prev_seal_volume',
        'bid1_volume', 'ask1_volume',
        'total_debt',
    ]

    for field in chinese_num_fields:
        if field in df.columns:
            df[field] = _convert_chinese_number_series(df[field])

    # 普通数值字段（不含中文数量词）
    plain_numeric_fields = [
        'zt_close', 'latest_price', 'price_change', 'high', 'low', 'open',
        'pre_close', 'avg_price', 'bid_price', 'ask_price',
        'vol_ratio', 'inner_outer_ratio', 'bid_ask_diff',
        'pe_ratio', 'pe_dynamic', 'pe_ttm', 'pb_ratio', 'ps_ratio',
        'eps', 'eps_ttm', 'debt_ratio',
        'zt_open_count', 'zt_days_ytd',
        'consecutive_up',
        'prev_vol_ratio', 'auction_prev_vol_ratio',
    ]

    for field in plain_numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')

    # ---- 2.6 解析"几天几板"字段 ----
    if 'days_boards' in df.columns:
        df['days_boards'] = df['days_boards'].apply(_parse_days_boards)

    # ---- 2.7 股票代码标准化为 JQ 格式 ----
    if 'code' in df.columns:
        df['code'] = df['code'].astype(str).str.strip()
        df['jq_code'] = df['code'].apply(_normalize_jq_code)

    # ---- 2.8 过滤：涨停日距当前 > 6 天 ----
    if 'zt_date' in df.columns:
        today = dt.datetime.now()
        max_date = today - timedelta(days=CONFIG['max_days_from_zt'] + 3)  # 加缓冲
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

    # ---- 2.9 过滤：停牌 ----
    if JQ_AVAILABLE and 'jq_code' in df.columns:
        try:
            code_list = df['jq_code'].dropna().unique().tolist()
            if code_list:
                current_data = get_current_data(code_list)
                paused_codes = [c for c in code_list if current_data[c].paused]
                df = df[~df['jq_code'].isin(paused_codes)]
        except Exception as e:
            print(f"[clean_data] 停牌过滤异常（跳过）: {e}")

    # ---- 2.10 过滤：ST（可选） ----
    if filter_st and JQ_AVAILABLE and 'jq_code' in df.columns:
        try:
            code_list = df['jq_code'].dropna().unique().tolist()
            if code_list:
                current_data = get_current_data(code_list)
                st_codes = [c for c in code_list if current_data[c].is_st]
                df = df[~df['jq_code'].isin(st_codes)]
        except Exception as e:
            print(f"[clean_data] ST过滤异常（跳过）: {e}")

    # ---- 2.11 过滤：一字板（无法交易） ----
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

    zt_close = df['zt_close'] if 'zt_close' in df.columns else pd.Series([10.0] * len(df))

    for n in CONFIG['N_days']:
        # 模拟 N 日收益率（小数形式，如 0.02 表示 2%）
        base_pct = df['pct_change'] if 'pct_change' in df.columns else pd.Series([0.0] * len(df))
        # pct_change 是百分比形式（如 2.5 表示 2.5%），转为小数
        base_return = base_pct / 100.0
        df[f'return_n{n}'] = base_return * np.random.uniform(0.5, 1.5, len(df)) * (n / 3.0)
        df[f'max_return_n{n}'] = df[f'return_n{n}'] * np.random.uniform(1.0, 2.0, len(df))
        df[f'max_drawdown_n{n}'] = -abs(df[f'return_n{n}']) * np.random.uniform(0.3, 1.5, len(df))
        df[f'close_n{n}'] = zt_close * (1 + df[f'return_n{n}'])
        df[f'high_n{n}'] = zt_close * (1 + df[f'max_return_n{n}'])
        df[f'low_n{n}'] = zt_close * (1 + df[f'max_drawdown_n{n}'])
        vol_base = df['volume'] if 'volume' in df.columns else pd.Series([1e6] * len(df))
        df[f'volume_n{n}'] = vol_base * np.random.uniform(0.5, 1.5, len(df))

    df['ma5'] = zt_close * np.random.uniform(0.95, 1.05, len(df))
    df['ma10'] = zt_close * np.random.uniform(0.90, 1.05, len(df))
    df['bias_ma5'] = np.random.uniform(-0.05, 0.05, len(df))
    df['overall_max_drawdown'] = np.random.uniform(-0.08, -0.01, len(df))
    df['overall_max_return'] = np.random.uniform(0.01, 0.15, len(df))
    df['consecutive_up_post'] = np.random.randint(0, 4, len(df))
    df['days_above_zt'] = np.random.randint(0, n_days + 1, len(df))
    df['ratio_above_zt'] = df['days_above_zt'] / n_days
    df['days_above_defense'] = np.random.randint(0, n_days + 1, len(df))
    df['ratio_above_defense'] = df['days_above_defense'] / n_days

    return df


def _normalize_price_df_time(price_df):
    """
    兼容不同版本 JoinQuant / pandas 的 get_price 返回格式。
    
    在新版 JQ 中，get_price(panel=False) 返回的 DataFrame
    日期在 'time' 列中；
    在旧版 JQ (pandas < 0.25) 中，日期可能是索引名 'time' 或 'date'，
    也可能没有 'time' 列。
    
    此函数确保返回的 DataFrame 有一个 'time' 列（datetime 类型）。
    """
    # 情况1：'time' 已经是列
    if 'time' in price_df.columns:
        price_df['time'] = pd.to_datetime(price_df['time'])
        return price_df
    
    # 情况2：索引名是 'time' 或 'date' 或其他日期索引
    if price_df.index.name in ('time', 'date', 'Time', 'Date') or isinstance(price_df.index, pd.DatetimeIndex):
        price_df = price_df.reset_index()
        # 找到日期列并重命名为 'time'
        for col in price_df.columns:
            if col.lower() in ('time', 'date', 'index'):
                price_df = price_df.rename(columns={col: 'time'})
                break
        if 'time' in price_df.columns:
            price_df['time'] = pd.to_datetime(price_df['time'])
        return price_df
    
    # 情况3：索引没有名字但是 DatetimeIndex
    if isinstance(price_df.index, pd.DatetimeIndex):
        price_df = price_df.reset_index()
        # reset_index 后的列名可能是 'index'
        if 'index' in price_df.columns:
            price_df = price_df.rename(columns={'index': 'time'})
        price_df['time'] = pd.to_datetime(price_df['time'])
        return price_df
    
    # 情况4：无法确定时间列，尝试第一列
    first_col = price_df.columns[0]
    try:
        price_df['time'] = pd.to_datetime(price_df[first_col])
    except Exception:
        pass
    
    return price_df


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
            # JQ API: start_date + end_date（不能用 start_date + count）
            # end_date 设为 zt_date + 足够缓冲日历日，确保覆盖 N 个交易日
            end_date_str = (zt_date + timedelta(days=n_days * 2 + 10)).strftime('%Y-%m-%d') if isinstance(zt_date, pd.Timestamp) else zt_date_str
            price_df = get_price(
                jq_code,
                start_date=zt_date_str,
                end_date=end_date_str,
                frequency='daily',
                fields=['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'high_limit', 'low_limit'],
                panel=False,
                fill_paused=False,
                skip_paused=True
            )

            if price_df is None or price_df.empty:
                price_data_list.append({})
                continue

            # 找到涨停日所在行
            zt_date_pd = pd.to_datetime(zt_date_str)
            # 兼容不同版本 JQ：time 可能是列名，也可能是索引
            price_df = _normalize_price_df_time(price_df)

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


# ## 6.5 模块 3.5: `supplement_jq_data()` — 补充 JQ 缺失数据

# In[8.5]:


def supplement_jq_data(cleaned_df):
    """
    对 CSV 中未提供的列，调用 JoinQuant API 补充数据。
    仅填充 NaN 的字段，不覆盖已有数据。

    补充内容：
    - get_price() → MA5, MA10, 近期收盘价
    - get_fundamentals() → PE, PB, ROE, EPS, 市值等
    - get_money_flow() → 主力净流入（如缺失）
    - get_current_data() → 最新价、是否停牌、是否ST

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        clean_data() 输出的标准化数据

    Returns
    -------
    pd.DataFrame
        补充了缺失字段的 DataFrame
    """
    df = cleaned_df.copy()

    print(f"[supplement_jq_data] 开始补充缺失数据，共 {len(df)} 只股票")

    if not JQ_AVAILABLE:
        print("[supplement_jq_data] JQ 不可用，跳过补充（使用 CSV 已有数据 + 模拟数据）")
        return df

    # 统计缺失情况
    key_fields = ['pe_ttm', 'pb_ratio', 'roe', 'eps', 'market_cap', 'float_market_cap',
                  'main_net_inflow', 'vol_ratio', 'turnover_rate']
    missing_report = {}
    for f in key_fields:
        if f in df.columns:
            na_count = df[f].isna().sum()
            if na_count > 0:
                missing_report[f] = na_count
    if missing_report:
        print(f"[supplement_jq_data] 缺失字段统计: {missing_report}")

    # ---- 3.5.1 用 get_fundamentals 补充基本面/估值/规模 ----
    fundamental_fields = {
        'pe_ttm': 'valuation.pe_ratio',           # 市盈率TTM
        'pe_dynamic': 'valuation.pe_ratio_lyr',    # 市盈率(动)
        'pb_ratio': 'valuation.pb_ratio',          # 市净率
        'ps_ratio': 'valuation.ps_ratio',          # 市销率
        'market_cap': 'valuation.market_cap',      # 总市值
        'float_market_cap': 'valuation.circulating_market_cap',  # 流通市值
        'eps': 'indicator.eps',                    # 每股收益
        'eps_ttm': 'indicator.eps',               # 每股收益TTM (近似)
        'roe': 'indicator.roe',                   # ROE
        'roa': 'indicator.roa',                   # ROA
        'gross_margin': 'indicator.gross_profit_margin',  # 销售毛利率
        'revenue_yoy': 'indicator.inc_revenue_year_on_year',  # 营业收入同比
        'profit_yoy': 'indicator.inc_net_profit_year_on_year',  # 净利润同比
        'debt_ratio': 'indicator.debt_to_asset_ratio',  # 资产负债率
    }

    # 检查哪些字段需要补充
    need_fundamental = False
    for f, _ in fundamental_fields.items():
        if f in df.columns and df[f].isna().any():
            need_fundamental = True
            break
        elif f not in df.columns:
            need_fundamental = True
            break

    if need_fundamental:
        code_list = df['jq_code'].dropna().unique().tolist()
        if code_list:
            try:
                # 获取最近一个交易日的财务数据
                # 用 zt_date 作为查询日期参考
                for idx, row in df.iterrows():
                    jq_code = row.get('jq_code', '')
                    if not jq_code or pd.isna(jq_code):
                        continue

                    zt_date = row.get('zt_date', None)
                    if pd.isna(zt_date):
                        query_date = dt.datetime.now().strftime('%Y-%m-%d')
                    else:
                        query_date = zt_date.strftime('%Y-%m-%d') if isinstance(zt_date, pd.Timestamp) else str(zt_date)

                    try:
                        q = query(
                            valuation.code,
                            valuation.pe_ratio,
                            valuation.pe_ratio_lyr,
                            valuation.pb_ratio,
                            valuation.ps_ratio,
                            valuation.market_cap,
                            valuation.circulating_market_cap,
                            indicator.eps,
                            indicator.roe,
                            indicator.roa,
                            indicator.gross_profit_margin,
                            indicator.inc_revenue_year_on_year,
                            indicator.inc_net_profit_year_on_year,
                            indicator.debt_to_asset_ratio,
                        ).filter(valuation.code == jq_code)

                        fund_df = get_fundamentals(q, date=query_date)

                        if fund_df is not None and not fund_df.empty:
                            fund_row = fund_df.iloc[0]

                            # 仅填充 NaN 的字段
                            field_map = {
                                'pe_ttm': fund_row.get('pe_ratio', np.nan),
                                'pe_dynamic': fund_row.get('pe_ratio_lyr', np.nan),
                                'pb_ratio': fund_row.get('pb_ratio', np.nan),
                                'ps_ratio': fund_row.get('ps_ratio', np.nan),
                                'market_cap': fund_row.get('market_cap', np.nan),
                                'float_market_cap': fund_row.get('circulating_market_cap', np.nan),
                                'eps': fund_row.get('eps', np.nan),
                                'roe': fund_row.get('roe', np.nan),
                                'roa': fund_row.get('roa', np.nan),
                                'gross_margin': fund_row.get('gross_profit_margin', np.nan),
                                'revenue_yoy': fund_row.get('inc_revenue_year_on_year', np.nan),
                                'profit_yoy': fund_row.get('inc_net_profit_year_on_year', np.nan),
                                'debt_ratio': fund_row.get('debt_to_asset_ratio', np.nan),
                            }

                            for field, value in field_map.items():
                                if field in df.columns and pd.isna(df.at[idx, field]):
                                    df.at[idx, field] = value
                                elif field not in df.columns:
                                    df.at[idx, field] = value

                    except Exception as e:
                        pass  # 单只股票失败不影响整体

                print(f"[supplement_jq_data] 基本面数据补充完成")

            except Exception as e:
                print(f"[supplement_jq_data] 基本面数据补充异常（跳过）: {e}")

    # ---- 3.5.2 用 get_price 补充 MA5/MA10 ----
    ma_fields_missing = False
    if 'ma5' not in df.columns or 'ma10' not in df.columns:
        ma_fields_missing = True
    elif df.get('ma5', pd.Series(dtype=float)).isna().all() or df.get('ma10', pd.Series(dtype=float)).isna().all():
        ma_fields_missing = True

    if ma_fields_missing:
        print(f"[supplement_jq_data] 补充 MA5/MA10 数据...")
        for idx, row in df.iterrows():
            jq_code = row.get('jq_code', '')
            zt_date = row.get('zt_date', None)
            if not jq_code or pd.isna(zt_date):
                continue

            zt_date_str = zt_date.strftime('%Y-%m-%d') if isinstance(zt_date, pd.Timestamp) else str(zt_date)

            try:
                price_df = get_price(
                    jq_code,
                    end_date=zt_date_str,
                    frequency='daily',
                    fields=['close'],
                    count=15,
                    skip_paused=True,
                    panel=False,
                )
                if price_df is not None and len(price_df) >= 5:
                    if 'ma5' not in df.columns or pd.isna(df.at[idx, 'ma5']):
                        df.at[idx, 'ma5'] = price_df['close'].tail(5).mean()
                if price_df is not None and len(price_df) >= 10:
                    if 'ma10' not in df.columns or pd.isna(df.at[idx, 'ma10']):
                        df.at[idx, 'ma10'] = price_df['close'].tail(10).mean()
            except Exception:
                pass

        print(f"[supplement_jq_data] MA5/MA10 补充完成")

    # ---- 3.5.3 用 get_money_flow 补充主力净流入 ----
    if 'main_net_inflow' in df.columns and df['main_net_inflow'].isna().any():
        print(f"[supplement_jq_data] 补充主力资金流向数据...")
        for idx, row in df.iterrows():
            jq_code = row.get('jq_code', '')
            zt_date = row.get('zt_date', None)
            if not jq_code or pd.isna(zt_date):
                continue
            if not pd.isna(df.at[idx, 'main_net_inflow']):
                continue

            zt_date_str = zt_date.strftime('%Y-%m-%d') if isinstance(zt_date, pd.Timestamp) else str(zt_date)

            try:
                money_df = get_money_flow([jq_code], start_date=zt_date_str,
                                          end_date=zt_date_str)
                if money_df is not None and not money_df.empty:
                    # 主力 = 大单 + 超大单
                    net_inflow = (money_df['sec_large_net_inflow'].sum() +
                                  money_df['large_net_inflow'].sum())
                    if pd.isna(df.at[idx, 'main_net_inflow']):
                        df.at[idx, 'main_net_inflow'] = net_inflow
            except Exception:
                pass

        print(f"[supplement_jq_data] 主力资金流向补充完成")

    # ---- 3.5.4 补充乖离率 ----
    if 'bias_ma5' not in df.columns and 'ma5' in df.columns and 'latest_price' in df.columns:
        df['bias_ma5'] = np.where(
            df['ma5'] > 0,
            (df['latest_price'] - df['ma5']) / df['ma5'],
            np.nan
        )
        print(f"[supplement_jq_data] 乖离率(bias_ma5)已计算")

    # 统计补充后缺失情况
    remaining_missing = {}
    for f in key_fields:
        if f in df.columns:
            na_count = df[f].isna().sum()
            if na_count > 0:
                remaining_missing[f] = na_count
    if remaining_missing:
        print(f"[supplement_jq_data] 补充后仍缺失: {remaining_missing}")
    else:
        print(f"[supplement_jq_data] 所有关键字段已完整")

    return df


print('✅ supplement_jq_data() 已定义')


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
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (含分类列的完整DataFrame, 强势股, 平稳股, 弱势股)
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

    return df, strong_df, neutral_df, weak_df


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


# ## 10.5 模块 7.5: `predict_next_day()` — 次日建仓预测

# In[12.5]:


def predict_next_day(scored_df):
    """
    基于当前收盘情况，预测未来一天可建仓的股票。

    预测逻辑：
    1. 从已评分股票中筛选候选（强势/平稳 + 未破防守线）
    2. 综合评分 + 技术面 + 资金面 + 涨停特征，计算"次日建仓指数"
    3. 输出：主选标的、候补标的、建议买入价、止损价、目标价

    Parameters
    ----------
    scored_df : pd.DataFrame
        score_stock() 输出的含评分和分类数据

    Returns
    -------
    pd.DataFrame
        次日建仓预测表，按建仓指数降序排列
    """
    print(f"[predict_next_day] 开始次日建仓预测")

    df = scored_df.copy()

    # ---- 筛选候选池 ----
    # 条件：强势或平稳 + 未跌破防守线
    candidates = df[df['classification'].isin(['强势', '平稳'])].copy()

    if len(candidates) == 0:
        print("[predict_next_day] 无符合条件的候选股票")
        return pd.DataFrame()

    # ---- 计算次日建仓指数 (0~100) ----
    predict_scores = []

    for idx, row in candidates.iterrows():
        # === 1. 综合评分权重 (40分) ===
        total_score = row.get('total_score', 0)
        if pd.isna(total_score):
            total_score = 0
        score_component = min(total_score / 100 * 40, 40)

        # === 2. 价格位置权重 (20分) ===
        # 收盘价相对涨停价的位置
        price_component = 0
        ratio_above_zt = row.get('ratio_above_zt', 0)
        if pd.isna(ratio_above_zt):
            ratio_above_zt = 0

        if ratio_above_zt >= 0.8:
            price_component = 20  # 大部分时间在涨停价上方
        elif ratio_above_zt >= 0.6:
            price_component = 16
        elif ratio_above_zt >= 0.4:
            price_component = 12
        elif ratio_above_zt >= 0.2:
            price_component = 8
        else:
            price_component = 4

        # === 3. 防守线权重 (15分) ===
        defense_component = 0
        ratio_above_defense = row.get('ratio_above_defense', 0)
        if pd.isna(ratio_above_defense):
            ratio_above_defense = 0

        if ratio_above_defense >= 0.8:
            defense_component = 15
        elif ratio_above_defense >= 0.6:
            defense_component = 12
        elif ratio_above_defense >= 0.4:
            defense_component = 8
        else:
            defense_component = 3

        # === 4. 资金面权重 (15分) ===
        capital_component = 0
        main_net = row.get('main_net_inflow', 0)
        if pd.isna(main_net):
            main_net = 0
        main_pct = row.get('main_net_pct', 0)
        if pd.isna(main_pct):
            main_pct = 0

        if main_net > 0 and main_pct > 0:
            capital_component = 15
        elif main_net > 0:
            capital_component = 10
        elif main_net > -5e7:
            capital_component = 6
        else:
            capital_component = 2

        # === 5. 涨停特征权重 (10分) ===
        zt_feature_component = 0
        # 封成比越高越好
        seal_ratio = row.get('seal_volume_ratio', 0)
        if pd.isna(seal_ratio):
            seal_ratio = 0
        # 开板次数越少越好
        zt_open = row.get('zt_open_count', 0)
        if pd.isna(zt_open):
            zt_open = 0
        # 连板数
        consec_up = row.get('consecutive_up', 0)
        if pd.isna(consec_up):
            consec_up = 0

        if seal_ratio > 5:
            zt_feature_component += 4
        elif seal_ratio > 2:
            zt_feature_component += 3
        elif seal_ratio > 0:
            zt_feature_component += 2

        if zt_open == 0:
            zt_feature_component += 4
        elif zt_open == 1:
            zt_feature_component += 2
        else:
            zt_feature_component += 0

        if consec_up >= 3:
            zt_feature_component += 2
        elif consec_up >= 2:
            zt_feature_component += 1

        zt_feature_component = min(zt_feature_component, 10)

        # === 汇总建仓指数 ===
        entry_index = score_component + price_component + defense_component + capital_component + zt_feature_component
        entry_index = max(0, min(100, entry_index))

        # === 计算建议买入价、止损价、目标价 ===
        zt_close = row.get('zt_close', np.nan)
        if pd.isna(zt_close) or zt_close <= 0:
            buy_price = np.nan
            stop_loss = np.nan
            target_price = np.nan
        else:
            # 建议买入价：涨停价 * 0.98~1.02（根据强度调整）
            if row.get('classification') == '强势':
                buy_price = zt_close * 1.00  # 强势可按涨停价附近买入
            else:
                buy_price = zt_close * 0.98  # 平稳股等回踩买入

            # 止损价：涨停价 * 0.97（-3%防守线）
            stop_loss = zt_close * 0.97

            # 目标价：根据评分推算预期收益
            if total_score >= 70:
                target_price = zt_close * 1.10  # 强势目标+10%
            elif total_score >= 50:
                target_price = zt_close * 1.05  # 平稳目标+5%
            else:
                target_price = zt_close * 1.03  # 保守目标+3%

        # === 预测次日涨跌方向 ===
        if entry_index >= 70:
            prediction = '看多'
            signal = '🟢 积极建仓'
        elif entry_index >= 55:
            prediction = '偏多'
            signal = '🟡 适度建仓'
        elif entry_index >= 40:
            prediction = '震荡'
            signal = '🟠 观望为主'
        else:
            prediction = '偏空'
            signal = '🔴 不建议建仓'

        predict_scores.append({
            'entry_index': entry_index,
            'score_component': score_component,
            'price_component': price_component,
            'defense_component': defense_component,
            'capital_component': capital_component,
            'zt_feature_component': zt_feature_component,
            'buy_price': round(buy_price, 2) if not pd.isna(buy_price) else np.nan,
            'stop_loss': round(stop_loss, 2) if not pd.isna(stop_loss) else np.nan,
            'target_price': round(target_price, 2) if not pd.isna(target_price) else np.nan,
            'prediction': prediction,
            'signal': signal,
        })

    predict_df = pd.DataFrame(predict_scores, index=candidates.index)
    result_df = pd.concat([candidates, predict_df], axis=1)

    # 按建仓指数降序排列
    result_df = result_df.sort_values('entry_index', ascending=False).reset_index(drop=True)

    # 输出预测汇总
    positive_count = len(result_df[result_df['prediction'].isin(['看多', '偏多'])])
    print(f"[predict_next_day] 预测完成: {len(result_df)} 只候选, {positive_count} 只看多/偏多")

    return result_df


print('✅ predict_next_day() 已定义')


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

    # 只返回因子与目标之间的相关性
    factor_target_corr = corr_matrix.loc[available_factors, available_targets]

    print(f"[correlation_analysis] 分析完成，{len(available_factors)} 个因子 × {len(available_targets)} 个目标")

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
                   scored_df, corr_df, predict_df=None):
    """
    输出6个表 + 交易候选排序。

    表1: 清洗后股票列表
    表2: N日表现表
    表3: 强势股列表（重点）
    表4: 因子相关性表
    表5: 每日交易候选排序表
    表6: 次日建仓预测表（新增）
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
        _display(cleaned_df[available_cols_1].head(20))
    else:
        _display(cleaned_df.head(20))

    # ---- 表2: N日表现表 ----
    print('\n' + '─' * 60)
    print('📊 表2: N日表现统计')
    print('─' * 60)
    if not n_day_df.empty:
        _display(n_day_df)
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
        _display(strong_df[available_cols_3])
    else:
        print('无强势股')

    # ---- 表4: 因子相关性表 ----
    print('\n' + '─' * 60)
    print('📊 表4: 因子相关性表（因子 vs N日收益/总分）')
    print('─' * 60)

    if not corr_df.empty:
        _display(corr_df.round(3))
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
            _display(candidates[available_cols_5].head(top_k))
            print(f'\n📋 候补股票:')
            _display(candidates[available_cols_5].iloc[top_k:])
        else:
            _display(candidates[available_cols_5])
    else:
        print('无候选股票')

    # ---- 表6: 次日建仓预测表（新增）----
    print('\n' + '─' * 60)
    print('📊 表6: 次日建仓预测表（🔮 新增）')
    print('─' * 60)

    if predict_df is not None and not predict_df.empty:
        display_cols_6 = ['code', 'name', 'zt_date', 'zt_close', 'total_score',
                          'classification', 'entry_index', 'prediction', 'signal',
                          'buy_price', 'stop_loss', 'target_price',
                          'main_net_inflow', 'vol_ratio', 'consecutive_up']
        available_cols_6 = [c for c in display_cols_6 if c in predict_df.columns]

        top_k = CONFIG['top_k']
        if len(predict_df) > top_k:
            print(f'\n🏆 主选建仓标的 (Top {top_k}):')
            _display(predict_df[available_cols_6].head(top_k))
            print(f'\n📋 候补建仓标的:')
            _display(predict_df[available_cols_6].iloc[top_k:])
        else:
            _display(predict_df[available_cols_6])

        # 输出预测汇总
        print(f'\n💡 次日建仓建议汇总:')
        for _, row in predict_df.head(top_k).iterrows():
            name = row.get('name', row.get('code', 'N/A'))
            signal = row.get('signal', 'N/A')
            buy = row.get('buy_price', np.nan)
            stop = row.get('stop_loss', np.nan)
            target = row.get('target_price', np.nan)
            entry = row.get('entry_index', 0)
            print(f'  {signal} {name}: 建议买入价 {buy:.2f}, 止损价 {stop:.2f}, 目标价 {target:.2f} (建仓指数: {entry:.0f})'
                  if not pd.isna(buy) else f'  {signal} {name}: 数据不足')
    else:
        print('无次日建仓预测数据')

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
    if predict_df is not None and not predict_df.empty:
        positive = predict_df[predict_df['prediction'].isin(['看多', '偏多'])]
        print(f'  🔮 次日看多/偏多: {len(positive)} 只')
        if len(positive) > 0:
            best = positive.iloc[0]
            print(f'  🔮 次日首选: {best.get("name", best.get("code", "N/A"))} (建仓指数: {best["entry_index"]:.0f})')

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

    # Step 3: 补充 JQ 缺失数据（如 JQ 可用）
    supplemented_df = supplement_jq_data(cleaned_df)

    # Step 4: 获取行情数据
    price_df = get_price_data(supplemented_df)

    # Step 5: 计算因子
    factor_df = calc_factors(price_df)

    # Step 6: N日统计分析
    n_day_df = analyze_N_day(factor_df)

    # Step 7: 股票分类（返回含 classification 列的完整 DataFrame）
    classified_df, strong_df, neutral_df, weak_df = classify_stock(factor_df)

    # Step 8: 评分（在含 classification 列的 classified_df 上评分）
    scored_df = score_stock(classified_df)

    # Step 9: 因子相关性分析
    corr_df = correlation_analysis(scored_df)

    # Step 10: 次日建仓预测
    predict_df = predict_next_day(scored_df)

    # Step 11: 输出结果
    output_results(cleaned_df, n_day_df, strong_df, neutral_df, weak_df,
                   scored_df, corr_df, predict_df)

    # 返回所有结果
    results = {
        'cleaned_df': cleaned_df,
        'supplemented_df': supplemented_df,
        'n_day_df': n_day_df,
        'strong_df': strong_df,
        'neutral_df': neutral_df,
        'weak_df': weak_df,
        'scored_df': scored_df,
        'corr_df': corr_df,
        'predict_df': predict_df,
    }

    return results


print('✅ run_analysis() 已定义')


# ---
# 
# ## 14. 运行分析
# 
# ### 方式 A: 使用自己的数据文件
# 
# 将文件上传到 JoinQuant Research 环境，然后修改下方路径运行。

# In[16]:


# ============================================================
# 使用自己的数据文件运行（请修改文件路径）
# ============================================================
results = run_analysis(file_path='Table_1429.csv')

# 访问结果:
# results['strong_df']    → 强势股列表
# results['scored_df']    → 评分排序表
# results['n_day_df']     → N日统计表
# results['corr_df']      → 因子相关性表

print('请取消注释上方代码并修改文件路径后运行')


# ### 方式 B: 使用内嵌演示数据运行

# In[3]:


# ============================================================
# 演示模式：使用内嵌模拟数据
# ============================================================

demo_data = {
    '代码': ['000001', '600036', '000858', '002475', '300750',
             '601318', '000333', '600519', '002714', '300059'],
    '名称': ['平安银行', '招商银行', '五粮液', '立讯精密', '宁德时代',
             '中国平安', '美的集团', '贵州茅台', '牧原股份', '东方财富'],
    '自选时间': pd.date_range('2026-04-22', periods=10, freq='D'),
    '自选价格': [12.5, 35.2, 158.0, 33.8, 215.0, 48.5, 62.3, 1750.0, 42.8, 18.5],
    '自选收益': [2.5, -1.2, 3.8, -0.5, 5.2, 1.8, -2.1, 0.8, -3.5, 4.1],
    '连涨天数': [1, 2, 1, 1, 3, 1, 1, 1, 1, 2],
    '昨日涨幅%': [10.0, 10.0, 10.0, 9.9, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    '最新': [12.8, 34.8, 162.0, 33.5, 225.0, 49.2, 61.0, 1760.0, 41.5, 19.2],
    '涨幅%': [2.4, -1.1, 2.5, -0.9, 4.7, 1.4, -2.1, 0.6, -3.0, 3.8],
    '涨跌': [0.3, -0.4, 4.0, -0.3, 10.0, 0.7, -1.3, 10.0, -1.3, 0.7],
    '最高': [13.0, 35.5, 163.0, 34.2, 228.0, 49.8, 62.5, 1765.0, 43.5, 19.5],
    '最低': [12.3, 34.5, 157.0, 33.0, 218.0, 48.0, 60.5, 1745.0, 41.0, 18.0],
    '开盘': [12.5, 35.0, 159.0, 33.6, 220.0, 48.5, 62.0, 1752.0, 42.5, 18.8],
    '昨收': [12.5, 35.2, 158.0, 33.8, 215.0, 48.5, 62.3, 1750.0, 42.8, 18.5],
    '振幅%': [5.6, 2.8, 3.8, 3.6, 4.7, 3.7, 3.2, 1.1, 5.8, 8.1],
    '均价': [12.65, 35.0, 160.0, 33.6, 222.0, 48.8, 61.5, 1755.0, 42.0, 18.9],
    '总量': [5e6, 3e6, 2e6, 4e6, 1.5e6, 3.5e6, 2.5e6, 8e5, 1.8e6, 6e6],
    '现量': [1e5, 8e4, 5e4, 9e4, 4e4, 7e4, 6e4, 2e4, 4.5e4, 1.2e5],
    '金额': [6.325e7, 1.05e8, 3.2e8, 1.344e8, 3.33e8, 1.708e8, 1.5375e8, 1.404e9, 7.56e7, 1.134e8],
    '量比': [1.2, 0.8, 1.5, 1.1, 2.0, 0.9, 1.3, 0.7, 1.8, 2.5],
    '换手%': [3.2, 1.5, 4.8, 5.5, 6.2, 2.1, 3.8, 0.6, 7.5, 8.2],
    '内盘': [2.2e6, 1.6e6, 9e5, 2.1e6, 6.5e5, 1.8e6, 1.35e6, 4.2e5, 9.5e5, 2.8e6],
    '外盘': [2.8e6, 1.4e6, 1.1e6, 1.9e6, 8.5e5, 1.7e6, 1.15e6, 3.8e5, 8.5e5, 3.2e6],
    '内外比': [0.79, 1.14, 0.82, 1.11, 0.76, 1.06, 1.17, 1.11, 1.12, 0.88],
    '3日涨幅%': [8.5, -2.3, 12.0, -1.5, 15.0, 5.2, -3.8, 2.5, -5.0, 10.5],
    '6日涨幅%': [12.0, -1.0, 18.0, 3.0, 22.0, 8.0, -1.0, 5.0, -2.0, 15.0],
    '5日涨幅%': [10.0, -1.5, 15.0, 1.0, 18.0, 6.5, -2.0, 3.5, -3.5, 12.0],
    '本月涨幅%': [15.0, 3.0, 20.0, 5.0, 25.0, 10.0, 0.0, 8.0, -5.0, 18.0],
    '今年涨幅%': [25.0, 8.0, 30.0, 12.0, 35.0, 15.0, 5.0, 12.0, -10.0, 28.0],
    '近一月涨幅%': [18.0, 2.0, 22.0, 8.0, 28.0, 12.0, -2.0, 10.0, -8.0, 20.0],
    '近一年涨幅%': [35.0, 15.0, 45.0, 20.0, 50.0, 25.0, 10.0, 20.0, -15.0, 40.0],
    '3日换手%': [9.5, 4.2, 14.0, 16.0, 18.0, 6.0, 11.0, 1.8, 22.0, 24.0],
    '6日换手%': [18.0, 8.5, 25.0, 28.0, 32.0, 12.0, 20.0, 3.5, 38.0, 42.0],
    '5日换手率%': [15.0, 7.0, 22.0, 24.0, 28.0, 10.0, 18.0, 3.0, 33.0, 36.0],
    '10日换手率%': [28.0, 14.0, 40.0, 45.0, 50.0, 20.0, 32.0, 6.0, 55.0, 60.0],
    '主力净流入': [5e7, -2e7, 8e7, -1e7, 1.2e8, 3e7, -5e7, 2e7, -8e7, 1e8],
    '主力净比': [5.2, -3.1, 8.5, -2.0, 12.0, 3.5, -5.8, 2.5, -8.5, 10.5],
    '3日主力净流入': [1.5e8, -5e7, 2.2e8, -3e7, 3.5e8, 1e8, -1.2e8, 8e7, -2e8, 3e8],
    '市盈率': [6.5, 8.2, 28.0, 35.0, 55.0, 10.0, 15.0, 35.0, -5.0, 40.0],
    '市盈率(动)': [6.8, 8.5, 30.0, 38.0, 60.0, 10.5, 16.0, 38.0, -8.0, 45.0],
    '市盈率(TTM)': [6.2, 7.8, 25.0, 32.0, 50.0, 9.5, 14.0, 32.0, -3.0, 38.0],
    '市净率': [0.8, 1.2, 8.0, 5.5, 12.0, 1.5, 4.0, 10.0, 3.5, 6.0],
    '市销率': [2.5, 3.0, 8.5, 3.5, 5.0, 1.8, 2.0, 15.0, 2.5, 12.0],
    '股息率TTM%': [5.2, 3.8, 2.5, 0.8, 0.3, 4.5, 3.0, 1.5, 0.0, 0.5],
    '总股本': [1.94e10, 2.52e10, 3.88e9, 7.1e9, 2.43e9, 1.83e10, 6.97e9, 1.26e9, 5.47e9, 1.32e10],
    '总市值': [2.4e11, 8.8e11, 6.1e11, 2.4e11, 5.2e11, 8.9e11, 4.3e11, 2.2e12, 2.3e11, 2.4e11],
    '流通股本': [1.94e10, 2.52e10, 3.88e9, 7.1e9, 2.17e9, 1.09e10, 6.97e9, 1.26e9, 3.28e9, 1.05e10],
    '流通市值': [2.4e11, 8.8e11, 6.1e11, 2.4e11, 4.7e11, 5.3e11, 4.3e11, 2.2e12, 1.4e11, 1.9e11],
    '人均持股数': [15000, 25000, 8000, 12000, 5000, 20000, 10000, 3000, 7000, 8000],
    '每股收益': [1.92, 4.28, 6.32, 1.06, 4.30, 5.11, 4.45, 54.69, -8.56, 0.49],
    'ROE': [12.5, 15.8, 25.0, 18.0, 22.0, 14.0, 25.0, 30.0, -15.0, 12.0],
    'ROA': [1.0, 1.2, 15.0, 8.0, 10.0, 1.5, 12.0, 20.0, -8.0, 5.0],
    '营业收入同比%': [10.0, 8.0, 15.0, 25.0, 30.0, 5.0, 10.0, 12.0, -20.0, 35.0],
    '净利润同比%': [15.0, 12.0, 20.0, 30.0, 40.0, 8.0, 12.0, 15.0, -50.0, 45.0],
    '扣非净利润同比%': [12.0, 10.0, 18.0, 28.0, 35.0, 6.0, 10.0, 13.0, -55.0, 40.0],
    '销售毛利率%': [45.0, 50.0, 75.0, 18.0, 25.0, 35.0, 28.0, 90.0, 15.0, 60.0],
    '资产负债率': [92.0, 90.0, 30.0, 55.0, 60.0, 88.0, 65.0, 25.0, 70.0, 75.0],
    '首次涨停时间': ['09:35', '10:15', '09:32', '14:30', '09:31', '10:00', '13:30', '09:45', '14:50', '09:33'],
    '最终涨停时间': ['09:35', '10:15', '09:32', '14:55', '09:31', '10:00', '13:30', '09:45', '14:50', '09:33'],
    '封单额': [5e8, 3e8, 8e8, 1e8, 1.2e9, 4e8, 2e8, 2e9, 5e7, 6e8],
    '封单量': [4e7, 8.5e6, 5e6, 3e6, 5.6e6, 8.2e6, 3.2e6, 1.1e6, 1.2e6, 3.2e7],
    '封成比%': [8.0, 2.8, 2.5, 0.8, 3.7, 2.3, 1.3, 1.4, 0.7, 5.3],
    '封流比%': [20.0, 10.0, 25.0, 5.0, 30.0, 15.0, 8.0, 35.0, 3.0, 22.0],
    '涨停开板次数': [0, 0, 0, 2, 0, 0, 1, 0, 3, 0],
    '今年累计涨停天数': [5, 3, 8, 4, 12, 2, 3, 1, 6, 10],
    '几天几板': ['1天1板', '2天2板', '1天1板', '1天1板', '3天3板', '1天1板', '1天1板', '1天1板', '1天1板', '2天2板'],
    '竞价涨幅%': [2.0, -1.0, 3.0, 0.5, 5.0, 1.5, -0.5, 0.8, -2.0, 3.5],
    '竞价换手率%': [0.3, 0.1, 0.2, 0.4, 0.5, 0.2, 0.3, 0.05, 0.6, 0.8],
    '竞价实际换手率%': [0.35, 0.12, 0.25, 0.45, 0.55, 0.22, 0.35, 0.06, 0.65, 0.85],
    '竞价量': [5e5, 3e5, 2e5, 4e5, 1.5e5, 3.5e5, 2.5e5, 8e4, 1.8e5, 6e5],
    '竞价金额': [6.3e6, 1.05e7, 3.2e7, 1.34e7, 3.33e7, 1.71e7, 1.54e7, 1.4e8, 7.56e6, 1.13e7],
    '未匹配量': [1e5, -5e4, 2e5, -3e4, 3e5, 1e5, -8e4, 5e4, -1.5e5, 2.5e5],
    '未匹配金额': [1.26e6, -1.75e6, 3.2e7, -1.01e6, 6.66e7, 4.9e6, -4.92e6, 8.75e6, -6.3e6, 4.63e6],
}

demo_df = pd.DataFrame(demo_data)
print(f'演示数据已创建，共 {len(demo_df)} 只股票, {len(demo_df.columns)} 个字段')


# In[4]:


# 使用演示数据运行完整分析流程
raw_df = demo_df
cleaned_df = clean_data(raw_df, filter_st=False, filter_yizhi=False)

if len(cleaned_df) > 0:
    supplemented_df = supplement_jq_data(cleaned_df)
    price_df = get_price_data(supplemented_df)
    factor_df = calc_factors(price_df)
    n_day_df = analyze_N_day(factor_df)
    classified_df, strong_df, neutral_df, weak_df = classify_stock(factor_df)
    scored_df = score_stock(classified_df)
    corr_df = correlation_analysis(scored_df)
    predict_df = predict_next_day(scored_df)
    output_results(cleaned_df, n_day_df, strong_df, neutral_df, weak_df,
                   scored_df, corr_df, predict_df)


# ---
# 
# ## 附录：JQ API 字段替代说明
# 
# | 需补充指标 | JQ API | 替代方案（如不可用） |
# |---|---|---|
# | T+N 收盘价 | `get_price()` | 使用文件中'最新'字段近似 |
# | MA5/MA10 | `get_price()` + `mean()` | 使用近5/10日'均价'近似 |
# | 5日均线乖离率 | `(close-MA5)/MA5` | 使用'3日涨幅%'近似趋势 |
# | N日最大回撤 | `get_price()` low | 使用'振幅%'近似 |
# | N日最大涨幅 | `get_price()` high | 使用'涨幅%'近似 |
# | 连续上涨判断 | 逐日比较 | 使用'连涨天数'字段 |
# | 主力净流入 | `get_money_flow()` | 使用文件字段'主力净流入' |
# 
# > ⚠️ 本程序仅供研究参考，不构成投资建议。

# In[ ]:




