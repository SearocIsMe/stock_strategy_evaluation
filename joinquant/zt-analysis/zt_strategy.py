#!/usr/bin/env python
# coding: utf-8

# # 涨停板交易策略 (ZT Strategy)
# # JoinQuant 交易策略程序
#
# ## 核心逻辑
# 1. 每日盘前获取昨日涨停股，更新股票池
# 2. 运行分析管线（因子→分类→评分→预测）
# 3. 生成建仓信号（4种类型）
# 4. 盘中按规则执行建仓/止盈/止损
#
# ## 建仓类型
# - TYPE_A: 强势+积极建仓 → 开盘50% + MA5回踩50%
# - TYPE_B: 平稳+积极建仓 → 10点后金叉全仓
# - TYPE_C: 强势+适度建仓 → 10点后金叉半仓 + MA5/MA10半仓
# - NO_ENTRY: 其他组合不建仓
#
# ## 止盈条件（任一触发）
# - 最大持仓天数 (默认5天)
# - T+1利润 > 9%
# - 移动止盈 (从最高价回撤 > 3%)
# - 达到目标价
#
#
# ## 止损条件（任一触发）
# - 日内亏损 > 5%
# - 崩盘检测 (涨跌比 < 1:4)
# - 跌破止损价

# ============================================================================
# Section 1: 环境导入
# ============================================================================

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import traceback
import warnings
warnings.filterwarnings('ignore')

# scipy.stats for IC calculation (Spearman correlation)
try:
    from scipy import stats
except ImportError:
    stats = None  # IC监控功能将不可用

# JoinQuant 环境导入
try:
    from jqdata import *
except ImportError:
    pass  # 策略文件仅在 JQ 环境中运行


# ============================================================================
# Section 2: 策略配置
# ============================================================================

STRATEGY_CONFIG = {
    # --- 股票池 ---
    'pool_max_size': 100,           # 股票池最大容量
    'pool_zt_expire_days': 10,      # ZT日超过此天数则淘汰
    'min_list_days': 63,            # 上市不足此天数则过滤 (约3个月)

    # --- 建仓 ---
    'max_entry_count': 5,           # 每日最大建仓数
    'max_holdings': 5,              # 最大同时持仓数
    'zt_count_threshold': 30,       # 昨日ZT数<=此值则不交易
    'min_score': 40,                # 建仓最低评分 (可配置, 8维模型分数分布更分散)
    'min_entry_index': 55,          # 建仓最低建仓指数 (可配置, 与"适度建仓"信号阈值对齐)

    # --- 止盈 ---
    'max_hold_days': 5,             # 最大持仓天数
    't1_profit_take_pct': 0.09,     # T+1利润>9%止盈
    'trailing_stop_pct': 0.04,      # 从最高价回撤4%移动止盈（让利润奔跑）

    # --- 止损 ---
    'daily_stop_loss_pct': 0.05,    # 日内亏损>5%止损
    'crash_ratio': 4,               # 涨跌比 < 1:4 判定崩盘
    'crash_check_time': {'hour': 11, 'minute': 25},  # 崩盘检测时间

    # --- 技术指标 ---
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'kdj_n': 9,
    'kdj_m1': 3,
    'kdj_m2': 3,
    'ma5_bias_threshold': 0.01,     # MA5回踩阈值: 距MA5<1%视为回踩
    'ma10_bias_threshold': 0.015,   # MA10回踩阈值

    # --- 评分 (与 zt_analysis.py CONFIG 一致) ---
    'score_weights': {
        'price_strength': 30,
        'trend_structure': 20,
        'volume': 20,
        'capital_flow': 15,
        'fundamental': 10,
        'risk_deduction': 5,
        'alpha_factors': 5,         # 量价背离/波动率 Alpha 信号
        'zt_exclusive': 5,          # 涨停板专属因子
    },
    'defense_line': -0.03,
    'zt_threshold': 9.8,
    'N_days': [1, 2, 3, 4, 5],
}


# ============================================================================
# Section 3: 工具函数 (从 zt_analysis.py 复用)
# ============================================================================

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    安全地从 DataFrame 获取一列作为 Series。
    处理重复列名导致 df[col] 返回 DataFrame 的情况。
    若列不存在，返回全 NaN 的 Series。
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def _safe_get(df: pd.DataFrame, col: str, default: Optional[float] = np.nan) -> Optional[float]:
    """
    类似 df.get(col, default)，但处理重复列名。
    列不存在时返回 default，存在时返回首个 Series。
    """
    if col not in df.columns:
        return default
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去除重复列名，保留首次出现的列。"""
    return df.loc[:, ~df.columns.duplicated()]


def _normalize_jq_code(code_str: str) -> str:
    """
    将各种格式的股票代码标准化为 JoinQuant 格式。
    例: '000001' → '000001.XSHE', '600000' → '600000.XSHG'
    """
    code_str = str(code_str).strip().upper()
    if '.XSHE' in code_str or '.XSHG' in code_str:
        return code_str
    code_str = code_str.replace('SH', '').replace('SZ', '').replace('BJ', '')
    code_str = code_str.replace('.', '')
    code_str = code_str.zfill(6)
    if code_str.startswith(('6', '9', '5')):
        return f"{code_str}.XSHG"
    elif code_str.startswith(('0', '1', '2', '3')):
        return f"{code_str}.XSHE"
    elif code_str.startswith(('4', '8')):
        return f"{code_str}.XSHE"  # 北交所暂归XSHE
    else:
        return f"{code_str}.XSHE"


def _normalize_price_df_time(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容不同版本 JoinQuant / pandas 的 get_price 返回格式。
    确保返回的 DataFrame 有一个 'time' 列（datetime 类型）。
    """
    if 'time' in price_df.columns:
        price_df['time'] = pd.to_datetime(price_df['time'])
        return price_df
    if price_df.index.name in ('time', 'date', 'Time', 'Date') or isinstance(price_df.index, pd.DatetimeIndex):
        price_df = price_df.reset_index()
        for col in price_df.columns:
            if col.lower() in ('time', 'date', 'index'):
                price_df = price_df.rename(columns={col: 'time'})
                break
        if 'time' in price_df.columns:
            price_df['time'] = pd.to_datetime(price_df['time'])
        return price_df
    if isinstance(price_df.index, pd.DatetimeIndex):
        price_df = price_df.reset_index()
        if 'index' in price_df.columns:
            price_df = price_df.rename(columns={'index': 'time'})
        price_df['time'] = pd.to_datetime(price_df['time'])
        return price_df
    first_col = price_df.columns[0]
    try:
        price_df['time'] = pd.to_datetime(price_df[first_col])
    except Exception:
        pass
    return price_df


# ============================================================================
# Section 4: 涨停股筛选与过滤
# ============================================================================

def get_yesterday_zt_stocks(context) -> pd.DataFrame:
    """
    获取昨日涨停股列表。

    通过 JQ API 获取全A股票昨日涨跌幅，筛选涨幅 >= zt_threshold 的股票。

    Returns
    -------
    pd.DataFrame
        昨日涨停股，包含 jq_code, code, name, zt_date, zt_close, pct_change 等列
    """
    yesterday = context.previous_date

    # 获取全A股票列表
    all_stocks = get_all_securities('stock', date=yesterday)
    stock_codes = all_stocks.index.tolist()

    if not stock_codes:
        return pd.DataFrame()

    # 批量获取昨日行情 (分批，每批200只)
    batch_size = 200
    all_prices = []

    for i in range(0, len(stock_codes), batch_size):
        batch = stock_codes[i:i + batch_size]
        try:
            prices = get_price(
                batch,
                end_date=yesterday,
                count=1,
                frequency='daily',
                fields=['close', 'pre_close', 'high', 'low', 'volume', 'money', 'open'],
                panel=False,
                skip_paused=True
            )
            if prices is not None and not prices.empty:
                # JQ get_price 不支持 pct_change 字段，手动计算
                prices['pct_change'] = (prices['close'] - prices['pre_close']) / prices['pre_close'] * 100
                all_prices.append(prices)
        except Exception as e:
            log.info(f"[get_yesterday_zt_stocks] 批量获取行情失败: {e}")

    if not all_prices:
        return pd.DataFrame()

    price_df = pd.concat(all_prices, ignore_index=True)

    # 筛选涨停股 (涨幅 >= zt_threshold 且 收盘价=最高价，即封涨停)
    zt_df = price_df[
        (price_df['pct_change'] >= STRATEGY_CONFIG['zt_threshold']) &
        (price_df['close'] == price_df['high'])
    ].copy()

    if zt_df.empty:
        return pd.DataFrame()

    # 标准化格式: 确保 code 列存在
    if 'code' not in zt_df.columns:
        zt_df = zt_df.reset_index()
        for col in zt_df.columns:
            if col.lower() in ('code', 'level_0'):
                zt_df = zt_df.rename(columns={col: 'code'})
                break

    result = pd.DataFrame()
    result['jq_code'] = zt_df['code'].values if 'code' in zt_df.columns else zt_df.iloc[:, 0].values
    result['code'] = result['jq_code'].apply(lambda x: str(x).split('.')[0] if '.' in str(x) else str(x))
    result['name'] = result['jq_code'].apply(
        lambda x: all_stocks.loc[x].display_name if x in all_stocks.index else ''
    )
    result['zt_date'] = yesterday
    result['zt_close'] = zt_df['close'].values
    result['pct_change'] = zt_df['pct_change'].values
    result['volume'] = zt_df['volume'].values if 'volume' in zt_df.columns else np.nan
    result['money'] = zt_df['money'].values if 'money' in zt_df.columns else np.nan
    result['open'] = zt_df['open'].values if 'open' in zt_df.columns else np.nan
    result['high'] = zt_df['high'].values if 'high' in zt_df.columns else np.nan
    result['low'] = zt_df['low'].values if 'low' in zt_df.columns else np.nan
    result['pre_close'] = zt_df['pre_close'].values if 'pre_close' in zt_df.columns else np.nan

    result = result.reset_index(drop=True)

    log.info(f"[get_yesterday_zt_stocks] 昨日涨停股: {len(result)} 只")

    return result


def filter_stocks(context, stock_codes: List[str]) -> List[str]:
    """
    过滤ST股和上市不足3个月的股票。

    Parameters
    ----------
    context : JQ context
    stock_codes : list
        JQ格式股票代码列表

    Returns
    -------
    list
        过滤后的股票代码列表
    """
    if not stock_codes:
        return []

    yesterday = context.previous_date

    # 1. 过滤ST股 (双重检测: API + 名称)
    st_codes = set()

    # 1a. 用 get_extras 检测ST
    try:
        st_flags = get_extras('is_st', stock_codes, end_date=yesterday, count=1)
        if st_flags is not None and not st_flags.empty:
            st_row = st_flags.iloc[-1]
            for code in stock_codes:
                if code in st_row.index and st_row[code]:
                    st_codes.add(code)
    except Exception:
        pass

    # 1b. 用名称检测ST (补充: 部分情况API可能不准确)
    try:
        all_stocks = get_all_securities('stock', date=yesterday)
        for code in stock_codes:
            if code in all_stocks.index:
                name = all_stocks.loc[code].display_name
                if 'ST' in str(name) or 'st' in str(name).lower():
                    st_codes.add(code)
    except Exception:
        pass

    non_st_codes = [code for code in stock_codes if code not in st_codes]

    # 2. 过滤上市不足3个月的股票
    min_date = yesterday - timedelta(days=STRATEGY_CONFIG['min_list_days'])
    filtered = []
    for code in non_st_codes:
        try:
            info = get_security_info(code)
            if info.start_date <= min_date:
                filtered.append(code)
        except Exception:
            continue

    removed_count = len(stock_codes) - len(filtered)
    if removed_count > 0:
        log.info(f"[filter_stocks] 过滤掉 {removed_count} 只股票 (ST/次新)，剩余 {len(filtered)} 只")

    return filtered


# ============================================================================
# Section 5: 股票池管理
# ============================================================================

def update_stock_pool(context, new_zt_df: pd.DataFrame) -> None:
    """
    将新ZT股加入股票池，去重。如果股票已在池中且再次涨停，更新ZT信息。

    Parameters
    ----------
    context : JQ context
    new_zt_df : pd.DataFrame
        get_yesterday_zt_stocks() 返回的新ZT股数据
    """
    if new_zt_df.empty:
        return

    pool = g.stock_pool

    # ---- 批量更新：分离已有股票和新增股票 ----
    existing_codes = set(pool['jq_code'].values)
    update_cols = ['zt_date', 'zt_close', 'pct_change']

    # 已有股票：批量 loc 更新
    existing_mask = new_zt_df['jq_code'].isin(existing_codes)
    existing_updates = new_zt_df[existing_mask]

    if not existing_updates.empty:
        for _, row in existing_updates.iterrows():
            code = row.get('jq_code', '')
            pool_idx = pool.index[pool['jq_code'] == code][0]
            for col in update_cols:
                pool.at[pool_idx, col] = row.get(col, np.nan if col != 'pct_change' else 0)
            log.info(f"[update_stock_pool] 更新 {code} ZT信息 (再次涨停)")

    # 新增股票：批量 concat
    new_stocks = new_zt_df[~existing_mask]
    if not new_stocks.empty:
        new_rows = pd.DataFrame({
            'jq_code': new_stocks['jq_code'].values,
            'code': new_stocks.get('code', '').values,
            'name': new_stocks.get('name', '').values,
            'zt_date': new_stocks.get('zt_date', np.nan).values,
            'zt_close': new_stocks.get('zt_close', np.nan).values,
            'pct_change': new_stocks.get('pct_change', 0).values,
            'total_score': np.nan,
            'classification': '',
            'signal': '',
            'entry_index': np.nan,
            'buy_price': np.nan,
            'stop_loss': np.nan,
            'target_price': np.nan,
        })
        pool = pd.concat([pool, new_rows], ignore_index=True)

    g.stock_pool = pool
    log.info(f"[update_stock_pool] 股票池更新后大小: {len(g.stock_pool)}")


def prune_stock_pool(context) -> None:
    """
    按规则淘汰股票池中的股票。

    淘汰优先级:
    0. ST股票
    1. 信号为"不建议建仓"的股票
    2. ZT日期超过 pool_zt_expire_days 天的股票
    3. 总评分最低的股票 (直到池大小 <= pool_max_size)
    """
    pool = g.stock_pool
    if pool.empty:
        return

    today = context.current_dt.date()
    initial_size = len(pool)

    # 0. 淘汰ST股票 (双重检测: API + 名称)
    if 'jq_code' in pool.columns and len(pool) > 0:
        st_codes = set()
        stock_codes = pool['jq_code'].tolist()

        # 0a. 用 get_extras 检测ST
        try:
            st_flags = get_extras('is_st', stock_codes, end_date=context.previous_date, count=1)
            if st_flags is not None and not st_flags.empty:
                st_row = st_flags.iloc[-1]
                for code in stock_codes:
                    if code in st_row.index and st_row[code]:
                        st_codes.add(code)
        except Exception:
            pass

        # 0b. 用名称检测ST (补充)
        try:
            all_stocks = get_all_securities('stock', date=context.previous_date)
            for code in stock_codes:
                if code in all_stocks.index:
                    name = all_stocks.loc[code].display_name
                    if 'ST' in str(name) or 'st' in str(name).lower():
                        st_codes.add(code)
        except Exception:
            pass

        # 0c. 用池中name列检测ST (补充)
        if 'name' in pool.columns:
            name_st_mask = pool['name'].astype(str).str.contains('ST', case=False, na=False)
            for idx in pool[name_st_mask].index:
                st_codes.add(pool.at[idx, 'jq_code'])

        if st_codes:
            st_mask = pool['jq_code'].isin(st_codes)
            removed_st = st_mask.sum()
            pool = pool[~st_mask].copy()
            if removed_st > 0:
                log.info(f"[prune_stock_pool] 淘汰 {removed_st} 只ST股票")

    if pool.empty:
        g.stock_pool = pool.reset_index(drop=True)
        return

    # 1. 淘汰"不建议建仓"信号的股票
    if 'signal' in pool.columns:
        no_entry_mask = pool['signal'].str.contains('不建议', na=False)
        removed_no_entry = no_entry_mask.sum()
        pool = pool[~no_entry_mask].copy()
        if removed_no_entry > 0:
            log.info(f"[prune_stock_pool] 淘汰 {removed_no_entry} 只'不建议建仓'股票")

    # 2. 淘汰ZT日期过期的股票
    if 'zt_date' in pool.columns and len(pool) > 0:
        expire_days = STRATEGY_CONFIG['pool_zt_expire_days']
        pool['zt_date_parsed'] = pd.to_datetime(pool['zt_date'], errors='coerce')
        expired_mask = (today - pool['zt_date_parsed'].dt.date) > timedelta(days=expire_days)
        removed_expired = expired_mask.sum()
        pool = pool[~expired_mask].copy()
        if removed_expired > 0:
            log.info(f"[prune_stock_pool] 淘汰 {removed_expired} 只ZT日过期股票")
        pool = pool.drop(columns=['zt_date_parsed'], errors='ignore')

    # 3. 如果池仍超过最大容量，按评分淘汰最低的
    max_size = STRATEGY_CONFIG['pool_max_size']
    if len(pool) > max_size:
        if 'total_score' in pool.columns:
            pool = pool.sort_values('total_score', ascending=False, na_position='last').head(max_size).copy()
        else:
            pool = pool.head(max_size).copy()
        log.info(f"[prune_stock_pool] 按评分淘汰至 {max_size} 只")

    g.stock_pool = pool.reset_index(drop=True)
    log.info(f"[prune_stock_pool] 池大小: {initial_size} → {len(g.stock_pool)}")


# ============================================================================
# Section 6: 数据获取与因子计算
# ============================================================================

def build_stock_data(context, pool_df: pd.DataFrame) -> pd.DataFrame:
    """
    为股票池中的股票构建分析数据。
    整合了 zt_analysis.py 中 get_price_data + supplement_jq_data 的逻辑。

    关键：对于昨日涨停股（无涨停后数据），使用涨停日当天数据
    和涨停前数据来计算可用的指标，确保评分/分类/信号不为全NaN。

    Parameters
    ----------
    context : JQ context
    pool_df : pd.DataFrame
        股票池数据，至少包含 jq_code, zt_date, zt_close 列

    Returns
    -------
    pd.DataFrame
        包含行情数据和补充数据的完整 DataFrame
    """
    if pool_df.empty:
        return pd.DataFrame()

    df = pool_df.copy()
    n_days = 5

    log.info(f"[build_stock_data] 开始构建 {len(df)} 只股票的数据")

    # ---- 获取行情数据 ----
    price_data_list = []

    for idx, row in df.iterrows():
        jq_code = row.get('jq_code', '')
        zt_date = row.get('zt_date', None)
        zt_close = row.get('zt_close', np.nan)

        if pd.isna(zt_date) or not jq_code:
            price_data_list.append({})
            continue

        zt_date_pd = pd.to_datetime(zt_date)
        zt_date_str = zt_date_pd.strftime('%Y-%m-%d')

        try:
            stock_price_data = {}

            # ======== 1. 获取涨停日当天数据 ========
            zt_day_df = get_price(
                jq_code,
                end_date=zt_date_str,
                count=1,
                frequency='daily',
                fields=['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close'],
                panel=False,
                skip_paused=True
            )

            if zt_day_df is not None and not zt_day_df.empty:
                zt_row_data = zt_day_df.iloc[-1]
                if pd.isna(zt_close):
                    zt_close = zt_row_data['close']
                # 记录涨停日成交量/成交额（供因子计算使用）
                stock_price_data['volume'] = zt_row_data['volume']
                stock_price_data['money'] = zt_row_data['money']
                # 涨停日振幅
                if zt_row_data['pre_close'] > 0:
                    stock_price_data['amplitude'] = (zt_row_data['high'] - zt_row_data['low']) / zt_row_data['pre_close'] * 100
                else:
                    stock_price_data['amplitude'] = np.nan
                # 涨停日量比 (用 volume / pre_volume 近似，暂设NaN由supplement补充)
                stock_price_data['vol_ratio'] = np.nan

            stock_price_data['zt_close_jq'] = zt_close

            # ======== 2. 获取涨停日及后 N 天的数据 ========
            end_date_str = (zt_date_pd + timedelta(days=n_days * 2 + 10)).strftime('%Y-%m-%d')
            price_df = get_price(
                jq_code,
                start_date=zt_date_str,
                end_date=end_date_str,
                frequency='daily',
                fields=['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close'],
                panel=False,
                fill_paused=False,
                skip_paused=True
            )

            after_zt = None
            if price_df is not None and not price_df.empty:
                price_df = _normalize_price_df_time(price_df)
                mask = price_df['time'] >= zt_date_pd
                after_zt = price_df[mask].head(n_days + 1)

            has_post_zt = (after_zt is not None and len(after_zt) > 1)

            # ======== 3. 计算 N 日收益数据 ========
            for n in STRATEGY_CONFIG['N_days']:
                if after_zt is not None and len(after_zt) > n:
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
                    for key in [f'close_n{n}', f'high_n{n}', f'low_n{n}',
                                f'volume_n{n}', f'amount_n{n}', f'return_n{n}',
                                f'max_return_n{n}', f'max_drawdown_n{n}']:
                        stock_price_data[key] = np.nan

            # ======== 4. MA5 / MA10 (始终计算，使用涨停日及之前数据) ========
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
                else:
                    stock_price_data['ma5'] = np.nan
                if ma_df is not None and len(ma_df) >= 10:
                    stock_price_data['ma10'] = ma_df['close'].tail(10).mean()
                else:
                    stock_price_data['ma10'] = np.nan
            except Exception:
                stock_price_data['ma5'] = np.nan
                stock_price_data['ma10'] = np.nan

            # ======== 5. 乖离率 (使用涨停日收盘价和MA5) ========
            if not pd.isna(stock_price_data.get('ma5')) and stock_price_data.get('ma5', 0) > 0:
                stock_price_data['bias_ma5'] = (zt_close - stock_price_data['ma5']) / stock_price_data['ma5']
            else:
                stock_price_data['bias_ma5'] = np.nan

            # ======== 6. 涨停前连续上涨天数 ========
            try:
                pre_df = get_price(
                    jq_code,
                    end_date=zt_date_str,
                    frequency='daily',
                    fields=['close', 'pre_close'],
                    count=10,
                    panel=False,
                    skip_paused=True
                )
                consecutive_up = 0
                if pre_df is not None and len(pre_df) > 1:
                    for i in range(len(pre_df) - 1, 0, -1):
                        if pre_df.iloc[i]['close'] > pre_df.iloc[i]['pre_close']:
                            consecutive_up += 1
                        else:
                            break
                stock_price_data['consecutive_up'] = consecutive_up
            except Exception:
                stock_price_data['consecutive_up'] = 0

            # ======== 7. 涨停后综合指标 ========
            if has_post_zt:
                post_zt = after_zt.iloc[1:]  # 涨停日之后的数据

                # N日最大回撤/涨幅（跨所有N日）
                if len(post_zt) > 0 and zt_close > 0:
                    stock_price_data['overall_max_drawdown'] = (post_zt['low'].min() - zt_close) / zt_close
                    stock_price_data['overall_max_return'] = (post_zt['high'].max() - zt_close) / zt_close
                else:
                    stock_price_data['overall_max_drawdown'] = np.nan
                    stock_price_data['overall_max_return'] = np.nan

                # 涨停后连续上涨天数
                consecutive_up_post = 0
                for _, day_row in post_zt.iterrows():
                    if day_row['close'] > day_row['pre_close']:
                        consecutive_up_post += 1
                    else:
                        break
                stock_price_data['consecutive_up_post'] = consecutive_up_post

                # 收盘价 > 涨停价的天数
                stock_price_data['days_above_zt'] = int((post_zt['close'] > zt_close).sum())
                stock_price_data['ratio_above_zt'] = (post_zt['close'] > zt_close).sum() / len(post_zt) if len(post_zt) > 0 else 0

                # 收盘价 > 涨停价*0.97 的天数（-3%防守线）
                stock_price_data['days_above_defense'] = int((post_zt['close'] > zt_close * 0.97).sum())
                stock_price_data['ratio_above_defense'] = (post_zt['close'] > zt_close * 0.97).sum() / len(post_zt) if len(post_zt) > 0 else 0
            else:
                # 昨日涨停股：无涨停后数据，使用涨停日当天作为参考
                # 涨停日收盘价 = zt_close，所以 ratio_above_zt = 1.0
                stock_price_data['overall_max_drawdown'] = 0  # 涨停日无回撤
                stock_price_data['overall_max_return'] = 0    # 涨停日无额外涨幅
                stock_price_data['consecutive_up_post'] = 0
                stock_price_data['days_above_zt'] = 0
                stock_price_data['ratio_above_zt'] = 1.0      # 涨停日收盘=涨停价
                stock_price_data['days_above_defense'] = 0
                stock_price_data['ratio_above_defense'] = 1.0  # 涨停价 > 防守线

            price_data_list.append(stock_price_data)

        except Exception as e:
            log.info(f"[build_stock_data] 获取 {jq_code} 行情失败: {e}")
            price_data_list.append({})

    # 合并行情数据
    price_df_result = pd.DataFrame(price_data_list)
    df = pd.concat([df, price_df_result], axis=1)
    df = _dedup_columns(df)

    # ---- 补充基本面和资金数据 ----
    df = _supplement_jq_data_strategy(context, df)

    log.info(f"[build_stock_data] 数据构建完成")

    return df


def _supplement_jq_data_strategy(context, df: pd.DataFrame) -> pd.DataFrame:
    """
    补充基本面、资金流向等数据 (策略版)。
    仅填充 NaN 的字段，不覆盖已有数据。
    """
    yesterday = context.previous_date

    # ---- 预初始化可能缺失的列，防止后续 classify_stock() 等函数 KeyError ----
    for _col in ['main_net_inflow', 'main_net_pct', 'turnover_rate',
                 'pe_ttm', 'pb_ratio', 'market_cap', 'float_market_cap',
                 'eps', 'roe', 'roa', 'gross_margin',
                 'revenue_yoy', 'profit_yoy', 'debt_ratio']:
        if _col not in df.columns:
            df[_col] = np.nan

    for idx, row in df.iterrows():
        jq_code = row.get('jq_code', '')
        if not jq_code or pd.isna(jq_code):
            continue

        zt_date = row.get('zt_date', None)
        if pd.isna(zt_date):
            query_date = yesterday.strftime('%Y-%m-%d')
        else:
            query_date = pd.to_datetime(zt_date).strftime('%Y-%m-%d')

        # ---- 基本面数据 ----
        try:
            q = query(
                valuation.code,
                valuation.pe_ratio,
                valuation.pb_ratio,
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
                field_map = {
                    'pe_ttm': fund_row.get('pe_ratio', np.nan),
                    'pb_ratio': fund_row.get('pb_ratio', np.nan),
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
                    if field not in df.columns:
                        df.at[idx, field] = value
                    elif pd.isna(df.at[idx, field]):
                        df.at[idx, field] = value
        except Exception:
            pass

        # ---- 资金流向数据 ----
        try:
            if 'main_net_inflow' not in df.columns or pd.isna(row.get('main_net_inflow', np.nan)):
                zt_date_str = query_date
                money_df = get_money_flow([jq_code], start_date=zt_date_str, end_date=zt_date_str)
                if money_df is not None and not money_df.empty:
                    net_inflow = (money_df['sec_large_net_inflow'].sum() +
                                  money_df['large_net_inflow'].sum())
                    if 'main_net_inflow' not in df.columns:
                        df.at[idx, 'main_net_inflow'] = net_inflow
                    elif pd.isna(df.at[idx, 'main_net_inflow']):
                        df.at[idx, 'main_net_inflow'] = net_inflow
        except Exception:
            pass

        # ---- 换手率 (从日K获取) ----
        try:
            if 'turnover_rate' not in df.columns or pd.isna(row.get('turnover_rate', np.nan)):
                day_df = get_price(
                    jq_code,
                    end_date=query_date,
                    count=1,
                    frequency='daily',
                    fields=['close'],
                    panel=False,
                    skip_paused=True
                )
                # JQ get_price 不直接返回换手率，用 get_fundamentals 补充
                q2 = query(
                    valuation.code,
                    valuation.turnover_ratio,
                ).filter(valuation.code == jq_code)
                turn_df = get_fundamentals(q2, date=query_date)
                if turn_df is not None and not turn_df.empty:
                    tr = turn_df.iloc[0].get('turnover_ratio', np.nan)
                    if 'turnover_rate' not in df.columns:
                        df.at[idx, 'turnover_rate'] = tr
                    elif pd.isna(df.at[idx, 'turnover_rate']):
                        df.at[idx, 'turnover_rate'] = tr
        except Exception:
            pass

    return df


# ============================================================================
# Section 7: 因子计算 (从 zt_analysis.py 复用)
# ============================================================================

def calc_factors(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算多维度因子，为评分模型提供输入。

    Parameters
    ----------
    price_df : pd.DataFrame
        build_stock_data() 输出的含行情数据

    Returns
    -------
    pd.DataFrame
        增加了因子列的 DataFrame
    """
    df = price_df.copy()

    # ---- 4.1 价格强度因子 ----
    df['factor_return_3d'] = _safe_get(df, 'return_n3', np.nan)
    df['factor_return_5d'] = _safe_get(df, 'return_n5', np.nan)
    df['factor_max_return'] = _safe_get(df, 'overall_max_return', np.nan)
    df['factor_defense_ratio'] = _safe_get(df, 'ratio_above_defense', np.nan)
    df['factor_above_zt_ratio'] = _safe_get(df, 'ratio_above_zt', np.nan)

    # ---- 4.2 趋势结构因子 ----
    if 'close_n3' in df.columns and 'ma5' in df.columns:
        _c3 = _safe_series(df, 'close_n3').astype(float, errors='ignore')
        _m5 = _safe_series(df, 'ma5').astype(float, errors='ignore')
        df['factor_ma5_position'] = np.nan
        valid = _m5 > 0
        df.loc[valid, 'factor_ma5_position'] = _c3[valid] / _m5[valid] - 1
    else:
        df['factor_ma5_position'] = np.nan

    df['factor_bias_ma5'] = _safe_get(df, 'bias_ma5', np.nan)

    if 'consecutive_up_post' in df.columns:
        df['factor_consecutive_up'] = _safe_get(df, 'consecutive_up', 0) + _safe_series(df, 'consecutive_up_post')
    else:
        df['factor_consecutive_up'] = _safe_get(df, 'consecutive_up', 0)

    df['factor_pct_3d'] = _safe_get(df, 'pct_3d', np.nan)
    df['factor_pct_5d'] = _safe_get(df, 'pct_5d', np.nan)

    # ---- 4.3 成交量因子 ----
    df['factor_vol_ratio'] = _safe_get(df, 'vol_ratio', np.nan)
    df['factor_turnover'] = _safe_get(df, 'turnover_rate', np.nan)
    df['factor_inner_outer'] = _safe_get(df, 'inner_outer_ratio', np.nan)

    if 'return_n1' in df.columns and 'volume_n1' in df.columns and 'volume' in df.columns:
        _r1 = _safe_series(df, 'return_n1').astype(float, errors='ignore')
        _vn1 = _safe_series(df, 'volume_n1').astype(float, errors='ignore')
        _vol = _safe_series(df, 'volume').astype(float, errors='ignore').replace(0, np.nan)
        vol_change = _vn1 / _vol
        df['factor_vol_price'] = np.nan
        pos = _r1 > 0
        neg = _r1 <= 0
        df.loc[pos, 'factor_vol_price'] = vol_change[pos]
        df.loc[neg, 'factor_vol_price'] = -vol_change[neg]
    else:
        df['factor_vol_price'] = np.nan

    # ---- 4.4 资金因子 ----
    df['factor_main_net_inflow'] = _safe_get(df, 'main_net_inflow', np.nan)
    df['factor_main_net_pct'] = _safe_get(df, 'main_net_pct', np.nan)
    df['factor_main_net_3d'] = _safe_get(df, 'main_net_inflow_3d', np.nan)

    # ---- 4.5 基本面因子 ----
    _pe = _safe_get(df, 'pe_ttm', None)
    if _pe is None or (isinstance(_pe, pd.Series) and _pe.isna().all()):
        _pe = _safe_get(df, 'pe_ratio', np.nan)
    df['factor_pe'] = _pe
    df['factor_roe'] = _safe_get(df, 'roe', np.nan)
    df['factor_profit_yoy'] = _safe_get(df, 'profit_yoy', np.nan)
    df['factor_gross_margin'] = _safe_get(df, 'gross_margin', np.nan)

    # ---- 4.6 风险因子（扣分项） ----
    df['factor_max_drawdown'] = _safe_get(df, 'overall_max_drawdown', np.nan)
    df['factor_zt_open_count'] = _safe_get(df, 'zt_open_count', np.nan)
    df['factor_amplitude'] = _safe_get(df, 'amplitude', np.nan)

    # ---- 4.7 涨停特征因子（辅助） ----
    df['factor_seal_amount'] = _safe_get(df, 'seal_amount', np.nan)
    df['factor_seal_ratio'] = _safe_get(df, 'seal_volume_ratio', np.nan)
    df['factor_days_boards'] = _safe_get(df, 'days_boards', np.nan)
    df['factor_first_zt_time'] = _safe_get(df, 'first_zt_time', np.nan)

    # ---- 4.8 互补 Alpha 因子 (WorldQuant 101 Alphas 精选) ----
    # Alpha#6: -1 * correlation(open, volume, 10)
    # 量价背离检测：开盘价与成交量的相关性，负相关表示量价背离（看跌信号）
    if all(c in df.columns for c in ['open', 'volume']):
        _open = _safe_series(df, 'open').astype(float, errors='ignore')
        _vol_s = _safe_series(df, 'volume').astype(float, errors='ignore')
        try:
            df['factor_alpha6'] = -1 * _open.rolling(window=10, min_periods=6).corr(_vol_s)
        except Exception:
            df['factor_alpha6'] = np.nan
    else:
        df['factor_alpha6'] = np.nan

    # Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    # 量增价跌/量缩价涨的短期反转信号
    if all(c in df.columns for c in ['close', 'volume']):
        _close_s = _safe_series(df, 'close').astype(float, errors='ignore')
        _vol_s2 = _safe_series(df, 'volume').astype(float, errors='ignore')
        try:
            _delta_vol = _vol_s2.diff(1)
            _delta_close = _close_s.diff(1)
            df['factor_alpha12'] = np.sign(_delta_vol) * (-1 * _delta_close)
        except Exception:
            df['factor_alpha12'] = np.nan
    else:
        df['factor_alpha12'] = np.nan

    # Alpha#33: rank(-1 * (1 - (open / close)))
    # 日内反转：开盘价相对收盘价的位置，rank标准化
    if all(c in df.columns for c in ['open', 'close']):
        _open2 = _safe_series(df, 'open').astype(float, errors='ignore').replace(0, np.nan)
        _close2 = _safe_series(df, 'close').astype(float, errors='ignore')
        try:
            _intraday_ret = 1 - (_open2 / _close2)
            df['factor_alpha33'] = _intraday_ret.rank(pct=True) * -1
        except Exception:
            df['factor_alpha33'] = np.nan
    else:
        df['factor_alpha33'] = np.nan

    # Alpha#54: -1 * ((low - close) * (low^5)) / ((low - high) * (close^5))
    # 日内价格位置加权：衡量收盘价在日内区间中的位置，低收时放大信号
    if all(c in df.columns for c in ['low', 'close', 'high']):
        _low = _safe_series(df, 'low').astype(float, errors='ignore').replace(0, np.nan)
        _close3 = _safe_series(df, 'close').astype(float, errors='ignore').replace(0, np.nan)
        _high = _safe_series(df, 'high').astype(float, errors='ignore')
        try:
            _numerator = (_low - _close3) * (_low ** 5)
            _denominator = (_low - _high).replace(0, np.nan) * (_close3 ** 5)
            df['factor_alpha54'] = -1 * _numerator / _denominator
            df['factor_alpha54'] = df['factor_alpha54'].replace([np.inf, -np.inf], np.nan)
        except Exception:
            df['factor_alpha54'] = np.nan
    else:
        df['factor_alpha54'] = np.nan

    # Alpha#41: (high - low) / close 的滚动均值 — 波动率因子
    if all(c in df.columns for c in ['high', 'low', 'close']):
        _high2 = _safe_series(df, 'high').astype(float, errors='ignore')
        _low2 = _safe_series(df, 'low').astype(float, errors='ignore')
        _close4 = _safe_series(df, 'close').astype(float, errors='ignore').replace(0, np.nan)
        try:
            _range_ratio = (_high2 - _low2) / _close4
            df['factor_alpha41'] = _range_ratio.rolling(window=5, min_periods=3).mean()
        except Exception:
            df['factor_alpha41'] = np.nan
    else:
        df['factor_alpha41'] = np.nan

    # Alpha#49: sum(((high+close)/2 - (low+close)/2)^2, 5) / (5*volume)
    # 日内振幅的平方和归一化 — 波动率/成交量比
    if all(c in df.columns for c in ['high', 'low', 'close', 'volume']):
        _high3 = _safe_series(df, 'high').astype(float, errors='ignore')
        _low3 = _safe_series(df, 'low').astype(float, errors='ignore')
        _close5 = _safe_series(df, 'close').astype(float, errors='ignore')
        _vol_s3 = _safe_series(df, 'volume').astype(float, errors='ignore').replace(0, np.nan)
        try:
            _half_spread = ((_high3 + _close5) / 2 - (_low3 + _close5) / 2) ** 2
            df['factor_alpha49'] = _half_spread.rolling(window=5, min_periods=3).sum() / (5 * _vol_s3)
            df['factor_alpha49'] = df['factor_alpha49'].replace([np.inf, -np.inf], np.nan)
        except Exception:
            df['factor_alpha49'] = np.nan
    else:
        df['factor_alpha49'] = np.nan

    # ---- 4.9 涨停板专属因子 ----
    # 封板速度: 首次涨停时间越早，封板越坚决，预测力越强
    # 将 HH:MM 时间转换为分钟数（09:30=0, 15:00=330），越早值越大
    if 'first_zt_time' in df.columns:
        _zt_time = _safe_series(df, 'first_zt_time')
        try:
            def _parse_zt_time_to_minutes(val) -> Optional[float]:
                """将涨停时间字符串转为距开盘的分钟数，越早封板值越大"""
                if pd.isna(val) or not isinstance(val, str) or ':' not in val:
                    return np.nan
                try:
                    parts = val.strip().split(':')
                    hour, minute = int(parts[0]), int(parts[1])
                    total_minutes = hour * 60 + minute
                    open_minutes = 9 * 60 + 30  # 09:30 开盘
                    elapsed = total_minutes - open_minutes
                    if elapsed < 0:
                        elapsed = 0  # 集合竞价涨停
                    return float(elapsed)
                except (ValueError, IndexError):
                    return np.nan

            _elapsed_minutes = _zt_time.apply(_parse_zt_time_to_minutes)
            # 反转：越早封板，factor_seal_speed 越大 (330-elapsed)
            df['factor_seal_speed'] = 330.0 - _elapsed_minutes
            df.loc[_elapsed_minutes.isna(), 'factor_seal_speed'] = np.nan
        except Exception:
            df['factor_seal_speed'] = np.nan
    else:
        df['factor_seal_speed'] = np.nan

    # 开板次数: 已有 factor_zt_open_count，此处增加二次确认
    # (0次=一字板/秒板，1次=开过1次，2+次=弱势)
    # factor_zt_open_count 已在 4.6 中定义，此处不再重复

    # 涨停板类型: 基于连板天数分类 (首板=0, 2连板=1, 3+连板=2)
    if 'days_boards' in df.columns:
        _db = _safe_series(df, 'days_boards').astype(float, errors='ignore')
        try:
            # 解析 "几天几板" 格式，取连板数
            def _parse_days_boards(val) -> Optional[float]:
                """解析连板数，如 '2天2板' -> 2, '昨日首板' -> 1"""
                if pd.isna(val):
                    return np.nan
                if isinstance(val, (int, float)):
                    return float(val)
                val_str = str(val).strip()
                # 尝试匹配 "N天N板" 格式
                import re
                m = re.search(r'(\d+)天.*?(\d+)板', val_str)
                if m:
                    return float(m.group(2))
                # "昨日首板" / "首板" -> 1
                if '首板' in val_str:
                    return 1.0
                # 纯数字
                try:
                    return float(val_str)
                except ValueError:
                    return np.nan

            _board_count = _db.apply(_parse_days_boards)
            df['factor_zt_board_type'] = 0.0  # 首板
            df.loc[_board_count >= 2, 'factor_zt_board_type'] = 1.0  # 连板
            df.loc[_board_count >= 3, 'factor_zt_board_type'] = 2.0  # 多连板
            df.loc[_board_count.isna(), 'factor_zt_board_type'] = np.nan
        except Exception:
            df['factor_zt_board_type'] = np.nan
    else:
        df['factor_zt_board_type'] = np.nan

    # 封流比: 封单金额 / 流通市值 — 封板强度相对流通盘的占比
    if 'seal_amount' in df.columns and 'float_market_cap' in df.columns:
        _seal = _safe_series(df, 'seal_amount').astype(float, errors='ignore')
        _float_mcap = _safe_series(df, 'float_market_cap').astype(float, errors='ignore').replace(0, np.nan)
        try:
            df['factor_seal_float_ratio'] = _seal / _float_mcap
            df['factor_seal_float_ratio'] = df['factor_seal_float_ratio'].replace([np.inf, -np.inf], np.nan)
        except Exception:
            df['factor_seal_float_ratio'] = np.nan
    else:
        df['factor_seal_float_ratio'] = np.nan

    return df


# ============================================================================
# Section 8: 股票分类 (从 zt_analysis.py 复用，适配 STRATEGY_CONFIG)
# ============================================================================

def classify_stock(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据多维度条件将股票分为强势/平稳/弱势。

    Returns
    -------
    pd.DataFrame
        含 classification 列的完整 DataFrame
    """
    df = factor_df.copy()
    df['classification'] = '平稳'  # 默认平稳

    # ---- 向量化分类 (替代逐行 iterrows+at[]) ----
    # 使用 _safe_series() 防止列不存在时 KeyError 崩溃整个管线
    _ratio_zt = _safe_series(df, 'ratio_above_zt').fillna(0).astype(float)
    _max_dd = _safe_series(df, 'overall_max_drawdown').fillna(0).astype(float)
    _vol_ratio = _safe_series(df, 'vol_ratio').fillna(1).astype(float)
    _main_net = _safe_series(df, 'main_net_inflow').fillna(0).astype(float)
    _turnover = _safe_series(df, 'turnover_rate').fillna(0).astype(float)
    _consec_up = _safe_series(df, 'consecutive_up').fillna(0).astype(float)
    _return_n1 = _safe_series(df, 'return_n1').fillna(0).astype(float)

    _not_break_defense = _max_dd > STRATEGY_CONFIG['defense_line']
    _vol_stable = _vol_ratio >= 0.8
    _main_positive = _main_net >= 0
    _turnover_active = _turnover >= 3
    _high_consec = _consec_up >= 2

    # 强势评分
    strong_score = pd.Series(0, index=df.index, dtype=float)
    strong_score += ((_ratio_zt > 0.6).astype(int) * 2 + (_ratio_zt > 0.4).astype(int) * 1)
    strong_score = strong_score.clip(upper=2)  # 最多2分
    strong_score += _not_break_defense.astype(int) * 2
    strong_score += _vol_stable.astype(int) * 1
    strong_score += _main_positive.astype(int) * 1
    strong_score += _turnover_active.astype(int) * 1
    strong_score += _high_consec.astype(int) * 1

    # 弱势评分
    weak_score = pd.Series(0, index=df.index, dtype=float)
    weak_score += (_ratio_zt < 0.3).astype(int) * 2
    weak_score += (~_not_break_defense).astype(int) * 2
    weak_score += ((_return_n1 < 0) & (_vol_ratio > 1.5)).astype(int) * 2
    weak_score += (_main_net < 0).astype(int) * 1

    # 分类
    df['classification'] = '平稳'
    df.loc[strong_score >= 5, 'classification'] = '强势'
    df.loc[weak_score >= 4, 'classification'] = '弱势'

    return df


# ============================================================================
# Section 9: 评分模型 (从 zt_analysis.py 复用)
# ============================================================================

def score_stock(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建评分模型，对每只股票打分（0~100）。

    评分维度：
      1. 价格强度（30分）
      2. 趋势结构（20分）
      3. 成交量（20分）
      4. 资金（15分）
      5. 基本面（10分）
      6. 风险扣分（5分）
      7. Alpha因子（5分）— 量价背离/波动率信号
      8. 涨停板专属（5分）— 封板速度/涨停板类型/封流比
    """
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

        # ======== 7. Alpha因子（5分）========
        alpha_score = 0

        # 7a. Alpha#6 量价背离（2分）
        alpha6 = row.get('factor_alpha6', np.nan)
        if not pd.isna(alpha6):
            if alpha6 < -0.3:
                alpha_score += 2   # 量价背离明显（看跌信号反转后偏多）
            elif alpha6 < -0.1:
                alpha_score += 1

        # 7b. Alpha#12 短期反转（2分）
        alpha12 = row.get('factor_alpha12', np.nan)
        if not pd.isna(alpha12):
            if alpha12 > 0:
                alpha_score += 2   # 量缩价涨（偏多）
            elif alpha12 > -0.5:
                alpha_score += 1

        # 7c. Alpha#41 波动率（1分）
        alpha41 = row.get('factor_alpha41', np.nan)
        if not pd.isna(alpha41):
            if 0.02 < alpha41 < 0.05:
                alpha_score += 1   # 适度波动
            # 过高或过低不加分

        # ======== 8. 涨停板专属（5分）========
        zt_exclusive_score = 0

        # 8a. 封板速度（2分）
        seal_speed = row.get('factor_seal_speed', np.nan)
        if not pd.isna(seal_speed):
            if seal_speed >= 300:   # 开盘30分钟内封板
                zt_exclusive_score += 2
            elif seal_speed >= 200:  # 上午封板
                zt_exclusive_score += 1

        # 8b. 涨停板类型（2分）
        board_type = row.get('factor_zt_board_type', np.nan)
        if not pd.isna(board_type):
            if board_type >= 2:     # 多连板
                zt_exclusive_score += 2
            elif board_type >= 1:   # 连板
                zt_exclusive_score += 1

        # 8c. 封流比（1分）
        seal_float = row.get('factor_seal_float_ratio', np.nan)
        if not pd.isna(seal_float):
            if seal_float > 0.05:
                zt_exclusive_score += 1   # 封单占流通盘5%以上

        # ======== 总分 ========
        total_score = (price_score + trend_score + vol_score +
                       capital_score + fund_score - risk_deduction +
                       alpha_score + zt_exclusive_score)
        total_score = max(0, min(100, total_score))

        scores.append({
            'price_score': price_score,
            'trend_score': trend_score,
            'vol_score': vol_score,
            'capital_score': capital_score,
            'fund_score': fund_score,
            'risk_deduction': risk_deduction,
            'alpha_score': alpha_score,
            'zt_exclusive_score': zt_exclusive_score,
            'total_score': total_score,
        })

    scores_df = pd.DataFrame(scores, index=df.index)
    df = pd.concat([df, scores_df], axis=1)
    df = _dedup_columns(df)

    # 按 total_score 降序排列
    df = df.sort_values('total_score', ascending=False, na_position='last').reset_index(drop=True)

    return df


# ============================================================================
# Section 10: 次日建仓预测 (从 zt_analysis.py 复用)
# ============================================================================

def predict_next_day(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    基于当前收盘情况，预测未来一天可建仓的股票。

    Returns
    -------
    pd.DataFrame
        次日建仓预测表，按建仓指数降序排列
    """
    df = scored_df.copy()

    # ---- 筛选候选池 ----
    candidates = df[df['classification'].isin(['强势', '平稳'])].copy()

    if len(candidates) == 0:
        return pd.DataFrame()

    # ---- 计算次日建仓指数 (0~100) ----
    predict_scores = []

    for idx, row in candidates.iterrows():
        # === 1. 综合评分权重 (35分) ===
        total_score = row.get('total_score', 0)
        if pd.isna(total_score):
            total_score = 0
        score_component = min(total_score / 100 * 35, 35)

        # === 2. 价格位置权重 (20分) ===
        price_component = 0
        ratio_above_zt = row.get('ratio_above_zt', 0)
        if pd.isna(ratio_above_zt):
            ratio_above_zt = 0

        if ratio_above_zt >= 0.8:
            price_component = 20
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
        main_net_is_nan = pd.isna(main_net)
        if main_net_is_nan:
            main_net = 0
        main_pct = row.get('main_net_pct', 0)
        main_pct_is_nan = pd.isna(main_pct)
        if main_pct_is_nan:
            main_pct = 0

        # 数据缺失时给中性分数(10/15)，避免因API失败惩罚整个建仓指数
        if main_net_is_nan and main_pct_is_nan:
            capital_component = 10  # 数据缺失=中性
        elif main_net > 0 and main_pct > 0:
            capital_component = 15
        elif main_net > 0:
            capital_component = 12
        elif main_net > -5e7:
            capital_component = 8
        else:
            capital_component = 2

        # === 5. 涨停特征权重 (5分) ===
        zt_feature_component = 0
        seal_ratio = row.get('seal_volume_ratio', 0)
        if pd.isna(seal_ratio):
            seal_ratio = 0
        zt_open = row.get('zt_open_count', 0)
        if pd.isna(zt_open):
            zt_open = 0
        consec_up = row.get('consecutive_up', 0)
        if pd.isna(consec_up):
            consec_up = 0

        if seal_ratio > 5:
            zt_feature_component += 2
        elif seal_ratio > 2:
            zt_feature_component += 1.5
        elif seal_ratio > 0:
            zt_feature_component += 1

        if zt_open == 0:
            zt_feature_component += 2
        elif zt_open == 1:
            zt_feature_component += 1

        if consec_up >= 3:
            zt_feature_component += 1
        elif consec_up >= 2:
            zt_feature_component += 0.5

        zt_feature_component = min(zt_feature_component, 5)

        # === 6. Alpha因子权重 (5分) ===
        alpha_component = 0
        _alpha6 = row.get('factor_alpha6', 0)
        if pd.isna(_alpha6):
            _alpha6 = 0
        _alpha12 = row.get('factor_alpha12', 0)
        if pd.isna(_alpha12):
            _alpha12 = 0
        _alpha41 = row.get('factor_alpha41', 0)
        if pd.isna(_alpha41):
            _alpha41 = 0

        # Alpha#6: 量价背离 — 负相关越强(值越负)越看多
        if _alpha6 < -0.3:
            alpha_component += 2
        elif _alpha6 < -0.1:
            alpha_component += 1.5
        elif _alpha6 < 0.1:
            alpha_component += 1
        else:
            alpha_component += 0

        # Alpha#12: 短期反转 — 正值表示量增价跌反转信号
        if _alpha12 > 0.02:
            alpha_component += 2
        elif _alpha12 > 0:
            alpha_component += 1
        elif _alpha12 > -0.02:
            alpha_component += 0.5
        else:
            alpha_component += 0

        # Alpha#41: 波动率 — 适度波动率最佳
        if 0.02 <= _alpha41 <= 0.05:
            alpha_component += 1
        elif 0.01 <= _alpha41 <= 0.08:
            alpha_component += 0.5
        else:
            alpha_component += 0

        alpha_component = min(alpha_component, 5)

        # === 7. 涨停板专属因子权重 (5分) ===
        zt_exclusive_component = 0
        _seal_speed = row.get('factor_seal_speed', 0)
        if pd.isna(_seal_speed):
            _seal_speed = 0
        _board_type = row.get('factor_zt_board_type', 0)
        if pd.isna(_board_type):
            _board_type = 0
        _seal_float = row.get('factor_seal_float_ratio', 0)
        if pd.isna(_seal_float):
            _seal_float = 0

        # 封板速度: 值越大表示封板越早 (330=开盘即封, 0=尾盘封)
        if _seal_speed >= 300:
            zt_exclusive_component += 2
        elif _seal_speed >= 240:
            zt_exclusive_component += 1.5
        elif _seal_speed >= 120:
            zt_exclusive_component += 1
        else:
            zt_exclusive_component += 0.5

        # 涨停板类型: 2=多连板 > 1=连板 > 0=首板
        if _board_type >= 2:
            zt_exclusive_component += 2
        elif _board_type >= 1:
            zt_exclusive_component += 1.5
        else:
            zt_exclusive_component += 0.5

        # 封流比: 封单金额/流通市值, 越高封板越稳
        if _seal_float >= 0.05:
            zt_exclusive_component += 1
        elif _seal_float >= 0.02:
            zt_exclusive_component += 0.7
        elif _seal_float >= 0.005:
            zt_exclusive_component += 0.3
        else:
            zt_exclusive_component += 0

        zt_exclusive_component = min(zt_exclusive_component, 5)

        # === 汇总建仓指数 ===
        entry_index = (score_component + price_component + defense_component
                       + capital_component + zt_feature_component
                       + alpha_component + zt_exclusive_component)
        entry_index = max(0, min(100, entry_index))

        # === 计算建议买入价、止损价、目标价 ===
        zt_close = row.get('zt_close', np.nan)
        if pd.isna(zt_close) or zt_close <= 0:
            buy_price = np.nan
            stop_loss = np.nan
            target_price = np.nan
        else:
            if row.get('classification') == '强势':
                buy_price = zt_close * 1.00
            else:
                buy_price = zt_close * 0.98

            stop_loss = zt_close * 0.97

            if total_score >= 70:
                target_price = zt_close * 1.10
            elif total_score >= 50:
                target_price = zt_close * 1.05
            else:
                target_price = zt_close * 1.03

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
            'alpha_component': alpha_component,
            'zt_exclusive_component': zt_exclusive_component,
            'buy_price': round(buy_price, 2) if not pd.isna(buy_price) else np.nan,
            'stop_loss': round(stop_loss, 2) if not pd.isna(stop_loss) else np.nan,
            'target_price': round(target_price, 2) if not pd.isna(target_price) else np.nan,
            'prediction': prediction,
            'signal': signal,
        })

    predict_df = pd.DataFrame(predict_scores, index=candidates.index)
    result_df = pd.concat([candidates, predict_df], axis=1)
    result_df = _dedup_columns(result_df)

    # 按建仓指数降序排列
    result_df = result_df.sort_values('entry_index', ascending=False).reset_index(drop=True)

    return result_df


# ============================================================================
# Section 11: 建仓信号生成
# ============================================================================

def generate_entry_signals(context, predict_df: pd.DataFrame) -> Dict:
    """
    根据分类+信号生成4类建仓信号。

    信号映射:
    - 强势 + 积极建仓 → TYPE_A (开盘100%笼子上限挂单)
    - 平稳 + 积极建仓 → TYPE_B (10点后金叉70% + MA5/MA10补30%)
    - 强势 + 适度建仓 → TYPE_C (9:26集合竞价50% + 条件分支)
    - 其他 → NO_ENTRY

    Parameters
    ----------
    context : JQ context
    predict_df : pd.DataFrame
        predict_next_day() 输出的含建仓指数数据

    Returns
    -------
    dict
        {jq_code: {entry_type, classification, signal, buy_price, stop_loss,
                    target_price, first_leg_done, second_leg_done}}
    """
    signals = {}

    if predict_df is None or predict_df.empty:
        return signals

    # 当前持仓代码
    held_codes = set(g.holdings.keys())

    # 筛选有建仓信号的股票
    entry_candidates = predict_df[
        predict_df['signal'].str.contains('积极|适度', na=False)
    ].copy()

    # Feature 11: 过滤最低评分和最低建仓指数
    min_score = STRATEGY_CONFIG.get('min_score', 0)
    min_entry_index = STRATEGY_CONFIG.get('min_entry_index', 0)
    if min_score > 0:
        entry_candidates = entry_candidates[
            entry_candidates['total_score'].fillna(0) >= min_score
        ]
    if min_entry_index > 0:
        entry_candidates = entry_candidates[
            entry_candidates['entry_index'].fillna(0) >= min_entry_index
        ]

    # 按建仓指数降序，取前 max_entry_count 只
    max_entry = STRATEGY_CONFIG['max_entry_count']
    entry_candidates = entry_candidates.head(max_entry)

    skipped_held = []
    for idx, row in entry_candidates.iterrows():
        jq_code = row.get('jq_code', '')
        classification = row.get('classification', '')
        signal = row.get('signal', '')

        # Feature 9: 跳过已持仓的股票（不对持仓股隔日补仓）
        if jq_code in held_codes:
            skipped_held.append(jq_code)
            continue

        # 确定建仓类型
        entry_type = 'NO_ENTRY'

        if classification == '强势' and '积极' in signal:
            entry_type = 'TYPE_A'
        elif classification == '平稳' and '积极' in signal:
            entry_type = 'TYPE_B'
        elif classification == '强势' and '适度' in signal:
            entry_type = 'TYPE_C'

        if entry_type == 'NO_ENTRY':
            continue

        signals[jq_code] = {
            'entry_type': entry_type,
            'classification': classification,
            'signal': signal,
            'buy_price': row.get('buy_price', np.nan),
            'stop_loss': row.get('stop_loss', np.nan),
            'target_price': row.get('target_price', np.nan),
            'entry_index': row.get('entry_index', 0),
            'total_score': row.get('total_score', 0),
            'first_leg_done': False,
            'second_leg_done': False,
        }

    if skipped_held:
        log.info(f"[generate_entry_signals] 跳过已持仓股票 {len(skipped_held)} 只: {skipped_held}")

    if signals:
        type_counts = {}
        for s in signals.values():
            t = s['entry_type']
            type_counts[t] = type_counts.get(t, 0) + 1
        log.info(f"[generate_entry_signals] 生成 {len(signals)} 个建仓信号: {type_counts}, "
                 f"筛选条件: 评分>={STRATEGY_CONFIG.get('min_score', 0)}, 建仓指数>={STRATEGY_CONFIG.get('min_entry_index', 0)}")

    return signals


# ============================================================================
# Section 12: 技术指标
# ============================================================================

def calc_macd(close_series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算 MACD 指标。

    Returns
    -------
    tuple of (pd.Series, pd.Series, pd.Series)
        (DIF, DEA, MACD柱)
    """
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal_period, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist


def calc_kdj(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算 KDJ 指标。

    Returns
    -------
    tuple of (pd.Series, pd.Series, pd.Series)
        (K, D, J)
    """
    low_n = low_series.rolling(n).min()
    high_n = high_series.rolling(n).max()
    rsv = (close_series - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)

    k = pd.Series(np.nan, index=close_series.index, dtype=float)
    d = pd.Series(np.nan, index=close_series.index, dtype=float)

    k.iloc[0] = 50.0
    d.iloc[0] = 50.0

    for i in range(1, len(close_series)):
        k.iloc[i] = (m1 - 1) / m1 * k.iloc[i - 1] + 1 / m1 * rsv.iloc[i]
        d.iloc[i] = (m2 - 1) / m2 * d.iloc[i - 1] + 1 / m2 * k.iloc[i]

    j = 3 * k - 2 * d
    return k, d, j


def check_15min_golden_cross(context, stock_code: str) -> bool:
    """
    检查15分钟MACD/KDJ金叉。

    金叉判定（宽松条件，满足任一）:
    1. MACD柱由负转正 且 K线上穿D线
    2. MACD柱为正 且 K>D 且 K<80（非超买区金叉）

    Returns
    -------
    bool
        是否检测到金叉信号
    """
    try:
        bars = get_bars(
            stock_code,
            count=50,
            unit='15m',
            fields=['date', 'open', 'close', 'high', 'low', 'volume'],
            include_now=True,
            df=True
        )

        if bars is None or len(bars) < 30:
            return False

        close = bars['close']
        high = bars['high']
        low = bars['low']

        # 计算 MACD
        dif, dea, macd_hist = calc_macd(
            close,
            fast=STRATEGY_CONFIG['macd_fast'],
            slow=STRATEGY_CONFIG['macd_slow'],
            signal_period=STRATEGY_CONFIG['macd_signal']
        )

        # 计算 KDJ
        k, d, j = calc_kdj(
            high, low, close,
            n=STRATEGY_CONFIG['kdj_n'],
            m1=STRATEGY_CONFIG['kdj_m1'],
            m2=STRATEGY_CONFIG['kdj_m2']
        )

        # 检查金叉条件
        # 条件1: MACD柱由负转正 且 K上穿D
        macd_cross = (macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0)
        kd_cross = (k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2])

        if macd_cross and kd_cross:
            return True

        # 条件2: MACD柱为正 且 K>D 且 K<80 (非超买区金叉)
        macd_positive = macd_hist.iloc[-1] > 0
        kd_golden = (k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80)

        if macd_positive and kd_golden:
            return True

        return False

    except Exception as e:
        log.info(f"[check_15min_golden_cross] {stock_code} 检查失败: {e}")
        return False


def check_ma_dip(context, stock_code: str, ma_type: str = 'ma5') -> bool:
    """
    检查MA5/MA10回踩条件。

    当现价距MA线 < 阈值时视为回踩。

    Parameters
    ----------
    ma_type : str
        'ma5' 或 'ma10'

    Returns
    -------
    bool
        是否回踩到MA线
    """
    try:
        # 从缓存获取MA值
        ma_cache = g.ma_cache.get(stock_code, {})
        ma_value = ma_cache.get(ma_type, np.nan)

        if pd.isna(ma_value) or ma_value <= 0:
            return False

        # 获取当前价格
        cur_data = get_current_data()
        if stock_code not in cur_data:
            return False

        current_price = cur_data[stock_code].last_price
        if current_price is None or current_price <= 0:
            return False

        bias = (current_price - ma_value) / ma_value

        if ma_type == 'ma5':
            return abs(bias) < STRATEGY_CONFIG['ma5_bias_threshold']
        elif ma_type == 'ma10':
            return abs(bias) < STRATEGY_CONFIG['ma10_bias_threshold']

        return False

    except Exception as e:
        log.info(f"[check_ma_dip] {stock_code} MA回踩检查失败: {e}")
        return False


def update_ma_cache(context) -> None:
    """
    更新MA5/MA10缓存，供盘中回踩检测使用。
    """
    pool = g.stock_pool
    if pool.empty:
        return

    g.ma_cache = {}

    for idx, row in pool.iterrows():
        jq_code = row.get('jq_code', '')
        if not jq_code:
            continue

        try:
            ma_df = get_price(
                jq_code,
                end_date=context.current_dt.date(),
                frequency='daily',
                fields=['close'],
                count=15,
                panel=False,
                skip_paused=True
            )

            if ma_df is not None and len(ma_df) >= 5:
                g.ma_cache.setdefault(jq_code, {})['ma5'] = ma_df['close'].tail(5).mean()
            if ma_df is not None and len(ma_df) >= 10:
                g.ma_cache.setdefault(jq_code, {})['ma10'] = ma_df['close'].tail(10).mean()
        except Exception:
            pass


# ============================================================================
# Section 13: 建仓执行逻辑
# ============================================================================

def calc_position_size(context, stock_code: str, ratio: float = 1.0) -> int:
    """
    计算买入股数。

    单股分配金额 = 总资产 / max_holdings * ratio
    买入股数取整到100股。

    Parameters
    ----------
    ratio : float
        仓位比例 (0.5 = 半仓, 1.0 = 全仓)

    Returns
    -------
    int
        买入股数
    """
    total_value = context.portfolio.total_value
    per_stock = total_value / STRATEGY_CONFIG['max_holdings']
    buy_amount = per_stock * ratio

    try:
        cur_data = get_current_data()
        current_price = cur_data[stock_code].last_price
        if current_price is None or current_price <= 0:
            return 0
        shares = int(buy_amount / current_price / 100) * 100
        return max(shares, 0)
    except Exception:
        return 0


def execute_entry(context, data) -> None:
    """
    盘中建仓主入口，遍历 entry_signals 执行建仓。
    """
    signals = g.entry_signals
    if not signals:
        return

    # 当前持仓数
    current_holdings = len(g.holdings)
    max_holdings = STRATEGY_CONFIG['max_holdings']

    for code in list(signals.keys()):
        if current_holdings >= max_holdings:
            log.info(f"[execute_entry] 持仓数已达上限 {max_holdings}，停止建仓")
            break

        signal = signals[code]
        entry_type = signal['entry_type']

        if entry_type == 'TYPE_A':
            executed = execute_type_a(context, data, code, signal)
        elif entry_type == 'TYPE_B':
            executed = execute_type_b(context, data, code, signal)
        elif entry_type == 'TYPE_C':
            executed = execute_type_c(context, data, code, signal)
        else:
            continue

        if executed:
            current_holdings = len(g.holdings)


def execute_type_a(context, data, code: str, signal: Dict) -> bool:
    """
    TYPE_A: 强势+积极建仓
    - 开盘100%市价买入（JQ笼子机制自动限制价格在上限内）
    """
    if not signal['first_leg_done']:
        # 100%仓位，按照笼子上限（涨停价）挂单
        shares = calc_position_size(context, code, ratio=1.0)
        if shares > 0:
            try:
                cur_data = get_current_data()
                high_limit = cur_data[code].high_limit
                current_price = cur_data[code].last_price
                if current_price and current_price > 0:
                    # 市价买入，JQ笼子机制自动限制成交价在上限内
                    order_result = order(code, shares)
                    if order_result is not None and _check_order_filled(context, code):
                        signal['first_leg_done'] = True
                        signal['second_leg_done'] = True  # 单腿完成
                        log.info(f"[TYPE_A] {code} 市价买入 {shares} 股，笼子上限 {high_limit:.2f}")
                        _record_holding(context, code, signal, shares, leg='full')
                        return True
                    else:
                        log.info(f"[TYPE_A] {code} 市价买入未成交")
                else:
                    log.info(f"[TYPE_A] {code} 无法获取当前价格")
            except Exception as e:
                log.info(f"[TYPE_A] {code} 笼子上限挂单失败: {e}")
        return False

    return False


def execute_type_b(context, data, code: str, signal: Dict) -> bool:
    """
    TYPE_B: 平稳+积极建仓
    - 10点后15min金叉买入70%
    - MA5/MA10附近补仓30%
    """
    current_time = context.current_dt.time()

    if not signal['first_leg_done']:
        # 10点前不操作
        if current_time.hour < 10:
            return False

        # 检查15min金叉
        if not check_15min_golden_cross(context, code):
            return False

        # 金叉买入70%
        shares = calc_position_size(context, code, ratio=0.7)
        if shares > 0:
            try:
                order_result = order(code, shares)
                if order_result is not None and _check_order_filled(context, code):
                    signal['first_leg_done'] = True
                    log.info(f"[TYPE_B] {code} 金叉买入 {shares} 股 (70%仓位)")
                    _record_holding(context, code, signal, shares, leg='first')
                    return True
                else:
                    log.debug(f"[TYPE_B] {code} 金叉买入未成交（可能涨停/停牌）")
            except Exception as e:
                log.debug(f"[TYPE_B] {code} 金叉买入失败: {e}")
        return False

    elif not signal['second_leg_done']:
        # 第二腿: MA5或MA10附近补仓30%
        ma5_dip = check_ma_dip(context, code, ma_type='ma5')
        ma10_dip = check_ma_dip(context, code, ma_type='ma10')

        if ma5_dip or ma10_dip:
            shares = calc_position_size(context, code, ratio=0.3)
            if shares > 0:
                try:
                    order_result = order(code, shares)
                    if order_result is not None and _check_order_filled(context, code):
                        signal['second_leg_done'] = True
                        dip_type = 'MA5' if ma5_dip else 'MA10'
                        log.info(f"[TYPE_B] {code} {dip_type}附近补仓 {shares} 股 (30%仓位)")
                        _update_holding(context, code, shares, leg='second')
                        return True
                    else:
                        log.debug(f"[TYPE_B] {code} 补仓买入未成交")
                except Exception as e:
                    log.info(f"[TYPE_B] {code} 补仓买入失败: {e}")
        return False

    return False


def execute_type_c(context, data, code: str, signal: Dict) -> bool:
    """
    TYPE_C: 强势+适度建仓
    - 9:26集合竞价50% (JQ中在9:30首tick以开盘价限价挂单模拟)
    - 开盘价 ≤ 集合竞价 → 10点后金叉买入50%
    - 开盘价 > 集合竞价 → 9:31市价挂单50%
    - 已涨停 → 当天不再挂单
    """
    current_time = context.current_dt.time()

    # 检查是否已涨停
    try:
        cur_data = get_current_data()
        high_limit = cur_data[code].high_limit
        current_price = cur_data[code].last_price
        if high_limit and current_price >= high_limit:
            log.debug(f"[TYPE_C] {code} 已涨停，当天不再挂单买入")
            signal['skip_today'] = True
            return False
    except Exception:
        pass

    if signal.get('skip_today'):
        return False

    if not signal['first_leg_done']:
        # 第一腿: 集合竞价挂单50%
        # JQ中无法在9:26下单，在9:30首tick以开盘价限价挂单模拟集合竞价
        auction_price = getattr(g, 'auction_prices', {}).get(code)
        if auction_price is None or auction_price <= 0:
            return False

        # 确定第二腿策略: 根据开盘价与集合竞价的关系
        if signal.get('second_leg_type') is None:
            try:
                cur_data = get_current_data()
                open_price = cur_data[code].last_price
            except Exception:
                open_price = auction_price

            if open_price <= auction_price:
                signal['second_leg_type'] = 'golden_cross'
            else:
                signal['second_leg_type'] = 'market_order'

        shares = calc_position_size(context, code, ratio=0.5)
        if shares > 0:
            try:
                # 市价买入（首tick时市价≈集合竞价价格）
                order_result = order(code, shares)
                if order_result is not None and _check_order_filled(context, code):
                    signal['first_leg_done'] = True
                    log.info(f"[TYPE_C] {code} 集合竞价买入 {shares} 股，竞价 {auction_price:.2f}")
                    _record_holding(context, code, signal, shares, leg='first')
                    return True
                else:
                    signal['first_leg_done'] = True  # 标记已尝试，进入第二腿
                    log.info(f"[TYPE_C] {code} 集合竞价未成交，进入第二腿 ({signal['second_leg_type']})")
            except Exception as e:
                log.info(f"[TYPE_C] {code} 集合竞价挂单失败: {e}")
                return False
        return False

    elif not signal['second_leg_done']:
        second_leg_type = signal.get('second_leg_type', 'golden_cross')

        if second_leg_type == 'golden_cross':
            # 开盘价 ≤ 集合竞价 → 10点后金叉买入50%
            if current_time.hour < 10:
                return False
            if not check_15min_golden_cross(context, code):
                return False
            shares = calc_position_size(context, code, ratio=0.5)
            if shares > 0:
                try:
                    order_result = order(code, shares)
                    if order_result is not None and _check_order_filled(context, code):
                        signal['second_leg_done'] = True
                        log.debug(f"[TYPE_C] {code} 金叉买入 {shares} 股 (50%仓位)")
                        if code in g.holdings:
                            _update_holding(context, code, shares, leg='second')
                        else:
                            _record_holding(context, code, signal, shares, leg='full')
                        return True
                    else:
                        log.debug(f"[TYPE_C] {code} 金叉买入未成交")
                except Exception as e:
                    log.debug(f"[TYPE_C] {code} 金叉买入失败: {e}")
            return False

        elif second_leg_type == 'market_order':
            # 开盘价 > 集合竞价 → 9:31市价挂单50%
            if current_time.hour == 9 and current_time.minute < 31:
                return False
            shares = calc_position_size(context, code, ratio=0.5)
            if shares > 0:
                try:
                    order_result = order(code, shares)
                    if order_result is not None and _check_order_filled(context, code):
                        signal['second_leg_done'] = True
                        log.info(f"[TYPE_C] {code} 9:31市价买入 {shares} 股 (50%仓位)")
                        if code in g.holdings:
                            _update_holding(context, code, shares, leg='second')
                        else:
                            _record_holding(context, code, signal, shares, leg='full')
                        return True
                    else:
                        log.info(f"[TYPE_C] {code} 市价买入未成交")
                except Exception as e:
                    log.info(f"[TYPE_C] {code} 市价买入失败: {e}")
            return False

    return False


def _check_order_filled(context, code: str) -> bool:
    """
    检查订单是否实际成交（持仓是否存在于portfolio中）。
    JQ的order()可能返回Order对象但订单被取消（如涨停买不进）。
    """
    try:
        position = context.portfolio.positions.get(code)
        if position is not None and position.total_amount > 0:
            return True
    except Exception:
        pass
    return False


def _record_holding(context, code: str, signal: Dict, shares: int, leg: str = 'full') -> None:
    """
    记录持仓信息到 g.holdings。
    仅在订单实际成交后调用。
    """
    try:
        cur_data = get_current_data()
        buy_price = cur_data[code].last_price
    except Exception:
        buy_price = signal.get('buy_price', np.nan)

    # 用实际持仓数据校正
    try:
        position = context.portfolio.positions.get(code)
        if position is not None and position.total_amount > 0:
            actual_shares = int(position.total_amount)
            if actual_shares > 0:
                shares = actual_shares
    except Exception:
        pass

    if code not in g.holdings:
        g.holdings[code] = {
            'buy_date': context.current_dt.date(),
            'buy_price': buy_price,
            'shares': shares,
            'amount': buy_price * shares if not pd.isna(buy_price) else 0,
            'stop_loss': signal.get('stop_loss', np.nan),
            'target_price': signal.get('target_price', np.nan),
            'highest_price': buy_price if not pd.isna(buy_price) else 0,
            'entry_type': signal.get('entry_type', ''),
            'entry_index': signal.get('entry_index', 0),
        }
    else:
        # 更新已有持仓 (第二腿)
        h = g.holdings[code]
        total_shares = h['shares'] + shares
        total_amount = h['amount'] + (buy_price * shares if not pd.isna(buy_price) else 0)
        h['shares'] = total_shares
        h['amount'] = total_amount
        h['buy_price'] = total_amount / total_shares if total_shares > 0 else h['buy_price']


def _update_holding(context, code: str, shares: int, leg: str = 'second') -> None:
    """
    更新持仓 (第二腿买入后)。
    """
    if code not in g.holdings:
        return

    try:
        cur_data = get_current_data()
        buy_price = cur_data[code].last_price
    except Exception:
        buy_price = np.nan

    h = g.holdings[code]
    total_shares = h['shares'] + shares
    total_amount = h['amount'] + (buy_price * shares if not pd.isna(buy_price) else 0)
    h['shares'] = total_shares
    h['amount'] = total_amount
    h['buy_price'] = total_amount / total_shares if total_shares > 0 else h['buy_price']


# ============================================================================
# Section 14: 止盈逻辑
# ============================================================================

def check_take_profit(context, data) -> None:
    """
    检查所有持仓的止盈条件。

    止盈优先级:
    1. T+1利润 > 9% (涨停开板则止盈，封涨停则继续持股)
    2. 达到目标价
    3. 移动止盈 (从最高价回撤 > 4%，让利润奔跑)
    4. 最大持仓天数
    """
    if not g.holdings:
        return

    today = context.current_dt.date()

    for code in list(g.holdings.keys()):
        holding = g.holdings[code]
        buy_date = holding.get('buy_date')
        buy_price = holding.get('buy_price', 0)
        target_price = holding.get('target_price', 0)
        highest_price = holding.get('highest_price', 0)

        if not buy_date or buy_price <= 0:
            continue

        # 获取当前价格
        try:
            cur_data = get_current_data()
            current_price = cur_data[code].last_price
            if current_price is None or current_price <= 0:
                continue
        except Exception:
            continue

        # 持仓天数
        hold_days = (today - buy_date).days

        # Feature 10: T+1规则 — 当日新建仓股票不能止盈（A股T+1限制）
        if hold_days == 0:
            log.debug(f"[止盈] {code} 当日新建仓(T+0)，跳过止盈检查")
            continue

        # 1. T+1利润 > 9% + 涨停开板检查
        if hold_days >= 1:
            profit_pct = (current_price - buy_price) / buy_price
            if profit_pct > STRATEGY_CONFIG['t1_profit_take_pct']:
                # 检查是否封涨停：当前价格在涨停价则继续持股，开板则止盈
                high_limit = cur_data[code].high_limit
                if high_limit and current_price >= high_limit:
                    log.info(f"[止盈] {code} T+{hold_days}利润 {profit_pct:.1%} > {STRATEGY_CONFIG['t1_profit_take_pct']:.0%}，但封涨停，继续持股")
                    # 封涨停，不卖出，继续持股让利润奔跑
                else:
                    log.info(f"[止盈] {code} T+{hold_days}利润 {profit_pct:.1%} > {STRATEGY_CONFIG['t1_profit_take_pct']:.0%}，涨停开板，止盈")
                    _sell_position(context, code, reason='T+1高利止盈(开板)')
                    continue

        # 2. 达到目标价 (仅当目标价 > 买入价时才有意义)
        if (not pd.isna(target_price) and target_price > 0 and
                target_price > buy_price and current_price >= target_price):
            log.info(f"[止盈] {code} 现价 {current_price:.2f} 达到目标价 {target_price:.2f}")
            _sell_position(context, code, reason='目标价止盈')
            continue

        # 3. 移动止盈 (从最高价回撤 > 4%，让利润奔跑)
        if hold_days >= 1 and highest_price > 0:
            drawdown = (highest_price - current_price) / highest_price
            if drawdown > STRATEGY_CONFIG['trailing_stop_pct']:
                log.info(f"[止盈] {code} 从最高 {highest_price:.2f} 回撤 {drawdown:.1%} > {STRATEGY_CONFIG['trailing_stop_pct']:.0%}")
                _sell_position(context, code, reason='移动止盈')
                continue

        # 4. 最大持仓天数
        if hold_days >= STRATEGY_CONFIG['max_hold_days']:
            log.info(f"[止盈] {code} 持仓 {hold_days} 天达到上限 {STRATEGY_CONFIG['max_hold_days']}")
            _sell_position(context, code, reason='最大持仓天数')
            continue


# ============================================================================
# Section 15: 止损逻辑
# ============================================================================

def check_stop_loss(context, data) -> None:
    """
    检查所有持仓的止损条件。

    止损优先级:
    1. 跌破止损价
    2. 日内亏损 > 5%
    3. 崩盘检测 (涨跌比 < 1:4)
    """
    if not g.holdings:
        return

    today = context.current_dt.date()

    for code in list(g.holdings.keys()):
        holding = g.holdings[code]
        buy_price = holding.get('buy_price', 0)
        stop_loss_price = holding.get('stop_loss', 0)
        buy_date = holding.get('buy_date')

        if buy_price <= 0:
            continue

        # T+1规则：当日新建仓股票不能卖出，跳过止损检查
        if buy_date is not None and buy_date == today:
            log.debug(f"[止损] {code} 当日新建仓(T+0)，跳过止损检查")
            continue

        # 获取当前价格
        try:
            cur_data = get_current_data()
            current_price = cur_data[code].last_price
            if current_price is None or current_price <= 0:
                continue
        except Exception:
            continue

        # 1. 跌破止损价
        if not pd.isna(stop_loss_price) and stop_loss_price > 0 and current_price <= stop_loss_price:
            log.info(f"[止损] {code} 现价 {current_price:.2f} <= 止损价 {stop_loss_price:.2f}")
            _sell_position(context, code, reason='止损价止损')
            continue

        # 2. 日内亏损 > 5%
        loss_pct = (current_price - buy_price) / buy_price
        if loss_pct < -STRATEGY_CONFIG['daily_stop_loss_pct']:
            log.info(f"[止损] {code} 亏损 {loss_pct:.1%} < -{STRATEGY_CONFIG['daily_stop_loss_pct']:.0%}")
            _sell_position(context, code, reason='日亏止损')
            continue

    # 3. 崩盘检测 (11:25检查)
    crash_time = STRATEGY_CONFIG['crash_check_time']
    current_time = context.current_dt.time()

    if (current_time.hour == crash_time['hour'] and
        current_time.minute == crash_time['minute'] and
        not g.crash_checked_today):

        g.crash_checked_today = True

        if check_market_crash(context):
            g.crash_detected_today = True
            log.info(f"[止损] ⚠️ 崩盘检测触发！涨跌比 < 1:{STRATEGY_CONFIG['crash_ratio']}，清仓所有持仓")
            for code in list(g.holdings.keys()):
                _sell_position(context, code, reason='崩盘清仓')


def check_market_crash(context) -> bool:
    """
    检测市场崩盘：涨跌比 < 1:4。

    Returns
    -------
    bool
        是否检测到崩盘
    """
    try:
        today = context.current_dt.date()
        all_stocks = get_all_securities('stock', date=today)
        stock_codes = all_stocks.index.tolist()

        if not stock_codes:
            return False

        # 分批获取涨跌幅
        batch_size = 500
        advance = 0
        decline = 0

        for i in range(0, len(stock_codes), batch_size):
            batch = stock_codes[i:i + batch_size]
            try:
                prices = get_price(
                    batch,
                    end_date=today,
                    count=1,
                    frequency='daily',
                    fields=['close', 'pre_close'],
                    panel=False,
                    skip_paused=True
                )
                if prices is not None and not prices.empty:
                    # JQ get_price 不支持 pct_change 字段，手动计算
                    pct = (prices['close'] - prices['pre_close']) / prices['pre_close']
                    advance += (pct > 0).sum()
                    decline += (pct < 0).sum()
            except Exception:
                continue

        if decline == 0:
            return False

        ratio = advance / decline
        log.info(f"[check_market_crash] 涨跌比: {advance}:{decline} = 1:{decline/advance:.1f}" if advance > 0 else f"[check_market_crash] 涨跌比: {advance}:{decline}")

        return ratio < 1 / STRATEGY_CONFIG['crash_ratio']

    except Exception as e:
        log.info(f"[check_market_crash] 检测失败: {e}")
        return False


def _get_stock_name(code: str) -> str:
    """
    从股票池或JQ API获取股票名称。
    """
    # 1. 从股票池查找
    pool = g.stock_pool
    if not pool.empty and 'jq_code' in pool.columns:
        match = pool[pool['jq_code'] == code]
        if not match.empty:
            name = match.iloc[0].get('name', '')
            if name:
                return name
    # 2. 从JQ API查找
    try:
        info = get_security_info(code)
        if info and info.display_name:
            return info.display_name
    except Exception:
        pass
    return ''


def _sell_position(context, code: str, reason: str = '') -> None:
    """
    卖出持仓。
    先检查T+1限制（当日买入不能卖出），再检查实际持仓是否存在。
    卖出失败时也清理 g.holdings，避免反复尝试。
    """
    if code not in g.holdings:
        return

    holding = g.holdings[code]
    name = _get_stock_name(code)

    # T+1规则：当日新建仓股票不能卖出（A股T+1限制）
    buy_date = holding.get('buy_date')
    today = context.current_dt.date()
    if buy_date is not None and buy_date == today:
        log.debug(f"[_sell_position] {code} {name} 当日新建仓(T+0)，不能卖出 (原因: {reason})")
        return

    shares = holding.get('shares', 0)

    if shares <= 0:
        # 清理无效持仓记录
        del g.holdings[code]
        if code in g.entry_signals:
            del g.entry_signals[code]
        return

    # 检查实际持仓是否存在
    try:
        position = context.portfolio.positions.get(code)
        if position is None or position.total_amount <= 0:
            # 实际无持仓，清理记录
            log.info(f"[_sell_position] {code} {name} 实际无持仓，清理记录")
            del g.holdings[code]
            if code in g.entry_signals:
                del g.entry_signals[code]
            return
    except Exception:
        pass

    try:
        order_result = order_target(code, 0)
        if order_result is not None:
            # 检查订单是否有错误（如"可平仓数量不足"）
            order_error = getattr(order_result, 'error', None) or getattr(order_result, 'comment', '')
            if order_error:
                log.info(f"[_sell_position] {code} {name} 卖出失败: {order_error}, 清理持仓记录")
                del g.holdings[code]
                if code in g.entry_signals:
                    del g.entry_signals[code]
                return

            # 检查是否实际成交
            try:
                position = context.portfolio.positions.get(code)
                if position is not None and position.total_amount > 0:
                    # 卖出未成交（可能跌停/停牌）
                    log.info(f"[_sell_position] {code} {name} 卖出未成交（可能跌停/停牌）")
                    return
            except Exception:
                pass

            # 计算盈亏
            buy_price = holding.get('buy_price', 0)
            try:
                cur_data = get_current_data()
                sell_price = cur_data[code].last_price
            except Exception:
                sell_price = np.nan

            if buy_price > 0 and not pd.isna(sell_price):
                profit_pct = (sell_price - buy_price) / buy_price
                log.info(f"[_sell_position] {code} {name} 卖出 {shares} 股, 原因: {reason}, "
                         f"买入价: {buy_price:.2f}, 卖出价: {sell_price:.2f}, 盈亏: {profit_pct:.1%}")
            else:
                log.info(f"[_sell_position] {code} {name} 卖出 {shares} 股, 原因: {reason}")

            # 从持仓中移除
            del g.holdings[code]

            # 从建仓信号中移除
            if code in g.entry_signals:
                del g.entry_signals[code]

    except Exception as e:
        log.info(f"[_sell_position] {code} {name} 卖出异常: {e}, 清理持仓记录")
        # 异常时也清理，避免反复尝试
        if code in g.holdings:
            del g.holdings[code]
        if code in g.entry_signals:
            del g.entry_signals[code]


# ============================================================================
# Section 16: 日志输出
# ============================================================================

def log_daily_summary(context) -> None:
    """
    每日盘后输出完整日志。

    日志内容:
    1. 股票池大小和ZT股数量
    2. Top 10 评分股票 (含因子数据)
    3. 昨日涨停股数量
    4. 当前持仓信息 (买入日期、盈亏、股数、金额、总盈亏)
    5. 今日建仓建议 (分类+信号+建仓类型)
    """
    today = context.current_dt.date()
    log.info("=" * 80)
    log.info(f"📊 涨停板策略日报 — {today}")
    log.info("=" * 80)

    # 1. 股票池大小和ZT股数量
    pool = g.stock_pool
    zt_count = g.zt_count_yesterday
    log.info(f"📋 股票池大小: {len(pool)}, 昨日涨停数: {zt_count}")
    log.info(f"📋 交易开关: {'开启' if g.trade_enabled_today else '关闭 (ZT数≤30)'}")

    # 2. Top 10 评分股票
    if not pool.empty and 'total_score' in pool.columns:
        top10 = pool.nlargest(10, 'total_score') if pool['total_score'].notna().any() else pool.head(10)
        log.info(f"🏆 Top 10 评分股票:")
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            code = row.get('code', '')
            name = row.get('name', '')
            score = row.get('total_score', np.nan)
            cls = row.get('classification', '')
            signal = row.get('signal', '')
            entry_idx = row.get('entry_index', np.nan)
            log.info(f"  {i:2d}. {code} {name} | 评分: {score:.0f} | 分类: {cls} | "
                     f"信号: {signal} | 建仓指数: {entry_idx:.0f}")

    # 3. 昨日涨停股数量
    log.info(f"📈 昨日涨停股数量: {zt_count}")

    # 4. 当前持仓信息
    if g.holdings:
        log.info(f"💰 当前持仓 ({len(g.holdings)} 只):")
        total_profit = 0
        total_amount = 0

        try:
            cur_data = get_current_data()
        except Exception:
            cur_data = {}

        for code, h in g.holdings.items():
            buy_date = h.get('buy_date', '')
            buy_price = h.get('buy_price', 0)
            shares = h.get('shares', 0)
            amount = h.get('amount', 0)
            highest = h.get('highest_price', 0)
            name = _get_stock_name(code)

            # 获取当前价格
            current_price = np.nan
            if code in cur_data:
                try:
                    current_price = cur_data[code].last_price
                except Exception:
                    pass

            if buy_price > 0 and not pd.isna(current_price):
                profit_pct = (current_price - buy_price) / buy_price
                profit_amount = (current_price - buy_price) * shares
                total_profit += profit_amount
                total_amount += amount
                log.info(f"  {code} {name} | 买入日: {buy_date} | 买入价: {buy_price:.2f} | "
                         f"现价: {current_price:.2f} | 股数: {shares} | 金额: {amount:.0f} | "
                         f"盈亏: {profit_pct:.1%} ({profit_amount:.0f}元) | 最高: {highest:.2f}")
            else:
                log.info(f"  {code} {name} | 买入日: {buy_date} | 买入价: {buy_price:.2f} | "
                         f"股数: {shares} | 金额: {amount:.0f} | 最高: {highest:.2f}")

        log.info(f"💰 总持仓金额: {total_amount:.0f}元, 总盈亏: {total_profit:.0f}元")
    else:
        log.info(f"💰 当前无持仓")

    # 5. 今日建仓建议
    signals = g.entry_signals
    if signals:
        log.info(f"🔔 今日建仓建议 ({len(signals)} 只):")
        for code, sig in signals.items():
            entry_type = sig.get('entry_type', '')
            cls = sig.get('classification', '')
            signal_text = sig.get('signal', '')
            buy_price = sig.get('buy_price', np.nan)
            stop_loss = sig.get('stop_loss', np.nan)
            target = sig.get('target_price', np.nan)
            first_done = '✅' if sig.get('first_leg_done') else '⬜'
            second_done = '✅' if sig.get('second_leg_done') else '⬜'
            name = _get_stock_name(code)
            log.info(f"  {code} {name} | 类型: {entry_type} | 分类: {cls} | 信号: {signal_text} | "
                     f"买入价: {buy_price:.2f} | 止损: {stop_loss:.2f} | 目标: {target:.2f} | "
                     f"第一腿: {first_done} 第二腿: {second_done}")
    else:
        log.info(f"🔔 今日无建仓建议")

    log.info("=" * 80)


# ============================================================================
# Section 17: IC衰减监控 & 因子共线性检测
# ============================================================================

# --- IC衰减监控配置 ---
IC_MONITOR_CONFIG = {
    'enabled': True,                     # 是否启用IC监控
    'factor_cols': [                     # 需监控的因子列
        'factor_alpha6', 'factor_alpha12', 'factor_alpha33',
        'factor_alpha41', 'factor_alpha49', 'factor_alpha54',
        'factor_seal_speed', 'factor_zt_board_type', 'factor_seal_float_ratio',
    ],
    'target_col': 'next_day_return',     # 预测目标列 (需在 after_trading_end 中计算)
    'ic_warning_threshold': 0.03,        # IC绝对值低于此值发出警告
    'ic_decay_window': 20,               # IC衰减检测窗口 (交易日数)
    'ic_history_max_len': 60,            # IC历史最大保留天数
    'collinearity_threshold': 0.7,       # 相关系数超过此值视为高共线
    'vif_threshold': 5.0,                # VIF超过此值视为高共线
}


def calc_ic(factor_df: pd.DataFrame, factor_col: str, target_col: str) -> Optional[float]:
    """
    计算单个因子的 IC (Information Coefficient) = Spearman秩相关系数。

    Parameters
    ----------
    factor_df : pd.DataFrame
        包含因子值和目标收益的数据
    factor_col : str
        因子列名
    target_col : str
        目标收益列名

    Returns
    -------
    Optional[float]
        IC值，若数据不足返回 None
    """
    if factor_col not in factor_df.columns or target_col not in factor_df.columns:
        return None
    valid = factor_df[[factor_col, target_col]].dropna()
    if len(valid) < 5:
        return None
    try:
        ic, _ = stats.spearmanr(valid[factor_col], valid[target_col])
        return float(ic) if not np.isnan(ic) else None
    except Exception:
        return None


def calc_ic_batch(factor_df: pd.DataFrame, factor_cols: Optional[List[str]] = None,
                  target_col: str = 'next_day_return') -> Dict[str, Optional[float]]:
    """
    批量计算多个因子的 IC 值。

    Parameters
    ----------
    factor_df : pd.DataFrame
        包含因子值和目标收益的数据
    factor_cols : Optional[List[str]]
        因子列名列表，默认使用 IC_MONITOR_CONFIG 中的配置
    target_col : str
        目标收益列名

    Returns
    -------
    Dict[str, Optional[float]]
        {因子名: IC值} 字典
    """
    if factor_cols is None:
        factor_cols = IC_MONITOR_CONFIG['factor_cols']
    ic_results = {}
    for col in factor_cols:
        ic_results[col] = calc_ic(factor_df, col, target_col)
    return ic_results


def update_ic_history(context, ic_results: Dict[str, Optional[float]]) -> None:
    """
    将当日 IC 结果追加到历史记录中，并检测 IC 衰减。

    Parameters
    ----------
    context : object
        JQ context 对象 (通过 g.ic_history 访问历史)
    ic_results : Dict[str, Optional[float]]
        当日 {因子名: IC值} 字典
    """
    if not hasattr(g, 'ic_history'):
        g.ic_history = {}  # {factor_name: [ic_day1, ic_day2, ...]}

    for factor_name, ic_val in ic_results.items():
        if factor_name not in g.ic_history:
            g.ic_history[factor_name] = []
        history = g.ic_history[factor_name]
        history.append(ic_val)
        # 保留最近 N 天
        max_len = IC_MONITOR_CONFIG['ic_history_max_len']
        if len(history) > max_len:
            g.ic_history[factor_name] = history[-max_len:]


def check_ic_decay(context) -> Dict[str, Dict]:
    """
    检测各因子的 IC 衰减情况。

    对每个因子，比较最近窗口内的平均IC与历史平均IC，
    若衰减幅度超过阈值则标记为衰减。

    Returns
    -------
    Dict[str, Dict]
        {因子名: {'current_ic': float, 'history_avg_ic': float,
                  'decay_ratio': float, 'is_decayed': bool}}
    """
    if not hasattr(g, 'ic_history'):
        return {}

    decay_report = {}
    window = IC_MONITOR_CONFIG['ic_decay_window']
    warning_threshold = IC_MONITOR_CONFIG['ic_warning_threshold']

    for factor_name, history in g.ic_history.items():
        if len(history) < 5:
            continue

        # 最近窗口的IC均值
        recent = [v for v in history[-window:] if v is not None]
        if len(recent) < 3:
            continue
        current_avg_ic = np.mean(recent)

        # 历史IC均值 (排除最近窗口)
        older = [v for v in history[:-window] if v is not None]
        if len(older) < 3:
            history_avg_ic = current_avg_ic
        else:
            history_avg_ic = np.mean(older)

        # 衰减比 = (历史|IC| - 当前|IC|) / 历史|IC|
        if abs(history_avg_ic) > 1e-6:
            decay_ratio = (abs(history_avg_ic) - abs(current_avg_ic)) / abs(history_avg_ic)
        else:
            decay_ratio = 0.0

        is_decayed = (abs(current_avg_ic) < warning_threshold) or (decay_ratio > 0.5)

        decay_report[factor_name] = {
            'current_ic': round(current_avg_ic, 4),
            'history_avg_ic': round(history_avg_ic, 4),
            'decay_ratio': round(decay_ratio, 4),
            'is_decayed': is_decayed,
        }

    return decay_report


def log_ic_monitor_report(context) -> None:
    """
    输出 IC 监控报告到日志，包含衰减警告。
    """
    if not IC_MONITOR_CONFIG['enabled']:
        return

    decay_report = check_ic_decay(context)
    if not decay_report:
        return

    log.info("📊 IC衰减监控报告:")
    log.info(f"  {'因子':<25} {'当前IC':>8} {'历史IC':>8} {'衰减比':>8} {'状态':>6}")
    log.info("  " + "-" * 60)

    for factor_name, report in sorted(decay_report.items(), key=lambda x: x[1]['current_ic'], reverse=True):
        status = '⚠️ 衰减' if report['is_decayed'] else '✅ 正常'
        log.info(f"  {factor_name:<25} {report['current_ic']:>8.4f} "
                 f"{report['history_avg_ic']:>8.4f} {report['decay_ratio']:>8.2%} {status:>6}")


def detect_factor_collinearity(factor_df: pd.DataFrame,
                                factor_cols: Optional[List[str]] = None,
                                method: str = 'both') -> Dict:
    """
    检测因子间的共线性 (相关系数 + VIF)。

    Parameters
    ----------
    factor_df : pd.DataFrame
        包含因子值的数据
    factor_cols : Optional[List[str]]
        因子列名列表，默认使用 IC_MONITOR_CONFIG 中的配置
    method : str
        'correlation' 仅相关系数, 'vif' 仅VIF, 'both' 两者都算

    Returns
    -------
    Dict
        {
            'correlation_matrix': pd.DataFrame or None,
            'high_corr_pairs': List[Tuple[str, str, float]],
            'vif_results': Dict[str, float] or None,
            'high_vif_factors': List[str],
            'recommendation': str
        }
    """
    if factor_cols is None:
        factor_cols = IC_MONITOR_CONFIG['factor_cols']

    # 仅保留存在的列
    available_cols = [c for c in factor_cols if c in factor_df.columns]
    if len(available_cols) < 2:
        return {
            'correlation_matrix': None,
            'high_corr_pairs': [],
            'vif_results': None,
            'high_vif_factors': [],
            'recommendation': '因子数量不足，无法进行共线性检测',
        }

    valid_data = factor_df[available_cols].dropna()
    if len(valid_data) < 10:
        return {
            'correlation_matrix': None,
            'high_corr_pairs': [],
            'vif_results': None,
            'high_vif_factors': [],
            'recommendation': '有效数据不足，无法进行共线性检测',
        }

    result = {
        'correlation_matrix': None,
        'high_corr_pairs': [],
        'vif_results': None,
        'high_vif_factors': [],
        'recommendation': '',
    }

    # ---- 1. 相关系数矩阵 ----
    if method in ('correlation', 'both'):
        corr_matrix = valid_data.corr(method='spearman')
        result['correlation_matrix'] = corr_matrix

        # 找出高相关因子对
        threshold = IC_MONITOR_CONFIG['collinearity_threshold']
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        round(corr_val, 4)
                    ))
        result['high_corr_pairs'] = high_corr_pairs

    # ---- 2. VIF (方差膨胀因子) ----
    if method in ('vif', 'both'):
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from statsmodels.tools.tools import add_constant

            X = valid_data[available_cols]
            X_const = add_constant(X)

            vif_results = {}
            for idx, col in enumerate(available_cols):
                try:
                    vif_val = variance_inflation_factor(X_const.values, idx + 1)
                    vif_results[col] = round(vif_val, 2)
                except Exception:
                    vif_results[col] = np.nan

            result['vif_results'] = vif_results

            vif_threshold = IC_MONITOR_CONFIG['vif_threshold']
            high_vif_factors = [k for k, v in vif_results.items()
                                if not np.isnan(v) and v > vif_threshold]
            result['high_vif_factors'] = high_vif_factors

        except ImportError:
            result['recommendation'] = 'statsmodels 未安装，VIF 检测不可用。请 pip install statsmodels'

    # ---- 3. 生成建议 ----
    recommendations = []
    if result['high_corr_pairs']:
        for f1, f2, corr in result['high_corr_pairs']:
            recommendations.append(
                f"高相关 ({corr:+.2f}): {f1} ↔ {f2}，建议保留IC更高者"
            )
    if result.get('high_vif_factors'):
        for f in result['high_vif_factors']:
            vif_val = result['vif_results'][f]
            recommendations.append(
                f"高VIF ({vif_val:.1f}): {f}，建议剔除或降权"
            )
    if not recommendations:
        recommendations.append('✅ 因子间共线性在可接受范围内')

    result['recommendation'] = '\n'.join(recommendations)
    return result


def log_collinearity_report(factor_df: pd.DataFrame,
                             factor_cols: Optional[List[str]] = None) -> None:
    """
    输出因子共线性检测报告到日志。
    """
    result = detect_factor_collinearity(factor_df, factor_cols, method='both')

    log.info("📊 因子共线性检测报告:")

    # 高相关对
    if result['high_corr_pairs']:
        log.info("  高相关因子对 (|ρ| > {:.1f}):".format(IC_MONITOR_CONFIG['collinearity_threshold']))
        for f1, f2, corr in result['high_corr_pairs']:
            log.info(f"    {f1} ↔ {f2}: ρ = {corr:+.4f}")
    else:
        log.info("  ✅ 无高相关因子对")

    # VIF
    if result['vif_results']:
        log.info("  VIF检测结果 (阈值 > {:.1f}):".format(IC_MONITOR_CONFIG['vif_threshold']))
        for factor, vif_val in sorted(result['vif_results'].items(),
                                       key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                                       reverse=True):
            flag = ' ⚠️' if (not np.isnan(vif_val) and vif_val > IC_MONITOR_CONFIG['vif_threshold']) else ''
            log.info(f"    {factor}: VIF = {vif_val:.2f}{flag}")

    # 建议
    log.info("  建议:")
    for line in result['recommendation'].split('\n'):
        log.info(f"    {line}")


def run_factor_diagnostics(context, factor_df: pd.DataFrame) -> Dict:
    """
    运行完整的因子诊断 (IC衰减 + 共线性)，返回综合报告。
    可在 after_trading_end 中调用。

    Parameters
    ----------
    context : object
        JQ context 对象
    factor_df : pd.DataFrame
        当日因子数据 (需包含 next_day_return 列用于IC计算)

    Returns
    -------
    Dict
        {'ic_decay': Dict, 'collinearity': Dict}
    """
    diagnostics = {'ic_decay': {}, 'collinearity': {}}

    if not IC_MONITOR_CONFIG['enabled']:
        return diagnostics

    # 1. IC计算与历史更新
    target_col = IC_MONITOR_CONFIG['target_col']
    if target_col in factor_df.columns:
        ic_results = calc_ic_batch(factor_df, target_col=target_col)
        update_ic_history(context, ic_results)
        diagnostics['ic_decay'] = check_ic_decay(context)
    else:
        # 无目标收益列时，仅记录因子值统计
        log.info(f"⚠️ IC监控: 未找到目标列 '{target_col}'，跳过IC计算")

    # 2. 共线性检测
    diagnostics['collinearity'] = detect_factor_collinearity(factor_df, method='both')

    return diagnostics


# ============================================================================
# Section 18: JQ策略框架
# ============================================================================

def initialize(context):
    """
    策略初始化，仅在回测/实盘开始时调用一次。
    """
    # 设置策略参数
    set_option('use_real_price', True)          # 使用真实价格交易
    set_option('order_volume_ratio', 1)          # 无成交量限制
    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))  # 佣金
    set_slippage(FixedSlippage(0.02))            # 滑点

    # 初始化全局状态
    g.stock_pool = pd.DataFrame(columns=[
        'jq_code', 'code', 'name', 'zt_date', 'zt_close', 'pct_change',
        'total_score', 'classification', 'signal', 'entry_index',
        'buy_price', 'stop_loss', 'target_price'
    ])
    g.holdings = {}                # 持仓 dict
    g.entry_signals = {}           # 当日建仓信号
    g.zt_count_yesterday = 0       # 昨日ZT数
    g.daily_log = []               # 日志
    g.trailing_stops = {}          # 移动止盈线
    g.ma_cache = {}                # MA5/MA10缓存
    g.auction_prices = {}          # TYPE_C集合竞价价格缓存
    g.auction_captured_today = False  # 今日是否已捕获集合竞价
    g.crash_checked_today = False  # 今日是否已检查崩盘
    g.crash_detected_today = False # 今日是否检测到崩盘
    g.trade_enabled_today = True   # 今日是否允许交易

    log.info("[initialize] 涨停板交易策略初始化完成")
    log.info(f"[initialize] 最大持仓: {STRATEGY_CONFIG['max_holdings']}, "
             f"每日最大建仓: {STRATEGY_CONFIG['max_entry_count']}, "
             f"ZT阈值: {STRATEGY_CONFIG['zt_count_threshold']}")
    # 设置日志级别
    log.set_level('order', 'error')   # 订单日志只报错
    log.set_level('system', 'error')  # 系统日志只报错
    log.set_level('strategy', 'info') # 策略日志显示debug信息

def before_trading_start(context):
    """
    每日盘前运行 (约8:30)。

    流程:
    1. 获取昨日ZT股列表
    2. 过滤ST和次新股
    3. 记录ZT数量
    4. 判断是否允许交易
    5. 更新股票池
    6. 淘汰过期/低分股票
    7. 为池中股票构建数据
    8. 运行分析管线
    9. 生成建仓信号
    10. 更新MA缓存
    11. 输出盘前日志
    """
    # 重置每日状态
    g.crash_checked_today = False
    g.crash_detected_today = False
    g.entry_signals = {}
    g.auction_prices = {}           # 重置集合竞价价格缓存
    g.auction_captured_today = False  # 重置竞价捕获标志

    # Step 1: 获取昨日涨停股
    new_zt_df = get_yesterday_zt_stocks(context)

    # Step 2: 过滤ST和次新股
    if not new_zt_df.empty:
        zt_codes = new_zt_df['jq_code'].tolist()
        filtered_codes = filter_stocks(context, zt_codes)
        new_zt_df = new_zt_df[new_zt_df['jq_code'].isin(filtered_codes)].copy()
        log.info(f"[before_trading_start] 过滤后昨日ZT股: {len(new_zt_df)} 只")

    # Step 3: 记录ZT数量
    g.zt_count_yesterday = len(new_zt_df)

    # Step 4: 判断是否允许交易
    if g.zt_count_yesterday <= STRATEGY_CONFIG['zt_count_threshold']:
        g.trade_enabled_today = False
        log.info(f"[before_trading_start] ⚠️ 昨日ZT数 {g.zt_count_yesterday} ≤ {STRATEGY_CONFIG['zt_count_threshold']}，今日不交易")
    else:
        g.trade_enabled_today = True

    # Step 5: 更新股票池
    update_stock_pool(context, new_zt_df)

    # Step 6: 淘汰过期/低分股票
    prune_stock_pool(context)

    # Step 7-9: 为池中股票构建数据并运行分析管线
    pool = g.stock_pool
    if not pool.empty:
        try:
            # 传入分析管线前，先去除池中的分析结果列
            # 这些列会在 score_stock/classify_stock/predict_next_day 中重新计算
            # 如果不去除，pd.concat + _dedup_columns 会保留旧的NaN列而丢弃新计算的列
            analysis_cols = ['total_score', 'classification', 'signal', 'entry_index',
                             'buy_price', 'stop_loss', 'target_price',
                             'price_score', 'trend_score', 'vol_score',
                             'capital_score', 'fund_score', 'risk_deduction']
            pool_for_analysis = pool.drop(columns=[c for c in analysis_cols if c in pool.columns], errors='ignore')

            # 构建行情+补充数据
            enriched_df = build_stock_data(context, pool_for_analysis)

            if enriched_df is not None and not enriched_df.empty:
                # 计算因子
                factor_df = calc_factors(enriched_df)

                # 分类
                classified_df = classify_stock(factor_df)

                # 评分
                scored_df = score_stock(classified_df)

                # 预测
                predict_df = predict_next_day(scored_df)

                # 批量更新股票池中的评分/分类/信号 (替代逐行 iterrows+at[])
                if predict_df is not None and not predict_df.empty:
                    _update_cols = ['total_score', 'classification', 'signal',
                                    'entry_index', 'buy_price', 'stop_loss', 'target_price']
                    _available_cols = [c for c in _update_cols if c in predict_df.columns]
                    _predict_subset = predict_df[['jq_code'] + _available_cols].drop_duplicates('jq_code', keep='first')
                    _predict_indexed = _predict_subset.set_index('jq_code')
                    _pool_mask = g.stock_pool['jq_code'].isin(_predict_indexed.index)

                    if _pool_mask.any():
                        _matched_codes = g.stock_pool.loc[_pool_mask, 'jq_code']
                        for _col in _available_cols:
                            if _col in _predict_indexed.columns:
                                g.stock_pool.loc[_pool_mask, _col] = _predict_indexed.loc[_matched_codes, _col].values

                # Step 9: 生成建仓信号 (仅当允许交易时)
                if g.trade_enabled_today and predict_df is not None and not predict_df.empty:
                    g.entry_signals = generate_entry_signals(context, predict_df)
                else:
                    g.entry_signals = {}

        except Exception as e:
            log.info(f"[before_trading_start] 分析管线异常: {e}")
            log.info(f"[before_trading_start] 异常堆栈:\n{traceback.format_exc()}")

    # Step 10: 更新MA缓存
    update_ma_cache(context)

    # Step 11: 输出盘前日志
    log.info(f"[before_trading_start] 盘前准备完成 | 池: {len(g.stock_pool)} | "
             f"持仓: {len(g.holdings)} | 信号: {len(g.entry_signals)} | "
             f"交易: {'✅' if g.trade_enabled_today else '❌'}")


def handle_data(context, data):
    """
    盘中每个Tick调用。

    执行顺序:
    0. 捕获集合竞价价格 (TYPE_C需要，仅首tick)
    1. 检查止损条件 (最优先)
    2. 检查止盈条件
    3. 检查建仓条件 (仅当允许交易时)
    """
    # 0. 首tick捕获集合竞价价格 (用于TYPE_C)
    if not g.auction_captured_today and g.entry_signals:
        g.auction_captured_today = True
        try:
            cur_data = get_current_data()
            for code, signal in g.entry_signals.items():
                if signal.get('entry_type') == 'TYPE_C':
                    open_price = cur_data[code].last_price
                    if open_price and open_price > 0:
                        g.auction_prices[code] = open_price
                        log.debug(f"[TYPE_C] {code} 捕获集合竞价(开盘价): {open_price:.2f}")
        except Exception as e:
            log.debug(f"[TYPE_C] 捕获集合竞价异常: {e}")

    # 1. 检查止损
    check_stop_loss(context, data)

    # 2. 检查止盈
    check_take_profit(context, data)

    # 3. 检查建仓条件
    if g.trade_enabled_today and g.entry_signals:
        execute_entry(context, data)


def after_trading_end(context):
    """
    每日盘后运行 (15:10)。

    流程:
    1. 更新所有持仓的最高价
    2. 更新移动止盈线
    3. 清理已完成的entry_signals
    4. 清理已平仓的holdings
    5. 输出盘后日志
    """
    # 1. 更新所有持仓的最高价
    today = context.current_dt.date()

    for code in list(g.holdings.keys()):
        holding = g.holdings[code]

        try:
            # 获取今日最高价
            price_df = get_price(
                code,
                end_date=today,
                count=1,
                frequency='daily',
                fields=['high'],
                panel=False,
                skip_paused=True
            )

            if price_df is not None and not price_df.empty:
                today_high = price_df['high'].iloc[0]
                if not pd.isna(today_high):
                    old_highest = holding.get('highest_price', 0)
                    holding['highest_price'] = max(old_highest, today_high)
        except Exception:
            pass

    # 2. 更新移动止盈线
    for code, holding in g.holdings.items():
        highest = holding.get('highest_price', 0)
        if highest > 0:
            trailing_line = highest * (1 - STRATEGY_CONFIG['trailing_stop_pct'])
            g.trailing_stops[code] = {
                'highest_price': highest,
                'trailing_line': trailing_line,
            }

    # 3. 清理已完成的entry_signals (两腿都完成/已跳过/已不在持仓中)
    for code in list(g.entry_signals.keys()):
        signal = g.entry_signals[code]
        if signal.get('first_leg_done') and signal.get('second_leg_done'):
            del g.entry_signals[code]
        elif signal.get('skip_today'):
            # TYPE_C 涨停跳过，当天不再挂单，盘后清理
            del g.entry_signals[code]
        elif code not in g.holdings and signal.get('first_leg_done'):
            # 已卖出，清理信号
            del g.entry_signals[code]

    # 4. 同步持仓信息 (用实际仓位校正)
    try:
        positions = context.portfolio.positions
        for code in list(g.holdings.keys()):
            if code in positions and positions[code].total_amount > 0:
                # 更新实际股数
                g.holdings[code]['shares'] = int(positions[code].total_amount)
            else:
                # 实际已无持仓，清理
                if code in g.holdings:
                    del g.holdings[code]
                if code in g.entry_signals:
                    del g.entry_signals[code]
    except Exception:
        pass

    # 5. 因子诊断 (IC衰减监控 + 共线性检测)
    if IC_MONITOR_CONFIG['enabled'] and not g.stock_pool.empty:
        try:
            # 构建当日因子数据 (从股票池中提取因子列)
            _factor_cols_in_pool = [c for c in IC_MONITOR_CONFIG['factor_cols']
                                     if c in g.stock_pool.columns]
            if _factor_cols_in_pool:
                _factor_data = g.stock_pool[['jq_code'] + _factor_cols_in_pool].copy()

                # 计算次日收益 (仅对已持仓股票可计算实际收益)
                # 对于IC监控，使用 entry_index 作为代理目标
                if 'entry_index' in g.stock_pool.columns:
                    _factor_data['next_day_return'] = g.stock_pool['entry_index']
                    _diagnostics = run_factor_diagnostics(context, _factor_data)
                else:
                    _diagnostics = run_factor_diagnostics(context, _factor_data)

                # 输出IC衰减报告
                log_ic_monitor_report(context)

                # 输出共线性报告
                log_collinearity_report(_factor_data)
        except Exception as e:
            log.info(f"[after_trading_end] 因子诊断异常: {e}")

    # 6. 输出盘后日志
    log_daily_summary(context)