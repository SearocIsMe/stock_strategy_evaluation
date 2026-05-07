#!/usr/bin/env python
# coding: utf-8

# # 涨停板交易策略 (ZT Strategy) - v11 Regime Adaptive Hot Rotation
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
import re
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
    'strategy_version': 'v20_emotion_ignition_core_absorb',
    # --- 股票池 ---
    'pool_max_size': 100,           # 股票池最大容量
    'pool_zt_expire_days': 10,      # ZT日超过此天数则淘汰
    # --- v8 Hot Rotation Engine ---
    # 维护最近N个交易日的热点记忆池：TOP1_TICK / SCORE_BUY / 成交股 / 近期强势评分股。
    'hot_memory_days': 10,
    'hot_memory_max_size': 180,
    'hot_memory_decay': 0.92,
    'hot_memory_min_score': 8,
    'hot_memory_keep_top_n_daily': 20,
    'hot_memory_signal_bonus': 25,
    'hot_memory_buy_bonus': 30,
    'hot_memory_sell_win_bonus': 20,
    'hot_memory_sell_loss_penalty': 18,
    'hot_memory_entry_bonus_weight': 0.35,
    'hot_memory_total_score_weight': 0.18,
    'hot_memory_state_core': 95,
    'hot_memory_state_continue': 65,
    'hot_memory_state_trend': 35,
    'v8_hot_score_weight_total_score': 0.18,
    'v8_hot_score_weight_entry_index': 0.22,
    'v8_hot_score_weight_memory': 0.20,
    'v8_hot_pool_signal_top_n': 8,
    'v8_allow_low_open_memory_states': ['CORE_LEADER', 'HOT_CONTINUE', 'TREND'],
    'v8_low_open_min_ret': -0.035,
    'v8_low_open_min_strength': -0.020,
    'min_list_days': 63,            # 上市不足此天数则过滤 (约3个月)

    # --- 建仓 ---
    'max_entry_count': 3,           # v9：每日最多3只，降低低质量补位交易
    'max_holdings': 4,              # v9：最多4只，单票标准仓位=25%总资产
    'zt_count_threshold': 30,       # 全市场原始昨日ZT数<=此值则不交易（不再使用过滤后数量）

    # --- v11 市场风格状态机：解决“指数涨、接力跌”的反向趋势问题 ---
    # RELAY_HOT/RELAY_OK：允许打板与利润奔跑；TREND_INDEX：禁用打板，只允许少量趋势低吸；ICE/RELAY_WEAK：空仓或极少交易。
    'v11_enable_market_regime': True,
    'v11_index_code': '000300.XSHG',          # 用沪深300判断指数趋势
    'v11_index_short_window': 5,
    'v11_index_mid_window': 20,
    'v11_index_up_threshold': 0.015,          # 20日指数涨幅超过1.5%视为指数趋势偏强
    'v11_index_down_threshold': -0.025,       # 20日指数跌幅低于-2.5%视为弱市/冰点风险
    'v11_relay_recent_trade_n': 8,            # 最近N笔平仓用于判断接力生态
    'v11_relay_recent_bad_avg_pnl': -0.006,   # 最近接力平均收益低于-0.6%，视为接力退潮
    'v11_relay_recent_good_avg_pnl': 0.012,   # 最近接力平均收益高于1.2%，视为接力有效
    'v11_trend_disable_top1_tick': True,      # 趋势指数市禁用Top1打板
    'v11_trend_max_score_buys': 1,
    'v11_trend_score_buy_position_ratio': 0.62, # v14：趋势市核心低吸仓位提高，单票约15.5%总资产
    'v11_trend_min_total_score': 52,
    'v11_trend_min_entry_index': 58,
    'v11_trend_max_0931_ret': 0.045,          # 趋势市不追高，超过4.5%不买
    'v14_trend_min_0931_ret': -0.006,        # v14：趋势市低吸允许小幅水下核心股，不再要求+0.6%
    'v11_relay_weak_no_new_buy': True,
    'v11_runner_only_relay_market': True,

    # --- v13 Guarded Hybrid ---
    # v12把rawZT>=100的指数趋势市全部放成HYBRID，并且只做多tick打板，回测变差。
    # v13退回v11的防守框架，只在“指数趋势+短线极热+情绪HOT”的共振日，开放一个小仓位Top1试探；
    # 同时仍保留v11的趋势低吸，避免错过真正可成交的低吸机会。
    'v13_enable_guarded_hybrid': True,
    'v14_hybrid_rawzt_threshold': 130,
    'v14_hybrid_require_emotion_hot': True,
    'v14_hybrid_tick_candidate_n': 1,
    'v14_hybrid_min_entry_index': 60,
    'v14_hybrid_min_total_score': 42,
    'v14_hybrid_min_hot_score': 130,
    'v14_hybrid_max_score_buys': 1,
    'v14_hybrid_top1_position_ratio': 0.50,
    # --- 一进二打板增强参数（盘后选股 + 次日盘中确认，避免未来函数）---
    'one_two_min_score': 50,          # 回测版放宽：低于此值不直接硬剔除，而用于软惩罚
    'one_two_core_score': 75,
    'one_two_max_float_mcap_yuan': 8e9,
    'one_two_max_price': 10.0,
    'one_two_max_pct_1y': 100.0,
    'one_two_min_zt_days_ytd': 3,
    'one_two_max_zt_open_count': 2,
    'min_score': 35,                # 回测版放宽：建仓最低评分
    'min_entry_index': 50,          # 回测版放宽：建仓最低建仓指数
    'one_two_total_score_weight': 0.30,   # total_score中一进二盘后因子权重
    'one_two_entry_weight': 0.55,         # entry_index中一进二因子权重
    'one_two_hard_filter_penalty': 10,       # 一进二硬过滤改为软惩罚
    'one_two_low_score_penalty_coef': 0.2,   # one_two_score<60时的线性惩罚系数
    'minute_limit_buy_ratio': 0.997,         # 分钟级近似打板：当前价达到涨停价的99.7%可触发
    'minute_near_limit_buy_ratio': 0.985,    # 核心候选接近涨停触发
    'enable_minute_entry_confirm': True,     # 使用分钟级盘中确认，不再强制日线打板确认
    'top1_tick_buy_ratio': 0.997,           # Top1 tick打板触发比例：tick.current >= high_limit*0.997
    'top1_tick_start': '09:30:00',          # Top1 tick打板开始时间
    'top1_tick_end': '10:30:00',            # Top1 tick打板截止时间
    'score_buy_min_entry_index': 50,        # v10：略放宽Top2~3，避免过度空仓；真实执行仍由09:31质量过滤
    'top1_position_ratio': 1.5,             # Top1 仓位：单股标准仓位*1.5，约30%总资产
    'score_buy_position_ratio': 0.75,       # Top2~5 仓位：单股标准仓位*0.75，约15%总资产
    'score_buy_time': '09:31',              # Top2~5固定执行时间：实时过滤后买入
    'score_buy_min_return': -0.02,          # 09:31实时过滤：当前价/昨收 >= 0.98
    'score_buy_max_return': 0.075,          # v10：强势核心允许更高涨幅追随，但仍避免接近涨停普通买
    'score_buy_near_limit_ratio': 0.985,    # Top2~5若接近涨停则不普通买入，留给打板逻辑

    # --- v7 买入质量增强 ---
    # 目的：保留v6.5交易活跃度，但过滤掉09:31弱转强失败、刚亏损过的重复买入票。
    'score_buy_min_open_strength': 0.0,          # 普通票仍要求不弱于开盘
    'v14_core_min_open_strength': -0.018,      # v14：CORE/HOT核心低吸允许开盘后小幅回落，不再错过强趋势核心
    # --- v15 后半段行情再入场模块 ---
    # 解决 v14 在上涨后半段因 RELAY_WEAK/TREND_INDEX 过度防守而长期空仓的问题。
    'v15_enable_late_rebound': True,
    'v15_rebound_rawzt_min': 55,
    'v15_rebound_index_ret5_min': 0.012,
    'v15_rebound_index_ret20_min': -0.015,
    'v15_rebound_max_score_buys': 1,
    'v15_rebound_min_total_score': 45,
    'v15_rebound_min_entry_index': 52,
    'v15_rebound_min_hot_score': 80,
    'v15_rebound_min_0931_ret': -0.004,
    'v15_rebound_max_0931_ret': 0.055,
    'v15_rebound_score_buy_position_ratio': 0.58,

    # --- v16 龙头衰退退出模块 ---
    # v15 已解决后半段空仓问题；v16 重点控制利润回吐：热点衰退、排名掉队、动态利润保护。
    'v16_enable_runner_decay_exit': True,
    'v16_hot_score_decay_pct': 0.12,          # 单日hot_score较上次记录下降超过12%，记一次衰退
    'v16_hot_decay_days_to_exit': 2,          # 连续2次衰退后，不再继续利润奔跑
    'v16_runner_rank_exit_threshold': 8,      # 核心票跌出热点Top8，视为龙头地位下降
    'v16_runner_rank_warn_threshold': 5,      # 跌出Top5后收紧保护
    'v16_decay_exit_min_profit': 0.018,       # 有利润时才用衰退退出，避免低位反复割肉
    'v16_decay_exit_min_max_profit': 0.035,   # 曾浮盈达到3.5%后才启动衰退退出
    'v16_dynamic_trail_enabled': True,
    'v16_runner_dd_after_3pct': 0.018,        # 曾浮盈3%+，允许从高点回撤1.8%
    'v16_runner_dd_after_5pct': 0.024,
    'v16_runner_dd_after_8pct': 0.035,
    'v16_runner_dd_after_12pct': 0.050,
    'v16_runner_min_hold_for_decay_exit': 2,


    # --- v18 情绪优先市场状态引擎 ---
    # v16/v17 的问题：指数仍弱时容易长期 ICE，错过“冰点后情绪修复主升”。
    # v18 让短线情绪优先于指数：指数弱 + 情绪强 => REBOUND_RELAY / MAIN_UPTREND，而不是 ICE。
    'v18_enable_emotion_first_regime': True,
    'v18_rebound_rawzt_min': 70,
    'v18_rebound_core_min': 3,
    'v18_rebound_hot_continue_min': 5,
    'v18_rebound_top_hot_min': 80,
    'v18_rebound_top_entry_min': 58,
    'v18_rebound_idx5_min': -0.065,
    'v18_rebound_idx20_min': -0.085,
    'v18_main_rawzt_min': 90,
    'v18_main_core_min': 6,
    'v18_main_top_hot_min': 105,
    'v18_main_top_entry_min': 62,
    'v18_main_max_score_buys': 2,
    'v18_rebound_max_score_buys': 1,
    'v18_rebound_min_total_score': 42,
    'v18_rebound_min_entry_index': 52,
    'v18_rebound_min_hot_score': 70,
    'v18_rebound_min_0931_ret': -0.018,
    'v18_rebound_max_0931_ret': 0.065,
    'v18_rebound_score_buy_position_ratio': 0.65,
    'v18_main_score_buy_position_ratio': 0.72,
    'v18_main_top1_position_ratio': 1.15,
    'v18_rebound_top1_position_ratio': 0.90,
    'v18_ice_rawzt_hard_min': 45,
    'v18_ice_top_hot_max': 55,
    'v18_ice_core_max': 1,
    'v18_temperature_ice': 25,
    'v18_temperature_rebound': 42,
    'v18_temperature_main': 58,

    # --- v20 情绪点火确认引擎 ---
    # v18解决了“会不会进场”，但过早、过宽，导致弱修复日噪音交易太多。
    # v20在情绪点火早期只做核心低吸，主升确认后才开放Top1；降低错过1月/4月修复行情。
    'v19_enable_confirmed_rebound': True,
    'v19_rebound_rawzt_min': 70,
    'v19_rebound_core_min': 3,
    'v19_rebound_hot_continue_min': 3,
    'v19_rebound_top_hot_min': 80,
    'v19_rebound_top_entry_min': 58,
    'v19_rebound_temperature_min': 68,
    'v19_rebound_idx5_min': -0.070,
    'v19_rebound_idx20_min': -0.12,
    'v19_rebound_disable_top1': True,
    'v19_rebound_max_score_buys': 1,
    'v19_rebound_min_total_score': 50,
    'v19_rebound_min_entry_index': 59,
    'v19_rebound_min_hot_score': 80,
    'v19_rebound_min_0931_ret': -0.025,
    'v19_rebound_max_0931_ret': 0.030,
    'v19_rebound_score_buy_position_ratio': 0.52,
    'v19_main_rawzt_min': 95,
    'v19_main_core_min': 7,
    'v19_main_top_hot_min': 115,
    'v19_main_top_entry_min': 63,
    'v19_main_temperature_min': 82,
    'v19_main_max_score_buys': 1,
    'v19_main_top1_position_ratio': 0.80,
    'v19_main_score_buy_position_ratio': 0.55,
    # --- v20 情绪点火早期：只核心低吸，不追板 ---
    'v20_ignition_disable_top1': True,
    'v20_ignition_core_only': True,
    'v20_rebound_prefer_low_absorb': True,
    'v20_runner_core_only': True,

    'score_buy_weak_reversal_open_strength': 0.004, # v10：深水票仍需反包，但不过度严苛
    'score_buy_loss_cooldown_days': 5,           # v9：亏损票冷却更久，减少重复试错
    'score_buy_low_quality_entry_index': 58,     # v10：低分票更严格，高分/核心票放行
    'score_buy_low_quality_total_score': 36,     # v9：提高低质量判定线
    'top1_reseal_mode': False,
    'trend_hold_min_days': 2,
    'trend_allow_low_open_reversal': True,
    'top1_position_ratio': 1.20,                 # v10：Top1龙头仓位约30%总资产，是策略收益核心
    'score_buy_position_ratio': 0.72,            # v10：SCORE_BUY略降仓，给Top1和持仓利润奔跑留空间
                    # Top1从单纯涨停价排板，改为“触板-开板-回封”优先


    # --- v7 市场情绪过滤与动态仓位 ---
    # 说明：使用“原始昨日涨停数”作为市场情绪，不使用过滤后的可交易池数量。
    # 弱市不新增仓位；普通市场降低买入数量和提高门槛；强市场才允许更积极交易。
    'enable_emotion_filter': True,
    'emotion_stop_zt_threshold': 40,       # v7: 极弱市才停止新增买入；40~89转为控仓而非禁买
    'emotion_caution_zt_threshold': 90,    # 60~89：谨慎，仅允许Top2~5最多1只
    'emotion_hot_zt_threshold': 130,       # >=130：强势，可按原计划多买
    'emotion_caution_max_score_buys': 1,
    'emotion_normal_max_score_buys': 1,
    'emotion_hot_max_score_buys': 2,
    'emotion_caution_min_entry_index': 55,
    'emotion_normal_min_entry_index': 50,
    'emotion_hot_min_entry_index': 49,
    'emotion_caution_min_return': 0.003,  # v10：谨慎市仍要求转强，但不再过度空仓
    'emotion_normal_min_return': 0.006,
    'emotion_hot_min_return': 0.008,

    # --- v6.3 风控与交易闭环 ---
    'risk_check_times': ['09:35', '10:30', '11:25', '14:00', '14:50'],
    'profit_protect_min_pct': 0.06,       # v7: 浮盈超过3%后启用利润保护
    'profit_protect_drawdown_pct': 0.025,  # 普通票仍严格保护；v10长效龙头使用独立利润奔跑参数
    'runner_hot_score_min': 140,          # v10：长效利润候选：热点记忆分阈值
    'runner_entry_index_min': 62,         # v10：长效利润候选：建仓指数阈值
    'runner_total_score_min': 45,         # v10：长效利润候选：总评分阈值
    'runner_min_hold_days': 3,            # v10：长效龙头最少观察持有天数，避免T+1洗盘卖飞
    'runner_max_hold_days': 8,            # v10：长效龙头最大持有天数
    'runner_profit_protect_min_pct': 0.12,# v10：最高浮盈>=12%才启动龙头利润保护
    'runner_profit_drawdown_pct': 0.08,   # v10：龙头允许8%高位回撤
    'runner_trailing_stop_pct': 0.10,     # v10：泛化移动止盈更宽
    'runner_break_even_after_profit_pct': 0.10,
    'runner_break_even_buffer_pct': -0.015,
    'break_even_after_profit_pct': 0.05,  # 曾浮盈超过5%后，不允许重新跌破买入价太多
    'break_even_buffer_pct': -0.005,      # 允许最多回落到买入价下方0.5%
    'force_sell_on_max_hold_days': True,  # 超过最大持仓天数强制卖出

    # --- 止盈 ---
    'max_hold_days': 5,             # 最大持仓天数
    't1_profit_take_pct': 0.09,     # T+1利润>9%止盈
    'trailing_stop_pct': 0.06,      # 泛化移动止盈阈值；v7主要使用3%浮盈/2%回撤保护

    # --- 止损 ---
    'daily_stop_loss_pct': 0.04,   # v9：仓位提升后硬止损收紧到4%
    'structure_stop_enabled': True,  # v8：结构止损，跌破开盘且亏损时退出
    'structure_stop_min_loss_pct': -0.018, # v9：结构走坏更早退出
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
# 一进二打板增强模块：盘后选股 + 次日盘中执行提示（无未来函数）
# ============================================================================

def _cfg_get(name, default=None):
    """兼容 CONFIG / STRATEGY_CONFIG 的配置读取。"""
    if 'CONFIG' in globals() and isinstance(CONFIG, dict) and name in CONFIG:
        return CONFIG.get(name, default)
    if 'STRATEGY_CONFIG' in globals() and isinstance(STRATEGY_CONFIG, dict) and name in STRATEGY_CONFIG:
        return STRATEGY_CONFIG.get(name, default)
    return default


def _to_float_safe(val, default=np.nan):
    """安全转 float；支持 万/亿/百分号。"""
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    s = str(val).strip().replace(',', '').replace('%', '')
    if s in ('', '-', '—', 'NA', 'N/A', 'None', 'nan'):
        return default
    try:
        if '亿' in s:
            return float(s.replace('亿', '')) * 1e8
        if '万' in s:
            return float(s.replace('万', '')) * 1e4
        return float(s)
    except Exception:
        return default


def _series_num(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype='float64')
    s = _safe_series(df, col)
    return s.apply(lambda x: _to_float_safe(x, default))


def _parse_time_minutes(val):
    """把 09:35 / 93500 / 093500 / Timestamp 统一成分钟数。"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (pd.Timestamp, dt.datetime)):
        return val.hour * 60 + val.minute + val.second / 60.0
    if isinstance(val, dt.time):
        return val.hour * 60 + val.minute + val.second / 60.0
    s = str(val).strip()
    if s in ('', '-', '—', 'nan', 'None'):
        return np.nan
    try:
        if ':' in s:
            parts = s.split(':')
            h, m = int(parts[0]), int(parts[1])
            sec = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            return h * 60 + m + sec / 60.0
        digits = re.sub(r'\D', '', s)
        if len(digits) >= 6:
            h, m, sec = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
            return h * 60 + m + sec / 60.0
        if len(digits) == 4:
            h, m = int(digits[:2]), int(digits[2:4])
            return h * 60 + m
        if len(digits) <= 2:
            return float(digits)
    except Exception:
        return np.nan
    return np.nan


def _mcap_to_yuan_series(s):
    """流通市值兼容：若数值明显是“亿元”，转为元；若已是元则保持。"""
    x = s.astype(float)
    # A股流通市值如果小于 10000，通常是“亿元”单位；CSV 中文量词转换后通常已是元。
    return np.where((x > 0) & (x < 10000), x * 1e8, x)


def calc_one_two_board_factors(df):
    """
    一进二打板的盘后因子，严格只使用涨停日收盘后已知数据。

    输出核心列：
    - one_two_hard_filter: 是否硬过滤
    - one_two_filter_reason: 过滤原因
    - one_two_score: 一进二盘后综合分 0~100
    - one_two_rank_bucket: 核心/备选/观察/剔除
    - intraday_plan: 次日盘中执行计划
    """
    d = df.copy()
    n = len(d)
    if n == 0:
        return d

    # 基础数据
    open_s = _series_num(d, 'open')
    high_s = _series_num(d, 'high')
    low_s = _series_num(d, 'low')
    close_s = _series_num(d, 'zt_close')
    latest_s = _series_num(d, 'latest_price')
    latest_s = latest_s.where(~latest_s.isna(), close_s)
    vol_ratio = _series_num(d, 'vol_ratio')
    turnover = _series_num(d, 'turnover_rate')
    inner_outer = _series_num(d, 'inner_outer_ratio')
    outer_vol = _series_num(d, 'outer_vol')
    inner_vol = _series_num(d, 'inner_vol')
    bid_ask_ratio = _series_num(d, 'bid_ask_ratio')
    seal_amount = _series_num(d, 'seal_amount')
    seal_ratio = _series_num(d, 'seal_volume_ratio')
    zt_open_count = _series_num(d, 'zt_open_count').fillna(0)
    pct_1y = _series_num(d, 'pct_1y')
    pct_1m = _series_num(d, 'pct_1m')
    pct_month = _series_num(d, 'pct_month')
    main_net = _series_num(d, 'main_net_inflow')
    consecutive_up = _series_num(d, 'consecutive_up').fillna(0)
    zt_days_ytd = _series_num(d, 'zt_days_ytd')
    days_boards = _series_num(d, 'days_boards')
    float_mcap_raw = _series_num(d, 'float_market_cap')
    float_mcap_yuan = pd.Series(_mcap_to_yuan_series(float_mcap_raw), index=d.index)

    # 首次涨停时间：越早越好；没有则给中性
    if 'first_zt_time' in d.columns:
        first_minutes = _safe_series(d, 'first_zt_time').apply(_parse_time_minutes)
    else:
        first_minutes = pd.Series(np.nan, index=d.index)
    open_minutes = 9 * 60 + 30
    first_elapsed = (first_minutes - open_minutes).clip(lower=0)

    # 硬过滤：完全不依赖未来数据
    one_price_board = ((open_s == high_s) & (high_s == low_s) & (low_s == close_s))
    repeated_broken = zt_open_count >= _cfg_get('one_two_max_zt_open_count', 2)
    large_float = float_mcap_yuan > _cfg_get('one_two_max_float_mcap_yuan', 8e9)
    high_price = latest_s > _cfg_get('one_two_max_price', 10.0)
    overextended = pct_1y > _cfg_get('one_two_max_pct_1y', 100.0)
    no_zt_basis = (zt_days_ytd.notna()) & (zt_days_ytd < _cfg_get('one_two_min_zt_days_ytd', 3))

    reasons = []
    hard_filter = pd.Series(False, index=d.index)
    for idx in d.index:
        r = []
        if bool(one_price_board.loc[idx]): r.append('一字板')
        if bool(repeated_broken.loc[idx]): r.append('反复炸板')
        if bool(large_float.loc[idx]): r.append('流通市值>80亿')
        if bool(high_price.loc[idx]): r.append('股价>10元')
        if bool(overextended.loc[idx]): r.append('近一年涨幅过大')
        if bool(no_zt_basis.loc[idx]): r.append('涨停基因不足')
        hard_filter.loc[idx] = len(r) > 0
        reasons.append('、'.join(r) if r else '')
    d['one_two_hard_filter'] = hard_filter
    d['one_two_filter_reason'] = reasons

    # 打分：涨停质量 30，量能 25，盘口/资金 20，位置热度 15，题材/低位 10
    zt_quality = pd.Series(0.0, index=d.index)
    zt_quality += np.select(
        [first_elapsed <= 30, first_elapsed <= 60, first_elapsed <= 120, first_elapsed.notna()],
        [14, 11, 8, 5], default=7
    )
    zt_quality += np.select([zt_open_count == 0, zt_open_count == 1, zt_open_count >= 2], [8, 4, 0], default=4)
    zt_quality += np.select([seal_ratio >= 5, seal_ratio >= 2, seal_ratio > 0], [8, 5, 3], default=2)
    d['one_two_zt_quality_score'] = zt_quality.clip(0, 30)

    volume_score = pd.Series(0.0, index=d.index)
    volume_score += np.select([vol_ratio >= 2.0, vol_ratio >= 1.5, vol_ratio >= 1.0], [12, 9, 5], default=2)
    volume_score += np.select([turnover.between(5, 20), turnover.between(3, 25), turnover > 0], [8, 5, 2], default=1)
    # 封板当天放量但不过分：给正分；过度巨量不额外加分
    volume_score += np.select([vol_ratio.between(1.5, 4.0), vol_ratio > 4.0], [5, 2], default=1)
    d['one_two_volume_score'] = volume_score.clip(0, 25)

    order_score = pd.Series(0.0, index=d.index)
    order_score += np.select([inner_outer < 0.8, inner_outer < 1.0, inner_outer.notna()], [7, 5, 2], default=3)
    order_score += np.select([outer_vol > inner_vol, outer_vol.notna() & inner_vol.notna()], [5, 2], default=3)
    order_score += np.select([main_net > 0, main_net >= -1e7, main_net.notna()], [5, 3, 1], default=3)
    order_score += np.select([bid_ask_ratio > 20, bid_ask_ratio > 0, bid_ask_ratio.notna()], [3, 2, 1], default=1)
    d['one_two_order_score'] = order_score.clip(0, 20)

    position_score = pd.Series(0.0, index=d.index)
    position_score += np.select([pct_1y <= 30, pct_1y <= 60, pct_1y <= 100, pct_1y.notna()], [7, 5, 3, 1], default=4)
    position_score += np.select([pct_1m > 0, pct_month > 0], [4, 3], default=2)
    position_score += np.select([consecutive_up == 1, consecutive_up == 2, consecutive_up >= 3], [4, 3, 1], default=2)
    d['one_two_position_score'] = position_score.clip(0, 15)

    theme_score = pd.Series(0.0, index=d.index)
    theme_score += np.select([zt_days_ytd >= 4, zt_days_ytd >= 2, zt_days_ytd.notna()], [4, 2, 1], default=2)
    theme_score += np.select([days_boards == 1, days_boards == 2, days_boards >= 3], [3, 2, 1], default=2)
    # 如果有行业列，程序不做未来判断，只保留中性分，外部可按当日题材再人工确认
    theme_score += 3
    d['one_two_theme_score'] = theme_score.clip(0, 10)

    total = (d['one_two_zt_quality_score'] + d['one_two_volume_score'] +
             d['one_two_order_score'] + d['one_two_position_score'] + d['one_two_theme_score'])
    total = total.where(~hard_filter, total * 0.25)
    d['one_two_score'] = total.clip(0, 100).round(2)

    d['one_two_rank_bucket'] = np.select(
        [d['one_two_hard_filter'], d['one_two_score'] >= 75, d['one_two_score'] >= 65, d['one_two_score'] >= 50],
        ['剔除', '核心Top3候选', '备选', '观察'], default='放弃'
    )

    d['intraday_plan'] = np.select(
        [d['one_two_rank_bucket'].eq('核心Top3候选'), d['one_two_rank_bucket'].eq('备选'), d['one_two_rank_bucket'].eq('观察')],
        [
            '次日9:20-9:25只看竞价：竞价量/昨量10%-15%、价格最后1-2分钟拐头向上；满足则竞价小仓或打板确认。',
            '只做板上确认或炸板回封；不得低吸追涨。',
            '仅观察，不主动买入；除非板块涨停数>=10且盘口显著增强。'
        ],
        default='剔除/放弃，不参与。'
    )
    return d


def calc_auction_precheck(df):
    """
    竞价预检：仅在 9:20-9:25 已真实产生竞价数据后使用；盘后回测不得提前使用。
    """
    d = df.copy()
    auction_vol = _series_num(d, 'auction_volume')
    prev_vol = _series_num(d, 'prev_volume')
    auction_pct = _series_num(d, 'auction_pct')
    unmatched = _series_num(d, 'unmatched_volume')
    ratio = auction_vol / prev_vol.replace(0, np.nan)
    d['auction_volume_ratio'] = ratio
    score = pd.Series(0.0, index=d.index)
    score += np.select([ratio.between(0.10, 0.15), ratio.between(0.06, 0.20), ratio.notna()], [40, 25, 10], default=15)
    score += np.select([auction_pct.between(3, 9.5), auction_pct.between(0, 10), auction_pct.notna()], [25, 15, 5], default=10)
    score += np.select([unmatched > 0, unmatched.notna()], [20, 8], default=10)
    base = _series_num(d, 'one_two_score').fillna(50)
    score += np.select([base >= 75, base >= 65, base >= 50], [15, 10, 5], default=0)
    d['auction_precheck_score'] = score.clip(0, 100).round(2)
    d['auction_action'] = np.select(
        [d['auction_precheck_score'] >= 75, d['auction_precheck_score'] >= 60, d['auction_precheck_score'] >= 45],
        ['竞价可小仓/打板确认', '只打板确认', '观察'], default='放弃'
    )
    return d


# ============================================================================
# 实盘/回测盘中执行增强：只使用当前时刻已知数据，不读取未来K线
# ============================================================================

def check_intraday_one_two_entry(context, code: str, signal: Dict) -> Tuple[bool, str]:
    """
    v5 分钟级近似实盘执行确认。

    只使用当前分钟已知数据：get_current_data().last_price / high_limit。
    逻辑：
    - 9:30前不假设成交；
    - 9:30后，当前价 >= 涨停价*99.7%：视为分钟级打板/排板触发；
    - 9:31-10:00，核心候选且当前价 >= 涨停价*98.5%：允许小仓提前排板；
    - 10:00后，只接受更严格的接近涨停触发，避免普通半路追涨；
    - 不读取未来K线，不用当日收盘/最高价判断。
    """
    if not STRATEGY_CONFIG.get('enable_minute_entry_confirm', True):
        return True, '分钟确认关闭：按盘前信号执行'

    now_t = context.current_dt.time()
    try:
        cur_data = get_current_data()
        cd = cur_data[code]
        price = cd.last_price
        high_limit = cd.high_limit
        if price is None or high_limit is None or high_limit <= 0:
            return False, '当前价/涨停价不可用'
    except Exception as e:
        return False, f'行情不可用: {e}'

    if now_t.hour < 9 or (now_t.hour == 9 and now_t.minute < 30):
        return False, '9:30前不假设成交'

    entry_index = float(signal.get('entry_index', 0) or 0)
    one_two_score = float(signal.get('one_two_score', 0) or 0)
    bucket = str(signal.get('one_two_rank_bucket', '') or '')

    limit_ratio = STRATEGY_CONFIG.get('minute_limit_buy_ratio', 0.997)
    near_ratio = STRATEGY_CONFIG.get('minute_near_limit_buy_ratio', 0.985)

    # 分钟级近似打板：当前分钟价格已经非常接近涨停价。
    if price >= high_limit * limit_ratio:
        if entry_index >= STRATEGY_CONFIG.get('min_entry_index', 50):
            return True, f'分钟级打板触发：price/high_limit={price/high_limit:.3f}'
        return False, '接近涨停但建仓指数不足'

    # 早盘强势拉升：只允许高entry_index或核心候选，作为近涨停排板近似。
    if (now_t.hour == 9 and now_t.minute >= 31) or (now_t.hour == 10 and now_t.minute == 0):
        if price >= high_limit * near_ratio and (entry_index >= 58 or one_two_score >= 65 or '核心' in bucket):
            return True, f'早盘近涨停触发：price/high_limit={price/high_limit:.3f}'
        return False, '早盘未达到近涨停触发条件'

    # 10点后只接受更强的触板/近板确认。
    if price >= high_limit * 0.995 and entry_index >= 55:
        return True, f'10点后回封/近板确认：price/high_limit={price/high_limit:.3f}'

    return False, '分钟级未触发打板/近板条件'




# ============================================================================
# Section 4: 涨停股筛选与过滤（v5.1 修复版）
# ============================================================================

def _ensure_jq_zt_dataframe(price_df: pd.DataFrame, all_stocks: pd.DataFrame, zt_date) -> pd.DataFrame:
    """
    将 JoinQuant get_price(panel=False) 的各种返回格式统一为标准涨停池格式。
    必须输出 jq_code/code/name/zt_date/zt_close/pct_change 等列，供后续股票池使用。
    """
    if price_df is None or len(price_df) == 0:
        return pd.DataFrame(columns=['jq_code', 'code', 'name', 'zt_date', 'zt_close', 'pct_change'])

    df = price_df.copy()

    # 兼容 MultiIndex / index 中带 code 的情况
    if 'code' not in df.columns:
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            # 常见列名可能是 level_0/level_1，其中一个是 code
            if 'code' not in df.columns:
                for col in df.columns:
                    sample = df[col].dropna().astype(str).head(5).tolist()
                    if any(('.XSHE' in x or '.XSHG' in x) for x in sample):
                        df = df.rename(columns={col: 'code'})
                        break
        else:
            df = df.reset_index()
            if 'code' not in df.columns:
                for col in df.columns:
                    sample = df[col].dropna().astype(str).head(5).tolist()
                    if any(('.XSHE' in x or '.XSHG' in x) for x in sample):
                        df = df.rename(columns={col: 'code'})
                        break

    if 'code' not in df.columns:
        # 无 code 列就不能更新股票池，返回空表而不是让后续 KeyError
        return pd.DataFrame(columns=['jq_code', 'code', 'name', 'zt_date', 'zt_close', 'pct_change'])

    # 必要列兜底
    for col in ['close', 'pre_close', 'high', 'low', 'open', 'volume', 'money']:
        if col not in df.columns:
            df[col] = np.nan

    # 涨跌幅
    if 'pct_change' not in df.columns:
        df['pct_change'] = np.where(df['pre_close'] > 0,
                                    (df['close'] - df['pre_close']) / df['pre_close'] * 100,
                                    np.nan)

    # 涨停判断：优先使用 high_limit；否则用涨幅阈值 + 收盘等于最高价近似
    if 'high_limit' in df.columns:
        zt_mask = df['close'] >= df['high_limit'] * 0.999
    else:
        zt_mask = (df['pct_change'] >= STRATEGY_CONFIG.get('zt_threshold', 9.8)) & (df['close'] >= df['high'] * 0.999)

    zt_df = df[zt_mask].copy()
    if zt_df.empty:
        return pd.DataFrame(columns=['jq_code', 'code', 'name', 'zt_date', 'zt_close', 'pct_change'])

    result = pd.DataFrame()
    result['jq_code'] = zt_df['code'].astype(str).values
    result['code'] = result['jq_code'].apply(lambda x: x.split('.')[0] if '.' in x else x)
    result['name'] = result['jq_code'].apply(
        lambda x: all_stocks.loc[x].display_name if x in all_stocks.index else ''
    )
    result['zt_date'] = pd.to_datetime(zt_date)
    result['zt_close'] = zt_df['close'].values
    result['pct_change'] = zt_df['pct_change'].values
    result['volume'] = zt_df['volume'].values
    result['money'] = zt_df['money'].values
    result['open'] = zt_df['open'].values
    result['high'] = zt_df['high'].values
    result['low'] = zt_df['low'].values
    result['pre_close'] = zt_df['pre_close'].values
    if 'high_limit' in zt_df.columns:
        result['high_limit'] = zt_df['high_limit'].values
    result = result.dropna(subset=['jq_code']).reset_index(drop=True)
    return result


def get_yesterday_zt_stocks(context) -> pd.DataFrame:
    """
    获取昨日涨停股列表（原始全市场口径，不过滤 ST/次新）。

    返回标准化字段：
    jq_code/code/name/zt_date/zt_close/pct_change/volume/money/open/high/low/pre_close
    """
    yesterday = context.previous_date

    try:
        all_stocks = get_all_securities('stock', date=yesterday)
    except TypeError:
        all_stocks = get_all_securities(types=['stock'], date=yesterday)

    stock_codes = list(all_stocks.index)
    if not stock_codes:
        return pd.DataFrame(columns=['jq_code', 'code', 'name', 'zt_date', 'zt_close', 'pct_change'])

    batch_size = 200
    parts = []
    fields_try = ['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'high_limit']
    fields_fallback = ['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close']

    for i in range(0, len(stock_codes), batch_size):
        batch = stock_codes[i:i + batch_size]
        price_df = None
        try:
            price_df = get_price(batch, end_date=yesterday, count=1, frequency='daily',
                                 fields=fields_try, panel=False, skip_paused=True)
        except Exception:
            try:
                price_df = get_price(batch, end_date=yesterday, count=1, frequency='daily',
                                     fields=fields_fallback, panel=False, skip_paused=True)
            except Exception as e:
                try:
                    log.info(f"[get_yesterday_zt_stocks] 批次行情失败: {e}")
                except Exception:
                    pass
                continue

        part = _ensure_jq_zt_dataframe(price_df, all_stocks, yesterday)
        if part is not None and not part.empty:
            parts.append(part)

    if not parts:
        result = pd.DataFrame(columns=['jq_code', 'code', 'name', 'zt_date', 'zt_close', 'pct_change'])
    else:
        result = pd.concat(parts, ignore_index=True)
        result = result.drop_duplicates(subset=['jq_code']).reset_index(drop=True)

    try:
        log.info(f"[get_yesterday_zt_stocks] 昨日涨停股原始数: {len(result)} 只")
    except Exception:
        pass
    return result


def filter_stocks(context, stock_codes: List[str]) -> List[str]:
    """
    过滤 ST、停牌、上市不足 min_list_days 的股票。
    只返回 jq_code 列表；不改变原始 ZT 数量口径。
    """
    if not stock_codes:
        return []

    yesterday = context.previous_date
    stock_codes = [c for c in stock_codes if isinstance(c, str) and ('.XSHE' in c or '.XSHG' in c)]
    if not stock_codes:
        return []

    filtered = list(stock_codes)

    # 1) ST 过滤：get_extras + 名称双保险
    st_codes = set()
    try:
        st_flags = get_extras('is_st', filtered, end_date=yesterday, count=1)
        if st_flags is not None and not st_flags.empty:
            st_row = st_flags.iloc[-1]
            for code in filtered:
                if code in st_row.index and bool(st_row[code]):
                    st_codes.add(code)
    except Exception:
        pass

    try:
        all_stocks = get_all_securities('stock', date=yesterday)
        for code in filtered:
            if code in all_stocks.index:
                name = str(all_stocks.loc[code].display_name)
                if 'ST' in name.upper():
                    st_codes.add(code)
    except Exception:
        pass

    filtered = [c for c in filtered if c not in st_codes]

    # 2) 停牌过滤
    paused_codes = set()
    try:
        cur = get_current_data()
        for code in filtered:
            try:
                if cur[code].paused:
                    paused_codes.add(code)
            except Exception:
                pass
    except Exception:
        pass
    filtered = [c for c in filtered if c not in paused_codes]

    # 3) 上市天数过滤
    min_days = STRATEGY_CONFIG.get('min_list_days', 63)
    min_date = yesterday - timedelta(days=min_days)
    listed = []
    for code in filtered:
        try:
            info = get_security_info(code)
            if info is not None and info.start_date <= min_date:
                listed.append(code)
        except Exception:
            # 获取不到信息时保守剔除，避免异常代码进入池
            continue

    removed = len(stock_codes) - len(listed)
    try:
        if removed > 0:
            log.info(f"[filter_stocks] 过滤掉 {removed} 只股票 (ST/停牌/次新)，剩余 {len(listed)} 只")
    except Exception:
        pass
    return listed


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
            'one_two_score': np.nan,
            'one_two_rank_bucket': '',
            'one_two_filter_reason': '',
            'intraday_plan': '',
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

            # ======== 2. 获取涨停日至昨日的数据 ========
            # 实盘/回测统一: end_date 固定为昨日，只获取已知数据，避免未来函数
            end_date_str = context.previous_date.strftime('%Y-%m-%d')
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
                # 昨日涨停股：无涨停后数据，使用中性默认值
                # ratio_above_zt/defense 设为 0.5（中性），避免给未知表现最高分
                # 这使得 predict_next_day() 给出中等评分而非最高评分
                stock_price_data['overall_max_drawdown'] = np.nan
                stock_price_data['overall_max_return'] = np.nan
                stock_price_data['consecutive_up_post'] = 0
                stock_price_data['days_above_zt'] = np.nan
                stock_price_data['ratio_above_zt'] = 0.5      # 中性：无数据时不偏袒
                stock_price_data['days_above_defense'] = np.nan
                stock_price_data['ratio_above_defense'] = 0.5  # 中性：无数据时不偏袒

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

    # 一进二打板增强因子（盘后已知数据）
    df = calc_one_two_board_factors(df)

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


    # ---- 一进二因子融合：total_score = 70% 原始多因子 + 30% 一进二盘后因子 ----
    # 说明：total_score 仍用于“选股质量/候选池”过滤，不直接等同于建仓。
    if 'total_score' in df.columns:
        df['base_total_score'] = pd.to_numeric(df['total_score'], errors='coerce')
        if 'one_two_score' in df.columns:
            _ot = pd.to_numeric(_safe_series(df, 'one_two_score'), errors='coerce').fillna(50)
            df['one_two_score_weighted'] = (_ot * 0.30).round(2)
            df['multi_factor_score_weighted'] = (df['base_total_score'].fillna(0) * 0.70).round(2)
            df['total_score'] = (df['multi_factor_score_weighted'] + df['one_two_score_weighted']).clip(0, 100).round(2)
            if 'one_two_hard_filter' in df.columns:
                _hf = _safe_series(df, 'one_two_hard_filter').fillna(False).astype(bool)
                df.loc[_hf, 'total_score'] = df.loc[_hf, 'total_score'].clip(upper=35)
        else:
            df['one_two_score_weighted'] = np.nan
            df['multi_factor_score_weighted'] = df['base_total_score']

    # 按 total_score 降序排列
    df = df.sort_values('total_score', ascending=False, na_position='last').reset_index(drop=True)

    return df


# ============================================================================
# Section 10: 次日建仓预测 (从 zt_analysis.py 复用)
# ============================================================================

def predict_next_day(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成次日建仓预测表。

    新版逻辑：
    - total_score：70% 多因子 + 30% 一进二盘后因子，作为候选池质量分。
    - entry_index：55% 一进二盘后因子 + 25% total_score + 10% 量能 + 10% 资金/盘口。
    - 不使用竞价字段，避免盘后回测中误用未来函数；竞价只在 9:20-9:25 后由实时逻辑确认。
    """
    df = scored_df.copy()
    candidates = df[df['classification'].isin(['强势', '平稳'])].copy()
    if len(candidates) == 0:
        return pd.DataFrame()

    predict_scores = []
    for idx, row in candidates.iterrows():
        total_score = row.get('total_score', 0)
        if pd.isna(total_score):
            total_score = 0
        score_component = min(max(total_score, 0) / 100 * 25, 25)

        one_two_score = row.get('one_two_score', np.nan)
        if pd.isna(one_two_score):
            one_two_score = 50
        one_two_component = min(max(one_two_score, 0) / 100 * 55, 55)

        vol_ratio = row.get('vol_ratio', np.nan)
        turnover = row.get('turnover_rate', np.nan)
        vol_price = row.get('factor_vol_price', np.nan)
        volume_component = 0
        if not pd.isna(vol_ratio):
            if 1.5 <= vol_ratio <= 4.0:
                volume_component += 4
            elif vol_ratio >= 1.0:
                volume_component += 2.5
            elif vol_ratio > 0:
                volume_component += 1
        else:
            volume_component += 2
        if not pd.isna(turnover):
            if 5 <= turnover <= 20:
                volume_component += 4
            elif 3 <= turnover <= 25:
                volume_component += 2.5
            elif turnover > 0:
                volume_component += 1
        else:
            volume_component += 2
        if not pd.isna(vol_price):
            if vol_price > 1.0:
                volume_component += 2
            elif vol_price > 0:
                volume_component += 1
        else:
            volume_component += 1
        volume_component = min(volume_component, 10)

        capital_component = 0
        main_net = row.get('main_net_inflow', np.nan)
        main_pct = row.get('main_net_pct', np.nan)
        inner_outer = row.get('inner_outer_ratio', np.nan)
        bid_ask_ratio = row.get('bid_ask_ratio', np.nan)
        if pd.isna(main_net) and pd.isna(main_pct):
            capital_component += 3
        elif (not pd.isna(main_net) and main_net > 0) and (pd.isna(main_pct) or main_pct >= 0):
            capital_component += 4
        elif not pd.isna(main_net) and main_net >= -1e7:
            capital_component += 2
        if not pd.isna(inner_outer):
            if inner_outer < 0.8:
                capital_component += 3
            elif inner_outer < 1.0:
                capital_component += 2
            else:
                capital_component += 1
        else:
            capital_component += 2
        if not pd.isna(bid_ask_ratio):
            if bid_ask_ratio > 20:
                capital_component += 3
            elif bid_ask_ratio > 0:
                capital_component += 2
            else:
                capital_component += 1
        else:
            capital_component += 1
        capital_component = min(capital_component, 10)

        entry_index = score_component + one_two_component + volume_component + capital_component

        # v5: 一进二硬过滤/低分不再一票否决，改为软惩罚，避免回测阶段被锁死为0交易。
        if bool(row.get('one_two_hard_filter', False)):
            entry_index -= STRATEGY_CONFIG.get('one_two_hard_filter_penalty', 10)
        if one_two_score < 60:
            entry_index -= (60 - one_two_score) * STRATEGY_CONFIG.get('one_two_low_score_penalty_coef', 0.2)
        entry_index = max(0, min(100, entry_index))

        zt_close = row.get('zt_close', np.nan)
        if pd.isna(zt_close) or zt_close <= 0:
            buy_price = stop_loss = target_price = np.nan
        else:
            buy_price = zt_close * (1.00 if row.get('classification') == '强势' else 0.98)
            stop_loss = zt_close * 0.97
            target_price = zt_close * (1.10 if total_score >= 70 else 1.05 if total_score >= 50 else 1.03)

        if entry_index >= 75:
            prediction = '看多'
            signal = '🟢 积极建仓'
        elif entry_index >= 60:
            prediction = '偏多'
            signal = '🟡 适度建仓'
        elif entry_index >= 45:
            prediction = '震荡'
            signal = '🟠 观察/只打板确认'
        else:
            prediction = '偏空'
            signal = '🔴 不建议建仓'

        predict_scores.append({
            'entry_index': round(entry_index, 2),
            'score_component': round(score_component, 2),
            'one_two_component': round(one_two_component, 2),
            'volume_component': round(volume_component, 2),
            'capital_component': round(capital_component, 2),
            'one_two_score': row.get('one_two_score', np.nan),
            'one_two_rank_bucket': row.get('one_two_rank_bucket', ''),
            'one_two_filter_reason': row.get('one_two_filter_reason', ''),
            'intraday_plan': row.get('intraday_plan', ''),
            'buy_price': round(buy_price, 2) if not pd.isna(buy_price) else np.nan,
            'stop_loss': round(stop_loss, 2) if not pd.isna(stop_loss) else np.nan,
            'target_price': round(target_price, 2) if not pd.isna(target_price) else np.nan,
            'prediction': prediction,
            'signal': signal,
        })

    predict_df = pd.DataFrame(predict_scores, index=candidates.index)
    result_df = pd.concat([candidates, predict_df], axis=1)
    result_df = _dedup_columns(result_df)
    result_df = result_df.sort_values(['entry_index', 'total_score'], ascending=False).reset_index(drop=True)
    return result_df


# ============================================================================
# Section 11: 建仓信号生成
# ============================================================================

def generate_entry_signals(context, predict_df: pd.DataFrame) -> Dict:
    """
    v6: 生成“Top1 tick打板 + Top2~5 建仓指数买入”的交易信号。

    设计目标：
    - Top1：只做tick级打板确认，subscribe 后由 handle_tick(context, tick) 执行；
    - Top2~5：不再要求打板确认，只要 entry_index >= score_buy_min_entry_index，即可由 09:31 buy_score_candidates() 做实时过滤后买入；
    - 避免旧版必须 signal 含“积极/适度”才入选，导致全市场无交易。
    """
    signals = {}
    if predict_df is None or predict_df.empty:
        return signals

    held_codes = set(g.holdings.keys()) if hasattr(g, 'holdings') else set()
    df = predict_df.copy()

    # 基础列兜底
    if 'jq_code' not in df.columns:
        return signals
    if 'entry_index' not in df.columns:
        df['entry_index'] = 0
    if 'total_score' not in df.columns:
        df['total_score'] = 0
    if 'classification' not in df.columns:
        df['classification'] = ''
    if 'signal' not in df.columns:
        df['signal'] = ''

    # 跳过已持仓股票，按建仓指数优先、评分次之排序
    df = df[~df['jq_code'].isin(held_codes)].copy()
    if df.empty:
        return signals
    df['entry_index'] = pd.to_numeric(df['entry_index'], errors='coerce').fillna(0)
    df['total_score'] = pd.to_numeric(df['total_score'], errors='coerce').fillna(0)
    df = df.sort_values(['entry_index', 'total_score'], ascending=False).head(STRATEGY_CONFIG.get('max_entry_count', 5))

    min_score_buy = STRATEGY_CONFIG.get('score_buy_min_entry_index', 45)
    regime = getattr(g, 'v11_market_regime', 'RELAY_OK')
    disable_top1 = bool(getattr(g, 'disable_top1_tick_today', False))

    if regime in ('ICE', 'RELAY_WEAK'):
        log.info(f"[generate_entry_signals] v11市场状态={regime}，停止新增信号")
        return signals

    # v19：确认型情绪修复/主升。
    # 关键修正：REBOUND_RELAY 不再追Top1涨停，只做核心低吸；MAIN_UPTREND 才允许Top1。
    if regime in ('REBOUND_RELAY', 'MAIN_UPTREND'):
        df['v8_hot_score'] = pd.to_numeric(df.get('v8_hot_score', 0), errors='coerce').fillna(0)
        state_series = df.get('v8_hot_state', pd.Series('', index=df.index)).astype(str)
        if regime == 'MAIN_UPTREND':
            min_total = float(STRATEGY_CONFIG.get('v19_rebound_min_total_score', 55) or 55)
            min_entry = float(STRATEGY_CONFIG.get('v19_main_top_entry_min', 64) or 64)
            min_hot = float(STRATEGY_CONFIG.get('v19_main_top_hot_min', 125) or 125)
        else:
            min_total = float(STRATEGY_CONFIG.get('v19_rebound_min_total_score', 55) or 55)
            min_entry = float(STRATEGY_CONFIG.get('v19_rebound_min_entry_index', 63) or 63)
            min_hot = float(STRATEGY_CONFIG.get('v19_rebound_min_hot_score', 105) or 105)

        rdf = df[(df['total_score'] >= min_total) &
                 (df['entry_index'] >= min_entry) &
                 (df['v8_hot_score'] >= min_hot) &
                 (state_series.isin(['CORE_LEADER','HOT_CONTINUE']))].copy()
        # v20：修复期优先CORE_LEADER，避免HOT_CONTINUE泛化太宽；若无CORE再用HOT_CONTINUE。
        if regime == 'REBOUND_RELAY':
            core_rdf = rdf[state_series.loc[rdf.index].isin(['CORE_LEADER'])].copy()
            if not core_rdf.empty:
                rdf = core_rdf

        if rdf.empty:
            log.info(f"[generate_entry_signals] v20 {regime} 无确认核心候选 total>={min_total}, entry>={min_entry}, hot>={min_hot}")
            return signals

        rdf = rdf.sort_values(['entry_index', 'v8_hot_score', 'total_score'], ascending=False)

        allow_top1_tick = (regime == 'MAIN_UPTREND' and not disable_top1)
        if allow_top1_tick:
            top1_row = rdf.iloc[0]
            top1_code = top1_row.get('jq_code', '')
            if top1_code:
                signals[top1_code] = {
                    'entry_type': 'TOP1_TICK',
                    'rank': 1,
                    'classification': top1_row.get('classification', ''),
                    'signal': f'v20 {regime} 主升确认Top1 tick',
                    'buy_price': top1_row.get('buy_price', np.nan),
                    'stop_loss': top1_row.get('stop_loss', np.nan),
                    'target_price': top1_row.get('target_price', np.nan),
                    'entry_index': float(top1_row.get('entry_index', 0) or 0),
                    'total_score': float(top1_row.get('total_score', 0) or 0),
                    'v8_hot_state': top1_row.get('v8_hot_state', ''),
                    'v8_hot_score': float(top1_row.get('v8_hot_score', 0) or 0),
                    'v8_seen_count': int(top1_row.get('v8_seen_count', 0) or 0),
                    'one_two_score': top1_row.get('one_two_score', 0),
                    'one_two_rank_bucket': top1_row.get('one_two_rank_bucket', ''),
                    'one_two_filter_reason': top1_row.get('one_two_filter_reason', ''),
                    'intraday_plan': top1_row.get('intraday_plan', ''),
                    'first_leg_done': False,
                    'second_leg_done': True,
                    'v20_confirmed_rebound': True,
                }

        max_buys = int(STRATEGY_CONFIG.get('v19_main_max_score_buys', 1) if regime == 'MAIN_UPTREND' else STRATEGY_CONFIG.get('v19_rebound_max_score_buys', 1))
        score_rows = rdf[~rdf['jq_code'].isin(signals.keys())].head(max_buys)
        for rank, (_, row) in enumerate(score_rows.iterrows(), start=2 if signals else 1):
            code = row.get('jq_code', '')
            if not code:
                continue
            signals[code] = {
                'entry_type': 'SCORE_BUY',
                'rank': rank,
                'classification': row.get('classification', ''),
                'signal': f'v20 {regime} 确认型核心低吸Top{rank}',
                'buy_price': row.get('buy_price', np.nan),
                'stop_loss': row.get('stop_loss', np.nan),
                'target_price': row.get('target_price', np.nan),
                'entry_index': float(row.get('entry_index', 0) or 0),
                'total_score': float(row.get('total_score', 0) or 0),
                'v8_hot_state': row.get('v8_hot_state', ''),
                'v8_hot_score': float(row.get('v8_hot_score', 0) or 0),
                'v8_seen_count': int(row.get('v8_seen_count', 0) or 0),
                'one_two_score': row.get('one_two_score', 0),
                'one_two_rank_bucket': row.get('one_two_rank_bucket', ''),
                'one_two_filter_reason': row.get('one_two_filter_reason', ''),
                'intraday_plan': row.get('intraday_plan', ''),
                'first_leg_done': False,
                'second_leg_done': True,
                'v20_confirmed_rebound': True,
            }
        log.info(f"[generate_entry_signals] v20 {regime} 生成 {len(signals)} 个信号: { {v.get('entry_type'): list(s.get('entry_type') for s in signals.values()).count(v.get('entry_type')) for v in signals.values()} }")
        return signals

    # v11/v15：指数趋势市不做Top1打板；Top1也必须转成SCORE_BUY趋势低吸，且后续由09:31二次过滤。
    if regime in ('TREND_INDEX', 'TREND_REBOUND'):
        if regime == 'TREND_REBOUND':
            min_total = float(STRATEGY_CONFIG.get('v15_rebound_min_total_score', 45) or 45)
            min_entry = float(STRATEGY_CONFIG.get('v15_rebound_min_entry_index', 52) or 52)
            min_hot = float(STRATEGY_CONFIG.get('v15_rebound_min_hot_score', 80) or 80)
            df['v8_hot_score'] = pd.to_numeric(df.get('v8_hot_score', 0), errors='coerce').fillna(0)
            state_series = df.get('v8_hot_state', pd.Series('', index=df.index)).astype(str)
            df = df[(df['total_score'] >= min_total) & (df['entry_index'] >= min_entry) & ((df['v8_hot_score'] >= min_hot) | state_series.isin(['CORE_LEADER','HOT_CONTINUE']))].copy()
        else:
            min_total = float(STRATEGY_CONFIG.get('v11_trend_min_total_score', 55) or 55)
            min_entry = float(STRATEGY_CONFIG.get('v11_trend_min_entry_index', 60) or 60)
            df = df[(df['total_score'] >= min_total) & (df['entry_index'] >= min_entry)].copy()
        if df.empty:
            label = 'v15趋势修复' if regime == 'TREND_REBOUND' else 'v11趋势指数市'
            log.info(f"[generate_entry_signals] {label}无合格趋势低吸候选 total>={min_total}, entry>={min_entry}")
            return signals
        max_buys = STRATEGY_CONFIG.get('v15_rebound_max_score_buys', 1) if regime == 'TREND_REBOUND' else STRATEGY_CONFIG.get('v11_trend_max_score_buys', 1)
        for rank, (_, row) in enumerate(df.head(max_buys).iterrows(), start=1):
            code = row.get('jq_code', '')
            if not code:
                continue
            signals[code] = {
                'entry_type': 'SCORE_BUY',
                'rank': rank,
                'classification': row.get('classification', ''),
                'signal': f"{'v15趋势修复' if regime == 'TREND_REBOUND' else 'v11趋势指数市'}Top{rank}低吸",
                'buy_price': row.get('buy_price', np.nan),
                'stop_loss': row.get('stop_loss', np.nan),
                'target_price': row.get('target_price', np.nan),
                'entry_index': float(row.get('entry_index', 0) or 0),
                'total_score': float(row.get('total_score', 0) or 0),
                'v8_hot_state': row.get('v8_hot_state', ''),
                'v8_hot_score': float(row.get('v8_hot_score', 0) or 0),
                'v8_seen_count': int(row.get('v8_seen_count', 0) or 0),
                'one_two_score': row.get('one_two_score', 0),
                'one_two_rank_bucket': row.get('one_two_rank_bucket', ''),
                'one_two_filter_reason': row.get('one_two_filter_reason', ''),
                'intraday_plan': row.get('intraday_plan', ''),
                'first_leg_done': False,
                'second_leg_done': True,
            }
        log.info(f"[generate_entry_signals] {regime}生成 {len(signals)} 个SCORE_BUY低吸信号，禁用Top1打板")
        return signals

    # v13 HYBRID_GUARDED：不是v12的多tick打板，只允许一个严格筛选的Top1试探；
    # 同时继续保留后面的SCORE_BUY低吸补位，避免全天无交易。
    if regime == 'HYBRID_GUARDED':
        hdf = df.copy()
        hdf['v8_hot_score'] = pd.to_numeric(hdf.get('v8_hot_score', 0), errors='coerce').fillna(0)
        min_entry = float(STRATEGY_CONFIG.get('v14_hybrid_min_entry_index', 62) or 62)
        min_total = float(STRATEGY_CONFIG.get('v14_hybrid_min_total_score', 45) or 45)
        min_hot = float(STRATEGY_CONFIG.get('v14_hybrid_min_hot_score', 150) or 150)
        hdf = hdf[(hdf['entry_index'] >= min_entry) & (hdf['total_score'] >= min_total) & (hdf['v8_hot_score'] >= min_hot)].copy()
        if not hdf.empty and not disable_top1:
            top1_row = hdf.sort_values(['entry_index', 'total_score', 'v8_hot_score'], ascending=False).iloc[0]
            top1_code = top1_row.get('jq_code', '')
            if top1_code:
                signals[top1_code] = {
                    'entry_type': 'TOP1_TICK',
                    'rank': 1,
                    'classification': top1_row.get('classification', ''),
                    'signal': 'v14混合行情严格Top1小仓tick打板确认',
                    'buy_price': top1_row.get('buy_price', np.nan),
                    'stop_loss': top1_row.get('stop_loss', np.nan),
                    'target_price': top1_row.get('target_price', np.nan),
                    'entry_index': float(top1_row.get('entry_index', 0) or 0),
                    'total_score': float(top1_row.get('total_score', 0) or 0),
                    'v8_hot_state': top1_row.get('v8_hot_state', ''),
                    'v8_hot_score': float(top1_row.get('v8_hot_score', 0) or 0),
                    'v8_seen_count': int(top1_row.get('v8_seen_count', 0) or 0),
                    'one_two_score': top1_row.get('one_two_score', 0),
                    'one_two_rank_bucket': top1_row.get('one_two_rank_bucket', ''),
                    'one_two_filter_reason': top1_row.get('one_two_filter_reason', ''),
                    'intraday_plan': top1_row.get('intraday_plan', ''),
                    'first_leg_done': False,
                    'second_leg_done': True,
                    'v14_guarded_hybrid': True,
                }
        else:
            log.info(f"[generate_entry_signals] v14混合行情无严格tick候选 entry>={min_entry}, total>={min_total}, hot>={min_hot}")
    else:
        # Top1：tick打板专用。即便 entry_index 略低，也允许“触板才买”，因为tick确认本身是强过滤。
        top1_row = df.iloc[0]
        top1_code = top1_row.get('jq_code', '')
        if top1_code and not disable_top1:
            signals[top1_code] = {
                'entry_type': 'TOP1_TICK',
                'rank': 1,
                'classification': top1_row.get('classification', ''),
                'signal': 'Top1 tick打板确认',
                'buy_price': top1_row.get('buy_price', np.nan),
                'stop_loss': top1_row.get('stop_loss', np.nan),
                'target_price': top1_row.get('target_price', np.nan),
                'entry_index': float(top1_row.get('entry_index', 0) or 0),
                'total_score': float(top1_row.get('total_score', 0) or 0),
                'v8_hot_state': top1_row.get('v8_hot_state', ''),
                'v8_hot_score': float(top1_row.get('v8_hot_score', 0) or 0),
                'v8_seen_count': int(top1_row.get('v8_seen_count', 0) or 0),
                'one_two_score': top1_row.get('one_two_score', 0),
                'one_two_rank_bucket': top1_row.get('one_two_rank_bucket', ''),
                'one_two_filter_reason': top1_row.get('one_two_filter_reason', ''),
                'intraday_plan': top1_row.get('intraday_plan', ''),
                'first_leg_done': False,
                'second_leg_done': True,
            }

    # Top2~5：保交易和稳定性，只看 entry_index >= 45；真正执行由 buy_score_candidates() 在09:31统一处理。
    for rank, (_, row) in enumerate(df.iloc[1:5].iterrows(), start=2):
        code = row.get('jq_code', '')
        if not code:
            continue
        entry_index = float(row.get('entry_index', 0) or 0)
        if entry_index < min_score_buy:
            continue
        signals[code] = {
            'entry_type': 'SCORE_BUY',
            'rank': rank,
            'classification': row.get('classification', ''),
            'signal': f'Top{rank} 建仓指数买入',
            'buy_price': row.get('buy_price', np.nan),
            'stop_loss': row.get('stop_loss', np.nan),
            'target_price': row.get('target_price', np.nan),
            'entry_index': entry_index,
            'total_score': float(row.get('total_score', 0) or 0),
            'v8_hot_state': row.get('v8_hot_state', ''),
            'v8_hot_score': float(row.get('v8_hot_score', 0) or 0),
            'v8_seen_count': int(row.get('v8_seen_count', 0) or 0),
            'one_two_score': row.get('one_two_score', 0),
            'one_two_rank_bucket': row.get('one_two_rank_bucket', ''),
            'one_two_filter_reason': row.get('one_two_filter_reason', ''),
            'intraday_plan': row.get('intraday_plan', ''),
            'first_leg_done': False,
            'second_leg_done': True,
        }

    if signals:
        type_counts = {}
        for sig in signals.values():
            t = sig.get('entry_type', '')
            type_counts[t] = type_counts.get(t, 0) + 1
        log.info(f"[generate_entry_signals] v14生成 {len(signals)} 个建仓信号: {type_counts}; "
                 f"Top1=tick打板/混合严格试探, Top2~5 entry_index>={min_score_buy}")
    else:
        log.info(f"[generate_entry_signals] v13无信号：候选不足或Top2~5均低于entry_index>={min_score_buy}")

    return signals


def setup_tick_subscriptions(context) -> None:
    if getattr(g, 'disable_top1_tick_today', False):
        try:
            unsubscribe_all()
        except Exception:
            pass
        log.info(f"[setup_tick_subscriptions] v11市场状态={getattr(g, 'v11_market_regime', 'UNKNOWN')}，禁用Top1 tick订阅")
        return
    """订阅 Top1 的 tick。Top2~5 不订阅tick，由09:31 buy_score_candidates()执行。"""
    try:
        unsubscribe_all()
    except Exception:
        pass

    g.top1_tick_code = None
    g.top1_tick_limit = None

    if not getattr(g, 'entry_signals', None):
        return

    top1_code = None
    for code, sig in g.entry_signals.items():
        if sig.get('entry_type') == 'TOP1_TICK':
            top1_code = code
            break
    if not top1_code:
        return

    try:
        cur_data = get_current_data()
        high_limit = cur_data[top1_code].high_limit
        if high_limit and high_limit > 0:
            g.top1_tick_code = top1_code
            g.top1_tick_limit = high_limit
            subscribe(top1_code, 'tick')
            log.info(f"[setup_tick_subscriptions] 已订阅Top1 tick: {top1_code}, high_limit={high_limit:.2f}")
    except Exception as e:
        log.info(f"[setup_tick_subscriptions] 订阅Top1 tick失败: {top1_code}, {e}")


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
            # 仅在整点输出日志，避免每分钟重复
            current_minute = context.current_dt.minute
            if current_minute == 0 or not hasattr(g, '_last_max_holdings_log_date'):
                log.info(f"[execute_entry] 持仓数已达上限 {max_holdings}，停止建仓")
                g._last_max_holdings_log_date = context.current_dt.date()
            break

        signal = signals[code]
        entry_type = signal['entry_type']

        if entry_type == 'TOP1_TICK':
            # Top1 由 handle_tick(context, tick) 执行，handle_data 不买，避免非打板成交。
            continue
        elif entry_type == 'SCORE_BUY':
            # v6.1：Top2~5由 buy_score_candidates() 在09:31固定执行，
            # 不再依赖 handle_data/every_bar，避免tick回测环境中 every_bar 不稳定或重复下单。
            continue
        elif entry_type == 'TYPE_A':
            executed = execute_type_a(context, data, code, signal)
        elif entry_type == 'TYPE_B':
            executed = execute_type_b(context, data, code, signal)
        elif entry_type == 'TYPE_C':
            executed = execute_type_c(context, data, code, signal)
        else:
            continue

        if executed:
            current_holdings = len(g.holdings)


def execute_score_buy(context, data, code: str, signal: Dict) -> bool:
    """
    v6: Top2~5 建仓指数买入。
    不要求打板确认；只做基础风控：交易时间、未买过、未持仓、未涨停一字买不到、entry_index阈值。
    """
    if signal.get('first_leg_done'):
        return False
    if code in getattr(g, 'bought_today', set()):
        return False
    if code in context.portfolio.positions and context.portfolio.positions[code].total_amount > 0:
        return False

    now_t = context.current_dt.time()
    if now_t < dt.time(9, 31) or now_t > dt.time(14, 30):
        return False

    entry_index = float(signal.get('entry_index', 0) or 0)
    if entry_index < STRATEGY_CONFIG.get('score_buy_min_entry_index', 45):
        return False

    try:
        cur_data = get_current_data()
        cd = cur_data[code]
        price = cd.last_price
        if price is None or price <= 0 or cd.paused:
            return False
        # 已经涨停时，普通评分票不追板；Top1才打板。
        if cd.high_limit and price >= cd.high_limit * 0.997:
            log.debug(f"[SCORE_BUY] {code} 已接近涨停，非Top1不追板")
            return False
        # 若当日大幅低开/弱势，可按需过滤；这里先保留，以便回测产生交易。
        shares = calc_position_size(context, code, ratio=STRATEGY_CONFIG.get('score_buy_position_ratio', 0.75))
        if shares <= 0:
            return False
        order_result = order(code, shares)
        g.bought_today.add(code)
        if order_result is not None:
            signal['first_leg_done'] = True
            log.info(f"[SCORE_BUY] Top{signal.get('rank','?')} {code} 建仓指数买入 {shares} 股 | entry_index={entry_index:.1f}")
            _record_holding(context, code, signal, shares, leg='full')
            return True
    except Exception as e:
        log.info(f"[SCORE_BUY] {code} 买入失败: {e}")
    return False


def execute_type_a(context, data, code: str, signal: Dict) -> bool:
    """
    TYPE_A: 强势+积极建仓
    - 开盘100%市价买入（JQ笼子机制自动限制价格在上限内）
    """
    if not signal['first_leg_done']:
        ok, reason = check_intraday_one_two_entry(context, code, signal)
        if not ok:
            log.debug(f"[TYPE_A] {code} 一进二盘中确认未通过: {reason}")
            return False
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
        ok, reason = check_intraday_one_two_entry(context, code, signal)
        if not ok:
            log.debug(f"[TYPE_C] {code} 一进二盘中确认未通过: {reason}")
            return False
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



def _get_position_amount(context, code: str) -> int:
    """安全获取真实持仓股数。"""
    try:
        pos = context.portfolio.positions.get(code)
        if pos is not None and getattr(pos, 'total_amount', 0) > 0:
            return int(pos.total_amount)
    except Exception:
        pass
    return 0


def _sync_holdings_with_portfolio(context) -> None:
    """
    v6.2：用 JoinQuant 真实 portfolio 校正 g.holdings，避免日志显示持仓但真实账户无持仓，
    也避免平台胜率与内部统计长期不闭环。
    """
    try:
        positions = context.portfolio.positions
    except Exception:
        positions = {}

    # 1) 删除内部有、真实无的持仓
    for code in list(getattr(g, 'holdings', {}).keys()):
        amount = _get_position_amount(context, code)
        if amount <= 0:
            g.holdings.pop(code, None)
            g.trailing_stops.pop(code, None)
            g.open_trades.pop(code, None)
            g.pending_sells.discard(code)
        else:
            g.holdings[code]['shares'] = amount
            try:
                pos = positions.get(code)
                if pos is not None and getattr(pos, 'avg_cost', 0) > 0:
                    g.holdings[code]['buy_price'] = float(pos.avg_cost)
                    g.holdings[code]['amount'] = float(pos.avg_cost) * amount
            except Exception:
                pass

    # 2) 真实有、内部无的持仓，补一个最小记录，避免风控漏检
    for code, pos in getattr(positions, 'items', lambda: [])():
        try:
            amount = int(getattr(pos, 'total_amount', 0) or 0)
            if amount <= 0 or code in g.holdings:
                continue
            avg_cost = float(getattr(pos, 'avg_cost', 0) or 0)
            if avg_cost <= 0:
                avg_cost = float(getattr(pos, 'price', 0) or 0)
            g.holdings[code] = {
                'buy_date': context.current_dt.date(),
                'buy_price': avg_cost,
                'shares': amount,
                'amount': avg_cost * amount,
                'stop_loss': avg_cost * (1 - STRATEGY_CONFIG.get('daily_stop_loss_pct', 0.05)),
                'target_price': avg_cost * (1 + STRATEGY_CONFIG.get('t1_profit_take_pct', 0.09)),
                'highest_price': avg_cost,
                'entry_type': 'SYNCED',
                'entry_index': 0,
                'total_score': 0,
                'v8_hot_state': '',
                'v8_hot_score': 0,
                'v10_runner': False,
            }
            log.info(f"[持仓同步] 补录真实持仓 {code} amount={amount}, avg_cost={avg_cost:.2f}")
        except Exception:
            continue


def _record_trade_buy(context, code: str, signal: Dict, price: float, shares: int) -> None:
    """v6.2：记录买入流水与未平仓交易。"""
    if shares <= 0 or price is None or pd.isna(price) or price <= 0:
        return
    rec = {
        'side': 'BUY',
        'code': code,
        'name': _get_stock_name(code),
        'date': context.current_dt.date(),
        'datetime': context.current_dt,
        'price': float(price),
        'shares': int(shares),
        'amount': float(price) * int(shares),
        'entry_type': signal.get('entry_type', ''),
        'entry_index': float(signal.get('entry_index', 0) or 0),
        'rank': signal.get('rank', None),
    }
    # 同一股票未平仓时不重复覆盖为新的独立交易；加仓则更新加权成本
    if code in g.open_trades:
        old = g.open_trades[code]
        old_amount = old['price'] * old['shares']
        new_amount = rec['price'] * rec['shares']
        total_shares = old['shares'] + rec['shares']
        old['price'] = (old_amount + new_amount) / total_shares if total_shares > 0 else old['price']
        old['shares'] = total_shares
        old['amount'] = old['price'] * total_shares
    else:
        g.open_trades[code] = rec.copy()
    g.trade_records.append(rec)


def _record_trade_sell(context, code: str, price: float, shares: int, reason: str) -> None:
    """v6.2：记录卖出流水，并根据 open_trades 形成闭环盈亏。"""
    if shares <= 0 or price is None or pd.isna(price) or price <= 0:
        return
    buy_rec = g.open_trades.get(code, {})
    buy_price = float(buy_rec.get('price', g.holdings.get(code, {}).get('buy_price', price)) or price)
    pnl = (float(price) - buy_price) * int(shares)
    pnl_pct = (float(price) - buy_price) / buy_price if buy_price > 0 else 0.0
    rec = {
        'side': 'SELL',
        'code': code,
        'name': _get_stock_name(code),
        'date': context.current_dt.date(),
        'datetime': context.current_dt,
        'price': float(price),
        'shares': int(shares),
        'amount': float(price) * int(shares),
        'buy_price': buy_price,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'reason': reason,
        'entry_type': buy_rec.get('entry_type', g.holdings.get(code, {}).get('entry_type', '')),
        'entry_index': buy_rec.get('entry_index', g.holdings.get(code, {}).get('entry_index', 0)),
    }
    g.trade_records.append(rec)
    g.open_trades.pop(code, None)


def _log_trade_stats(context) -> None:
    """输出自定义胜率/盈亏比，避免平台因未闭环或持仓统计异常显示0。"""
    sells = [r for r in getattr(g, 'trade_records', []) if r.get('side') == 'SELL' and 'pnl_pct' in r]
    if not sells:
        log.info("📊 v6.2交易统计: 暂无已平仓交易，胜率/盈亏比待形成闭环")
        return
    wins = [r for r in sells if r.get('pnl_pct', 0) > 0]
    losses = [r for r in sells if r.get('pnl_pct', 0) <= 0]
    win_rate = len(wins) / len(sells) if sells else 0
    avg_win = np.mean([r['pnl_pct'] for r in wins]) if wins else 0
    avg_loss = abs(np.mean([r['pnl_pct'] for r in losses])) if losses else 0
    profit_factor = (sum([r['pnl'] for r in wins]) / abs(sum([r['pnl'] for r in losses]))) if losses and abs(sum([r['pnl'] for r in losses])) > 0 else np.nan
    avg_pnl = np.mean([r['pnl_pct'] for r in sells]) if sells else 0
    log.info(f"📊 v6.2交易统计: 平仓{len(sells)}笔 | 胜率{win_rate:.1%} | 平均收益{avg_pnl:.2%} | 平均盈利{avg_win:.2%} | 平均亏损{avg_loss:.2%} | 盈亏比/ProfitFactor={profit_factor if not pd.isna(profit_factor) else 'NA'}")


def _record_holding(context, code: str, signal: Dict, shares: int, leg: str = 'full') -> None:
    """
    记录持仓信息到 g.holdings。
    v6.2：仅在订单实际成交后调用，并同步记录 BUY 流水，便于后续计算真实胜率。
    """
    try:
        cur_data = get_current_data()
        buy_price = cur_data[code].last_price
    except Exception:
        buy_price = signal.get('buy_price', np.nan)

    # 用实际持仓数据校正
    actual_shares = shares
    try:
        position = context.portfolio.positions.get(code)
        if position is not None and position.total_amount > 0:
            actual_shares = int(position.total_amount)
            if getattr(position, 'avg_cost', 0) and position.avg_cost > 0:
                buy_price = float(position.avg_cost)
    except Exception:
        pass

    if actual_shares <= 0:
        return

    if code not in g.holdings:
        g.holdings[code] = {
            'buy_date': context.current_dt.date(),
            'buy_price': buy_price,
            'shares': actual_shares,
            'amount': buy_price * actual_shares if not pd.isna(buy_price) else 0,
            'stop_loss': signal.get('stop_loss', np.nan),
            'target_price': signal.get('target_price', np.nan),
            'highest_price': buy_price if not pd.isna(buy_price) else 0,
            'entry_type': signal.get('entry_type', ''),
            'entry_index': signal.get('entry_index', 0),
            'total_score': signal.get('total_score', 0),
            'v8_hot_state': signal.get('v8_hot_state', ''),
            'v8_hot_score': signal.get('v8_hot_score', 0),
            'v8_seen_count': signal.get('v8_seen_count', 0),
            'v10_runner': False,
        }
        _record_trade_buy(context, code, signal, buy_price, actual_shares)
        _v10_mark_holding_runner(code)
    else:
        # 更新已有持仓（加仓）：按真实持仓校正；BUY流水只记录新增部分，无法确认新增时只更新成本。
        h = g.holdings[code]
        old_shares = int(h.get('shares', 0) or 0)
        new_shares = max(actual_shares - old_shares, 0)
        h['shares'] = actual_shares
        if not pd.isna(buy_price) and buy_price > 0:
            h['buy_price'] = buy_price
            h['amount'] = buy_price * actual_shares
            h['highest_price'] = max(h.get('highest_price', 0), buy_price)
        if new_shares > 0:
            _record_trade_buy(context, code, signal, buy_price, new_shares)


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
                    # 封涨停日志每15分钟输出一次，避免刷屏
                    _log_throttle(context, code, 'zt_hold',
                                  f"[止盈] {code} T+{hold_days}利润 {profit_pct:.1%} > {STRATEGY_CONFIG['t1_profit_take_pct']:.0%}，但封涨停，继续持股",
                                  interval_minutes=15)
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


def _log_throttle(context, key, message, interval_minutes=15):
    """
    节流日志输出：同一key在指定分钟间隔内只输出一次。
    
    Parameters
    ----------
    context : JQ context
    key : str
        唯一标识，如 'zt_hold_000001.XSHE'
    message : str
        日志内容
    interval_minutes : int
        最小输出间隔（分钟）
    """
    if not hasattr(g, '_log_throttle_cache'):
        g._log_throttle_cache = {}

    now = context.current_dt
    cache = g._log_throttle_cache

    if key in cache:
        elapsed = (now - cache[key]).total_seconds() / 60
        if elapsed < interval_minutes:
            return

    cache[key] = now
    log.info(message)


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
    v6.2 卖出持仓。
    - 遵守A股T+1；
    - 用固定时间风控触发；
    - 下单后记录 SELL 流水，形成自定义胜率/盈亏比统计；
    - 卖出后清理 g.holdings / entry_signals，避免长期假持仓导致回撤失真。
    """
    if code not in g.holdings:
        return

    holding = g.holdings.get(code, {})
    name = _get_stock_name(code)
    today = context.current_dt.date()
    buy_date = holding.get('buy_date')

    # T+1规则：当日新建仓股票不能卖出
    if buy_date is not None and buy_date == today:
        log.debug(f"[_sell_position] {code} {name} 当日新建仓(T+0)，不能卖出 (原因: {reason})")
        return

    # 真实持仓检查
    real_shares = _get_position_amount(context, code)
    if real_shares <= 0:
        log.info(f"[_sell_position] {code} {name} 真实无持仓，清理内部记录")
        g.holdings.pop(code, None)
        g.entry_signals.pop(code, None)
        g.trailing_stops.pop(code, None)
        g.open_trades.pop(code, None)
        return

    try:
        cur_data = get_current_data()
        sell_price = float(cur_data[code].last_price or 0)
    except Exception:
        sell_price = np.nan

    try:
        order_result = order_target(code, 0)
    except Exception as e:
        log.info(f"[_sell_position] {code} {name} 卖出异常: {e}")
        return

    if order_result is None:
        log.info(f"[_sell_position] {code} {name} 卖出委托返回None，可能停牌/跌停，保留持仓继续风控")
        g.pending_sells.add(code)
        return

    order_error = getattr(order_result, 'error', None) or getattr(order_result, 'comment', '')
    if order_error:
        log.info(f"[_sell_position] {code} {name} 卖出委托异常: {order_error}，保留持仓继续风控")
        g.pending_sells.add(code)
        return

    # 回测中多数可立即成交；为了让胜率形成闭环，按当前价记录SELL。
    buy_price = holding.get('buy_price', np.nan)
    if pd.isna(sell_price) or sell_price <= 0:
        sell_price = buy_price if not pd.isna(buy_price) else 0
    if sell_price > 0:
        _record_trade_sell(context, code, sell_price, real_shares, reason)
        if buy_price and not pd.isna(buy_price) and buy_price > 0:
            pnl_pct = (sell_price - buy_price) / buy_price
            log.info(f"[_sell_position] {code} {name} 卖出 {real_shares}股 | 原因: {reason} | 买入价:{buy_price:.2f} 卖出价:{sell_price:.2f} 盈亏:{pnl_pct:.2%}")
        else:
            log.info(f"[_sell_position] {code} {name} 卖出 {real_shares}股 | 原因: {reason} | 卖出价:{sell_price:.2f}")

    # 清理内部记录，下一次 _sync_holdings_with_portfolio 如发现真实仍持仓会补录，避免遗漏。
    g.holdings.pop(code, None)
    g.entry_signals.pop(code, None)
    g.trailing_stops.pop(code, None)
    g.pending_sells.discard(code)


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
    _sync_holdings_with_portfolio(context)
    today = context.current_dt.date()
    log.info("=" * 80)
    log.info(f"📊 涨停板策略日报 — {today}")
    log.info("=" * 80)

    # 1. 股票池大小和ZT股数量
    pool = g.stock_pool
    zt_count = getattr(g, 'zt_count_raw', g.zt_count_yesterday)
    zt_tradeable = getattr(g, 'zt_count_tradeable', g.zt_count_yesterday)
    log.info(f"📋 股票池大小: {len(pool)}, 昨日ZT原始数: {zt_count}, 过滤后ZT数: {zt_tradeable}")
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
    log.info(f"📈 昨日ZT原始数量: {zt_count}, 过滤后可交易数量: {zt_tradeable}")

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

    # v6.3：每日盘后固定输出自定义胜率/平均收益/盈亏比。
    # 之前虽然有 _log_trade_stats()，但没有挂到日报函数里，因此日志中看不到统计。
    try:
        _log_trade_stats_v63(context)
    except Exception as e:
        log.info(f"[log_daily_summary] v8交易统计输出异常: {e}")

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


# ============================================================================
# v6.3 市场情绪与统计增强
# ============================================================================

def _v11_safe_get_index_trend(context) -> Dict:
    """
    v11：用沪深300最近5/20日收益判断是否处于“指数趋势市”。
    只使用当前日前可见日线数据；取不到数据时返回中性。
    """
    out = {'ret5': 0.0, 'ret20': 0.0, 'ok': False}
    try:
        idx = STRATEGY_CONFIG.get('v11_index_code', '000300.XSHG')
        mid = int(STRATEGY_CONFIG.get('v11_index_mid_window', 20) or 20)
        short = int(STRATEGY_CONFIG.get('v11_index_short_window', 5) or 5)
        cnt = max(mid, short) + 2
        end_dt = context.current_dt - timedelta(days=1)
        px = get_price(idx, end_date=end_dt, count=cnt, frequency='daily', fields=['close'], panel=False)
        if px is None or len(px) < max(mid, short) + 1:
            return out
        close = pd.Series(px['close']).astype(float).dropna()
        if len(close) < max(mid, short) + 1:
            return out
        out['ret5'] = float(close.iloc[-1] / close.iloc[-1-short] - 1.0)
        out['ret20'] = float(close.iloc[-1] / close.iloc[-1-mid] - 1.0)
        out['ok'] = True
    except Exception as e:
        try:
            log.info(f"[v11市场状态] 指数趋势读取失败: {e}")
        except Exception:
            pass
    return out


def _v11_recent_relay_stats(context) -> Dict:
    """
    v11：最近平仓交易的平均收益/胜率，用来判断当前涨停接力是否还赚钱。
    没有足够交易时保持中性，避免样本不足导致误杀。
    """
    n = int(STRATEGY_CONFIG.get('v11_relay_recent_trade_n', 8) or 8)
    sells = [r for r in getattr(g, 'trade_records', []) if r.get('side') == 'SELL' and 'pnl_pct' in r]
    recent = sells[-n:]
    if len(recent) < max(3, n // 2):
        return {'n': len(recent), 'avg': 0.0, 'win_rate': 0.5, 'ok': False}
    vals = [float(r.get('pnl_pct', 0) or 0) for r in recent]
    return {
        'n': len(vals),
        'avg': float(np.mean(vals)),
        'win_rate': float(len([v for v in vals if v > 0]) / len(vals)),
        'ok': True,
    }



def _v18_hot_breadth_snapshot(context) -> Dict:
    """
    v18：从10日热点记忆池读取情绪宽度。
    这是盘前已知数据，不使用未来函数。
    """
    out = {
        'core': 0,
        'hot_continue': 0,
        'trend': 0,
        'top_hot': 0.0,
        'top_entry': 0.0,
        'temperature': 0.0,
    }
    try:
        _v8_ensure_hot_memory()
        items = list(getattr(g, 'hot_memory_pool', {}).values())
        if not items:
            return out
        valid = []
        for item in items:
            try:
                score = float(item.get('hot_score', 0) or 0)
                entry = float(item.get('last_entry_index', 0) or 0)
            except Exception:
                score, entry = 0.0, 0.0
            state = str(item.get('state', ''))
            if score <= 0:
                continue
            valid.append((score, entry, state))
            if state == 'CORE_LEADER':
                out['core'] += 1
            elif state == 'HOT_CONTINUE':
                out['hot_continue'] += 1
            elif state == 'TREND':
                out['trend'] += 1
        if not valid:
            return out
        valid.sort(key=lambda x: x[0], reverse=True)
        top = valid[:10]
        out['top_hot'] = float(np.mean([x[0] for x in top]))
        out['top_entry'] = float(np.mean([x[1] for x in top]))
        # 情绪温度：0~100。情绪宽度优先，指数只作为辅助过滤。
        raw_count = int(getattr(g, 'zt_count_raw', 0) or 0)
        temp = 0.0
        temp += min(raw_count / 2.0, 35.0)              # rawZT=70 => 35分
        temp += min(out['core'] * 4.0, 25.0)            # 龙头数量
        temp += min(out['hot_continue'] * 2.0, 20.0)    # 续强扩散
        temp += min(max(out['top_hot'] - 40.0, 0) / 3.0, 15.0)
        temp += min(max(out['top_entry'] - 50.0, 0), 10.0)
        out['temperature'] = float(min(temp, 100.0))
    except Exception as e:
        try:
            log.info(f"[v18情绪宽度] 计算异常: {e}")
        except Exception:
            pass
    return out


def _v18_emotion_regime_override(raw_count: int, emotion: str, idx: Dict, relay: Dict, hb: Dict) -> Optional[str]:
    """
    v18：情绪优先覆盖器。
    旧逻辑是“指数弱 => ICE”；v18 改为“指数弱 + 情绪强 => REBOUND_RELAY/MAIN_UPTREND”。
    """
    if not STRATEGY_CONFIG.get('v18_enable_emotion_first_regime', True):
        return None

    idx5 = float(idx.get('ret5', 0) or 0)
    idx20 = float(idx.get('ret20', 0) or 0)
    core = int(hb.get('core', 0) or 0)
    hotc = int(hb.get('hot_continue', 0) or 0)
    top_hot = float(hb.get('top_hot', 0) or 0)
    top_entry = float(hb.get('top_entry', 0) or 0)
    temp = float(hb.get('temperature', 0) or 0)

    # 硬冰点：指数弱 + 情绪也弱，才允许 ICE。
    hard_ice = (
        raw_count < int(STRATEGY_CONFIG.get('v18_ice_rawzt_hard_min', 45) or 45)
        and core <= int(STRATEGY_CONFIG.get('v18_ice_core_max', 1) or 1)
        and top_hot <= float(STRATEGY_CONFIG.get('v18_ice_top_hot_max', 55) or 55)
    )
    if hard_ice:
        return 'ICE'

    # v19 主升：必须是情绪确认后的主线扩散，不再因为单日温度高就放开攻击。
    main_ok = (
        raw_count >= int(STRATEGY_CONFIG.get('v19_main_rawzt_min', 100) or 100)
        and core >= int(STRATEGY_CONFIG.get('v19_main_core_min', 8) or 8)
        and top_hot >= float(STRATEGY_CONFIG.get('v19_main_top_hot_min', 125) or 125)
        and top_entry >= float(STRATEGY_CONFIG.get('v19_main_top_entry_min', 64) or 64)
        and temp >= float(STRATEGY_CONFIG.get('v19_main_temperature_min', 85) or 85)
    )
    if main_ok:
        return 'MAIN_UPTREND'

    # v19 冰点修复：只在修复被确认后开启；过早修复日继续观望，避免v18高频噪音。
    rebound_ok = (
        raw_count >= int(STRATEGY_CONFIG.get('v19_rebound_rawzt_min', 85) or 85)
        and core >= int(STRATEGY_CONFIG.get('v19_rebound_core_min', 5) or 5)
        and hotc >= int(STRATEGY_CONFIG.get('v19_rebound_hot_continue_min', 5) or 5)
        and top_hot >= float(STRATEGY_CONFIG.get('v19_rebound_top_hot_min', 105) or 105)
        and top_entry >= float(STRATEGY_CONFIG.get('v19_rebound_top_entry_min', 61) or 61)
        and idx5 >= float(STRATEGY_CONFIG.get('v19_rebound_idx5_min', -0.045) or -0.045)
        and idx20 >= float(STRATEGY_CONFIG.get('v19_rebound_idx20_min', -0.09) or -0.09)
        and temp >= float(STRATEGY_CONFIG.get('v19_rebound_temperature_min', 78) or 78)
    )
    if rebound_ok:
        return 'REBOUND_RELAY'

    return None

def _update_market_regime_v11(context) -> None:
    """
    v11 市场风格状态机。
    目标：当指数趋势上涨但接力生态走弱时，停止高位打板，避免策略与基准反向。
    """
    if not STRATEGY_CONFIG.get('v11_enable_market_regime', True):
        g.v11_market_regime = 'LEGACY'
        g.disable_top1_tick_today = False
        return

    raw_count = int(getattr(g, 'zt_count_raw', 0) or 0)
    emotion = getattr(g, 'market_emotion_state', 'UNKNOWN')
    idx = _v11_safe_get_index_trend(context)
    relay = _v11_recent_relay_stats(context)
    hb = _v18_hot_breadth_snapshot(context)

    idx_up = bool(idx.get('ok') and idx.get('ret20', 0) >= float(STRATEGY_CONFIG.get('v11_index_up_threshold', 0.015)))
    idx_down = bool(idx.get('ok') and idx.get('ret20', 0) <= float(STRATEGY_CONFIG.get('v11_index_down_threshold', -0.025)))
    relay_bad = bool(relay.get('ok') and relay.get('avg', 0) <= float(STRATEGY_CONFIG.get('v11_relay_recent_bad_avg_pnl', -0.006)))
    relay_good = bool(relay.get('ok') and relay.get('avg', 0) >= float(STRATEGY_CONFIG.get('v11_relay_recent_good_avg_pnl', 0.012)))

    # v15：后半段上涨修复。
    # 当指数短线重新转强、昨日ZT仍不低、且不是指数大级别下跌时，
    # 不再把市场完全判成 RELAY_WEAK/ICE；允许一个小仓位核心趋势低吸。
    rebound_ok = bool(
        STRATEGY_CONFIG.get('v15_enable_late_rebound', True)
        and idx.get('ok')
        and idx.get('ret5', 0) >= float(STRATEGY_CONFIG.get('v15_rebound_index_ret5_min', 0.012))
        and idx.get('ret20', 0) >= float(STRATEGY_CONFIG.get('v15_rebound_index_ret20_min', -0.015))
        and raw_count >= int(STRATEGY_CONFIG.get('v15_rebound_rawzt_min', 55) or 55)
        and not idx_down
        and emotion != 'WEAK'
    )

    # v13：保留v11“指数趋势市防守”的主框架，但增加受保护的 HYBRID_GUARDED。
    # 只有当指数向上、昨日涨停数达到HOT阈值、且情绪为HOT时，才允许一个小仓位Top1试探；
    # rawZT 100~129 仍按 TREND_INDEX 处理，避免v12那种把普通趋势市误判成接力市。
    hybrid_raw = int(STRATEGY_CONFIG.get('v14_hybrid_rawzt_threshold', 130) or 130)
    require_hot = bool(STRATEGY_CONFIG.get('v14_hybrid_require_emotion_hot', True))
    hybrid_ok = bool(
        STRATEGY_CONFIG.get('v13_enable_guarded_hybrid', True)
        and idx_up
        and raw_count >= hybrid_raw
        and ((emotion == 'HOT') or (not require_hot))
        and not relay_bad
    )

    # v18：先让情绪覆盖器判断。情绪足够强时，不能被指数弱直接压成ICE。
    override_regime = _v18_emotion_regime_override(raw_count, emotion, idx, relay, hb)
    if override_regime:
        regime = override_regime
    elif emotion == 'WEAK' or idx_down:
        regime = 'ICE'
    elif hybrid_ok:
        regime = 'HYBRID_GUARDED'
    elif rebound_ok and relay_bad:
        regime = 'TREND_REBOUND'
    elif idx_up and (relay_bad or raw_count < STRATEGY_CONFIG.get('emotion_hot_zt_threshold', 130)):
        regime = 'TREND_INDEX'
    elif relay_bad:
        regime = 'RELAY_WEAK'
    elif rebound_ok and raw_count < STRATEGY_CONFIG.get('emotion_hot_zt_threshold', 130):
        regime = 'TREND_REBOUND'
    elif emotion == 'HOT' or relay_good:
        regime = 'RELAY_HOT'
    else:
        regime = 'RELAY_OK'

    g.v11_market_regime = regime
    g.v11_index_ret5 = idx.get('ret5', 0.0)
    g.v11_index_ret20 = idx.get('ret20', 0.0)
    g.v11_recent_relay_avg = relay.get('avg', 0.0)
    g.v11_recent_relay_win_rate = relay.get('win_rate', 0.5)
    g.v18_hot_breadth = hb
    g.v18_emotion_temperature = float(hb.get('temperature', 0) or 0)
    g.disable_top1_tick_today = False

    if regime == 'ICE':
        g.allow_new_entries_today = False
        g.max_score_buys_today = 0
        g.dynamic_score_buy_min_entry_index = 999
        g.dynamic_score_buy_min_return = 999
        g.disable_top1_tick_today = True
    elif regime == 'MAIN_UPTREND':
        # v19：主升确认后才允许Top1，且只保留一个核心低吸名额。
        g.disable_top1_tick_today = False
        g.allow_new_entries_today = True
        g.max_score_buys_today = int(STRATEGY_CONFIG.get('v19_main_max_score_buys', 1) or 1)
        g.dynamic_score_buy_min_entry_index = max(float(getattr(g, 'dynamic_score_buy_min_entry_index', 50) or 50), float(STRATEGY_CONFIG.get('v19_rebound_min_entry_index', 63) or 63))
        g.dynamic_score_buy_min_return = min(float(getattr(g, 'dynamic_score_buy_min_return', 0.006) or 0.006), float(STRATEGY_CONFIG.get('v19_rebound_min_0931_ret', -0.018) or -0.018))
    elif regime == 'REBOUND_RELAY':
        # v19：冰点修复只低吸，不打板；避免v18在修复初期追高被反复洗。
        g.disable_top1_tick_today = bool(STRATEGY_CONFIG.get('v19_rebound_disable_top1', True))
        g.allow_new_entries_today = True
        g.max_score_buys_today = int(STRATEGY_CONFIG.get('v19_rebound_max_score_buys', 1) or 1)
        g.dynamic_score_buy_min_entry_index = max(float(getattr(g, 'dynamic_score_buy_min_entry_index', 50) or 50), float(STRATEGY_CONFIG.get('v19_rebound_min_entry_index', 63) or 63))
        g.dynamic_score_buy_min_return = min(float(getattr(g, 'dynamic_score_buy_min_return', 0.006) or 0.006), float(STRATEGY_CONFIG.get('v19_rebound_min_0931_ret', -0.018) or -0.018))
    elif regime == 'TREND_INDEX':
        # 指数趋势市：禁打板，少量做强趋势低吸；防止“基准涨、策略跌”。
        if STRATEGY_CONFIG.get('v11_trend_disable_top1_tick', True):
            g.disable_top1_tick_today = True
        g.max_score_buys_today = min(int(getattr(g, 'max_score_buys_today', 1) or 1), int(STRATEGY_CONFIG.get('v11_trend_max_score_buys', 1) or 1))
        g.dynamic_score_buy_min_entry_index = max(float(getattr(g, 'dynamic_score_buy_min_entry_index', 50) or 50), float(STRATEGY_CONFIG.get('v11_trend_min_entry_index', 60) or 60))
        g.dynamic_score_buy_min_return = min(float(getattr(g, 'dynamic_score_buy_min_return', 0.006) or 0.006), float(STRATEGY_CONFIG.get('v14_trend_min_0931_ret', -0.006) or -0.006))
        g.allow_new_entries_today = True
    elif regime == 'HYBRID_GUARDED':
        # 指数趋势 + 短线极热共振：允许一个Top1试探，同时保留最多一个SCORE_BUY低吸。
        # 仓位降低，避免v12多tick打板放大噪音。
        g.disable_top1_tick_today = False
        g.max_score_buys_today = min(int(getattr(g, 'max_score_buys_today', 1) or 1), int(STRATEGY_CONFIG.get('v14_hybrid_max_score_buys', 1) or 1))
        g.dynamic_score_buy_min_entry_index = max(float(getattr(g, 'dynamic_score_buy_min_entry_index', 50) or 50), 55.0)
        g.dynamic_score_buy_min_return = min(float(getattr(g, 'dynamic_score_buy_min_return', 0.008) or 0.008), float(STRATEGY_CONFIG.get('v14_trend_min_0931_ret', -0.006) or -0.006))
        g.allow_new_entries_today = True
    elif regime == 'TREND_REBOUND':
        # v15：上涨后半段/修复段。禁打板，只允许一个高质量核心趋势低吸，防止继续空仓。
        g.disable_top1_tick_today = True
        g.max_score_buys_today = min(int(getattr(g, 'max_score_buys_today', 1) or 1), int(STRATEGY_CONFIG.get('v15_rebound_max_score_buys', 1) or 1))
        g.dynamic_score_buy_min_entry_index = max(float(getattr(g, 'dynamic_score_buy_min_entry_index', 50) or 50), float(STRATEGY_CONFIG.get('v15_rebound_min_entry_index', 52) or 52))
        g.dynamic_score_buy_min_return = min(float(getattr(g, 'dynamic_score_buy_min_return', 0.006) or 0.006), float(STRATEGY_CONFIG.get('v15_rebound_min_0931_ret', -0.004) or -0.004))
        g.allow_new_entries_today = True
    elif regime == 'RELAY_WEAK':
        if STRATEGY_CONFIG.get('v11_relay_weak_no_new_buy', True):
            g.allow_new_entries_today = False
            g.max_score_buys_today = 0
            g.disable_top1_tick_today = True
    # RELAY_OK / RELAY_HOT 使用原来的情绪参数

    log.info(
        f"[v11市场状态] regime={regime} | emotion={emotion} | rawZT={raw_count} | "
        f"idx5={idx.get('ret5', 0):.2%} idx20={idx.get('ret20', 0):.2%} | "
        f"recentRelay n={relay.get('n',0)} avg={relay.get('avg',0):.2%} win={relay.get('win_rate',0.5):.1%} | "
        f"hotCore={hb.get('core',0)} hotCont={hb.get('hot_continue',0)} topHot={hb.get('top_hot',0):.1f} topEntry={hb.get('top_entry',0):.1f} temp={hb.get('temperature',0):.1f} | "
        f"disableTop1={getattr(g,'disable_top1_tick_today',False)} maxScoreBuy={getattr(g,'max_score_buys_today',0)}"
    )


def _update_market_emotion_state(context) -> None:
    """
    v8：根据“原始全市场昨日涨停数”定义市场情绪。
    注意：这里只控制“新增买入”，不影响已有持仓的风控卖出。
    """
    raw_count = int(getattr(g, 'zt_count_raw', 0) or 0)
    if raw_count < STRATEGY_CONFIG.get('emotion_stop_zt_threshold', 60):
        state = 'WEAK'
        max_score_buys = 0
        min_entry_index = 999
        min_return = 999
        allow_new_entries = False
    elif raw_count < STRATEGY_CONFIG.get('emotion_caution_zt_threshold', 90):
        state = 'CAUTION'
        max_score_buys = STRATEGY_CONFIG.get('emotion_caution_max_score_buys', 1)
        min_entry_index = STRATEGY_CONFIG.get('emotion_caution_min_entry_index', 55)
        min_return = STRATEGY_CONFIG.get('emotion_caution_min_return', 0.02)
        allow_new_entries = True
    elif raw_count < STRATEGY_CONFIG.get('emotion_hot_zt_threshold', 130):
        state = 'NORMAL'
        max_score_buys = STRATEGY_CONFIG.get('emotion_normal_max_score_buys', 2)
        min_entry_index = STRATEGY_CONFIG.get('emotion_normal_min_entry_index', 52)
        min_return = STRATEGY_CONFIG.get('emotion_normal_min_return', 0.01)
        allow_new_entries = True
    else:
        state = 'HOT'
        max_score_buys = STRATEGY_CONFIG.get('emotion_hot_max_score_buys', 3)
        min_entry_index = STRATEGY_CONFIG.get('emotion_hot_min_entry_index', 45)
        min_return = STRATEGY_CONFIG.get('emotion_hot_min_return', 0.01)
        allow_new_entries = True

    g.market_emotion_state = state
    g.max_score_buys_today = max_score_buys
    g.dynamic_score_buy_min_entry_index = min_entry_index
    g.dynamic_score_buy_min_return = min_return
    g.allow_new_entries_today = allow_new_entries
    log.info(f"[emotion_v64] 原始ZT={raw_count} | 状态={state} | 新增买入={allow_new_entries} | "
             f"Top2~5最多{max_score_buys}只 | entry_index阈值={min_entry_index} | 09:31涨幅阈值={min_return:.2%}")


def _log_trade_stats_v63(context) -> None:
    """v6.3：盘后输出自定义交易统计，直接挂到log_daily_summary里，保证一定能看到。"""
    sells = [r for r in getattr(g, 'trade_records', []) if r.get('side') == 'SELL' and 'pnl_pct' in r]
    buys = [r for r in getattr(g, 'trade_records', []) if r.get('side') == 'BUY']
    if not sells:
        log.info(f"📊 v8交易统计: 买入{len(buys)}笔，暂无已平仓交易，胜率/盈亏比待形成闭环")
        return
    wins = [r for r in sells if float(r.get('pnl_pct', 0) or 0) > 0]
    losses = [r for r in sells if float(r.get('pnl_pct', 0) or 0) <= 0]
    win_rate = len(wins) / len(sells) if sells else 0.0
    avg_win = np.mean([float(r['pnl_pct']) for r in wins]) if wins else 0.0
    avg_loss = abs(np.mean([float(r['pnl_pct']) for r in losses])) if losses else 0.0
    avg_pnl = np.mean([float(r['pnl_pct']) for r in sells]) if sells else 0.0
    gross_profit = sum([float(r.get('pnl', 0) or 0) for r in wins])
    gross_loss = abs(sum([float(r.get('pnl', 0) or 0) for r in losses]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    recent = sells[-20:]
    recent_win_rate = len([r for r in recent if float(r.get('pnl_pct', 0) or 0) > 0]) / len(recent) if recent else 0.0
    log.info("📊 ===== v8 自定义交易统计 =====")
    log.info(f"📊 累计买入{len(buys)}笔 | 平仓{len(sells)}笔 | 未平仓{len(getattr(g, 'open_trades', {}))}笔")
    log.info(f"📊 胜率{win_rate:.1%} | 最近20笔胜率{recent_win_rate:.1%} | 平均收益{avg_pnl:.2%}")
    log.info(f"📊 平均盈利{avg_win:.2%} | 平均亏损{avg_loss:.2%} | ProfitFactor={profit_factor if not pd.isna(profit_factor) else 'NA'}")
    # v7 自动诊断：用于快速判断策略是否假运行/过度交易/交易质量下降。
    try:
        all_signals = len(getattr(g, 'entry_signals', {}) or {})
        open_n = len(getattr(g, 'open_trades', {}) or {})
        if len(sells) >= 10:
            if win_rate < 0.45:
                log.info("🧠 [v8诊断] 胜率偏低：建议检查09:31买入质量过滤或降低弱市交易数量")
            if not pd.isna(profit_factor) and profit_factor < 1.3:
                log.info("🧠 [v8诊断] 盈亏比偏低：盈利保护/止盈延展可能不足")
        log.info(f"🧠 [v8诊断] 信号{all_signals} | 累计BUY{len(buys)} | SELL{len(sells)} | 未平仓{open_n}")
        _v8_log_hot_memory_summary(context)
    except Exception as e:
        log.info(f"🧠 [v8诊断异常] {e}")

def initialize(context):
    """
    策略初始化，仅在回测/实盘开始时调用一次。
    """
    # 设置策略参数
    set_option('use_real_price', True)          # 使用真实价格交易
    set_option('order_volume_ratio', 1)          # 无成交量限制
    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))  # 佣金
    # default slippage
    # set_slippage(FixedSlippage(0.02))            # 滑点
    
    set_slippage(FixedSlippage(3/1000))
    # 设置交易成本（股票万2.5，卖出印花税0.1%）
    set_order_cost(
        OrderCost(
            open_tax=0,                # 买入印花税
            close_tax=0.001,           # 卖出印花税
            open_commission=2.5/10000, # 买入佣金
            close_commission=2.5/10000,# 卖出佣金
            close_today_commission=0,  # 平今佣金（股票无）
            min_commission=5           # 最低佣金
        ),
        type='stock'
    )
    
    # 初始化全局状态
    g.stock_pool = pd.DataFrame(columns=[
        'jq_code', 'code', 'name', 'zt_date', 'zt_close', 'pct_change',
        'total_score', 'classification', 'signal', 'entry_index',
        'buy_price', 'stop_loss', 'target_price'
    ])
    g.holdings = {}                # 持仓 dict
    g.entry_signals = {}           # 当日建仓信号
    g.zt_count_yesterday = 0       # 过滤后昨日ZT数（可交易池口径）
    g.zt_count_raw = 0             # 全市场原始昨日ZT数（市场情绪口径）
    g.zt_count_tradeable = 0       # 过滤ST/次新后的昨日ZT数
    g.daily_log = []               # 日志
    g.trailing_stops = {}          # 移动止盈线
    g.ma_cache = {}                # MA5/MA10缓存
    g.auction_prices = {}          # TYPE_C集合竞价价格缓存
    g.auction_captured_today = False  # 今日是否已捕获集合竞价
    g.crash_checked_today = False  # 今日是否已检查崩盘
    g.crash_detected_today = False # 今日是否检测到崩盘
    g.trade_enabled_today = True   # 今日是否允许交易
    g.top1_tick_code = None        # 今日Top1 tick打板标的
    g.top1_tick_limit = None       # 今日Top1涨停价
    g.bought_today = set()         # 今日已成功买入/已放弃股票，防止重复下单
    g.top1_order_attempted = set() # Top1已提交过打板单的股票；避免tick连续刷单
    g.score_buy_executed_today = False  # Top2~5 09:31固定买入是否已执行
    g.trade_records = []            # v6.2 完整交易流水：BUY/SELL，用于自定义胜率统计
    g.open_trades = {}              # v6.2 当前未平仓交易，code -> BUY记录
    # v8：热点记忆池。跨交易日保存最近10日TOP1/SCORE_BUY/成交股，并参与次日重新评分。
    g.hot_memory_pool = {}
    g.hot_memory_last_update_date = None
    g.hot_memory_today_added = set()
    g.pending_sells = set()         # v6.2 已提交卖出但等待成交/同步的股票
    # v6.3 市场情绪状态：用于控制新增买入，避免弱市过山车
    g.market_emotion_state = 'UNKNOWN'
    g.allow_new_entries_today = True
    g.max_score_buys_today = STRATEGY_CONFIG.get('emotion_normal_max_score_buys', 2)
    g.dynamic_score_buy_min_entry_index = STRATEGY_CONFIG.get('emotion_normal_min_entry_index', 52)
    g.dynamic_score_buy_min_return = STRATEGY_CONFIG.get('emotion_normal_min_return', 0.01)

    # v6.1：Top2~5 不依赖 every_bar/handle_data，固定在09:31做一次实时过滤后买入。
    try:
        run_daily(buy_score_candidates, time=STRATEGY_CONFIG.get('score_buy_time', '09:31'))
    except Exception as e:
        log.info(f"[initialize] 注册buy_score_candidates失败: {e}")

    # v6.2：tick回测中 every_bar/handle_data 不稳定，风控必须用固定时间点调度。
    # 这会稳定触发止损、移动止盈、最大持仓天数退出，并形成交易闭环。
    for _t in STRATEGY_CONFIG.get('risk_check_times', ['09:35','09:44','10:00', '10:15', '10:30', '11:00', '11:25', '13:08', '13:39', '14:00', '14:30', '14:50']):
        try:
            run_daily(risk_management_v62, time=_t)
        except Exception as e:
            log.info(f"[initialize] 注册risk_management_v62({_t})失败: {e}")

    log.info("[initialize] 涨停板交易策略 v19 初始化完成")
    log.info(f"[initialize] 最大持仓: {STRATEGY_CONFIG['max_holdings']}, "
             f"每日最大建仓: {STRATEGY_CONFIG['max_entry_count']}, "
             f"ZT阈值: {STRATEGY_CONFIG['zt_count_threshold']}")
    # 设置日志级别
    #log.set_level('order', 'error')   # 订单日志只报错
    #log.set_level('system', 'error')  # 系统日志只报错
    #log.set_level('strategy', 'warning') # 策略日志显示debug信息


# ============================================================================
# Section 10.5: v8 Hot Rotation Engine - 热点记忆池与生命周期
# ============================================================================

def _v8_get_today_date(context):
    try:
        return context.current_dt.date()
    except Exception:
        try:
            return context.previous_date
        except Exception:
            return datetime.now().date()


def _v8_ensure_hot_memory():
    if not hasattr(g, 'hot_memory_pool') or g.hot_memory_pool is None:
        g.hot_memory_pool = {}
    if not hasattr(g, 'hot_memory_today_added') or g.hot_memory_today_added is None:
        g.hot_memory_today_added = set()


def _v8_update_state(item: Dict) -> str:
    score = float(item.get('hot_score', 0) or 0)
    if score >= STRATEGY_CONFIG.get('hot_memory_state_core', 95):
        state = 'CORE_LEADER'
    elif score >= STRATEGY_CONFIG.get('hot_memory_state_continue', 65):
        state = 'HOT_CONTINUE'
    elif score >= STRATEGY_CONFIG.get('hot_memory_state_trend', 35):
        state = 'TREND'
    elif score >= STRATEGY_CONFIG.get('hot_memory_min_score', 8):
        state = 'WEAKENING'
    else:
        state = 'DEAD'
    item['state'] = state
    return state


def _v8_decay_hot_memory(context):
    """每日只衰减一次热点记忆池，避免热点一天失忆。"""
    _v8_ensure_hot_memory()
    today = _v8_get_today_date(context)
    if getattr(g, 'hot_memory_last_update_date', None) == today:
        return
    decay = float(STRATEGY_CONFIG.get('hot_memory_decay', 0.92) or 0.92)
    max_days = int(STRATEGY_CONFIG.get('hot_memory_days', 10) or 10)
    min_score = float(STRATEGY_CONFIG.get('hot_memory_min_score', 8) or 8)
    for code in list(g.hot_memory_pool.keys()):
        item = g.hot_memory_pool.get(code, {})
        item['hot_score'] = float(item.get('hot_score', 0) or 0) * decay
        item['days_alive'] = int(item.get('days_alive', 0) or 0) + 1
        item['last_seen_days'] = int(item.get('last_seen_days', 0) or 0) + 1
        _v8_update_state(item)
        if item.get('days_alive', 0) > max_days or float(item.get('hot_score', 0) or 0) < min_score:
            del g.hot_memory_pool[code]
    g.hot_memory_last_update_date = today
    g.hot_memory_today_added = set()


def _v8_touch_hot_memory(context, code: str, row: Optional[Dict] = None, source: str = 'UNKNOWN', bonus: float = 0.0):
    """把股票写入/更新热点记忆池。只使用已知数据，不读取未来。"""
    _v8_ensure_hot_memory()
    if not code or not isinstance(code, str):
        return
    today = _v8_get_today_date(context)
    row = row or {}
    try:
        total_score = float(row.get('total_score', 0) or 0)
    except Exception:
        total_score = 0.0
    try:
        entry_index = float(row.get('entry_index', 0) or 0)
    except Exception:
        entry_index = 0.0
    add = float(bonus or 0)
    add += total_score * float(STRATEGY_CONFIG.get('hot_memory_total_score_weight', 0.18) or 0.18)
    add += entry_index * float(STRATEGY_CONFIG.get('hot_memory_entry_bonus_weight', 0.35) or 0.35)
    if code not in g.hot_memory_pool:
        g.hot_memory_pool[code] = {
            'jq_code': code,
            'first_seen_date': today,
            'last_seen_date': today,
            'days_alive': 0,
            'last_seen_days': 0,
            'seen_count': 0,
            'signal_count': 0,
            'buy_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'hot_score': 0.0,
            'state': 'NEW_HOT',
            'source': source,
            'last_total_score': 0.0,
            'last_entry_index': 0.0,
            'last_classification': '',
            'last_signal': '',
        }
    item = g.hot_memory_pool[code]
    item['last_seen_date'] = today
    item['last_seen_days'] = 0
    item['seen_count'] = int(item.get('seen_count', 0) or 0) + 1
    item['source'] = source
    item['hot_score'] = float(item.get('hot_score', 0) or 0) + add
    item['last_total_score'] = total_score
    item['last_entry_index'] = entry_index
    if row.get('classification') is not None:
        item['last_classification'] = row.get('classification', '')
    if row.get('signal') is not None:
        item['last_signal'] = row.get('signal', '')
    _v8_update_state(item)


def _v8_update_hot_memory_from_predict(context, predict_df: pd.DataFrame):
    """每天把评分靠前的候选持续写入热点池。"""
    if predict_df is None or predict_df.empty or 'jq_code' not in predict_df.columns:
        return
    df = predict_df.copy()
    for col in ['entry_index', 'total_score']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    keep_n = int(STRATEGY_CONFIG.get('hot_memory_keep_top_n_daily', 20) or 20)
    df = df.sort_values(['entry_index', 'total_score'], ascending=False).head(keep_n)
    for _, row in df.iterrows():
        _v8_touch_hot_memory(context, row.get('jq_code', ''), row.to_dict(), source='DAILY_TOP', bonus=5)


def _v8_update_hot_memory_from_signals(context, signals: Dict):
    """把当日TOP1_TICK/SCORE_BUY目标标的写入热点池。"""
    if not signals:
        return
    for code, sig in signals.items():
        b = STRATEGY_CONFIG.get('hot_memory_signal_bonus', 25)
        if sig.get('entry_type') == 'TOP1_TICK':
            b += 5
        _v8_touch_hot_memory(context, code, sig, source=sig.get('entry_type', 'SIGNAL'), bonus=b)
        try:
            g.hot_memory_pool[code]['signal_count'] = int(g.hot_memory_pool[code].get('signal_count', 0) or 0) + 1
        except Exception:
            pass


def _v8_update_hot_memory_from_trade_record(context, rec: Dict):
    """在买卖成交记录后更新热点池。"""
    try:
        code = rec.get('code') or rec.get('stock')
        side = rec.get('side') or rec.get('action')
        if not code or not side:
            return
        if side == 'BUY':
            _v8_touch_hot_memory(context, code, rec, source='BUY', bonus=STRATEGY_CONFIG.get('hot_memory_buy_bonus', 30))
            g.hot_memory_pool[code]['buy_count'] = int(g.hot_memory_pool[code].get('buy_count', 0) or 0) + 1
        elif side == 'SELL':
            pnl_pct = float(rec.get('pnl_pct', rec.get('pnl', 0)) or 0)
            if pnl_pct > 0:
                _v8_touch_hot_memory(context, code, rec, source='SELL_WIN', bonus=STRATEGY_CONFIG.get('hot_memory_sell_win_bonus', 20))
                g.hot_memory_pool[code]['win_count'] = int(g.hot_memory_pool[code].get('win_count', 0) or 0) + 1
            else:
                _v8_touch_hot_memory(context, code, rec, source='SELL_LOSS', bonus=-STRATEGY_CONFIG.get('hot_memory_sell_loss_penalty', 18))
                g.hot_memory_pool[code]['loss_count'] = int(g.hot_memory_pool[code].get('loss_count', 0) or 0) + 1
    except Exception as e:
        try:
            log.info(f"[v8热点交易更新异常] {e}")
        except Exception:
            pass


def _v8_build_analysis_pool(context, current_pool: pd.DataFrame) -> pd.DataFrame:
    """
    合并“昨日涨停池 + 10日热点记忆池”。
    热点记忆池里的股票会重新走 build_stock_data/calc_factors/score/predict，保证连续对待。
    """
    _v8_ensure_hot_memory()
    base = current_pool.copy() if current_pool is not None else pd.DataFrame()
    rows = []
    existing = set(base['jq_code'].tolist()) if (not base.empty and 'jq_code' in base.columns) else set()
    today = _v8_get_today_date(context)
    for code, item in g.hot_memory_pool.items():
        if code in existing:
            continue
        if item.get('state') == 'DEAD':
            continue
        # 记忆池候选需要基本字段才能进入原分析管线。
        rows.append({
            'jq_code': code,
            'code': code.split('.')[0],
            'name': '',
            'zt_date': item.get('last_seen_date', context.previous_date),
            'zt_close': np.nan,
            'pct_change': np.nan,
            'v8_memory_only': True,
            'v8_hot_score': float(item.get('hot_score', 0) or 0),
            'v8_hot_state': item.get('state', 'WEAKENING'),
            'v8_days_alive': int(item.get('days_alive', 0) or 0),
        })
    if rows:
        base = pd.concat([base, pd.DataFrame(rows)], ignore_index=True, sort=False)
        log.info(f"[v8热点池] 合并历史热点候选 {len(rows)} 只，分析池合计 {len(base)} 只")
    else:
        log.info(f"[v8热点池] 无额外历史热点候选，分析池 {len(base)} 只")
    return base.drop_duplicates('jq_code', keep='first') if not base.empty and 'jq_code' in base.columns else base


def _v8_apply_hot_memory_scores(context, predict_df: pd.DataFrame) -> pd.DataFrame:
    """把热点记忆加权到 total_score / entry_index 中，但仍保留原评分作为基础。"""
    if predict_df is None or predict_df.empty or 'jq_code' not in predict_df.columns:
        return predict_df
    _v8_ensure_hot_memory()
    df = predict_df.copy()
    for col in ['total_score', 'entry_index']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    mem_w = float(STRATEGY_CONFIG.get('v8_hot_score_weight_memory', 0.20) or 0.20)
    entry_w = float(STRATEGY_CONFIG.get('v8_hot_score_weight_entry_index', 0.22) or 0.22)
    total_w = float(STRATEGY_CONFIG.get('v8_hot_score_weight_total_score', 0.18) or 0.18)
    states = []
    hot_scores = []
    for code in df['jq_code'].astype(str).tolist():
        item = g.hot_memory_pool.get(code, {})
        hs = float(item.get('hot_score', 0) or 0)
        state = item.get('state', '')
        hot_scores.append(hs)
        states.append(state)
    df['v8_hot_score'] = hot_scores
    df['v8_hot_state'] = states
    state_bonus = df['v8_hot_state'].map({'CORE_LEADER': 12, 'HOT_CONTINUE': 8, 'TREND': 4, 'WEAKENING': -2}).fillna(0)
    hot_component = np.log1p(df['v8_hot_score'].clip(lower=0)) * 6
    df['v8_total_score'] = df['total_score'] + hot_component * mem_w + state_bonus
    df['v8_entry_index'] = df['entry_index'] + hot_component * entry_w + state_bonus * 0.6
    df['total_score_raw'] = df['total_score']
    df['entry_index_raw'] = df['entry_index']
    df['total_score'] = df['v8_total_score'].round(2)
    df['entry_index'] = df['v8_entry_index'].round(2)
    log.info(f"[v8热点评分] 已对 {len(df)} 只股票叠加热点记忆评分")
    return df


def _v8_allow_low_open_buy(code: str, sig: Dict, price: float, open_price: float, ret: float) -> bool:
    """热点记忆股允许低开/平开后继续观察，不再09:31一刀切。"""
    try:
        item = getattr(g, 'hot_memory_pool', {}).get(code, {})
        state = item.get('state', '')
        allow_states = STRATEGY_CONFIG.get('v8_allow_low_open_memory_states', ['CORE_LEADER','HOT_CONTINUE','TREND'])
        if state not in allow_states:
            return False
        open_strength = price / open_price - 1.0 if open_price > 0 else -9
        return ret >= STRATEGY_CONFIG.get('v8_low_open_min_ret', -0.035) and open_strength >= STRATEGY_CONFIG.get('v8_low_open_min_strength', -0.018)
    except Exception:
        return False


def _v8_log_hot_memory_summary(context):
    try:
        _v8_ensure_hot_memory()
        if not g.hot_memory_pool:
            log.info('[v8热点池] 当前为空')
            return
        items = sorted(g.hot_memory_pool.items(), key=lambda kv: float(kv[1].get('hot_score', 0) or 0), reverse=True)
        log.info(f"[v8热点池] 存量 {len(items)} 只，Top10:")
        for code, item in items[:10]:
            log.info(f"   {code} state={item.get('state')} hot={float(item.get('hot_score',0) or 0):.1f} seen={item.get('seen_count',0)} buy={item.get('buy_count',0)} win/loss={item.get('win_count',0)}/{item.get('loss_count',0)}")
    except Exception as e:
        log.info(f"[v8热点池日志异常] {e}")

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
    g._log_throttle_cache = {}  # 清空日志节流缓存
    g.entry_signals = {}
    g.auction_prices = {}           # 重置集合竞价价格缓存
    g.auction_captured_today = False  # 重置竞价捕获标志
    g.top1_tick_code = None
    g.top1_tick_limit = None
    g.bought_today = set()
    g.top1_order_attempted = set()
    g.score_buy_executed_today = False
    # v8：每日盘前先对热点记忆池做一次自然衰减。
    try:
        _v8_decay_hot_memory(context)
    except Exception as e:
        log.info(f"[v8热点池] 衰减异常: {e}")
    try:
        unsubscribe_all()
    except Exception:
        pass

    # Step 1: 获取昨日涨停股（原始全市场口径，用于市场情绪判断）
    raw_zt_df = get_yesterday_zt_stocks(context)
    g.zt_count_raw = len(raw_zt_df) if raw_zt_df is not None else 0

    # Step 2: 过滤ST和次新股（可交易池口径，用于建池，不用于市场情绪开关）
    new_zt_df = raw_zt_df.copy() if raw_zt_df is not None else pd.DataFrame()
    if not new_zt_df.empty:
        zt_codes = new_zt_df['jq_code'].tolist()
        filtered_codes = filter_stocks(context, zt_codes)
        new_zt_df = new_zt_df[new_zt_df['jq_code'].isin(filtered_codes)].copy()
    g.zt_count_tradeable = len(new_zt_df)
    g.zt_count_yesterday = g.zt_count_tradeable  # 兼容旧日志字段
    log.info(f"[before_trading_start] 昨日ZT原始数: {g.zt_count_raw} 只 | 过滤后可交易ZT: {g.zt_count_tradeable} 只")

    # Step 3: 判断是否允许交易：必须使用原始全市场ZT数，避免被ST/次新过滤后误判弱市
    # v6.3：在原有30只硬阈值基础上，增加更严格的情绪过滤与动态买入门槛。
    _update_market_emotion_state(context)
    _update_market_regime_v11(context)
    if g.zt_count_raw <= STRATEGY_CONFIG['zt_count_threshold']:
        g.trade_enabled_today = False
        log.info(f"[before_trading_start] ⚠️ 原始昨日ZT数 {g.zt_count_raw} ≤ {STRATEGY_CONFIG['zt_count_threshold']}，今日不交易")
    elif STRATEGY_CONFIG.get('enable_emotion_filter', True) and not getattr(g, 'allow_new_entries_today', True):
        g.trade_enabled_today = False
        log.info(f"[before_trading_start] ⚠️ v8情绪过滤：原始昨日ZT数 {g.zt_count_raw} < {STRATEGY_CONFIG.get('emotion_stop_zt_threshold', 60)}，今日不新增买入，只做持仓风控")
    else:
        g.trade_enabled_today = True

    # Step 4: 更新股票池（使用过滤后的可交易ZT池）
    update_stock_pool(context, new_zt_df)

    # Step 6: 淘汰过期/低分股票
    prune_stock_pool(context)

    # Step 7-9: 为池中股票构建数据并运行分析管线
    pool = _v8_build_analysis_pool(context, g.stock_pool)
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

                # v8：对“昨日涨停池 + 10日热点记忆池”统一重新评分，并叠加热点记忆分。
                if predict_df is not None and not predict_df.empty:
                    predict_df = _v8_apply_hot_memory_scores(context, predict_df)
                    _v8_update_hot_memory_from_predict(context, predict_df)

                # 批量更新股票池中的评分/分类/信号 (替代逐行 iterrows+at[])
                if predict_df is not None and not predict_df.empty:
                    _update_cols = ['total_score', 'classification', 'signal',
                                    'entry_index', 'buy_price', 'stop_loss', 'target_price',
                                    'one_two_score', 'one_two_rank_bucket', 'one_two_filter_reason',
                                    'intraday_plan']
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
                    _v8_update_hot_memory_from_signals(context, g.entry_signals)
                    setup_tick_subscriptions(context)
                else:
                    g.entry_signals = {}
                    setup_tick_subscriptions(context)

        except Exception as e:
            log.info(f"[before_trading_start] 分析管线异常: {e}")
            log.info(f"[before_trading_start] 异常堆栈:\n{traceback.format_exc()}")

    # Step 10: 更新MA缓存
    update_ma_cache(context)

    # Step 11: 输出盘前日志
    log.info(f"[before_trading_start] 盘前准备完成 | 池: {len(g.stock_pool)} | "
             f"持仓: {len(g.holdings)} | 信号: {len(g.entry_signals)} | "
             f"交易: {'✅' if g.trade_enabled_today else '❌'}")
    _v8_log_hot_memory_summary(context)


def _get_tick_code(tick):
    """兼容不同JQ tick对象字段：tick.code / tick.security。"""
    return getattr(tick, 'code', None) or getattr(tick, 'security', None)


def _get_pre_close_from_current_data(cd):
    """尽量从 current_data 取昨收；取不到时用涨停价近似反推。"""
    for attr in ['pre_close', 'prev_close', 'previous_close', 'day_pre_close']:
        try:
            v = getattr(cd, attr, None)
            if v and v > 0:
                return float(v)
        except Exception:
            pass
    try:
        # 普通A股大多数为10%涨停；ST/创业板等会不准，但过滤后主要用于兜底。
        if getattr(cd, 'high_limit', None) and cd.high_limit > 0:
            return float(cd.high_limit) / 1.1
    except Exception:
        pass
    return np.nan


def _submit_limit_buy(code: str, shares: int, limit_price: float):
    """
    用涨停价/指定价限价买入。
    JoinQuant 中 LimitOrderStyle 在部分环境可用；若不可用，退化为普通 order()。
    """
    try:
        return order(code, shares, style=LimitOrderStyle(limit_price))
    except Exception:
        try:
            return order(code, shares)
        except Exception as e:
            log.info(f"[_submit_limit_buy] {code} 下单失败: {e}")
            return None



def _recent_loss_cooldown_hit(context, code: str) -> bool:
    """
    v8：亏损冷却。
    如果同一股票近期刚被亏损卖出，短期内不再重复买入，避免在弱势票上连续打脸。
    """
    try:
        days = int(STRATEGY_CONFIG.get('score_buy_loss_cooldown_days', 3) or 0)
        if days <= 0:
            return False
        today = context.current_dt.date()
        for rec in reversed(getattr(g, 'trade_records', [])):
            if rec.get('side') != 'SELL' or rec.get('code') != code:
                continue
            pnl_pct = float(rec.get('pnl_pct', 0) or 0)
            sell_date = rec.get('date')
            if pnl_pct >= 0 or sell_date is None:
                continue
            try:
                delta_days = (today - sell_date).days
            except Exception:
                delta_days = 999
            if 0 <= delta_days <= days:
                log.info(f"[v7冷却跳过] {code} 最近{delta_days}天亏损卖出过，暂停买入")
                return True
            return False
    except Exception:
        return False
    return False


def _score_buy_quality_ok(context, code: str, sig: Dict, price: float, open_price: float, ret: float) -> Tuple[bool, str]:
    """
    v8：Top2~5 09:31买入二次质量过滤。
    只使用当前已知行情，不使用未来数据。
    """
    try:
        entry_index = float(sig.get('entry_index', 0) or 0)
        total_score = float(sig.get('total_score', 0) or 0)
        rank = int(sig.get('rank', 99) or 99)

        if _recent_loss_cooldown_hit(context, code):
            return False, "近期亏损冷却"

        if open_price <= 0 or price <= 0:
            return False, "价格字段异常"

        open_strength = price / open_price - 1.0
        min_open_strength = float(STRATEGY_CONFIG.get('score_buy_min_open_strength', 0.0015) or 0)

        # v14：强热点核心股允许小幅低开/开盘后回落低吸。
        # v13 日志显示，多只CORE_LEADER因 open_strength -0.1%~-0.5% 被过滤，错过后续主升。
        hot_state = str(sig.get('v8_hot_state', '') or '')
        hot_score = float(sig.get('v8_hot_score', 0) or 0)
        is_core_absorb = (
            hot_state in ('CORE_LEADER', 'HOT_CONTINUE')
            and entry_index >= float(STRATEGY_CONFIG.get('v11_trend_min_entry_index', 58) or 58)
            and (total_score >= float(STRATEGY_CONFIG.get('v11_trend_min_total_score', 52) or 52) or hot_score >= 150)
        )
        if is_core_absorb:
            min_open_strength = min(min_open_strength, float(STRATEGY_CONFIG.get('v14_core_min_open_strength', -0.006) or -0.006))

        # v8/v14：普通候选要求不弱于开盘；核心趋势股允许小幅低吸。
        if open_strength < min_open_strength:
            return False, f"低开过弱 open_strength={open_strength:.2%} < {min_open_strength:.2%}" 

        # 如果9:31涨幅偏弱，必须看到明显的开盘后拉升，否则容易买到低开弱反抽。
        weak_reversal_strength = float(STRATEGY_CONFIG.get('score_buy_weak_reversal_open_strength', 0.008) or 0)
        # v8：允许小水下转强
        if ret < -0.03 and open_strength < weak_reversal_strength:
            return False, f"弱势下跌过深 ret={ret:.2%}, open_strength={open_strength:.2%}" 

        # v9：回测日志显示亏损主要来自结构止损/硬止损，说明低质量补位票不能再靠宽松阈值进入。
        low_score_line = float(STRATEGY_CONFIG.get('score_buy_low_quality_total_score', 38) or 38)
        low_quality_entry = float(STRATEGY_CONFIG.get('score_buy_low_quality_entry_index', 56) or 56)
        if total_score < low_score_line:
            return False, f"v9 total_score不足 total_score={total_score:.1f} < {low_score_line:.1f}"
        if entry_index < low_quality_entry and rank >= 4:
            return False, f"v9 Rank靠后且entry_index不足 rank={rank}, entry_index={entry_index:.1f}"
        if rank >= 5:
            return False, f"v9 不再买Top5补位票 rank={rank}"

        return True, "OK"
    except Exception as e:
        return False, f"质量过滤异常: {e}"


def buy_score_candidates(context):
    """
    v6.1：Top2~5 固定时间买入函数。

    不依赖 every_bar/handle_data，也不依赖 tick 订阅；在09:31执行一次，
    对 SCORE_BUY 信号做实时过滤后买入：
    1. entry_index >= 45；
    2. 当前价强于开盘价；
    3. 当前价/昨收在 0.98~1.07；
    4. 当前价未接近涨停，避免普通评分票追板；
    5. 未停牌、未持仓、持仓数未超限。
    """
    if getattr(g, 'score_buy_executed_today', False):
        return
    g.score_buy_executed_today = True

    if not getattr(g, 'trade_enabled_today', True):
        log.info("[SCORE_BUY_0931] 今日交易开关关闭，跳过Top2~5")
        return
    if not getattr(g, 'entry_signals', None):
        log.info("[SCORE_BUY_0931] 无entry_signals，跳过")
        return

    signals = [(code, sig) for code, sig in g.entry_signals.items()
               if sig.get('entry_type') == 'SCORE_BUY']
    if not signals:
        log.info("[SCORE_BUY_0931] 无SCORE_BUY信号")
        return

    # 按rank排序，确保Top2~5顺序执行。
    signals = sorted(signals, key=lambda x: x[1].get('rank', 99))
    cur_data = get_current_data()
    max_holdings = STRATEGY_CONFIG.get('max_holdings', 5)
    # v6.3：根据市场情绪动态控制 Top2~5 买入数量、entry_index 与9:31涨幅阈值。
    max_score_buys = int(getattr(g, 'max_score_buys_today', STRATEGY_CONFIG.get('emotion_normal_max_score_buys', 2)) or 0)
    dyn_min_entry = float(getattr(g, 'dynamic_score_buy_min_entry_index', STRATEGY_CONFIG.get('score_buy_min_entry_index', 45)) or 0)
    dyn_min_ret = float(getattr(g, 'dynamic_score_buy_min_return', STRATEGY_CONFIG.get('score_buy_min_return', 0.01)) or 0)
    bought_count = 0

    if max_score_buys <= 0:
        log.info(f"[SCORE_BUY_0931] v8情绪状态={getattr(g, 'market_emotion_state', 'UNKNOWN')}，Top2~5今日不新增买入")
        return

    for code, sig in signals:
        try:
            if len(context.portfolio.positions) >= max_holdings:
                log.info(f"[SCORE_BUY_0931] 持仓已达上限 {max_holdings}，停止买入")
                break
            if code in getattr(g, 'bought_today', set()):
                continue
            if code in context.portfolio.positions and context.portfolio.positions[code].total_amount > 0:
                continue

            if bought_count >= max_score_buys:
                log.info(f"[SCORE_BUY_0931] v8情绪状态={getattr(g, 'market_emotion_state', 'UNKNOWN')}，已买满Top2~5上限 {max_score_buys} 只")
                break

            entry_index = float(sig.get('entry_index', 0) or 0)
            if entry_index < dyn_min_entry:
                log.info(f"[SCORE_BUY_0931跳过] {code} entry_index={entry_index:.1f} < v8动态阈值{dyn_min_entry:.1f}")
                continue

            cd = cur_data[code]
            if getattr(cd, 'paused', False):
                log.info(f"[SCORE_BUY_0931跳过] {code} 停牌")
                continue

            price = float(getattr(cd, 'last_price', 0) or 0)
            open_price = float(getattr(cd, 'day_open', 0) or 0)
            high_limit = float(getattr(cd, 'high_limit', 0) or 0)
            pre_close = _get_pre_close_from_current_data(cd)

            if price <= 0 or open_price <= 0 or high_limit <= 0 or pd.isna(pre_close) or pre_close <= 0:
                log.info(f"[SCORE_BUY_0931跳过] {code} 行情字段不完整 price={price}, open={open_price}, high_limit={high_limit}, pre_close={pre_close}")
                continue

            ret = price / pre_close - 1.0

            # 2. v8：普通候选仍要求强于开盘；热点记忆股允许低开/平开后转强，不再一刀切。
            if price <= open_price and not _v8_allow_low_open_buy(code, sig, price, open_price, ret):
                log.info(f"[SCORE_BUY_0931跳过] {code} 当前价未强于开盘 price={price:.2f}, open={open_price:.2f}")
                continue

            # v8：二次质量过滤，避免v6.5放宽后买入过多弱质量票。
            quality_ok, quality_reason = _score_buy_quality_ok(context, code, sig, price, open_price, ret)
            if not quality_ok:
                log.info(f"[SCORE_BUY_0931跳过] {code} v8质量过滤: {quality_reason}")
                continue

            if ret < dyn_min_ret:
                log.info(f"[SCORE_BUY_0931跳过] {code} 09:31涨幅不足 ret={ret:.2%} < v8动态阈值{dyn_min_ret:.2%}")
                continue
            max_ret_allowed = STRATEGY_CONFIG.get('score_buy_max_return', 0.07)
            if getattr(g, 'v11_market_regime', '') == 'TREND_INDEX':
                max_ret_allowed = min(float(max_ret_allowed or 0.07), float(STRATEGY_CONFIG.get('v11_trend_max_0931_ret', 0.045) or 0.045))
            elif getattr(g, 'v11_market_regime', '') == 'TREND_REBOUND':
                max_ret_allowed = min(float(max_ret_allowed or 0.07), float(STRATEGY_CONFIG.get('v15_rebound_max_0931_ret', 0.055) or 0.055))
            elif getattr(g, 'v11_market_regime', '') == 'REBOUND_RELAY':
                max_ret_allowed = min(float(max_ret_allowed or 0.07), float(STRATEGY_CONFIG.get('v19_rebound_max_0931_ret', 0.035) or 0.035))
            elif getattr(g, 'v11_market_regime', '') == 'MAIN_UPTREND':
                max_ret_allowed = min(float(max_ret_allowed or 0.07), float(STRATEGY_CONFIG.get('v18_rebound_max_0931_ret', 0.065) or 0.065))
            if ret > max_ret_allowed:
                log.info(f"[SCORE_BUY_0931跳过] {code} 涨幅过高，避免追高 ret={ret:.2%} > {max_ret_allowed:.2%}")
                continue
            if price >= high_limit * STRATEGY_CONFIG.get('score_buy_near_limit_ratio', 0.985):
                log.info(f"[SCORE_BUY_0931跳过] {code} 已接近涨停，普通评分票不追板 price/high_limit={price/high_limit:.3f}")
                continue

            pos_ratio = STRATEGY_CONFIG.get('score_buy_position_ratio', 0.75)
            if getattr(g, 'v11_market_regime', '') == 'TREND_INDEX':
                pos_ratio = STRATEGY_CONFIG.get('v11_trend_score_buy_position_ratio', 0.38)
            elif getattr(g, 'v11_market_regime', '') == 'TREND_REBOUND':
                pos_ratio = STRATEGY_CONFIG.get('v15_rebound_score_buy_position_ratio', 0.58)
            elif getattr(g, 'v11_market_regime', '') == 'REBOUND_RELAY':
                pos_ratio = STRATEGY_CONFIG.get('v19_rebound_score_buy_position_ratio', 0.55)
            elif getattr(g, 'v11_market_regime', '') == 'MAIN_UPTREND':
                pos_ratio = STRATEGY_CONFIG.get('v19_main_score_buy_position_ratio', 0.58)
            shares = calc_position_size(context, code, ratio=pos_ratio)
            if shares <= 0:
                log.info(f"[SCORE_BUY_0931跳过] {code} 仓位不足，shares=0")
                continue

            # 非涨停附近，用普通下单即可；下单后必须检查真实持仓，不再把Order对象当成交。
            order_result = order(code, shares)
            if order_result is None:
                log.info(f"[SCORE_BUY_0931未成交] {code} order返回None")
                continue

            if _check_order_filled(context, code):
                sig['first_leg_done'] = True
                g.bought_today.add(code)
                _record_holding(context, code, sig, shares, leg='full')
                bought_count += 1
                log.info(f"[SCORE_BUY_0931成交] Top{sig.get('rank','?')} {code} {shares}股 | price={price:.2f}, ret={ret:.2%}, entry_index={entry_index:.1f}")
            else:
                log.info(f"[SCORE_BUY_0931提交未确认成交] {code} {shares}股 | price={price:.2f}, entry_index={entry_index:.1f}")
        except Exception as e:
            log.info(f"[SCORE_BUY_0931异常] {code}: {e}")

    log.info(f"[SCORE_BUY_0931] Top2~5执行完成，确认成交 {bought_count} 只")


def handle_tick(context, tick):
    """
    v6.1 tick级执行引擎：只处理 Top1 龙头候选的打板买入。

    Top1采用 tick 打板：tick.current 接近/达到涨停价时，使用涨停价限价单排板。
    重要修复：order对象不等于成交；只有 portfolio 里真实出现持仓，才记录为买入成功。
    """
    if not getattr(g, 'trade_enabled_today', True):
        return
    if getattr(g, 'disable_top1_tick_today', False):
        return
    if not getattr(g, 'entry_signals', None):
        return

    code = _get_tick_code(tick)
    if not code or code != getattr(g, 'top1_tick_code', None):
        return

    signal = g.entry_signals.get(code, {})
    if signal.get('entry_type') != 'TOP1_TICK':
        return

    # 如果此前提交过订单，先检查是否已成交；成交后才记录并取消订阅。
    if code in context.portfolio.positions and context.portfolio.positions[code].total_amount > 0:
        if code not in g.holdings:
            _record_holding(context, code, signal, int(context.portfolio.positions[code].total_amount), leg='full')
        g.bought_today.add(code)
        signal['first_leg_done'] = True
        try:
            unsubscribe(code, 'tick')
        except Exception:
            pass
        log.info(f"[TOP1_TICK成交确认] {code} 已持仓，停止监听")
        return

    now_str = context.current_dt.strftime('%H:%M:%S')
    if now_str < STRATEGY_CONFIG.get('top1_tick_start', '09:30:00') or now_str > STRATEGY_CONFIG.get('top1_tick_end', '10:30:00'):
        return

    # 避免tick连续刷单。若已提交过但没成交，保持监听，但不重复下单。
    if code in getattr(g, 'top1_order_attempted', set()):
        return

    try:
        current_price = float(getattr(tick, 'current', 0) or 0)
        high_limit = getattr(g, 'top1_tick_limit', None)
        if not high_limit or high_limit <= 0:
            cur_data = get_current_data()
            high_limit = cur_data[code].high_limit
        high_limit = float(high_limit or 0)
        if current_price <= 0 or high_limit <= 0:
            return

        trigger_ratio = STRATEGY_CONFIG.get('top1_tick_buy_ratio', 0.997)

        # 破坏性弱势过滤：如果tick已跌破昨收附近较多，放弃Top1打板监听。
        # 用high_limit/1.1估算昨收，仅作为Top1止损式放弃监听阈值。
        if current_price < high_limit / 1.1 * 0.97:
            g.bought_today.add(code)  # 当日不再尝试
            log.info(f"[TOP1_TICK] {code} 走弱，停止监听: current/high_limit={current_price/high_limit:.3f}")
            try:
                unsubscribe(code, 'tick')
            except Exception:
                pass
            return

        if current_price >= high_limit * trigger_ratio:
            
            pos_ratio = STRATEGY_CONFIG.get('top1_position_ratio', 1.5)
            try:
                if signal.get('v14_guarded_hybrid'):
                    pos_ratio = STRATEGY_CONFIG.get('v14_hybrid_top1_position_ratio', 0.60)
            except Exception:
                pass
            shares = calc_position_size(context, code, ratio=pos_ratio)
            if shares <= 0:
                return
            order_result = _submit_limit_buy(code, shares, high_limit)
            g.top1_order_attempted.add(code)
            if order_result is None:
                log.info(f"[TOP1_TICK下单失败] {code} order返回None | tick={current_price:.2f}, high_limit={high_limit:.2f}")
                return

            if _check_order_filled(context, code):
                signal['first_leg_done'] = True
                g.bought_today.add(code)
                _record_holding(context, code, signal, shares, leg='full')
                try:
                    unsubscribe(code, 'tick')
                except Exception:
                    pass
                log.info(f"[TOP1_TICK成交] {code} {shares}股 | tick={current_price:.2f}, high_limit={high_limit:.2f}, entry_index={signal.get('entry_index',0):.1f}")
            else:
                # 排板未成交是正常情况；不要记录买入成功，也不要取消订阅。
                log.info(f"[TOP1_TICK已提交排板未成交] {code} {shares}股 | tick={current_price:.2f}, high_limit={high_limit:.2f}, entry_index={signal.get('entry_index',0):.1f}")
    except Exception as e:
        log.info(f"[TOP1_TICK] {code} tick处理异常: {e}")

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




def _v10_get_hot_item(code: str) -> Dict:
    """读取热点记忆池中的状态，用于识别可利润奔跑的长效核心。"""
    try:
        return getattr(g, 'hot_memory_pool', {}).get(code, {}) or {}
    except Exception:
        return {}


def _v10_runner_profile(code: str, holding: Dict) -> Dict:
    """
    v10：识别“长效利润候选/可持股龙头”。
    只使用买入时信号、当前热点记忆池和已知持仓信息，不使用未来数据。
    """
    item = _v10_get_hot_item(code)
    entry_type = str(holding.get('entry_type', '') or '')
    hot_state = str(holding.get('v8_hot_state', '') or item.get('state', '') or '')
    try:
        hot_score = float(holding.get('v8_hot_score', 0) or item.get('hot_score', 0) or 0)
    except Exception:
        hot_score = 0.0
    try:
        entry_index = float(holding.get('entry_index', 0) or 0)
    except Exception:
        entry_index = 0.0
    try:
        total_score = float(holding.get('total_score', 0) or 0)
    except Exception:
        total_score = 0.0
    try:
        seen_count = int(holding.get('v8_seen_count', 0) or item.get('seen_count', 0) or 0)
    except Exception:
        seen_count = 0

    is_core_state = hot_state in ('CORE_LEADER', 'HOT_CONTINUE')
    is_top1 = entry_type == 'TOP1_TICK'
    is_hot_memory = hot_score >= float(STRATEGY_CONFIG.get('runner_hot_score_min', 120) or 120)
    is_strong_entry = (is_core_state
                       and entry_index >= float(STRATEGY_CONFIG.get('runner_entry_index_min', 62) or 62)
                       and total_score >= float(STRATEGY_CONFIG.get('runner_total_score_min', 45) or 45))
    # v20：TREND不再仅因反复出现就自动runner，避免普通趋势票持有过久拖累收益。
    is_repeated_seen = seen_count >= 4 and hot_state in ('CORE_LEADER', 'HOT_CONTINUE')

    is_runner = bool(is_top1 or is_core_state or is_hot_memory or is_strong_entry or is_repeated_seen)
    return {
        'is_runner': is_runner,
        'entry_type': entry_type,
        'hot_state': hot_state,
        'hot_score': hot_score,
        'entry_index': entry_index,
        'total_score': total_score,
        'seen_count': seen_count,
    }


def _v10_mark_holding_runner(code: str) -> None:
    """买入后立即打上利润奔跑标签，便于日志和风控稳定识别。"""
    try:
        if code not in g.holdings:
            return
        profile = _v10_runner_profile(code, g.holdings[code])
        g.holdings[code]['v10_runner'] = profile['is_runner']
        g.holdings[code]['v10_runner_profile'] = profile
        if profile['is_runner']:
            log.info(f"[v10利润奔跑标记] {code} runner=True | type={profile['entry_type']} state={profile['hot_state']} hot={profile['hot_score']:.1f} entry={profile['entry_index']:.1f} total={profile['total_score']:.1f} seen={profile['seen_count']}")
    except Exception as e:
        log.info(f"[v10利润奔跑标记异常] {code}: {e}")


def _v16_hot_rank(code: str) -> int:
    """返回当前热点记忆池排名；不存在则给大数。"""
    try:
        pool = getattr(g, 'hot_memory_pool', {}) or {}
        if not pool or code not in pool:
            return 999
        items = sorted(pool.items(), key=lambda kv: float(kv[1].get('hot_score', 0) or 0), reverse=True)
        for i, (c, _) in enumerate(items, 1):
            if c == code:
                return i
    except Exception:
        pass
    return 999


def _v16_update_runner_decay_state(code: str, holding: Dict, profile: Dict, today) -> Dict:
    """
    v16：跟踪龙头衰退，不使用未来函数。
    每个交易日只更新一次：hot_score下降、热点状态降级、热点排名掉队。
    """
    info = {
        'rank': _v16_hot_rank(code),
        'decay_days': int(holding.get('v16_hot_decay_days', 0) or 0),
        'is_decaying': False,
        'rank_warning': False,
        'rank_exit': False,
        'reason': ''
    }
    if not STRATEGY_CONFIG.get('v16_enable_runner_decay_exit', True):
        return info
    try:
        today_key = str(today)
        last_update = str(holding.get('v16_last_decay_update', '') or '')
        cur_hot = float(profile.get('hot_score', 0) or 0)
        last_hot = float(holding.get('v16_last_hot_score', cur_hot) or cur_hot)
        cur_state = str(profile.get('hot_state', '') or '')
        last_state = str(holding.get('v16_last_hot_state', cur_state) or cur_state)
        rank = int(info['rank'])
        warn_rank = int(STRATEGY_CONFIG.get('v16_runner_rank_warn_threshold', 5) or 5)
        exit_rank = int(STRATEGY_CONFIG.get('v16_runner_rank_exit_threshold', 8) or 8)
        decay_pct = float(STRATEGY_CONFIG.get('v16_hot_score_decay_pct', 0.12) or 0.12)
        state_order = {'CORE_LEADER': 4, 'HOT_CONTINUE': 3, 'TREND': 2, 'WEAKENING': 1, 'DEAD': 0, '': 0}
        state_down = state_order.get(cur_state, 0) < state_order.get(last_state, 0)
        hot_down = (last_hot > 0 and cur_hot < last_hot * (1.0 - decay_pct))
        rank_bad = rank > exit_rank
        rank_warn = rank > warn_rank
        info['rank_warning'] = bool(rank_warn)
        info['rank_exit'] = bool(rank_bad)
        if last_update != today_key:
            if hot_down or state_down or rank_bad:
                info['decay_days'] = int(info['decay_days']) + 1
                holding['v16_hot_decay_days'] = info['decay_days']
                reasons = []
                if hot_down:
                    reasons.append(f"hot {last_hot:.1f}->{cur_hot:.1f}")
                if state_down:
                    reasons.append(f"state {last_state}->{cur_state}")
                if rank_bad:
                    reasons.append(f"rank {rank}")
                info['reason'] = ', '.join(reasons)
                log.info(f"[v16龙头衰退计数] {code} decay_days={info['decay_days']} | {info['reason']}")
            else:
                # 回到Top队列或hot恢复，衰退计数缓慢清零，避免一次噪音误杀。
                if int(info['decay_days']) > 0 and (cur_state in ('CORE_LEADER','HOT_CONTINUE')) and rank <= warn_rank:
                    info['decay_days'] = max(0, int(info['decay_days']) - 1)
                    holding['v16_hot_decay_days'] = info['decay_days']
            holding['v16_last_hot_score'] = cur_hot
            holding['v16_last_hot_state'] = cur_state
            holding['v16_last_hot_rank'] = rank
            holding['v16_last_decay_update'] = today_key
        info['is_decaying'] = int(info['decay_days']) >= int(STRATEGY_CONFIG.get('v16_hot_decay_days_to_exit', 2) or 2)
    except Exception as e:
        try:
            log.info(f"[v16龙头衰退检测异常] {code}: {e}")
        except Exception:
            pass
    return info


def _v16_runner_dynamic_dd(max_profit_pct: float, decay_info: Dict) -> float:
    """根据曾经浮盈和热点衰退状态，动态收紧runner回撤阈值。"""
    if not STRATEGY_CONFIG.get('v16_dynamic_trail_enabled', True):
        return float(STRATEGY_CONFIG.get('runner_profit_drawdown_pct', 0.08) or 0.08)
    if max_profit_pct >= 0.12:
        dd = float(STRATEGY_CONFIG.get('v16_runner_dd_after_12pct', 0.050) or 0.050)
    elif max_profit_pct >= 0.08:
        dd = float(STRATEGY_CONFIG.get('v16_runner_dd_after_8pct', 0.035) or 0.035)
    elif max_profit_pct >= 0.05:
        dd = float(STRATEGY_CONFIG.get('v16_runner_dd_after_5pct', 0.024) or 0.024)
    elif max_profit_pct >= 0.03:
        dd = float(STRATEGY_CONFIG.get('v16_runner_dd_after_3pct', 0.018) or 0.018)
    else:
        dd = float(STRATEGY_CONFIG.get('runner_profit_drawdown_pct', 0.08) or 0.08)
    # 跌出Top5但尚未彻底衰退，进一步收紧，保护利润。
    try:
        if decay_info.get('rank_warning'):
            dd = min(dd, 0.026)
        if decay_info.get('is_decaying'):
            dd = min(dd, 0.020)
    except Exception:
        pass
    return dd

def risk_management_v62(context):
    """
    v16 固定时间风控入口。
    在v15跟上后半段行情的基础上，增加龙头衰退识别和动态利润保护，
    目标是减少利润回吐，把最大回撤压回更健康区间。
    """
    if not hasattr(g, 'holdings'):
        return
    _sync_holdings_with_portfolio(context)
    if not g.holdings:
        return

    now_str = context.current_dt.strftime('%H:%M')
    log.info(f"[risk_management_v62] {now_str} 开始风控检查，持仓 {len(g.holdings)} 只")

    try:
        cur_data = get_current_data()
    except Exception:
        cur_data = {}

    today = context.current_dt.date()
    for code in list(g.holdings.keys()):
        h = g.holdings.get(code, {})
        buy_date = h.get('buy_date')
        buy_price = float(h.get('buy_price', 0) or 0)
        if buy_price <= 0:
            continue
        if buy_date is not None and buy_date == today:
            # A股T+1，买入当天不能卖。仍更新最高价并打runner标签。
            try:
                p0 = float(cur_data[code].last_price or 0)
                if p0 > 0:
                    h['highest_price'] = max(float(h.get('highest_price', 0) or 0), p0)
                    _v10_mark_holding_runner(code)
            except Exception:
                pass
            continue
        try:
            cd = cur_data[code]
            price = float(cd.last_price or 0)
            high_limit = float(getattr(cd, 'high_limit', 0) or 0)
        except Exception:
            continue
        if price <= 0:
            continue

        highest = max(float(h.get('highest_price', 0) or 0), price)
        h['highest_price'] = highest
        pnl_pct = (price - buy_price) / buy_price
        hold_days = (today - buy_date).days if buy_date is not None else 0
        stop_loss = h.get('stop_loss', np.nan)
        target_price = h.get('target_price', np.nan)

        profile = _v10_runner_profile(code, h)
        is_runner = bool(profile.get('is_runner'))
        h['v10_runner'] = is_runner
        h['v10_runner_profile'] = profile
        decay_info = _v16_update_runner_decay_state(code, h, profile, today) if is_runner else {'rank': 999, 'is_decaying': False, 'rank_warning': False}

        # 1. 止损：runner允许正常波动，但不能无底线扛错。
        try:
            day_open = float(getattr(cd, 'day_open', 0) or 0)
        except Exception:
            day_open = 0

        hard_stop = STRATEGY_CONFIG.get('daily_stop_loss_pct', 0.04)
        if is_runner:
            # 龙头/长效热点给更宽一点空间，但浮亏过大仍退出。
            hard_stop = max(hard_stop, 0.05)

        if STRATEGY_CONFIG.get('structure_stop_enabled', True) and day_open > 0:
            structure_line = STRATEGY_CONFIG.get('structure_stop_min_loss_pct', -0.018)
            if is_runner and hold_days < int(STRATEGY_CONFIG.get('runner_min_hold_days', 3) or 3):
                # T+1/T+2龙头洗盘不因“跌破开盘+小亏”直接卖；只有明显亏损才离场。
                structure_line = min(structure_line, -0.04)
            if price < day_open and pnl_pct <= structure_line:
                _sell_position(context, code, reason=f"v10结构止损({'RUNNER' if is_runner else 'NORMAL'} 跌破开盘且浮亏{pnl_pct:.2%})")
                continue
        if pnl_pct <= -hard_stop:
            _sell_position(context, code, reason=f"v10硬止损({'RUNNER' if is_runner else 'NORMAL'} {pnl_pct:.2%})")
            continue

        # v16：龙头衰退退出。只在已有利润/曾有明显浮盈后触发，避免把正常低位波动当衰退。
        if is_runner and STRATEGY_CONFIG.get('v16_enable_runner_decay_exit', True):
            try:
                max_profit_for_decay = (highest - buy_price) / buy_price if highest > buy_price else 0
                min_hold_decay = int(STRATEGY_CONFIG.get('v16_runner_min_hold_for_decay_exit', 2) or 2)
                min_pnl_decay = float(STRATEGY_CONFIG.get('v16_decay_exit_min_profit', 0.018) or 0.018)
                min_max_decay = float(STRATEGY_CONFIG.get('v16_decay_exit_min_max_profit', 0.035) or 0.035)
                if (hold_days >= min_hold_decay and decay_info.get('is_decaying')
                        and pnl_pct >= min_pnl_decay and max_profit_for_decay >= min_max_decay):
                    _sell_position(context, code, reason=f"v16龙头衰退止盈(rank={decay_info.get('rank')}, decay_days={decay_info.get('decay_days')}, pnl={pnl_pct:.2%}, max={max_profit_for_decay:.2%})")
                    continue
            except Exception:
                pass

        # 2. 高利润开板：普通票可止盈；runner在最小持有期内优先拿住。
        if pnl_pct >= STRATEGY_CONFIG.get('t1_profit_take_pct', 0.09):
            if high_limit > 0 and price >= high_limit * 0.999:
                log.info(f"[risk_management_v62] {code} 盈利{pnl_pct:.2%}且封涨停，继续持有")
            else:
                if is_runner and hold_days < int(STRATEGY_CONFIG.get('runner_min_hold_days', 3) or 3):
                    log.info(f"[v10利润奔跑] {code} RUNNER 高利开板但仍在最小持有期 hold_days={hold_days}, pnl={pnl_pct:.2%}，继续持有")
                elif is_runner and pnl_pct < float(STRATEGY_CONFIG.get('runner_profit_protect_min_pct', 0.12) or 0.12):
                    log.info(f"[v10利润奔跑] {code} RUNNER 高利开板但未达到runner保护线 pnl={pnl_pct:.2%}，继续观察")
                else:
                    _sell_position(context, code, reason=f"v10高利开板止盈({'RUNNER' if is_runner else 'NORMAL'} {pnl_pct:.2%})")
                    continue

        # 3. 目标价止盈：runner禁用固定目标价，避免5%目标价卖飞主升浪。
        if (not is_runner) and (not pd.isna(target_price)) and target_price > buy_price and price >= target_price:
            _sell_position(context, code, reason=f'固定风控-目标价止盈({price:.2f}>={target_price:.2f})')
            continue

        # 4. 利润保护 / 移动止盈：runner使用独立、更宽的回撤阈值。
        if highest > buy_price:
            max_profit_pct = (highest - buy_price) / buy_price
            dd_from_high = (highest - price) / highest if highest > 0 else 0

            if is_runner:
                # v16：runner不再一律给8%大回撤，而是按已获得浮盈动态收紧。
                protect_min = min(float(STRATEGY_CONFIG.get('runner_profit_protect_min_pct', 0.12) or 0.12), 0.03)
                protect_dd = _v16_runner_dynamic_dd(max_profit_pct, decay_info)
                trailing_dd = max(protect_dd + 0.012, float(STRATEGY_CONFIG.get('runner_trailing_stop_pct', 0.10) or 0.10) * 0.55)
                be_after = float(STRATEGY_CONFIG.get('runner_break_even_after_profit_pct', 0.10) or 0.10)
                be_buffer = float(STRATEGY_CONFIG.get('runner_break_even_buffer_pct', -0.015) or -0.015)
            else:
                protect_min = STRATEGY_CONFIG.get('profit_protect_min_pct', 0.04)
                protect_dd = STRATEGY_CONFIG.get('profit_protect_drawdown_pct', 0.025)
                trailing_dd = STRATEGY_CONFIG.get('trailing_stop_pct', 0.06)
                be_after = STRATEGY_CONFIG.get('break_even_after_profit_pct', 0.05)
                be_buffer = STRATEGY_CONFIG.get('break_even_buffer_pct', -0.005)

            if max_profit_pct >= protect_min and dd_from_high >= protect_dd:
                if pnl_pct > 0 or max_profit_pct >= be_after:
                    _sell_position(context, code, reason=f"v10利润保护回撤({'RUNNER' if is_runner else 'NORMAL'} dd={dd_from_high:.2%}, max={max_profit_pct:.2%})")
                    continue
            if max_profit_pct >= be_after and pnl_pct <= be_buffer:
                _sell_position(context, code, reason=f"v10回本保护({'RUNNER' if is_runner else 'NORMAL'} pnl={pnl_pct:.2%}, max={max_profit_pct:.2%})")
                continue
            if max_profit_pct >= protect_min and dd_from_high >= trailing_dd:
                _sell_position(context, code, reason=f"v10移动止盈回撤({'RUNNER' if is_runner else 'NORMAL'} dd={dd_from_high:.2%})")
                continue

        # 5. 最大持仓天数：runner更长，普通票保持原规则。
        max_days = STRATEGY_CONFIG.get('max_hold_days', 5)
        if is_runner:
            max_days = STRATEGY_CONFIG.get('runner_max_hold_days', 8)
        if STRATEGY_CONFIG.get('force_sell_on_max_hold_days', True) and hold_days >= max_days:
            _sell_position(context, code, reason=f"v10最大持仓天数({'RUNNER' if is_runner else 'NORMAL'} {hold_days})")
            continue

    _sync_holdings_with_portfolio(context)

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
    # v6.2：先同步真实持仓，避免日志/风控基于假持仓。
    _sync_holdings_with_portfolio(context)

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

# ============================================================
# v7 核心思想
# ============================================================
# Top2~5 强趋势换手股作为核心利润来源
# 允许低开、水下后转强
# Top1 打板降级为辅助仓位
# 盈利票尽量持有至 T+2
# ============================================================
