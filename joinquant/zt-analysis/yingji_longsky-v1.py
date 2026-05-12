# 克隆自聚宽文章：https://www.joinquant.com/post/63587
# 标题：【实盘数据分享】鹰击长空打板策略
# 作者：实测狂魔~王子玉
#
# 重构说明:
# 1. 所有硬编码参数提取到 CONFIG 字典统一管理，便于调参与回测
# 2. 新增风险控制模块（均可通过 CONFIG['risk_control'] 开关独立控制）:
#    - 移动止盈: 盈利超过阈值后，从最高价回撤时自动卖出
#    - 市场环境过滤: 指数低于均线时降低仓位
#    - 日亏损限制: 当日亏损超过阈值时减仓并禁止新买入
#    - 板块集中度限制: 同行业最多持有N只股票
#    - 分批止盈: 盈利达到目标后先卖出一部分，剩余由移动止盈管理
#    - 时间止损: 持仓超过N天且未达盈利要求时卖出

from jqdata import *         # 聚宽数据模块，提供基础数据API
from jqfactor import *       # 聚宽因子模块，提供因子计算功能
from jqlib.technical_analysis import *  # 聚宽技术分析模块
import datetime as dt        # 日期时间处理
import pandas as pd          # 数据处理和分析
from datetime import datetime
from datetime import timedelta

import newqmt_sql

# ⭐ 在这里设置这个策略的分类标签（写入 trade.fenlei）
newqmt_sql.FENLEI = 'eagles-75-300-3000-test'      

from newqmt_sql import (
    order_zzy as order,
    order_target_zzy as order_target,
    order_value_zzy as order_value,
    order_target_value_zzy as order_target_value
)

# ================================================
# 统一参数配置 (CONFIG)
# ================================================
CONFIG = {
    # ------ 全局参数 ------
    'global': {
        'max_stock_num': 3,           # 最大持仓数量
        'min_shares': 100,            # 最小买入股数
        'cash_ratio_min': 0.3,        # 最低现金比例要求（低于此值不买入）
    },

    # ------ 定时任务时间 ------
    'schedule': {
        'get_stock_list': '9:10',         # 获取选股列表
        'buy': '09:30',                   # 执行买入操作（集合竞价结束后）
        'sell_heavy_turnover': '9:58',   # 高位放量卖出
        'sell_am': '11:25',              # 上午收盘前止盈
        'sell_pm': '13:15',              # 下午收盘前止盈/止损
        'log_position_stats': '14:55',   # 每日持仓统计
        'log_position_mid_stats': '11:25',   # 午时持仓统计
    },

    # ------ 一进二策略参数 (gap_up) ------
    'gap_up': {
        'avg_price_increase_min': 0.07,     # 均价增长最低要求（相对收盘价×1.1）
        'money_min': 5.5e8,                 # 最低成交金额
        'money_max': 20e8,                  # 最高成交金额
        'market_cap_min': 5,               # 最低总市值（亿）
        'circulating_market_cap_max': 75,  # 最高流通市值（亿）
        'auction_vol_ratio_min': 0.03,      # 集合竞价成交量/昨日成交量 最低比例
        'current_ratio_min': 1.0,           # 开盘价/昨日涨停价 下限
        'current_ratio_max': 1.06,          # 开盘价/昨日涨停价 上限
    },

    # ------ 首板低开策略参数 (gap_down) ------
    'gap_down': {
        'relative_position_max': 0.5,   # 60日内相对位置上限
        'watch_days': 60,               # 相对位置观察天数
        'open_pct_min': 0.955,          # 低开幅度下限（低开3%）
        'open_pct_max': 0.97,           # 低开幅度上限（低开4.5%）
        'money_min': 1e8,               # 最低成交金额
    },

    # ------ 弱转强策略参数 (reversal) ------
    'reversal': {
        'increase_ratio_max': 0.20,         # N日内最大涨幅限制
        'increase_days': 4,                 # 涨幅计算天数
        'open_close_ratio_min': -0.05,      # 开收比最低要求（过滤大幅低走）
        'avg_price_increase_min': -0.04,    # 均价增长最低要求
        'money_min': 3e8,                   # 最低成交金额
        'money_max': 19e8,                  # 最高成交金额
        'market_cap_min': 300,               # 最低总市值（亿）
        'circulating_market_cap_max': 3000,  # 最高流通市值（亿）
        'auction_vol_ratio_min': 0.03,      # 集合竞价成交量/昨日成交量 最低比例
        'current_ratio_min': 0.98,          # 开盘价/昨日涨停价 下限
        'current_ratio_max': 1.09,          # 开盘价/昨日涨停价 上限
    },

    # ------ 左压检测参数 (rise_low_volume) ------
    'rise_low_volume': {
        'hist_days': 106,       # 历史数据获取天数
        'high_days': 102,       # 高价数据截取天数
        'volume_ratio': 0.9,    # 成交量比例阈值（低于最大量×此值视为左压）
        'buffer': 5,            # 观察期缓冲天数
    },

    # ------ 卖出参数 ------
    'sell': {
        'profit_multiplier': 1.0,               # 止盈倍数（相对avg_cost）
        'ma_stop_period': 5,                    # 均线止损周期（MA5）
        'heavy_turnover_watch_days': 60,        # 高位放量观察天数
        'heavy_turnover_rp_min': 0.85,          # 高位放量相对位置最低值
        'heavy_turnover_vol_ratio_min': 0.5,    # 高位放量成交量比例最低值
        'heavy_turnover_close_pos_max': 0.65,   # 高位放量收盘位置最高值
    },

    # ------ 过滤参数 ------
    'filter': {
        'new_stock_days': 50,       # 新股过滤天数（一进二/弱转强）
        'new_stock2_days': 250,     # 新股过滤天数（首板低开）
    },

    # ------ 风险控制参数 ------
    'risk_control': {
        'enabled': False,  # 风险控制总开关（False时所有风控措施均不生效）

        # ====== 核心风控（对回撤影响最大）======

        # 硬止损: 单只股票亏损超过阈值时无条件卖出（优先级最高，在所有卖出逻辑之前执行）
        # 涨停板策略个股波动大，MA5止损是滞后指标，硬止损提供绝对保护
        'hard_stop_loss_enabled': True,
        'hard_stop_loss_pct': -0.08,               # 硬止损阈值: -8%（亏损8%无条件卖出）

        # 组合回撤熔断: 当组合总价值从最高点回撤超过阈值时，停止新买入并减仓
        # 针对持续亏损期（如2025/02-04的25%回撤），防止在亏损期继续加仓
        'portfolio_drawdown_enabled': True,
        'portfolio_drawdown_threshold': 0.10,       # 触发阈值: 从峰值回撤10%
        'portfolio_drawdown_reduce_ratio': 0.5,     # 触发后持仓减仓比例
        'portfolio_drawdown_no_buy': True,          # 触发后是否禁止新买入

        # 权益曲线过滤: 当策略净值低于自身N日均线时，降低仓位
        # 比沪深300过滤更精准——直接反映策略自身状态而非大盘
        'equity_curve_filter_enabled': True,
        'equity_curve_ma_period': 10,               # 权益曲线均线周期（交易日）
        'equity_curve_position_scale': 0.5,         # 低于均线时仓位缩减比例

        # ====== 辅助风控 ======

        # 移动止盈: 盈利超过 activation_profit 后，从最高价回撤 trailing_stop_pct 时卖出
        'trailing_stop_enabled': True,
        'trailing_stop_activation_profit': 0.05,    # 激活条件: 盈利5%
        'trailing_stop_pct': 0.03,                  # 回撤阈值: 从最高价回撤3%

        # 市场环境过滤: 当指数低于N日均线时，降低新买入仓位
        'market_filter_enabled': True,
        'market_index': '000300.XSHG',              # 参考指数（沪深300）
        'market_ma_period': 20,                      # 均线周期
        'market_filter_position_scale': 0.5,         # 熊市时仓位缩减比例

        # 单日最大亏损限制: 当日亏损超过阈值时减仓并禁止新买入
        'daily_loss_limit_enabled': True,
        'daily_loss_limit': -0.03,                   # 日亏损阈值: -3%
        'daily_loss_reduce_ratio': 0.5,              # 触发后减仓比例

        # 板块集中度限制: 同一行业最多持有N只股票
        'sector_limit_enabled': True,
        'sector_max_stocks': 1,                      # 同行业最大持仓数

        # 分批止盈: 盈利达到目标后先卖出一部分，剩余由移动止盈管理
        'partial_profit_enabled': True,
        'partial_profit_target': 0.08,               # 分批止盈目标: 盈利8%
        'partial_profit_sell_ratio': 0.5,            # 首次卖出比例: 50%

        # 时间止损: 持仓超过N天且盈利未达要求时卖出
        'time_stop_enabled': True,
        'time_stop_days': 5,                         # 持仓天数阈值
        'time_stop_min_profit': -0.02,               # 最低盈利要求（低于此值触发）
    },
}

# ================================================
# 策略初始化
# ================================================
def initialize(context):
    # 设置策略基本参数
    set_option('use_real_price', True)  # 使用真实价格，而非复权价格
    log.set_level('system', 'error')    # 设置日志级别，只显示错误信息
    set_option('avoid_future_data', True)  # 避免使用未来数据

    # 设置定时任务（时间从CONFIG读取）
    _setup_schedules()

    # 风险控制追踪变量
    g.trailing_high = {}            # {stock: highest_price_since_purchase} 移动止盈最高价
    g.purchase_dates = {}           # {stock: purchase_date_str} 买入日期
    g.stock_strategy = {}           # {stock: strategy_type} 持仓股票的策略来源（'一进二'/'首板低开'/'弱转强'）
    g.day_start_value = 0           # 每日开始时组合价值（用于日亏损计算）
    g.daily_loss_triggered = False  # 是否触发日亏损限制
    g.market_bullish = True         # 市场是否处于多头趋势
    g.partial_profit_taken = {}     # {stock: bool} 是否已执行分批止盈
    g.portfolio_peak = 0            # 组合历史最高价值（用于回撤熔断）
    g.equity_history = []           # 每日净值历史（用于权益曲线过滤）
    g.portfolio_drawdown_triggered = False  # 组合回撤熔断是否触发

    # 记录CONFIG配置
    _log_config()


# ================================================
# 定时任务注册
# ================================================
def _setup_schedules():
    """注册所有定时任务（供 initialize 和 after_code_changed 调用）"""
    run_daily(get_stock_list, CONFIG['schedule']['get_stock_list'])
    run_daily(buy, CONFIG['schedule']['buy'])
    run_daily(sell_heavy_turnover, time=CONFIG['schedule']['sell_heavy_turnover'])
    run_daily(sell_am, time=CONFIG['schedule']['sell_am'])
    run_daily(sell_pm, time=CONFIG['schedule']['sell_pm'])
    run_daily(log_position_stats, time=CONFIG['schedule']['log_position_stats'])
    run_daily(log_position_mid_stats, time=CONFIG['schedule']['log_position_mid_stats'])
    log.info(f"[FOOTPRINT] _setup_schedules 已注册 {len(CONFIG['schedule'])} 个定时任务")


# ================================================
# 代码修改后处理
# ================================================
def after_code_changed(context):
    """策略代码修改后触发：取消所有定时任务并重新注册，确保schedule与最新代码一致"""
    log.info("[FOOTPRINT] after_code_changed 触发，重新注册定时任务")
    unschedule_all()
    _setup_schedules()
    log.info("[FOOTPRINT] after_code_changed 完成")


# ================================================
# 配置日志
# ================================================
def _log_config():
    """记录CONFIG配置到日志，便于回测时确认参数"""
    log.info("=" * 50)
    log.info("策略配置:")
    log.info(f"  最大持仓: {CONFIG['global']['max_stock_num']}")
    log.info(f"  分类标签: {CONFIG['global']['fenlei']}")
    log.info(f"  一进二: 均价增长>={CONFIG['gap_up']['avg_price_increase_min']:.0%}, "
             f"金额{CONFIG['gap_up']['money_min']/1e8:.1f}-{CONFIG['gap_up']['money_max']/1e8:.0f}亿, "
             f"市值>={CONFIG['gap_up']['market_cap_min']}亿, "
             f"流通市值<={CONFIG['gap_up']['circulating_market_cap_max']}亿")
    log.info(f"  首板低开: 相对位置<={CONFIG['gap_down']['relative_position_max']}, "
             f"低开{1-CONFIG['gap_down']['open_pct_max']:.1%}-{1-CONFIG['gap_down']['open_pct_min']:.1%}")
    log.info(f"  弱转强: {CONFIG['reversal']['increase_days']}日涨幅<={CONFIG['reversal']['increase_ratio_max']:.0%}, "
             f"金额{CONFIG['reversal']['money_min']/1e8:.0f}-{CONFIG['reversal']['money_max']/1e8:.0f}亿")

    rc = CONFIG['risk_control']
    if rc['enabled']:
        log.info("  风险控制: 已启用")
        # 核心风控
        if rc['hard_stop_loss_enabled']:
            log.info(f"    硬止损: 亏损{rc['hard_stop_loss_pct']:.0%}无条件卖出")
        if rc['portfolio_drawdown_enabled']:
            log.info(f"    组合回撤熔断: 回撤>{rc['portfolio_drawdown_threshold']:.0%}时"
                     f"减仓{rc['portfolio_drawdown_reduce_ratio']:.0%}+禁止买入")
        if rc['equity_curve_filter_enabled']:
            log.info(f"    权益曲线过滤: 净值<MA{rc['equity_curve_ma_period']}时"
                     f"仓位{rc['equity_curve_position_scale']:.0%}")
        # 辅助风控
        if rc['trailing_stop_enabled']:
            log.info(f"    移动止盈: 盈利>{rc['trailing_stop_activation_profit']:.0%}后, "
                     f"回撤{rc['trailing_stop_pct']:.0%}卖出")
        if rc['market_filter_enabled']:
            log.info(f"    市场过滤: {rc['market_index']} MA{rc['market_ma_period']}, "
                     f"熊市仓位{rc['market_filter_position_scale']:.0%}")
        if rc['daily_loss_limit_enabled']:
            log.info(f"    日亏损限制: {rc['daily_loss_limit']:.0%}, "
                     f"减仓{rc['daily_loss_reduce_ratio']:.0%}")
        if rc['sector_limit_enabled']:
            log.info(f"    板块集中度: 同行业最多{rc['sector_max_stocks']}只")
        if rc['partial_profit_enabled']:
            log.info(f"    分批止盈: 盈利{rc['partial_profit_target']:.0%}时"
                     f"卖出{rc['partial_profit_sell_ratio']:.0%}")
        if rc['time_stop_enabled']:
            log.info(f"    时间止损: 持仓>{rc['time_stop_days']}天"
                     f"且盈利<{rc['time_stop_min_profit']:.0%}时卖出")
    else:
        log.info("  风险控制: 未启用")
    log.info("=" * 50)


# ================================================
# 风险控制辅助函数
# ================================================
def _check_market_regime(context):
    """检查市场环境：指数是否处于N日均线上方"""
    rc = CONFIG['risk_control']
    if not rc['enabled'] or not rc['market_filter_enabled']:
        g.market_bullish = True
        return

    index_code = rc['market_index']
    ma_period = rc['market_ma_period']

    try:
        close_data = attribute_history(index_code, ma_period, '1d', ['close'], skip_paused=True)
        if len(close_data) < ma_period:
            g.market_bullish = True
            return

        ma_value = close_data['close'].mean()
        current_data = get_current_data()
        current_price = current_data[index_code].last_price

        g.market_bullish = current_price >= ma_value
        log.info(f"市场环境: {'多头' if g.market_bullish else '空头'}, "
                 f"指数={current_price:.2f}, MA{ma_period}={ma_value:.2f}")
    except Exception as e:
        log.warning(f"市场环境检查失败: {e}")
        g.market_bullish = True  # 默认多头，避免错过交易机会


def _update_trailing_highs(context):
    """更新持仓股票的最高价追踪（用于移动止盈）"""
    current_data = get_current_data()
    current_positions = set(context.portfolio.positions.keys())

    # 清理已卖出股票的追踪数据
    for tracking_dict in [g.trailing_high, g.purchase_dates, g.partial_profit_taken, g.stock_strategy]:
        for s in list(tracking_dict.keys()):
            if s not in current_positions:
                del tracking_dict[s]

    # 更新当前持仓的最高价
    for s in current_positions:
        current_price = current_data[s].last_price
        if s in g.trailing_high:
            g.trailing_high[s] = max(g.trailing_high[s], current_price)
        # 注意: 新买入股票的 trailing_high 在 buy() 中初始化


def _check_daily_loss(context):
    """检查当日亏损是否超过限制，若超过则减仓"""
    rc = CONFIG['risk_control']
    if not rc['enabled'] or not rc['daily_loss_limit_enabled']:
        g.daily_loss_triggered = False
        return

    if g.day_start_value <= 0:
        g.daily_loss_triggered = False
        return

    current_value = context.portfolio.total_value
    daily_return = (current_value - g.day_start_value) / g.day_start_value

    if daily_return <= rc['daily_loss_limit']:
        if not g.daily_loss_triggered:  # 避免重复触发
            g.daily_loss_triggered = True
            log.warning(f"触发日亏损限制: 日收益率={daily_return:.2%}, 阈值={rc['daily_loss_limit']:.2%}")
            # 减仓: 对所有可卖持仓按比例减仓
            for s in list(context.portfolio.positions):
                pos = context.portfolio.positions[s]
                if pos.closeable_amount != 0:
                    target_value = pos.value * (1 - rc['daily_loss_reduce_ratio'])
                    order_target_value(s, target_value)
                    log.info(f"日亏损减仓: {s} 减仓{rc['daily_loss_reduce_ratio']:.0%}")
    else:
        g.daily_loss_triggered = False


def _check_portfolio_drawdown(context):
    """检查组合回撤是否超过阈值，若超过则减仓并标记（禁止新买入）"""
    rc = CONFIG['risk_control']
    if not rc['enabled'] or not rc['portfolio_drawdown_enabled']:
        g.portfolio_drawdown_triggered = False
        return

    if g.portfolio_peak <= 0:
        g.portfolio_drawdown_triggered = False
        return

    current_value = context.portfolio.total_value
    drawdown = (g.portfolio_peak - current_value) / g.portfolio_peak

    if drawdown >= rc['portfolio_drawdown_threshold']:
        if not g.portfolio_drawdown_triggered:  # 避免重复触发
            g.portfolio_drawdown_triggered = True
            log.warning(f"触发组合回撤熔断: 回撤={drawdown:.2%}, 阈值={rc['portfolio_drawdown_threshold']:.2%}, "
                        f"峰值={g.portfolio_peak:.0f}, 当前={current_value:.0f}")
            # 减仓: 对所有可卖持仓按比例减仓
            for s in list(context.portfolio.positions):
                pos = context.portfolio.positions[s]
                if pos.closeable_amount != 0:
                    target_value = pos.value * (1 - rc['portfolio_drawdown_reduce_ratio'])
                    order_target_value(s, target_value)
                    log.info(f"组合回撤减仓: {s} 减仓{rc['portfolio_drawdown_reduce_ratio']:.0%}")
    else:
        if g.portfolio_drawdown_triggered:
            log.info(f"组合回撤恢复: 回撤={drawdown:.2%} < 阈值{rc['portfolio_drawdown_threshold']:.2%}")
        g.portfolio_drawdown_triggered = False


def _get_industry_codes(stock_list):
    """获取股票行业代码（申万一级），用于板块集中度检查
    
    JoinQuant get_industry() 返回格式:
        {'000001.XSHE': {'行业代码': 'J66', '行业名称': '银行', ...}}
    需要从嵌套dict中提取行业代码字符串
    """
    if not stock_list:
        return {}
    result = {}
    for stock in stock_list:
        try:
            ind = get_industry(stock)
            if isinstance(ind, dict) and ind:
                # 取第一个value（股票代码对应的行业信息）
                ind_val = list(ind.values())[0]
                if isinstance(ind_val, dict):
                    # 嵌套dict: {'行业代码': 'J66', '行业名称': '银行', ...}
                    # 优先取行业代码，其次取行业名称，最后取str
                    result[stock] = ind_val.get('行业代码') or ind_val.get('行业名称') or str(ind_val)
                elif isinstance(ind_val, str):
                    result[stock] = ind_val
                else:
                    result[stock] = str(ind_val) if ind_val is not None else None
            elif isinstance(ind, str):
                result[stock] = ind
            else:
                result[stock] = str(ind) if ind else None
        except Exception:
            result[stock] = None
    return result


def _check_sector_concentration(qualified_stocks, context):
    """检查板块集中度，过滤掉同行业超限的股票"""
    rc = CONFIG['risk_control']
    if not rc['enabled'] or not rc['sector_limit_enabled']:
        return qualified_stocks

    if not qualified_stocks:
        return qualified_stocks

    # 获取所有相关股票的行业代码
    all_stocks = list(context.portfolio.positions.keys()) + qualified_stocks
    industry_codes = _get_industry_codes(all_stocks)

    # 统计当前持仓中各行业的股票数
    position_sector_count = {}
    for s in context.portfolio.positions:
        ind = industry_codes.get(s, None)
        if ind is not None:
            position_sector_count[ind] = position_sector_count.get(ind, 0) + 1

    # 过滤qualified_stocks
    filtered = []
    new_sector_count = {}  # 本次新增的同行业计数
    for s in qualified_stocks:
        ind = industry_codes.get(s, None)
        if ind is None:
            # 无法获取行业信息，允许买入
            filtered.append(s)
            continue

        current_count = position_sector_count.get(ind, 0)
        new_count = new_sector_count.get(ind, 0)

        if current_count + new_count < rc['sector_max_stocks']:
            filtered.append(s)
            new_sector_count[ind] = new_count + 1
        else:
            log.info(f"板块集中度限制: 跳过{s} (行业{ind}已有{current_count + new_count}只)")

    return filtered


# ================================================
# 选股函数
# ================================================
def get_stock_list(context):
    log.info(f"[FOOTPRINT] get_stock_list 触发 @ {CONFIG['schedule']['get_stock_list']} | 日期={context.previous_date}")

    # 记录每日起始组合价值（用于日亏损计算）
    g.day_start_value = context.portfolio.total_value
    g.daily_loss_triggered = False

    # 更新组合峰值和权益历史（用于回撤熔断和权益曲线过滤）
    current_value = context.portfolio.total_value
    g.portfolio_peak = max(g.portfolio_peak, current_value)
    g.equity_history.append(current_value)
    log.info(f"[FOOTPRINT] get_stock_list | 总资产={current_value:.0f}, 峰值={g.portfolio_peak:.0f}")

    # 检查市场环境
    _check_market_regime(context)

    # 获取交易日期
    date = context.previous_date  # 前一个交易日
    date_2, date_1, date = get_trade_days(end_date=date, count=3)  # 获取最近3个交易日

    # 获取初始股票池
    initial_list = prepare_stock_list(date)
    log.info(f"[FOOTPRINT] get_stock_list | 初始股票池={len(initial_list)}只")

    # 获取不同日期的涨停股票列表
    hl0_list = get_hl_stock(initial_list, date)       # 昨日涨停股票
    hl1_list = get_ever_hl_stock(initial_list, date_1)  # 前日曾涨停股票
    hl2_list = get_ever_hl_stock(initial_list, date_2)  # 前前日曾涨停股票
    log.info(f"[FOOTPRINT] get_stock_list | 昨日涨停={len(hl0_list)}, 前日曾涨停={len(hl1_list)}, 前前日曾涨停={len(hl2_list)}")

    # 一进二策略：昨日涨停且前两日未涨停的股票
    elements_to_remove = set(hl1_list + hl2_list)  # 合并前两日涨停股票，用于快速查找
    g.gap_up = [stock for stock in hl0_list if stock not in elements_to_remove]  # 昨日涨停且前两日未涨停

    # 首板低开策略：昨日首次涨停的股票
    g.gap_down = [s for s in hl0_list if s not in hl1_list]  # 昨日涨停但前日未涨停

    # 弱转强策略：昨日曾涨停但收盘未涨停，且前日未涨停的股票
    h1_list = get_ever_hl_stock2(initial_list, date)  # 昨日曾涨停但收盘未涨停的股票
    elements_to_remove = get_hl_stock(initial_list, date_1)  # 前日涨停的股票
    g.reversal = [stock for stock in h1_list if stock not in elements_to_remove]  # 昨日曾涨停但收盘未涨停，且前日未涨停

    log.info(f"[FOOTPRINT] get_stock_list 完成 | 一进二={len(g.gap_up)}, 首板低开={len(g.gap_down)}, 弱转强={len(g.reversal)}")


def check_position_limit(context):
    """检查持仓数量是否超过限制"""
    max_stock_num = CONFIG['global']['max_stock_num']
    current_positions = len(context.portfolio.positions)
    if current_positions > max_stock_num:
        log.warning(f"持仓数量 {current_positions} 超过限制 {max_stock_num}")
        # 强制平仓多余的持仓
        positions = list(context.portfolio.positions.keys())
        # 卖出超出限制的部分
        for s in positions[max_stock_num:]:
            order_target_value(s, 0)
            log.info(f"强制卖出 {s} 以符合持仓限制")


# ================================================
# 买入函数
# ================================================
def buy(context):
    log.info(f"[FOOTPRINT] buy 触发 @ {CONFIG['schedule']['buy']} | 持仓={len(context.portfolio.positions)}只, 可用资金={context.portfolio.available_cash:.0f}")

    check_position_limit(context)

    rc = CONFIG['risk_control']

    # 风控检查: 日亏损限制触发时不买入
    if rc['enabled'] and g.daily_loss_triggered:
        log.info("日亏损限制已触发，不买入新股票")
        return

    # 风控检查: 组合回撤熔断触发时不买入
    if rc['enabled'] and rc['portfolio_drawdown_enabled'] and rc['portfolio_drawdown_no_buy'] and g.portfolio_drawdown_triggered:
        log.info("组合回撤熔断已触发，不买入新股票")
        return

    # 初始化股票列表
    qualified_stocks = []
    gk_stocks = []
    dk_stocks = []
    rzq_stocks = []

    # 获取当前市场数据和时间
    current_data = get_current_data()
    date_now = context.current_dt.strftime("%Y-%m-%d")
    mid_time1 = '09:15:00'
    end_times1 = '09:26:00'
    start = date_now + mid_time1
    end = date_now + end_times1

    # ====== 一进二策略 (gap_up) ======
    gu = CONFIG['gap_up']
    for s in g.gap_up:
        # 条件一：筛选均价、成交金额
        prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
        # 计算均价增长值，要求大于阈值
        avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][0] * 1.1 - 1
        # 成交金额要求在指定范围内
        if avg_price_increase_value < gu['avg_price_increase_min'] or prev_day_data['money'][0] < gu['money_min'] or prev_day_data['money'][0] > gu['money_max']:
            continue

        # 条件二：筛选市值
        turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                          fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
        if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < gu['market_cap_min'] or turnover_ratio_data['circulating_market_cap'][0] > gu['circulating_market_cap_max']:
            continue

        # 条件三：排除左压（上涨时未放量）的股票
        if rise_low_volume(s, context):
            continue

        # 条件四：检查集合竞价数据
        auction_data = get_call_auction(s, start_date=date_now, end_date=date_now, fields=['time', 'volume', 'current'])
        # 集合竞价成交量要求大于昨日成交量的指定比例
        if auction_data.empty or auction_data['volume'][0] / prev_day_data['volume'][-1] < gu['auction_vol_ratio_min']:
            continue
        # 开盘价相对于昨日涨停价的比例要求在指定范围内
        current_ratio = auction_data['current'][0] / (current_data[s].high_limit / 1.1)
        if current_ratio <= gu['current_ratio_min'] or current_ratio >= gu['current_ratio_max']:
            continue

        # 如果股票满足所有条件，则添加到列表中
        gk_stocks.append(s)
        qualified_stocks.append(s)

    log.info(f"[FOOTPRINT] buy | 一进二筛选: 候选={len(g.gap_up)}, 通过={len(gk_stocks)}")

    date = transform_date(context.previous_date, 'str')

    # ====== 首板低开策略 (gap_down) ======
    gd = CONFIG['gap_down']
    if g.gap_down:
        stock_list = g.gap_down

        # 条件一：筛选相对位置
        rpd = get_relative_position_df(stock_list, date, gd['watch_days'])
        rpd = rpd[rpd['rp'] <= gd['relative_position_max']]
        stock_list = list(rpd.index)

        # 条件二：筛选低开幅度
        if len(stock_list) != 0:
            df = get_price(stock_list, end_date=date, frequency='daily', fields=['close'],
                          count=1, panel=False, fill_paused=False, skip_paused=True).set_index('code')
            df['open_pct'] = [current_data[s].day_open / df.loc[s, 'close'] for s in stock_list]
            df = df[(gd['open_pct_min'] <= df['open_pct']) & (df['open_pct'] <= gd['open_pct_max'])]
            stock_list = list(df.index)

        # 条件三：筛选成交金额
        for s in stock_list:
            prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            if prev_day_data['money'][0] >= gd['money_min']:
                dk_stocks.append(s)
                qualified_stocks.append(s)

    log.info(f"[FOOTPRINT] buy | 首板低开筛选: 候选={len(g.gap_down)}, 通过={len(dk_stocks)}")

    # ====== 弱转强策略 (reversal) ======
    rv = CONFIG['reversal']
    for s in g.reversal:
        # 条件一：过滤前N天涨幅超过限制的股票
        price_data = attribute_history(s, rv['increase_days'], '1d', fields=['close'], skip_paused=True)
        if len(price_data) < rv['increase_days']:
            continue
        increase_ratio = (price_data['close'][-1] - price_data['close'][0]) / price_data['close'][0]
        if increase_ratio > rv['increase_ratio_max']:
            continue

        # 条件二：过滤前一日收盘价小于开盘价超过阈值的股票（大幅低走）
        prev_day_data = attribute_history(s, 1, '1d', fields=['open', 'close'], skip_paused=True)
        if len(prev_day_data) < 1:
            continue
        open_close_ratio = (prev_day_data['close'][0] - prev_day_data['open'][0]) / prev_day_data['open'][0]
        if open_close_ratio < rv['open_close_ratio_min']:
            continue

        # 条件三：筛选均价和成交金额
        prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
        avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][0] - 1
        if avg_price_increase_value < rv['avg_price_increase_min'] or prev_day_data['money'][0] < rv['money_min'] or prev_day_data['money'][0] > rv['money_max']:
            continue

        # 条件四：筛选市值
        turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                          fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
        if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < rv['market_cap_min'] or turnover_ratio_data['circulating_market_cap'][0] > rv['circulating_market_cap_max']:
            continue

        # 条件五：排除左压（上涨时未放量）的股票
        if rise_low_volume(s, context):
            continue

        # 条件六：检查集合竞价数据
        auction_data = get_call_auction(s, start_date=date_now, end_date=date_now, fields=['time', 'volume', 'current'])
        # 集合竞价成交量要求大于昨日成交量的指定比例
        if auction_data.empty or auction_data['volume'][0] / prev_day_data['volume'][-1] < rv['auction_vol_ratio_min']:
            continue
        # 开盘价相对于昨日涨停价的比例要求在指定范围内
        current_ratio = auction_data['current'][0] / (current_data[s].high_limit / 1.1)
        if current_ratio <= rv['current_ratio_min'] or current_ratio >= rv['current_ratio_max']:
            continue

        # 如果股票满足所有条件，则添加到列表中
        rzq_stocks.append(s)
        qualified_stocks.append(s)

    log.info(f"[FOOTPRINT] buy | 弱转强筛选: 候选={len(g.reversal)}, 通过={len(rzq_stocks)}")

    # ====== 构建股票→策略类型映射 ======
    stock_strategy_map = {}
    for s in gk_stocks:
        stock_strategy_map[s] = '一进二'
    for s in dk_stocks:
        stock_strategy_map[s] = '首板低开'
    for s in rzq_stocks:
        stock_strategy_map[s] = '弱转强'

    # ====== 板块集中度过滤 ======
    qualified_stocks = _check_sector_concentration(qualified_stocks, context)

    # ====== 执行买入 ======
    max_stock_num = CONFIG['global']['max_stock_num']
    cash_ratio_min = CONFIG['global']['cash_ratio_min']
    min_shares = CONFIG['global']['min_shares']

    # 仅当有符合条件的股票且可用现金占总资产比例>阈值时执行买入
    if len(qualified_stocks) != 0 and context.portfolio.available_cash / context.portfolio.total_value > cash_ratio_min:
        # 获取当前持仓数量
        current_position_count = len(context.portfolio.positions)

        # 计算还能买入多少只股票
        can_buy_count = min(max_stock_num - current_position_count, len(qualified_stocks))

        # 如果没有持仓名额了，直接返回
        if can_buy_count <= 0:
            log.info(f"持仓已达上限 {max_stock_num} 只，不再买入新股票")
            return

        # 限制只买入前 can_buy_count 只股票
        qualified_stocks = qualified_stocks[:can_buy_count]

        # 计算每只股票的买入金额，平均分配可用资金
        value = context.portfolio.available_cash / len(qualified_stocks)

        # 熊市减仓: 如果市场环境为空头，按比例缩减买入金额
        rc = CONFIG['risk_control']
        if rc['enabled'] and rc['market_filter_enabled'] and not g.market_bullish:
            value = value * rc['market_filter_position_scale']
            log.info(f"熊市减仓: 买入金额缩减至{rc['market_filter_position_scale']:.0%}")

        # 权益曲线过滤: 策略净值低于自身均线时，缩减买入金额
        if rc['enabled'] and rc['equity_curve_filter_enabled']:
            ma_period = rc['equity_curve_ma_period']
            if len(g.equity_history) >= ma_period:
                equity_ma = sum(g.equity_history[-ma_period:]) / ma_period
                current_portfolio_value = context.portfolio.total_value
                if current_portfolio_value < equity_ma:
                    value = value * rc['equity_curve_position_scale']
                    log.info(f"权益曲线过滤: 净值{current_portfolio_value:.0f} < MA{ma_period}={equity_ma:.0f}, "
                             f"买入金额缩减至{rc['equity_curve_position_scale']:.0%}")

        for s in qualified_stocks:
            # 确保有足够资金买入至少min_shares股
            if context.portfolio.available_cash / current_data[s].last_price > min_shares:
                # 以开盘价买入
                order_value(s, value, MarketOrderStyle(current_data[s].day_open))
                # 初始化风险控制追踪数据
                g.trailing_high[s] = current_data[s].day_open
                g.purchase_dates[s] = context.current_dt.strftime("%Y-%m-%d")
                g.partial_profit_taken[s] = False
                g.stock_strategy[s] = stock_strategy_map.get(s, '未知')

        log.info(f"买入 {len(qualified_stocks)} 只股票，当前持仓 {current_position_count + len(qualified_stocks)} 只")

    log.info(f"[FOOTPRINT] buy 完成 | 合格={len(qualified_stocks)}只, 买入后持仓={len(context.portfolio.positions)}只")


# ================================================
# 日期处理相关函数
# ================================================
def transform_date(date, date_type):

    if type(date) == str:
        str_date = date
        dt_date = dt.datetime.strptime(date, '%Y-%m-%d')
        d_date = dt_date.date()
    elif type(date) == dt.datetime:
        str_date = date.strftime('%Y-%m-%d')
        dt_date = date
        d_date = dt_date.date()
    elif type(date) == dt.date:
        str_date = date.strftime('%Y-%m-%d')
        dt_date = dt.datetime.strptime(str_date, '%Y-%m-%d')
        d_date = date
    dct = {'str': str_date, 'dt': dt_date, 'd': d_date}
    return dct[date_type]

def get_shifted_date(date, days, days_type='T'):

    # 获取上一个自然日
    d_date = transform_date(date, 'd')
    yesterday = d_date + dt.timedelta(-1)

    # 按自然日平移
    if days_type == 'N':
        shifted_date = yesterday + dt.timedelta(days+1)

    # 按交易日平移
    if days_type == 'T':
        all_trade_days = [i.strftime('%Y-%m-%d') for i in list(get_all_trade_days())]

        # 如果上一个自然日是交易日，根据其在交易日列表中的index计算平移后的交易日
        if str(yesterday) in all_trade_days:
            shifted_date = all_trade_days[all_trade_days.index(str(yesterday)) + days + 1]
        # 否则，从上一个自然日向前数，先找到最近一个交易日，再开始平移
        else:
            for i in range(100):
                last_trade_date = yesterday - dt.timedelta(i)
                if str(last_trade_date) in all_trade_days:
                    shifted_date = all_trade_days[all_trade_days.index(str(last_trade_date)) + days + 1]
                    break
    return str(shifted_date)



# ================================================
# 股票过滤函数
# ================================================
def filter_new_stock(initial_list, date, days=None):
    """过滤上市不足N天的新股"""
    if days is None:
        days = CONFIG['filter']['new_stock_days']
    d_date = transform_date(date, 'd')
    return [stock for stock in initial_list if d_date - get_security_info(stock).start_date > dt.timedelta(days=days)]

def filter_st_paused_stock(initial_list):
    current_data = get_current_data()
    # 使用列表推导式结合any()函数，筛选出符合条件的股票
    return [stock for stock in initial_list
            if not any([
                current_data[stock].is_st,          # 排除ST股
                current_data[stock].paused,         # 排除停牌股
                '退' in current_data[stock].name    # 排除名称中含'退'字的股票，避免退市股
            ])]

def filter_kcbj_stock(initial_list):
    return [stock for stock in initial_list if stock[:2] in (('60','00','30'))]

def filter_st_stock(initial_list, date):
    str_date = transform_date(date, 'str')
    # 如果当前日期不是交易日，则使用前一个交易日
    if get_shifted_date(str_date, 0, 'N') != get_shifted_date(str_date, 0, 'T'):
        str_date = get_shifted_date(str_date, -1, 'T')
    # 获取股票的ST状态
    df = get_extras('is_st', initial_list, start_date=str_date, end_date=str_date, df=True)
    df = df.T
    df.columns = ['is_st']
    # 过滤掉ST股票
    df = df[df['is_st'] == False]
    filter_list = list(df.index)
    return filter_list

def filter_paused_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['paused'], count=1, panel=False, fill_paused=True)
    # 过滤掉停牌股票（paused=0表示未停牌）
    df = df[df['paused'] == 0]
    paused_list = list(df.code)
    return paused_list

def filter_extreme_limit_stock(context, stock_list, date):
    tmp = []
    for stock in stock_list:
        df = get_price(stock, end_date=date, frequency='daily', fields=['low','high_limit'], count=1, panel=False)
        # 如果最低价小于涨停价，说明不是一字涨停
        if df.iloc[0,0] < df.iloc[0,1]:
            tmp.append(stock)
    return tmp



def prepare_stock_list(date):
    # 获取所有A股
    initial_list = get_all_securities('stock', date).index.tolist()
    # 过滤掉科创板
    initial_list = filter_kcbj_stock(initial_list)
    # 过滤掉新股
    initial_list = filter_new_stock(initial_list, date)
    # 过滤掉ST股和停牌股
    initial_list = filter_st_paused_stock(initial_list)
    return initial_list

def rise_low_volume(s, context):
    """左压检测：上涨时未放量则存在左压"""
    rlv = CONFIG['rise_low_volume']
    # 获取股票的历史高价和成交量数据
    hist = attribute_history(s, rlv['hist_days'], '1d', fields=['high','volume'], skip_paused=True, df=False)
    high_prices = hist['high'][:rlv['high_days']]
    prev_high = high_prices[-1]  # 最近一日的高价

    # 寻找前面最近一次高于当前高价的日期，计算中间的天数
    zyts_0 = next((i-1 for i, high in enumerate(high_prices[-3::-1], 2) if high >= prev_high), 100)
    zyts = zyts_0 + rlv['buffer']  # 增加buffer天作为观察期

    # 如果当前成交量小于观察期内最大成交量的指定比例，则认为存在左压
    if hist['volume'][-1] <= max(hist['volume'][-zyts:-1]) * rlv['volume_ratio']:
        return True
    return False

def get_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close','high_limit'], count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    # 筛选收盘价等于涨停价的股票
    df = df[df['close'] == df['high_limit']]
    hl_list = list(df.code)
    return hl_list

def get_ever_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['high','high_limit'], count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    # 筛选最高价等于涨停价的股票
    df = df[df['high'] == df['high_limit']]
    hl_list = list(df.code)
    return hl_list

def get_ever_hl_stock2(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close','high','high_limit'], count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    cd1 = df['high'] == df['high_limit']  # 条件1：最高价等于涨停价（曾经涨停）
    cd2 = df['close'] != df['high_limit']  # 条件2：收盘价不等于涨停价（收盘未涨停）
    df = df[cd1 & cd2]  # 同时满足两个条件
    hl_list = list(df.code)
    return hl_list

# ================================================
# 涨停统计和指数涨幅计算函数
# ================================================
def get_hl_count_df(hl_list, date, watch_days):
    # 获取watch_days的数据
    df = get_price(hl_list, end_date=date, frequency='daily', fields=['close','high_limit','low'], count=watch_days, panel=False, fill_paused=False, skip_paused=False)
    df.index = df.code

    # 计算涨停与一字涨停数，一字涨停定义为最低价等于涨停价
    hl_count_list = []
    extreme_hl_count_list = []
    for stock in hl_list:
        df_sub = df.loc[stock]
        # 计算收盘涨停的天数
        hl_days = df_sub[df_sub.close==df_sub.high_limit].high_limit.count()
        # 计算一字涨停的天数（最低价等于涨停价）
        extreme_hl_days = df_sub[df_sub.low==df_sub.high_limit].high_limit.count()
        hl_count_list.append(hl_days)
        extreme_hl_count_list.append(extreme_hl_days)

    # 创建DataFrame记录结果
    df = pd.DataFrame(index=hl_list, data={'count':hl_count_list, 'extreme_count':extreme_hl_count_list})
    return df

def get_continue_count_df(hl_list, date, watch_days):
    df = pd.DataFrame()
    # 从2板开始，逐步检查到watch_days板
    for d in range(2, watch_days+1):
        # 获取d天内的涨停次数
        HLC = get_hl_count_df(hl_list, date, d)
        # 筛选出涨停次数等于d的股票（即连续d天涨停）
        CHLC = HLC[HLC['count'] == d]
        df = df.append(CHLC)

    # 处理可能的重复记录，保留最大连板数
    stock_list = list(set(df.index))
    ccd = pd.DataFrame()
    for s in stock_list:
        tmp = df.loc[[s]]
        if len(tmp) > 1:  # 如果一只股票有多条记录
            M = tmp['count'].max()  # 取最大连板数
            tmp = tmp[tmp['count'] == M]  # 只保留最大连板数的记录
        ccd = ccd.append(tmp)

    # 按连板数降序排列
    if len(ccd) != 0:
        ccd = ccd.sort_values(by='count', ascending=False)
    return ccd

def get_index_increase_ratio(index_code, context):
    # 获取指数昨天和前天的收盘价
    close_prices = attribute_history(index_code, 2, '1d', fields=['close'], skip_paused=True)
    if len(close_prices) < 2:
        return 0  # 如果数据不足，返回0

    day_before_yesterday_close = close_prices['close'][0]  # 前天收盘价
    yesterday_close = close_prices['close'][1]  # 昨天收盘价

    # 计算涨幅
    increase_ratio = (yesterday_close - day_before_yesterday_close) / day_before_yesterday_close
    return increase_ratio

# ================================================
# 卖出函数
# ================================================
def sell_heavy_turnover(context):
    """高位放量卖出：处于高位、成交量放大、收盘位置低、价格下跌时卖出"""
    log.info(f"[FOOTPRINT] sell_heavy_turnover 触发 @ {CONFIG['schedule']['sell_heavy_turnover']} | 持仓={len(context.portfolio.positions)}只")
    sl = CONFIG['sell']
    date = transform_date(context.previous_date, 'str')
    current_data = get_current_data()

    # 更新移动止盈追踪
    _update_trailing_highs(context)

    # 计算相对位置
    rpd = get_relative_position_df2(list(context.portfolio.positions), date, sl['heavy_turnover_watch_days'])
    rpd = rpd[rpd['rp'] >= sl['heavy_turnover_rp_min']]  # 筛选处于高位的股票
    stock_list = list(rpd.index)

    for s in stock_list:
        df = get_price(s, end_date=date, fields=['high', 'low', 'close', 'open', 'volume', 'high_limit'], count=2, fill_paused=False, skip_paused=False, panel=False)
        pullback_ratio = (df['close'][-1] - df['high'][-1]) / df['high'][-1]  # 回调比例
        vol_ratio = df['volume'][-1] / df['volume'][-2]  # 成交量比例
        close_pos = (df['close'][-1] - min(df['high'][-1], df['low'][-1])) / abs(df['high'][-1] - df['low'][-1])  # 收盘位置

        # 如果处于高位，成交量放大，回调位置低于阈值，当天在跌, 则卖出
        if ((context.portfolio.positions[s].closeable_amount != 0) and (vol_ratio >= sl['heavy_turnover_vol_ratio_min']) and (close_pos <= sl['heavy_turnover_close_pos_max']) and (current_data[s].last_price < df['close'][-1])):
            order_target_value(s, 0)  # 清仓
            log.info(f"高位放量卖出: {s}")


def sell_am(context):
    """上午止盈：未涨停且有盈利时卖出；分批止盈：盈利达到目标时先卖出一部分"""
    log.info(f"[FOOTPRINT] sell_am 触发 @ {CONFIG['schedule']['sell_am']} | 持仓={len(context.portfolio.positions)}只")
    sl = CONFIG['sell']
    rc = CONFIG['risk_control']
    date = transform_date(context.previous_date, 'str')
    current_data = get_current_data()

    # 更新移动止盈追踪
    _update_trailing_highs(context)

    # 遍历所有持仓股票
    for s in list(context.portfolio.positions):
        pos = context.portfolio.positions[s]
        if pos.closeable_amount == 0:
            continue

        profit_ratio = (current_data[s].last_price - pos.avg_cost) / pos.avg_cost
        is_at_limit = current_data[s].last_price >= current_data[s].high_limit

        # 分批止盈：未涨停 + 盈利达到目标 + 未执行过分批止盈
        if (rc['enabled'] and rc['partial_profit_enabled'] and
            not is_at_limit and
            not g.partial_profit_taken.get(s, False) and
            profit_ratio >= rc['partial_profit_target']):
            sell_ratio = rc['partial_profit_sell_ratio']
            target_value = pos.value * (1 - sell_ratio)
            order_target_value(s, target_value)
            g.partial_profit_taken[s] = True
            log.info(f"分批止盈: {s} 卖出{sell_ratio:.0%}, 盈利{profit_ratio:.2%}")
            continue

        # 已分批止盈的股票，不执行常规止盈（剩余部分由移动止盈/止损管理）
        if g.partial_profit_taken.get(s, False):
            continue

        # 常规止盈：未涨停且有盈利
        if not is_at_limit and profit_ratio > 0:
            order_target_value(s, 0)  # 清仓


def sell_pm(context):
    """下午止盈/止损：常规止盈 + MA止损 + 移动止盈 + 时间止损 + 日亏损减仓"""
    log.info(f"[FOOTPRINT] sell_pm 触发 @ {CONFIG['schedule']['sell_pm']} | 持仓={len(context.portfolio.positions)}只")
    sl = CONFIG['sell']
    rc = CONFIG['risk_control']
    date = transform_date(context.previous_date, 'str')
    current_data = get_current_data()

    # 更新移动止盈追踪
    _update_trailing_highs(context)

    # 检查日亏损限制
    _check_daily_loss(context)

    # 检查组合回撤熔断
    _check_portfolio_drawdown(context)

    # 遍历所有持仓股票
    for s in list(context.portfolio.positions):
        pos = context.portfolio.positions[s]
        if pos.closeable_amount == 0:
            continue

        profit_ratio = (current_data[s].last_price - pos.avg_cost) / pos.avg_cost
        is_at_limit = current_data[s].last_price >= current_data[s].high_limit

        # 0. 硬止损：亏损超过阈值无条件卖出（优先级最高，在所有卖出逻辑之前）
        if (rc['enabled'] and rc['hard_stop_loss_enabled'] and
            profit_ratio <= rc['hard_stop_loss_pct']):
            order_target_value(s, 0)
            log.info(f"硬止损: {s} 亏损{profit_ratio:.2%} <= {rc['hard_stop_loss_pct']:.0%}")
            continue

        # 1. 移动止盈：盈利超过激活阈值后，从最高价回撤超过阈值时卖出
        if (rc['enabled'] and rc['trailing_stop_enabled'] and
            s in g.trailing_high and
            profit_ratio >= rc['trailing_stop_activation_profit']):
            trailing_high = g.trailing_high[s]
            if current_data[s].last_price <= trailing_high * (1 - rc['trailing_stop_pct']):
                order_target_value(s, 0)
                log.info(f"移动止盈: {s} 从最高价{trailing_high:.2f}回撤{rc['trailing_stop_pct']:.0%}, 盈利{profit_ratio:.2%}")
                continue

        # 2. 常规止盈：未涨停且有盈利（已分批止盈的由移动止盈管理，不再常规止盈）
        if not is_at_limit and profit_ratio > 0 and not g.partial_profit_taken.get(s, False):
            order_target_value(s, 0)  # 清仓
            continue

        # 3. MA止损：价格跌破N日均线
        ma_period = sl['ma_stop_period']
        close_data = attribute_history(s, ma_period - 1, '1d', ['close'])
        if len(close_data) < ma_period - 1:
            continue
        M_prev = close_data['close'].mean()
        MA = (M_prev * (ma_period - 1) + current_data[s].last_price) / ma_period

        if current_data[s].last_price < MA:
            order_target_value(s, 0)  # 清仓
            log.info(f"MA{ma_period}止损: {s} 价格{current_data[s].last_price:.2f} < MA{ma_period}={MA:.2f}")
            continue

        # 4. 时间止损：持仓超过N天且盈利未达要求
        if (rc['enabled'] and rc['time_stop_enabled'] and
            s in g.purchase_dates):
            purchase_date = g.purchase_dates[s]
            current_date = context.current_dt.strftime("%Y-%m-%d")
            try:
                holding_days = len(get_trade_days(start_date=purchase_date, end_date=current_date)) - 1
            except Exception:
                holding_days = 0
            if (holding_days >= rc['time_stop_days'] and
                profit_ratio < rc['time_stop_min_profit']):
                order_target_value(s, 0)
                log.info(f"时间止损: {s} 持仓{holding_days}天, 盈利{profit_ratio:.2%}")
                continue

def _log_position_stats(context):
    current_data = get_current_data()
    positions = context.portfolio.positions

    if not positions:
        log.info("持仓统计: 当前无持仓")
        return

    total_profit = 0.0          # 总体持仓盈利金额
    strategy_profit = {}        # {策略类型: 盈利金额}
    strategy_count = {}         # {策略类型: 持仓数量}

    log.info("=" * 60)
    log.info("持仓统计:")

    for s in list(positions):
        pos = positions[s]
        if pos.total_amount == 0:
            continue

        profit = (current_data[s].last_price - pos.avg_cost) * pos.total_amount
        profit_ratio = (current_data[s].last_price - pos.avg_cost) / pos.avg_cost
        strategy = g.stock_strategy.get(s, '未知')

        total_profit += profit
        strategy_profit[strategy] = strategy_profit.get(strategy, 0) + profit
        strategy_count[strategy] = strategy_count.get(strategy, 0) + 1

        stock_name = current_data[s].name
        log.info(f"  {s} {stock_name} | 策略={strategy} | 盈利={profit_ratio:+.2%} | 盈亏额={profit:+.0f}")

    # 总体盈利
    total_value = context.portfolio.total_value
    log.info(f"  总体持仓盈亏额: {total_profit:+.0f} | 总资产: {total_value:.0f}")

    # 各策略统计
    log.info("  各策略统计:")
    for strategy in sorted(strategy_profit.keys()):
        sp = strategy_profit[strategy]
        sc = strategy_count[strategy]
        ratio = sp / total_profit * 100 if total_profit != 0 else 0
        log.info(f"    {strategy}: {sc}只, 盈亏额={sp:+.0f}, 占比={ratio:.1f}%")

    log.info("=" * 60)

    
# ================================================
# 每日持仓统计
# ================================================
def log_position_stats(context):
    """每日持仓统计：每股盈利、总体盈利、策略类型、各策略盈利占比"""
    log.info(f"[FOOTPRINT] log_position_stats 触发")
    _log_position_stats(context)
    
def log_position_mid_stats(context):
    """每日持仓统计：每股盈利、总体盈利、策略类型、各策略盈利占比"""
    log.info(f"[FOOTPRINT] log_position_mid_stats 触发")
    _log_position_stats(context)

# ================================================
# 首板低开策略辅助函数
# ================================================
def filter_new_stock2(initial_list, date, days=None):
    """过滤上市不足N天的新股（首板低开策略用）"""
    if days is None:
        days = CONFIG['filter']['new_stock2_days']
    d_date = transform_date(date, 'd')
    return [stock for stock in initial_list if d_date - get_security_info(stock).start_date > dt.timedelta(days=days)]


# 每日初始股票池
def prepare_stock_list2(date):
    initial_list = get_all_securities('stock', date).index.tolist()
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_new_stock2(initial_list, date)
    initial_list = filter_st_stock(initial_list, date)
    initial_list = filter_paused_stock(initial_list, date)
    return initial_list

# 计算股票处于一段时间内相对位置
def get_relative_position_df(stock_list, date, watch_days):
    if len(stock_list) != 0:
        df = get_price(stock_list, end_date=date, fields=['high', 'low', 'close'], count=watch_days, fill_paused=False, skip_paused=False, panel=False).dropna()
        close = df.groupby('code').apply(lambda df: df.iloc[-1,-1])
        high = df.groupby('code').apply(lambda df: df['high'].max())
        low = df.groupby('code').apply(lambda df: df['low'].min())
        result = pd.DataFrame()
        result['rp'] = (close-low) / (high-low)
        return result
    else:
        return pd.DataFrame(columns=['rp'])

# 计算股票最高位处于一段时间内相对位置
def get_relative_position_df2(stock_list, date, watch_days):
    if len(stock_list) != 0:
        df = get_price(stock_list, end_date=date, fields=['high', 'low', 'close'], count=watch_days, fill_paused=False, skip_paused=False, panel=False).dropna()
        yestoday_high = df.groupby('code').apply(lambda df: df.iloc[-1,-3])
        high = df.groupby('code').apply(lambda df: df['high'].max())
        low = df.groupby('code').apply(lambda df: df['low'].min())
        result = pd.DataFrame()
        result['rp'] = (yestoday_high-low) / (high-low)
        return result
    else:
        return pd.DataFrame(columns=['rp'])


