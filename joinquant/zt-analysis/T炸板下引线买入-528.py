# 克隆自聚宽文章：https://www.joinquant.com/post/70871
# 标题：一字板下引线买入法
# 作者：天香膏

from datetime import datetime, timedelta
from jqdata import *
import numpy as np
import pandas as pd

# ===========================
# 全局配置
# ===========================
BUY_MAX_COUNT = 2
TODAY_BUY_PRICE_RATIO = 0.97
MAX_3D_INCREASE = 1.25
MAX_HOLDING_DAYS = 5

# ===========================
# 获利筹码计算
# ===========================
def calc_custom_winner(code, end_date):
    try:
        df = get_bars(
            security=code,
            count=120,
            unit='1d',
            fields=['close','high','low','money'],
            end_dt=end_date
        )
        close_arr = df['close']
        A1 = close_arr[-2]
        sum_An = 0.0
        for i in range(120):
            H = df['high'][i]
            L = df['low'][i]
            M = df['money'][i]
            if H - L < 0.01:
                part = M
            else:
                part = M * (H - A1) / (H - L) if H > A1 else M
            sum_An += part

        val_df = get_valuation(code, end_date=end_date, count=1, fields=['circulating_market_cap'])
        circ_value = val_df['circulating_market_cap'].iloc[0] * 100000000
        return round(100 * sum_An / circ_value, 2)
    except Exception as e:
        log.warning(f"calc_custom_winner({code}) 异常: {e}")
        return -1

# ===========================
# 9:26 竞价一字板
# ===========================
def get_yizi_auction_stocks(context):
    log.info("→ 开始获取集合竞价一字板股票...")
    today = context.current_dt.date()
    yesterday = context.previous_date
    stock_list = get_all_securities(['stock']).index.tolist()
    current_data = get_current_data()
    stock_list = [s for s in stock_list if not current_data[s].paused and not current_data[s].is_st]

    start_time = f"{today} 09:15:00"
    end_time = f"{today} 09:25:00"
    df_auction = get_call_auction(
        stock_list,
        start_date=start_time,
        end_date=end_time,
        fields=['time', 'current', 'volume']
    )

    if df_auction is None or df_auction.empty:
        log.warning("→ 集合竞价数据为空！")
        return []

    df_auction = df_auction.reset_index()
    df_auction['time'] = pd.to_datetime(df_auction['time'])
    df_auction = df_auction.sort_values('time').groupby('code').last().reset_index()

    df_prev = get_price(
        stock_list,
        end_date=yesterday,
        count=1,
        fields=['close'],
        panel=False
    )

    df_merged = pd.merge(df_prev, df_auction[['code', 'current']], on='code', how='inner')
    if df_merged.empty:
        log.warning("→ 竞价与昨日收盘价匹配失败")
        return []

    df_merged['high_limit'] = df_merged['close'] * 1.098
    df_yizi = df_merged.query('current >= high_limit - 0.01')
    yizi_codes = df_yizi['code'].tolist()
    yizi_codes = [s for s in yizi_codes if 300 <= calc_custom_winner(s, yesterday) <= 1000]
    return yizi_codes

# ===========================
# 【无未来数据】9:34 涨停股票（1分钟K线）
# ===========================
def get_934_limit_up_stocks(context):
    log.info("→ 开始获取 09:34 涨停股票（1分钟K线 无未来数据）...")
    today = context.current_dt.date()
    current_data = get_current_data()
    all_stocks = get_all_securities(['stock']).index.tolist()
    valid_stocks = []
    check_time = context.current_dt.replace(hour=9, minute=34, second=0)

    for code in all_stocks:
        if current_data[code].paused or current_data[code].is_st:
            continue

        try:
            bar = get_bars(code, count=1, unit='1m', end_dt=check_time, fields=['high'])
            if bar is None or len(bar) == 0:
                continue

            pre_bar = get_bars(code, count=1, unit='1d', end_dt=today - timedelta(days=1), fields=['close'])
            if pre_bar is None or len(pre_bar) == 0:
                continue
            pre_close = pre_bar['close'].iloc[0]
            high_limit = pre_close * 1.098

            if bar['high'].iloc[0] >= high_limit - 0.01:
                valid_stocks.append(code)
        except Exception:
            continue
    log.info(f"→ 9:34 涨停数量：{len(valid_stocks)}")
    return valid_stocks

# ===========================
# 昨日开盘+收盘均涨停
# ===========================
def check_yesterday_open_close_limit(code, today):
    try:
        df = get_bars(code, count=2, unit='1d', end_dt=today, fields=['open', 'close'])
        if len(df) < 2:
            return False
        pre2_close = df['close'].iloc[0]
        high_limit = pre2_close * 1.098
        yd_open = df['open'].iloc[1]
        yd_close = df['close'].iloc[1]
        if yd_open < high_limit - 0.01:
            return False
        if yd_close < high_limit - 0.01:
            return False

        df3 = get_bars(code, count=3, unit='1d', end_dt=today, fields=['close'])
        if df3.shape[0] <3:
            return False
        if df3['close'].iloc[-1] / df3['close'].iloc[0] > MAX_3D_INCREASE:
            return False
        return True
    except Exception:
        return False

# ===========================
# 6:00 扫描
# ===========================
def morning_6am_scan(context):
    all_stocks = get_all_securities(['stock']).index.tolist()
    current_data = get_current_data()
    valid_stocks = [s for s in all_stocks if not current_data[s].paused and not current_data[s].is_st]
    target_list = []
    for code in valid_stocks:
        if check_yesterday_open_close_limit(code, context.current_dt):
            target_list.append(code)
    g.temp_target_pool = target_list
    log.info(f"✅ 06:00 选出：{len(target_list)} 只")

# ===========================
# 1分钟数据 & 炸板判断
# ===========================
def get_safe_data(code, now):
    try:
        # 使用日线获取昨日收盘价（非分钟线同时刻价格）
        pre_bar = get_bars(code, count=1, unit='1d', end_dt=now - timedelta(days=1), fields=['close'])
        pre_close = pre_bar['close'][0]
        bar = get_bars(code, count=1, unit='1m', end_dt=now, fields=['close', 'high'])
        price = bar['close'][0]
        high  = bar['high'][0]
        return pre_close, price, high
    except Exception as e:
        log.warning(f"get_safe_data({code}) 异常: {e}")
        return None, None, None

def check_zhaban(code, now):
    pre_close, price, high = get_safe_data(code, now)
    if pre_close is None:
        return False, 0.0
    high_limit = pre_close * 1.098
    zhang = (price / pre_close - 1) * 100
    is_zhaban = (high >= high_limit - 0.01) and (price < high_limit - 0.005)
    return is_zhaban, round(zhang, 2)

# ===========================
# 买入逻辑（放在最前，避免NameError）
# ===========================
def scan_buy(context):
    now = context.current_dt
    holds = [s for s in context.portfolio.positions if context.portfolio.positions[s].total_amount > 0]
    if len(holds) >= BUY_MAX_COUNT:
        return

    cash = context.portfolio.available_cash
    need = BUY_MAX_COUNT - len(holds)
    per_cash = cash / need * 0.95 if need > 0 else 0
    current_data = get_current_data()

    # 1. 竞价一字板炸板
    for code in g.yizi_list:
        if len(holds) >= BUY_MAX_COUNT: break
        ok, zhang = check_zhaban(code, now)
        if ok:
            if current_data[code].paused:
                continue
            result = order_value(code, per_cash)
            if result is not None:
                g.buy_dates[code] = now.date()
                log.info(f"✅ [一字炸板] {code} 涨幅{zhang}%")
                holds.append(code)

    # 2. 9:34 涨停炸板
    for code in g.stock_at_934:
        if len(holds) >= BUY_MAX_COUNT: break
        if code in holds: continue
        ok, zhang = check_zhaban(code, now)
        if ok:
            if current_data[code].paused:
                continue
            result = order_value(code, per_cash)
            if result is not None:
                g.buy_dates[code] = now.date()
                log.info(f"✅ [9:34炸板] {code} 涨幅{zhang}%")
                holds.append(code)

    # 3. 昨日双涨停 低开低吸
    for code in g.temp_target_pool:
        if len(holds) >= BUY_MAX_COUNT: break
        if code in holds: continue
        try:
            df = get_bars(code, count=1, unit='1d', end_dt=now, fields=['open'])
            today_open = df['open'].iloc[0]
            now_price = current_data[code].last_price
            if now_price < today_open * TODAY_BUY_PRICE_RATIO:
                if current_data[code].paused:
                    continue
                result = order_value(code, per_cash)
                if result is not None:
                    g.buy_dates[code] = now.date()
                    log.info(f"✅ [低开买入] {code} 开盘{today_open:.2f} 现价{now_price:.2f}")
                    holds.append(code)
        except Exception as e:
            log.warning(f"[低开买入] {code} 异常: {e}")
            continue

# ===========================
# 初始化
# ===========================
def initialize(context):
    set_option('avoid_future_data', True)
    set_option('use_real_price', True)
    set_benchmark('000300.XSHG')
    set_slippage(PriceRelatedSlippage(0.003))
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0001, close_commission=0.0001, min_commission=5), type='stock')

    run_daily(morning_6am_scan, '06:00')
    run_daily(yesterday_profit_stat, '06:05')
    run_daily(prepare_list, '09:26')
    run_daily(prepare_934_list, '09:34')
    run_daily(scan_buy, '09:35')
    run_daily(scan_buy, '09:50')
    run_daily(scan_buy, '10:05')
    run_daily(scan_buy, '10:20')
    run_daily(scan_buy, '10:35')
    run_daily(scan_buy, '11:00')
    register_sell_tasks()

g.yizi_list = []
g.temp_target_pool = []
g.stock_at_934 = []
g.buy_dates = {}

def prepare_list(context):
    g.yizi_list = get_yizi_auction_stocks(context)
    log.info("📅 9:26 一字板：" + str(g.yizi_list))

def prepare_934_list(context):
    g.stock_at_934 = get_934_limit_up_stocks(context)
    log.info("📅 9:34 涨停：" + str(g.stock_at_934))

# ===========================
# 收益统计 & 卖出
# ===========================
def yesterday_profit_stat(context):
    log.info("="*60)
    log.info("📊 06:05 昨日持仓收益统计")
    log.info("="*60)
    
    holds = context.portfolio.positions
    if not holds:
        log.info("📭 昨日无持仓")
        log.info("="*60)
        return
    
    total_profit = 0.0
    for code, pos in holds.items():
        if pos.total_amount <= 0:
            continue
        try:
            name = get_security_info(code).display_name
            cost = pos.avg_cost
            last_close = get_bars(code, count=1, unit='1d', end_dt=context.previous_date, fields=['close'])['close'].iloc[0]
            rise = (last_close / cost - 1) * 100
            profit = (last_close - cost) * pos.total_amount
            total_profit += profit
            log.info(f"📈 {name}({code}) | 成本：{cost:.2f} | 昨收：{last_close:.2f} | 涨幅：{rise:.2f}% | 收益：{profit:.2f}")
        except Exception as e:
            log.warning(f"yesterday_profit_stat({code}) 异常: {e}")
            continue
    
    log.info(f"💰 昨日总收益：{total_profit:.2f} 元")
    log.info("="*60)

PROFIT_TAKE_RATIO = 1.2
STOP_LOSS_MA5_RATIO = 0.986
SELL_1ST_TIMES = ['14:30:00']
SELL_2ND_TIMES = ['11:00:00', '14:50:00']

def register_sell_tasks():
    for t in SELL_1ST_TIMES: run_daily(sell_1st, time=t)
    for t in SELL_2ND_TIMES: run_daily(sell_2nd, time=t)
def sell_1st(context): sell_out(context, 1)
def sell_2nd(context): sell_out(context, 2)

def sell_out(context, times=1):
    current_data = get_current_data()
    hold_codes = list(context.portfolio.positions.keys())
    today = context.current_dt.date()
    
    for code in hold_codes:
        try:
            pos = context.portfolio.positions[code]
            if pos.closeable_amount <= 0:
                continue
            
            px = current_data[code].last_price
            cost = pos.avg_cost
            high_limit = current_data[code].high_limit

            # 涨停不卖
            if px >= high_limit - 0.001:
                log.info(f"🚫 {code} 涨停，锁定不卖")
                continue

            # 止盈: 盈利20%
            if px > cost * PROFIT_TAKE_RATIO:
                order_target(code, 0)
                g.buy_dates.pop(code, None)
                log.info(f"📈 {code} 止盈卖出")
                continue

            # 跌破20日均线 → 清仓
            bars_20 = get_bars(code, count=20, unit='1d', end_dt=context.current_dt, fields=['close'])
            if len(bars_20) >= 20:
                ma20 = bars_20['close'].mean()
                if px < ma20:
                    order_target(code, 0)
                    g.buy_dates.pop(code, None)
                    log.info(f"📉 {code} 跌破MA20卖出 (MA20={ma20:.2f}, px={px:.2f})")
                    continue

            # 持仓超过5个交易日 → 清仓
            if code in g.buy_dates:
                trade_days_count = len(get_trade_days(start_date=g.buy_dates[code], end_date=today))
                hold_days = trade_days_count - 1  # 排除买入当天
                if hold_days >= MAX_HOLDING_DAYS:
                    order_target(code, 0)
                    g.buy_dates.pop(code, None)
                    log.info(f"⏰ {code} 持仓{hold_days}个交易日超限卖出")
                    continue

            # MA5止损 (times==2 only)
            if times == 2:
                bars = get_bars(code, 5, '1m', fields=['close'])
                if len(bars) >= 5:
                    ma5 = bars['close'].mean()
                    if px < ma5 * STOP_LOSS_MA5_RATIO:
                        order_target(code, 0)
                        g.buy_dates.pop(code, None)
                        log.info(f"📉 {code} MA5止损卖出")
                        continue
        except Exception as e:
            log.warning(f"sell_out({code}) 异常: {e}")
            continue

def before_trading_start(context):
    pass