# -*- coding: utf-8 -*-
"""
六一中路（分钟回测 + 实盘风控 + TWAP 优化版）

在原策略基础上做了：
1）情绪周期：保留你现在的情绪指标与分型；
2）大盘风控：用沪深300做破位 + 波动率判断，给出“正常 / 谨慎 / 高风险”；
3）买入优化：日内 TWAP + 对手价-2档限价单 + 单票参与率限制；
4）卖出优化：退潮期清仓 + 回撤止盈 + 原有卖出规则 + 跌停解封卖出。

适配「分钟回测」：
- perpare 调到 09:31（此时 day_open 有值）；
- buy 在 09:32 只生成买入计划；
- minute_main 每分钟执行 TWAP 买入 + 风控 + 跌停解封 + 回撤止盈。
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta

from jqlib.technical_analysis import *
from jqdata import *

# ===================== 情绪周期模块（沿用你现有的） =====================

def get_market_sentiment_indicators(context, check_date=None):
    """
    计算市场情绪核心指标。
    """
    if check_date is None:
        check_date = context.previous_date

    # 1. 获取所有A股，排除 ST、退市、科创板、北交所、新股
    all_stocks = get_all_securities(['stock'], date=check_date)
    all_stocks = all_stocks[all_stocks.start_date < check_date]
    all_stocks = all_stocks[~all_stocks['display_name'].str.contains('ST')]
    all_stocks = all_stocks[~all_stocks['display_name'].str.contains('退')]
    all_stocks = all_stocks[~all_stocks.index.str.startswith('688')]
    all_stocks = all_stocks[~all_stocks.index.str.startswith('8')]
    
    stock_list = list(all_stocks.index)

    # 近两日行情
    df_prices = get_price(
        stock_list, end_date=check_date, count=2, frequency='daily',
        fields=['open', 'close', 'high', 'low', 'high_limit', 'low_limit', 'volume', 'money'],
        skip_paused=False, fq='pre', panel=False
    )
    
    if df_prices.empty:
        return {
            '最高连板高度': 0,
            '涨停家数': 0,
            '昨日涨停表现': 0.0,
            '炸板率': 0.0,
            '跌停家数': 0
        }

    today_data = df_prices[df_prices.time.dt.date == pd.to_datetime(check_date).date()].set_index('code')
    today_data = today_data[today_data['volume'] > 0]

    yesterday_data = df_prices[df_prices.time.dt.date < pd.to_datetime(check_date).date()]
    if not yesterday_data.empty:
        yesterday_data = yesterday_data.set_index('code')
    
    # 涨停 / 跌停家数
    limit_up_stocks = today_data[today_data['close'] >= today_data['high_limit']]
    limit_up_count = len(limit_up_stocks)

    limit_down_stocks = today_data[today_data['close'] <= today_data['low_limit']]
    limit_down_count = len(limit_down_stocks)

    # 炸板率
    touched_limit_up = today_data[today_data['high'] >= today_data['high_limit']]
    blown_board_stocks = touched_limit_up[touched_limit_up['close'] < touched_limit_up['high_limit']]
    blown_board_rate = len(blown_board_stocks) / len(touched_limit_up) if len(touched_limit_up) > 0 else 0.0

    # 昨涨停今日表现
    yesterday_limit_up_performance = 0.0
    if not yesterday_data.empty:
        yesterday_limit_up_stocks = yesterday_data[yesterday_data['close'] >= yesterday_data['high_limit']]
        today_performance_stocks = today_data.loc[today_data.index.isin(yesterday_limit_up_stocks.index)]
        if not today_performance_stocks.empty:
            yesterday_closes = yesterday_data.loc[today_performance_stocks.index]['close']
            today_opens = today_performance_stocks['open']
            performance = (today_opens - yesterday_closes) / yesterday_closes
            yesterday_limit_up_performance = performance.mean()

    # 最高连板高度（近 15 日）
    highest_board = 0
    if not limit_up_stocks.empty:
        limit_up_list = list(limit_up_stocks.index)
        history_prices_df = get_price(
            security=limit_up_list, count=15, end_date=check_date, frequency='1d',
            fields=['close', 'high_limit'], skip_paused=True, panel=False, fq='pre'
        )
        
        for stock in limit_up_list:
            current_board = 1
            stock_history = history_prices_df[history_prices_df['code'] == stock].sort_values('time', ascending=False)
            for idx, row in stock_history.iloc[1:].iterrows():
                if row['close'] >= row['high_limit']:
                    current_board += 1
                else:
                    break
            highest_board = max(highest_board, current_board)

    return {
        '最高连板高度': highest_board,
        '涨停家数': limit_up_count,
        '昨日涨停表现': yesterday_limit_up_performance,
        '炸板率': blown_board_rate,
        '跌停家数': limit_down_count
    }


def determine_market_phase(indicators):
    """根据情绪指标判断市场阶段（沿用你原来的分型）"""
    if indicators['炸板率'] > 0.40 and indicators['跌停家数'] > 10:
        return '退潮期'
    if indicators['最高连板高度'] <= 3 and indicators['跌停家数'] > indicators['涨停家数']:
        return '冰点期'
    if indicators['炸板率'] > 0.35 or (indicators['跌停家数'] > 5 and indicators['最高连板高度'] > 6):
        return '高潮期'
    if indicators['最高连板高度'] >= 5 and indicators['昨日涨停表现'] > 0.02:
        return '发酵期'
    if indicators['最高连板高度'] > 3 and indicators['涨停家数'] < 80 and indicators['昨日涨停表现'] > 0.01:
        return '启动期'
    return '震荡期'


# ===================== 大盘风险 + 实盘回撤风控 =====================

def update_index_risk(context):
    """
    用沪深300做大盘风控，综合 MA20 / 回撤 / 波动率：
    - 正常 / 谨慎 / 高风险
    """
    idx = g.index
    hist = get_price(idx,
                     end_date=context.previous_date,
                     frequency='daily',
                     fields=['close'],
                     count=30,
                     panel=False)
    if hist is None or hist.empty:
        g.index_risk = '正常'
        return

    closes = hist['close'].values
    last = closes[-1]
    window = closes[-20:] if len(closes) >= 20 else closes
    ma20 = window.mean()
    peak = closes.max()
    drawdown = last / peak - 1
    rets = np.diff(closes) / closes[:-1]
    vol = rets.std() if len(rets) > 1 else 0

    if (last < ma20 * 0.99) and (drawdown < -0.05) and (vol > 0.02):
        g.index_risk = '高风险'
    elif (last < ma20) and (drawdown < -0.03):
        g.index_risk = '谨慎'
    else:
        g.index_risk = '正常'


def check_intraday_risk(context):
    """
    实盘风控：
    - 当日相对昨收收益 < max_daily_loss_pct -> 停止买入
    - 当日从日内高点回撤 < max_intraday_drawdown_pct -> 强制减仓
    """
    if g.yesterday_total_value is None:
        return

    total_value = context.portfolio.total_value
    if g.daily_max_value is None or total_value > g.daily_max_value:
        g.daily_max_value = total_value

    daily_ret = total_value / g.yesterday_total_value - 1
    dd_from_high = total_value / g.daily_max_value - 1

    if (not g.intraday_trading_blocked) and (daily_ret <= g.max_daily_loss_pct):
        g.intraday_trading_blocked = True
        log.warn("【实盘风控】当日收益 {:.2%} <= 阈值 {:.2%}，今日停止买入。"
                 .format(daily_ret, g.max_daily_loss_pct))

    if dd_from_high <= g.max_intraday_drawdown_pct:
        log.warn("【实盘风控】当日从高点回撤 {:.2%} <= 阈值 {:.2%}，执行强制减仓。"
                 .format(dd_from_high, g.max_intraday_drawdown_pct))
        for s, pos in list(context.portfolio.positions.items()):
            if pos.closeable_amount > 0:
                order_target_value(s, 0)


# ===================== 原六一中路的选股相关函数 =====================

def today_is_between(context):
    today = context.current_dt.strftime('%m-%d')
    return ('01-15' <= today <= '01-31') or \
           ('04-15' <= today <= '04-30') or \
           ('12-15' <= today <= '12-31')


def get_st(context):
    stocks = get_index_stocks('399101.XSHE', date=context.previous_date)
    st_data = get_extras('is_st', stocks, count=1, end_date=context.previous_date).T
    st_data.columns = ['is_st']
    return st_data[st_data['is_st'] == False].index.tolist()


def transform_date(date, date_type):
    if isinstance(date, str):
        dt_date = dt.datetime.strptime(date, '%Y-%m-%d')
    elif isinstance(date, dt.datetime):
        dt_date = date
    elif isinstance(date, dt.date):
        dt_date = dt.datetime.combine(date, dt.time())
    else:
        dt_date = date
    dct = {'str': dt_date.strftime('%Y-%m-%d'), 'dt': dt_date, 'd': dt_date.date()}
    return dct[date_type]


def get_shifted_date(date, days, days_type='T'):
    d_date = transform_date(date, 'd')
    yesterday = d_date + dt.timedelta(-1)
    if days_type == 'N':
        return str(yesterday + dt.timedelta(days + 1))
    if days_type == 'T':
        all_trade_days = [i.strftime('%Y-%m-%d') for i in list(get_all_trade_days())]
        if str(yesterday) in all_trade_days:
            return all_trade_days[all_trade_days.index(str(yesterday)) + days + 1]
        else:
            for i in range(100):
                last_trade_date = yesterday - dt.timedelta(i)
                if str(last_trade_date) in all_trade_days:
                    return all_trade_days[all_trade_days.index(str(last_trade_date)) + days + 1]
    return None


def get_ever_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily',
                   fields=['close', 'high', 'high_limit'],
                   count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()
    return list(df[df['close'] != df['high_limit']].code)


def get_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily',
                   fields=['close', 'low', 'high_limit'],
                   count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()
    return list(df[df['close'] == df['high_limit']].code)


def rzq_list(context, initial_list):
    date = transform_date(context.previous_date, 'str')
    date_1 = get_shifted_date(date, -1, 'T')
    h1_list = get_ever_hl_stock(initial_list, date)
    elements_to_remove = get_hl_stock(initial_list, date_1)
    return [stock for stock in h1_list if stock in elements_to_remove]


def filter_stocks(context, stocks):
    yesterday = context.previous_date
    df = get_price(stocks, count=11, frequency='1d',
                   fields=['close', 'low', 'volume'],
                   end_date=yesterday, panel=False).reset_index()
    valid_stocks = []
    for code, group in df.groupby('code'):
        if len(group) < 11:
            continue
        group = group.copy()
        group['ma10'] = group['close'].rolling(10).mean()
        group['prev_low'] = group['low'].shift(1)
        group['prev_volume'] = group['volume'].shift(1)
        last_row = group.iloc[-1]
        if (not pd.isna(last_row[['ma10', 'prev_low', 'prev_volume']]).any() and
            last_row['close'] > last_row['prev_low'] and
            last_row['close'] > last_row['ma10'] and
            last_row['volume'] > last_row['prev_volume'] and
            last_row['volume'] < 10 * last_row['prev_volume'] and
            last_row['close'] > 1):
            valid_stocks.append(code)
    return valid_stocks


def GJT_filter_stocks(stocks):
    q = query(
        valuation.code, income.np_parent_company_owners, income.net_profit,
        income.operating_revenue
    ).filter(
        valuation.code.in_(stocks),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8,
        indicator.roe > 0,
        indicator.roa > 0
    )
    return list(get_fundamentals(q).code)


def filter_stocks_by_b_s(context, stock_list):
    """
    竞价多空资金过滤：保留 b_s > 0 的标的。
    """
    date = context.current_dt.strftime("%Y-%m-%d")
    valid_stocks = []
    for stock in stock_list:
        auction_df = get_call_auction(stock, start_date=date, end_date=date)
        if auction_df is None or auction_df.empty:
            continue
        auction_df = auction_df.assign(
            sellmoney=lambda df: df['a1_p']*df['a1_v'] + df['a2_p']*df['a2_v'] +
                                 df['a3_p']*df['a3_v'] + df['a4_p']*df['a4_v'] +
                                 df['a5_p']*df['a5_v'],
            buymoney=lambda df: df['b1_p']*df['b1_v'] + df['b2_p']*df['b2_v'] +
                                df['b3_p']*df['b3_v'] + df['b4_p']*df['b4_v'] +
                                df['b5_p']*df['b5_v']
        ).assign(b_s=lambda df: (df['buymoney'] - df['sellmoney']) / df['sellmoney'])
        if not auction_df.empty and auction_df['b_s'].iloc[0] > 0:
            valid_stocks.append(stock)
    return valid_stocks


# ===================== 卖出：原逻辑 + 回撤止盈 =====================

def update_trailing_stop_and_sell(context):
    """
    回撤止盈：
    - 当持仓曾盈利 >= sell_trailing_start（10%）
    - 且从高点回撤 >= sell_trailing_back（6%）
    -> 限价稍低于现价卖出。
    """
    current_data = get_current_data()
    for code, pos in list(context.portfolio.positions.items()):
        if pos.total_amount <= 0:
            g.position_high.pop(code, None)
            continue

        last = current_data[code].last_price
        prev_high = g.position_high.get(code, pos.avg_cost)
        if last > prev_high:
            g.position_high[code] = last
            continue

        high = prev_high
        high_profit = high / pos.avg_cost - 1
        if high_profit < g.sell_trailing_start:
            continue

        pullback = last / high - 1
        if pullback <= -g.sell_trailing_back and pos.closeable_amount > 0 and (not current_data[code].paused):
            limit_price = max(last * 0.995, current_data[code].low_limit)
            order = order_target_value(code, 0, style=LimitOrderStyle(limit_price))
            if order and order.filled > 0:
                g.today_sold_stocks.add(code)
                log.info("【回撤止盈】{} 高位回落 {:.2%} 卖出 {} 股"
                         .format(code, pullback, order.filled))
            g.position_high.pop(code, None)


def dieting(context):
    """
    跌停监控 + 解封卖出（原逻辑）
    """
    current_data = get_current_data()
    for s in list(context.portfolio.positions):
        if s not in g.dieting:
            dtj = current_data[s].low_limit
            zxj = current_data[s].last_price
            if zxj == dtj and (context.portfolio.positions[s].closeable_amount != 0):
                if s not in g.dieting:
                    g.dieting.append(s)
    g.dieting = list(set(g.dieting))
    if len(g.dieting) > 0:
        for s in g.dieting[:]:
            dtj = current_data[s].low_limit
            zxj = current_data[s].last_price
            if zxj > dtj:
                pos = context.portfolio.positions[s]
                closeable_amount = pos.closeable_amount
                if closeable_amount <= 0:
                    log.warn(f'跌停打开跳过卖出 {s}: 可卖数量不足: {closeable_amount}')
                    g.dieting.remove(s)
                    continue
                if current_data[s].paused:
                    log.warn(f'跌停打开跳过卖出 {s}: 股票停牌')
                    continue
                order = order_target_value(s, 0)
                if order and order.filled > 0:
                    g.today_sold_stocks.add(s)
                    log.info(f"跌停打开卖出:{s} 成交:{order.filled}股")
                else:
                    log.warn(f"跌停打开卖出失败:{s} 可卖:{closeable_amount}股")
                g.dieting.remove(s)


def sell(context):
    """
    卖出逻辑：
    - 退潮期：无条件清仓；
    - 其他：按原六一中路的均线 + 盈利 + 昨涨停卖出逻辑。
    """
    # 退潮期清仓
    if g.market_phase == '退潮期':
        log.info(f"情绪周期为，执行清仓操作。")
        for s in list(context.portfolio.positions.keys()):
            pos = context.portfolio.positions[s]
            if pos.closeable_amount > 0:
                order = order_target_value(s, 0)
                if order and order.filled > 0:
                    g.today_sold_stocks.add(s)
        return

    hold_list = list(context.portfolio.positions)
    if not hold_list:
        return

    current_data = get_current_data()
    yesterday = context.previous_date

    sellable_list = []
    for s in hold_list:
        pos = context.portfolio.positions[s]
        if s not in g.today_bought_stocks and pos.closeable_amount > 0:
            sellable_list.append(s)

    if not sellable_list:
        log.info('没有可卖出的股票（全部为当日买入）')
        return

    # MA7
    hist_data = get_price(
        sellable_list, end_date=yesterday, frequency='daily',
        fields=['close'], count=8, panel=False
    )
    ma7_data = hist_data.groupby('code')['close'].apply(
        lambda x: x.rolling(7).mean().iloc[-1]
    ).to_dict()

    df_history = get_price(
        sellable_list, end_date=yesterday, frequency='daily',
        fields=['close', 'high_limit'], count=1, panel=False
    )

    df_history['avg_cost'] = [context.portfolio.positions[s].avg_cost for s in sellable_list]
    df_history['price'] = [context.portfolio.positions[s].price for s in sellable_list]
    df_history['high_limit'] = [current_data[s].high_limit for s in sellable_list]
    df_history['low_limit'] = [current_data[s].low_limit for s in sellable_list]
    df_history['last_price'] = [current_data[s].last_price for s in sellable_list]
    df_history['ma7'] = [ma7_data.get(s, 0) for s in sellable_list]
    df_history['closeable_amount'] = [context.portfolio.positions[s].closeable_amount for s in sellable_list]

    cond1 = (df_history['last_price'] != df_history['high_limit'])
    cond2_1 = df_history['last_price'] < df_history['ma7']
    ret_matrix = (df_history['price'] / df_history['avg_cost'] - 1) * 100
    cond2_2 = ret_matrix > 0
    cond2_3 = (df_history['close'] == df_history['high_limit'])  # 昨涨停

    sell_condition = cond1 & (cond2_1 | cond2_2 | cond2_3)

    sell_list = df_history[
        sell_condition &
        (df_history['last_price'] > df_history['low_limit']) &
        (df_history['closeable_amount'] > 0)
    ].code.tolist()

    for s in sell_list:
        pos = context.portfolio.positions[s]
        if pos.closeable_amount <= 0 or current_data[s].paused:
            continue
        order = order_target_value(s, 0)
        if order and order.filled > 0:
            g.today_sold_stocks.add(s)
            log.info(f'卖出 {s} | 成本价:{pos.avg_cost:.2f} 现价:{pos.price:.2f} 可卖:{pos.closeable_amount}股 成交:{order.filled}股')
        else:
            log.warn(f'卖出失败 {s} | 成本价:{pos.avg_cost:.2f} 现价:{pos.price:.2f} 可卖:{pos.closeable_amount}股')
        print('-'*50)


# ===================== 日内 TWAP 买入执行 =====================

def execute_buy_plans(context):
    """
    在指定时间区间（09:35-09:55）按 TWAP 思路分批执行买入计划：
    - 每分钟根据 target_value * 已过比例 计算应有持仓金额；
    - 与已成交金额差额 > min_slice_value 时，下限价单补齐；
    - 限价 = last_price * (1 + g.buy_limit_offset)，近似“对手价 - 2 档”。
    """
    if not g.buy_plans:
        return

    # 大盘高风险直接不买
    if g.index_risk == '高风险':
        log.info("【大盘风控】指数风险为【高风险】，暂停执行所有买入计划。")
        return

    # 当日亏损过大也不买
    if g.intraday_trading_blocked:
        log.info("【实盘风控】今日已触发亏损阈值，暂停执行所有买入计划。")
        return

    now = context.current_dt.time()
    if now < g.twap_start or now > g.twap_end:
        return

    total_minutes = int(
        (dt.datetime.combine(dt.date(2000, 1, 1), g.twap_end) -
         dt.datetime.combine(dt.date(2000, 1, 1), g.twap_start)
         ).seconds / 60
    )
    elapsed_minutes = int(
        (dt.datetime.combine(dt.date(2000, 1, 1), now) -
         dt.datetime.combine(dt.date(2000, 1, 1), g.twap_start)
         ).seconds / 60
    ) + 1
    elapsed_minutes = max(1, min(elapsed_minutes, total_minutes))

    current_data = get_current_data()

    for stock, plan in list(g.buy_plans.items()):
        if stock not in current_data:
            continue
        d = current_data[stock]
        if d.paused or d.last_price in (d.low_limit, d.high_limit):
            continue

        target_value = plan['target_value']
        executed_value = plan['executed_value']

        should_value = target_value * (elapsed_minutes / float(total_minutes))
        delta = should_value - executed_value

        if delta < g.min_slice_value:
            continue

        delta = min(delta, context.portfolio.available_cash)
        if delta < g.min_slice_value:
            continue

        limit_price = d.last_price * (1 + g.buy_limit_offset)
        limit_price = min(max(limit_price, d.low_limit), d.high_limit)

        order = order_value(stock, delta, style=LimitOrderStyle(limit_price))
        if order and order.filled > 0:
            filled_value = order.filled * limit_price
            plan['executed_value'] += filled_value
            g.today_bought_stocks.add(stock)
            log.info("【TWAP买入】{} 本次金额:{:.0f} 累计:{:.0f}/{:.0f} 限价:{:.2f}"
                     .format(stock, filled_value,
                             plan['executed_value'], target_value,
                             limit_price))

        if plan['executed_value'] >= target_value * 0.98:
            log.info("【TWAP完成】{} 计划金额 {:.0f} 实际成交 {:.0f}"
                     .format(stock, target_value, plan['executed_value']))
            del g.buy_plans[stock]


def minute_main(context):
    """
    分钟主循环：
    1）更新实盘风控（当日回撤）；
    2）跌停解封卖出；
    3）回撤止盈；
    4）执行 TWAP 买入计划。
    """
    check_intraday_risk(context)
    dieting(context)
    update_trailing_stop_and_sell(context)
    execute_buy_plans(context)


# ===================== 买入计划生成（替代原 buy） =====================

def buy(context):
    """
    不在这里直接下单，改为：
    - 用原有「六一中路」逻辑选股；
    - 按持仓上限和资金均分 + 参与率限制，生成 g.buy_plans；
    - 具体买入由 minute_main -> execute_buy_plans 日内分批完成。
    """
    # 情绪退潮期不做多
    if g.market_phase == '退潮期':
        log.info(f"情绪周期为，今日不生成买入计划。")
        return

    target = filter_stocks_by_b_s(context, g.today_list)
    hold_list = list(context.portfolio.positions)

    # 冰点期只试错 1 只
    if g.market_phase == '冰点期':
        log.info("情绪周期为【冰点期】，执行小仓位试错，最多持仓1只。")
        adjusted_stock_num = 1
    else:
        adjusted_stock_num = g.stock_num

    num = adjusted_stock_num - len(hold_list)
    if num <= 0:
        log.info("持仓已满，不生成新的买入计划。")
        return

    target = [x for x in target if x not in hold_list][:num]
    if len(target) == 0:
        log.info("没有可用标的，不生成买入计划。")
        return

    value = context.portfolio.available_cash
    if value <= 0:
        log.info("可用现金为 0，不生成买入计划。")
        return

    cash_per_stock = value / len(target)
    current_data = get_current_data()

    for stock in target:
        d = current_data[stock]
        if d.paused or d.last_price in (d.low_limit, d.high_limit):
            continue

        # 参与率限制：单只股票当日计划买入金额 <= 昨日成交额 * max_participation_rate
        hist = get_price(stock,
                         end_date=context.previous_date,
                         frequency='daily',
                         fields=['money'],
                         count=1,
                         panel=False)
        if hist is not None and not hist.empty:
            y_money = hist['money'].iloc[0]
            cap_value = y_money * g.max_participation_rate
        else:
            cap_value = cash_per_stock

        plan_value = min(cash_per_stock, cap_value)
        if plan_value < g.min_slice_value:
            continue

        g.buy_plans[stock] = {
            'target_value': plan_value,
            'executed_value': 0.0
        }
        log.info("【买入计划生成】{} 目标金额:{:.0f}".format(stock, plan_value))

    if not g.buy_plans:
        log.info("未生成任何买入计划。")


# ===================== 盘前选股（perpare 调整时间到 09:31） =====================

def perpare(context):
    """
    盘前 / 开盘后立即的选股：
    - 沿用原六一中路 get_st -> GJT_filter -> filter_stocks -> rzq_list；
    - 再用昨收 + day_open 限定开盘涨幅区间；
    - 再按换手率 * 高开因子排序。
    """
    g.dieting = []
    g.yesterday_high_dict = {}
    g.today_list = []
    g.today_bought_stocks = set()
    g.today_sold_stocks = set()
    g.buy_plans = {}

    current_data = get_current_data()

    stk_list = get_st(context)
    singal = today_is_between(context)
    if singal:
        log.info(f'筛选前：{len(stk_list)}')
        stk_list = GJT_filter_stocks(stk_list)
        log.info(f'筛选后：{len(stk_list)}')

    stk_list = filter_stocks(context, stk_list)
    if len(stk_list) == 0:
        return
    stk_list = rzq_list(context, stk_list)
    if len(stk_list) == 0:
        return

    df = get_price(
        stk_list, end_date=context.previous_date, frequency='daily',
        fields=['close'], count=1, panel=False, fill_paused=False, skip_paused=True
    ).set_index('code')

    # 在分钟回测下，09:31 以后 day_open 已经有值
    open_now_values = []
    for s in stk_list:
        try:
            open_now_values.append(current_data[s].day_open)
        except KeyError as e:
            log.warn(f"警告: 股票 {s} 的数据不可用, 错误: {e}")
            open_now_values.append(None)

    df['open_now'] = open_now_values
    df = df.dropna(subset=['open_now'])
    df = df[(df['open_now'] / df['close']) < 1.01]
    df = df[(df['open_now'] / df['close']) > 0.95]

    stk_list = list(df.index)
    hold_list = list(context.portfolio.positions)
    stk_list = list(set(stk_list) - set(hold_list))

    if len(stk_list) == 0:
        return

    df_val = get_valuation(
        stk_list, start_date=context.previous_date, end_date=context.previous_date,
        fields=['turnover_ratio', 'market_cap', 'circulating_market_cap']
    )

    df.index = df.index.astype(str)
    df_val['code'] = df_val['code'].astype(str)
    df_combined = pd.merge(df.reset_index(), df_val, on='code')
    df_combined['factor'] = df_combined['turnover_ratio'] * (df_combined['open_now'] / df_combined['close'])
    df_sorted = df_combined.sort_values(by='factor', ascending=False)

    g.today_list = list(df_sorted['code'])
    log.info(f"股池数: {len(g.today_list)}")


# ===================== 盘前情绪 & 报告 =====================

def before_trading_start(context):
    """
    - 计算情绪指标 & 分型；
    - 初始化当日实盘风控基准；
    - 计算大盘风险等级。
    """
    g.sentiment_indicators = get_market_sentiment_indicators(context)
    g.market_phase = determine_market_phase(g.sentiment_indicators)

    g.yesterday_total_value = context.portfolio.total_value
    g.daily_max_value = g.yesterday_total_value
    g.intraday_trading_blocked = False

    update_index_risk(context)

    log.info(f"--- 市场情绪判断 | 大盘风险 ---")
    log.info(f"最高连板: {g.sentiment_indicators['最高连板高度']}板, 涨停: {g.sentiment_indicators['涨停家数']}家, 跌停: {g.sentiment_indicators['跌停家数']}家")
    log.info(f"昨日涨停表现: {g.sentiment_indicators['昨日涨停表现']:.2%}, 炸板率: {g.sentiment_indicators['炸板率']:.2%}")
    log.info("-------------------------------------------------")


def daily_trading_report(context):
    """
    收盘后打印当日交易情况 & 持仓情况。
    """
    try:
        current_date = context.current_dt.strftime('%Y-%m-%d')
        total_value = context.portfolio.total_value
        cash = context.portfolio.cash
        
        daily_return = total_value - g.last_total_value if g.last_total_value > 0 else 0
        daily_return_pct = (daily_return / g.last_total_value * 100) if g.last_total_value > 0 else 0
        g.last_total_value = total_value
        
        log.info(f"📊【{current_date} 交易报告】| 情绪周期: | 大盘风险:")
        log.info(f"💰 总资产: {total_value:,.2f}元 | 现金: {cash:,.2f}元")
        log.info(f"📈 当日收益: {daily_return:+,.2f}元 ({daily_return_pct:+.2f}%)")
        
        log.info(f"📈【当日买入】{len(g.today_bought_stocks)}只: {', '.join(g.today_bought_stocks) or '无'}")
        log.info(f"📉【当日卖出】{len(g.today_sold_stocks)}只: {', '.join(g.today_sold_stocks) or '无'}")
        
        positions_info = []
        for code, pos in context.portfolio.positions.items():
            if pos.total_amount > 0:
                stock_name = get_security_info(code).display_name
                profit_loss = (pos.price - pos.avg_cost) * pos.total_amount
                profit_loss_pct = (pos.price / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else 0
                positions_info.append(
                    f"  {stock_name}({code}): {pos.total_amount}股 | 成本:{pos.avg_cost:.2f} | "
                    f"现价:{pos.price:.2f} | 市值:{pos.value:,.2f}元 | "
                    f"盈亏:{profit_loss:+,.2f}元 ({profit_loss_pct:+.2f}%)"
                )
        
        if positions_info:
            log.info(f"📋【持仓详情】共{len(positions_info)}只股票")
            for info in positions_info:
                log.info(info)
        else:
            log.info("📋【持仓详情】当前无持仓")
        
        if g.dieting:
            log.info(f"⚠️【跌停监控】{len(g.dieting)}只股票跌停: {', '.join(g.dieting)}")
        else:
            log.info("✅【跌停监控】无跌停股票")
        
        log.info("=" * 80)
        
    except Exception as e:
        log.error(f"生成每日交易报告失败: {e}")


# ===================== 初始化 =====================

def initialize(context):
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_slippage(FixedSlippage(0.0001))
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.0005,
            open_commission=0.0001,
            close_commission=0.0001,
            close_today_commission=0,
            min_commission=1
        ),
        type='stock'
    )
    # 基本参数
    g.stock_num = 4
    g.today_list = []
    g.down = 0.4
    g.today_bought_stocks = set()
    g.today_sold_stocks = set()
    g.last_total_value = 0
    g.dieting = []

    # 情绪周期
    g.sentiment_indicators = {}
    g.market_phase = '未知'

    # 大盘风控
    g.index = '000300.XSHG'
    g.index_risk = '正常'
    g.max_daily_loss_pct = -0.03
    g.max_intraday_drawdown_pct = -0.06
    g.yesterday_total_value = None
    g.daily_max_value = None
    g.intraday_trading_blocked = False

    # 日内 TWAP 买入计划
    g.buy_plans = {}
    g.twap_start = dt.time(9, 35)
    g.twap_end = dt.time(9, 55)
    g.buy_limit_offset = -0.002     # 对手价 - 0.2%
    g.min_slice_value = 2000        # 单次下单金额下限
    g.max_participation_rate = 0.05 # 单票参与率 <= 5%

    # 卖出：回撤止盈参数
    g.position_high = {}
    g.sell_trailing_start = 0.10    # 盈利 10% 开始启用回撤止盈
    g.sell_trailing_back = 0.06     # 从高点回撤 6% 卖出

    # 调度：注意 perpare / buy 时间后移，适配分钟回测
    run_daily(before_trading_start, time="09:00")
    run_daily(perpare,              time="09:31")
    run_daily(buy,                  time="09:32")
    run_daily(sell,                 time="13:00")
    run_daily(sell,                 time="14:55")
    run_daily(minute_main,          time="every_bar")
    run_daily(daily_trading_report, time="15:05")

    log.set_level('order', 'warning')
    log.set_level('system', 'warning')
