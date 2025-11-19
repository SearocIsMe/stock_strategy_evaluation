# coding: gbk


import datetime
import traceback

# ====== 基本配置：资金账号 & 账号类型（建议后续改成策略参数） ======
ACCOUNT_ID   = 'sf'   # 你的资金账号
ACCOUNT_TYPE = 'stock'        # 文档示例里用的是小写 'stock'


# ---------- 简单日志封装 ----------
def log_info(msg):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('[QMT-AccSummary {}] {}'.format(now, msg))


# ---------- 通用工具 ----------
def pick_attr(obj, candidates, default=None):
    """
    在 obj 上按候选字段名列表依次尝试 getattr，返回第一个存在的
    例如 pick_attr(o, ['m_strInstrumentID', 'm_strCode'], '')
    """
    for name in candidates:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


# ========== 1. 账户资金 / 资产信息 ==========
def fetch_account_info():
    """
    使用 get_trade_detail_data(..., 'account') 获取资金账号信息
    返回：第一个账户对象（通常只有一个），或者 None
    """
    try:
        lst = get_trade_detail_data(ACCOUNT_ID, ACCOUNT_TYPE, 'account')
    except Exception:
        log_info('💥 获取账户资金信息失败：\n{}'.format(traceback.format_exc()))
        return None

    if not lst:
        log_info('⚠️ get_trade_detail_data(account) 返回为空')
        return None

    acct = lst[0]
    # 打印一次 dir(acct)，方便你以后自己看有哪些字段可用
    log_info('账户对象字段示例：{}'.format([x for x in dir(acct) if x.startswith("m_")][:15]))

    # 常见字段名（按你的实际字段名微调即可）
    total_asset = pick_attr(acct, ['m_dTotalAsset', 'm_dAsset', 'm_dNetAsset'], 0.0)
    available   = pick_attr(acct, ['m_dAvailable', 'm_dAvailableCash'], 0.0)
    frozen      = pick_attr(acct, ['m_dFrozenCash', 'm_dFrozen'], 0.0)

    log_info('资金汇总：总资产={:.2f} 可用资金={:.2f} 冻结资金={:.2f}'.format(
        float(total_asset), float(available), float(frozen))
    )

    return {
        'raw': acct,
        'total_asset': float(total_asset),
        'available': float(available),
        'frozen': float(frozen),
    }


# ========== 2. 当前持仓 + 对账户的贡献 ==========
def fetch_positions(acct_total_asset):
    """
    用 get_trade_detail_data(..., 'position') 获取当前持仓列表，
    计算每只股票的市值以及占整个账户总资产的比例。
    """
    try:
        pos_list = get_trade_detail_data(ACCOUNT_ID, ACCOUNT_TYPE, 'position')
    except Exception:
        log_info('💥 获取持仓信息失败：\n{}'.format(traceback.format_exc()))
        return []

    if not pos_list:
        log_info('当前无持仓（position 列表为空）')
        return []

    results = []
    total_mv = 0.0

    for pos in pos_list:
        # 代码 / 名称
        code = pick_attr(pos, ['m_strInstrumentID', 'm_strCode', 'm_strStockCode'], '')
        name = pick_attr(pos, ['m_strInstrumentName', 'm_strName'], '')

        # 持仓数量 & 成本价
        volume = pick_attr(pos, ['m_nVolume', 'm_nPosition', 'm_nTodayPosition', 'm_nAmount'], 0)
        cost   = pick_attr(pos, ['m_dAvgPrice', 'm_dOpenPrice', 'm_dCostPrice'], 0.0)

        try:
            volume = float(volume)
        except Exception:
            volume = 0.0

        # 当前市值：优先用持仓结构里自带的市值字段，其次用最新价*数量
        mv = pick_attr(pos, ['m_dMarketValue', 'm_dPositionValue'], None)

        if mv is None:
            # 没有现成市值字段，用最新行情算
            try:
                md = get_market_data_ex(code, subscribe=False)   # 文档推荐用法：只取本地数据
                last_px = pick_attr(md, ['m_dLastPrice', 'm_lastPrice', 'last_price'], 0.0)
            except Exception:
                last_px = 0.0
            mv = volume * float(last_px)
        else:
            mv = float(mv)

        total_mv += mv
        results.append({
            'code': code,
            'name': name,
            'volume': volume,
            'cost_price': float(cost),
            'market_value': mv,
        })

    base = acct_total_asset if acct_total_asset > 0 else total_mv

    log_info('持仓总市值 = {:.2f}，用于计算权重的基数 = {:.2f}'.format(total_mv, base))

    # 计算权重并打印
    for r in sorted(results, key=lambda x: x['market_value'], reverse=True):
        weight = (r['market_value'] / base * 100) if base > 0 else 0.0
        r['weight'] = weight
        log_info(
            '持仓: {name}({code}) 数量={vol:.0f} 成本价={cost:.3f} '
            '市值={mv:.2f} 占账户={w:.2f}%'.format(
                name=r['name'],
                code=r['code'],
                vol=r['volume'],
                cost=r['cost_price'],
                mv=r['market_value'],
                w=weight
            )
        )

    return results


# ========== 3. 当日订单执行统计 ==========
def fetch_orders_and_deals():
    """
    通过：
      - get_trade_detail_data(..., 'order') 获取当日委托
      - get_trade_detail_data(..., 'deal')  获取当日成交
    做三层统计：
      1）逐笔委托的执行情况（原来已有）
      2）全日总体执行统计
      3）按股票维度的执行统计
    """
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    # ========= 1. 成交明细，按 order_id 聚合 =========
    try:
        deal_list = get_trade_detail_data(ACCOUNT_ID, ACCOUNT_TYPE, 'deal')
    except Exception:
        log_info('💥 获取成交明细失败：\n{}'.format(traceback.format_exc()))
        deal_list = []

    deal_agg = {}
    for d in deal_list:
        order_id = pick_attr(d, ['m_strOrderID', 'm_strOrderId', 'm_order_id'], '')
        if not order_id:
            continue

        deal_time = str(pick_attr(d, ['m_strTradeTime', 'm_strDealTime', 'm_trade_time'], ''))
        deal_date = deal_time[:10]
        if deal_date != today:
            continue  # 只统计当日

        code   = pick_attr(d, ['m_strInstrumentID', 'm_strCode'], '')
        bs_flag = pick_attr(d, ['m_nDirection', 'm_nTradeType', 'm_nOffsetFlag'], 0)
        price  = pick_attr(d, ['m_dTradePrice', 'm_dPrice'], 0.0)
        vol    = pick_attr(d, ['m_nTradeVolume', 'm_nVolume'], 0)

        try:
            vol = float(vol)
            price = float(price)
        except Exception:
            vol = 0.0
            price = 0.0

        agg = deal_agg.setdefault(order_id, {
            'order_id': order_id,
            'code': code,
            'bs_flag': bs_flag,
            'trade_volume': 0.0,
            'trade_amount': 0.0,
        })
        agg['trade_volume'] += vol
        agg['trade_amount'] += vol * price

    for agg in deal_agg.values():
        if agg['trade_volume'] > 0:
            agg['trade_price'] = agg['trade_amount'] / agg['trade_volume']
        else:
            agg['trade_price'] = 0.0

    # ========= 2. 委托明细，合并成交信息，形成逐笔 summary =========
    try:
        order_list = get_trade_detail_data(ACCOUNT_ID, ACCOUNT_TYPE, 'order')
    except Exception:
        log_info('💥 获取委托明细失败：\n{}'.format(traceback.format_exc()))
        order_list = []

    summary = {}
    for o in order_list:
        order_id = pick_attr(o, ['m_strOrderID', 'm_strOrderId', 'm_order_id'], '')
        if not order_id:
            continue

        order_time = str(pick_attr(o, ['m_strOrderTime', 'm_order_time'], ''))
        order_date = order_time[:10]
        if order_date != today:
            continue

        code   = pick_attr(o, ['m_strInstrumentID', 'm_strCode'], '')
        bs_flag = pick_attr(o, ['m_nDirection', 'm_nTradeType'], 0)
        vol    = pick_attr(o, ['m_nVolume', 'm_nOrderVolume'], 0)
        price  = pick_attr(o, ['m_dPrice', 'm_dOrderPrice'], 0.0)
        status = pick_attr(o, ['m_nOrderStatus', 'm_order_status'], 0)

        try:
            vol = float(vol)
            price = float(price)
        except Exception:
            vol = 0.0
            price = 0.0

        s = summary.setdefault(order_id, {
            'order_id': order_id,
            'code': code,
            'bs_flag': bs_flag,
            'order_volume': vol,
            'order_price': price,
            'order_time': order_time,
            'order_status': status,
            'trade_volume': 0.0,
            'trade_amount': 0.0,
            'trade_price': 0.0,
            'cancel_volume': 0.0,
            'fill_ratio': 0.0,
        })

    # 把成交信息 merge 回去
    for oid, agg in deal_agg.items():
        s = summary.setdefault(oid, {
            'order_id': oid,
            'code': agg['code'],
            'bs_flag': agg['bs_flag'],
            'order_volume': 0.0,
            'order_price': 0.0,
            'order_time': '',
            'order_status': '',
            'trade_volume': 0.0,
            'trade_amount': 0.0,
            'trade_price': 0.0,
            'cancel_volume': 0.0,
            'fill_ratio': 0.0,
        })
        s['trade_volume'] = agg['trade_volume']
        s['trade_amount'] = agg['trade_amount']
        s['trade_price']  = agg['trade_price']

    # 计算撤单量和成交率
    for s in summary.values():
        s['cancel_volume'] = max(0.0, s['order_volume'] - s['trade_volume'])
        if s['order_volume'] > 0:
            s['fill_ratio'] = s['trade_volume'] / s['order_volume']
        else:
            s['fill_ratio'] = 0.0

    # ========= 3A. 全日总体统计（总委托 / 总成交 / 总成交率等） =========
    total_orders = len(summary)
    total_order_vol = sum(s['order_volume'] for s in summary.values())
    total_trade_vol = sum(s['trade_volume'] for s in summary.values())
    total_trade_amt = sum(s['trade_amount'] for s in summary.values())
    overall_fill_ratio = (total_trade_vol / total_order_vol) if total_order_vol > 0 else 0.0
    avg_trade_price = (total_trade_amt / total_trade_vol) if total_trade_vol > 0 else 0.0

    log_info('====== 当日委托执行情况（{}，账号 {}） ======'.format(today, ACCOUNT_ID))
    log_info(
        '总体：委托笔数={}，委托总量={:.0f}，成交总量={:.0f}，整体成交率={:.1f}%，'
        '成交总额={:.2f}，成交均价={:.3f}'.format(
            total_orders,
            total_order_vol,
            total_trade_vol,
            overall_fill_ratio * 100,
            total_trade_amt,
            avg_trade_price
        )
    )

    # ========= 3B. 按股票维度汇总 =========
    per_stock = {}
    for s in summary.values():
        code = s['code']
        if not code:
            continue
        ps = per_stock.setdefault(code, {
            'code': code,
            'order_volume': 0.0,
            'trade_volume': 0.0,
            'trade_amount': 0.0,
            'orders': 0,
        })
        ps['orders'] += 1
        ps['order_volume'] += s['order_volume']
        ps['trade_volume'] += s['trade_volume']
        ps['trade_amount'] += s['trade_amount']

    log_info('—— 按股票维度的执行统计 ——')
    for code, ps in per_stock.items():
        if ps['trade_volume'] > 0:
            avg_px = ps['trade_amount'] / ps['trade_volume']
        else:
            avg_px = 0.0
        fill_ratio = (ps['trade_volume'] / ps['order_volume']) if ps['order_volume'] > 0 else 0.0
        log_info(
            '{} | 委托笔数={}，委托量={:.0f}，成交量={:.0f}，成交率={:.1f}%，成交均价={:.3f}，成交额={:.2f}'.format(
                code,
                ps['orders'],
                ps['order_volume'],
                ps['trade_volume'],
                fill_ratio * 100,
                avg_px,
                ps['trade_amount'],
            )
        )

    # ========= 3C. 逐笔订单明细（和你之前的一样，保留） =========
    log_info('—— 逐笔订单明细 ——')
    for s in summary.values():
        side = '买入' if s['bs_flag'] in (23, 27, 33) else '卖出'
        log_info(
            '[{oid}] {time} {code} {side} | '
            '委托:{ov:.0f} 成交:{tv:.0f} 撤单:{cv:.0f} 成交率:{fr:.1f}% 委托价:{op:.3f} 成交均价:{tp:.3f}'.format(
                oid=s['order_id'],
                time=s['order_time'],
                code=s['code'],
                side=side,
                ov=s['order_volume'],
                tv=s['trade_volume'],
                cv=s['cancel_volume'],
                fr=s['fill_ratio'] * 100,
                op=s['order_price'],
                tp=s['trade_price'],
            )
        )

    # 返回结构方便以后扩展（比如写入数据库或导出文件）
    return {
        'per_order': list(summary.values()),
        'per_stock': per_stock,
        'totals': {
            'total_orders': total_orders,
            'total_order_vol': total_order_vol,
            'total_trade_vol': total_trade_vol,
            'total_trade_amt': total_trade_amt,
            'overall_fill_ratio': overall_fill_ratio,
            'avg_trade_price': avg_trade_price,
        }
    }
# ========== 4. 统一入口：在 QMT 里调用 ==========
def run_account_summary(ContextInfo):
    """
    可以在 init(ContextInfo) 里直接调用，也可以用 schedule_run 盘后跑。
    """
    log_info('====== QMT 实盘账户统计开始 ======')

    acct = fetch_account_info()
    total_asset = acct['total_asset'] if acct else 0.0

    positions = fetch_positions(total_asset)
    orders    = fetch_orders_and_deals()

    log_info('====== QMT 实盘账户统计结束 ======')


def init(ContextInfo):
    # 策略加载时跑一次；你也可以注释掉这行，改成用 ContextInfo.schedule_run 定时跑
    run_account_summary(ContextInfo)
