# -*- coding: utf-8 -*-
"""
QMT 实盘统计脚本：
1）统计当日委托&成交执行情况
2）统计当前持仓对账户净值的贡献
只能在 QMT 策略环境中运行，不依赖 xt_trader
"""

import datetime
import traceback

# ---------- 通用日志封装（直接用 QMT 的 log 对象，如果不存在就退化到 print） ----------
try:
    log  # noqa
except NameError:
    class _DummyLog(object):
        def info(self, *a, **k):  print(*a)
        def error(self, *a, **k): print(*a)
    log = _DummyLog()


# ========== 一些小工具 ==========

def _today_str():
    """返回 yyyy-mm-dd 形式的今天日期字符串"""
    return datetime.datetime.now().strftime("%Y-%m-%d")


def _safe_get(obj, name, default=None):
    """安全地从 QMT 返回的对象上取属性，避免某些字段不存在报错"""
    return getattr(obj, name, default)


# ========== 1. 获取当日订单 / 成交信息 ==========

def get_today_trades(account_id, account_type):
    """
    使用 QMT 内部 API 获取“当日成交明细列表”
    具体函数名和参数，请按你的文档 3.2.4.1 中
    【取某资金账号当日成交明细数据 get_trade_detail_data】那一行进行调整。
    下面是一个典型用法示例：
        trade_list = get_trade_detail_data(account_id, account_type, '')
    如果你本地实际签名不一致，请自行改函数名和参数。
    """
    try:
        # ★ 这里的 get_trade_detail_data 需要你从 QMT 内部模块 import，或者直接使用全局函数
        trade_list = get_trade_detail_data(account_id, account_type, "")
    except Exception:
        log.error("[STAT] 获取成交明细失败：\n%s" % traceback.format_exc())
        trade_list = []

    return trade_list


def get_today_orders(account_id, account_type):
    """
    使用 QMT 内部 API 获取“当日委托列表”
    对应文档 3.2.4.1 中 【取某资金账号当日委托数据 get_order_data】
    典型用法示例：
        order_list = get_order_data(account_id, account_type, '')
    """
    try:
        # ★ 同样，这里的 get_order_data 请按你本机文档调整
        order_list = get_order_data(account_id, account_type, "")
    except Exception:
        log.error("[STAT] 获取委托列表失败：\n%s" % traceback.format_exc())
        order_list = []

    return order_list


def summarize_order_execution(account_id, account_type):
    """
    综合“当日委托 + 当日成交”，按 order_id 给出执行统计：
        - 委托数量 / 成交数量 / 撤单数量
        - 成交均价 / 成交金额
        - 成交率
    结果以 list[dict] 形式返回，并打印到日志。
    """
    today = _today_str()
    trade_list = get_today_trades(account_id, account_type)
    order_list = get_today_orders(account_id, account_type)

    # 1) 先把成交按 order_id 聚合
    trade_agg = {}
    for t in trade_list:
        order_id = _safe_get(t, "order_id", "")
        if not order_id:
            continue
        trade_date = str(_safe_get(t, "trade_time", ""))[:10]
        if trade_date != today:
            # 某些接口一次性返回多日数据，这里只保留今天
            continue

        d = trade_agg.setdefault(order_id, {
            "order_id": order_id,
            "stock_code": _safe_get(t, "stock_code", ""),
            "buy_sell": _safe_get(t, "trade_type", ""),
            "trade_volume": 0,
            "trade_amount": 0.0,
        })
        vol = float(_safe_get(t, "volume", 0))
        price = float(_safe_get(t, "price", 0.0))
        d["trade_volume"] += vol
        d["trade_amount"] += vol * price

    for d in trade_agg.values():
        if d["trade_volume"] > 0:
            d["trade_price"] = d["trade_amount"] / d["trade_volume"]
        else:
            d["trade_price"] = 0.0

    # 2) 再把委托信息填进去
    summary_dict = {}
    for o in order_list:
        order_id = _safe_get(o, "order_id", "")
        if not order_id:
            continue
        order_date = str(_safe_get(o, "order_time", ""))[:10]
        if order_date != today:
            continue

        total_vol = float(_safe_get(o, "volume", 0))
        status = _safe_get(o, "order_status", "")

        s = summary_dict.setdefault(order_id, {
            "order_id": order_id,
            "stock_code": _safe_get(o, "stock_code", ""),
            "buy_sell": _safe_get(o, "trade_type", ""),
            "order_volume": 0,
            "trade_volume": 0,
            "cancel_volume": 0,
            "trade_amount": 0.0,
            "trade_price": 0.0,
            "order_status": status,
        })
        s["order_volume"] = total_vol
        s["order_status"] = status

    # 3) 把成交聚合结果 merge 到 summary
    for oid, tinfo in trade_agg.items():
        s = summary_dict.setdefault(oid, {
            "order_id": oid,
            "stock_code": tinfo["stock_code"],
            "buy_sell": tinfo["buy_sell"],
            "order_volume": 0,
            "trade_volume": 0,
            "cancel_volume": 0,
            "trade_amount": 0.0,
            "trade_price": 0.0,
            "order_status": "",
        })
        s["trade_volume"] = tinfo["trade_volume"]
        s["trade_amount"] = tinfo["trade_amount"]
        s["trade_price"] = tinfo["trade_price"]

    # 4) 计算撤单数量和成交率
    for s in summary_dict.values():
        s["cancel_volume"] = max(0.0, s["order_volume"] - s["trade_volume"])
        if s["order_volume"] > 0:
            s["fill_ratio"] = s["trade_volume"] / s["order_volume"]
        else:
            s["fill_ratio"] = 0.0

    # 打到日志
    log.info("====== 当日委托执行情况汇总（账号 %s, %s）======" % (account_id, account_type))
    for s in summary_dict.values():
        log.info(
            "[%s] %s %s | 委托: %.0f 手, 成交: %.0f 手, 撤单: %.0f 手, 成交率: %.1f%%, 均价: %.3f, 成交额: %.2f"
            % (
                s["order_id"],
                s["stock_code"],
                s["buy_sell"],
                s["order_volume"],
                s["trade_volume"],
                s["cancel_volume"],
                s["fill_ratio"] * 100,
                s["trade_price"],
                s["trade_amount"],
            )
        )

    return list(summary_dict.values())


# ========== 2. 获取当前持仓 + 贡献度 ==========

def get_position_list(context, account_id, account_type):
    """
    获取当前持仓列表：
    这里我采用“ContextInfo 自带的查询函数 + 行情函数”的组合，
    因为你的 ContextInfo dump 里出现了这些方法：
        - get_universe / get_total_share / get_stock_name / get_industry
        - get_market_data_ex / get_product_asset_value / get_net_value
    一个通用思路是：
        1）从券商或本地接口拿到持仓股票代码列表（不同版本函数名不同）
        2）再用 context.get_total_share / get_market_data_ex 算出市值
    由于不同券商适配层差异较大，这里给出一个“模板式”的实现，
    你只需要把『获取持仓列表的那一行』替换成自己的即可。
    """

    try:
        # ★★ 这一步需要你替换成你实际版本的“取持仓列表”函数 ★★
        # 例如某些版本是：position_list = get_position_data(account_id, account_type)
        # position_list 是一堆对象，至少包含 stock_code / volume 字段
        position_list = get_position_data(account_id, account_type)
    except Exception:
        log.error("[STAT] 获取持仓失败：\n%s" % traceback.format_exc())
        position_list = []

    results = []
    for p in position_list:
        code = _safe_get(p, "stock_code", "")
        vol = float(_safe_get(p, "volume", 0))
        if not code or vol <= 0:
            continue

        # 使用 ContextInfo 提供的行情 / 基本信息函数
        name = context.get_stock_name(code)
        try:
            industry = context.get_industry(code)
        except Exception:
            industry = ""

        # 取最新价：如果你的版本提供了 get_market_data_ex，可以这样写
        last_price = 0.0
        try:
            md = context.get_market_data_ex(code)
            last_price = float(_safe_get(md, "last_price", 0.0))
        except Exception:
            pass

        market_value = vol * last_price
        results.append({
            "stock_code": code,
            "stock_name": name,
            "industry": industry,
            "volume": vol,
            "last_price": last_price,
            "market_value": market_value,
        })

    # 再用 ContextInfo 的账户整体资产函数，计算贡献度
    total_asset = 0.0
    try:
        # 你 dump 出来的 ContextInfo 有 get_product_asset_value / get_net_value 等
        total_asset = float(context.get_product_asset_value())
    except Exception:
        # 如果上面的函数在你版本中不存在，可以改成 get_net_value() 等
        try:
            total_asset = float(context.get_net_value())
        except Exception:
            total_asset = 0.0

    if total_asset > 0:
        for r in results:
            r["weight_in_portfolio"] = r["market_value"] / total_asset
    else:
        for r in results:
            r["weight_in_portfolio"] = 0.0

    # 打到日志
    log.info("====== 当前持仓及对资产贡献（账号 %s, %s）======" % (account_id, account_type))
    log.info("账户总资产估算：%.2f" % total_asset)
    for r in sorted(results, key=lambda x: x["market_value"], reverse=True):
        log.info(
            "%s %s | 行业:%s, 持仓:%.0f 股, 现价:%.3f, 市值:%.2f, 占比:%.2f%%"
            % (
                r["stock_code"],
                r["stock_name"],
                r["industry"],
                r["volume"],
                r["last_price"],
                r["market_value"],
                r["weight_in_portfolio"] * 100,
            )
        )

    return results, total_asset


# ========== 3. 对外入口：供 QMT 调用 ==========

def run_once(context):
    """
    一个方便你在 QMT 中“点一次就跑完统计”的入口。
    你可以在 init 里调用，也可以在菜单里绑定到某个按钮。
    """
    # 资金账号和账户类型建议在参数里配置，这里给一个默认写死的方式
    account_id = context.account_id if hasattr(context, "account_id") else ""
    account_type = "STOCK"  # 或从参数中读取

    if not account_id:
        log.info("[STAT] 当前 context 不含 account_id，请从参数或配置中传入。")
        # 如：account_id = g.my_account_id
        return

    log.info("====== QMT 实盘统计开始 ======")
    summarize_order_execution(account_id, account_type)
    get_position_list(context, account_id, account_type)
    log.info("====== QMT 实盘统计结束 ======")


def init(ContextInfo):
    """
    QMT 策略的入口函数。
    这里简单地在策略启动时就跑一次统计，
    如果你只想在盘后跑，可以改成在收盘时通过 schedule_run 调用 run_once。
    """
    global context
    context = ContextInfo
    log.info("[STAT] 账户统计脚本 init 完成，准备执行 run_once()")
    run_once(ContextInfo)
