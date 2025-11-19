# coding: gbk
# QMT 侧 JoinQuant → MSSQL → QMT 下单桥接策略（tick 轮询版）

import datetime
import time
import pymssql

# =============== 参数区：请根据实际情况修改 ==================

# —— MSSQL 连接参数 ——
MSSQL_SERVER   = '8.138.38.43'   # 你的数据库服务器 IP / 域名
MSSQL_PORT     = 1433            # 端口
MSSQL_USER     = 'sa'
MSSQL_PWD      = '你的数据库密码'
MSSQL_DB       = 'touzi'
MSSQL_TABLE    = 'trade'

# —— 交易账号参数（必须改成你自己的资金号） ——
ACCOUNT_ID     = '6000000248'    # QMT 资金账号
ACCOUNT_TYPE   = 'STOCK'         # 股票账户，固定写 'STOCK'

# —— 信号过滤 & 处理节奏 ——
# 只处理这些 fenlei 的记录；如果想全部处理，可以设成空列表 []
FENLEI_WHITELIST = [u'未分类']

MAX_ROWS_PER_TICK   = 20   # 每个 tick 最多处理多少条信号
POLL_INTERVAL_SEC   = 1.0  # 最快轮询间隔（秒）
MAX_RETRY_PER_ORDER = 3    # 每条信号最多重试下单次数

# flag 约定：
FLAG_NEW        = None  # 数据库层面：NULL 表示新鲜信号，不在代码中直接用
FLAG_UNFILLED   = 0     # 已委托但没有成交 / 暂时没有成交回报
FLAG_FILLED     = 1     # 已成交（能拿到成交回报）
FLAG_ERROR      = 9     # 三次尝试仍失败或严重错误

# ============================================================

# 内部：记录上次轮询时间，避免 tick 过于频繁时反复查询 DB
_last_poll_ts = 0.0


def _get_db_conn():
    """获取 MSSQL 连接。"""
    conn = pymssql.connect(
        server=MSSQL_SERVER,
        user=MSSQL_USER,
        password=MSSQL_PWD,
        database=MSSQL_DB,
        port=MSSQL_PORT,
        charset='utf8'  # trade 表用 NVARCHAR，UTF-8 读写没问题
    )
    return conn


def _fetch_new_signals():
    """
    从 MSSQL 取待处理信号：
    条件：guoqi = 0 且 flag IS NULL，且 fenlei 在白名单（如果配置了的话）
    """
    conn = _get_db_conn()
    cursor = conn.cursor()

    where_clauses = ["guoqi = 0", "flag IS NULL"]
    params = []

    if FENLEI_WHITELIST:
        # 生成 fenlei IN (...)
        placeholders = ",".join(["%s"] * len(FENLEI_WHITELIST))
        where_clauses.append("fenlei IN ({})".format(placeholders))
        params.extend(FENLEI_WHITELIST)

    where_sql = " AND ".join(where_clauses)
    sql = (
        "SELECT TOP {top} id, name, code, price, jiner, num, date, guoqi, flag, "
        "fenlei, zhixing_time, type "
        "FROM dbo.{table} WITH (READPAST) "
        "WHERE {where} "
        "ORDER BY id ASC;"
    ).format(
        top=MAX_ROWS_PER_TICK,
        table=MSSQL_TABLE,
        where=where_sql
    )

    cursor.execute(sql, tuple(params))
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]

    cursor.close()
    conn.close()

    result = []
    for row in rows:
        record = dict(zip(col_names, row))
        result.append(record)
    return result


def _extract_price_amount(obj_order, obj_deal):
    """
    尝试从 QMT 的委托/成交对象中提取价格和成交额。
    不同柜台字段名可能略有不同，这里做了一些兼容性处理。
    拿不到就返回 (None, None)。
    """
    price = None
    amount = None
    volume = None

    # 尝试从成交对象拿成交价、成交额、成交量
    if obj_deal is not None:
        for attr in ('m_dPrice', 'm_dTradePrice', 'tradeprice', 'm_dLimitPrice'):
            if hasattr(obj_deal, attr):
                price = getattr(obj_deal, attr)
                break

        for attr in ('m_nVolume', 'm_nVolumeTraded', 'volumetraded'):
            if hasattr(obj_deal, attr):
                volume = getattr(obj_deal, attr)
                break

        for attr in ('m_dTradeAmount', 'm_dTurnover', 'tradeamount'):
            if hasattr(obj_deal, attr):
                amount = getattr(obj_deal, attr)
                break

    # 如果成交对象没拿到，就从委托对象尝试拿成交均价和成交量
    if obj_order is not None:
        if price is None:
            for attr in ('m_dTradedPrice', 'tradeprice', 'm_dPrice'):
                if hasattr(obj_order, attr):
                    price = getattr(obj_order, attr)
                    break

        if volume is None:
            for attr in ('m_nVolumeTraded', 'volumetraded', 'm_nVolume'):
                if hasattr(obj_order, attr):
                    volume = getattr(obj_order, attr)
                    break

    # 如果只有 price 和 volume，算一个金额出来
    try:
        if amount is None and price is not None and volume is not None:
            amount = float(price) * float(volume)
    except Exception:
        pass

    return price, amount


def _update_trade_record(trade_id, price, amount, flag_value, expired=True):
    """
    更新 trade 表：成交价、成交额、flag、guoqi、执行时间。
    """
    conn = _get_db_conn()
    cursor = conn.cursor()

    guoqi_val = 1 if expired else 0
    now = datetime.datetime.now()

    sql = (
        "UPDATE dbo.{table} "
        "SET price = %s, jiner = %s, flag = %s, guoqi = %s, zhixing_time = %s "
        "WHERE id = %s;"
    ).format(table=MSSQL_TABLE)

    cursor.execute(sql, (price, amount, flag_value, guoqi_val, now, trade_id))
    conn.commit()
    cursor.close()
    conn.close()


def _mark_trade_error(trade_id, errmsg):
    """下单失败时，标记为错误并置 guoqi=1。"""
    print(u"[QMT桥接] 订单 id={} 下单失败：{}".format(trade_id, errmsg))
    _update_trade_record(trade_id, None, None, FLAG_ERROR, expired=True)


def _send_qmt_order(record, ContextInfo):
    """
    根据 trade 表中的一条记录，调用对应的 QMT 下单函数。
    返回 (成功?, 是否认为已成交?, 成交价, 成交额)
    """
    trade_id = record['id']
    code = record['code']
    ttype = (record.get('type') or '').strip().lower()
    num = record.get('num') or 0

    # num 为 0 没有意义，直接视为错误
    try:
        num = float(num)
    except Exception:
        raise Exception(u"num 字段无法转为数字: {}".format(num))

    if num == 0:
        raise Exception(u"num 为 0，忽略该信号")

    # 方向：正数买入，负数卖出（QMT 的 order_value / order_shares 都支持正负号）
    direction = 1 if num > 0 else -1
    abs_num = abs(num)

    # =============== 按 type 映射到 QMT 交易函数 ==================
    # JoinQuant → QMT 对应关系：
    #   order_target       → order_shares（目标股数）
    #   order_value        → order_value（金额下单）
    #   order_target_value → order_target_value（目标金额）
    #   order              → 用 order_shares 模拟“按手下单”，num 换算成手
    # ============================================================

    # 下单实现作为内部函数，方便做重试
    def _do_order():
        if ttype == 'order_value':
            # 金额下单，num 正负代表买卖方向
            order_value(code, direction * abs_num, ContextInfo, ACCOUNT_ID)

        elif ttype == 'order_target_value':
            # 目标金额：QMT 的 order_target_value 也是填目标金额，非负
            order_target_value(code, abs_num, ContextInfo, ACCOUNT_ID)

        elif ttype == 'order_target':
            # 目标股数：num 为目标持仓股数，非负
            order_shares(code, abs_num, ContextInfo, ACCOUNT_ID)

        else:
            # 默认视作 order：按手下单（lots），num 为股数，需要 /100 转成手
            lots = int(abs_num // 100)
            if lots <= 0:
                raise Exception(u"order 类型且 num < 100 股，无法转为 1 手")
            shares = lots * 100 * direction
            order_shares(code, shares, ContextInfo, ACCOUNT_ID)

    # ===================== 重试逻辑 =============================
    last_error = u''
    for i in range(1, MAX_RETRY_PER_ORDER + 1):
        try:
            _do_order()
            print(u"[QMT桥接] 订单 id={} 第 {} 次尝试下单成功，code={} type={} num={}".format(
                trade_id, i, code, ttype, num))
            break
        except Exception as e:
            last_error = u"{}".format(e)
            print(u"[QMT桥接] 订单 id={} 第 {} 次尝试下单失败：{}".format(trade_id, i, last_error))
            if i == MAX_RETRY_PER_ORDER:
                raise
            # 给柜台一点缓冲时间
            time.sleep(0.2)

    # ===================== 查询成交回报 =========================
    # 简单做法：拿“最新的委托号 / 成交号”，适用于当前只有本策略在下单的情况。
    price = None
    amount = None
    filled = False

    try:
        # 先取最新委托号，再尝试取委托 & 成交细节
        orderid = get_last_order_id(ACCOUNT_ID, ACCOUNT_TYPE.lower(), 'ORDER')
        if orderid and orderid != '-1':
            obj_order = get_value_by_order_id(orderid, ACCOUNT_ID,
                                              ACCOUNT_TYPE.lower(), 'ORDER')
        else:
            obj_order = None

        dealid = get_last_order_id(ACCOUNT_ID, ACCOUNT_TYPE.lower(), 'DEAL')
        if dealid and dealid != '-1':
            obj_deal = get_value_by_order_id(dealid, ACCOUNT_ID,
                                             ACCOUNT_TYPE.lower(), 'DEAL')
        else:
            obj_deal = None

        price, amount = _extract_price_amount(obj_order, obj_deal)
        if amount is not None and amount != 0:
            filled = True
    except Exception as e:
        # 只打印提醒，不影响主流程
        print(u"[QMT桥接] 订单 id={} 获取成交信息异常：{}".format(trade_id, e))

    # 若取不到成交额，则认为“已委托未成交”，flag=0
    # 若能拿到成交额，则 flag=1
    flag_value = FLAG_FILLED if filled else FLAG_UNFILLED

    return True, filled, price, amount, flag_value


# ===================== QMT 策略入口函数 =========================

def init(ContextInfo):
    """
    QMT 初始化函数：设置资金账号等。
    在 QMT 策略参数里不需要再配置 accid，这里统一写。
    """
    ContextInfo.accid = ACCOUNT_ID
    print(u"[QMT桥接] init 完成，使用资金账号：{}".format(ACCOUNT_ID))


def handlebar(ContextInfo):
    """
    每个 tick 调用一次。
    轮询 MSSQL，取 trade 表的新信号并下单。
    """
    global _last_poll_ts

    now_ts = time.time()
    if (now_ts - _last_poll_ts) < POLL_INTERVAL_SEC:
        return
    _last_poll_ts = now_ts

    try:
        signals = _fetch_new_signals()
    except Exception as e:
        print(u"[QMT桥接] 从 MSSQL 读取信号失败：{}".format(e))
        return

    if not signals:
        return

    for rec in signals:
        trade_id = rec['id']
        try:
            ok, filled, price, amount, flag_value = _send_qmt_order(rec, ContextInfo)
            if ok:
                _update_trade_record(trade_id, price, amount, flag_value, expired=True)
        except Exception as e:
            _mark_trade_error(trade_id, e)
