#coding:gbk
"""
QMT 实时跟单交易插件 v2.0 (支持回测/调试模式)
================================================
功能：从 SQL Server 数据库 touzi.dbo.trade 表读取交易信号，
      调用 QMT passorder API 下单，并更新信号状态防止重复执行。

运行模式：
  live      = 实盘模式，从 SQL Server 读取真实 trade 表信号，调用 passorder 下单
  backtest  = 回测/调试模式，使用模拟 trade 信号，不真实下单，不连接数据库

运行环境：QMT 自定义 Python 策略编辑器
编码声明：GBK（QMT 规范要求首行 #coding:gbk）

注意：
  - 本文件在 VS Code 中以 UTF-8 保存，导入 QMT 编辑器后请另存为 GBK 编码
  - 所有不确定的 QMT API 参数均以 TODO 标出，详见文件末尾清单
  - 默认 run_mode='backtest'，调试完成后再改为 'live'
"""

import datetime
import traceback
import re
import sys
import copy

# ==================== 数据库驱动导入 ====================
# 优先使用 pyodbc，备选 pymssql
# TODO: 需人工确认 QMT Python 环境已安装哪个驱动
try:
    import pyodbc
    DB_DRIVER = "pyodbc"
except ImportError:
    try:
        import pymssql
        DB_DRIVER = "pymssql"
    except ImportError:
        DB_DRIVER = None

# ==================== QMT passorder 常量定义 ====================

# ---------- opType（操作类型）----------
# TODO: 以下 opType 数值需人工确认，仅列出股票买卖相关
OP_TYPE_BUY = 23     # 股票买入
OP_TYPE_SELL = 24    # 股票卖出
# TODO: 完整 opType 枚举需人工确认（融资融券、逆回购等未列出）

# ---------- orderType（下单类型）----------
ORDER_TYPE_BY_QUANTITY = 1101   # 按数量下单（对应 type=order）
ORDER_TYPE_BY_AMOUNT = 1202     # 按金额下单（对应 type=order_value）
# TODO: order_target_value 对应的 orderType 需人工确认
#       目前按金额处理，使用 1202，但目标市值语义可能需要特殊 orderType
ORDER_TYPE_TARGET_VALUE = 1202  # TODO: 需人工确认是否有独立 orderType

# ---------- prType（委托方式）----------
PR_TYPE_LIMIT = 0       # 限价委托，需指定价格
PR_TYPE_LATEST = 5      # 最新价
PR_TYPE_LIMIT_UP_DOWN = 6   # 涨停价买入 / 跌停价卖出
PR_TYPE_COUNTERPART = 7     # 对手价（卖一价买入 / 买一价卖出）
PR_TYPE_BEST5 = 2           # 最优五档即时成交剩余撤销

# ---------- quickTrade ----------
QUICK_TRADE_SAFE = 1     # 仅在 is_last_bar() 且非历史 bar 时触发（推荐）
QUICK_TRADE_AGGRESSIVE = 2  # 不判断 bar 状态，风险高，不推荐

# ==================== 委托方式映射表 ====================
# TODO: 以下 prType 数值部分为推测，需人工确认

# 买入委托方式 -> prType
BUY_PRICE_TYPE_MAP = {
    "卖五价": 11,       # TODO: 需人工确认 prType 数值
    "卖四价": 10,       # TODO: 需人工确认 prType 数值
    "卖三价": 9,        # TODO: 需人工确认 prType 数值
    "卖二价": 8,        # TODO: 需人工确认 prType 数值
    "卖一价": 7,        # TODO: 需人工确认（与对手价是否相同 prType=7）
    "笼子上限": -1,     # TODO: 需人工确认 prType 数值及含义（价格笼子机制）
    "涨跌停价": PR_TYPE_LIMIT_UP_DOWN,   # 6
    "对手价": PR_TYPE_COUNTERPART,        # 7
    "最优五档成交": PR_TYPE_BEST5,        # 2
}

# 卖出委托方式 -> prType
SELL_PRICE_TYPE_MAP = {
    "买五价": 15,       # TODO: 需人工确认 prType 数值
    "买四价": 14,       # TODO: 需人工确认 prType 数值
    "买三价": 13,       # TODO: 需人工确认 prType 数值
    "买二价": 12,       # TODO: 需人工确认 prType 数值
    "买一价": 7,        # TODO: 需人工确认（与对手价是否相同 prType=7）
    "涨跌停价": PR_TYPE_LIMIT_UP_DOWN,   # 6
    "对手价": PR_TYPE_COUNTERPART,        # 7
    "最优五档成交": PR_TYPE_BEST5,        # 2
}

# ==================== 数据库默认配置 ====================
DB_NAME = "touzi"           # 数据库名
DB_TABLE = "dbo.trade"      # 表名
DB_USER_DEFAULT = "sa"      # 默认数据库用户名

# ==================== SQL 参数占位符 ====================
# pymssql 使用 %s，pyodbc 使用 ?
if DB_DRIVER == "pymssql":
    SQL_PARAM = "%s"
else:
    SQL_PARAM = "?"

# ==================== 日志前缀 ====================
LOG_PREFIX = "[QMT跟单]"
LOG_PREFIX_BT = "[BACKTEST]"

# ==================== 运行模式常量 ====================
RUN_MODE_LIVE = "live"          # 实盘模式
RUN_MODE_BACKTEST = "backtest"  # 回测/调试模式

# ==================== 回测报告打印间隔（bar 数）====================
BACKTEST_REPORT_INTERVAL = 50


# =====================================================================
#  模块1：参数读取模块
# =====================================================================

def read_config(ContextInfo):
    """
    从 QMT XML 注入的变量中读取配置参数
    XML 变量通过 ContextInfo 属性传递，使用 getattr 安全读取
    """
    cfg = {}
    # 服务器地址（XML bind="name"）
    cfg['db_server'] = str(getattr(ContextInfo, 'name', '127.0.0.1')).strip()
    # 数据库密码（XML bind="pws"）
    cfg['db_password'] = str(getattr(ContextInfo, 'pws', '')).strip()
    # 开始时间（XML bind="start_time"，格式 HHMMSS）
    cfg['start_time'] = str(getattr(ContextInfo, 'start_time', '093000')).strip()
    # 结束时间（XML bind="end_time"，格式 HHMMSS）
    cfg['end_time'] = str(getattr(ContextInfo, 'end_time', '153000')).strip()
    # 买入金额上限（XML bind="Buy_Amount"，0=不限制，使用数据库 num）
    cfg['Buy_Amount'] = float(getattr(ContextInfo, 'Buy_Amount', 0))
    # 最多买入股票数（XML bind="Buy_num"）
    cfg['Buy_num'] = int(getattr(ContextInfo, 'Buy_num', 100))
    # 卖出比例（XML bind="sell_ratio"，百分比，100=全卖）
    cfg['sell_ratio'] = float(getattr(ContextInfo, 'sell_ratio', 100))
    # 跟单比例（XML bind="bili"，1=全跟，0.5=跟一半）
    cfg['bili'] = float(getattr(ContextInfo, 'bili', 1))
    # 数据库用户名（XML bind="username"）
    cfg['db_user'] = str(getattr(ContextInfo, 'username', DB_USER_DEFAULT)).strip()
    # 策略分类（XML bind="fenlei"）
    cfg['fenlei'] = str(getattr(ContextInfo, 'fenlei', '')).strip()
    # 卖出委托方式（XML bind="fs_sell"）
    cfg['fs_sell'] = str(getattr(ContextInfo, 'fs_sell', '最优五档成交')).strip()
    # 买入委托方式（XML bind="fs_buy"）
    cfg['fs_buy'] = str(getattr(ContextInfo, 'fs_buy', '最优五档成交')).strip()
    # 运行模式（XML bind="run_mode"，默认 backtest）
    cfg['run_mode'] = str(getattr(ContextInfo, 'run_mode', RUN_MODE_BACKTEST)).strip().lower()

    return cfg


# =====================================================================
#  模块2：数据库连接模块（仅 live 模式使用）
# =====================================================================

def get_db_connection(server, user, password, database, retries=3):
    """
    获取 SQL Server 数据库连接，支持重试
    优先使用 pyodbc，备选 pymssql

    参数:
        server:   服务器地址
        user:     数据库用户名
        password: 数据库密码
        database: 数据库名
        retries:  重试次数

    返回:
        连接对象 或 None
    """
    if DB_DRIVER is None:
        print("%s 严重错误：未找到 pyodbc 或 pymssql，无法连接数据库！" % LOG_PREFIX)
        return None

    for attempt in range(1, retries + 1):
        try:
            if DB_DRIVER == "pymssql":
                conn = pymssql.connect(
                    server=server,
                    user=user,
                    password=password,
                    database=database,
                    login_timeout=10,
                    timeout=30
                )
            else:  # pyodbc
                conn_str = (
                    "DRIVER={SQL Server};"
                    "SERVER=%s;"
                    "DATABASE=%s;"
                    "UID=%s;"
                    "PWD=%s;"
                    "TrustServerCertificate=yes;"
                ) % (server, database, user, password)
                conn = pyodbc.connect(conn_str, timeout=10)

            print("%s 数据库连接成功 server=%s db=%s driver=%s" % (
                LOG_PREFIX, server, database, DB_DRIVER))
            return conn

        except Exception as e:
            print("%s 数据库连接失败 第%d/%d次 server=%s error=%s" % (
                LOG_PREFIX, attempt, retries, server, str(e)))
            if attempt < retries:
                import time
                time.sleep(2 * attempt)  # 递增等待

    print("%s 错误：数据库连接失败，已重试%d次" % (LOG_PREFIX, retries))
    return None


def check_db_connection(conn):
    """
    检查数据库连接是否有效

    参数:
        conn: 数据库连接对象

    返回:
        True=连接有效, False=连接已断开
    """
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        return True
    except Exception:
        return False


# =====================================================================
#  模块3：交易时间判断模块
# =====================================================================

def is_trading_time(now_time_str, start_time, end_time):
    """
    判断当前时间是否在交易时间窗口内

    参数:
        now_time_str: 当前时间字符串，格式 "HHMMSS"
        start_time:   开始时间字符串，格式 "HHMMSS"
        end_time:     结束时间字符串，格式 "HHMMSS"

    返回:
        True=在交易时间内, False=不在
    """
    try:
        now = int(now_time_str)
        start = int(start_time)
        end = int(end_time)
        return start <= now <= end
    except (ValueError, TypeError):
        print("%s 警告：时间格式错误 now=%s start=%s end=%s" % (
            LOG_PREFIX, now_time_str, start_time, end_time))
        return False


# =====================================================================
#  模块4：信号查询模块（统一接口，分派 live/backtest）
# =====================================================================

def query_signals(ContextInfo, fenlei, bar_date=None):
    """
    统一信号查询接口：根据 run_mode 分派到数据库查询或模拟数据

    参数:
        ContextInfo: QMT 上下文
        fenlei:      策略分类筛选条件
        bar_date:    当前 K 线日期（仅 backtest 模式使用）

    返回:
        信号字典列表，按 date ASC, id ASC 排序
    """
    run_mode = getattr(ContextInfo, 'run_mode', RUN_MODE_BACKTEST)

    if run_mode == RUN_MODE_BACKTEST:
        return query_mock_signals(ContextInfo, fenlei, bar_date)
    else:
        return query_db_signals(ContextInfo.db_conn, fenlei)


def query_db_signals(conn, fenlei):
    """
    从 SQL Server 数据库查询今日未过期、新鲜的交易信号

    查询条件:
        - 日期为今日
        - guoqi = 0（未过期）
        - flag IS NULL（新鲜信号）
        - fenlei 匹配配置

    参数:
        conn:   数据库连接
        fenlei: 策略分类筛选条件

    返回:
        信号字典列表
    """
    if conn is None:
        return []

    try:
        cursor = conn.cursor()

        # 使用 sargable 日期范围条件，利于索引优化
        sql = """
        SELECT id, name, code, price, jiner, num, date, guoqi, flag, fenlei, type
        FROM dbo.trade
        WHERE [date] >= CONVERT(date, GETDATE())
          AND [date] < DATEADD(day, 1, CONVERT(date, GETDATE()))
          AND ISNULL(guoqi, 0) = 0
          AND flag IS NULL
          AND fenlei = %s
        ORDER BY date ASC, id ASC
        """ % SQL_PARAM

        cursor.execute(sql, (fenlei,))

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        signals = []
        for row in rows:
            signal = dict(zip(columns, row))
            signals.append(signal)

        cursor.close()
        return signals

    except Exception as e:
        print("%s 查询信号异常 error=%s" % (LOG_PREFIX, str(e)))
        traceback.print_exc()
        return []


def query_mock_signals(ContextInfo, fenlei, bar_date=None):
    """
    从内存模拟信号中查询匹配的未过期、新鲜信号（backtest 模式）

    参数:
        ContextInfo: QMT 上下文（含 mock_trade_signals）
        fenlei:      策略分类筛选条件
        bar_date:    当前 K 线日期（datetime.date），None 则不过滤日期

    返回:
        信号字典列表
    """
    mock_signals = getattr(ContextInfo, 'mock_trade_signals', [])
    if not mock_signals:
        return []

    result = []
    for sig in mock_signals:
        # guoqi 检查：0 或 None 表示未过期
        guoqi = sig.get('guoqi', 0)
        if guoqi is not None and guoqi != 0:
            continue

        # flag 检查：None 表示新鲜信号
        if sig.get('flag') is not None:
            continue

        # fenlei 匹配
        if str(sig.get('fenlei', '')).strip() != fenlei:
            continue

        # 日期过滤（如果提供了 bar_date）
        if bar_date is not None:
            sig_date = sig.get('date')
            if sig_date is not None:
                # 兼容字符串和 datetime 对象
                if isinstance(sig_date, str):
                    try:
                        sig_date_obj = datetime.datetime.strptime(sig_date[:10], "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        continue
                elif isinstance(sig_date, datetime.datetime):
                    sig_date_obj = sig_date.date()
                elif isinstance(sig_date, datetime.date):
                    sig_date_obj = sig_date
                else:
                    continue

                if sig_date_obj != bar_date:
                    continue

        result.append(sig)

    # 按 date ASC, id ASC 排序
    def sort_key(s):
        d = s.get('date', '')
        i = s.get('id', 0)
        return (str(d), i)

    result.sort(key=sort_key)
    return result


# =====================================================================
#  模块5：信号校验模块
# =====================================================================

def validate_signal(signal, processed_ids, Buy_Amount, Buy_num, buy_count, bili):
    """
    校验单条交易信号是否合法、可执行

    校验内容:
        1. 必要字段非空（id, code, type, num）
        2. id 未被处理过（幂等检查）
        3. type 在合法范围内
        4. 买入金额不超过 Buy_Amount
        5. 买入股票数不超过 Buy_num

    参数:
        signal:        信号字典
        processed_ids: 已处理信号 ID 集合
        Buy_Amount:    买入金额上限（0=不限制）
        Buy_num:       最多买入股票数
        buy_count:     当前已买入股票计数
        bili:          跟单比例

    返回:
        (True, "") 或 (False, "错误原因")
    """
    # 检查必要字段
    signal_id = signal.get('id')
    if signal_id is None:
        return False, "信号 id 为空"

    code = signal.get('code')
    if not code or not str(code).strip():
        return False, "信号 code 为空 id=%s" % signal_id

    sig_type = str(signal.get('type', '')).strip().lower()
    if sig_type not in ('order', 'order_value', 'order_target_value'):
        return False, "信号 type 不合法 type=%s id=%s" % (signal.get('type'), signal_id)

    # 幂等检查
    if signal_id in processed_ids:
        return False, "信号已处理过 id=%s" % signal_id

    # 判断买卖方向
    num = signal.get('num')
    try:
        num_val = float(num) if num is not None else 0
    except (ValueError, TypeError):
        return False, "信号 num 格式错误 num=%s id=%s" % (num, signal_id)

    is_sell = num_val < 0
    is_close = (sig_type == 'order_target_value' and num_val == 0)

    # 买入金额限制（仅对买入信号检查，不拒绝但会在映射时截断）
    if not is_sell and not is_close and Buy_Amount > 0:
        if sig_type == 'order_value':
            adjusted_amount = abs(num_val) * bili
            if adjusted_amount > Buy_Amount:
                pass  # 不拒绝，映射时截断
        elif sig_type == 'order_target_value' and num_val > 0:
            adjusted_amount = num_val * bili
            if adjusted_amount > Buy_Amount:
                pass

    # 买入股票数限制（仅对买入信号检查）
    if not is_sell and not is_close and Buy_num > 0 and buy_count >= Buy_num:
        return False, "已达最大买入股票数 Buy_num=%s id=%s" % (Buy_num, signal_id)

    return True, ""


# =====================================================================
#  模块6：QMT 下单参数映射模块
# =====================================================================

def normalize_code(code):
    """
    将股票代码转换为 QMT 标准格式：XXXXXX.SH 或 XXXXXX.SZ

    支持输入格式:
        - 000001.SZ / 600000.SH（已是 QMT 格式）
        - 000001.XSHE / 600000.XSHG（聚宽格式）
        - 000001 / 600000（纯6位数字）
        - SZ000001 / SH600000（带市场前缀）

    参数:
        code: 股票代码字符串

    返回:
        QMT 格式代码 或 None（格式无效时）
    """
    if not code:
        return None

    code = str(code).strip().upper()

    # 已经是 QMT 格式：XXXXXX.SH / XXXXXX.SZ
    if re.match(r'^\d{6}\.(SH|SZ)$', code):
        return code

    # 聚宽格式：000001.XSHE -> 000001.SZ, 600000.XSHG -> 600000.SH
    if code.endswith('.XSHE'):
        return code[:-5] + '.SZ'
    elif code.endswith('.XSHG'):
        return code[:-5] + '.SH'

    # 带市场前缀：SZ000001 -> 000001.SZ, SH600000 -> 600000.SH
    m_pre = re.match(r'^(SH|SZ)(\d{6})$', code)
    if m_pre:
        return m_pre.group(2) + '.' + m_pre.group(1)

    # 纯6位数字：根据代码规则判断市场
    pure_code = re.sub(r'[^0-9]', '', code)
    if len(pure_code) == 6:
        # 6xx, 9xx, 68xxx -> 上海
        if pure_code.startswith(('6', '9', '68')):
            return pure_code + '.SH'
        # 0xx, 1xx, 2xx, 3xx -> 深圳
        elif pure_code.startswith(('0', '1', '2', '3')):
            return pure_code + '.SZ'
        else:
            print("%s 警告：无法确定股票 %s 的市场，默认按深圳处理" % (LOG_PREFIX, code))
            return pure_code + '.SZ'

    print("%s 错误：无法识别的股票代码格式：%s" % (LOG_PREFIX, code))
    return None


def map_order_params(signal, fs_buy, fs_sell, Buy_Amount, sell_ratio, bili):
    """
    将数据库信号映射为 QMT passorder 参数

    业务规则:
        - type=order:           按数量下单，orderType=1101
        - type=order_value:     按金额下单，orderType=1202
        - type=order_target_value + num=0: 清仓（卖出全部持仓）
        - type=order_target_value + num>0: 目标市值（TODO: 需计算差额）
        - num<0:                卖出（取绝对值）
        - price 非 NULL 且 >0:  限价委托 prType=0
        - price 为 NULL 或 =0:  使用界面委托方式对应的 prType

    参数:
        signal:     信号字典
        fs_buy:     买入委托方式（中文）
        fs_sell:    卖出委托方式（中文）
        Buy_Amount: 买入金额上限（0=不限制）
        sell_ratio: 卖出比例（百分比）
        bili:       跟单比例

    返回:
        参数字典 或 None（映射失败时）
        字典键: op_type, order_type, order_code, pr_type, order_price, volume, is_close
    """
    sig_type = str(signal.get('type', '')).strip().lower()
    num = signal.get('num', 0)
    price = signal.get('price', None)
    code = signal.get('code', '')

    # ---- 规范化股票代码 ----
    order_code = normalize_code(code)
    if order_code is None:
        print("%s 错误：股票代码格式无效 code=%s" % (LOG_PREFIX, code))
        return None

    # ---- 解析 num ----
    try:
        num_val = float(num) if num is not None else 0
    except (ValueError, TypeError):
        print("%s 错误：num 格式错误 num=%s" % (LOG_PREFIX, num))
        return None

    # ---- 判断买卖方向 ----
    is_sell = num_val < 0
    is_close = (sig_type == 'order_target_value' and num_val == 0)

    # ---- 确定 opType ----
    op_type = OP_TYPE_SELL if (is_sell or is_close) else OP_TYPE_BUY

    # ---- 确定 orderType、volume、prType、price ----
    order_price = 0
    pr_type = PR_TYPE_BEST5  # 默认最优五档

    if is_close:
        # ===== 清仓信号 =====
        order_type = ORDER_TYPE_BY_QUANTITY
        volume = 0  # 占位，执行时需查询持仓
        pr_type = SELL_PRICE_TYPE_MAP.get(fs_sell, PR_TYPE_BEST5)

    elif sig_type == 'order':
        # ===== 按数量下单 =====
        order_type = ORDER_TYPE_BY_QUANTITY
        raw_volume = abs(num_val)

        # 应用跟单比例
        volume = int(raw_volume * bili)

        # 卖出时应用卖出比例
        if is_sell:
            volume = int(volume * sell_ratio / 100.0)

        # 买入时按100股取整（A股最小交易单位）
        if not is_sell:
            volume = max(100, (volume // 100) * 100)

        # 委托方式
        pr_type = SELL_PRICE_TYPE_MAP.get(fs_sell, PR_TYPE_BEST5) if is_sell else \
                  BUY_PRICE_TYPE_MAP.get(fs_buy, PR_TYPE_BEST5)

    elif sig_type == 'order_value':
        # ===== 按金额下单 =====
        order_type = ORDER_TYPE_BY_AMOUNT
        raw_volume = abs(num_val)

        # 应用跟单比例
        volume = round(raw_volume * bili, 2)

        # 卖出时应用卖出比例
        if is_sell:
            volume = round(volume * sell_ratio / 100.0, 2)

        # 买入时检查金额上限
        if not is_sell and Buy_Amount > 0:
            volume = min(volume, Buy_Amount)

        # 委托方式
        pr_type = SELL_PRICE_TYPE_MAP.get(fs_sell, PR_TYPE_BEST5) if is_sell else \
                  BUY_PRICE_TYPE_MAP.get(fs_buy, PR_TYPE_BEST5)

    elif sig_type == 'order_target_value':
        # ===== 目标市值下单 =====
        # TODO: 需人工确认 order_target_value 对应的 orderType
        order_type = ORDER_TYPE_BY_AMOUNT
        target_value = abs(num_val)

        # 应用跟单比例
        target_value = round(target_value * bili, 2)

        # 买入时检查金额上限
        if not is_sell and Buy_Amount > 0:
            target_value = min(target_value, Buy_Amount)

        volume = target_value

        # 委托方式
        pr_type = SELL_PRICE_TYPE_MAP.get(fs_sell, PR_TYPE_BEST5) if is_sell else \
                  BUY_PRICE_TYPE_MAP.get(fs_buy, PR_TYPE_BEST5)
    else:
        print("%s 错误：未知的订单类型 type=%s" % (LOG_PREFIX, sig_type))
        return None

    # ---- 处理价格 ----
    try:
        price_val = float(price) if price is not None else 0
    except (ValueError, TypeError):
        price_val = 0

    if price_val > 0:
        pr_type = PR_TYPE_LIMIT
        order_price = price_val
    else:
        order_price = 0

    # ---- 有效性检查 ----
    if not is_close and (volume is None or volume <= 0):
        print("%s 跳过：下单数量/金额无效 volume=%s code=%s id=%s" % (
            LOG_PREFIX, volume, order_code, signal.get('id')))
        return None

    return {
        'op_type': op_type,
        'order_type': order_type,
        'order_code': order_code,
        'pr_type': pr_type,
        'order_price': order_price,
        'volume': volume,
        'is_close': is_close,
    }


# =====================================================================
#  模块7：下单执行模块（统一接口，分派 live/backtest）
# =====================================================================

def get_position_quantity(ContextInfo, account, stock_code):
    """
    获取当前持仓数量

    TODO: 需人工确认 QMT 持仓查询 API，以下为参考实现

    参数:
        ContextInfo: QMT 上下文
        account:     账号ID
        stock_code:  QMT 格式股票代码（如 000001.SZ）

    返回:
        持仓数量（股），查询失败返回 0
    """
    run_mode = getattr(ContextInfo, 'run_mode', RUN_MODE_BACKTEST)

    if run_mode == RUN_MODE_BACKTEST:
        # 回测模式：从模拟持仓中查询
        mock_positions = getattr(ContextInfo, 'mock_positions', {})
        return mock_positions.get(stock_code, 0)

    try:
        # TODO: 需人工确认以下 API 是否可用
        # 方式1：使用 get_trade_detail_data
        # position_data = get_trade_detail_data(account, "STOCK", "POSITION")
        # for pos in position_data:
        #     pos_code = pos.m_strInstrumentID + "." + pos.m_strExchangeID
        #     if pos_code == stock_code:
        #         return max(0, pos.m_nVolume)

        print("%s 警告：get_position_quantity 实盘未实现 stock=%s" % (
            LOG_PREFIX, stock_code))
        return 0

    except Exception as e:
        print("%s 获取持仓失败 stock=%s error=%s" % (LOG_PREFIX, stock_code, str(e)))
        return 0


def execute_order(ContextInfo, account, strategy_name, params, signal_id):
    """
    统一下单接口：根据 run_mode 分派到模拟下单或实盘下单

    参数:
        ContextInfo:   QMT 上下文
        account:       账号ID
        strategy_name: 策略名称
        params:        下单参数字典（来自 map_order_params）
        signal_id:     信号ID

    返回:
        (True, "") 或 (False, "错误信息")
    """
    run_mode = getattr(ContextInfo, 'run_mode', RUN_MODE_BACKTEST)

    if run_mode == RUN_MODE_BACKTEST:
        return mock_passorder(ContextInfo, account, strategy_name, params, signal_id)
    else:
        return real_passorder(ContextInfo, account, strategy_name, params, signal_id)


def mock_passorder(ContextInfo, account, strategy_name, params, signal_id):
    """
    模拟下单（backtest 模式）
    不调用真实 passorder，只打印日志并返回模拟结果

    参数:
        ContextInfo:   QMT 上下文
        account:       账号ID
        strategy_name: 策略名称
        params:        下单参数字典
        signal_id:     信号ID

    返回:
        (True, "") 或 (False, "错误信息")
    """
    op_type = params['op_type']
    order_type = params['order_type']
    order_code = params['order_code']
    pr_type = params['pr_type']
    order_price = params['order_price']
    volume = params['volume']
    is_close = params['is_close']

    # ---- 清仓特殊处理：查询模拟持仓 ----
    if is_close:
        pos_qty = get_position_quantity(ContextInfo, account, order_code)
        if pos_qty <= 0:
            print("%s %s 清仓跳过：无持仓 code=%s" % (LOG_PREFIX, LOG_PREFIX_BT, order_code))
            return True, ""
        sell_ratio = getattr(ContextInfo, 'cfg_sell_ratio', 100)
        volume = int(pos_qty * sell_ratio / 100.0)
        volume = max(1, volume)
        order_type = ORDER_TYPE_BY_QUANTITY
        op_type = OP_TYPE_SELL
        pr_type = SELL_PRICE_TYPE_MAP.get(
            getattr(ContextInfo, 'cfg_fs_sell', '最优五档成交'), PR_TYPE_BEST5)
        order_price = 0

    # ---- 模拟下单日志 ----
    action = "买入" if op_type == OP_TYPE_BUY else "卖出"
    order_type_desc = "按数量" if order_type == ORDER_TYPE_BY_QUANTITY else "按金额"

    print("%s %s 模拟下单: %s %s %s prType=%s price=%s volume=%s strategy=%s" % (
        LOG_PREFIX, LOG_PREFIX_BT, action, order_type_desc, order_code,
        pr_type, order_price, volume, strategy_name
    ))

    # ---- 更新模拟持仓 ----
    mock_positions = getattr(ContextInfo, 'mock_positions', {})
    if op_type == OP_TYPE_BUY:
        if order_type == ORDER_TYPE_BY_QUANTITY:
            mock_positions[order_code] = mock_positions.get(order_code, 0) + int(volume)
        # 按金额买入无法精确计算股数，跳过持仓更新
    elif op_type == OP_TYPE_SELL:
        if order_type == ORDER_TYPE_BY_QUANTITY:
            current = mock_positions.get(order_code, 0)
            mock_positions[order_code] = max(0, current - int(volume))
    ContextInfo.mock_positions = mock_positions

    # ---- 生成模拟订单ID ----
    order_id = "MOCK_%s_%s" % (signal_id, datetime.datetime.now().strftime("%H%M%S"))

    print("%s %s 模拟下单成功: orderId=%s code=%s volume=%s" % (
        LOG_PREFIX, LOG_PREFIX_BT, order_id, order_code, volume))

    return True, ""


def real_passorder(ContextInfo, account, strategy_name, params, signal_id):
    """
    实盘下单（live 模式）
    调用 QMT passorder API

    passorder 签名:
        passorder(opType, orderType, accountID, orderCode,
                  prType, price, volume, strategyName,
                  quickTrade, userOrderId, ContextInfo)

    参数:
        ContextInfo:   QMT 上下文
        account:       账号ID
        strategy_name: 策略名称
        params:        下单参数字典
        signal_id:     信号ID

    返回:
        (True, "") 或 (False, "错误信息")
    """
    op_type = params['op_type']
    order_type = params['order_type']
    order_code = params['order_code']
    pr_type = params['pr_type']
    order_price = params['order_price']
    volume = params['volume']
    is_close = params['is_close']

    # ---- 清仓特殊处理：查询持仓数量 ----
    if is_close:
        pos_qty = get_position_quantity(ContextInfo, account, order_code)
        if pos_qty <= 0:
            print("%s 清仓跳过：无持仓 code=%s" % (LOG_PREFIX, order_code))
            return True, ""
        sell_ratio = getattr(ContextInfo, 'cfg_sell_ratio', 100)
        volume = int(pos_qty * sell_ratio / 100.0)
        volume = max(1, volume)
        order_type = ORDER_TYPE_BY_QUANTITY
        op_type = OP_TYPE_SELL
        pr_type = SELL_PRICE_TYPE_MAP.get(
            getattr(ContextInfo, 'cfg_fs_sell', '最优五档成交'), PR_TYPE_BEST5)
        order_price = 0

    # ---- 生成用户订单ID ----
    now_str = datetime.datetime.now().strftime("%H%M%S")
    user_order_id = "QMT_%s_%s" % (signal_id, now_str)

    # ---- 打印下单参数 ----
    print("%s 下单参数: opType=%s orderType=%s code=%s prType=%s price=%s volume=%s strategy=%s quickTrade=%s orderId=%s" % (
        LOG_PREFIX, op_type, order_type, order_code, pr_type,
        order_price, volume, strategy_name, QUICK_TRADE_SAFE, user_order_id
    ))

    # ---- 调用 passorder ----
    try:
        passorder(
            op_type,                # opType: 买入23/卖出24
            order_type,             # orderType: 按数量1101/按金额1202
            account,                # accountID: 资金账号
            order_code,             # orderCode: 股票代码 XXXXXX.SH/SZ
            pr_type,                # prType: 委托方式
            order_price,            # price: 委托价格（0=按prType市价）
            volume,                 # volume: 委托数量/金额
            strategy_name,          # strategyName: 策略名称
            QUICK_TRADE_SAFE,       # quickTrade: 1=安全模式
            user_order_id,          # userOrderId: 用户订单ID
            ContextInfo             # ContextInfo: 上下文对象
        )
        print("%s 下单调用成功 id=%s code=%s volume=%s" % (
            LOG_PREFIX, signal_id, order_code, volume))
        return True, ""

    except Exception as e:
        error_msg = str(e)
        print("%s 下单调用异常 id=%s code=%s error=%s" % (
            LOG_PREFIX, signal_id, order_code, error_msg))
        traceback.print_exc()
        return False, error_msg


# =====================================================================
#  模块8：数据库状态更新模块（统一接口，分派 live/backtest）
# =====================================================================

def update_signal_status(ContextInfo, signal_id, flag, error_msg=None):
    """
    统一信号状态更新接口：根据 run_mode 分派到数据库或内存更新

    无论成功或失败，都将 guoqi 设为 1，防止 tick 重复处理

    参数:
        ContextInfo: QMT 上下文
        signal_id:   信号主键 ID
        flag:        执行状态（1=成功, -1=失败, 0=未成交）
        error_msg:   错误信息（可选）

    返回:
        True=更新成功, False=更新失败
    """
    run_mode = getattr(ContextInfo, 'run_mode', RUN_MODE_BACKTEST)

    if run_mode == RUN_MODE_BACKTEST:
        return update_mock_signal_status(ContextInfo, signal_id, flag, error_msg)
    else:
        return update_db_signal_status(ContextInfo.db_conn, signal_id, flag)


def update_db_signal_status(conn, signal_id, flag):
    """
    更新 SQL Server 数据库中的信号状态（live 模式）

    参数:
        conn:      数据库连接
        signal_id: 信号主键 ID
        flag:      执行状态

    返回:
        True=更新成功, False=更新失败
    """
    if conn is None:
        print("%s 错误：数据库连接为空，无法更新状态 id=%s" % (LOG_PREFIX, signal_id))
        return False

    try:
        cursor = conn.cursor()

        sql = """
        UPDATE dbo.trade
        SET guoqi = 1,
            flag = %s,
            zhixing_time = GETDATE()
        WHERE id = %s
        """ % (SQL_PARAM, SQL_PARAM)

        cursor.execute(sql, (flag, signal_id))
        conn.commit()
        cursor.close()

        print("%s 状态更新成功 id=%s flag=%s" % (LOG_PREFIX, signal_id, flag))
        return True

    except Exception as e:
        print("%s 状态更新失败 id=%s flag=%s error=%s" % (
            LOG_PREFIX, signal_id, flag, str(e)))
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def update_mock_signal_status(ContextInfo, signal_id, flag, error_msg=None):
    """
    更新内存中模拟信号的状态（backtest 模式）

    参数:
        ContextInfo: QMT 上下文
        signal_id:   信号主键 ID
        flag:        执行状态
        error_msg:   错误信息（可选）

    返回:
        True=更新成功
    """
    mock_signals = getattr(ContextInfo, 'mock_trade_signals', [])

    for sig in mock_signals:
        if sig.get('id') == signal_id:
            sig['guoqi'] = 1
            sig['flag'] = flag
            sig['zhixing_time'] = datetime.datetime.now()
            if error_msg:
                sig['error_msg'] = error_msg
            print("%s %s 模拟状态更新: id=%s flag=%s" % (
                LOG_PREFIX, LOG_PREFIX_BT, signal_id, flag))
            return True

    print("%s %s 模拟状态更新警告：未找到信号 id=%s" % (
        LOG_PREFIX, LOG_PREFIX_BT, signal_id))
    return False


# =====================================================================
#  模块9：日志与异常处理模块
# =====================================================================

def log_signal(signal, action, detail="", run_mode=RUN_MODE_LIVE):
    """
    记录信号处理日志

    参数:
        signal:   信号字典
        action:   动作描述
        detail:   补充详情
        run_mode: 运行模式
    """
    signal_id = signal.get('id', '?')
    code = signal.get('code', '?')
    sig_type = signal.get('type', '?')
    num = signal.get('num', '?')
    price = signal.get('price', '?')
    fenlei = signal.get('fenlei', '?')

    prefix = LOG_PREFIX_BT if run_mode == RUN_MODE_BACKTEST else LOG_PREFIX

    msg = "%s %s: id=%s code=%s type=%s num=%s price=%s fenlei=%s"
    args = (prefix, action, signal_id, code, sig_type, num, price, fenlei)
    if detail:
        msg += " %s"
        args += (detail,)

    print(msg % args)


# =====================================================================
#  模块10：防重复执行机制
# =====================================================================

def cleanup_processed_ids(processed_ids, max_size=10000, keep_size=5000):
    """
    清理过大的已处理信号缓存，防止内存泄漏

    参数:
        processed_ids: 已处理信号 ID 集合
        max_size:      触发清理的阈值
        keep_size:     清理后保留的数量

    返回:
        清理后的集合
    """
    if len(processed_ids) > max_size:
        id_list = list(processed_ids)
        processed_ids = set(id_list[-keep_size:])
        print("%s 清理已处理缓存：保留最近%d条" % (LOG_PREFIX, keep_size))
    return processed_ids


# =====================================================================
#  模块11：模拟交易数据生成模块（仅 backtest 模式使用）
# =====================================================================

def generate_mock_trade_signals(ContextInfo):
    """
    生成模拟交易信号，覆盖10种测试场景

    字段结构必须与 SQL Server 的 dbo.trade 表一致

    测试场景:
        1. 正常按金额买入
        2. 正常按股数买入
        3. 清仓信号
        4. 卖出部分仓位
        5. 已过期信号（应跳过）
        6. 已处理信号（应跳过）
        7. 策略分类不匹配（应跳过）
        8. 股票代码格式异常（测试自动修正）
        9. price 非 NULL（限价委托）
        10. 重复 id（防重复执行）

    参数:
        ContextInfo: QMT 上下文

    返回:
        模拟信号列表
    """
    fenlei = getattr(ContextInfo, 'cfg', {}).get('fenlei', 'eagles')
    today = datetime.datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

    signals = [
        # ---- 场景1：正常按金额买入 ----
        {
            "id": 1,
            "name": "平安银行",
            "code": "000001.SZ",
            "price": None,
            "jiner": 10000,
            "num": 10000,
            "date": today,
            "guoqi": 0,
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order_value"
        },
        # ---- 场景2：正常按股数买入 ----
        {
            "id": 2,
            "name": "贵州茅台",
            "code": "600519.SH",
            "price": None,
            "jiner": 0,
            "num": 100,
            "date": today + datetime.timedelta(minutes=5),
            "guoqi": 0,
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order"
        },
        # ---- 场景3：清仓信号 ----
        {
            "id": 3,
            "name": "招商银行",
            "code": "600036.SH",
            "price": None,
            "jiner": 0,
            "num": 0,
            "date": today + datetime.timedelta(minutes=10),
            "guoqi": 0,
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order_target_value"
        },
        # ---- 场景4：卖出部分仓位 ----
        {
            "id": 4,
            "name": "中国平安",
            "code": "601318.SH",
            "price": None,
            "jiner": 0,
            "num": -200,
            "date": today + datetime.timedelta(minutes=15),
            "guoqi": 0,
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order"
        },
        # ---- 场景5：已过期信号（应跳过）----
        {
            "id": 5,
            "name": "万科A",
            "code": "000002.SZ",
            "price": None,
            "jiner": 5000,
            "num": 5000,
            "date": today + datetime.timedelta(minutes=20),
            "guoqi": 1,  # 已过期
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order_value"
        },
        # ---- 场景6：已处理信号（应跳过）----
        {
            "id": 6,
            "name": "海康威视",
            "code": "002415.SZ",
            "price": None,
            "jiner": 8000,
            "num": 8000,
            "date": today + datetime.timedelta(minutes=25),
            "guoqi": 0,
            "flag": 1,  # 已处理
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order_value"
        },
        # ---- 场景7：策略分类不匹配（应跳过）----
        {
            "id": 7,
            "name": "宁德时代",
            "code": "300750.SZ",
            "price": None,
            "jiner": 20000,
            "num": 20000,
            "date": today + datetime.timedelta(minutes=30),
            "guoqi": 0,
            "flag": None,
            "fenlei": "other_strategy",  # 不匹配
            "zhixing_time": None,
            "type": "order_value"
        },
        # ---- 场景8：股票代码格式异常（测试自动修正）----
        {
            "id": 8,
            "name": "比亚迪",
            "code": "002594",  # 缺少 .SZ 后缀
            "price": None,
            "jiner": 15000,
            "num": 15000,
            "date": today + datetime.timedelta(minutes=35),
            "guoqi": 0,
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order_value"
        },
        # ---- 场景9：price 非 NULL（限价委托）----
        {
            "id": 9,
            "name": "五粮液",
            "code": "000858.SZ",
            "price": 150.50,  # 指定价格
            "jiner": 0,
            "num": 200,
            "date": today + datetime.timedelta(minutes=40),
            "guoqi": 0,
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order"
        },
        # ---- 场景10：重复 id（防重复执行）----
        # 与场景1相同的 id=1，但不同内容
        {
            "id": 1,  # 重复 id
            "name": "平安银行",
            "code": "000001.SZ",
            "price": None,
            "jiner": 20000,
            "num": 20000,
            "date": today + datetime.timedelta(minutes=45),
            "guoqi": 0,
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order_value"
        },
        # ---- 额外场景：按金额卖出 ----
        {
            "id": 11,
            "name": "招商银行",
            "code": "600036.SH",
            "price": None,
            "jiner": 0,
            "num": -5000,
            "date": today + datetime.timedelta(minutes=50),
            "guoqi": 0,
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order_value"
        },
        # ---- 额外场景：目标市值买入 ----
        {
            "id": 12,
            "name": "中信证券",
            "code": "600030.SH",
            "price": None,
            "jiner": 0,
            "num": 30000,
            "date": today + datetime.timedelta(minutes=55),
            "guoqi": 0,
            "flag": None,
            "fenlei": fenlei,
            "zhixing_time": None,
            "type": "order_target_value"
        },
    ]

    return signals


# =====================================================================
#  模块12：K 线日期获取模块（backtest 模式使用）
# =====================================================================

def get_current_bar_datetime(ContextInfo):
    """
    获取当前 K 线的日期时间

    TODO: 根据 QMT 实际 API 获取当前 bar 时间
    可选方案：
      1. ContextInfo.get_bar_timetag(ContextInfo.barpos)
      2. ContextInfo.get_bar_timetag()
      3. 其他 QMT 时间函数

    参数:
        ContextInfo: QMT 上下文

    返回:
        datetime.datetime 对象
    """
    try:
        # TODO: 尝试 QMT API 获取 bar 时间
        # 方式1：get_bar_timetag 返回毫秒时间戳
        # timetag = ContextInfo.get_bar_timetag(ContextInfo.barpos)
        # if timetag:
        #     return datetime.datetime.fromtimestamp(timetag / 1000.0)

        # 方式2：直接从 ContextInfo 获取
        # bar_time = ContextInfo.get_time()
        # if bar_time:
        #     return bar_time

        # 回退：使用当前系统时间
        return datetime.datetime.now()

    except Exception as e:
        print("%s %s 获取 bar 时间失败，使用系统时间 error=%s" % (
            LOG_PREFIX, LOG_PREFIX_BT, str(e)))
        return datetime.datetime.now()


# =====================================================================
#  模块13：回测报告模块
# =====================================================================

def init_backtest_report(ContextInfo):
    """
    初始化回测报告数据结构

    参数:
        ContextInfo: QMT 上下文
    """
    ContextInfo.backtest_report = []
    ContextInfo.backtest_stats = {
        "total_signals": 0,
        "processed_signals": 0,
        "success_signals": 0,
        "skipped_signals": 0,
        "failed_signals": 0,
        "fail_reasons": {},
    }
    ContextInfo.bar_count = 0


def append_backtest_report(ContextInfo, bar_time, signal, action, status, reason=""):
    """
    追加回测报告条目

    参数:
        ContextInfo: QMT 上下文
        bar_time:    当前 bar 时间
        signal:      信号字典
        action:      动作（BUY/SELL/CLOSE/SKIP）
        status:      状态（SUCCESS/FAIL/SKIP）
        reason:      原因
    """
    entry = {
        "bar_time": str(bar_time),
        "signal_id": signal.get('id', '?'),
        "code": signal.get('code', '?'),
        "type": signal.get('type', '?'),
        "num": signal.get('num', '?'),
        "action": action,
        "status": status,
        "reason": reason,
    }
    ContextInfo.backtest_report.append(entry)

    # 更新统计
    stats = ContextInfo.backtest_stats
    stats["total_signals"] += 1

    if status == "SUCCESS":
        stats["success_signals"] += 1
        stats["processed_signals"] += 1
    elif status == "FAIL":
        stats["failed_signals"] += 1
        stats["processed_signals"] += 1
        # 记录失败原因
        reason_key = reason[:50] if reason else "unknown"
        stats["fail_reasons"][reason_key] = stats["fail_reasons"].get(reason_key, 0) + 1
    elif status == "SKIP":
        stats["skipped_signals"] += 1


def print_backtest_summary(ContextInfo):
    """
    打印回测摘要报告

    参数:
        ContextInfo: QMT 上下文
    """
    stats = getattr(ContextInfo, 'backtest_stats', {})
    if not stats:
        return

    print("=" * 60)
    print("%s %s 回测摘要报告" % (LOG_PREFIX, LOG_PREFIX_BT))
    print("=" * 60)
    print("  总信号数:       %d" % stats.get("total_signals", 0))
    print("  已处理信号数:   %d" % stats.get("processed_signals", 0))
    print("  成功模拟下单数: %d" % stats.get("success_signals", 0))
    print("  跳过信号数:     %d" % stats.get("skipped_signals", 0))
    print("  失败信号数:     %d" % stats.get("failed_signals", 0))

    fail_reasons = stats.get("fail_reasons", {})
    if fail_reasons:
        print("  失败原因汇总:")
        for reason, count in fail_reasons.items():
            print("    - %s: %d次" % (reason, count))

    # 打印已处理信号 ID 集合
    processed_ids = getattr(ContextInfo, 'processed_ids', set())
    print("  已处理信号ID:   %s" % sorted(processed_ids))

    # 打印模拟持仓
    mock_positions = getattr(ContextInfo, 'mock_positions', {})
    if mock_positions:
        print("  模拟持仓:")
        for code, qty in mock_positions.items():
            print("    - %s: %d股" % (code, qty))

    print("=" * 60)


# =====================================================================
#  QMT 核心函数：init(ContextInfo)
# =====================================================================

def init(ContextInfo):
    """
    QMT 策略初始化函数，在策略启动时调用一次

    完成以下初始化:
        1. 读取 QMT XML 注入的变量
        2. 初始化运行模式（live/backtest）
        3. 初始化数据库连接配置（live 模式）
        4. 初始化账号信息
        5. 调用 ContextInfo.set_account(account)
        6. 初始化策略名 strategyName
        7. 初始化已处理信号缓存
        8. 初始化委托方式映射表
        9. 初始化运行时间窗口
        10. 初始化模拟数据（backtest 模式）
        11. 初始化回测报告（backtest 模式）
    """
    print("=" * 60)
    print("%s 策略初始化开始 v2.0" % LOG_PREFIX)
    print("=" * 60)

    # ---- 1. 读取 QMT XML 注入的变量 ----
    cfg = read_config(ContextInfo)
    ContextInfo.cfg = cfg

    # ---- 2. 初始化运行模式 ----
    ContextInfo.run_mode = cfg.get('run_mode', RUN_MODE_BACKTEST)
    is_backtest = (ContextInfo.run_mode == RUN_MODE_BACKTEST)

    # ---- 打印配置参数 ----
    print("%s 配置参数:" % LOG_PREFIX)
    print("  运行模式:       %s%s" % (
        ContextInfo.run_mode,
        " (模拟模式，不会真实下单)" if is_backtest else " (实盘模式！)"
    ))
    print("  服务器地址:     %s" % cfg['db_server'])
    print("  数据库用户:     %s" % cfg['db_user'])
    print("  数据库密码:     %s" % ("*" * len(cfg['db_password'])))
    print("  开始时间:       %s" % cfg['start_time'])
    print("  结束时间:       %s" % cfg['end_time'])
    print("  买入金额上限:   %s" % cfg['Buy_Amount'])
    print("  最多买入股票数: %s" % cfg['Buy_num'])
    print("  卖出比例:       %s%%" % cfg['sell_ratio'])
    print("  跟单比例:       %s" % cfg['bili'])
    print("  策略分类:       %s" % cfg['fenlei'])
    print("  买入委托方式:   %s" % cfg['fs_buy'])
    print("  卖出委托方式:   %s" % cfg['fs_sell'])

    # ---- 3. 初始化数据库连接（live 模式）----
    if not is_backtest:
        ContextInfo.db_conn = get_db_connection(
            server=cfg['db_server'],
            user=cfg['db_user'],
            password=cfg['db_password'],
            database=DB_NAME
        )
        if ContextInfo.db_conn is None:
            print("%s 严重错误：数据库连接失败！策略将以空连接运行，无法处理信号。" % LOG_PREFIX)
    else:
        ContextInfo.db_conn = None
        print("%s %s 回测模式：跳过数据库连接" % (LOG_PREFIX, LOG_PREFIX_BT))

    # ---- 4. 初始化账号信息 ----
    # TODO: 需人工确认 QMT 账号属性名（accountid 或 account）
    account = str(getattr(ContextInfo, 'accountid', '')).strip()
    if not account:
        account = str(getattr(ContextInfo, 'account', '')).strip()
    if not account:
        print("%s 警告：未检测到账号信息，请确认 QMT 策略交易窗口已绑定账号" % LOG_PREFIX)
    ContextInfo.account = account

    # ---- 5. 调用 set_account ----
    if account:
        try:
            ContextInfo.set_account(account)
            print("%s 账号绑定成功: %s" % (LOG_PREFIX, account))
        except Exception as e:
            print("%s set_account 异常: %s" % (LOG_PREFIX, str(e)))
    else:
        print("%s 警告：账号为空，跳过 set_account" % LOG_PREFIX)

    # ---- 6. 初始化策略名 ----
    ContextInfo.strategy_name = "QMT跟单策略v2.0"

    # ---- 7. 初始化已处理信号缓存 ----
    ContextInfo.processed_ids = set()
    ContextInfo.buy_count = 0  # 当前已买入股票计数

    # ---- 8. 缓存委托方式到 ContextInfo（供执行模块读取）----
    ContextInfo.cfg_fs_buy = cfg['fs_buy']
    ContextInfo.cfg_fs_sell = cfg['fs_sell']
    ContextInfo.cfg_sell_ratio = cfg['sell_ratio']

    # ---- 9. 初始化运行时间窗口 ----
    ContextInfo.start_time = cfg['start_time']
    ContextInfo.end_time = cfg['end_time']

    # ---- 10. 初始化模拟数据（backtest 模式）----
    if is_backtest:
        ContextInfo.mock_trade_signals = generate_mock_trade_signals(ContextInfo)
        ContextInfo.mock_positions = {}
        print("%s %s 已生成 %d 条模拟信号" % (
            LOG_PREFIX, LOG_PREFIX_BT, len(ContextInfo.mock_trade_signals)))

    # ---- 11. 初始化回测报告（backtest 模式）----
    if is_backtest:
        init_backtest_report(ContextInfo)

    print("=" * 60)
    print("%s 策略初始化完成 v2.0 run_mode=%s account=%s" % (
        LOG_PREFIX, ContextInfo.run_mode, ContextInfo.account))
    print("=" * 60)


# =====================================================================
#  QMT 核心函数：handlebar(ContextInfo)
# =====================================================================

def handlebar(ContextInfo):
    """
    QMT 策略主循环函数，每个 tick/bar 调用一次

    核心流程:
        1. 仅在 is_last_bar() 且非历史 bar 时执行（live 模式）
           backtest 模式下每个 bar 都执行
        2. 判断交易时间窗口
        3. 获取当前 bar 日期时间
        4. 查询信号（统一接口，分派 live/backtest）
        5. 逐条处理信号：校验 → 映射 → 下单 → 更新状态
        6. 定期打印回测报告（backtest 模式）
    """
    run_mode = getattr(ContextInfo, 'run_mode', RUN_MODE_BACKTEST)
    is_backtest = (run_mode == RUN_MODE_BACKTEST)

    # ---- 1. bar 状态检查 ----
    if not is_backtest:
        # 实盘模式：仅在 is_last_bar() 为 True 时执行
        if not is_last_bar():
            return

    # ---- 2. 交易时间判断 ----
    now = datetime.datetime.now()
    now_time_str = now.strftime("%H%M%S")
    start_time = getattr(ContextInfo, 'start_time', '093000')
    end_time = getattr(ContextInfo, 'end_time', '153000')

    if not is_trading_time(now_time_str, start_time, end_time):
        return

    # ---- 3. 获取当前 bar 日期时间 ----
    bar_datetime = get_current_bar_datetime(ContextInfo)
    bar_date = bar_datetime.date() if hasattr(bar_datetime, 'date') else None

    # ---- 4. 数据库连接检查（live 模式）----
    if not is_backtest:
        conn = getattr(ContextInfo, 'db_conn', None)
        if conn is None or not check_db_connection(conn):
            print("%s 数据库连接断开，尝试重连..." % LOG_PREFIX)
            cfg = getattr(ContextInfo, 'cfg', {})
            ContextInfo.db_conn = get_db_connection(
                server=cfg.get('db_server', ''),
                user=cfg.get('db_user', DB_USER_DEFAULT),
                password=cfg.get('db_password', ''),
                database=DB_NAME
            )
            if ContextInfo.db_conn is None:
                print("%s 重连失败，本次 tick 跳过" % LOG_PREFIX)
                return

    # ---- 5. 查询信号 ----
    cfg = getattr(ContextInfo, 'cfg', {})
    fenlei = cfg.get('fenlei', '')
    signals = query_signals(ContextInfo, fenlei, bar_date)

    if not signals:
        return

    print("%s 查询到 %d 条待处理信号 (run_mode=%s)" % (
        LOG_PREFIX, len(signals), run_mode))

    # ---- 6. 逐条处理信号 ----
    processed_ids = getattr(ContextInfo, 'processed_ids', set())
    buy_count = getattr(ContextInfo, 'buy_count', 0)
    account = getattr(ContextInfo, 'account', '')
    strategy_name = getattr(ContextInfo, 'strategy_name', 'QMT跟单策略')

    Buy_Amount = cfg.get('Buy_Amount', 0)
    Buy_num = cfg.get('Buy_num', 100)
    sell_ratio = cfg.get('sell_ratio', 100)
    bili = cfg.get('bili', 1)
    fs_buy = cfg.get('fs_buy', '最优五档成交')
    fs_sell = cfg.get('fs_sell', '最优五档成交')

    for signal in signals:
        signal_id = signal.get('id')
        sig_type = str(signal.get('type', '')).strip().lower()
        num_val = 0
        try:
            num_val = float(signal.get('num', 0)) if signal.get('num') is not None else 0
        except (ValueError, TypeError):
            pass
        is_sell = num_val < 0
        is_close = (sig_type == 'order_target_value' and num_val == 0)

        # ---- 6a. 校验信号 ----
        valid, reason = validate_signal(
            signal, processed_ids, Buy_Amount, Buy_num, buy_count, bili
        )

        if not valid:
            log_signal(signal, "SKIP", reason, run_mode)
            if is_backtest:
                append_backtest_report(ContextInfo, bar_datetime, signal, "SKIP", "SKIP", reason)
            # 即使跳过，也标记为已处理（防止重复校验）
            processed_ids.add(signal_id)
            continue

        # ---- 6b. 先标记 id 到 processed_ids（幂等保护：先标记后执行）----
        processed_ids.add(signal_id)

        # ---- 6c. 映射下单参数 ----
        params = map_order_params(signal, fs_buy, fs_sell, Buy_Amount, sell_ratio, bili)

        if params is None:
            log_signal(signal, "SKIP", "参数映射失败", run_mode)
            update_signal_status(ContextInfo, signal_id, -1, "参数映射失败")
            if is_backtest:
                append_backtest_report(ContextInfo, bar_datetime, signal, "SKIP", "FAIL", "参数映射失败")
            continue

        # ---- 6d. 执行下单 ----
        success, error_msg = execute_order(
            ContextInfo, account, strategy_name, params, signal_id
        )

        # ---- 6e. 更新信号状态 ----
        if success:
            flag = 1  # 成功
            action = "BUY" if params['op_type'] == OP_TYPE_BUY else "SELL"
            if params['is_close']:
                action = "CLOSE"
            log_signal(signal, action, "下单成功", run_mode)

            # 更新买入计数
            if not is_sell and not is_close:
                buy_count += 1
        else:
            flag = -1  # 失败
            log_signal(signal, "FAIL", error_msg, run_mode)

        update_signal_status(ContextInfo, signal_id, flag, error_msg)

        # ---- 6f. 回测报告 ----
        if is_backtest:
            status = "SUCCESS" if success else "FAIL"
            action = "BUY" if params['op_type'] == OP_TYPE_BUY else "SELL"
            if params['is_close']:
                action = "CLOSE"
            append_backtest_report(ContextInfo, bar_datetime, signal, action, status,
                                   "" if success else (error_msg or ""))

    # ---- 7. 保存运行状态回 ContextInfo ----
    ContextInfo.processed_ids = processed_ids
    ContextInfo.buy_count = buy_count

    # ---- 8. 清理缓存 ----
    ContextInfo.processed_ids = cleanup_processed_ids(ContextInfo.processed_ids)

    # ---- 9. 定期打印回测报告（backtest 模式）----
    if is_backtest:
        bar_count = getattr(ContextInfo, 'bar_count', 0) + 1
        ContextInfo.bar_count = bar_count

        if bar_count % BACKTEST_REPORT_INTERVAL == 0:
            print_backtest_summary(ContextInfo)


# =====================================================================
#  QMT 回测结束回调（可选）
# =====================================================================

def after_backtest(ContextInfo):
    """
    回测结束后调用，打印最终回测报告
    仅在 backtest 模式下有效
    """
    run_mode = getattr(ContextInfo, 'run_mode', RUN_MODE_BACKTEST)
    if run_mode == RUN_MODE_BACKTEST:
        print("\n")
        print("*" * 60)
        print("%s %s 回测结束，最终报告" % (LOG_PREFIX, LOG_PREFIX_BT))
        print("*" * 60)
        print_backtest_summary(ContextInfo)

        # 打印详细回测条目
        report = getattr(ContextInfo, 'backtest_report', [])
        if report:
            print("\n%s %s 详细回测记录:" % (LOG_PREFIX, LOG_PREFIX_BT))
            print("-" * 60)
            for i, entry in enumerate(report, 1):
                print("  [%02d] bar=%s id=%s code=%s type=%s num=%s action=%s status=%s %s" % (
                    i,
                    entry.get('bar_time', ''),
                    entry.get('signal_id', ''),
                    entry.get('code', ''),
                    entry.get('type', ''),
                    entry.get('num', ''),
                    entry.get('action', ''),
                    entry.get('status', ''),
                    ("reason=%s" % entry['reason']) if entry.get('reason') else ""
                ))
            print("-" * 60)
