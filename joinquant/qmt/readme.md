# QMT 实时跟单交易插件 — 设计说明与使用文档

> 版本：v2.0（支持回测/调试模式）  
> 文件：[`qmt_trade_follow.py`](qmt_trade_follow.py)  
> 配置：[`聚宽QMT信号买卖sql.xml`](聚宽QMT信号买卖sql.xml)

---

## A. 设计说明

### A.1 整体架构

本插件采用 **信号驱动 + 幂等处理 + 模式分派** 架构，核心流程为：

```
QMT tick/bar 驱动 → handlebar() → 时间判断 → 查询信号(统一接口) → 校验 → 映射参数 → 下单(统一接口) → 更新状态(统一接口)
```

每个 tick 周期（约 3 秒）执行一次 `handlebar()`，仅在 `is_last_bar()` 为 True 的实时行情阶段触发交易逻辑，避免历史 K 线重复下单。

**v2.0 新增**：通过 `run_mode` 参数（`live`/`backtest`）实现统一信号处理流程，所有核心接口（查询、下单、状态更新）均根据模式自动分派：

```
                  ┌─────────────────────────────────────┐
                  │         handlebar(ContextInfo)      │
                  └──────────────┬──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   run_mode = ?          │
                    └───┬─────────────────┬───┘
                        │                 │
                 backtest               live
                        │                 │
            ┌───────────▼────────┐  ┌──────▼───────────┐
            │ query_mock_signals │  │ query_db_signals │
            │ mock_passorder     │  │ real_passorder   │
            │ update_mock_status │  │ update_db_status │
            └────────────────────┘  └──────────────────┘
```

### A.2 模块划分

| 模块编号 | 模块名称 | 核心函数 | 职责 |
|---------|---------|---------|------|
| 1 | 参数读取 | `read_config()` | 从 QMT XML 注入的 ContextInfo 属性读取配置 |
| 2 | 数据库连接 | `get_db_connection()`, `check_db_connection()` | 建立/检查 SQL Server 连接，支持重试 |
| 3 | 交易时间判断 | `is_trading_time()` | 判断当前是否在配置的时间窗口内 |
| 4 | 信号查询 | `query_signals()`, `query_db_signals()`, `query_mock_signals()` | 统一查询接口，分派到数据库或模拟数据 |
| 5 | 信号校验 | `validate_signal()` | 校验信号合法性、幂等性、风控限制 |
| 6 | QMT 下单参数映射 | `map_order_params()`, `normalize_code()` | 将业务信号映射为 passorder 参数 |
| 7 | 下单执行 | `execute_order()`, `mock_passorder()`, `real_passorder()`, `get_position_quantity()` | 统一下单接口，分派到模拟或实盘 |
| 8 | 数据库状态更新 | `update_signal_status()`, `update_db_signal_status()`, `update_mock_signal_status()` | 统一状态更新接口，分派到数据库或内存 |
| 9 | 日志与异常处理 | `log_signal()` | 统一日志格式 |
| 10 | 防重复执行机制 | `cleanup_processed_ids()` | 内存缓存清理，防止泄漏 |
| 11 | **模拟交易数据生成** | `generate_mock_trade_signals()` | 生成覆盖10种测试场景的模拟信号（v2.0 新增） |
| 12 | **K 线日期获取** | `get_current_bar_datetime()` | 获取当前 bar 日期时间，用于回测信号过滤（v2.0 新增） |
| 13 | **回测报告** | `init_backtest_report()`, `append_backtest_report()`, `print_backtest_summary()` | 回测统计与报告输出（v2.0 新增） |

### A.3 关键设计决策

1. **双层数幂等保护**：内存 `processed_ids` 集合 + 数据库 `guoqi` 字段，确保同一信号不会被重复处理
2. **先标记后执行**：在调用 `passorder` 之前先将 `signal_id` 加入 `processed_ids`，即使下单异常也不会重复尝试
3. **guoqi 必更新**：无论下单成功或失败，都将 `guoqi` 设为 1，防止 tick 重复处理导致无限下单
4. **连接自愈**：每次 `handlebar()` 检查数据库连接有效性，断线自动重连
5. **驱动兼容**：优先使用 `pyodbc`，备选 `pymssql`，运行时自动检测
6. **统一接口分派**（v2.0 新增）：`query_signals()`、`execute_order()`、`update_signal_status()` 三个核心接口根据 `run_mode` 自动分派，live 和 backtest 共享同一套信号处理流程
7. **默认安全模式**（v2.0 新增）：`run_mode` 默认为 `backtest`，必须显式切换为 `live` 才会真实下单

### A.4 数据流图

```
┌──────────────┐                    ┌──────────────────┐
│  SQL Server  │  (live 模式)       │  模拟信号数据     │  (backtest 模式)
│  touzi.dbo   │                    │  mock_trade_     │
│   .trade     │                    │  signals         │
└──────┬───────┘                    └──────┬───────────┘
       │ SELECT (guoqi=0, flag=NULL)       │ 内存过滤
       ▼                                   ▼
┌───────────────┐       ┌──────────────────┐
│ query_signals │ ────▶│ validate_signal   │
│ (统一接口)     │       └──────┬───────────┘
└───────────────┘              │ valid
                               ▼
                        ┌────────────────┐
                        │map_order_params│
                        └──────┬─────────┘
                               │ params dict
                               ▼
                        ┌──────────────────┐
                        │ execute_order    │──▶ mock_passorder (backtest)
                        │ (统一接口)        │──▶ real_passorder (live)
                        └──────┬───────────┘
                               │ result
                               ▼
                        ┌────────────────────┐
                        │update_signal_status│──▶ update_mock_signal_status (backtest)
                        │(统一接口)           │──▶ update_db_signal_status (live)
                        └────────────────────┘
```

---

## B. 完整 QMT Python 策略代码

见 [`qmt_trade_follow.py`](qmt_trade_follow.py)

### B.1 代码结构概览

```python
#coding:gbk                          # QMT 规范：首行 GBK 编码声明

# === 导入区 ===
import datetime, traceback, re, sys, copy
import pyodbc / pymssql               # 数据库驱动（自动检测）

# === 常量定义区 ===
OP_TYPE_BUY = 23                      # passorder opType
OP_TYPE_SELL = 24
ORDER_TYPE_BY_QUANTITY = 1101         # passorder orderType
ORDER_TYPE_BY_AMOUNT = 1202
PR_TYPE_LIMIT = 0                     # passorder prType
PR_TYPE_LATEST = 5
PR_TYPE_LIMIT_UP_DOWN = 6
PR_TYPE_COUNTERPART = 7
PR_TYPE_BEST5 = 2
QUICK_TRADE_SAFE = 1

RUN_MODE_LIVE = "live"                # v2.0: 运行模式常量
RUN_MODE_BACKTEST = "backtest"

BUY_PRICE_TYPE_MAP = {...}            # 买入委托方式映射
SELL_PRICE_TYPE_MAP = {...}           # 卖出委托方式映射

# === 13个功能模块 ===
def read_config(ContextInfo): ...                  # 模块1: 参数读取
def get_db_connection(...): ...                     # 模块2: 数据库连接
def check_db_connection(conn): ...                  # 模块2
def is_trading_time(...): ...                       # 模块3: 交易时间判断
def query_signals(ContextInfo, fenlei, bar_date): ..# 模块4: 统一信号查询
def query_db_signals(conn, fenlei): ...             # 模块4: DB查询
def query_mock_signals(ContextInfo, fenlei, bar_date): . # 模块4: 模拟查询
def validate_signal(...): ...                       # 模块5: 信号校验
def normalize_code(code): ...                       # 模块6: 代码规范化
def map_order_params(...): ...                      # 模块6: 参数映射
def get_position_quantity(ContextInfo, ...): ...    # 模块7: 持仓查询
def execute_order(ContextInfo, ...): ...            # 模块7: 统一下单
def mock_passorder(ContextInfo, ...): ...           # 模块7: 模拟下单
def real_passorder(ContextInfo, ...): ...           # 模块7: 实盘下单
def update_signal_status(ContextInfo, ...): ...     # 模块8: 统一状态更新
def update_db_signal_status(conn, ...): ...         # 模块8: DB状态更新
def update_mock_signal_status(ContextInfo, ...): .. # 模块8: 模拟状态更新
def log_signal(...): ...                            # 模块9: 日志
def cleanup_processed_ids(...): ...                 # 模块10: 防重复
def generate_mock_trade_signals(ContextInfo): ...   # 模块11: 模拟数据生成
def get_current_bar_datetime(ContextInfo): ...      # 模块12: K线日期
def init_backtest_report(ContextInfo): ...          # 模块13: 回测报告
def append_backtest_report(ContextInfo, ...): ...   # 模块13
def print_backtest_summary(ContextInfo): ...        # 模块13

# === QMT 核心函数 ===
def init(ContextInfo): ...                          # 初始化
def handlebar(ContextInfo): ...                     # 主循环
def after_backtest(ContextInfo): ...                # 回测结束回调
```

### B.2 编码注意事项

- 本文件在 VS Code 中以 UTF-8 保存
- 导入 QMT 编辑器后，请 **另存为 GBK 编码**（QMT 要求首行 `#coding:gbk`）
- 中文注释在 GBK 环境下可正常显示
- 如遇编码错误，可用 QMT 编辑器打开后重新保存

---

## C. XML 参数与 Python 变量对应表

| XML bind 属性 | XML name（界面标签） | 默认值 | Python 变量（cfg 字典键） | 类型 | 说明 |
|--------------|-------------------|--------|------------------------|------|------|
| `name` | 服务器地址 | `49.234.12.146` | `cfg['db_server']` | str | SQL Server 地址 |
| `pws` | 数据库密码 | `Fyyunfei336...` | `cfg['db_password']` | str | 数据库登录密码 |
| `start_time` | 开始时间 | `093000` | `cfg['start_time']` | str | 格式 HHMMSS |
| `end_time` | 结束时间 | `153000` | `cfg['end_time']` | str | 格式 HHMMSS |
| `Buy_Amount` | 买入金额 | `0` | `cfg['Buy_Amount']` | float | 0=不限制，使用数据库 num |
| `Buy_num` | 最多买入股票数 | `100` | `cfg['Buy_num']` | int | 单次运行最多买入不同股票数 |
| `sell_ratio` | 卖出比例 | `100` | `cfg['sell_ratio']` | float | 百分比，100=全卖 |
| `bili` | 跟单比例 | `1` | `cfg['bili']` | float | 1=全跟，0.5=跟一半 |
| `username` | 用户名 | `qmt` | `cfg['db_user']` | str | 数据库用户名 |
| `fenlei` | 策略分类 | `未分类` | `cfg['fenlei']` | str | 只处理匹配的信号 |
| `fs_sell` | 卖出委托方式 | `最优五档成交` | `cfg['fs_sell']` | str | combo 下拉选择 |
| `fs_buy` | 买入委托方式 | `最优五档成交` | `cfg['fs_buy']` | str | combo 下拉选择 |
| `run_mode` | 运行模式 | `backtest` | `cfg['run_mode']` | str | **v2.0 新增**：`backtest`=模拟 / `live`=实盘 |

### XML 配置界面示例

```xml
<!-- 运行模式下拉框（v2.0 新增）-->
<item comboType="custom" position=""
      list="backtest,live"
      bind="run_mode" value="backtest"
      note="运行模式" name="运行模式" type="combo"/>

<!-- 买入委托方式下拉框 -->
<item comboType="custom" position=""
      list="卖五价,卖四价,卖三价,卖二价,卖一价,笼子上限,涨跌停价,对手价,最优五档成交"
      bind="fs_buy" value="最优五档成交"
      note="买入委托方式" name="买入委托方式" type="combo"/>
```

---

## D. trade 表字段与业务逻辑对应表

| 字段名 | 类型 | 业务含义 | 在策略中的用途 |
|-------|------|---------|--------------|
| `id` | int, PK | 信号唯一标识 | 幂等检查主键；UPDATE WHERE 条件；userOrderId 组成部分 |
| `name` | nvarchar(50) | 股票名称 | 日志记录（不参与下单逻辑） |
| `code` | nvarchar(50) | 股票代码 | 经 `normalize_code()` 转换后作为 passorder 的 orderCode |
| `price` | numeric(18,3) | 成交价/限价 | >0 时使用限价委托 prType=0；NULL/0 时使用界面委托方式 |
| `jiner` | numeric(18,2) | 成交总价 | 日志记录（当前版本未参与下单计算，金额由 num 字段决定） |
| `num` | numeric(18,0) | 下单数量/金额 | 核心字段：正数=买入，负数=卖出，0=清仓；具体含义由 type 决定 |
| `date` | datetime | 信号生成时间 | 查询条件：只取今日信号；backtest 模式按 bar_date 过滤 |
| `guoqi` | int | 过期状态 | 0=未过期（查询条件）；处理后设为1（防重复） |
| `flag` | int | 执行状态 | NULL=新鲜（查询条件）；1=成功；-1=失败；0=未成交 |
| `fenlei` | nvarchar(50) | 策略分类 | 查询条件：只处理与配置 fenlei 匹配的信号 |
| `zhixing_time` | datetime | 执行时间 | 下单后更新为 GETDATE() |
| `type` | nvarchar(50) | 订单类型 | 决定 passorder orderType 映射 |

### type 字段映射规则

| type 值 | 业务含义 | passorder orderType | num 含义 | 备注 |
|---------|---------|-------------------|---------|------|
| `order` | 按数量下单 | 1101 | 股数（正=买，负=卖） | 买入按100股取整 |
| `order_value` | 按金额下单 | 1202 | 金额（正=买，负=卖） | 买入受 Buy_Amount 限制 |
| `order_target_value` | 目标市值 | 1202 | 目标市值 | num=0 时为清仓信号 |

### num 字段决策树

```
num 值?
├── = 0  → 清仓信号（type 视为 order_target_value，查询持仓后全部卖出）
├── > 0  → 买入信号
│   ├── type=order         → 买入 num*100 股（经 bili 调整）
│   ├── type=order_value   → 买入 num 元（经 bili 调整，受 Buy_Amount 限制）
│   └── type=order_target_value → 调整持仓至 num 元
└── < 0  → 卖出信号
    ├── type=order         → 卖出 |num| 股（经 sell_ratio 调整）
    ├── type=order_value   → 卖出 |num| 元市值（经 sell_ratio 调整）
    └── type=order_target_value → 调整持仓至 |num| 元
```

### price 字段决策树

```
price 值?
├── NULL 或 0  → 使用界面委托方式（fs_buy/fs_sell）对应的 prType，price=0
└── > 0        → 限价委托 prType=0，price=数据库字段值
```

---

## E. 需要人工确认的 QMT 参数清单

> ⚠️ 以下参数无法从公开文档完全确认，**必须在实盘前逐一验证**。代码中已用 `TODO` 标注。

### E.1 passorder orderType 映射

| 业务类型 | 当前使用值 | 确认状态 | 说明 |
|---------|----------|---------|------|
| `order`（按数量下单） | 1101 | ⚠️ 待确认 | QMT 文档示例中股票买入使用 1101 |
| `order_value`（按金额下单） | 1202 | ⚠️ 待确认 | 文档提到 1202 表示按金额买入 |
| `order_target_value`（目标市值） | 1202 | ❌ 高度待确认 | 可能需要独立 orderType；当前简化为与 order_value 相同 |

### E.2 passorder opType 枚举

| 操作 | 当前使用值 | 确认状态 | 说明 |
|------|----------|---------|------|
| 股票买入 | 23 | ✅ 已确认 | QMT 文档示例明确使用 |
| 股票卖出 | 24 | ⚠️ 待确认 | 常见约定，但需验证 |
| 融资买入 | — | ❌ 未实现 | 需确认 opType |
| 融券卖出 | — | ❌ 未实现 | 需确认 opType |

### E.3 prType 委托方式映射

#### 买入委托方式 (fs_buy)

| 中文委托方式 | 当前 prType | 确认状态 |
|------------|-----------|---------|
| 卖五价 | 11 | ❌ 待确认 |
| 卖四价 | 10 | ❌ 待确认 |
| 卖三价 | 9 | ❌ 待确认 |
| 卖二价 | 8 | ❌ 待确认 |
| 卖一价 | 7 | ⚠️ 待确认（可能与对手价 prType=7 相同） |
| 笼子上限 | -1 | ❌ 待确认（价格笼子机制，prType 含义不明） |
| 涨跌停价 | 6 | ⚠️ 待确认 |
| 对手价 | 7 | ⚠️ 待确认 |
| 最优五档成交 | 2 | ⚠️ 待确认 |

#### 卖出委托方式 (fs_sell)

| 中文委托方式 | 当前 prType | 确认状态 |
|------------|-----------|---------|
| 买五价 | 15 | ❌ 待确认 |
| 买四价 | 14 | ❌ 待确认 |
| 买三价 | 13 | ❌ 待确认 |
| 买二价 | 12 | ❌ 待确认 |
| 买一价 | 7 | ⚠️ 待确认（可能与对手价 prType=7 相同） |
| 涨跌停价 | 6 | ⚠️ 待确认 |
| 对手价 | 7 | ⚠️ 待确认 |
| 最优五档成交 | 2 | ⚠️ 待确认 |

### E.4 其他待确认项

| 编号 | 待确认内容 | 影响范围 | 优先级 |
|------|----------|---------|-------|
| 1 | QMT 实盘中 account 变量是否由策略交易窗口自动注入 | `init()` 中账号获取 | 高 |
| 2 | ContextInfo 账号属性名是 `accountid` 还是 `account` | `init()` 中账号获取 | 高 |
| 3 | QMT Python 环境是否已安装 pyodbc 或 pymssql | 数据库连接 | 高 |
| 4 | `get_trade_detail_data` API 的可用性及属性名 | 清仓功能（查询持仓） | 高 |
| 5 | 持仓对象的属性名（`m_strInstrumentID` 等） | 清仓功能 | 高 |
| 6 | `ContextInfo.do_back_test` 属性名是否正确 | 回测模式判断 | 中 |
| 7 | `passorder` 的 `quickTrade=1` 在实时行情中的具体行为 | 下单触发时机 | 中 |
| 8 | 限价委托时 price 参数传 0 还是 -1 | 限价下单 | 中 |
| 9 | `order_target_value` 是否需要查询当前持仓计算差额 | 目标市值下单 | 中 |
| 10 | A股卖出数量是否需要按100股取整 | 卖出下单 | 低 |
| 11 | `get_bar_timetag()` API 的可用性及返回格式 | 回测 bar 时间获取 | 中 |

---

## F. 实盘前测试步骤

### F.1 环境准备

- [ ] **步骤1**：确认 QMT Python 环境中已安装数据库驱动
  ```python
  # 在 QMT Python 控制台中执行
  try:
      import pyodbc
      print("pyodbc 版本:", pyodbc.version)
  except ImportError:
      print("pyodbc 未安装")
  
  try:
      import pymssql
      print("pymssql 已安装")
  except ImportError:
      print("pymssql 未安装")
  ```
  如果两者都未安装，需在 QMT Python 目录中安装：
  ```
  # 找到 QMT Python 路径（通常在 QMT 安装目录下）
  # 例如: C:\国金QMT\user.x64\python.exe -m pip install pymssql
  ```

- [ ] **步骤2**：确认数据库连接
  ```python
  # 在 QMT Python 控制台中执行
  import pymssql  # 或 pyodbc
  conn = pymssql.connect(server='你的服务器', user='sa', password='你的密码', database='touzi')
  cursor = conn.cursor()
  cursor.execute("SELECT COUNT(*) FROM dbo.trade")
  print("trade 表记录数:", cursor.fetchone()[0])
  conn.close()
  ```

- [ ] **步骤3**：确认 QMT 账号获取方式
  ```python
  # 在 QMT 策略中临时添加打印
  def init(ContextInfo):
      print("accountid:", getattr(ContextInfo, 'accountid', 'NOT_FOUND'))
      print("account:", getattr(ContextInfo, 'account', 'NOT_FOUND'))
  ```

### F.2 回测模式验证（v2.0 新增）

- [ ] **步骤4**：使用 backtest 模式启动策略
  - 在 XML 界面中设置 `run_mode=backtest`
  - 启动策略，观察日志输出
  - 确认模拟信号被正确生成和处理
  - 确认不会调用真实 passorder

- [ ] **步骤5**：验证10种测试场景
  | 场景 | 信号ID | 预期行为 |
  |------|--------|---------|
  | 1. 按金额买入 | 1 | 模拟买入平安银行 10000元 |
  | 2. 按股数买入 | 2 | 模拟买入贵州茅台 100股 |
  | 3. 清仓信号 | 3 | 查询持仓后模拟卖出 |
  | 4. 卖出部分仓位 | 4 | 模拟卖出中国平安 200股 |
  | 5. 已过期信号 | 5 | 跳过（guoqi=1） |
  | 6. 已处理信号 | 6 | 跳过（flag=1） |
  | 7. 分类不匹配 | 7 | 跳过（fenlei不匹配） |
  | 8. 代码格式异常 | 8 | 自动修正 002594→002594.SZ |
  | 9. 限价委托 | 9 | prType=0, price=150.50 |
  | 10. 重复ID | 1 | 跳过（已处理过） |

- [ ] **步骤6**：检查回测报告
  - 确认 `print_backtest_summary()` 输出正确
  - 确认统计数字与预期一致
  - 确认模拟持仓更新正确

### F.3 参数验证

- [ ] **步骤7**：验证 prType 映射
  - 在 QMT 中手动下单，分别选择不同委托方式
  - 观察 QMT 日志中 passorder 调用的 prType 值
  - 更新 `BUY_PRICE_TYPE_MAP` 和 `SELL_PRICE_TYPE_MAP`

- [ ] **步骤8**：验证 orderType 映射
  - 在 QMT 中分别按数量、按金额下单
  - 确认 orderType 1101 和 1202 是否正确
  - 特别确认 order_target_value 的 orderType

- [ ] **步骤9**：验证 opType 映射
  - 确认买入 opType=23、卖出 opType=24

### F.4 模拟信号测试（live 模式）

- [ ] **步骤10**：在数据库中插入测试信号
  ```sql
  -- 按数量买入测试
  INSERT INTO dbo.trade (name, code, price, jiner, num, date, guoqi, flag, fenlei, type)
  VALUES ('平安银行', '000001.SZ', NULL, 0, 100, GETDATE(), 0, NULL, '未分类', 'order');
  
  -- 按金额买入测试
  INSERT INTO dbo.trade (name, code, price, jiner, num, date, guoqi, flag, fenlei, type)
  VALUES ('平安银行', '000001.SZ', NULL, 0, 5000, GETDATE(), 0, NULL, '未分类', 'order_value');
  
  -- 清仓测试
  INSERT INTO dbo.trade (name, code, price, jiner, num, date, guoqi, flag, fenlei, type)
  VALUES ('平安银行', '000001.SZ', NULL, 0, 0, GETDATE(), 0, NULL, '未分类', 'order_target_value');
  ```

- [ ] **步骤11**：启动 QMT 策略（run_mode=live），观察日志输出
  - 确认信号被正确查询
  - 确认参数映射正确
  - 确认 passorder 调用无异常
  - 确认数据库状态被正确更新

- [ ] **步骤12**：验证幂等性
  - 策略运行后，检查同一信号是否只处理一次
  - 检查数据库中 guoqi 是否已更新为 1
  - 重启策略后，已处理信号不应被重复执行

### F.5 异常场景测试

- [ ] **步骤13**：数据库断线恢复
  - 策略运行中断开数据库连接
  - 观察自动重连是否成功
  - 重连后信号处理是否恢复正常

- [ ] **步骤14**：无效信号处理
  - 插入 code 格式错误的信号
  - 插入 type 不合法的信号
  - 确认策略不会崩溃，且数据库状态被正确更新

- [ ] **步骤15**：非交易时间测试
  - 在非交易时间启动策略
  - 确认不会下单
  - 在交易时间到达后确认正常工作

### F.6 实盘小额验证

- [ ] **步骤16**：使用最小金额实盘测试
  - 设置 Buy_Amount=100（最小金额）
  - 插入一条小额买入信号
  - 确认下单成功且数量/金额正确

- [ ] **步骤17**：卖出测试
  - 对已持仓股票插入卖出信号
  - 确认卖出数量经 sell_ratio 调整后正确

- [ ] **步骤18**：清仓测试
  - 对已持仓股票插入清仓信号（num=0, type=order_target_value）
  - 确认全部持仓被卖出

### F.7 上线检查清单

- [ ] 所有 TODO 标记的参数已确认
- [ ] prType 映射表已验证
- [ ] orderType 映射已验证
- [ ] opType 映射已验证
- [ ] 数据库连接稳定
- [ ] 幂等性验证通过
- [ ] 异常场景验证通过
- [ ] 回测模式验证通过（v2.0 新增）
- [ ] 小额实盘验证通过
- [ ] 日志输出正常
- [ ] 风控参数（Buy_Amount, Buy_num, sell_ratio, bili）已按实际需求配置
- [ ] run_mode 已从 backtest 切换为 live（v2.0 新增）

---

## G. v2.0 回测/调试模式详细说明

### G.1 运行模式切换

| 模式 | run_mode 值 | 数据源 | 下单行为 | 状态更新 | 适用场景 |
|------|------------|--------|---------|---------|---------|
| 回测/调试 | `backtest` | 内存模拟信号 | `mock_passorder()`（仅打印日志） | 内存字典更新 | 开发调试、逻辑验证 |
| 实盘 | `live` | SQL Server 数据库 | `real_passorder()`（调用 QMT API） | 数据库 UPDATE | 生产环境 |

**安全设计**：默认值为 `backtest`，必须显式修改 XML 参数或 ContextInfo 属性才能切换到 `live` 模式。

### G.2 模拟信号数据结构

模拟信号与 SQL Server `dbo.trade` 表字段完全一致：

```python
{
    "id": 1,                          # 信号唯一标识
    "name": "平安银行",                # 股票名称
    "code": "000001.SZ",              # 股票代码
    "price": None,                    # 委托价格
    "jiner": 10000,                   # 成交金额
    "num": 10000,                     # 下单数量/金额
    "date": datetime.datetime(...),   # 信号时间
    "guoqi": 0,                       # 过期标志
    "flag": None,                     # 执行状态
    "fenlei": "eagles",              # 策略分类
    "zhixing_time": None,            # 执行时间
    "type": "order_value"            # 订单类型
}
```

### G.3 回测报告输出

回测报告包含两部分：

1. **逐条记录**：每个信号处理结果记录在 `backtest_report` 列表中
2. **统计摘要**：通过 `print_backtest_summary()` 输出

示例输出：
```
============================================================
[QMT跟单] [BACKTEST] 回测摘要报告
============================================================
  总信号数:       12
  已处理信号数:   8
  成功模拟下单数: 6
  跳过信号数:     4
  失败信号数:     2
  失败原因汇总:
    - 参数映射失败: 2次
  已处理信号ID:   [1, 2, 3, 4, 8, 9, 11, 12]
  模拟持仓:
    - 000001.SZ: 100股
    - 002594.SZ: 0股
============================================================
```

### G.4 bar 日期驱动过滤

在 backtest 模式下，`query_mock_signals()` 接受 `bar_date` 参数，仅返回日期匹配的信号。这模拟了实盘中"只取今日信号"的行为，使回测能按日期逐步处理信号。

### G.5 模拟持仓跟踪

`mock_passorder()` 维护 `ContextInfo.mock_positions` 字典，跟踪模拟持仓变化：
- 按数量买入：`mock_positions[code] += volume`
- 按数量卖出：`mock_positions[code] -= volume`
- 按金额操作：无法精确计算股数，跳过持仓更新

---

## 附录：快速参考

### passorder 函数签名

```python
passorder(opType, orderType, accountID, orderCode, prType, price, volume, strategyName, quickTrade, userOrderId, ContextInfo)
```

### 常用 passorder 示例

```python
# 股票按数量买入 100 股
passorder(23, 1101, account, "000001.SZ", 5, 0, 100, "策略名", 1, "备注", ContextInfo)

# 股票按金额买入 50000 元
passorder(23, 1202, account, "000001.SZ", 5, 0, 50000, "策略名", 1, "备注", ContextInfo)

# 股票按数量卖出 100 股
passorder(24, 1101, account, "000001.SZ", 5, 0, 100, "策略名", 1, "备注", ContextInfo)
```

### 数据库查询 SQL

```sql
-- 查询今日未处理信号
SELECT * FROM dbo.trade
WHERE [date] >= CONVERT(date, GETDATE())
  AND [date] < DATEADD(day, 1, CONVERT(date, GETDATE()))
  AND ISNULL(guoqi, 0) = 0
  AND flag IS NULL
  AND fenlei = '未分类'
ORDER BY date ASC, id ASC;

-- 手动重置信号（测试用）
UPDATE dbo.trade SET guoqi = 0, flag = NULL, zhixing_time = NULL WHERE id = ?;

-- 查看今日执行结果
SELECT id, code, type, num, flag, guoqi, zhixing_time
FROM dbo.trade
WHERE [date] >= CONVERT(date, GETDATE())
ORDER BY id ASC;
```

### v2.0 变更摘要

| 变更项 | v1.0 | v2.0 |
|-------|------|------|
| 运行模式 | 仅实盘 | `backtest` / `live` 双模式 |
| 信号查询 | 仅数据库 | 统一接口，分派 DB/内存 |
| 下单执行 | 仅 passorder | 统一接口，分派 mock/real |
| 状态更新 | 仅数据库 | 统一接口，分派 DB/内存 |
| 模拟数据 | 无 | 10+ 测试场景 |
| 回测报告 | 无 | 逐条记录 + 统计摘要 |
| bar 日期过滤 | 无 | backtest 模式按 bar_date 过滤 |
| 模拟持仓 | 无 | mock_positions 字典跟踪 |
| XML 参数 | 12 个 | 13 个（新增 run_mode） |
| 模块数 | 10 | 13 |
