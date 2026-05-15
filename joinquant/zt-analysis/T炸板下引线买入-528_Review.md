# T炸板下引线买入-528 策略深度评审报告

> **策略来源**: [聚宽文章](https://www.joinquant.com/post/70871)  
> **策略标题**: 一字板下引线买入法  
> **策略作者**: 天香膏  
> **评审日期**: 2026-05-14  
> **综合评分**: ⭐ 5.0 / 10

---

## 一、策略概览

本策略围绕 **涨停炸板（炸板=涨停价被打开）** 这一核心事件，设计了三种互补的买入方式：

| 买入策略 | 触发时间 | 核心逻辑 |
|---------|---------|---------|
| 竞价一字板炸板 | 09:26识别 → 09:35~11:00扫描 | 集合竞价一字板，盘中炸板时买入 |
| 9:34涨停炸板 | 09:34识别 → 09:35~11:00扫描 | 开盘4分钟内涨停，随后炸板时买入 |
| 昨日双涨停低开低吸 | 06:00识别 → 09:35~11:00扫描 | 昨日开盘+收盘均涨停，今日低开3%+时买入 |

**卖出逻辑**:
- 止盈: 盈利20%卖出 (`PROFIT_TAKE_RATIO = 1.2`)
- 止损: 价格 < MA5 × 0.986 时卖出
- 涨停不卖: 当前价 ≥ 涨停价时锁定
- 最大持仓: 2只 (`BUY_MAX_COUNT = 2`)

---

## 二、未来函数逐函数审查

### 全局保护设置 ✅

```python
set_option('avoid_future_data', True)   # 第243行 ✅
set_option('use_real_price', True)       # 第244行 ✅
set_slippage(PriceRelatedSlippage(0.001)) # 第245行 ✅ 0.1%比例滑点
```

### 逐函数审查表

| 函数 | 行号 | 数据调用 | end_dt / end_date | 调用时间 | 未来函数? |
|------|------|---------|-------------------|---------|----------|
| [`calc_custom_winner()`](T炸板下引线买入-528.py:20) | 22-28 | `get_bars(code, 120, '1d', ..., end_dt=end_date)` | `yesterday` (由调用方传入) | 09:26 | ✅ 安全 |
| [`calc_custom_winner()`](T炸板下引线买入-528.py:42) | 42 | `get_valuation(code, end_date=end_date, ...)` | `yesterday` | 09:26 | ✅ 安全 |
| [`get_yizi_auction_stocks()`](T炸板下引线买入-528.py:51) | 61-66 | `get_call_auction(..., start_date=09:15, end_date=09:25)` | 09:15~09:25 | 09:26 | ✅ 安全 |
| [`get_yizi_auction_stocks()`](T炸板下引线买入-528.py:76) | 76-82 | `get_price(stock_list, end_date=yesterday, ...)` | `yesterday` | 09:26 | ✅ 安全 |
| [`get_934_limit_up_stocks()`](T炸板下引线买入-528.py:98) | 111 | `get_bars(code, 1, '1m', end_dt=check_time)` | 09:34 | 09:34 | ✅ 安全 |
| [`get_934_limit_up_stocks()`](T炸板下引线买入-528.py:115) | 115 | `get_bars(code, 1, '1d', end_dt=today-1day)` | 昨日 | 09:34 | ✅ 安全 |
| [`check_yesterday_open_close_limit()`](T炸板下引线买入-528.py:131) | 133 | `get_bars(code, 2, '1d', end_dt=today)` | `today` (06:00) | 06:00 | ✅ 安全* |
| [`check_yesterday_open_close_limit()`](T炸板下引线买入-528.py:145) | 145 | `get_bars(code, 3, '1d', end_dt=today)` | `today` (06:00) | 06:00 | ✅ 安全* |
| [`get_safe_data()`](T炸板下引线买入-528.py:171) | 173 | `get_bars(code, 1, '1m', end_dt=now-1day)` | 昨日同时刻 | 盘中 | ⚠️ 逻辑错误 |
| [`get_safe_data()`](T炸板下引线买入-528.py:175) | 175 | `get_bars(code, 1, '1m', end_dt=now)` | 当前时刻 | 盘中 | ✅ 安全 |
| [`scan_buy()`](T炸板下引线买入-528.py:229) 策略3 | 229 | `get_bars(code, 1, '1d', end_dt=now, fields=['open'])` | 当前时刻(仅open) | 盘中 | ✅ 安全 |
| [`yesterday_profit_stat()`](T炸板下引线买入-528.py:293) | 293 | `get_bars(code, 1, '1d', end_dt=previous_date)` | 昨日 | 06:05 | ✅ 安全 |
| [`sell_out()`](T炸板下引线买入-528.py:338) | 338 | `get_bars(code, 5, '1m', fields=['close'])` | 默认当前 | 盘中 | ✅ 安全 |

> **\*注**: [`check_yesterday_open_close_limit()`](T炸板下引线买入-528.py:131) 在06:00调用，`end_dt=today`，但由于 `avoid_future_data=True`，JQ会自动排除当日未完成的日线数据，返回的是最近2个已完成交易日（前日+昨日）的数据，逻辑正确。

### 未来函数审查结论

**无严重未来函数问题** ✅。所有数据调用均使用了正确的时间边界，`avoid_future_data=True` 为关键调用提供了额外保护。

但 [`get_safe_data()`](T炸板下引线买入-528.py:171) 存在 **逻辑错误**（非未来函数，但影响信号准确性），详见问题 #2。

---

## 三、策略优点

### 1. 三策略互补设计 🌟
三种买入策略覆盖了不同的涨停炸板场景：
- **竞价一字板炸板**: 捕捉最强势股的首次开板机会
- **9:34涨停炸板**: 捕捉开盘快速涨停后回撤的机会
- **昨日双涨停低开低吸**: 利用强势股的隔日低开反弹

这种多策略组合增加了交易机会的多样性，避免单一场景的局限性。

### 2. 获利筹码过滤 🌟
[`calc_custom_winner()`](T炸板下引线买入-528.py:20) 计算获利筹码比例，在 [`get_yizi_auction_stocks()`](T炸板下引线买入-528.py:92) 中过滤 `30% ≤ winner ≤ 1000%` 的股票。这是一个有价值的筹码面过滤，避免买入获利盘过少（抛压大）或过多（获利了结压力大）的股票。

### 3. 合理的滑点模型 ✅
```python
set_slippage(PriceRelatedSlippage(0.001))  # 0.1%比例滑点
```
使用比例滑点而非固定滑点，更符合炸板股的成交特征（价格波动大，固定滑点会失真）。

### 4. 涨停不卖逻辑 ✅
[`sell_out()`](T炸板下引线买入-528.py:329) 中 `if px >= high_limit - 0.001: continue`，涨停时不卖出，让利润继续奔跑。对于炸板回封的股票，这一逻辑可以避免过早止盈。

### 5. 多时间点扫描 ✅
```python
run_daily(scan_buy, '09:35')
run_daily(scan_buy, '09:50')
run_daily(scan_buy, '10:05')
run_daily(scan_buy, '10:20')
run_daily(scan_buy, '10:35')
run_daily(scan_buy, '11:00')
```
6个时间点扫描，覆盖了早盘主要交易时段，不会错过盘中炸板机会。

### 6. 3日涨幅过滤 ✅
[`check_yesterday_open_close_limit()`](T炸板下引线买入-528.py:148) 中 `df3['close'].iloc[-1] / df3['close'].iloc[0] > MAX_3D_INCREASE`（25%），过滤掉短期涨幅过大的股票，避免追高。

---

## 四、问题清单（按严重程度排序）

### 🔴 CRITICAL #1: 涨停价硬编码 9.8%，不适用于创业板/科创板

**位置**: [第89行](T炸板下引线买入-528.py:89)、[第119行](T炸板下引线买入-528.py:119)、[第137行](T炸板下引线买入-528.py:137)、[第186行](T炸板下引线买入-528.py:186)

**问题**: 全局使用 `high_limit = pre_close * 1.098` 计算涨停价，但A股不同板块涨跌幅限制不同：

| 板块 | 代码特征 | 涨跌幅限制 | 应使用系数 |
|------|---------|-----------|-----------|
| 主板 | 60xxxx / 00xxxx | ±10% | 1.098 (近似) |
| 创业板 | 300xxx / 301xxx | ±20% | 1.198 |
| 科创板 | 688xxx | ±20% | 1.198 |
| 北交所 | 8xxxxx / 4xxxxx | ±30% | 1.298 |
| ST股 | — | ±5% | 1.048 (已过滤ST) |

**影响**:
- 创业板/科创板股票的涨停价被严重低估，`1.098` 远低于实际涨停价 `1.198`
- 导致这些股票永远不会被识别为"涨停"，策略完全错失创业板/科创板的炸板机会
- 创业板300xxx是炸板策略的重要标的来源，此bug导致策略覆盖面大幅缩窄

**修复建议**:
```python
def get_high_limit_ratio(code):
    """根据股票代码返回涨停比例"""
    if code.startswith(('300', '301')):  # 创业板
        return 1.198
    elif code.startswith('688'):         # 科创板
        return 1.198
    elif code.startswith(('8', '4')):    # 北交所
        return 1.298
    else:                                # 主板
        return 1.098
```
或更简洁地使用 JQ 内置的 `current_data[code].high_limit` 直接获取涨停价（[`sell_out()`](T炸板下引线买入-528.py:327) 中已正确使用）。

---

### 🔴 CRITICAL #2: `get_safe_data()` 使用分钟线收盘价作为 pre_close，逻辑错误

**位置**: [第171-180行](T炸板下引线买入-528.py:171)

**问题**:
```python
def get_safe_data(code, now):
    pre_bar = get_bars(code, count=1, unit='1m', end_dt=now - timedelta(days=1), fields=['close'])
    pre_close = pre_bar['close'][0]  # ← 这是昨日同时刻的分钟收盘价，不是昨日收盘价！
```

`now - timedelta(days=1)` 获取的是 **昨日同一时钟时间的1分钟K线收盘价**，而非 **昨日日线收盘价**。

**举例**: 若 `now = 2026-05-14 09:35:00`，则 `now - timedelta(days=1) = 2026-05-13 09:35:00`，取到的是昨日09:35分钟的收盘价，这是昨日开盘附近的价格，**不是昨日收盘价**。

**影响**:
- [`check_zhaban()`](T炸板下引线买入-528.py:182) 中的 `high_limit = pre_close * 1.098` 计算的涨停价错误
- 涨幅 `zhang = (price / pre_close - 1) * 100` 计算错误
- 炸板判断 `is_zhaban = (high >= high_limit - 0.01) and (price < high_limit - 0.005)` 可能误判
- 对于 `g.yizi_list`（昨日一字板），由于一字板全天价格不变，昨日09:35分钟收盘价 = 昨日收盘价，**恰好正确**
- 对于 `g.stock_at_934`（今日9:34涨停），昨日09:35分钟收盘价 ≠ 昨日收盘价，**判断错误**

**修复建议**:
```python
def get_safe_data(code, now):
    try:
        # 使用日线获取昨日收盘价
        pre_bar = get_bars(code, count=1, unit='1d', end_dt=now - timedelta(days=1), fields=['close'])
        pre_close = pre_bar['close'].iloc[0]
        # 当前分钟数据
        bar = get_bars(code, count=1, unit='1m', end_dt=now, fields=['close', 'high'])
        price = bar['close'][0]
        high  = bar['high'][0]
        return pre_close, price, high
    except:
        return None, None, None
```

---

### 🟠 HIGH #3: 全市场逐股遍历，性能极差

**位置**: [`get_934_limit_up_stocks()`](T炸板下引线买入-528.py:98) 第106-124行、[`morning_6am_scan()`](T炸板下引线买入-528.py:157) 第158-165行

**问题**:
- [`get_934_limit_up_stocks()`](T炸板下引线买入-528.py:106): 对全市场 ~5000 只股票逐一调用 `get_bars()`，每只2次API调用，共 ~10000 次
- [`morning_6am_scan()`](T炸板下引线买入-528.py:162): 对全市场股票逐一调用 `check_yesterday_open_close_limit()`，每只2次API调用
- [`calc_custom_winner()`](T炸板下引线买入-528.py:32): 在 [`get_yizi_auction_stocks()`](T炸板下引线买入-528.py:92) 中对每个一字板候选股调用，内部120次Python循环

**影响**:
- 回测时极易触发JQ的API调用频率限制或超时
- 实盘中06:00和09:34的扫描可能无法在时间窗口内完成

**修复建议**:
- 使用 `get_price()` 批量获取数据，避免逐股遍历
- `calc_custom_winner()` 使用 numpy 向量化替代 Python for 循环
- 预先过滤股票池（如仅关注沪深300+中证500成分股）

---

### 🟠 HIGH #4: 7处裸 `except:` 吞没所有异常

**位置**: [第45行](T炸板下引线买入-528.py:45)、[第123行](T炸板下引线买入-528.py:123)、[第151行](T炸板下引线买入-528.py:151)、[第179行](T炸板下引线买入-528.py:179)、[第236行](T炸板下引线买入-528.py:236)、[第298行](T炸板下引线买入-528.py:298)、[第344行](T炸板下引线买入-528.py:344)

**问题**: 裸 `except:` 会捕获所有异常，包括 `KeyboardInterrupt`、`SystemExit`、`TypeError` 等不应被静默处理的异常。在回测中，关键逻辑错误（如数据格式变化、API返回值变更）会被完全掩盖，导致策略静默产生错误交易。

**修复建议**:
```python
except Exception as e:
    log.warning(f"calc_custom_winner({code}) 异常: {e}")
    return -1
```

---

### 🟠 HIGH #5: 未设置基准

**位置**: [`initialize()`](T炸板下引线买入-528.py:242)

**问题**: 缺少 `set_benchmark()` 调用，JQ默认使用沪深300作为基准。对于炸板策略，基准的选择影响超额收益的计算和评估。建议显式设置。

**修复建议**:
```python
set_benchmark('000300.XSHG')  # 或 '000905.XSHG' 中证500
```

---

### 🟡 MEDIUM #6: 买入成交可行性 — 炸板瞬间市价单可能严重滑点

**位置**: [`scan_buy()`](T炸板下引线买入-528.py:210) 第210行、[第220行](T炸板下引线买入-528.py:220)

**问题**: 炸板发生时，股票从涨停价快速下跌，使用 `order_value(code, per_cash)` 市价单买入：
- 炸板瞬间卖单涌出，买一价可能远低于预期
- 实际成交价可能比检测到炸板时的价格低很多
- 0.1%的比例滑点可能不足以覆盖炸板股的实际滑点

**影响**: 回测中假设的成交价与实盘严重不符，回测收益可能被高估。

**修复建议**:
- 增加限价单逻辑，设定可接受的最高买入价
- 提高滑点比例至 0.2%~0.5%
- 在买入前检查买卖盘口（实盘）

---

### 🟡 MEDIUM #7: 卖出逻辑不完善

**位置**: [`sell_out()`](T炸板下引线买入-528.py:315)

**问题**:
1. **无绝对止损**: 只有MA5动态止损，没有从成本价计算的绝对止损（如 -8%）。若股票持续阴跌但始终在MA5附近，可能长期持有亏损股
2. **无最大持仓天数**: 缺少持仓时间限制，可能长期套牢
3. **MA5止损过紧**: `STOP_LOSS_MA5_RATIO = 0.986`，仅低于MA5 1.4%，对于波动较大的炸板股极易触发
4. **止盈止损互斥**: 代码中止盈和止损是 `if` 而非 `elif`，若止盈卖出后仍会检查止损（但 `closeable_amount` 已为0所以不会重复卖出，逻辑上无bug但不够清晰）

**修复建议**:
```python
# 增加绝对止损
if px < cost * 0.92:  # -8%绝对止损
    order_target(code, 0)
    log.info(f"📉 {code} 绝对止损卖出")

# 增加最大持仓天数
if holding_days > 5:
    order_target(code, 0)
    log.info(f"⏰ {code} 超时卖出")
```

---

### 🟡 MEDIUM #8: 变量命名误导 — `yd_open` 实为今日开盘价

**位置**: [第230行](T炸板下引线买入-528.py:230)

**问题**:
```python
df = get_bars(code, count=1, unit='1d', end_dt=now, fields=['open'])
yd_open = df['open'].iloc[0]  # ← 变量名暗示"昨日开盘"，实际是"今日开盘"
```

`end_dt=now` 配合 `avoid_future_data=True`，返回的是今日日线（仅open字段可用），所以这是今日开盘价。变量名 `yd_open` 严重误导。

**修复建议**: 重命名为 `today_open`。

---

### 🟡 MEDIUM #9: `calc_custom_winner()` 性能和逻辑问题

**位置**: [第20-46行](T炸板下引线买入-528.py:20)

**问题**:
1. **Python for循环**: 120次循环逐行计算，应使用numpy向量化
2. **`A1 = close_arr[-2]`**: 使用倒数第二根K线收盘价作为盈亏平衡点，语义不够清晰。如果 `end_date=yesterday`，则 `close_arr[-2]` 是前日收盘，`close_arr[-1]` 是昨日收盘。用前日收盘作为参考价来计算"获利筹码"的合理性需要验证
3. **`H - L < 0.01` 的处理**: 当日振幅小于1分钱时，将全部成交额计入获利筹码，这个假设可能不合理（一字涨停/跌停的情况应特殊处理）

**修复建议**:
```python
def calc_custom_winner(code, end_date):
    try:
        df = get_bars(security=code, count=120, unit='1d',
                      fields=['close','high','low','money'], end_dt=end_date)
        A1 = df['close'].iloc[-2]
        H, L, M = df['high'].values, df['low'].values, df['money'].values
        
        # 向量化计算
        mask = (H - L) >= 0.01
        part = np.where(mask & (H > A1), M * (H - A1) / (H - L), M)
        part = np.where(mask & (H <= A1), 0, part)  # H <= A1 时无获利筹码
        sum_An = part.sum()
        
        val_df = get_valuation(code, end_date=end_date, count=1, fields=['circulating_market_cap'])
        circ_value = val_df['circulating_market_cap'].iloc[0] * 1e8
        return round(100 * sum_An / circ_value, 2)
    except Exception as e:
        log.warning(f"calc_custom_winner({code}) 异常: {e}")
        return -1
```

---

### 🟡 MEDIUM #10: 持仓极度集中，风险敞口大

**位置**: [第13行](T炸板下引线买入-528.py:13)

**问题**: `BUY_MAX_COUNT = 2`，最多仅持有2只股票，单只股票占比可达50%。炸板股本身波动极大，2只持仓无法有效分散个股风险。

**影响**:
- 单只股票跌停可造成组合5%以上损失
- 回测结果受个别股票影响极大，策略稳定性差

**修复建议**: 增加到 3~5 只，并加入行业分散约束（同行业最多1只）。

---

### 🟢 LOW #11: 缺少市场环境过滤

**问题**: 策略在任何市场环境下都执行相同的买入逻辑，没有判断大盘趋势。炸板策略在弱势市场中炸板后继续下跌的概率远大于回封概率。

**修复建议**:
```python
def market_filter(context):
    """大盘过滤：沪深300在MA20以下时不买入"""
    df = get_bars('000300.XSHG', count=20, unit='1d', fields=['close'])
    ma20 = df['close'].mean()
    current = get_current_data()['000300.XSHG'].last_price
    return current > ma20
```

---

### 🟢 LOW #12: `before_trading_start()` 空实现

**位置**: [第347-348行](T炸板下引线买入-528.py:347)

**问题**: `before_trading_start()` 仅 `pass`，未利用此钩子做任何预处理。可以将06:00的扫描移至此处，利用JQ的框架保证执行时机。

---

### 🟢 LOW #13: 卖出时间点设计可优化

**位置**: [第306-307行](T炸板下引线买入-528.py:306)

**问题**:
```python
SELL_1ST_TIMES = ['14:30:00']   # 止盈检查
SELL_2ND_TIMES = ['11:00:00', '14:50:00']  # 止盈+止损检查
```
- 14:30 才第一次检查止盈，但炸板股可能在早盘就已冲高回落
- 11:00 才第一次检查止损，但买入从09:35开始，09:35~11:00之间无止损保护
- 建议增加盘中止损检查频率

---

### 🟢 LOW #14: `order_value()` 未检查返回值

**位置**: [第210行](T炸板下引线买入-528.py:210)、[第220行](T炸板下引线买入-528.py:220)、[第233行](T炸板下引线买入-528.py:233)

**问题**: `order_value(code, per_cash)` 的返回值未检查。若下单失败（停牌、涨跌停无法成交等），策略不会感知，且 `holds.append(code)` 仍会执行，导致后续扫描误认为已持有该股票。

**修复建议**:
```python
result = order_value(code, per_cash)
if result is not None:
    log.info(f"✅ [一字炸板] {code}")
    holds.append(code)
```

---

### 🟢 LOW #15: 科创板股票未过滤

**位置**: [`get_934_limit_up_stocks()`](T炸板下引线买入-528.py:107) 第107行

**问题**: 仅过滤了停牌和ST股，未过滤科创板（688xxx）。科创板20%涨跌幅、最低买入200股，与主板交易规则不同，需特殊处理。

---

## 五、问题汇总表

| # | 严重程度 | 问题 | 位置 | 类别 |
|---|---------|------|------|------|
| 1 | 🔴 CRITICAL | 涨停价硬编码9.8%，不适用创业板/科创板 | L89, L119, L137, L186 | 信号逻辑 |
| 2 | 🔴 CRITICAL | `get_safe_data()` 用分钟线收盘价当pre_close | L171-180 | 信号逻辑 |
| 3 | 🟠 HIGH | 全市场逐股遍历，性能极差 | L98-126, L157-166 | 性能 |
| 4 | 🟠 HIGH | 7处裸`except:`吞没异常 | L45,123,151,179,236,298,344 | 代码质量 |
| 5 | 🟠 HIGH | 未设置基准 | L242 | 规范 |
| 6 | 🟡 MEDIUM | 炸板瞬间市价单滑点不足 | L210, L220 | 交易执行 |
| 7 | 🟡 MEDIUM | 卖出逻辑不完善（无绝对止损/持仓天数限制） | L315-345 | 风控 |
| 8 | 🟡 MEDIUM | `yd_open`变量名误导（实为今日开盘） | L230 | 代码质量 |
| 9 | 🟡 MEDIUM | `calc_custom_winner()`性能和逻辑问题 | L20-46 | 性能/逻辑 |
| 10 | 🟡 MEDIUM | 持仓极度集中（仅2只） | L13 | 风控 |
| 11 | 🟢 LOW | 缺少市场环境过滤 | — | 策略逻辑 |
| 12 | 🟢 LOW | `before_trading_start()`空实现 | L347-348 | 规范 |
| 13 | 🟢 LOW | 卖出时间点设计可优化 | L306-307 | 策略逻辑 |
| 14 | 🟢 LOW | `order_value()`未检查返回值 | L210,220,233 | 代码质量 |
| 15 | 🟢 LOW | 科创板股票未过滤 | L107 | 规范 |

---

## 六、评分细项

| 评估维度 | 得分 | 满分 | 说明 |
|---------|------|------|------|
| 未来函数防护 | 9 | 10 | `avoid_future_data` + `use_real_price` 均已设置，数据调用时间边界正确 |
| 信号逻辑正确性 | 4 | 10 | 涨停价硬编码9.8%导致创业板/科创板完全失效；pre_close取值错误影响炸板判断 |
| 风控体系 | 4 | 10 | 有止盈止损但缺少绝对止损和持仓天数限制；持仓极度集中 |
| 代码质量 | 4 | 10 | 7处裸except；变量命名误导；性能极差；未检查下单返回值 |
| 策略设计 | 6 | 10 | 三策略互补设计好；获利筹码过滤有价值；但缺少市场环境过滤 |
| 交易执行真实性 | 4 | 10 | 炸板市价单滑点不足；0.1%比例滑点对炸板股偏低 |
| **综合** | **5.0** | **10** | |

---

## 七、总结

本策略的 **未来函数防护做得较好**，`avoid_future_data=True` 和 `use_real_price=True` 均已正确设置，数据调用的时间边界基本正确。策略设计上，三种炸板买入策略互补、获利筹码过滤有独到之处。

但存在 **两个CRITICAL级逻辑错误**：
1. **涨停价硬编码9.8%** — 创业板（300xxx）和科创板（688xxx）的涨停价为20%，此bug导致策略完全错失这两个重要板块的炸板机会
2. **`get_safe_data()` 用分钟线收盘价代替日线收盘价** — 导致9:34涨停炸板策略的pre_close计算错误，直接影响炸板判断的准确性

这两个问题修复后，策略评分可提升至 **6.0~6.5** 分。进一步优化性能（批量获取数据）、完善风控（绝对止损、持仓天数限制、市场环境过滤）、提高交易执行真实性（提高滑点、限价单），可提升至 **7.0+** 分。
