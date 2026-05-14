# 动量趋势日内做T策略 — 深度Review报告

> 审查日期: 2026-05-13  
> 策略来源: 聚宽文章 https://www.joinquant.com/post/70680  
> 策略文件: `动量趋势日内做T策略.py`

---

## 一、未来函数（Look-Ahead Bias）排查

### 1.1 全局防护设置 ✅

```python
set_option('use_real_price', True)      # 第23行：使用真实价格成交
set_option('avoid_future_data', True)   # 第24行：避免未来数据
```

策略在 [`initialize()`](动量趋势日内做T策略.py:20) 中已开启聚宽两大防未来函数开关，这是最基础也是最重要的防护。

### 1.2 逐函数排查结果

| 函数 | 调用时间 | 数据来源 | 是否存在未来函数 | 说明 |
|------|---------|---------|:---------------:|------|
| [`select_stocks()`](动量趋势日内做T策略.py:129) | 9:25 | `context.previous_date` + `history()` + `get_fundamentals()` | ❌ 无 | 所有历史数据截止到前一日收盘，选股在盘前完成 |
| [`rebalance_buy()`](动量趋势日内做T策略.py:293) | 9:32 | `get_current_data()` | ❌ 无 | 使用实时行情买入 |
| [`t_sell_high()`](动量趋势日内做T策略.py:322) | 每分钟 | `get_current_data()` + `history(2,'1d')` | ❌ 无 | `history(2)` 取 `iloc[-2]` 为昨收，正确 |
| [`t_buy_back()`](动量趋势日内做T策略.py:374) | 每分钟 | `get_current_data()` + `history(2,'1d')` | ❌ 无 | 同上 |
| [`forward_t_buy_low()`](动量趋势日内做T策略.py:502) | 每分钟 | `get_current_data()` + `history(2,'1d')` | ❌ 无 | 同上 |
| [`forward_t_sell_high()`](动量趋势日内做T策略.py:557) | 每分钟 | `get_current_data()` | ❌ 无 | 纯实时数据 |
| [`check_base_pnl()`](动量趋势日内做T策略.py:589) | 每分钟 | `get_current_data()` + `pos.avg_cost` | ❌ 无 | 成本价为实际持仓成本 |
| [`check_trend_break()`](动量趋势日内做T策略.py:663) | 14:40 | `get_price(end_date=previous_date)` + 实时价 | ❌ 无 | MA用昨日及之前数据，现价用实时 |
| [`t_force_close()`](动量趋势日内做T策略.py:429) | 14:30 | `get_current_data()` + `history(2,'1d')` | ❌ 无 | 同上 |
| [`is_market_crash()`](动量趋势日内做T策略.py:720) | 实时 | `get_current_data()` | ❌ 无 | 使用当日开盘价和实时价 |
| [`prepare_daily()`](动量趋势日内做T策略.py:106) | 9:05 | `get_price(end_date=previous_date)` | ❌ 无 | 使用前一日数据 |

### 1.3 关键细节确认

**✅ `history(2, unit='1d')` 取昨收价的处理（第344行）**

```python
prev_data = history(2, unit='1d', field='close', security_list=[stock])
pre_close = prev_data[stock].iloc[-2]  # 倒数第二根 = 昨日收盘
```

这是**正确**的做法。在分钟级回测中，`history(1, '1d')` 会返回包含当天未完成K线的数据（等于当前价），所以必须取 `count=2` 然后用 `iloc[-2]` 获取真正的昨收价。代码注释（第342-343行）也明确说明了这一点。

**✅ `get_fundamentals()` 日期参数（第152行）**

```python
df_val = get_fundamentals(q, date=dt_last)  # dt_last = context.previous_date
```

使用前一日日期查询基本面数据，配合 `avoid_future_data=True`，不会引入未来数据。

**✅ `get_price()` 的 end_date 参数**

所有 `get_price()` 调用均使用 `end_date=context.previous_date` 或 `end_date=dt_last`，确保不包含当日未完成数据。

### 1.4 未来函数排查结论

> **🟢 未发现严重未来函数问题。** 策略在数据时间边界处理上总体规范，关键的历史数据查询均正确使用了 `context.previous_date` 作为截止日期，分钟级行情中的昨收价提取也采用了正确的 `history(2) + iloc[-2]` 模式。

### 1.5 需注意的"准未来函数"风险

虽然没有严格的未来函数，但以下场景在实盘中可能产生偏差：

1. **分钟级成交假设**：策略假设每分钟末能以当前价成交，实盘存在执行延迟和滑点。`FixedSlippage(0.02)` 可能不足以覆盖快速波动时的真实滑点。

2. **`get_fundamentals()` 的市值数据**：`valuation.market_cap` 在回测中由聚宽按日提供，但实盘中市值数据的更新可能有T+1延迟，导致回测与实盘选股不一致。

3. **9:25选股 → 9:32买入的时间差**：选股基于昨收数据，但买入在开盘2分钟后，动量股可能已大幅高开，实际买入价可能远高于选股时的参考价。

---

## 二、策略评价

### 2.1 ✅ 值得学习的亮点

#### 1. 多因子动量评分体系（第189-278行）

策略不是简单按涨幅排序，而是构建了5个维度的动量因子：

| 因子 | 权重 | 作用 |
|------|------|------|
| 风险调整动量（波动率调整） | 50% | 涨得稳比涨得猛更重要 |
| 动量加速度 | 25% | 近期涨速 > 远期涨速 = 趋势健康 |
| 动量一致性（上涨天数占比） | 25% | 涨势均匀比暴涨暴跌更持久 |

这种多因子打分方式比单一动量排序更稳健，尤其是**波动率调整**和**一致性**因子能有效过滤掉"暴涨暴跌"型伪强势股。

#### 2. 动量跳过周期（Momentum Skip）（第38行、220-221行）

```python
g.momentum_skip = 3  # 跳过最近3日
raw_momentum = closes[-g.momentum_skip] / closes[-(g.momentum_period + g.momentum_skip)] - 1
```

这是学术研究中公认的最佳实践——短期（1周内）存在反转效应，中期（1-6个月）存在动量效应。跳过最近3天计算动量，避免了短期反转对中期动量信号的污染。

#### 3. 双向做T + 互斥锁机制（第98-102行、332行、514行）

```
反T（先卖后买）：冲高卖出 → 回踩买回
正T（先买后卖）：急杀低吸 → 企稳抛出
互斥：一只股票一天只做一个方向
```

互斥锁设计（`g.t_sold_today` / `g.t_bought_first` 互斥检查）避免了同一只股票同时做反T和正T导致的仓位混乱，这是实战经验的体现。

#### 4. 多层风控体系

```
Level 1: 大盘熔断（-2%全天暂停做T）
Level 2: 个股止损（-7%清仓）
Level 3: 趋势破位（跌破MA10清仓）
Level 4: 时间止损（5天不盈利出局）
Level 5: 次日快跑（昨日建仓今日盈利5%即走）
Level 6: 防接飞刀（深跌不接回）
Level 7: 摩擦防守（差价<1%不强制接回）
```

这种分层风控在实战中非常必要，尤其是**防接飞刀**和**摩擦防守**两个细节，体现了对A股T+0交易摩擦成本的深刻理解。

#### 5. 过热防护——连板股排除（第269-272行）

```python
hl_count = sum(np.round(recent_10['close'].values, 2) >= np.round(recent_10['high_limit'].values, 2))
if hl_count > 2:
    continue  # 10天内超过2个涨停板，排除
```

连板股虽然动量极高，但回撤风险也极大。排除连板股是防止"动量陷阱"的有效手段。

#### 6. 冲高回落过滤（第160-182行）

选股时过滤掉最近两天出现大跌超6%或冲高回落超5%的股票，这能有效排除"高位放量滞涨"的危险信号。

---

### 2.2 ❌ 需要改进的问题

#### 🔴 严重问题1：止盈条件存在死代码（Dead Code）

**位置**：[`check_base_pnl()`](动量趋势日内做T策略.py:608) 第608-621行

```python
# 条件1：pnl >= 0.8 (80%止盈)  ← 永远不会被执行！
if pnl >= getattr(g, 'base_take_profit', 0.8):
    order_target_value(stock, 0)
    ...
    continue

# 条件1.05：pnl >= 0.1 (10%止盈)  ← 会先于此触发
if pnl >= 0.1:
    order_target_value(stock, 0)
    ...
    continue
```

**问题**：当 `pnl >= 0.8` 时，`pnl >= 0.1` 必然也为 `True`。由于10%止盈在80%止盈**之前**检查，80%止盈条件**永远不会被执行到**，成为死代码。

**影响**：所有盈利超过10%的持仓都会被立即止盈，完全无法持有到80%。这与"动量趋势"策略的核心理念矛盾——动量策略应该让利润奔跑，而不是在10%就止盈。

**修复建议**：

```python
# 方案A：10%止盈仅对短线持仓生效
if pnl >= 0.8:
    order_target_value(stock, 0)
    ...
    continue

# 仅在建仓5日内适用10%止盈
if pnl >= 0.1 and held_trade_days <= 5:
    order_target_value(stock, 0)
    ...
    continue

# 方案B：改用移动止盈（推荐）
if pnl >= 0.1:
    # 不立即止盈，而是启动移动止盈跟踪
    # 从最高点回撤超过5%才止盈
    ...
```

#### 🔴 严重问题2：10%固定止盈与动量策略逻辑矛盾

**位置**：[`check_base_pnl()`](动量趋势日内做T策略.py:616) 第616行

策略选股要求最低动量10%（`g.min_momentum = 0.10`），意味着入选股票已经涨了10%以上。但建仓后又在盈利10%时强制止盈，这导致：

- 选出来的动量股，刚买入不久就可能因继续上涨而触发10%止盈
- 动量策略的核心利润来源是**趋势延续**，10%止盈直接截断了趋势利润
- 做T的差价收益通常只有2-4%，而底仓10%止盈会放弃远大于此的趋势利润

**修复建议**：改用移动止盈（Trailing Stop），从最高点回撤一定比例才止盈，让利润充分奔跑。

#### 🔴 严重问题3：参数注释与实际值严重不一致

**位置**：多处参数定义（第39-57行）

| 参数 | 注释描述 | 实际值 | 差异 |
|------|---------|--------|------|
| `g.t_sell_gain = 0.03` | "涨幅超过2.5%卖出" | 3% | 注释写2.5% |
| `g.t_buy_back_drop = -0.04` | "回落2.5%精准买回" | -4% | 注释写2.5% |
| `g.base_take_profit = 0.8` | "盈利5%强制止盈" | 80% | 注释写5% |
| `g.base_stop_loss = -0.07` | "回撤5%清仓" | -7% | 注释写5% |
| `g.ma_break_period = 10` | "跌破MA5清仓" | MA10 | 注释写MA5 |

**影响**：参数注释与实际值不一致，在调参时极易造成误解。例如以为止盈是5%实际是80%（虽然80%是死代码），以为止损是5%实际是7%。

**修复建议**：更新所有注释使其与实际参数值一致，或调整参数值使其符合注释意图。

#### 🟡 中等问题4：缺少移动止盈机制

**位置**：[`check_base_pnl()`](动量趋势日内做T策略.py:589)

当前止盈方式只有固定阈值（10%），没有移动止盈。对于动量策略，移动止盈是标配：

```python
# 建议增加移动止盈
g.trailing_stop = -0.05  # 从持仓最高点回撤5%止盈
g.trailing_high = {}     # 记录每只股票的持仓期间最高价

# 在 check_base_pnl() 中：
if stock not in g.trailing_high:
    g.trailing_high[stock] = current_price
else:
    g.trailing_high[stock] = max(g.trailing_high[stock], current_price)

if pnl > 0.1:  # 盈利超过10%后启动移动止盈
    drop_from_high = (current_price - g.trailing_high[stock]) / g.trailing_high[stock]
    if drop_from_high <= g.trailing_stop:
        order_target_value(stock, 0)
```

#### 🟡 中等问题5：正T做T盈亏未在盘后统计

**位置**：[`after_market()`](动量趋势日内做T策略.py:695) 第695-716行

盘后统计只计算了反T（先卖后买）的盈亏，完全忽略了正T（先买后卖）的盈亏统计：

```python
# 当前代码只统计了 g.t_sold_today → g.t_bought_back 的反T盈亏
# 缺少 g.t_bought_first → g.t_sold_back_first 的正T盈亏统计
```

**修复建议**：

```python
# 在 after_market() 中增加正T统计
if g.t_bought_first:
    for stock, buy_info in g.t_bought_first.items():
        buy_price = buy_info['price']
        if stock in g.t_sold_back_first:
            sell_price = g.t_sold_back_first[stock]
            pnl = (sell_price - buy_price) / buy_price * 100
            log.info("[正T日报] %s | 买: %.2f → 卖: %.2f | 盈亏: %.2f%%" %
                     (stock, buy_price, sell_price, pnl))
        else:
            log.info("[正T日报] %s | 买: %.2f | 未对冲（尾盘强制平仓）" %
                     (stock, buy_price))
```

#### 🟡 中等问题6：`t_force_close()` 与 `check_trend_break()` 时序冲突

**位置**：第81行 `run_daily(t_force_close, '14:30')` 和 第80行 `run_daily(check_trend_break, '14:40')`

时序问题：
1. **14:30** — 强制买回未接回的反T仓位
2. **14:40** — 检查趋势破位，可能清仓

如果某股票14:30被强制买回（花费现金），14:40又因趋势破位被清仓（卖出），则：
- 做T买回是**亏损的**（因为没跌到阈值就强制买回了）
- 紧接着又被清仓，底仓也丢了
- 两次交易产生双倍手续费

**修复建议**：在 `t_force_close()` 中先检查趋势破位条件，如果即将清仓则不强制买回：

```python
# 在 t_force_close() 中，强制买回前先检查趋势是否已破
df = get_price(stock, end_date=context.previous_date, frequency='daily',
               fields=['close'], count=g.ma_break_period, panel=False, skip_paused=True)
if len(df) >= g.ma_break_period:
    ma = df['close'].mean()
    if current_price < ma:
        # 趋势已破，不买回，让14:40的check_trend_break去清仓
        g.t_bought_back[stock] = current_price  # 标记为已处理
        continue
```

#### 🟡 中等问题7：`check_base_pnl()` 清仓后未清理正T状态

**位置**：[`check_base_pnl()`](动量趋势日内做T策略.py:609) 第609-610行

```python
order_target_value(stock, 0)
g.t_sold_today.pop(stock, None)  # 只清理了反T状态
# 缺少：g.t_bought_first.pop(stock, None)  # 正T状态未清理
```

如果某股票做了正T（已低吸买入），然后被 `check_base_pnl()` 清仓，`g.t_bought_first` 中仍保留该股票的记录。虽然 `t_force_close()` 中有 `pos.closeable_amount` 检查不会重复卖出，但状态不干净可能导致日志混乱和逻辑隐患。

#### 🟡 中等问题8：创业板全部过滤可能错失机会

**位置**：[`filter_kcbj_stock()`](动量趋势日内做T策略.py:735) 第737-738行

```python
return [s for s in stock_list
        if not (s.startswith('68') or s.startswith('4') or s.startswith('8') or s.startswith('3'))]
```

过滤了所有 `3` 开头的股票，即**整个创业板**（300xxx）。创业板有20%涨跌停限制，波动率更大，对于动量策略来说可能是更好的猎场。完全排除创业板会：
- 大幅缩小选股池
- 错失创业板强势股的机会
- 做T空间更大（20%振幅 vs 10%振幅）

**修复建议**：可以考虑保留创业板，但对其使用更严格的动量门槛或更小的做T仓位。

#### 🟢 轻微问题9：`rebalance_buy()` 未重新检查价格范围

**位置**：[`rebalance_buy()`](动量趋势日内做T策略.py:293) 第304-308行

选股时用昨收价过滤了价格范围（13-60元），但9:32买入时只检查了涨跌停和停牌，未重新检查实时价格是否仍在范围内。动量股可能高开超过60元上限。

#### 🟢 轻微问题10：做T卖出量计算公式可读性差

**位置**：[`t_sell_high()`](动量趋势日内做T策略.py:363) 第363行

```python
t_amount = int(pos.closeable_amount * g.t_ratio / 100) * 100
```

这个公式功能正确（计算40%仓位并取整百），但 `/ 100 * 100` 的写法容易让人误解为百分比转换，实际目的是取整到100股的整数倍。建议改为更清晰的写法：

```python
t_amount = int(pos.closeable_amount * g.t_ratio // 100) * 100
# 或者更明确：
raw_amount = int(pos.closeable_amount * g.t_ratio)
t_amount = raw_amount // 100 * 100  # 向下取整到整百
```

#### 🟢 轻微问题11：`after_market()` 中做T盈亏计算方式有误

**位置**：[`after_market()`](动量趋势日内做T策略.py:703) 第703行

```python
pnl = (sell_price - buy_back_price) / buy_back_price * 100
```

这里用买回价作为分母计算盈亏率，但做T的盈亏应该以卖出价为基准（卖出时锁定的差价）：

```python
# 更准确的计算：做T收益率 = (卖出价 - 买回价) / 卖出价 * 100
pnl = (sell_price - buy_back_price) / sell_price * 100
```

虽然差异很小，但逻辑上卖出价才是做T的"投入"。

---

## 三、改进优先级总结

| 优先级 | 问题 | 影响 | 修复难度 |
|:------:|------|------|:--------:|
| 🔴 P0 | 止盈条件死代码（80%永远不触发） | 严重：10%强制止盈截断趋势利润 | 低 |
| 🔴 P0 | 10%固定止盈与动量策略矛盾 | 严重：策略核心逻辑自相矛盾 | 中 |
| 🔴 P1 | 参数注释与实际值不一致 | 高：调参时极易出错 | 低 |
| 🟡 P2 | 缺少移动止盈机制 | 中：利润无法最大化 | 中 |
| 🟡 P2 | 正T盈亏未统计 | 中：无法评估正T效果 | 低 |
| 🟡 P2 | t_force_close与check_trend_break时序冲突 | 中：可能产生不必要交易 | 低 |
| 🟡 P3 | 清仓后正T状态未清理 | 中：状态不干净 | 低 |
| 🟡 P3 | 创业板全部过滤 | 中：选股池缩小 | 低 |
| 🟢 P4 | 买入时未重新检查价格范围 | 低：可能以不合理价格买入 | 低 |
| 🟢 P4 | 做T卖出量公式可读性 | 低：功能正确但可读性差 | 低 |
| 🟢 P4 | 做T盈亏计算分母选择 | 低：差异极小 | 低 |

---

## 四、总体评价

### 评分：7.0 / 10

**优点总结**：
- 未来函数防护到位，数据时间边界处理规范
- 多因子动量评分体系设计精巧，跳过周期、波动率调整、一致性检查都是学术最佳实践
- 双向做T + 互斥锁机制体现了丰富的实战经验
- 多层风控体系全面，防接飞刀和摩擦防守等细节处理到位
- 过热防护和冲高回落过滤有效规避了动量陷阱

**核心缺陷**：
- **止盈逻辑自相矛盾**是最严重的问题——选股追求强动量（涨10%以上），持仓却在盈利10%时强制止盈，直接否定了动量策略"让利润奔跑"的核心理念。这导致策略本质上是一个"短线波段+做T"策略，而非真正的"动量趋势"策略
- 参数注释与实际值大面积不一致，反映出策略在迭代过程中注释维护不足
- 缺少移动止盈机制，无法在保护利润的同时让趋势充分发展

**一句话总结**：这是一个框架扎实、风控细腻的做T策略，但止盈逻辑的矛盾使其无法发挥动量选股的真正威力——**选股是趋势派，止盈是短线派，两者需要统一**。
