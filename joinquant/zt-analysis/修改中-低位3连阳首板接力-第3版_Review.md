# 低位3连阳首板接力-第3版.py 策略深度评审报告

> **代码修改记录**:
> - CRITICAL #1: 修复 `get_buy()` 缓存回退逻辑，不再依赖 `before_open()` 返回值 ✅
> - CRITICAL #2: `MarketOrderStyle(op)` → `MarketOrderStyle()`，避免限价无法成交 ✅
> - HIGH #3: 裸`except:` → `except Exception as e:` + 竞价过滤异常日志 ✅
> - HIGH #4: `g.sold_stocks` 从死代码改为"当日已卖出防回买"机制 ✅
> - MEDIUM #5: 30+硬编码参数提取到 `g.*` 配置变量 ✅
> - MEDIUM #6: 新增 `filter_market_environment()` 大盘环境过滤 ✅
> - MEDIUM #7: `filter_auction()` 优化：dict O(1)查找 + 最大股票数限制 ✅
> - MEDIUM #8: `g.max_position_pct` → `g.per_stock_cap_pct` 语义清晰化 ✅
> - 附加：全局footprint日志（盘前/竞价/大盘/买入/卖出/日终总结） ✅

> **策略来源**: [聚宽文章](https://www.joinquant.com/post/72300) | **作者**: 顶级理解  
> **回测业绩**: 2025-01-01→2026-04-27 收益1856.78%，年化864.73%，最大回撤13.27%，胜率58.2%  
> **评审日期**: 2026-05-18

---

## 一、未来函数审查

### 1.1 全局防护设置

| 设置项 | 代码 | 状态 |
|--------|------|------|
| 真实价格交易 | `set_option('use_real_price', True)` (L16) | ✅ 正确 |
| 防止未来数据 | `set_option("avoid_future_data", True)` (L17) | ✅ 正确 |
| 滑点设置 | `set_slippage(FixedSlippage(0.005))` (L18) | ✅ 0.5%合理 |
| 交易成本 | `set_order_cost(...)` (L20) | ✅ 合理 |
| 基准指数 | `set_benchmark('399303.XSHE')` (L21) | ✅ 国证2000 |

### 1.2 逐函数未来函数审查

| 函数 | 调用时间 | 数据源 | end_date | 结论 |
|------|----------|--------|----------|------|
| `prepare_base_stocks()` | 周一09:10 | `get_all_securities(date=60天前)` | ✅ 历史日期 | ✅ 安全 |
| `before_open()` | 每日09:11 | `filter_first_board(y_day)` / `filter_volume_price(y_day)` / `filter_high_volatility(y_day)` | ✅ 前一日 | ✅ 安全 |
| `filter_high_volatility()` | 被09:11调用 | `get_price(end_date=y_day)` | ✅ 前一日 | ✅ 安全 |
| `filter_volume_price()` | 被09:11调用 | `get_price(end_date=y_day, count=35)` | ✅ 前一日 | ✅ 安全 |
| `filter_auction()` | 被09:27调用 | `get_price(end_date=y_day)` + `get_call_auction(09:15~09:26)` | ✅ 前一日 + 已结束竞价 | ✅ 安全 |
| `get_buy()` | 每日09:27 | `get_current_data()` + 竞价数据 | ✅ 盘前数据 | ✅ 安全 |
| `get_close_sell()` | 11:25/13:30/14:55 | `get_current_data()` + 缓存 | ✅ 实时数据 | ✅ 安全 |
| `get_minute_sell()` | 每分钟(9:30-11:25) | `get_current_data()` + 缓存 | ✅ 实时数据 | ✅ 安全 |
| `filter_first_board()` | 被09:11调用 | `get_price(end_date=y_day, count=2)` | ✅ 前一日 | ✅ 安全 |
| `precompute_minute_sell_cache()` | 每日09:11 | `get_price(end_date=y_day, count=2/5)` | ✅ 前一日 | ✅ 安全 |
| `after_trading_end()` | 收盘后自动 | 仅清理缓存 | ✅ 无数据访问 | ✅ 安全 |

### 1.3 未来函数审查结论

**✅ 未发现未来函数问题。** 策略在时间边界处理上非常规范：

1. 所有 `get_price()` 历史数据查询均使用 `end_date=y_day`（前一个交易日）
2. `get_call_auction()` 在09:27调用，竞价数据(09:15-09:26)已完整可用
3. `get_current_data()` 仅在交易时段内使用实时数据
4. 已启用 `avoid_future_data` 全局防护
5. `precompute_minute_sell_cache()` 在09:11预计算，使用前一日数据，避免交易时段重复API调用

---

## 二、策略评价

### 2.1 值得学习之处

#### ✅ 1. 多层过滤漏斗设计（优秀）

```
基础股票池(流通市值30-300亿) 
  → 首板筛选(昨日涨停+前日未涨停)
    → 量价过滤(成交额/放量/爆量/双阳线/涨幅限制)
      → 高波动过滤(5日波动≤20%)
        → 集合竞价过滤(竞价量≥3%+涨幅1-6%)
```

每层过滤都有明确的金融逻辑，层层递进缩小候选范围，最终只买入多条件共振的标的。这种"漏斗式"选股是短线策略的最佳实践。

#### ✅ 2. 盘前预计算缓存机制（优秀）

```python
# L435-474: precompute_minute_sell_cache()
# 09:11预计算，交易时段零API调用
g.minute_sell_cache = {...}  # 分钟止盈缓存
g.close_sell_cache = {...}   # 卖出缓存
```

将卖出所需的历史数据（昨日收盘、前日收盘、5日均价）在盘前一次性计算好，交易时段仅使用缓存+实时价格，**大幅减少API调用，避免超时风险**。这是聚宽策略的工程最佳实践。

#### ✅ 3. 多维度卖出体系（优秀）

| 卖出机制 | 触发时间 | 条件 | 设计意图 |
|----------|----------|------|----------|
| 收益止盈 | 3次/日 | 盈利>50% | 大赢收网 |
| 当日止损 | 3次/日 | 当日跌幅<-2% | 快速止损 |
| 开盘价止损 | 3次/日 | 现价<开盘价×0.96 | 弱势离场 |
| 持仓止损 | 3次/日 | 亏损>-7% | 硬止损 |
| 均线止损 | 3次/日 | 现价<MA5×0.97 | 趋势破位 |
| 线性止盈 | 每分钟 | 冲高回撤超容忍度 | 冲高回落锁定利润 |
| 昨亏今盈止盈 | 每分钟 | 昨亏>-2%且今盈>2% | 亏损反弹出局 |
| 开盘亏今盈止盈 | 每分钟 | 开盘亏>-2%且今盈>2% | 低开反弹出局 |
| 涨停持有 | 全部 | 价格≥涨停价×98.5% | 涨停不卖 |

**9种卖出条件覆盖了几乎所有持仓场景**，特别是"昨亏今盈"和"开盘亏今盈"两个条件，体现了对A股"低开反弹"特征的深刻理解。

#### ✅ 4. 集合竞价实时确认（良好）

```python
# L186-194: filter_auction()
auction = get_call_auction(stock, start_date=start, end_date=end, ...)
if (last.volume / hist['volume'] >= 0.03 and 1.0 < last.current / hist['close'] < 1.06):
    qualified.append(stock)
```

在09:27使用真实竞价数据确认买入意向，而非仅依赖历史数据选股。竞价量≥3%昨日成交量+竞价涨幅1-6%，有效过滤了"竞价冷清"和"竞价过高"的标的。

#### ✅ 5. 爆量排除逻辑（良好）

```python
# L138-143: filter_volume_price()
if (last.volume > avg_vol * 8 or last.volume > min_vol * 12) and last.close > group.high.iloc[-30:-1].max():
    continue  # 排除爆量创新高
```

识别"爆量+创新高"的危险组合——这通常是主力出货的信号，排除这类标的可以避免追高被套。

#### ✅ 6. 首板筛选的精确实现（良好）

```python
# L423-432: filter_first_board()
df['is_limit'] = df['close'] == df['high_limit']
groups = df[df['is_limit']].groupby('time')['code'].apply(set)
return list(groups.iloc[-1] - groups.iloc[-2])  # 昨日涨停 - 前日涨停 = 首板
```

使用集合差运算精确筛选"首板"（昨日涨停但前日未涨停），逻辑清晰且高效。`fill_paused=False` 确保停牌股票的NaN不会误判为涨停。

---

### 2.2 需要改进之处

#### 🔴 CRITICAL #1: `before_open()` 无返回值，`get_buy()` 的缓存回退逻辑会崩溃

```python
# L48-68: before_open() — 无return语句
def before_open(context): 
    ...
    g.today_pick = g.target_list.copy()  # 直接赋值给全局变量
    g.pick_cache_date = t_day
    # ❌ 没有return语句，隐式返回None

# L219-221: get_buy() — 缓存失效时调用before_open()
if g.pick_cache_date != t_day:
    g.today_pick = before_open(context)  # ❌ 赋值为None！
    g.pick_cache_date = t_day

# L224: 后续使用
candidates = [s for s in g.today_pick if s not in holdings]  # ❌ TypeError: NoneType不可迭代
```

**影响**: 正常流程下不会触发（`before_open`在09:11运行，`get_buy`在09:27运行），但如果`before_open()`因异常未执行，`get_buy()`的回退逻辑会导致策略崩溃。

**修复方案**:
```python
# 方案A: before_open()返回选股结果
def before_open(context):
    ...
    g.today_pick = g.target_list.copy()
    g.pick_cache_date = t_day
    return g.today_pick  # ✅ 添加返回值

# 方案B: get_buy()不依赖返回值
if g.pick_cache_date != t_day:
    before_open(context)  # ✅ before_open内部已设置g.today_pick
    g.pick_cache_date = t_day
```

---

#### 🔴 CRITICAL #2: `order_value()` 使用 `MarketOrderStyle(op)` 语义不清

```python
# L254
order = order_value(s, per_stock, MarketOrderStyle(op))
```

**问题**:
1. `MarketOrderStyle(limit_price=op)` 将开盘价作为限价，意味着订单不会以高于开盘价成交
2. 对于竞价高开的股票（涨幅1-6%），开盘后实际价格可能已高于开盘价，导致**订单无法成交**
3. 在09:27下单（市场未开盘），使用`day_open`作为限价——如果开盘后价格快速上冲，限价单会挂起无法成交

**影响**: 可能导致大量买入订单无法成交，实际买入率远低于预期。

**修复方案**:
```python
# 方案A: 使用市价单（无限制价格）
order = order_value(s, per_stock, MarketOrderStyle())

# 方案B: 使用限价单，给予一定上浮空间
order = order_value(s, per_stock, LimitOrderStyle(op * 1.02))  # 允许2%滑点
```

---

#### 🟠 HIGH #3: `filter_auction()` 中裸 `except:` 吞没所有异常 — ✅ 已修复

```python
# 修复前 L196-197
except:
    continue
```

**问题**: 裸`except:`会捕获所有异常，包括`KeyboardInterrupt`、`SystemExit`等不应被捕获的异常，且无法知道错误原因。

**修复方案**:
```python
except Exception as e:
    log.debug(f"[竞价过滤] {stock} 异常: {e}")
    continue
```

**实际修复**: `except Exception as e:` + `log.info(f"[竞价过滤异常] {stock}: {e}")` + error_count统计 + 汇总日志

---

#### 🟠 HIGH #4: `g.sold_stocks` 集合从未被使用 — ✅ 已修复

```python
# 修复前 L25: 初始化
g.sold_stocks = set()       # 已卖出股票集合（当前未使用）

# 修复前 L501: 清空（after_trading_end中）
g.sold_stocks.clear()
```

**问题**:
- 注释已标注"当前未使用"，但代码仍然初始化和清理
- 没有任何地方向`g.sold_stocks`添加元素
- 没有任何地方检查`g.sold_stocks`

**影响**: 死代码，增加理解成本。

**实际修复**: 将`g.sold_stocks`从死代码改为"当日已卖出股票集合（防止同日回买）"机制：
- `get_buy()`: 排除 `g.sold_stocks` 中的股票
- `get_close_sell()` / `get_minute_sell()`: 卖出后 `g.sold_stocks.add(s)`
- `after_trading_end()`: `g.sold_stocks.clear()`（次日可重新买入）

---

#### 🟡 MEDIUM #5: 所有阈值参数硬编码，无法灵活调优 — ✅ 已修复

| 参数 | 当前值 | 位置 | 含义 |
|------|--------|------|------|
| 成交额下限 | 5e8 | L125 | 5亿 |
| 成交额上限 | 30e8 | L125 | 30亿 |
| 放量倍数 | 2 | L129, L133 | 昨日量≥前日×2 |
| 爆量倍数 | 8/12 | L142 | 均量×8或最小量×12 |
| 短期涨幅限制 | 0.05 | L156 | 前两日涨幅<5% |
| 波动率上限 | 0.2 | L90 | 5日波动≤20% |
| 竞价量比 | 0.03 | L194 | 竞价量≥3%昨日量 |
| 竞价涨幅 | 1.0~1.06 | L194 | 竞价涨幅1-6% |
| 止盈线 | 0.5 | L303 | 盈利>50% |
| 当日止损 | -0.02 | L306 | 当日跌>-2% |
| 开盘价止损 | 0.96 | L309 | 现价<开盘价×96% |
| 持仓止损 | -0.07 | L312 | 亏损>-7% |
| 均线止损 | 0.97 | L315 | 现价<MA5×97% |
| 涨停持有线 | 0.985 | L296, L348 | 价格≥涨停价×98.5% |
| 冲高触发 | 0.03 | L397 | 最大涨幅>3% |
| 昨亏今盈 | -0.02/0.02 | L362 | 昨亏>-2%且今盈>2% |

**影响**: 16+个硬编码参数，调优时需要逐个修改源码，容易遗漏或引入错误。

**实际修复**: 30+个参数全部提取到`initialize()`中的`g.*`配置变量，按功能分组：
- 仓位控制: `g.max_hold_count`, `g.per_stock_cap_pct`
- 基础股票池: `g.min_market_cap`, `g.max_market_cap`, `g.new_stock_days`
- 量价过滤: `g.min_money`, `g.max_money`, `g.vol_ratio_min`, `g.prev_vol_ratio_max`, `g.explosion_vol_avg_ratio`, `g.explosion_vol_min_ratio`, `g.short_gain_limit`, `g.min_data_days`, `g.explosion_check_days`
- 波动过滤: `g.high_volatility_threshold`, `g.volatility_lookback`
- 竞价过滤: `g.auction_vol_ratio`, `g.auction_price_min`, `g.auction_price_max`, `g.auction_max_stocks`
- 卖出条件: `g.near_limit_pct`, `g.profit_take_pct`, `g.daily_drop_pct`, `g.open_drop_pct`, `g.max_loss_pct`, `g.ma5_drop_pct`
- 分钟止盈: `g.yesterday_loss_threshold`, `g.minute_profit_threshold`, `g.linear_tp_trigger`, `g.linear_tp_min_tolerance`, `g.linear_tp_base`, `g.linear_tp_offset`, `g.linear_tp_decay`
- 大盘过滤: `g.market_filter_enabled`, `g.market_index`, `g.market_ma_period`, `g.market_drop_pct`

---

#### 🟡 MEDIUM #6: 无大盘/市场环境过滤 — ✅ 已修复

**问题**: 策略在任何市场环境下都执行相同的买入逻辑，没有判断：
- 大盘是否处于下跌趋势（如沪深300低于20日均线）
- 市场整体情绪（如涨跌家数比）
- 是否处于特殊时期（如节假日前后、政策敏感期）

**影响**: 在熊市或急跌行情中，首板接力策略的胜率会大幅下降，回撤可能远超回测中的13.27%。

**实际修复**: 新增 `filter_market_environment(context)` 函数，在 `before_open()` 中调用：
- 条件1：参考指数收盘价 ≥ MA20（趋势向上）
- 条件2：参考指数前日跌幅未超过阈值（非暴跌日）
- 使用国证2000（`399303.XSHE`）作为参考指数（与策略基准一致）
- 优雅降级：数据不足或异常时默认允许买入
- 可通过 `g.market_filter_enabled = False` 关闭
- 大盘弱势时设置 `g.today_pick = []`，不执行任何买入

---

#### 🟡 MEDIUM #7: `filter_auction()` 逐只串行调用，效率低 — ✅ 已修复

```python
# 修复前 L184-198
for stock in stock_list:
    try:
        auction = get_call_auction(stock, start_date=start, end_date=end, ...)
        ...
    except:
        continue
```

**问题**: `get_call_auction()` 不支持批量查询，每只股票一次API调用。当候选股票较多时（如20-30只），串行调用耗时可能超过09:27→09:30的时间窗口。

**实际修复**:
1. 添加 `g.auction_max_stocks = 50` 限制，防止候选过多导致API超时
2. 预构建 `hist_map` 字典实现O(1)查找，替代 `hist_df[hist_df['code'] == stock]` 的O(n)逐行过滤
3. 预过滤无成交量/收盘价的股票（使用dict查找而非DataFrame过滤）
4. 添加统计日志：处理数/合格数/跳过数/异常数

---

#### 🟡 MEDIUM #8: `g.max_position_pct = 1` 变量名误导 — ✅ 已修复

```python
# 修复前 L34
g.max_position_pct = 1      # 单只股票最大仓位比例（100%）
```

**问题**:
- 注释说"单只股票最大仓位比例100%"，但实际用法是 `max_per = total * g.max_position_pct`（L242）
- 配合 `max_hold_count = 10`，每只股票实际分配约10%仓位
- 变量名暗示"单只最大仓位"，但值=1意味着"总资金的100%可作为单只上限"
- 如果真的允许单只100%仓位，与10只持仓上限的设计矛盾

**实际修复**: 重命名为 `g.per_stock_cap_pct = 1.0`，注释改为"单只股票最大资金分配比例（相对总资产）"，语义更清晰

---

#### 🔵 LOW #9: `order` 返回值未使用

```python
# L254, L319, L363, L374, L403
order = order_value(s, per_stock, MarketOrderStyle(op))
order = order_target_value(s, 0)
```

**问题**: 订单返回值赋给变量但从未使用，应检查订单是否成功。

**修复方案**:
```python
result = order_target_value(s, 0)
if result is None:
    log.warning(f"[卖出失败] {s}")
```

---

#### 🔵 LOW #10: `prepare_base_stocks()` 的股票代码前缀过滤可能误伤

```python
# L482
stock_list = [code for code in stock_list if not (
    code.startswith(('3', '68', '4', '8', '9'))
    ...
)]
```

**问题**: 
- `code.startswith('3')` 过滤掉所有300xxx（创业板）✅ 正确
- `code.startswith('68')` 过滤掉688xxx（科创板）✅ 正确
- `code.startswith('4')` 过滤掉400/430xxx（三板）✅ 正确
- `code.startswith('8')` 过滤掉8xxxxx（北证）✅ 正确
- `code.startswith('9')` 过滤掉9xxxxx — 但900xxx是B股，不是所有9开头都需要过滤

**影响**: 可能误过滤部分B股，但B股通常不在策略范围内，影响极小。

---

#### 🔵 LOW #11: 缺少买入后的确认和异常处理

```python
# L254
order = order_value(s, per_stock, MarketOrderStyle(op))
# ❌ 未检查订单是否成功
# L257-259
if not hasattr(g, 'information'):
    g.information = {}
g.information[s] = {'buy_date': t_day}  # ❌ 即使下单失败也记录买入日期
```

**问题**: 如果下单失败（如资金不足、股票停牌），仍然记录了买入日期，导致后续卖出逻辑跳过该股票（误认为"今日买入不卖"）。

**修复方案**:
```python
result = order_value(s, per_stock, MarketOrderStyle(op))
if result is not None and result.filled > 0:
    g.information[s] = {'buy_date': t_day}
    log.info(f"[买入成功] {s}")
else:
    log.warning(f"[买入失败] {s}")
```

---

#### 🔵 LOW #12: 回测业绩的可持续性存疑

| 指标 | 回测值 | 评估 |
|------|--------|------|
| 年化收益 | 864.73% | ⚠️ 极高，实盘难以复现 |
| 最大回撤 | 13.27% | ⚠️ 与高收益不匹配，可能低估 |
| 胜率 | 58.2% | 合理 |
| 交易次数 | 67次/16月 | 偏少，统计显著性不足 |
| 滑点 | 0.5% | ⚠️ 对小盘股可能不够 |

**风险因素**:
1. **小盘股流动性风险**: 流通市值30-300亿的股票，0.5%滑点可能不足以覆盖实际冲击成本
2. **涨停板买入困难**: 首板接力策略在实盘中面临"涨停买不到"的问题
3. **过拟合风险**: 16个月回测期、67次交易，参数空间大（16+个阈值），容易过拟合
4. **幸存者偏差**: 回测期(2025-2026)可能恰好适合首板接力策略

---

## 三、问题汇总

| # | 严重度 | 问题 | 位置 | 状态 |
|---|--------|------|------|------|
| 1 | 🔴 CRITICAL | `before_open()`无返回值，缓存回退崩溃 | L48/L219 | ✅ 已修复 |
| 2 | 🔴 CRITICAL | `MarketOrderStyle(op)`限价可能无法成交 | L254 | ✅ 已修复 |
| 3 | 🟠 HIGH | 裸`except:`吞没所有异常 | L196 | ✅ 已修复 |
| 4 | 🟠 HIGH | `g.sold_stocks`从未使用→改为防回买机制 | L25/L501 | ✅ 已修复 |
| 5 | 🟡 MEDIUM | 30+个阈值硬编码→提取到`g.*`配置 | 全局 | ✅ 已修复 |
| 6 | 🟡 MEDIUM | 无大盘/市场环境过滤→新增`filter_market_environment()` | — | ✅ 已修复 |
| 7 | 🟡 MEDIUM | `filter_auction()`效率低→dict O(1)+数量限制 | L184 | ✅ 已修复 |
| 8 | 🟡 MEDIUM | `g.max_position_pct`→`g.per_stock_cap_pct` | L34 | ✅ 已修复 |
| 9 | 🔵 LOW | `order`返回值未使用 | L254等 | ❌ 未修复 |
| 10 | 🔵 LOW | 股票代码前缀过滤可能误伤 | L482 | ❌ 未修复 |
| 11 | 🔵 LOW | 买入失败仍记录买入日期 | L257 | ❌ 未修复 |
| 12 | 🔵 LOW | 回测业绩可持续性存疑 | — | ❌ 未修复 |

**统计**: CRITICAL 2项(✅2) / HIGH 2项(✅2) / MEDIUM 4项(✅4) / LOW 4项(✅0)，共12项，已修复8项

---

## 四、评分细项

| 维度 | 评分 | 说明 |
|------|------|------|
| **未来函数防护** | 9.0/10 | 时间边界处理规范，所有历史查询使用y_day，已启用avoid_future_data |
| **选股逻辑** | 8.0/10 | 多层漏斗设计优秀，首板+量价+竞价三层确认，逻辑清晰 |
| **卖出体系** | 8.5/10 | 9种卖出条件覆盖全面，线性止盈设计精巧，涨停持有逻辑正确 |
| **代码质量** | 7.5/10 | ✅裸except已修复、✅死代码已激活、✅变量名已纠正、✅参数已集中配置 |
| **风控设计** | 7.5/10 | ✅大盘环境过滤已添加、个股止损完善、防同日回买机制已实现 |
| **可维护性** | 7.5/10 | ✅30+参数提取到g.*配置、✅竞价过滤效率优化、✅全局footprint日志 |
| **实盘可行性** | 5.0/10 | 小盘股流动性、涨停买入困难、滑点可能不足 |

### **综合评分: 7.5/10**

**评分说明**: 修复CRITICAL #1-2、HIGH #3-4、MEDIUM #5-8后，代码质量从6.0→7.5、风控从6.5→7.5、可维护性从5.5→7.5，综合评分从6.5提升至7.5。剩余LOW #9-12（order返回值检查、前缀过滤、买入确认、回测可持续性）不影响核心逻辑正确性，属于长期优化项。

---

## 五、改进建议

### ✅ 已完成修复（8项）

| # | 建议 | 优先级 | 状态 |
|---|------|--------|------|
| 1 | 修复`before_open()`返回值或`get_buy()`调用方式 | CRITICAL | ✅ 已修复 |
| 2 | 修改`MarketOrderStyle(op)`为`MarketOrderStyle()` | CRITICAL | ✅ 已修复 |
| 3 | 裸`except:`改为`except Exception as e:` + 异常日志 | HIGH | ✅ 已修复 |
| 4 | `g.sold_stocks`从死代码改为防同日回买机制 | HIGH | ✅ 已修复 |
| 5 | 30+个硬编码参数集中到`initialize()`的`g.*`配置 | MEDIUM | ✅ 已修复 |
| 6 | 新增`filter_market_environment()`大盘趋势过滤 | MEDIUM | ✅ 已修复 |
| 7 | `filter_auction()`优化：dict O(1)查找+数量限制 | MEDIUM | ✅ 已修复 |
| 8 | `g.max_position_pct`→`g.per_stock_cap_pct`语义清晰化 | MEDIUM | ✅ 已修复 |

### 长期优化（策略增强）

| # | 建议 | 优先级 |
|---|------|--------|
| 9 | 检查`order`返回值，下单失败不记录买入日期 | LOW |
| 10 | 增加更长的回测周期（3年+）验证策略稳健性 | LOW |
| 11 | 考虑实盘滑点对小盘股的影响，可能需要0.8-1% | LOW |
| 12 | 添加持仓时间限制（如持有超过N天强制卖出） | LOW |
