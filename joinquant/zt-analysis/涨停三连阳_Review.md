# 涨停三连阳 — 策略深度评审报告

> **策略来源**: [聚宽文章1](https://www.joinquant.com/post/71849) / [聚宽文章2](https://www.joinquant.com/post/71710)  
> **作者**: babakaifeiji / 股海掘金  
> **评审日期**: 2026-05-14  
> **综合评分**: ⭐ 6.5 / 10

---

## 一、策略概述

| 项目 | 内容 |
|------|------|
| 策略类型 | 首板次日竞价买入（涨停板接力） |
| 标的池 | 全A股（剔除ST/退市/科创板/北交所/新股） |
| 入场信号 | 昨日首板 + 5日波动≤20% + 市值30~300亿 + 量价过滤 + 集合竞价过滤 |
| 出场信号 | 线性止盈(分钟级) / 尾盘多条件止损 / 50%止盈 / -7%止损 / 跌破MA5*0.97 |
| 仓位管理 | 最多10只，单票最大100%总资产 |
| 大盘过滤 | 无 |
| 运行时间 | 09:10预计算 / 09:27买入 / 分钟止盈(9:30-11:25) / 尾盘卖出(11:25,13:30,14:55) |
| 基准指数 | 国证2000 (399303.XSHE) |

---

## 二、未来函数审查 ✅ 基本安全

### 2.1 全局防护设置 — ✅ 正确

```python
# 第18-19行
set_option('use_real_price', True)   # ✅ 使用真实价格
set_option("avoid_future_data", True) # ✅ 防止未来数据
```

两项基础防护均已开启，符合聚宽最佳实践。

---

### 2.2 逐函数审查

| 函数 | 行号 | 数据获取方式 | end_date | 结论 |
|------|------|-------------|----------|------|
| [`get_stock_list()`](涨停三连阳.py:49) | 49-84 | 调用子函数 | — | ✅ 安全 |
| [`filter_market_cap()`](涨停三连阳.py:87) | 98 | `get_valuation(..., start_date=y_day, end_date=y_day)` | `y_day=previous_date` | ✅ 安全 |
| [`filter_volume_price()`](涨停三连阳.py:107) | 120 | `get_price(..., end_date=y_day, frequency='1d', count=35)` | `y_day=previous_date` | ✅ 安全 |
| [`filter_auction()`](涨停三连阳.py:178) | 191 | `get_price(..., end_date=y_day, count=1)` | `y_day=previous_date` | ✅ 安全 |
| [`filter_auction()`](涨停三连阳.py:178) | 212 | `get_call_auction(s, start_date='T 09:15', end_date='T 09:26')` | 当日竞价时段 | ✅ 安全(09:27调用) |
| [`get_buy()`](涨停三连阳.py:230) | 234 | `get_current_data()` | 实时 | ✅ 安全 |
| [`get_close_sell()`](涨停三连阳.py:279) | 294 | `get_current_data()` | 实时 | ✅ 安全 |
| [`get_minute_sell()`](涨停三连阳.py:339) | 356 | `get_current_data()` | 实时 | ✅ 安全 |
| [`filter_high_volatility()`](涨停三连阳.py:389) | 400 | `get_price(..., end_date=y_day, count=5)` | `y_day=previous_date` | ✅ 安全 |
| [`prepare_stock_list()`](涨停三连阳.py:411) | 422 | `get_price(..., end_date=y_day, count=2)` | `y_day=previous_date` | ✅ 安全 |
| [`prepare_base_stocks()`](涨停三连阳.py:442) | 448-450 | `get_trade_days()` + `get_all_securities(date=by_date)` | 50交易日前 | ✅ 安全 |
| [`precompute_minute_sell_cache()`](涨停三连阳.py:464) | 477,491 | `get_price(..., end_date=y_day, count=2/5)` | `y_day=previous_date` | ✅ 安全 |

**结论**: 所有数据获取均使用 `context.previous_date`（昨日）或 `get_current_data()`（实时），无未来函数风险。

---

### 2.3 需关注的细节

#### 2.3.1 `get_call_auction()` 回测准确性 — 🟡 轻微关注

```python
# 第212行
auction = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'volume', 'current'])
```

- 在09:27调用，获取09:15-09:26的竞价数据，时间上无未来函数问题
- 但 `get_call_auction()` 在回测中返回的是**最终竞价结果**，而实盘中竞价数据是逐笔变化的
- 回测中取 `auction.iloc[-1]`（最后一条）即为最终竞价结果，与实盘在09:25后看到的结果一致 ✅
- **轻微风险**: 回测中竞价数据是确定性的，实盘中09:25-09:26的竞价数据可能仍在变化

#### 2.3.2 `MarketOrderStyle(op)` 执行假设 — 🟡 轻微关注

```python
# 第271行
order = order_value(s, per_stock, MarketOrderStyle(op))
```

- 在09:27以开盘价下单，假设能以开盘价成交
- 实盘中09:27下单，09:30开盘后成交价可能偏离开盘价
- 对于涨停板股票，买入本身极其困难，开盘价成交的假设偏乐观
- 这不是未来函数问题，而是**回测真实性**问题

---

## 三、策略评价

### 3.1 值得学习的地方 ✅

#### 1. 五层递进式选股过滤
```
首板标的 → 高波动过滤 → 市值过滤 → 量价过滤 → 集合竞价过滤
```
- 从粗到细，层层递进，逻辑清晰
- 每层过滤都有明确的业务含义，不是盲目堆砌因子
- 集合竞价作为最终过滤，利用了最接近交易时刻的信息

#### 2. 集合竞价数据实盘化
```python
# 条件1：竞价成交量 >= 昨日总成交量的3%
if al.volume / lv < 0.03:
    continue
# 条件2：竞价涨幅在0%~6%之间
ratio = al.current / lc
if ratio <= 1.0 or ratio >= 1.06:
    continue
```
- 使用真实竞价数据而非历史数据模拟，更接近实盘
- 竞价量≥3%过滤了无人关注的冷门股
- 竞价涨幅0%~6%区间：高开太多（>6%）追高风险大，低开/平开说明无延续性

#### 3. 盘前预计算缓存机制
```python
# 第464-503行 precompute_minute_sell_cache()
# 09:10预计算，交易时段零API调用
g.minute_sell_cache = {...}  # 分钟止盈缓存
g.close_sell_cache = {...}   # 尾盘卖出缓存
```
- 避免在每分钟/每次卖出检查时重复调用 `get_price()`
- 大幅减少API调用次数，提高运行效率
- 缓存与日期绑定，防止跨日数据污染

#### 4. 线性动态止盈
```python
# 第378-383行
if chr > 0.03:  # 最大涨幅超过3%
    lt = max(0.005, 0.02 - (chr - 0.01) * 0.5)  # 动态回撤容忍度
    if cp < th * (1 - lt):  # 回撤超过容忍度则卖出
        order_target_value(s, 0)
```
- 涨幅越大，回撤容忍度越小（从2%递减到0.5%）
- 3%涨幅→1%容忍，5%涨幅→0.5%容忍
- 冲高回落时快速锁利，避免利润回吐

#### 5. 涨停板持有策略
```python
# 第310行
if cp >= cd.high_limit * 0.99:  # 接近涨停继续持有
    continue
```
- 涨停股不轻易卖出，让利润奔跑
- 避免在涨停板打开前过早止盈

#### 6. 今日买入不卖出（T+1合规）
```python
# 第287-288行
hold = [s for s, p in context.portfolio.positions.items()
        if p.total_amount > 0 and g.information.get(s, {}).get('buy_date') != t_day]
```
- 正确实现A股T+1规则，当日买入的股票不参与卖出检查

#### 7. 爆量异常检测
```python
# 第152-157行
if (last.volume > avg_vol * 8 or last.volume > min_vol * 12) and last.close > group.high.iloc[-31:-1].max():
    continue  # 排除爆量创新高的股票
```
- 成交量异常放大（8倍均值或12倍最小值）+ 创31日新高 → 排除
- 这类股票可能是主力出货，追高风险极大

---

### 3.2 需要改进的地方 ❌

#### 🔴 P0 — 关键缺陷

**1. 涨停板买入可行性问题 — 回测与实盘严重脱节**

```python
# 第271行
order = order_value(s, per_stock, MarketOrderStyle(op))
```

- 策略核心是买入"昨日首板"股票，这类股票次日往往**高开甚至一字涨停**
- 一字涨停时买单远大于卖单，实际几乎无法买入
- 回测中 `MarketOrderStyle(op)` 假设以开盘价成交，但实盘中：
  - 一字涨停股：完全无法买入
  - 高开5%+：排队靠后，成交概率低
  - 正常开盘：可以买入，但这类股后续表现可能不佳
- **这是策略最大的实盘可行性问题**，回测收益可能大量来自实际无法成交的订单

**建议**: 增加可成交性检查，如开盘价 < 涨停价 * 0.98（排除一字涨停），或使用 `order_value()` 后检查订单状态

**2. `FixedSlippage(0.005)` 滑点设置与注释不符**

```python
# 第20行
set_slippage(FixedSlippage(0.005))  # 注释: 设置滑点为0.5%
```

- `FixedSlippage(0.005)` = 每股0.005元（0.5分钱），**不是0.5%**
- 对10元股：0.005/10 = 0.05%，对100元股：0.005/100 = 0.005%
- 涨停板股票波动大、流动性差，0.5分钱滑点严重低估实际滑点
- 应使用 `PriceRelatedSlippage(0.002)` (0.2%比例滑点) 更合理

**3. 尾盘卖出条件逻辑缺陷 — 盈利>50%可能被覆盖**

```python
# 第316-330行
if pp > 0.5:                          # 条件1: 盈利>50%
    sell, reason = True, "收益>50%"
if dg < -0.02:                        # 条件2: 当日跌>2% (独立if)
    sell, reason = True, f"当日涨幅<-2%"
elif cp < cd.day_open * 0.96:         # 条件3: 跌破开盘价4%
    ...
elif pp <= -0.07:                     # 条件4: 亏损>7%
    ...
elif cp < ma5 * 0.97:                 # 条件5: 跌破MA5 3%
    ...
```

- 条件1和条件2是**独立if**，若同时满足，条件2会覆盖条件1的reason
- 更严重的问题：若盈利>50%且当日跌幅<2%，条件1会触发卖出 — 这是正确的
- 但若盈利>50%且当日跌幅≥2%，条件2也会触发 — 卖出决策正确，只是reason被覆盖
- **实际Bug**: 条件1（盈利>50%）应该是独立判断，不应被后续条件影响reason

---

#### 🟡 P1 — 重要问题

**4. 分钟止盈仅覆盖上午，下午无监控**

```python
# 第346-347行
if t_day_time < time(9, 30) or t_day_time > time(11, 25):
    return  # 上午11:25后不再执行
```

- 下午13:00-15:00无分钟级止盈监控
- 下午冲高回落的场景完全无法捕获
- 应增加下午时段的监控，至少覆盖13:00-14:30

**5. `g.sold_stocks` 定义但从未使用**

```python
# 第27行
g.sold_stocks = set()       # 已卖出股票集合（当前未使用）

# 第511行
g.sold_stocks.clear()       # 每日清空，但从未add或查询
```

- 初始化了、清空了，但从未实际使用
- 疑似设计为"卖出后N日内不再买入"的冷却机制，但未实现

**6. `filter_auction()` 使用裸 `except`**

```python
# 第224行
except:
    continue
```

- 裸 `except` 会捕获所有异常，包括 `KeyboardInterrupt`、`SystemExit`
- 应改为 `except Exception:` 或更具体的异常类型
- 静默吞掉异常可能导致数据问题被隐藏

**7. 涨停判断容差过宽**

```python
# 第429行
df['is_limit'] = np.isclose(df.close, df.high_limit, rtol=0.005, atol=0.01)
```

- `rtol=0.005` = 0.5%相对容差
- 对10元股：收盘价在涨停价±0.05元内都算涨停，即9.95~10.05元都算涨停于10元
- 实际A股涨停价是精确的（前收盘×1.1，保留2位小数），收盘价应**等于**涨停价
- 0.5%容差可能将"接近涨停但未涨停"的股票也纳入，引入噪声
- 建议改为 `rtol=0.001` 或直接 `df.close >= df.high_limit * 0.998`

**8. 无行业/板块分散机制**

- 涨停板股票常呈板块效应（同一概念多股同日涨停）
- 10个仓位可能全部集中在同一热点板块
- 板块退潮时全部持仓同时亏损，集中度风险极高
- 建议限制同行业最多2-3只

---

#### 🟢 P2 — 改进建议

**9. `order` 变量名遮蔽内置函数**

```python
# 第271行
order = order_value(s, per_stock, MarketOrderStyle(op))
```

- 变量名 `order` 遮蔽了聚宽内置的 `order()` 函数
- 虽然是局部变量不影响其他函数，但容易引起混淆
- 建议改为 `order_result` 或 `order_id`

**10. 日志输出不一致**

- 混用 `log.info()`、`print()`、`log.set_level()`
- `log.set_level('order', 'error')` 抑制了订单日志，但 `print()` 输出不受控制
- 建议统一使用 `log.info()` 或自定义日志函数

**11. `g.max_position_pct = 1` (100%) 过于激进**

```python
# 第36行
g.max_position_pct = 1  # 单只股票最大仓位比例（100%）
```

- 允许单票占100%仓位，与"最多10只持仓"的设计矛盾
- 若首日只选出1只股票，全部资金买入一只涨停股，风险极大
- 建议设为0.15~0.20（15%~20%），确保分散

**12. 基础股票池仅每周更新**

```python
# 第40行
run_weekly(prepare_base_stocks, 1, '09:10')  # 每周一更新
```

- 新上市公司在周中不会被加入
- 更严重的是，若周一未运行（如回测起始日非周一），`g.base_stocks` 为空
- `prepare_stock_list()` 第418-419行有兜底检查，但依赖运行时触发
- 建议改为 `run_daily()` 或在 `prepare_stock_list()` 中确保初始化

**13. 无组合级回撤控制**

- 10只涨停股在极端行情下可能同时跌停
- 缺少整体组合级别的最大回撤保护
- 建议增加：当组合日内回撤超过X%时，全部清仓

**14. 量价过滤条件5命名与策略名不一致**

```python
# 第159-161行 — 条件5：双阳线要求
if not (d_b1.close > d_b1.open and d_b2.close > d_b2.open):
    continue
```

- 策略名"涨停三连阳"暗示3天阳线，但代码只要求T-2和T-3为阳线（2天）
- T-1为涨停（大阳线），加上T-2、T-3的阳线，确实是3连阳
- 但条件5的注释"双阳线"容易造成理解偏差

**15. 尾盘卖出函数名误导**

```python
# 第279行
def get_close_sell(context):
    """尾盘卖出函数：在11:25、13:30、14:55执行"""
```

- 11:25是上午盘中，不是"尾盘"
- 函数名 `get_close_sell` 暗示收盘前卖出，但实际在三个不同时段执行
- 建议改名为 `check_sell_conditions` 或 `intraday_sell_check`

**16. `precompute_minute_sell_cache` 中 `fill_paused=True` 可能引入失真**

```python
# 第479行
df2 = get_price(pos, end_date=y_day, frequency='1d',
                fields=['close'], count=2, panel=False,
                skip_paused=False, fill_paused=True)
```

- `fill_paused=True` 会用前值填充停牌日的数据
- 若股票停牌，填充的收盘价用于计算昨日涨幅，结果为0%
- 这可能导致 `y_day_gain = 0`，在分钟止盈中 `cache['y_day_gain'] >= 0` 判断为True，跳过止盈
- 停牌股不应参与止盈逻辑，应单独处理

---

## 四、问题优先级汇总

| # | 问题 | 优先级 | 类别 | 影响 |
|---|------|--------|------|------|
| 1 | 涨停板买入可行性 | P0 | 回测真实性 | 回测收益可能无法复现 |
| 2 | FixedSlippage注释与实际不符 | P0 | 回测真实性 | 严重低估滑点成本 |
| 3 | 尾盘卖出条件reason覆盖 | P0 | 逻辑Bug | 卖出原因记录错误 |
| 4 | 分钟止盈仅覆盖上午 | P1 | 策略逻辑 | 下午冲高回落无法捕获 |
| 5 | sold_stocks未使用 | P1 | 代码质量 | 死代码，冷却机制缺失 |
| 6 | 裸except | P1 | 代码质量 | 异常被静默吞掉 |
| 7 | 涨停判断容差过宽 | P1 | 选股准确性 | 可能选入非涨停股 |
| 8 | 无行业分散 | P1 | 风控 | 集中度风险 |
| 9 | order变量名遮蔽 | P2 | 代码质量 | 可读性差 |
| 10 | 日志不一致 | P2 | 代码质量 | 维护困难 |
| 11 | 单票100%仓位 | P2 | 风控 | 过度集中 |
| 12 | 基础池仅周更 | P2 | 数据时效 | 新股遗漏 |
| 13 | 无组合回撤控制 | P2 | 风控 | 极端行情风险 |
| 14 | 量价条件命名不一致 | P2 | 可读性 | 理解偏差 |
| 15 | 尾盘卖出函数名误导 | P2 | 可读性 | 理解偏差 |
| 16 | fill_paused失真 | P2 | 数据准确性 | 停牌股止盈逻辑错误 |

---

## 五、修复建议

### 5.1 涨停板可成交性检查

```python
def get_buy(context):
    # ... 现有逻辑 ...
    for s in to_buy:
        op = c_data[s].day_open
        hl = c_data[s].high_limit
        
        # 新增：排除一字涨停（无法买入）
        if op >= hl * 0.995:
            print(f"  [买入跳过] {s}: 一字涨停，无法买入")
            continue
        
        # 新增：高开超过5%降低仓位
        open_ratio = op / hl
        if open_ratio > 0.95:  # 高开接近涨停
            actual_per = per_stock * 0.5  # 减半仓位
        else:
            actual_per = per_stock
        
        shares = int(actual_per / op / 100) * 100
        # ...
```

### 5.2 滑点修复

```python
# 替换 FixedSlippage(0.005)
set_slippage(PriceRelatedSlippage(0.002))  # 0.2%比例滑点
```

### 5.3 分钟止盈覆盖下午

```python
def get_minute_sell(context):
    t_day_time = context.current_dt.time()
    # 扩展到下午14:30
    if t_day_time < time(9, 30) or t_day_time > time(14, 30):
        return
    # 午休时间跳过
    if time(11, 30) <= t_day_time <= time(13, 0):
        return
    # ... 其余逻辑不变
```

### 5.4 涨停判断容差收紧

```python
# 替换 np.isclose(rtol=0.005)
df['is_limit'] = df.close >= df.high_limit * 0.998  # 0.2%容差
```

### 5.5 卖出冷却机制（利用已有的sold_stocks）

```python
def get_buy(context):
    # ... 
    # 排除近期卖出的股票（3日冷却期）
    qualified = [s for s in g.today_pick 
                 if s not in holdings and s not in g.sold_stocks]

def close_position(context, stock, reason):
    # ...
    g.sold_stocks.add(stock)  # 记录卖出

def after_trading_end(context):
    # 3日后移除冷却（需记录卖出日期，此处简化为每日清空）
    g.sold_stocks.clear()
```

---

## 六、评分细项

| 维度 | 评分 | 说明 |
|------|------|------|
| 未来函数安全 | 9/10 | 两项防护齐全，数据获取规范，仅竞价数据回测准确性轻微存疑 |
| 策略逻辑 | 7/10 | 五层过滤+竞价数据+线性止盈设计精巧，但买入可行性是硬伤 |
| 风险控制 | 5/10 | 有多级止损止盈，但缺行业分散、组合回撤控制、单票100%仓位 |
| 代码质量 | 6/10 | 缓存机制优秀，但裸except、死代码、变量遮蔽、日志不一致 |
| 回测真实性 | 5/10 | 涨停板成交假设过于乐观，滑点设置不合理 |
| 可维护性 | 7/10 | 函数职责清晰，注释较完整，但硬编码参数、命名不一致 |
| **综合** | **6.5/10** | 框架设计优秀，未来函数防护到位，但实盘可行性和回测真实性是短板 |

---

## 七、总结

本策略是本次评审的**第四个策略**中未来函数防护做得最好的一个——`avoid_future_data` 和 `use_real_price` 均已开启，所有数据获取均使用 `context.previous_date` 或 `get_current_data()`，无未来函数污染。

策略的**设计亮点**突出：五层递进过滤、集合竞价实盘化、盘前缓存机制、线性动态止盈、爆量异常检测，这些设计思路值得借鉴。

然而，策略面临一个**根本性的实盘可行性问题**：买入昨日涨停股的次日开盘价，在实盘中大量订单无法成交（一字涨停买不到，高开太多成交率低）。回测中假设以开盘价100%成交，严重高估了策略的实际收益。配合偏低的固定滑点（0.5分钱），回测结果与实盘表现可能有巨大差距。

**建议优先级**:
1. 增加可成交性检查，过滤一字涨停股
2. 修正滑点为比例滑点
3. 扩展分钟止盈到下午时段
4. 增加行业分散和组合回撤控制
5. 修复代码质量问题（裸except、死代码等）

修正以上问题后重新回测，预期收益将显著下降，但策略的**实盘可执行性**和**结果可信度**将大幅提升。
