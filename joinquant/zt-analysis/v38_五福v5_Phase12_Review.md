# v38_五福v5_Phase12 策略深度Review

> **策略名称**: 五福v5-人已疯魔版-11年385倍  
> **来源**: https://www.joinquant.com/post/71748  
> **作者**: Heyibigboss / 烟花三月ETF  
> **代码行数**: 1339行  
> **Review日期**: 2026-05-13  

---

## 一、未来函数检查

### 1.1 全局保护设置

| 设置项 | 值 | 评价 |
|--------|-----|------|
| `avoid_future_data` | `True` | ✅ 已启用，聚宽底层保护 |
| `use_real_price` | `True` | ✅ 已启用，使用真实价格交易 |
| `set_slippage` | `PriceRelatedSlippage(0.0001)` | ✅ 基金类型使用比例滑点，合理 |
| `set_order_cost` | 0.01%双边 + 5元最低 | ✅ 接近真实ETF交易成本 |

### 1.2 逐函数数据访问审计

#### ✅ `calculate_global_etf_threshold()` (L357-389)
- `get_all_securities(['etf'], date=context.current_dt)` → 使用当前日期 ✅
- `get_trade_days(end_date=context.previous_date, count=3)` → 截止昨日 ✅
- `get_price(security=etf_list, start_date=start_day, end_date=context.previous_date, ...)` → 截止昨日 ✅

#### ✅ `filter_global_pool_by_volume()` (L392-431)
- `end_date = context.previous_date` → 截止昨日 ✅
- `get_price(g.global_etf_pool, end_date=end_date, count=3, ...)` → 截止昨日 ✅

#### ✅ `update_sector_pool()` (L434-634)
- `get_all_securities(['etf'])` → ⚠️ 未显式传入日期参数（见下方Issue #1）
- `end_date = context.previous_date` (L532) → 截止昨日 ✅
- `get_price(etf_codes, end_date=end_date, count=3, ...)` (L540) → 截止昨日 ✅

#### ✅ `filter_fixed_pool_by_volume()` (L637-675)
- `end_date = context.previous_date` (L647) → 截止昨日 ✅
- `get_price(g.fixed_etf_pool, end_date=end_date, count=3, ...)` (L650) → 截止昨日 ✅

#### ✅ `check_a_share_weak_period()` (L852-924)
- `attribute_history(code, g.weak_period_ma_lookback + 1, '1d', ['close'], skip_paused=False)` (L864) → 分钟级回测中返回至前一交易日数据 ✅
- `df['close'][-1]` → 昨日收盘价 ✅
- `get_trade_days(start_date=g.weak_start_date, end_date=today)` (L884) → 截止今日 ✅

#### ⚠️ `get_final_ranked_etfs()` (L944-1102) — 核心动量计算函数
- `end_date = context.previous_date` (L947) → 截止昨日 ✅
- `get_price(etf_set, count=safe_lookback, end_date=end_date, ...)` (L954) → 截止昨日 ✅
- `get_price(etf_set, start_date=today, end_date=context.current_dt, frequency='1m', fields=['volume'], ...)` (L955) → 今日截至当前的分钟级成交量 ✅（实时数据，非未来）
- `get_price(etf_set, start_date=end_date, end_date=end_date, fields=['close'], ...)` (L962) → 昨日收盘 ✅
- `get_extras('unit_net_value', etf_set, start_date=end_date, end_date=end_date)` (L965) → 昨日净值 ✅
- `current_data[etf].last_price` (L988) → 当前实时价格 ✅
- ⚠️ `price_series = np.append(hist_closes, current_price)` (L726) → 数据频率混合问题（见下方Issue #2）

#### ✅ `calculate_premium_rate()` (L820-840)
- 使用 `g.etf_yesterday_close_batch` 和 `g.etf_yesterday_nav_batch` → 昨日数据 ✅
- 回退查询 `get_price(etf, start_date=context.previous_date, end_date=context.previous_date, ...)` → 昨日 ✅
- 回退查询 `get_extras('unit_net_value', etf, start_date=context.previous_date, end_date=context.previous_date)` → 昨日 ✅

#### ✅ `get_volume_ratio()` (L797-817)
- 使用 `hist_volumes`（历史数据）和 `today_vol`（今日截至当前累计成交量） ✅
- 成交量线性投影 `projected_today_vol = today_vol * (240.0 / elapsed_minutes)` → 估算，非未来函数 ✅

#### ✅ `minute_level_stop_loss()` (L1240-1266)
- `get_current_data()` → 实时数据 ✅
- `current_data[security].last_price` → 当前价格 ✅
- `position.avg_cost` → 持仓成本 ✅

#### ✅ `minute_level_pct_stop_loss()` (L1269-1311)
- `attribute_history(security, 1, '1d', ['close'], skip_paused=False)` (L1292) → 昨日收盘 ✅
- `get_current_data()` → 实时数据 ✅

#### ✅ `execute_sell_trades()` / `execute_buy_trades()` (L1105-1183)
- 使用 `g.ranked_etfs_result`（已计算的排名结果）和 `context.portfolio` → 无数据访问 ✅

#### ✅ `smart_order_target_value()` (L1186-1237)
- `get_current_data()` → 实时数据 ✅
- `current_data[security].last_price` → 当前价格 ✅

### 1.3 未来函数检查结论

**🟢 未发现严重未来函数问题。** 策略在数据访问方面整体规范：

1. 所有历史数据查询均使用 `context.previous_date` 作为截止日期
2. 实时数据通过 `get_current_data()` 获取，符合聚宽规范
3. `avoid_future_data=True` 提供底层保护
4. `attribute_history()` 在分钟级回测中正确返回至前一交易日数据

**⚠️ 存在2个轻微隐患（非严格未来函数，但需注意）：**

| # | 隐患 | 位置 | 严重程度 | 说明 |
|---|------|------|----------|------|
| 1 | `get_all_securities(['etf'])` 未传日期 | L482 | 🟡低 | 启用`avoid_future_data`后聚宽会自动处理为当前日期，但显式传入`date=context.current_dt`更规范 |
| 2 | 日频收盘价+日内实时价混合 | L726 | 🟡低 | `hist_closes`为日收盘序列，`current_price`为日内实时价，数据频率不一致可能导致动量得分微小偏差 |

---

## 二、策略架构解析

### 2.1 整体流程

```
09:00  morning_routine()
       ├── check_positions()          → 持仓检查
       ├── monitor_drawdown()         → 回撤监控
       └── calculate_global_etf_threshold() → 全市场流动性阈值

09:40  check_weak_period_daily()
       ├── check_a_share_weak_period() → 大A走弱期判断
       └── midday_routine()           → ETF池更新
           ├── [走弱期] filter_global_pool_by_volume() → 仅全球池
           └── [正常期] update_sector_pool() + filter_fixed_pool_by_volume() + daily_merge_etf_pools()

13:10  afternoon_routine()
       ├── [走弱期/正常期] 确定merged_etf_pool
       ├── calculate_and_log_ranked_etfs() → 动量计算与排序
       ├── execute_sell_trades()       → 卖出
       └── execute_buy_trades()        → 买入

15:10  reset_daily_flags()            → 重置缓存
```

### 2.2 ETF池三层架构

| 层级 | 名称 | 内容 | 用途 |
|------|------|------|------|
| 第一层 | 固定池 (`fixed_etf_pool`) | 全球18只 + 国内100只 = 118只 | 手工精选，正常期使用 |
| 第二层 | 动态池 (`dynamic_etf_pool`) | 全市场ETF按行业分组，每组取流动性最佳1只，最多100只 | 自动发现新ETF |
| 第三层 | 合并池 (`merged_etf_pool`) | 固定池∩流动性过滤 ∪ 动态池 | 最终动量计算范围 |

### 2.3 动量得分计算

```
momentum_score = annualized_returns × R²

其中:
- annualized_returns = exp(slope × 250) - 1
  - slope 来自加权线性回归: log(price) ~ time
  - 权重: 线性递增 W = (1→2)², 近期权重更大
- R² = 1 - SS_res/SS_tot (加权)
```

### 2.4 七重过滤条件

| 过滤器 | 条件 | 正常期 | 走弱期 |
|--------|------|--------|--------|
| 动量得分 | `min_score ≤ score ≤ max_score` | ✅启用 | ✅启用 |
| R² | `R² > 0.4` | ✅启用 | ✅启用 |
| 均线 | `price > MA10 × 1.0` | ❌硬编码禁用 | ❌硬编码禁用 |
| 成交量 | `量比 < 1.8` | ✅启用 | ❌关闭 |
| 短期风控 | `近3日单日跌幅 > -5%` | ✅启用 | ❌关闭 |
| 溢价率 | `溢价率 ≤ 30%` | ❌禁用 | ❌禁用 |
| 拉普拉斯 | `price > L且slope > 0.002` | ✅启用 | ❌关闭 |

### 2.5 大A走弱期机制

- **进入条件**: 4个指数中≥3个低于MA10
  - 沪深300、中证综合、创业板指、中证A500
- **退出条件**: ≥3个站上MA10 或 持续≥20个交易日
- **走弱期效果**: 
  - ETF池切换为仅全球/海外ETF（黄金、原油、纳指等）
  - 关闭成交量、短期风控、溢价率、拉普拉斯过滤
  - 拉普拉斯参数放宽（s: 0.05→0.10）
  - 候选池得分阈值比例从0.9变为1.0（更严格）

---

## 三、策略评价

### 3.1 ✅ 值得学习的亮点

#### 1. 严谨的未来函数防护
双保险设置 `avoid_future_data=True` + `use_real_price=True`，所有历史数据查询统一使用 `context.previous_date`，代码规范度高。

#### 2. 三层ETF池架构设计精巧
固定池保证核心覆盖，动态池自动发现新上市ETF，合并池取并集。这种"人工+自动"混合模式既保证了稳定性又兼顾了扩展性。

#### 3. 行业分组去重机制
`update_sector_pool()` 将全市场ETF按名称关键词分组（香港组、科创组、创业组、美指组等），每组只保留流动性最高的1只。这避免了同质化ETF占据多个名额，提高了池的多样性。

#### 4. 大A走弱期自适应
通过多指数MA判断市场状态，走弱期自动切换到全球/海外ETF池，同时放宽部分过滤条件。这是一种简单有效的regime detection机制。

#### 5. 加权动量回归
使用加权线性回归计算动量（近期权重更大），并用R²过滤掉趋势不稳定的ETF。比简单的收益率排名更科学。

#### 6. 拉普拉斯滤波器
`laplace_filter()` 是一种指数平滑滤波，用于判断价格趋势方向和斜率。比简单MA更平滑，对噪声不敏感。

#### 7. 持仓优先保留逻辑
`get_final_ranked_etfs()` 第四步中，当前持仓ETF如果在候选池中则优先保留，减少了不必要的换手。

#### 8. 防御ETF机制
无符合条件的ETF时自动切换到货币基金(511880.XSHG)，避免空仓资金闲置。

### 3.2 ❌ 需要改进的问题

#### 🔴 严重问题

**Issue #1: 单只持仓集中度风险极高**
- **位置**: L170 `g.holdings_num = 1`
- **问题**: 仅持有1只ETF，任何单日黑天鹅事件（如ETF停牌、跌停无法卖出）都可能造成巨大损失
- **影响**: 策略回撤完全取决于单只ETF表现，无分散化保护
- **建议**: 至少持有2-3只ETF，或引入动态持仓数机制（高确定性时集中，低确定性时分散）

**Issue #2: 38个版本的参数过拟合风险**
- **位置**: L26 `v38: 风控放宽 -3%→-5%`
- **问题**: 策略经过38个版本迭代，参数很可能已经过拟合到历史数据。标题"11年385倍"暗示年化约60%，远超巴菲特长期年化20%
- **影响**: 样本外表现可能大幅衰减
- **建议**: (1) 使用滚动窗口样本外测试 (2) 参数敏感性分析 (3) 减少可调参数数量

**Issue #3: 均线过滤器参数与代码不一致（死代码）**
- **位置**: L185 `g.enable_ma_filter = True` vs L931 `('均线', lambda m: m['passed_ma'], False)` 
- **问题**: 参数设为启用，但`apply_filters()`中硬编码为`False`，注释说"v30: 全部关闭MA过滤"。参数声明与实际行为矛盾
- **影响**: 误导性代码，维护者可能以为MA过滤在生效
- **建议**: 统一参数与代码，删除`g.enable_ma_filter`或将其与`apply_filters`关联

**Issue #4: 成交量投影假设不合理**
- **位置**: L809-815 `get_volume_ratio()`
- **问题**: 假设日内成交量线性分布（`projected_today_vol = today_vol * 240 / elapsed_minutes`），但A股成交量呈U型分布（开盘和收盘量大，午盘量小）
- **影响**: 上午计算的量比偏高，下午计算的量比偏低，导致上午更容易被成交量过滤剔除
- **建议**: 使用分时段权重（如开盘30分钟权重1.5，盘中权重0.8，尾盘30分钟权重1.3）

#### 🟡 中等问题

**Issue #5: `update_sector_pool()` 过于复杂且脆弱**
- **位置**: L434-634（200+行）
- **问题**: 依赖大量硬编码关键词匹配（基金公司名50+个、噪声词60+个、特殊组关键词20+个、排除关键词30+个），ETF更名或新ETF命名不规范就会失效
- **影响**: 维护成本极高，新ETF可能被错误分类或排除
- **建议**: 使用聚宽的行业分类API（如`get_industry()`）替代关键词匹配

**Issue #6: 日频与日内数据混合计算动量**
- **位置**: L726 `price_series = np.append(hist_closes, current_price)`
- **问题**: `hist_closes`来自日收盘价序列（截止昨日），`current_price`是13:10的日内实时价。两者数据频率不一致
- **影响**: 动量回归中最后一个数据点的含义与其他点不同（日内价vs收盘价），可能导致动量得分偏差
- **建议**: 统一使用日收盘价（不含当日实时价），或在分钟级框架下全部使用分钟收盘价

**Issue #7: 溢价率过滤器被禁用**
- **位置**: L193 `g.enable_premium_filter = False`
- **问题**: ETF溢价率是重要的风险指标，尤其对于QDII类ETF（如纳指、标普），溢价率可能高达10%+，买入高溢价ETF相当于高位接盘
- **影响**: 可能在高溢价时买入海外ETF，承担溢价回归损失
- **建议**: 至少对QDII/海外ETF启用溢价率过滤

**Issue #8: `check_weak_period_daily()` 函数命名误导**
- **位置**: L249-251
- **问题**: 函数名暗示只检查走弱期，但实际还调用了`midday_routine()`（ETF池更新）。且`midday_routine`在09:40执行，实际是"早盘"而非"午盘"
- **影响**: 代码可读性差，容易误解执行逻辑
- **建议**: 重命名为更准确的名称，或将ETF池更新逻辑独立调度

**Issue #9: 止损机制可能过于宽松**
- **位置**: L205 `g.fixedStopLossThreshold = 0.95`（5%止损）, L206 `g.use_pct_stop_loss = False`
- **问题**: 仅启用基于成本价的5%固定止损，当日跌幅止损被禁用。对于单只持仓策略，5%的亏损已经很大
- **影响**: 在快速下跌行情中可能来不及止损
- **建议**: 启用`minute_level_pct_stop_loss`作为补充，或降低固定止损阈值

**Issue #10: `monitor_drawdown()` 只记录不行动**
- **位置**: L326-354
- **问题**: 回撤监控仅记录日志，不触发任何风控行为（如减仓、切换防御ETF）
- **影响**: 回撤预警形同虚设
- **建议**: 增加回撤触发的风控动作，如回撤超过阈值时自动切换到防御ETF或降低仓位

#### 🟢 轻微问题

**Issue #11: `fmt_status` 在循环内重复定义**
- **位置**: L1004, L1031
- **问题**: `fmt_status` 函数在两个for循环内部定义，每次迭代都重新创建函数对象
- **影响**: 微小的性能损失
- **建议**: 将`fmt_status`移到循环外或模块级别

**Issue #12: `trade()` 空函数**
- **位置**: L1338-1339
- **问题**: `def trade(context): pass` — 空函数，可能是遗留代码
- **影响**: 无实际影响，但降低代码整洁度
- **建议**: 删除或添加注释说明用途

**Issue #13: `get_all_securities(['etf'])` 未显式传入日期**
- **位置**: L482
- **问题**: 虽然启用`avoid_future_data`后聚宽会自动处理，但显式传入`date=context.current_dt`更规范
- **影响**: 低风险，但不够严谨
- **建议**: 改为 `get_all_securities(['etf'], date=context.current_dt)`

**Issue #14: 日志量过大**
- **位置**: 全局，特别是L1000-1101的排名日志
- **问题**: 每日输出100+行ETF排名详情，在长期回测中会产生海量日志
- **影响**: 影响回测性能和日志可读性
- **建议**: 仅输出前10名详情，或添加日志级别控制

**Issue #15: 固定池中包含未来上市ETF**
- **位置**: L30-153
- **问题**: 如`159206.XSHE`（卫星ETF永赢，上市日期2025-03-14）、`159529.XSHE`（标普消费ETF景顺，上市日期2024-02-02）等，在回测早期这些ETF尚未上市
- **影响**: `get_price()`会自动跳过未上市ETF，不会导致未来函数，但增加了无效查询
- **建议**: 可忽略，聚宽API会自动处理

---

## 四、关键逻辑流程图

```
                    ┌─────────────────────┐
                    │   09:00 晨间流水线    │
                    │  持仓检查/回撤监控    │
                    │  流动性阈值计算       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   09:40 走弱期判断    │
                    │  4指数 vs MA10       │
                    └──────────┬──────────┘
                               │
                  ┌────────────┼────────────┐
                  │ 走弱期     │            │ 正常期
                  ▼                         ▼
        ┌─────────────────┐     ┌─────────────────┐
        │ 仅全球池流动性   │     │ 动态池+固定池    │
        │ 过滤(18只→N只)  │     │ 过滤+合并        │
        └────────┬────────┘     └────────┬────────┘
                 │                       │
                 └───────────┬───────────┘
                             │
                  ┌──────────▼──────────┐
                  │   13:10 午盘流水线    │
                  │  确定merged_etf_pool │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │   动量得分计算        │
                  │  加权回归 × R²       │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │   七重过滤           │
                  │  动量/R²/MA/量/      │
                  │  风控/溢价/拉普拉斯   │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │ 有候选       │              │ 无候选
              ▼                             ▼
    ┌──────────────────┐         ┌──────────────────┐
    │ 持仓优先保留      │         │ 防御ETF(511880)  │
    │ + 候选池补充      │         │ 或空仓           │
    └────────┬─────────┘         └──────────────────┘
             │
    ┌────────▼─────────┐
    │ 先卖后买执行      │
    │ smart_order       │
    └──────────────────┘
```

---

## 五、参数敏感性分析

### 5.1 高敏感参数（小幅调整可能显著影响结果）

| 参数 | 当前值 | 影响 |
|------|--------|------|
| `g.holdings_num` | 1 | 持仓集中度，改为2+将大幅改变策略特征 |
| `g.lookback_days` | 25 | 动量计算周期，影响趋势判断灵敏度 |
| `g.score_threshold_ratio` | 0.9 | 换仓门槛，0.9意味着第1名得分×0.9以上的都可选 |
| `g.fixedStopLossThreshold` | 0.95 | 止损幅度，直接影响最大回撤 |
| `g.r2_threshold` | 0.4 | R²过滤门槛，过高则候选极少 |

### 5.2 低敏感参数（调整影响有限）

| 参数 | 当前值 | 原因 |
|------|--------|------|
| `g.enable_ma_filter` | True | 代码中硬编码禁用，参数无效 |
| `g.enable_premium_filter` | False | 已禁用 |
| `g.use_pct_stop_loss` | False | 已禁用 |
| `g.max_premium_rate` | 30 | 溢价率过滤已禁用，参数无效 |

---

## 六、回测可靠性评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 未来函数防护 | ⭐⭐⭐⭐⭐ | 双保险+规范的数据访问模式 |
| 滑点模型 | ⭐⭐⭐⭐ | PriceRelatedSlippage合理，但0.01%可能偏小 |
| 交易成本 | ⭐⭐⭐⭐ | 0.01%双边+5元最低，接近真实 |
| 流动性考虑 | ⭐⭐⭐⭐⭐ | 三层流动性过滤+最小交易额限制 |
| 参数过拟合风险 | ⭐⭐ | 38版迭代+大量可调参数 |
| 风控完整性 | ⭐⭐⭐ | 有止损但偏宽松，回撤监控无行动 |
| 代码可维护性 | ⭐⭐ | 关键词匹配脆弱，参数与代码不一致 |

---

## 七、综合评分

| 类别 | 评分(10分制) | 说明 |
|------|-------------|------|
| 未来函数安全 | 9.0 | 无严重未来函数，仅有2个轻微隐患 |
| 策略逻辑 | 7.0 | 动量+多过滤+走弱期自适应设计合理，但单持仓风险过大 |
| 风控机制 | 5.5 | 止损偏宽松，回撤监控无行动，溢价率过滤禁用 |
| 代码质量 | 5.0 | 参数与代码不一致、命名误导、空函数、循环内定义函数 |
| 回测可信度 | 5.0 | 38版迭代过拟合风险高，"11年385倍"需谨慎看待 |
| **综合评分** | **6.3** | 框架设计优秀但细节问题较多，实盘需大幅改进风控 |

---

## 八、改进建议优先级

| 优先级 | 改进项 | 预期效果 |
|--------|--------|----------|
| P0 | 增加持仓数至2-3只 | 大幅降低集中度风险 |
| P0 | 启用溢价率过滤（至少对QDII ETF） | 避免高溢价买入 |
| P1 | 修复均线过滤参数/代码不一致 | 消除死代码，提高可维护性 |
| P1 | 回撤监控增加风控行动 | 回撤超阈值时自动降仓或切防御 |
| P1 | 启用当日跌幅止损 | 补充快速下跌场景保护 |
| P2 | 改进成交量投影算法 | 提高量比计算准确性 |
| P2 | 统一动量计算数据频率 | 消除日频/日内混合偏差 |
| P2 | 用行业分类API替代关键词匹配 | 提高动态池更新鲁棒性 |
| P3 | 减少日志输出量 | 提高回测性能 |
| P3 | 清理空函数和重复定义 | 提高代码整洁度 |

---

*Review完成。策略框架设计有诸多亮点（三层ETF池、走弱期自适应、加权动量回归），但单持仓集中度风险、参数过拟合、风控不足是主要隐忧。建议优先解决P0级问题后再考虑实盘。*
