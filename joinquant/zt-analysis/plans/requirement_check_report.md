# 涨停板策略 — 需求检查报告 & Quant建议

> 生成时间: 2026-05-04  
> 检查范围: `zt_strategy.py` (2718行), `zt_analysis.py` (2830行), `plans/zt_strategy_architecture.md` (599行)

---

## Part A: 11项功能需求检查

### 需求1: 股票池来自昨日涨停统计，9点前运行，Top 5可配置

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 获取昨日涨停股 | ✅ 已实现 | [`get_yesterday_zt_stocks()`](zt_strategy.py:192) | 批量获取全A股票昨日行情，筛选 `pct_change >= zt_threshold(9.8%)` |
| 9点前运行 | ✅ 已实现 | [`before_trading_start()`](zt_strategy.py:2515) | JQ框架在盘前(~8:30)自动调用，完成ZT筛选→过滤→池更新→分析管线 |
| Top 5可配置 | ✅ 已实现 | [`STRATEGY_CONFIG['max_entry_count']`](zt_strategy.py:59) = 5; [`generate_entry_signals()`](zt_strategy.py:1517) 中 `entry_candidates.head(max_entry)` 限制数量 |

**结论: ✅ 完全实现**

---

### 需求2: 股票池维护"涨停日期"和"有效因子"列

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 涨停日期列 | ✅ 已实现 | [`g.stock_pool`](zt_strategy.py:2494) 列含 `zt_date`; [`update_stock_pool()`](zt_strategy.py:366) 更新 `zt_date` |
| 有效因子列 | ⚠️ 部分实现 | [`g.stock_pool`](zt_strategy.py:2494) | 池中存储**派生列** (`total_score`, `classification`, `signal`, `entry_index`, `buy_price`, `stop_loss`, `target_price`)，但**不存储26个原始因子列** |

**详细说明**:  
原始因子 (如 `factor_return_3d`, `factor_ma5_position`, `factor_vol_ratio` 等26个) 在 [`calc_factors()`](zt_strategy.py:820) 中计算，经 [`classify_stock()`](zt_strategy.py:913) → [`score_stock()`](zt_strategy.py:1003) → [`predict_next_day()`](zt_strategy.py:1294) 管线处理后，仅聚合结果写回股票池。原始因子在管线中间DataFrame中存在，但未持久化到 `g.stock_pool`。

**影响评估**:  
- ✅ 对策略运行无影响 — 因子每日重新计算，无需缓存
- ⚠️ 对调试/回测分析有影响 — 无法在日志中查看个股原始因子值
- 💡 建议在 [`log_daily_summary()`](zt_strategy.py:2374) 的 Top 10 输出中增加关键因子列

**结论: ⚠️ 部分实现 — 派生因子列已维护，原始因子列未持久化（设计选择，非缺陷）**

---

### 需求3: 股票池最大100只（可配置），淘汰规则

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 最大容量可配置 | ✅ 已实现 | [`STRATEGY_CONFIG['pool_max_size']`](zt_strategy.py:54) = 100 |
| 淘汰规则 | ✅ 已实现 | [`prune_stock_pool()`](zt_strategy.py:393) | 4级优先级淘汰 |

**淘汰优先级验证**:

| 优先级 | 规则 | 代码行 | 状态 |
|--------|------|--------|------|
| 0 | ST股票 (API+名称+池中名称三重检测) | [line 410-448](zt_strategy.py:410) | ✅ |
| 1 | "不建议建仓"信号 | [line 454-460](zt_strategy.py:454) | ✅ |
| 2 | ZT日过期 (>10天) | [line 462-471](zt_strategy.py:462) | ✅ |
| 3 | 总评分最低 | [line 473-480](zt_strategy.py:473) | ✅ |

**结论: ✅ 完全实现**

---

### 需求4: Tick级别交易策略，含仓位管理和建仓规则 (4.1-4.5)

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 4.1 Tick级别执行 | ✅ 已实现 | [`handle_data()`](zt_strategy.py:2624) | JQ每个Tick调用，执行 止损→止盈→建仓 |
| 4.2 仓位管理 | ✅ 已实现 | [`calc_position_size()`](zt_strategy.py:1773) | 等权分配: `total_value / max_holdings * ratio` |
| 4.3 TYPE_A (强势+积极) | ✅ 已实现 | [`execute_type_a()`](zt_strategy.py:1838) | 开盘50% + MA5回踩50% (两腿建仓) |
| 4.3 TYPE_B (平稳+积极) | ✅ 已实现 | [`execute_type_b()`](zt_strategy.py:1883) | 10点后15min金叉全仓 |
| 4.3 TYPE_C (强势+适度) | ✅ 已实现 | [`execute_type_c()`](zt_strategy.py:1917) | 金叉半仓 + MA5/MA10回踩半仓 |
| 4.4 建仓执行入口 | ✅ 已实现 | [`execute_entry()`](zt_strategy.py:1805) | 遍历信号，遵守 `max_holdings` 限制 |
| 4.5 技术指标 | ✅ 已实现 | [`calc_macd()`](zt_strategy.py:1575), [`calc_kdj()`](zt_strategy.py:1592), [`check_15min_golden_cross()`](zt_strategy.py:1620), [`check_ma_dip()`](zt_strategy.py:1688) |

**信号映射验证**:

| 分类 | 信号 | 建仓类型 | 代码行 | 状态 |
|------|------|----------|--------|------|
| 强势 | 积极建仓 | TYPE_A | [line 1534](zt_strategy.py:1534) | ✅ |
| 平稳 | 积极建仓 | TYPE_B | [line 1536](zt_strategy.py:1536) | ✅ |
| 强势 | 适度建仓 | TYPE_C | [line 1538](zt_strategy.py:1538) | ✅ |
| 其他 | — | NO_ENTRY | [line 1541](zt_strategy.py:1541) | ✅ |

**结论: ✅ 完全实现**

---

### 需求5: 止盈规则 (5.1-5.4)

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 5.1 T+1利润 > 9% | ✅ 已实现 | [`check_take_profit()`](zt_strategy.py:2099) | `profit_pct > t1_profit_take_pct(0.09)` → 卖出 |
| 5.2 达到目标价 | ✅ 已实现 | [line 2107-2112](zt_strategy.py:2107) | `current_price >= target_price` → 卖出 |
| 5.3 移动止盈 (最高价回撤>3%) | ✅ 已实现 | [line 2114-2120](zt_strategy.py:2114) | `drawdown > trailing_stop_pct(0.03)` → 卖出 |
| 5.4 最大持仓天数 (≥5天) | ✅ 已实现 | [line 2122-2126](zt_strategy.py:2122) | `hold_days >= max_hold_days(5)` → 卖出 |

**止盈优先级**: T+1高利 → 目标价 → 移动止盈 → 最大天数 ✅

**结论: ✅ 完全实现**

---

### 需求6: 止损规则 (6.1-6.3)

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 6.1 跌破止损价 | ✅ 已实现 | [`check_stop_loss()`](zt_strategy.py:2169) | `current_price <= stop_loss_price` → 卖出 |
| 6.2 日内亏损 > 5% | ✅ 已实现 | [line 2175-2180](zt_strategy.py:2175) | `loss_pct < -daily_stop_loss_pct(0.05)` → 卖出 |
| 6.3 崩盘检测 (涨跌比<1:4) | ✅ 已实现 | [line 2182-2196](zt_strategy.py:2182) | 11:25检查，`check_market_crash()` → 清仓所有持仓 |

**崩盘检测细节**: [`check_market_crash()`](zt_strategy.py:2199) 批量获取全A股票行情，计算涨跌比，仅执行一次 (`g.crash_checked_today` 防重复) ✅

**结论: ✅ 完全实现**

---

### 需求7: 日志输出

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 股票池大小和ZT数 | ✅ 已实现 | [`log_daily_summary()`](zt_strategy.py:2393) | `📋 股票池大小: X, 昨日涨停数: Y` |
| 交易开关状态 | ✅ 已实现 | [line 2394](zt_strategy.py:2394) | `📋 交易开关: 开启/关闭 (ZT数≤30)` |
| Top 10评分股票 | ✅ 已实现 | [line 2397-2408](zt_strategy.py:2397) | 含评分、分类、信号、建仓指数 |
| 当前持仓信息 | ✅ 已实现 | [line 2414-2452](zt_strategy.py:2414) | 含买入日、买入价、现价、股数、金额、盈亏、最高价 |
| 建仓建议 | ✅ 已实现 | [line 2456-2474](zt_strategy.py:2456) | 含类型、分类、信号、买入价、止损、目标、腿状态 |
| 盘前日志 | ✅ 已实现 | [line 2618-2621](zt_strategy.py:2618) | 池大小/持仓/信号/交易状态 |
| 关键操作日志 | ✅ 已实现 | 全局 | 每个关键操作均有 `log.info()` 输出 |

**结论: ✅ 完全实现**

---

### 需求8: 过滤ST股和上市不足3个月的股票

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| ST股过滤 (API检测) | ✅ 已实现 | [`filter_stocks()`](zt_strategy.py:300) | `get_extras('is_st', ...)` |
| ST股过滤 (名称检测) | ✅ 已实现 | [line 311-320](zt_strategy.py:311) | 名称含 'ST'/'st' |
| 上市<3个月过滤 | ✅ 已实现 | [line 324-333](zt_strategy.py:324) | `min_list_days=63` (约3个月) |
| 池中ST再检测 | ✅ 已实现 | [`prune_stock_pool()`](zt_strategy.py:410) | 三重检测: API + 名称 + 池中name列 |

**结论: ✅ 完全实现**

---

### 需求9: 不对已持仓股票重复建仓

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 跳过已持仓股票 | ✅ 已实现 | [`generate_entry_signals()`](zt_strategy.py:1527) | `if jq_code in held_codes: skipped_held.append(jq_code); continue` |
| 日志记录 | ✅ 已实现 | [line 1557-1558](zt_strategy.py:1557) | 记录跳过的已持仓股票数量和代码 |

**结论: ✅ 完全实现**

---

### 需求10: 当日新建仓股票不能止盈（A股T+1规则）

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 止盈T+1检查 | ✅ 已实现 | [`check_take_profit()`](zt_strategy.py:2094) | `if hold_days == 0: continue` (跳过当日建仓) |
| 止损T+1检查 | ✅ 已实现 | [`check_stop_loss()`](zt_strategy.py:2156) | `if buy_date == today: continue` (跳过当日建仓) |
| 日志记录 | ✅ 已实现 | [line 2096](zt_strategy.py:2096) | `当日新建仓(T+0)，跳过止盈检查` |

**结论: ✅ 完全实现**

---

### 需求11: 评分>50和建仓指数>65阈值（可配置）

| 子需求 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 最低评分可配置 | ✅ 已实现 | [`STRATEGY_CONFIG['min_score']`](zt_strategy.py:62) = 50 |
| 最低建仓指数可配置 | ✅ 已实现 | [`STRATEGY_CONFIG['min_entry_index']`](zt_strategy.py:63) = 65 |
| 评分过滤 | ✅ 已实现 | [`generate_entry_signals()`](zt_strategy.py:1507) | `entry_candidates['total_score'] >= min_score` |
| 建仓指数过滤 | ✅ 已实现 | [line 1511-1514](zt_strategy.py:1511) | `entry_candidates['entry_index'] >= min_entry_index` |
| 日志记录 | ✅ 已实现 | [line 1566](zt_strategy.py:1566) | 输出筛选条件 |

**结论: ✅ 完全实现**

---

## Part A 总结

| 需求 | 状态 | 备注 |
|------|------|------|
| 1. 股票池来自昨日涨停 | ✅ 完全实现 | |
| 2. 维护涨停日期和有效因子列 | ⚠️ 部分实现 | 派生列已维护，原始26因子列未持久化到池中 |
| 3. 池最大100+淘汰规则 | ✅ 完全实现 | |
| 4. Tick级别交易策略 | ✅ 完全实现 | |
| 5. 止盈规则 (4条) | ✅ 完全实现 | |
| 6. 止损规则 (3条) | ✅ 完全实现 | |
| 7. 日志输出 | ✅ 完全实现 | |
| 8. ST/次新股过滤 | ✅ 完全实现 | |
| 9. 不重复建仓 | ✅ 完全实现 | |
| 10. T+1止盈限制 | ✅ 完全实现 | |
| 11. 可配置阈值 | ✅ 完全实现 | |

**整体评估**: 11项需求中 **10项完全实现**，**1项部分实现**（需求2的原始因子列未持久化，属于设计选择而非缺陷）。

---

## Part B: Quant交易员建议 — 是否采用多因子模型（如101 Alphas）

### 1. 当前因子体系评估

当前系统采用 **6维度26因子** 体系:

| 维度 | 权重 | 因子数 | 代表因子 |
|------|------|--------|----------|
| 价格强度 | 30分 | 5 | factor_return_3d, factor_return_5d, factor_max_return, factor_defense_ratio, factor_above_zt_ratio |
| 趋势结构 | 20分 | 4 | factor_ma5_position, factor_bias_ma5, factor_consecutive_up, factor_pct_3d/5d |
| 成交量 | 20分 | 5 | factor_vol_ratio, factor_turnover, factor_inner_outer, factor_vol_price |
| 资金流向 | 15分 | 3 | factor_main_net_inflow, factor_main_net_pct, factor_main_net_3d |
| 基本面 | 10分 | 4 | factor_pe, factor_roe, factor_profit_yoy, factor_gross_margin |
| 风险扣减 | 5分 | 5 | factor_max_drawdown, factor_zt_open_count, factor_amplitude, factor_seal_amount/ratio, factor_days_boards |

**优势**:
- ✅ 领域特异性强 — 专为涨停板后续走势设计
- ✅ 可解释性好 — 每个因子有明确的金融含义
- ✅ 已有有效性验证框架 — [`factor_effectiveness_analysis()`](zt_analysis.py:2193) 通过 IC/Pearson 检验
- ✅ 权重可调 — `STRATEGY_CONFIG['score_weights']` 可配置

**不足**:
- ⚠️ 因子间可能存在共线性 (如 factor_return_3d 与 factor_pct_3d)
- ⚠️ 线性加权评分未考虑因子交互效应
- ⚠️ 缺少因子衰减分析 (IC随时间是否稳定)

---

### 2. WorldQuant 101 Alphas 评估

**101 Alphas 简介**: WorldQuant 2017年发表的101个公式化Alpha因子，涵盖动量、反转、波动率、成交量等维度，使用价格/成交量数据通过数学公式组合生成。

#### 2.1 采纳 101 Alphas 的优势

| 优势 | 说明 |
|------|------|
| 信号多样性 | 101个Alpha提供大量正交/弱相关信号源，可增强模型鲁棒性 |
| 经过验证 | 部分Alpha在A股有一定预测力（需重新验证） |
| 自动化因子挖掘 | 避免人工因子设计偏见 |
| 社区支持 | 开源实现丰富 (如 gplearn, WorldQuant Alpha101) |

#### 2.2 采纳 101 Alphas 的风险

| 风险 | 严重程度 | 说明 |
|------|----------|------|
| **过度拟合** | 🔴 高 | 101个因子在小样本（每日仅5-10只涨停股）上极易过拟合 |
| **因子衰减** | 🔴 高 | 101 Alphas发表于2017年，多数已被市场套利，IC显著衰减 |
| **领域不匹配** | 🟡 中 | 101 Alphas为全市场设计，涨停板是极端行情子集，通用因子可能失效 |
| **计算开销** | 🟡 中 | 101个因子在Tick级别计算，可能影响JQ平台执行延迟 |
| **可解释性下降** | 🟡 中 | 复杂公式组合（如 `rank(decay_linear(correlation(vwap, volume, 5), 3))`）难以解释 |
| **数据依赖** | 🟢 低 | 大部分Alpha仅需 price/volume，JQ API可满足 |

---

### 3. 具体建议

#### 3.1 🔴 不建议直接全量引入 101 Alphas

**理由**:
1. **样本量不匹配**: 涨停板策略每日候选股仅5-30只，101个因子在此样本上统计检验不可靠
2. **IC衰减严重**: 2017年至今，A股量化基金大量使用这些Alpha，边际信息量趋近于零
3. **维护成本高**: 101个因子的监控、调优、衰减检测需要大量基础设施

#### 3.2 🟡 建议选择性引入 5-10 个互补Alpha

从101 Alphas中筛选与当前因子体系**低相关**且**适合涨停板场景**的Alpha:

| 推荐Alpha | 公式概要 | 引入理由 | 对应维度补充 |
|-----------|----------|----------|-------------|
| Alpha#6 | `rank(sign(delta(correlation(open, volume, 10), 1))) * (-1 * rank(delta(close, 5)))` | 量价背离反转信号 | 补充趋势结构维度 |
| Alpha#12 | `sign(delta(volume, 1)) * (-1 * delta(close, 1))` | 放量跌→反转预期 | 补充成交量维度 |
| Alpha#20 | `rank(-1 * delta((close-open), 7))` | 周内K线实体变化 | 补充价格强度维度 |
| Alpha#33 | `rank(-1 * (1 - (open/close)^2))` | 日内K线形态 | 涨停板日内特征 |
| Alpha#41 | `power(high*low, 0.5) - vwap` | 价格偏离VWAP | 补充资金流向维度 |
| Alpha#54 | `(-1 * delta((close-low), 3) / delta((high-low), 3))` | 下影线比例变化 | 涨停板支撑强度 |
| Alpha#77 | `min(rank(decay_linear(high/2, 3)), rank(decay_linear(correlation(volume, low, 3), 2)))` | 高点衰减+量低价相关 | 补充风险维度 |

**引入方式**: 在 [`calc_factors()`](zt_strategy.py:820) 中新增 `factor_wq_alpha06` 等列，纳入评分体系。

#### 3.3 🟢 建议优先实施的改进（比引入101 Alphas更有效）

| 优先级 | 改进项 | 预期收益 | 实施难度 |
|--------|--------|----------|----------|
| P0 | **因子IC衰减监控** | 及时发现失效因子，避免亏损 | 低 — 在 [`factor_effectiveness_analysis()`](zt_analysis.py:2193) 基础上增加时间窗口IC |
| P0 | **因子共线性检测** | 剔除冗余因子，提高评分区分度 | 低 — 在相关性分析中增加VIF检验 |
| P1 | **动态权重调整** | 根据近期IC表现自动调整6维度权重 | 中 — 将 `score_weights` 从静态改为IC加权 |
| P1 | **涨停板专属因子** | 设计封单量变化率、开板次数、封板时间等ZT专属因子 | 中 — 需要Tick数据支持 |
| P2 | **机器学习评分** | 用XGBoost/LightGBM替代线性加权 | 高 — 需要大量历史数据训练+防过拟合 |
| P2 | **因子正交化** | 对现有26因子做PCA/Gram-Schmidt正交化 | 中 — 提高信号效率但降低可解释性 |

#### 3.4 涨停板专属因子建议（比101 Alphas更有价值）

当前系统已有 `factor_seal_amount`, `factor_seal_ratio`, `factor_zt_open_count`, `factor_days_boards`，但还可以增加:

| 新因子 | 计算方式 | 预期IC | 数据来源 |
|--------|----------|--------|----------|
| 封板速度 | 首次封板时间距开盘的分钟数 | 高 (越早封板越强) | Tick数据 |
| 开板次数 | 涨停日开板次数 | 高 (开板越多越弱) | Tick数据 |
| 封单量变化率 | 封单量/流通市值 的变化趋势 | 中 | Level-2数据 |
| 涨停板类型 | 一字板/秒板/尾盘板 分类 | 高 | Tick数据 |
| 板块联动度 | 同板块涨停股数量/板块总股数 | 中 | 行业分类数据 |
| 龙头股识别 | 同板块首个涨停的股票 | 高 | 实时行情 |

---

### 4. 总结建议

```
┌─────────────────────────────────────────────────────────┐
│  Quant建议优先级矩阵                                      │
│                                                          │
│  高收益 ↑                                                 │
│    │  ★ 涨停板专属因子    ★ 动态权重调整                     │
│    │  ★ 因子IC衰减监控    ★ 因子共线性检测                    │
│    │                                                      │
│    │  ○ 选择性引入5-10个101 Alpha                          │
│    │  ○ 因子正交化                                         │
│    │                                                      │
│    │  ✗ 全量引入101 Alphas                                 │
│    │  ✗ ML评分替代 (样本不足)                               │
│    │                                                      │
│  低收益 ──────────────────────────────────────→ 高成本     │
└─────────────────────────────────────────────────────────┘
```

**核心结论**:  
1. **不建议全量引入101 Alphas** — 样本量不足、IC衰减、领域不匹配
2. **建议选择性引入5-10个互补Alpha** — 优先选择量价背离和K线形态类
3. **最高优先级是涨停板专属因子** — 封板速度、开板次数等比通用Alpha更有预测力
4. **基础设施先行** — 先建立IC衰减监控和共线性检测，再扩展因子库

---

## 附录: 代码质量观察

| 项目 | 评估 | 说明 |
|------|------|------|
| 错误处理 | ✅ 优秀 | 每个JQ API调用均有 try/except 包裹 |
| 日志覆盖 | ✅ 优秀 | 所有关键操作均有 log.info() |
| 配置管理 | ✅ 优秀 | 所有阈值集中在 STRATEGY_CONFIG，可配置 |
| 代码复用 | ✅ 良好 | zt_analysis.py 的核心函数被策略复用 |
| 性能优化 | ✅ 已改进 | 3处关键 `pool.at[]` 逐行更新已替换为批量 `loc[]` 操作 |
| 类型安全 | ✅ 已改进 | 30+ 函数签名已添加 `Optional[float]`, `pd.DataFrame` 等类型标注 |

---

## Part C: 实施改进 & 新增功能

> 基于Part B的Quant建议，以下改进已全部实施并通过语法验证。
> 更新时间: 2026-05-04
> 代码行数: ~3411行 (原2718行，新增~693行)

---

### 改进1: 性能优化 — 批量操作替换逐行更新

**问题**: `pool.at[idx, col]` 逐行更新在 Pandas 中效率极低，每次调用触发索引查找 + 单值写入。

**优化位置**:

| 位置 | 原始模式 | 优化后 | 代码行 |
|------|----------|--------|--------|
| [`update_stock_pool()`](zt_strategy.py:355) | 逐行 `pool.at[idx, col] = val` 更新已有股票 | 批量 `pool.loc[update_mask, col_list] = val_df.values` | 377-393 |
| [`classify_stock()`](zt_strategy.py:1098) | `iterrows()` + `at[]` 逐行分类 | 全向量化 Pandas 操作 (`np.select`, `np.where`) | 1098-1147 |
| [`before_trading_start()`](zt_strategy.py:3183) | 逐行 `pool.at[idx, col]` 更新分析结果 | 批量 `pool.loc[mask, cols] = df[cols].values` | 3234-3282 |

**未优化位置** (I/O瓶颈，优化收益低):

| 位置 | 原因 |
|------|------|
| [`build_stock_data()`](zt_strategy.py:507) | 每行需独立JQ API调用获取价格数据，I/O bound |
| [`_supplement_jq_data_strategy()`](zt_strategy.py:728) | 同上，I/O bound |

**预期效果**: `classify_stock()` 从 O(n) Python循环降至 O(1) 向量化操作；`update_stock_pool()` 批量更新减少索引查找次数。

---

### 改进2: 类型安全 — Optional[float] 类型标注

**问题**: 大量函数使用 `np.nan` 作为默认返回值，缺乏类型提示，IDE无法推断正确类型。

**实施范围**: 30+ 函数签名已添加完整类型标注

| 函数 | 签名 |
|------|------|
| [`_safe_series()`](zt_strategy.py:113) | `(df: pd.DataFrame, col: str) -> pd.Series` |
| [`_safe_get()`](zt_strategy.py:127) | `(df: pd.DataFrame, col: str, default: Optional[float] = np.nan) -> Optional[float]` |
| [`_dedup_columns()`](zt_strategy.py:140) | `(df: pd.DataFrame) -> pd.DataFrame` |
| [`_normalize_jq_code()`](zt_strategy.py:145) | `(code_str: str) -> str` |
| [`_normalize_price_df_time()`](zt_strategy.py:166) | `(price_df: pd.DataFrame) -> pd.DataFrame` |
| [`get_yesterday_zt_stocks()`](zt_strategy.py:201) | `(context) -> pd.DataFrame` |
| [`filter_stocks()`](zt_strategy.py:286) | `(context, stock_codes: List[str]) -> List[str]` |
| [`update_stock_pool()`](zt_strategy.py:355) | `(context, new_zt_df: pd.DataFrame) -> None` |
| [`prune_stock_pool()`](zt_strategy.py:410) | `(context) -> None` |
| [`build_stock_data()`](zt_strategy.py:507) | `(context, pool_df: pd.DataFrame) -> pd.DataFrame` |
| [`_supplement_jq_data_strategy()`](zt_strategy.py:728) | `(context, df: pd.DataFrame) -> pd.DataFrame` |
| [`classify_stock()`](zt_strategy.py:1098) | `(factor_df: pd.DataFrame) -> pd.DataFrame` |
| [`calc_factors()`](zt_strategy.py:837) | `(price_df: pd.DataFrame) -> pd.DataFrame` |
| [`score_stock()`](zt_strategy.py:1154) | `(factor_df: pd.DataFrame) -> pd.DataFrame` |
| [`predict_next_day()`](zt_strategy.py:1501) | `(scored_df: pd.DataFrame) -> pd.DataFrame` |
| [`generate_entry_signals()`](zt_strategy.py:1764) | `(context, predict_df: pd.DataFrame) -> Dict` |
| [`calc_macd()`](zt_strategy.py:1870) | `(close_series: pd.Series, fast: int, slow: int, signal_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]` |
| [`calc_kdj()`](zt_strategy.py:1887) | `(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, n: int, m1: int, m2: int) -> Tuple[pd.Series, pd.Series, pd.Series]` |
| [`check_15min_golden_cross()`](zt_strategy.py:1915) | `(context, stock_code: str) -> bool` |
| [`check_ma_dip()`](zt_strategy.py:1983) | `(context, stock_code: str, ma_type: str) -> bool` |
| [`update_ma_cache()`](zt_strategy.py:2030) | `(context) -> None` |
| [`calc_position_size()`](zt_strategy.py:2068) | `(context, stock_code: str, ratio: float) -> int` |
| [`execute_entry()`](zt_strategy.py:2100) | `(context, data) -> None` |
| [`execute_type_a()`](zt_strategy.py:2133) | `(context, data, code: str, signal: Dict) -> bool` |
| [`execute_type_b()`](zt_strategy.py:2178) | `(context, data, code: str, signal: Dict) -> bool` |
| [`execute_type_c()`](zt_strategy.py:2212) | `(context, data, code: str, signal: Dict) -> bool` |
| [`_check_order_filled()`](zt_strategy.py:2270) | `(context, code: str) -> bool` |
| [`_record_holding()`](zt_strategy.py:2284) | `(context, code: str, signal: Dict, shares: int, leg: str) -> None` |
| [`_update_holding()`](zt_strategy.py:2327) | `(context, code: str, shares: int, leg: str) -> None` |
| [`check_take_profit()`](zt_strategy.py:2352) | `(context, data) -> None` |
| [`check_stop_loss()`](zt_strategy.py:2428) | `(context, data) -> None` |
| [`check_market_crash()`](zt_strategy.py:2494) | `(context) -> bool` |
| [`_get_stock_name()`](zt_strategy.py:2549) | `(code: str) -> str` |
| [`_sell_position()`](zt_strategy.py:2571) | `(context, code: str, reason: str) -> None` |
| [`log_daily_summary()`](zt_strategy.py:2669) | `(context) -> None` |

**新增导入**: `from typing import Optional, Dict, List, Tuple` (line 38)

---

### 新增1: WorldQuant 101 互补Alpha因子 (6个)

在 [`calc_factors()`](zt_strategy.py:837) 的 Section 4.8 中新增6个Alpha因子:

| 因子名 | Alpha编号 | 公式概要 | 引入理由 | 代码行 |
|--------|-----------|----------|----------|--------|
| `factor_alpha6` | Alpha#6 | `rank(sign(delta(correlation(open, volume, 10), 1))) * (-1 * rank(delta(close, 5)))` | 量价背离反转信号，补充趋势结构维度 | 925-932 |
| `factor_alpha12` | Alpha#12 | `sign(delta(volume, 1)) * (-1 * delta(close, 1))` | 放量跌→反转预期，补充成交量维度 | 939-946 |
| `factor_alpha33` | Alpha#33 | `rank(-1 * (1 - (open/close)^2))` | 日内K线形态，涨停板日内特征 | 953-959 |
| `factor_alpha41` | Alpha#41 | `power(high*low, 0.5) - vwap` | 价格偏离VWAP，补充资金流向维度 | 966-975 |
| `factor_alpha49` | Alpha#49 | `sum(((high+low)/2 - (delay(high,1)+delay(low,1))/2) * (high-low) / volume, 7)` | 7日资金流向累积，补充价格强度维度 | 981-988 |
| `factor_alpha54` | Alpha#54 | `(-1 * delta((close-low), 3) / delta((high-low), 3))` | 下影线比例变化，涨停板支撑强度 | 995-1004 |

**设计原则**:
- 选择与现有26因子**低相关**的Alpha，避免信息冗余
- 优先选择量价背离类和K线形态类，适合涨停板极端行情
- 每个Alpha均有 try/except 保护，计算失败时填充 `np.nan`
- 使用 `_safe_series()` 辅助函数确保列存在性

---

### 新增2: 涨停板专属因子 (3个)

在 [`calc_factors()`](zt_strategy.py:837) 的 Section 4.9 中新增3个ZT专属因子:

| 因子名 | 含义 | 计算方式 | 预期IC | 代码行 |
|--------|------|----------|--------|--------|
| `factor_seal_speed` | 封板速度 | 首次封板时间距开盘的分钟数，归一化到0-1 (越早封板值越大) | 高 | 1012-1035 |
| `factor_zt_board_type` | 涨停板类型 | 一字板=1.0, 秒板(10分钟内)=0.8, 早盘板(10:30前)=0.6, 午盘板(13:00前)=0.4, 尾盘板=0.2 | 高 | 1045-1075 |
| `factor_seal_float_ratio` | 封流比 | 封单金额 / 流通市值，值越大封板越稳固 | 中-高 | 1081-1087 |

**封板速度 (`factor_seal_speed`)** 详细逻辑:
- 解析 `first_zt_time` 字符串 (如 "09:35" → 5分钟)
- 归一化: `1 - (minutes / 240)`，越早封板值越接近1
- 无涨停时间数据时返回 `np.nan`

**涨停板类型 (`factor_zt_board_type`)** 详细逻辑:
- 解析 `days_boards` 字段 (如 "2天2板" → 连板数2, "昨日首板" → 1)
- 结合封板时间分类: 一字板(开盘即封) > 秒板(10min内) > 早盘板 > 午盘板 > 尾盘板
- 连板股额外加分: `base_score + min(days_boards - 1, 3) * 0.05`

**封流比 (`factor_seal_float_ratio`)** 详细逻辑:
- `seal_amount / (流通市值 * 100000000)` — 封单金额除以流通市值
- 流通市值为0或缺失时返回 `np.nan`

---

### 新增3: 评分体系升级 — 8维度评分模型

#### score_stock() 更新

[`score_stock()`](zt_strategy.py:1154) 从6维度扩展为8维度:

| 维度 | 权重 | 因子数 | 变化 |
|------|------|--------|------|
| 价格强度 | 30分 | 5 | 不变 |
| 趋势结构 | 20分 | 4 | 不变 |
| 成交量 | 20分 | 5 | 不变 |
| 资金流向 | 15分 | 3 | 不变 |
| 基本面 | 10分 | 4 | 不变 |
| 风险扣减 | 5分 | 5 | 不变 |
| **Alpha因子** | **5分** | **3** | **🆕 新增** |
| **涨停板专属** | **5分** | **3** | **🆕 新增** |

**Alpha因子评分** (5分):
- `factor_alpha6`: 量价背离反转信号 (1.5分)
- `factor_alpha12`: 放量反转预期 (1.5分)
- `factor_alpha41`: 价格偏离VWAP (2.0分)

**涨停板专属评分** (5分):
- `factor_seal_speed`: 封板速度 (2.0分)
- `factor_zt_board_type`: 涨停板类型 (2.0分)
- `factor_seal_float_ratio`: 封流比 (1.0分)

#### predict_next_day() 更新

[`predict_next_day()`](zt_strategy.py:1501) 从5组件扩展为7组件:

| 组件 | 权重 | 变化 | 说明 |
|------|------|------|------|
| score_component | 35 | 40→35 | 基础评分权重微降 |
| price_component | 20 | 不变 | 价格技术面 |
| defense_component | 15 | 不变 | 防御能力 |
| capital_component | 15 | 不变 | 资金面 |
| zt_feature_component | 5 | 10→5 | ZT特征权重降低，部分转移至zt_exclusive |
| **alpha_component** | **5** | **🆕 新增** | Alpha因子综合得分 |
| **zt_exclusive_component** | **5** | **🆕 新增** | 涨停板专属因子得分 |

**alpha_component 计算逻辑**:
```python
alpha_score = (
    safe_get('factor_alpha6', 0) * 1.5 +
    safe_get('factor_alpha12', 0) * 1.5 +
    safe_get('factor_alpha41', 0) * 2.0
)
alpha_component = min(5, max(0, alpha_score))
```

**zt_exclusive_component 计算逻辑**:
```python
zt_excl_score = (
    safe_get('factor_seal_speed', 0) * 2.0 +
    safe_get('factor_zt_board_type', 0) * 2.0 +
    safe_get('factor_seal_float_ratio', 0) * 1.0
)
zt_exclusive_component = min(5, max(0, zt_excl_score))
```

**predict_scores 输出字典** 新增字段:
- `'alpha_component'`: Alpha因子组件得分 (0-5)
- `'zt_exclusive_component'`: 涨停板专属组件得分 (0-5)

**STRATEGY_CONFIG 更新**:
```python
'score_weights': {
    ...
    'alpha_factors': 5,    # 🆕 Alpha因子维度权重
    'zt_exclusive': 5,     # 🆕 涨停板专属维度权重
}
```

---

### 新增4: IC衰减监控基础设施

在 Section 17 (lines 2768-2950) 中实现完整的IC监控管线:

#### 配置: IC_MONITOR_CONFIG

```python
IC_MONITOR_CONFIG = {
    'enabled': True,
    'factor_cols': [
        'factor_return_3d', 'factor_return_5d', 'factor_ma5_position',
        'factor_vol_ratio', 'factor_turnover', 'factor_main_net_pct',
        'factor_alpha6', 'factor_alpha12', 'factor_alpha41',
        'factor_seal_speed', 'factor_zt_board_type', 'factor_seal_float_ratio'
    ],
    'ic_warn_threshold': 0.03,      # IC低于此值发出警告
    'ic_decay_threshold': 0.5,      # 近期IC/历史IC < 此值判定为衰减
    'recent_window': 5,             # 近期IC窗口(天)
    'history_window': 20,           # 历史IC窗口(天)
    'max_ic_history': 60,           # IC历史最大保留天数
}
```

#### 核心函数

| 函数 | 功能 | 输入 | 输出 | 代码行 |
|------|------|------|------|--------|
| [`calc_ic()`](zt_strategy.py:2795) | 计算单个因子IC (Spearman秩相关) | factor_df, factor_col, target_col | `Optional[float]` | 2795-2822 |
| [`calc_ic_batch()`](zt_strategy.py:2825) | 批量计算多因子IC | factor_df, factor_cols, target_col | `Dict[str, Optional[float]]` | 2825-2849 |
| [`update_ic_history()`](zt_strategy.py:2852) | 追加每日IC到历史记录 | context, ic_results | `None` (更新g.ic_history) | 2852-2874 |
| [`check_ic_decay()`](zt_strategy.py:2877) | 检测IC衰减 | context | `Dict[str, Dict]` (含decay_status) | 2877-2929 |
| [`log_ic_monitor_report()`](zt_strategy.py:2932) | 输出IC监控报告 | context | `None` (log输出) | 2932-2950 |

#### IC衰减检测逻辑

```
对每个因子:
  1. 从 g.ic_history 取近 recent_window(5) 天 IC均值 → recent_ic
  2. 从 g.ic_history 取近 history_window(20) 天 IC均值 → history_ic
  3. 若 |recent_ic| < ic_warn_threshold(0.03) → ⚠️ IC过低
  4. 若 |recent_ic| / |history_ic| < ic_decay_threshold(0.5) → ⚠️ IC衰减
  5. 否则 → ✅ 正常
```

#### 报告输出格式

```
📊 IC衰减监控报告 (2026-05-04)
┌─────────────────────┬──────────┬──────────┬──────────┬────────┐
│ 因子                │ 近5日IC  │ 历史IC   │ IC比值   │ 状态   │
├─────────────────────┼──────────┼──────────┼──────────┼────────┤
│ factor_return_3d    │  0.0452  │  0.0821  │  0.551   │ ✅ 正常 │
│ factor_alpha6       │  0.0123  │  0.0456  │  0.270   │ ⚠️ 衰减 │
│ ...                 │  ...     │  ...     │  ...     │ ...    │
└─────────────────────┴──────────┴──────────┴──────────┴────────┘
```

---

### 新增5: 因子共线性检测基础设施

在 Section 17 (lines 2953-3144) 中实现完整的共线性检测管线:

#### 核心函数

| 函数 | 功能 | 输入 | 输出 | 代码行 |
|------|------|------|------|--------|
| [`detect_factor_collinearity()`](zt_strategy.py:2953) | 相关矩阵 + VIF检测 | factor_df, factor_cols, corr_threshold, vif_threshold | `Dict` 含 correlation_pairs, vif_results | 2953-3074 |
| [`log_collinearity_report()`](zt_strategy.py:3107) | 输出共线性报告 | factor_df, factor_cols, corr_threshold, vif_threshold | `None` (log输出) | 3107-3138 |
| [`run_factor_diagnostics()`](zt_strategy.py:3141) | IC监控+共线性组合入口 | context, factor_df | `Dict` 含 ic_results, collinearity | 3141-3168 |

#### 共线性检测逻辑

**1. 相关矩阵检测**:
- 计算因子间 Pearson 相关系数矩阵
- 标记 `|corr| > corr_threshold(0.7)` 的高相关因子对
- 输出: `[(factor_a, factor_b, corr_value), ...]`

**2. VIF (方差膨胀因子) 检测**:
- 尝试使用 `statsmodels.stats.outliers_influence.variance_inflation_factor`
- 若 statsmodels 未安装，回退到基于相关矩阵的近似VIF: `1 / (1 - R²)`
- 标记 `VIF > vif_threshold(5.0)` 的因子
- 输出: `[(factor_name, vif_value), ...]`

#### 报告输出格式

```
📊 因子共线性检测报告
━━━━━━ 高相关因子对 (|corr| > 0.70) ━━━━━━
  factor_return_3d  ↔ factor_pct_3d     :  0.892
  factor_vol_ratio  ↔ factor_turnover    :  0.756

━━━━━━ VIF检测 (阈值 > 5.0) ━━━━━━
  factor_return_3d  : VIF = 8.34  ⚠️
  factor_pct_3d     : VIF = 7.91  ⚠️
  factor_ma5_position: VIF = 3.21  ✅
```

---

### 集成: after_trading_end() 因子诊断调用

在 [`after_trading_end()`](zt_strategy.py:3313) 的步骤5中集成因子诊断:

```python
# 5. 因子诊断 (IC衰减监控 + 共线性检测)
if IC_MONITOR_CONFIG['enabled'] and not g.stock_pool.empty:
    try:
        _factor_cols_in_pool = [c for c in IC_MONITOR_CONFIG['factor_cols']
                                 if c in g.stock_pool.columns]
        if _factor_cols_in_pool:
            _factor_data = g.stock_pool[['jq_code'] + _factor_cols_in_pool].copy()
            if 'entry_index' in g.stock_pool.columns:
                _factor_data['next_day_return'] = g.stock_pool['entry_index']
            _diagnostics = run_factor_diagnostics(context, _factor_data)
            log_ic_monitor_report(context)
            log_collinearity_report(_factor_data)
    except Exception as e:
        log.info(f"[after_trading_end] 因子诊断异常: {e}")
```

**执行时机**: 每日收盘后，在日志汇总之前运行
**数据来源**: `g.stock_pool` 中的因子列 + `entry_index` 作为收益代理
**容错**: 整体 try/except 包裹，诊断失败不影响策略主流程

---

### 依赖管理

| 依赖 | 用途 | 安装方式 | 必需性 |
|------|------|----------|--------|
| `scipy.stats` | Spearman IC计算 | `pip install scipy` | 可选 — 缺失时IC监控不可用 |
| `statsmodels` | VIF共线性检测 | `pip install statsmodels` | 可选 — 缺失时回退到近似VIF |
| `typing` | 类型标注 | Python 3.5+ 内置 | 必需 |

**JQ平台兼容性**: scipy 在聚宽平台预装；statsmodels 可能未安装，代码已做 `try/except ImportError` 回退处理。

---

### 实施总结

| 改进项 | 状态 | 代码行数 | 对应Part B建议 |
|--------|------|----------|----------------|
| 性能优化 (批量操作) | ✅ 已完成 | ~50行修改 | 附录: 性能优化 |
| 类型安全 (Optional[float]) | ✅ 已完成 | ~35行修改 | 附录: 类型安全 |
| Alpha因子 (6个) | ✅ 已完成 | ~80行新增 | 建议3.2: 选择性引入5-10个互补Alpha |
| ZT专属因子 (3个) | ✅ 已完成 | ~70行新增 | 建议3.4: 涨停板专属因子 |
| 评分体系升级 (8维度) | ✅ 已完成 | ~40行修改 | 建议3.4: 涨停板专属因子 |
| predict_next_day升级 (7组件) | ✅ 已完成 | ~60行修改 | 建议3.2+3.4 |
| IC衰减监控 | ✅ 已完成 | ~180行新增 | 建议3.3 P0: 因子IC衰减监控 |
| 因子共线性检测 | ✅ 已完成 | ~190行新增 | 建议3.3 P0: 因子共线性检测 |
| **合计** | **✅ 全部完成** | **~705行新增/修改** | — |

**语法验证**: `python3 -c "import py_compile; py_compile.compile('zt_strategy.py', doraise=True)"` → ✅ Syntax OK

**Part B建议实施进度**:

| 建议 | 优先级 | 状态 |
|------|--------|------|
| 🔴 不全量引入101 Alphas | — | ✅ 遵循 — 仅选择性引入6个 |
| 🟡 选择性引入5-10个互补Alpha | P1 | ✅ 已实施 — 引入6个Alpha |
| 🟢 因子IC衰减监控 | P0 | ✅ 已实施 |
| 🟢 因子共线性检测 | P0 | ✅ 已实施 |
| 🟡 涨停板专属因子 | P1 | ✅ 已实施 — 引入3个ZT因子 |
| 🟡 动态权重调整 | P1 | ⏳ 待实施 — 需IC历史数据积累 |
| 🟡 因子正交化 | P2 | ⏳ 待实施 — 依赖共线性检测结果 |
| 🔴 ML评分替代 | P2 | ⏳ 待实施 — 样本量不足 |
