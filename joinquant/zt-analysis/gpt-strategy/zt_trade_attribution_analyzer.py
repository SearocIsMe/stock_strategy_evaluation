from pathlib import Path

script = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zt_trade_attribution_analyzer.py

Purpose
-------
Parse JoinQuant backtest logs for the ZT strategy family and produce a
trade-attribution pack. This is NOT another strategy version. It is a diagnostic
tool to answer:

1. Which module actually created each buy?
2. Which market/regime context was active at the time?
3. Which labels had positive/negative expectancy?
4. Which sell reasons caused most losses?
5. Did "dragon/leader" labels really improve outcomes, or only look good in logs?

Usage
-----
python zt_trade_attribution_analyzer.py --log "log(30).txt" --out attribution_v26

Outputs
-------
- trades.csv
- buys.csv
- sells.csv
- by_state.csv
- by_regime.csv
- by_signal_family.csv
- by_sell_reason.csv
- daily_context.csv
- attribution_report.md

Notes
-----
The parser is deliberately tolerant because log wording changed from v8-v26.
It supports GB18030/GBK and UTF-8 logs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "cp936", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("gb18030", errors="replace")


def fnum(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(str(x).replace("%", "").strip())
    except Exception:
        return default


def inum(x, default=0):
    try:
        if x is None or x == "":
            return default
        return int(str(x).replace(",", "").strip())
    except Exception:
        return default


def safe_div(a, b):
    return a / b if b else 0.0


def classify_signal_family(text: str) -> str:
    t = text or ""
    if "v26" in t or "龙头集中" in t:
        return "V26_CONCENTRATED_DRAGON"
    if "v25" in t or "穿越龙" in t:
        return "V25_CROSS_DRAGON"
    if "v24" in t or "龙头周期" in t:
        return "V24_DRAGON_CYCLE"
    if "v22" in t or "龙头池" in t:
        return "V22_DRAGON_POOL"
    if "v21" in t or "eagle" in t.lower() or "鹰" in t:
        return "V21_EAGLE"
    if "REBOUND_RELAY" in t:
        return "REBOUND_RELAY"
    if "TOP1_TICK" in t or "tick" in t.lower():
        return "TOP1_TICK"
    if "SCORE_BUY" in t:
        return "SCORE_BUY"
    return "UNKNOWN"


def simplify_sell_reason(reason: str) -> str:
    r = reason or ""
    if "结构止损" in r:
        return "STRUCT_STOP"
    if "移动止盈" in r:
        return "TRAIL_STOP"
    if "利润保护" in r:
        return "PROFIT_PROTECT"
    if "龙头衰退" in r:
        return "DRAGON_DECAY"
    if "止损" in r:
        return "STOP_LOSS"
    if "止盈" in r:
        return "TAKE_PROFIT"
    if "回撤" in r:
        return "DRAWDOWN_EXIT"
    return "OTHER"


@dataclass
class DailyContext:
    trade_date: str
    regime: str = ""
    emotion: str = ""
    rawZT: Optional[int] = None
    idx5: Optional[float] = None
    idx20: Optional[float] = None
    hotCore: Optional[int] = None
    hotCont: Optional[int] = None
    topHot: Optional[float] = None
    topEntry: Optional[float] = None
    temp: Optional[float] = None
    disableTop1: Optional[str] = None
    maxScoreBuy: Optional[int] = None
    signal_count: Optional[int] = None
    signal_mix: str = ""
    stock_pool_size: Optional[int] = None
    holdings_count_eod: Optional[int] = None


@dataclass
class BuyEvent:
    buy_id: int
    dt: str
    trade_date: str
    code: str
    name: str = ""
    shares: int = 0
    price: float = 0.0
    amount: float = 0.0
    open_ret: Optional[float] = None
    rank: Optional[int] = None
    entry_index: Optional[float] = None
    signal_type: str = ""
    hot_state: str = ""
    hot_score: Optional[float] = None
    total_score: Optional[float] = None
    seen: Optional[int] = None
    runner: Optional[str] = None
    regime: str = ""
    emotion: str = ""
    rawZT: Optional[int] = None
    idx5: Optional[float] = None
    idx20: Optional[float] = None
    hotCore: Optional[int] = None
    hotCont: Optional[int] = None
    topHot: Optional[float] = None
    topEntry: Optional[float] = None
    temp: Optional[float] = None
    signal_text: str = ""
    signal_family: str = ""


@dataclass
class SellEvent:
    sell_id: int
    dt: str
    trade_date: str
    code: str
    name: str = ""
    shares: int = 0
    buy_price_reported: float = 0.0
    sell_price: float = 0.0
    pnl_pct_reported: Optional[float] = None
    reason: str = ""
    reason_group: str = ""


@dataclass
class ClosedTrade:
    trade_id: int
    code: str
    name: str
    buy_dt: str
    sell_dt: str
    hold_days: Optional[int]
    shares: int
    buy_price: float
    sell_price: float
    amount: float
    pnl_pct: float
    pnl_cash: float
    buy_rank: Optional[int]
    entry_index: Optional[float]
    signal_type: str
    signal_family: str
    hot_state: str
    hot_score: Optional[float]
    total_score: Optional[float]
    seen: Optional[int]
    runner: Optional[str]
    regime: str
    emotion: str
    rawZT: Optional[int]
    idx5: Optional[float]
    idx20: Optional[float]
    hotCore: Optional[int]
    hotCont: Optional[int]
    topHot: Optional[float]
    topEntry: Optional[float]
    temp: Optional[float]
    sell_reason: str
    sell_reason_group: str


LINE_DT = re.compile(r"^(?P<dt>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
REGIME_RE = re.compile(
    r"regime=(?P<regime>[A-Z_]+)\s*\|\s*emotion=(?P<emotion>[A-Z_]+)"
    r".*?rawZT=(?P<rawZT>-?\d+)"
    r".*?idx5=(?P<idx5>-?\d+(?:\.\d+)?)%"
    r".*?idx20=(?P<idx20>-?\d+(?:\.\d+)?)%"
    r".*?hotCore=(?P<hotCore>-?\d+)\s+hotCont=(?P<hotCont>-?\d+)"
    r".*?topHot=(?P<topHot>-?\d+(?:\.\d+)?)\s+topEntry=(?P<topEntry>-?\d+(?:\.\d+)?)"
    r".*?temp=(?P<temp>-?\d+(?:\.\d+)?)"
    r".*?disableTop1=(?P<disableTop1>True|False)"
    r".*?maxScoreBuy=(?P<maxScoreBuy>-?\d+)"
)
SIGNAL_GEN_RE = re.compile(r"\[generate_entry_signals\].*?生成\s+(?P<n>\d+)\s+个信号:\s+(?P<mix>\{.*?\})")
BUY_RE = re.compile(
    r"\[(?P<tag>[^\]]*成交)\]\s*Top(?P<rank>\d+)\s+"
    r"(?P<code>\d{6}\.(?:XSHE|XSHG))\s+(?P<shares>\d+)股\s*\|\s*"
    r"price=(?P<price>-?\d+(?:\.\d+)?),\s*ret=(?P<ret>-?\d+(?:\.\d+)?)%,\s*entry_index=(?P<entry>-?\d+(?:\.\d+)?)"
)
SELL_RE = re.compile(
    r"\[_sell_position\]\s+(?P<code>\d{6}\.(?:XSHE|XSHG))\s+(?P<name>\S+)\s+卖出\s+(?P<shares>\d+)股\s*\|\s*"
    r"原因:\s*(?P<reason>.*?)\s*\|\s*买入价:(?P<buy>-?\d+(?:\.\d+)?)\s+卖出价:(?P<sell>-?\d+(?:\.\d+)?)\s+盈亏:(?P<pnl>-?\d+(?:\.\d+)?)%"
)
RUNNER_RE = re.compile(
    r"\[v10利润奔跑标记\]\s+(?P<code>\d{6}\.(?:XSHE|XSHG))\s+runner=(?P<runner>True|False)\s*\|\s*"
    r"type=(?P<type>\w+)\s+state=(?P<state>[A-Z_]+)\s+hot=(?P<hot>-?\d+(?:\.\d+)?)\s+"
    r"entry=(?P<entry>-?\d+(?:\.\d+)?)\s+total=(?P<total>-?\d+(?:\.\d+)?)\s+seen=(?P<seen>-?\d+)"
)
SUGGEST_RE = re.compile(
    r"(?P<code>\d{6}\.(?:XSHE|XSHG))\s+(?P<name>.*?)\s+\|\s+类型:\s*(?P<type>[^|]+)\|\s+分类:\s*(?P<class>[^|]+)\|\s+信号:\s*(?P<signal>.*?)\s+\|\s+买入价:"
)
TOP_RE = re.compile(
    r"\s+(?P<rank>\d+)\.\s+(?P<code>\d{6})\s+(?P<name>.*?)\s+\|\s+评分:\s*(?P<score>-?\d+(?:\.\d+)?)"
    r".*?建仓指数:\s*(?P<entry>-?\d+(?:\.\d+)?)"
)
HOTPOOL_RE = re.compile(
    r"(?P<code>\d{6}\.(?:XSHE|XSHG))\s+state=(?P<state>[A-Z_]+)\s+hot=(?P<hot>-?\d+(?:\.\d+)?)\s+seen=(?P<seen>\d+)"
)
STOCK_POOL_RE = re.compile(r"股票池大小:\s*(?P<n>\d+)")
HOLDING_COUNT_RE = re.compile(r"当前持仓\s+\((?P<n>\d+)\s+只\)")


def parse_log(path: Path):
    text = read_text_auto(path)
    daily: Dict[str, DailyContext] = {}
    buys: List[BuyEvent] = []
    sells: List[SellEvent] = []
    signal_meta: Dict[Tuple[str, str], Dict] = {}
    suggestion_meta: Dict[Tuple[str, str], Dict] = {}
    top_meta: Dict[Tuple[str, str], Dict] = {}
    hot_meta: Dict[Tuple[str, str], Dict] = {}
    open_lots: Dict[str, deque] = defaultdict(deque)

    buy_id = 0
    sell_id = 0
    current_date = ""

    for line in text.splitlines():
        mdt = LINE_DT.search(line)
        if not mdt:
            continue
        dt = mdt.group("dt")
        trade_date = dt[:10]
        current_date = trade_date
        ctx = daily.setdefault(trade_date, DailyContext(trade_date=trade_date))

        mr = REGIME_RE.search(line)
        if mr:
            ctx.regime = mr.group("regime")
            ctx.emotion = mr.group("emotion")
            ctx.rawZT = inum(mr.group("rawZT"))
            ctx.idx5 = fnum(mr.group("idx5"))
            ctx.idx20 = fnum(mr.group("idx20"))
            ctx.hotCore = inum(mr.group("hotCore"))
            ctx.hotCont = inum(mr.group("hotCont"))
            ctx.topHot = fnum(mr.group("topHot"))
            ctx.topEntry = fnum(mr.group("topEntry"))
            ctx.temp = fnum(mr.group("temp"))
            ctx.disableTop1 = mr.group("disableTop1")
            ctx.maxScoreBuy = inum(mr.group("maxScoreBuy"))
            continue

        ms = SIGNAL_GEN_RE.search(line)
        if ms:
            ctx.signal_count = inum(ms.group("n"))
            ctx.signal_mix = ms.group("mix")

        mpool = STOCK_POOL_RE.search(line)
        if mpool:
            ctx.stock_pool_size = inum(mpool.group("n"))

        mhc = HOLDING_COUNT_RE.search(line)
        if mhc:
            ctx.holdings_count_eod = inum(mhc.group("n"))

        mrn = RUNNER_RE.search(line)
        if mrn:
            signal_meta[(trade_date, mrn.group("code"))] = {
                "runner": mrn.group("runner"),
                "signal_type": mrn.group("type"),
                "hot_state": mrn.group("state"),
                "hot_score": fnum(mrn.group("hot")),
                "entry_index": fnum(mrn.group("entry")),
                "total_score": fnum(mrn.group("total")),
                "seen": inum(mrn.group("seen")),
            }
            continue

        msug = SUGGEST_RE.search(line)
        if msug:
            suggestion_meta[(trade_date, msug.group("code"))] = {
                "name": msug.group("name").strip(),
                "signal_type": msug.group("type").strip(),
                "signal_text": msug.group("signal").strip(),
            }
            continue

        mt = TOP_RE.search(line)
        if mt:
            code = mt.group("code") + (".XSHG" if mt.group("code").startswith("6") else ".XSHE")
            top_meta[(trade_date, code)] = {
                "name": mt.group("name").strip(),
                "top_score": fnum(mt.group("score")),
                "entry_index": fnum(mt.group("entry")),
            }
            continue

        mh = HOTPOOL_RE.search(line)
        if mh:
            hot_meta[(trade_date, mh.group("code"))] = {
                "hot_state": mh.group("state"),
                "hot_score": fnum(mh.group("hot")),
                "seen": inum(mh.group("seen")),
            }
            continue

        mb = BUY_RE.search(line)
        if mb:
            buy_id += 1
            code = mb.group("code")
            meta = {}
            meta.update(hot_meta.get((trade_date, code), {}))
            meta.update(top_meta.get((trade_date, code), {}))
            meta.update(suggestion_meta.get((trade_date, code), {}))
            meta.update(signal_meta.get((trade_date, code), {}))
            ctx = daily.get(trade_date, DailyContext(trade_date=trade_date))

            price = fnum(mb.group("price"), 0.0) or 0.0
            shares = inum(mb.group("shares"))
            signal_text = meta.get("signal_text", "")
            be = BuyEvent(
                buy_id=buy_id,
                dt=dt,
                trade_date=trade_date,
                code=code,
                name=meta.get("name", ""),
                shares=shares,
                price=price,
                amount=shares * price,
                open_ret=fnum(mb.group("ret")),
                rank=inum(mb.group("rank")),
                entry_index=fnum(mb.group("entry")) if meta.get("entry_index") is None else meta.get("entry_index"),
                signal_type=meta.get("signal_type", "SCORE_BUY"),
                hot_state=meta.get("hot_state", ""),
                hot_score=meta.get("hot_score"),
                total_score=meta.get("total_score"),
                seen=meta.get("seen"),
                runner=meta.get("runner"),
                regime=ctx.regime,
                emotion=ctx.emotion,
                rawZT=ctx.rawZT,
                idx5=ctx.idx5,
                idx20=ctx.idx20,
                hotCore=ctx.hotCore,
                hotCont=ctx.hotCont,
                topHot=ctx.topHot,
                topEntry=ctx.topEntry,
                temp=ctx.temp,
                signal_text=signal_text,
                signal_family=classify_signal_family(signal_text or meta.get("signal_type", "")),
            )
            if be.signal_family == "UNKNOWN":
                be.signal_family = classify_signal_family(line)
            buys.append(be)
            open_lots[code].append({"remaining": shares, "buy": be})
            continue

        msel = SELL_RE.search(line)
        if msel:
            sell_id += 1
            se = SellEvent(
                sell_id=sell_id,
                dt=dt,
                trade_date=trade_date,
                code=msel.group("code"),
                name=msel.group("name"),
                shares=inum(msel.group("shares")),
                buy_price_reported=fnum(msel.group("buy"), 0.0) or 0.0,
                sell_price=fnum(msel.group("sell"), 0.0) or 0.0,
                pnl_pct_reported=fnum(msel.group("pnl")),
                reason=msel.group("reason").strip(),
                reason_group=simplify_sell_reason(msel.group("reason").strip()),
            )
            sells.append(se)
            continue

    trades = match_fifo(buys, sells)
    return daily, buys, sells, trades


def days_between(a: str, b: str) -> Optional[int]:
    try:
        da = datetime.strptime(a[:10], "%Y-%m-%d").date()
        db = datetime.strptime(b[:10], "%Y-%m-%d").date()
        return (db - da).days
    except Exception:
        return None


def match_fifo(buys: List[BuyEvent], sells: List[SellEvent]) -> List[ClosedTrade]:
    lots: Dict[str, deque] = defaultdict(deque)
    for b in buys:
        lots[b.code].append({"remaining": b.shares, "buy": b})

    trades: List[ClosedTrade] = []
    trade_id = 0
    for s in sells:
        remaining_sell = s.shares
        while remaining_sell > 0 and lots[s.code]:
            lot = lots[s.code][0]
            b: BuyEvent = lot["buy"]
            qty = min(remaining_sell, lot["remaining"])
            lot["remaining"] -= qty
            remaining_sell -= qty
            if lot["remaining"] <= 0:
                lots[s.code].popleft()

            trade_id += 1
            buy_price = b.price or s.buy_price_reported
            sell_price = s.sell_price
            pnl_pct = (sell_price / buy_price - 1.0) * 100 if buy_price else (s.pnl_pct_reported or 0.0)
            amount = qty * buy_price
            pnl_cash = qty * (sell_price - buy_price)

            trades.append(ClosedTrade(
                trade_id=trade_id,
                code=s.code,
                name=b.name or s.name,
                buy_dt=b.dt,
                sell_dt=s.dt,
                hold_days=days_between(b.dt, s.dt),
                shares=qty,
                buy_price=buy_price,
                sell_price=sell_price,
                amount=amount,
                pnl_pct=pnl_pct,
                pnl_cash=pnl_cash,
                buy_rank=b.rank,
                entry_index=b.entry_index,
                signal_type=b.signal_type,
                signal_family=b.signal_family,
                hot_state=b.hot_state,
                hot_score=b.hot_score,
                total_score=b.total_score,
                seen=b.seen,
                runner=b.runner,
                regime=b.regime,
                emotion=b.emotion,
                rawZT=b.rawZT,
                idx5=b.idx5,
                idx20=b.idx20,
                hotCore=b.hotCore,
                hotCont=b.hotCont,
                topHot=b.topHot,
                topEntry=b.topEntry,
                temp=b.temp,
                sell_reason=s.reason,
                sell_reason_group=s.reason_group,
            ))

        if remaining_sell > 0:
            trade_id += 1
            bp = s.buy_price_reported
            sp = s.sell_price
            trades.append(ClosedTrade(
                trade_id=trade_id,
                code=s.code,
                name=s.name,
                buy_dt="",
                sell_dt=s.dt,
                hold_days=None,
                shares=remaining_sell,
                buy_price=bp,
                sell_price=sp,
                amount=remaining_sell * bp,
                pnl_pct=s.pnl_pct_reported if s.pnl_pct_reported is not None else ((sp / bp - 1) * 100 if bp else 0.0),
                pnl_cash=remaining_sell * (sp - bp),
                buy_rank=None,
                entry_index=None,
                signal_type="UNMATCHED",
                signal_family="UNMATCHED",
                hot_state="",
                hot_score=None,
                total_score=None,
                seen=None,
                runner=None,
                regime="",
                emotion="",
                rawZT=None,
                idx5=None,
                idx20=None,
                hotCore=None,
                hotCont=None,
                topHot=None,
                topEntry=None,
                temp=None,
                sell_reason=s.reason,
                sell_reason_group=s.reason_group,
            ))

    return trades


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate(rows: List[ClosedTrade], key: str) -> List[dict]:
    groups = defaultdict(list)
    for t in rows:
        groups[getattr(t, key) if getattr(t, key) not in (None, "") else "NA"].append(t)

    out = []
    for k, ts in groups.items():
        pnl = [t.pnl_pct for t in ts]
        cash = [t.pnl_cash for t in ts]
        wins = [x for x in pnl if x > 0]
        losses = [x for x in pnl if x <= 0]
        out.append({
            key: k,
            "n": len(ts),
            "win_rate": round(100 * safe_div(len(wins), len(ts)), 2),
            "avg_pnl_pct": round(sum(pnl) / len(pnl), 4) if pnl else 0,
            "median_pnl_pct": round(sorted(pnl)[len(pnl)//2], 4) if pnl else 0,
            "sum_pnl_cash": round(sum(cash), 2),
            "avg_win_pct": round(sum(wins) / len(wins), 4) if wins else 0,
            "avg_loss_pct": round(sum(losses) / len(losses), 4) if losses else 0,
            "profit_factor": round(abs(sum([x for x in cash if x > 0]) / sum([x for x in cash if x < 0])), 4) if sum([x for x in cash if x < 0]) != 0 else "",
            "worst_pct": round(min(pnl), 4) if pnl else 0,
            "best_pct": round(max(pnl), 4) if pnl else 0,
        })
    return sorted(out, key=lambda r: (r["sum_pnl_cash"], r["avg_pnl_pct"]))


def make_report(outdir: Path, daily, buys, sells, trades):
    total_cash = sum(t.pnl_cash for t in trades)
    n = len(trades)
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    avg = safe_div(sum(t.pnl_pct for t in trades), n)
    win_rate = 100 * safe_div(len(wins), n)
    worst = sorted(trades, key=lambda t: t.pnl_cash)[:10]
    best = sorted(trades, key=lambda t: t.pnl_cash, reverse=True)[:10]

    by_family = aggregate(trades, "signal_family")
    by_state = aggregate(trades, "hot_state")
    by_regime = aggregate(trades, "regime")
    by_reason = aggregate(trades, "sell_reason_group")

    lines = []
    lines.append("# ZT Strategy Trade Attribution Report\n")
    lines.append("## Executive Summary\n")
    lines.append(f"- Closed trade lots parsed: **{n}**")
    lines.append(f"- Buy events parsed: **{len(buys)}**")
    lines.append(f"- Sell events parsed: **{len(sells)}**")
    lines.append(f"- Win rate by matched trade lot: **{win_rate:.2f}%**")
    lines.append(f"- Average trade PnL: **{avg:.2f}%**")
    lines.append(f"- Total matched cash PnL approximation: **{total_cash:.2f}**")
    lines.append("")
    lines.append("## What to inspect first\n")
    lines.append("1. `by_signal_family.csv`: tells whether V26/V24/V22/REBOUND labels actually had positive expectancy.")
    lines.append("2. `by_sell_reason.csv`: tells whether losses come from structure stops, trail stops, decay exits, etc.")
    lines.append("3. `trades.csv`: sort by `pnl_cash` ascending to inspect the biggest damages.")
    lines.append("4. `daily_context.csv`: compare trading days with `regime`, `rawZT`, `hotCore`, `topHot`, and `signal_count`.")
    lines.append("")
    lines.append("## By signal family\n")
    lines.append("| signal_family | n | win_rate | avg_pnl_pct | sum_pnl_cash | worst_pct | best_pct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in by_family:
        lines.append(f"| {r['signal_family']} | {r['n']} | {r['win_rate']} | {r['avg_pnl_pct']} | {r['sum_pnl_cash']} | {r['worst_pct']} | {r['best_pct']} |")
    lines.append("")
    lines.append("## By hot_state\n")
    lines.append("| hot_state | n | win_rate | avg_pnl_pct | sum_pnl_cash | worst_pct | best_pct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in by_state:
        lines.append(f"| {r['hot_state']} | {r['n']} | {r['win_rate']} | {r['avg_pnl_pct']} | {r['sum_pnl_cash']} | {r['worst_pct']} | {r['best_pct']} |")
    lines.append("")
    lines.append("## By regime\n")
    lines.append("| regime | n | win_rate | avg_pnl_pct | sum_pnl_cash | worst_pct | best_pct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in by_regime:
        lines.append(f"| {r['regime']} | {r['n']} | {r['win_rate']} | {r['avg_pnl_pct']} | {r['sum_pnl_cash']} | {r['worst_pct']} | {r['best_pct']} |")
    lines.append("")
    lines.append("## By sell reason\n")
    lines.append("| sell_reason_group | n | win_rate | avg_pnl_pct | sum_pnl_cash | worst_pct | best_pct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in by_reason:
        lines.append(f"| {r['sell_reason_group']} | {r['n']} | {r['win_rate']} | {r['avg_pnl_pct']} | {r['sum_pnl_cash']} | {r['worst_pct']} | {r['best_pct']} |")
    lines.append("")
    lines.append("## Worst 10 trade lots\n")
    lines.append("| code | name | buy_dt | sell_dt | family | state | regime | pnl_pct | pnl_cash | reason |")
    lines.append("|---|---|---|---|---|---|---|---:|---:|---|")
    for t in worst:
        lines.append(f"| {t.code} | {t.name} | {t.buy_dt} | {t.sell_dt} | {t.signal_family} | {t.hot_state} | {t.regime} | {t.pnl_pct:.2f} | {t.pnl_cash:.2f} | {t.sell_reason_group} |")
    lines.append("")
    lines.append("## Best 10 trade lots\n")
    lines.append("| code | name | buy_dt | sell_dt | family | state | regime | pnl_pct | pnl_cash | reason |")
    lines.append("|---|---|---|---|---|---|---|---:|---:|---|")
    for t in best:
        lines.append(f"| {t.code} | {t.name} | {t.buy_dt} | {t.sell_dt} | {t.signal_family} | {t.hot_state} | {t.regime} | {t.pnl_pct:.2f} | {t.pnl_cash:.2f} | {t.sell_reason_group} |")
    lines.append("")
    lines.append("## Important limitation\n")
    lines.append("This parser reconstructs trades from logs. It is good for attribution and debugging, but the JoinQuant `交易详情` export is still the authoritative source for exact cash accounting. If available, add that CSV later and compare.")
    (outdir / "attribution_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to JoinQuant log txt")
    ap.add_argument("--out", default="attribution_output", help="Output directory")
    args = ap.parse_args()

    path = Path(args.log)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    daily, buys, sells, trades = parse_log(path)

    write_csv(outdir / "daily_context.csv", [asdict(v) for k, v in sorted(daily.items())])
    write_csv(outdir / "buys.csv", [asdict(x) for x in buys])
    write_csv(outdir / "sells.csv", [asdict(x) for x in sells])
    write_csv(outdir / "trades.csv", [asdict(x) for x in trades])

    write_csv(outdir / "by_signal_family.csv", aggregate(trades, "signal_family"))
    write_csv(outdir / "by_state.csv", aggregate(trades, "hot_state"))
    write_csv(outdir / "by_regime.csv", aggregate(trades, "regime"))
    write_csv(outdir / "by_sell_reason.csv", aggregate(trades, "sell_reason_group"))

    make_report(outdir, daily, buys, sells, trades)

    print(f"Done. Output directory: {outdir.resolve()}")
    print(f"Parsed buys={len(buys)}, sells={len(sells)}, closed_trade_lots={len(trades)}")


if __name__ == "__main__":
    main()
'''

out = Path('/mnt/data/zt_trade_attribution_analyzer.py')
out.write_text(script, encoding='utf-8')
print(f"saved {out}") 
