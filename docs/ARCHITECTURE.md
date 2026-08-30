# QuantCTA architecture

```text
data -> factors -> signals -> portfolio targets -> backtest -> metrics
approved target positions -> execution/ibkr (later)
```

The research path never imports IBKR. Closing TWS cannot prevent data validation,
factor calculation, historical backtesting or report generation.

Every source record has a market `timestamp` and an `available_at`. Research may
consume it only when `available_at <= decision_time`. A target decided at t is
delayed by at least one bar; PnL from t-1 to t belongs to the position held at t-1.

Factor, signal, price and position matrices have a unique increasing UTC index,
stable columns and finite float values. The engine never silently reindexes or
forward-fills them. A factor is a normal DataFrame-to-DataFrame function; it does
not inherit from a framework base class.

Adjusted continuous prices may be used for research, but PnL uses actual raw
contracts. `expand_root_positions` maps ES into dated contracts and a mapping
change creates two auditable roll trades.

Contract multiplier, tick size, currency, base-currency FX and costs are explicit. Notional, margin
and cash are different quantities. Broker margin reconciliation comes later.

Deliberately excluded from v0.1: tick/HFT simulation, queue modelling, partial
fills, portfolio optimization, automatic broker connectivity and plugin systems.
