# IBKR execution boundary

This directory is intentionally empty in the research MVP. The future adapter
will consume approved `target_positions` and translate them into IBKR contracts
and orders. Factors and the backtest engine must never import IBKR/TWS code.

Before order submission it must reconcile account, positions, open orders and
contract metadata, then enforce pre-trade limits and a kill switch.
