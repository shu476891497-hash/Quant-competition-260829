# Data sources

## Tushare

Tushare is useful for Chinese futures, Chinese markets and macro inputs. Its
documented futures exchanges are CFFEX, DCE, CZCE, SHFE, INE and GFEX. The adapter
in this repository deliberately rejects other codes. Full permission does not
turn these endpoints into CME data.

Credentials are supplied only through `TUSHARE_TOKEN`. Never put a token in source
code, notebooks, config files, shell history or CI output.

## ES/NQ and global commodity futures

Use a separate provider for CME/CBOT/NYMEX/COMEX history. Practical choices are
IBKR historical bars for prototyping and reconciliation, or a dedicated futures
vendor for longer and cleaner contract/roll history. Convert source files to the
canonical Parquet schema and cache them locally, so research works with TWS closed.

The provider must supply actual contract OHLCV, expiry metadata, multiplier, tick
size, timezone/session data, and preferably volume/open interest for safe rolls.
