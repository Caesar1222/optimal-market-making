"""
Order-book + trade recorder for the market-making project.

Collects periodic L2 order-book snapshots and recent trades from a crypto
exchange and appends them to JSONL files for later analysis.

Usage:
    python record.py
Stop with Ctrl+C. Data is flushed continuously, so partial runs are safe.
"""
import ccxt
import json
import os
import time

EXCHANGE_ID = "kraken"       # US-accessible; "coinbase" also works
SYMBOL = "BTC/USD"
DEPTH_LIMIT = 25             # order-book levels per side
POLL_SECONDS = 1.0          # snapshot interval
DATA_DIR = "data"
BOOK_FILE = os.path.join(DATA_DIR, "book.jsonl")
TRADE_FILE = os.path.join(DATA_DIR, "trades.jsonl")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    print(f"Recording {SYMBOL} from {EXCHANGE_ID}. Ctrl+C to stop.")

    seen_trades = set()

    with open(BOOK_FILE, "a") as bf, open(TRADE_FILE, "a") as tf:
        while True:
            try:
                # --- order-book snapshot ---
                book = exchange.fetch_order_book(SYMBOL, limit=DEPTH_LIMIT)
                ts = book.get("timestamp") or int(time.time() * 1000)
                bf.write(json.dumps({
                    "ts": ts,
                    "bids": book["bids"],   # [[price, amount], ...], best first
                    "asks": book["asks"],
                }) + "\n")
                bf.flush()

                # --- recent trades (for arrival-rate estimation) ---
                trades = exchange.fetch_trades(SYMBOL, limit=50)
                for t in trades:
                    tid = t.get("id") or f"{t['timestamp']}-{t['price']}-{t['amount']}"
                    if tid in seen_trades:
                        continue
                    seen_trades.add(tid)
                    tf.write(json.dumps({
                        "ts": t["timestamp"],
                        "price": t["price"],
                        "amount": t["amount"],
                        "side": t.get("side"),
                    }) + "\n")
                tf.flush()

                # cap the dedupe set so memory doesn't grow unbounded
                if len(seen_trades) > 5000:
                    seen_trades = set(list(seen_trades)[-2000:])

                time.sleep(POLL_SECONDS)

            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"Error: {e} — retrying in 5s")
                time.sleep(5)


if __name__ == "__main__":
    main()