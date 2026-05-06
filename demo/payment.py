"""
Payment processor — handles charge and retry logic.
"""

import time

_processed = {}  # tracks completed charges


def process_charge(order_id: str, amount: float) -> dict:
    """Charge a card for the given order. Returns result dict."""
    result = _charge_card(order_id, amount)
    _processed[order_id] = result
    return result


def _charge_card(order_id: str, amount: float) -> dict:
    """Simulate card charge. Raises TimeoutError 30% of the time."""
    import random
    if random.random() < 0.3:
        raise TimeoutError("Payment gateway timeout")
    return {"order_id": order_id, "amount": amount, "status": "charged"}


def process_with_retry(order_id: str, amount: float, retries: int = 2) -> dict:
    """Process a charge, retrying on timeout, ensuring idempotency."""
    if order_id in _processed:
        return _processed[order_id]
    for attempt in range(retries + 1):
        try:
            result = process_charge(order_id, amount)
            return result
        except TimeoutError:
            if attempt == retries:
                raise
            time.sleep(0.1)
