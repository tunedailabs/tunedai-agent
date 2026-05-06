"""
Tests for payment processor.
Run: python3 -m pytest demo/test_payment.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demo.payment as pm
from demo.payment import _processed


def test_no_double_charge():
    """A retried payment must not charge the card twice."""
    _processed.clear()
    charge_calls = []

    def mock_charge(oid, amt):
        if oid not in charge_calls:
            charge_calls.append(oid)
            raise TimeoutError("gateway timeout")
        return {"order_id": oid, "amount": amt, "status": "charged"}

    original = pm._charge_card
    pm._charge_card = mock_charge
    try:
        pm.process_with_retry("order-001", 99.99)
        # Ensure only one successful charge
        assert charge_calls.count("order-001") == 1, (
            f"order-001 was charged {charge_calls.count('order-001')} times — double charge bug!"
        )
    finally:
        pm._charge_card = original


def test_idempotent_retry():
    """process_charge called twice for same order should only charge the card once."""
    _processed.clear()
    charge_calls = []

    def mock_charge(oid, amt):
        charge_calls.append(oid)
        return {"order_id": oid, "amount": amt, "status": "charged"}

    original = pm._charge_card
    pm._charge_card = mock_charge
    try:
        pm.process_charge("order-002", 50.00)
        pm.process_charge("order-002", 50.00)  # duplicate call

        # This FAILS — no idempotency guard, card charged twice
        assert len(charge_calls) == 1, (
            f"Card charged {len(charge_calls)} times for same order — missing idempotency check!"
        )
    finally:
        pm._charge_card = original
