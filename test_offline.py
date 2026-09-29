#!/usr/bin/env python3
"""Offline verification of hw1.py logic using a mock chain (no API calls)."""
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import hw1

GT = json.loads(Path("public_test/ground_truth.json").read_text())

# Canned per-receipt JSON built from ground truth (simulates a perfect model).
CANNED = {}
for name, r in GT["receipts"].items():
    CANNED[name] = json.dumps({
        "subtotal_after_discounts": r["subtotal_after_discounts_before_rounding"],
        "discounts": [{"label": "X% OFF", "amount": r["discount_total"]}] if r["discount_total"] else [],
        "discount_total": r["discount_total"],
        "rounding": round(r["amount_paid_after_rounding"] - r["subtotal_after_discounts_before_rounding"], 2),
        "amount_paid": r["amount_paid_after_rounding"],
    })


class QueueChain:
    """Mimics a LangChain Runnable, returning queued canned text outputs."""

    def __init__(self, outputs):
        self.outputs = list(outputs)  # list of str (model text) to pop in order
        self.calls = 0

    def _pop(self):
        self.calls += 1
        if not self.outputs:
            raise AssertionError("mock chain ran out of canned outputs")
        return SimpleNamespace(content=self.outputs.pop(0))

    def batch(self, payloads, config=None):
        return [self._pop() for _ in payloads]

    def invoke(self, payload):
        return self._pop()


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {extra}")
    return cond


ok = True

# --- Test 1: perfect extraction -> correct totals, correct output format ---
images = sorted(Path("public_test").glob("receipt*.jpg"))
outputs = ["```json\n" + CANNED[p.name] + "\n```" for p in images]  # also test fence stripping
chain = QueueChain(outputs)
responses = hw1.answer_queries(chain, images)
ok &= check("Q1 response", responses[hw1.QUERY_1] == "HK$1974.30", repr(responses[hw1.QUERY_1]))
ok &= check("Q2 response", responses[hw1.QUERY_2] == "HK$2348.20", repr(responses[hw1.QUERY_2]))
ok &= check("batch called once per image", chain.calls == len(images), f"calls={chain.calls}")

# grade with the template's own scorer
truth = hw1.read_ground_truth(Path("public_test"))
out = hw1.write_results(responses, truth)
rows = out.read_text()
ok &= check("results.csv both correct", sum(1 for ln in rows.splitlines() if ln.rstrip("\r").endswith(",correct")) == 2)
print(rows)

# --- Test 2: one garbage + one inconsistent answer, then valid on retry ---
mixed = []
for p in images:
    if p.name == "receipt3.jpg":
        mixed.append("sorry, I cannot read this")            # invalid JSON
    elif p.name == "receipt4.jpg":
        mixed.append(json.dumps({                             # inconsistent math
            "subtotal_after_discounts": 514.09, "discount_total": 76.71,
            "rounding": -0.09, "amount_paid": 500.00}))
    else:
        mixed.append(CANNED[p.name])
# retry queue: fixes for receipt3 & receipt4 (invoked individually on failure)
retry_fixes = [CANNED["receipt3.jpg"], CANNED["receipt4.jpg"]]
chain2 = QueueChain(mixed + retry_fixes)
responses2 = hw1.answer_queries(chain2, images)
ok &= check("retry Q1", responses2[hw1.QUERY_1] == "HK$1974.30", repr(responses2[hw1.QUERY_1]))
ok &= check("retry Q2", responses2[hw1.QUERY_2] == "HK$2348.20", repr(responses2[hw1.QUERY_2]))
ok &= check("retries used", chain2.calls == len(images) + 2, f"calls={chain2.calls}")

# --- Test 3: template money regex accepts exactly our format ---
ok &= check("regex single amount", hw1.parse_single_amount("HK$1974.30") == Decimal("1974.30"))
ok &= check("regex rejects two amounts", hw1.parse_single_amount("HK$1974.30 and HK$100") is None)
ok &= check("regex rejects words", hw1.parse_single_amount("about 7 receipts totalling HK$1974.30") is None)

# --- Test 4: no-discount receipt (rounding only) ---
chain3 = QueueChain([json.dumps({
    "subtotal_after_discounts": 55.55, "discounts": [], "discount_total": 0,
    "rounding": -0.01, "amount_paid": 55.54})])
r3 = hw1.answer_queries(chain3, [Path("public_test/receipt5.jpg")])
ok &= check("no-discount receipt", r3[hw1.QUERY_1] == "HK$55.54" and r3[hw1.QUERY_2] == "HK$55.55",
            repr(r3))

print("\nALL PASS" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
