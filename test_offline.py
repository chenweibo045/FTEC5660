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
        "positive_lines": [round(r["subtotal_after_discounts_before_rounding"]
                                 + r["discount_total"], 2)],
    })


def fenced(text: str) -> str:
    """Wrap model JSON in markdown fences to test fence stripping."""
    return "```json\n" + text + "\n```"


class PerImageChain:
    """Mimics a LangChain Runnable with per-image canned sample queues.

    invoke() is called concurrently from hw1's worker pool, each call carrying
    the payload for a specific image; responses come from that image's own
    queue, which is exactly how the real chain behaves.
    """

    def __init__(self, samples_by_image: dict[str, list[str]]):
        import threading
        self._queues = {name: list(q) for name, q in samples_by_image.items()}
        self.calls = 0
        self._lock = threading.Lock()

    def invoke(self, payload):
        url = payload["image_url"]
        with self._lock:
            self.calls += 1
            # match payload to its image via the data-url encoding
            for name, queue in self._queues.items():
                if hw1.image_data_url(Path("public_test") / name) == url and queue:
                    return SimpleNamespace(content=queue.pop(0))
            raise AssertionError(f"no canned sample left for payload ({url[:40]}...)")

    def batch(self, payloads, config=None):
        return [self.invoke(p) for p in payloads]


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {extra}")
    return cond


images = sorted(Path("public_test").glob("receipt*.jpg"))
ok = True

# --- Test 1: two agreeing reads per receipt -> correct totals & format -------
# Round 1 reads sample 1 of each receipt; no consensus yet, so round 2 reads
# sample 2 of each. Two agreeing reads per receipt are enough.
chain = PerImageChain({p.name: [fenced(CANNED[p.name]), CANNED[p.name]] for p in images})
responses = hw1.answer_queries(chain, images)
ok &= check("Q1 response", responses[hw1.QUERY_1] == "HK$1974.30", repr(responses[hw1.QUERY_1]))
ok &= check("Q2 response", responses[hw1.QUERY_2] == "HK$2348.20", repr(responses[hw1.QUERY_2]))
ok &= check("two samples per receipt", chain.calls == 2 * len(images), f"calls={chain.calls}")

# grade with the template's own scorer
truth = hw1.read_ground_truth(Path("public_test"))
rows = hw1.write_results(responses, truth).read_text()
ok &= check("results.csv both correct", sum(1 for ln in rows.splitlines() if ln.rstrip("\r").endswith(",correct")) == 2)
print(rows)

# --- Test 2: garbage first read / inconsistent math get outvoted -------------
bad_math = json.dumps({
    "subtotal_after_discounts": 514.09, "discounts": [{"label": "X", "amount": 76.71}],
    "discount_total": 76.71, "rounding": -0.09, "amount_paid": 500.00,
    "positive_lines": [590.80]})
first_pass = [CANNED[p.name] for p in images]
first_pass[2] = "sorry, I cannot read this"   # receipt3: invalid JSON
first_pass[3] = bad_math                       # receipt4: fails paid identity
# Round 1: sample 1 (r3 garbage, r4 bad math -> no valid run). Round 2: sample
# 2 for all 7. Round 3: receipts 3 and 4 still lack 2 valid reads each and get
# a 3rd sample, after which two reads agree on every field.
samples2 = {p.name: [CANNED[p.name], CANNED[p.name]] for p in images}
samples2["receipt3.jpg"] = ["sorry, I cannot read this",
                            CANNED["receipt3.jpg"], CANNED["receipt3.jpg"]]
samples2["receipt4.jpg"] = [bad_math, CANNED["receipt4.jpg"], CANNED["receipt4.jpg"]]
chain2 = PerImageChain(samples2)
responses2 = hw1.answer_queries(chain2, images)
ok &= check("retry Q1", responses2[hw1.QUERY_1] == "HK$1974.30", repr(responses2[hw1.QUERY_1]))
ok &= check("retry Q2", responses2[hw1.QUERY_2] == "HK$2348.20", repr(responses2[hw1.QUERY_2]))
ok &= check("retry call count", chain2.calls == 2 * len(images) + 2, f"calls={chain2.calls}")

# --- Test 3: a wrong discount digit never enters the voting pool ------------
# true discount is 30.31; reading it as 29.31 breaks the receipt identity
# sum(positive_lines) - discount_total = subtotal (221.20 - 29.31 != 190.89)
wrong_digit = json.dumps({
    "subtotal_after_discounts": 190.89, "discounts": [{"label": "X", "amount": 29.31}],
    "discount_total": 29.31, "rounding": -0.09, "amount_paid": 190.80,
    "positive_lines": [221.20]})
first_pass3 = [CANNED[p.name] for p in images]
first_pass3[5] = wrong_digit
# receipt6's first read misreads the discount digit (29.31 vs 30.31), failing
# the receipt identity and never entering the voting pool. Round 2 reads a 2nd
# sample for all 7 receipts; round 3 reads a 3rd for receipt6 only, and the two
# correct reads agree.
samples3 = {p.name: [CANNED[p.name], CANNED[p.name]] for p in images}
samples3["receipt6.jpg"] = [wrong_digit, CANNED["receipt6.jpg"], CANNED["receipt6.jpg"]]
chain3 = PerImageChain(samples3)
responses3 = hw1.answer_queries(chain3, images)
ok &= check("outvote Q1", responses3[hw1.QUERY_1] == "HK$1974.30", repr(responses3[hw1.QUERY_1]))
ok &= check("outvote Q2", responses3[hw1.QUERY_2] == "HK$2348.20", repr(responses3[hw1.QUERY_2]))

# --- Test 4: template money regex accepts exactly our format -----------------
ok &= check("regex single amount", hw1.parse_single_amount("HK$1974.30") == Decimal("1974.30"))
ok &= check("regex rejects two amounts", hw1.parse_single_amount("HK$1974.30 and HK$100") is None)
ok &= check("regex rejects words", hw1.parse_single_amount("about 7 receipts totalling HK$1974.30") is None)

# --- Test 5: no-discount receipt (rounding only) ------------------------------
good55 = json.dumps({
    "subtotal_after_discounts": 55.55, "discounts": [], "discount_total": 0,
    "rounding": -0.01, "amount_paid": 55.54, "positive_lines": [55.55]})
chain5 = PerImageChain({"receipt5.jpg": [good55, good55]})
r5 = hw1.answer_queries(chain5, [Path("public_test/receipt5.jpg")])
ok &= check("no-discount receipt", r5[hw1.QUERY_1] == "HK$55.54" and r5[hw1.QUERY_2] == "HK$55.55",
            repr(r5))

print("\nALL PASS" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
