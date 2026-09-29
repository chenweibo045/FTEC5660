#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    import os

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Add it to your .env file "
            "(see the setup instructions in README.md)."
        )

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",  # required backbone model
        temperature=0,                          # deterministic extraction
        timeout=180,
        max_retries=3,
    )

    system_prompt = (
        "You are a meticulous OCR accountant for Hong Kong supermarket receipts "
        "(e.g. PARKnSHOP / 百佳, Fusion, Wellcome). You read one receipt image and "
        "return ONLY a JSON object - no markdown fences, no commentary. All amounts "
        "are in HKD. Money amounts appear on the right-hand column of the receipt."
    )

    human_prompt = (
        "Read this supermarket receipt carefully and extract the following fields.\n"
        "- \"subtotal_after_discounts\": the SUBTOTAL / 小計 line - the amount AFTER all "
        "discount/promotion/coupon lines but BEFORE the ROUNDING line. Do not compute it "
        "from item lines; read the printed subtotal.\n"
        "- \"discounts\": list every discount/promotion/coupon/member/app-off line as an "
        "object {{\"label\": <line text>, \"amount\": <positive number>}}. Discount lines are "
        "printed as negative amounts (e.g. \"5% OFF (CU-SCO) -$5.39\"); report the amount "
        "as a POSITIVE number. If the receipt has no discount line, use an empty list. "
        "Do NOT include the ROUNDING line here.\n"
        "- \"discount_total\": sum of all discount amounts (0 if none).\n"
        "- \"rounding\": the ROUNDING line amount, including its sign (0 if absent).\n"
        "- \"amount_paid\": the FINAL amount the customer actually paid - the payment line "
        "such as OCTOPUS / 八達通 / CASH / 現金 / VISA / MASTER / CREDIT CARD / FPS / Alipay / "
        "WeChat Pay, printed AFTER the ROUNDING line. This is the last total on the receipt.\n"
        "Rules:\n"
        "1. IGNORE card numbers, member numbers, points, 積分, quantities (數量), the change "
        "line (找續), and the Octopus card REMAINING BALANCE line (餘額).\n"
        "2. amount_paid = subtotal_after_discounts + rounding (they must be consistent; "
        "rounding is at most a few cents).\n"
        "3. discount_total must be >= 0.\n"
        "4. Respond with the JSON object only, exactly in this schema:\n"
        "{{\"subtotal_after_discounts\": <number>, \"discounts\": [{{\"label\": <string>, "
        "\"amount\": <number>}}], \"discount_total\": <number>, \"rounding\": <number>, "
        "\"amount_paid\": <number>}}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                [
                    {"type": "text", "text": human_prompt},
                    {"type": "image_url", "image_url": {"url": "{image_url}"}},
                ],
            ),
        ]
    )
    return prompt | llm


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    import json
    import re
    from decimal import Decimal

    if chain is None:
        raise RuntimeError("build_chain() returned None - the chain was not created.")

    _JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
    _MAX_ATTEMPTS = 4

    def _to_decimal(value: Any) -> Decimal | None:
        """Coerce a model-extracted numeric field to Decimal, or None if unusable."""
        try:
            return Decimal(str(value).replace(",", "").replace("$", "").strip())
        except Exception:
            return None

    def _parse_extraction(text: str) -> dict[str, Decimal] | None:
        """Parse and validate one receipt's JSON extraction.

        Returns {'paid': ..., 'without_discount': ...} or None when invalid.
        Validation catches OCR mistakes early so they can be retried.
        """
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0), parse_float=Decimal, parse_int=Decimal)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        subtotal = _to_decimal(data.get("subtotal_after_discounts"))
        discounts = _to_decimal(data.get("discount_total"))
        rounding = _to_decimal(data.get("rounding"))
        paid = _to_decimal(data.get("amount_paid"))
        if None in (subtotal, discounts, rounding, paid):
            return None
        # Basic sanity checks for a supermarket receipt.
        if subtotal <= 0 or paid <= 0 or discounts < 0:
            return None
        if abs(rounding) > Decimal("0.10"):  # HK receipts round to the nearest 10c
            return None
        # paid must equal subtotal + rounding (within a cent of tolerance).
        if abs(paid - (subtotal + rounding)) > Decimal("0.011"):
            return None
        return {
            "paid": paid,
            "without_discount": subtotal + discounts,
        }

    def _extract_one(image: Path) -> dict[str, Decimal]:
        """Extract one receipt with retries; every attempt is re-read from scratch."""
        payload = {"image_url": image_data_url(image)}
        last_text = ""
        for attempt in range(_MAX_ATTEMPTS):
            response = chain.invoke(payload)
            last_text = response_text(response)
            parsed = _parse_extraction(last_text)
            if parsed is not None:
                return parsed
        raise RuntimeError(
            f"could not extract a valid receipt summary from {image.name}; "
            f"last model output: {last_text[:300]!r}"
        )

    # Stage 1: extract every receipt. All images are independent, so the first
    # pass runs in parallel with chain.batch(); only failed parses are retried.
    payloads = [{"image_url": image_data_url(image)} for image in images]
    first_pass = chain.batch(payloads, config={"max_concurrency": 4})
    extracted: list[dict[str, Decimal]] = []
    for image, response in zip(images, first_pass):
        parsed = _parse_extraction(response_text(response))
        if parsed is None:
            parsed = _extract_one(image)  # retry individually
        extracted.append(parsed)

    # Stage 2: deterministic aggregation with Decimal - the LLM only reads
    # numbers off the paper; all arithmetic happens here, so no summation drift.
    total_paid = sum((row["paid"] for row in extracted), Decimal("0"))
    total_without_discount = sum(
        (row["without_discount"] for row in extracted), Decimal("0")
    )

    def _format_hkd(amount: Decimal) -> str:
        return f"HK${amount.quantize(Decimal('0.01'))}"

    return {
        QUERY_1: _format_hkd(total_paid),
        QUERY_2: _format_hkd(total_without_discount),
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
