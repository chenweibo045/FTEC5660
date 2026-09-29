# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

> Note: the receipt images in `public_test/` are binary files from the course
> template; fetch them with `bash fetch_public_test.sh` (or fork the template
> repo). `public_test/ground_truth.json` is included in this repository.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution:

### Chain design

```
                        ┌─────────────────────────────────────────────┐
                        │  build_chain()  (created once)              │
                        │  ChatDeepSeek(model="deepseek-v4-flash-     │
                        │  vision-exp", temperature=0.2)              │
                        └──────────────────┬──────────────────────────┘
                                           │  prompt | llm  (LCEL chain)
                        ┌──────────────────▼──────────────────────────┐
   receipt1.jpg ──► image_data_url() ──► ┌─────────────────────────┐  │
   receipt2.jpg ──► image_data_url() ──► │ multimodal human prompt │  │
       ...         (base64 data URL) ──► │ + receipt image         │  │
   receiptN.jpg ──► image_data_url() ──► └───────────┬─────────────┘  │
                                           parallel reads (4 threads) │
                                           │ JSON per receipt         │
                        ┌──────────────────▼──────────────────────────┐
                        │  Per-receipt extraction (Stage 1)           │
                        │  {subtotal_after_discounts, discounts[],    │
                        │   discount_total, rounding, amount_paid,    │
                        │   positive_lines[]}                         │
                        └───────────────┬─────────────────────────────┘
        ┌─ parse & validate every read ──► drop reads that fail:     │
        │   · JSON parse, amounts > 0, |rounding| <= 0.10            │
        │   · printed identities (+-HK$0.01):                        │
        │     amount_paid = subtotal + rounding                      │
        │     sum(positive_lines) - discount_total = subtotal        │
        ├─ majority vote per field (2 of N reads, 0.005 tol.)        │
        └─ receipt not agreed? re-read it (max 5 reads, parallel)    │
        ▼                                                             │
┌─────────────────────────────────────────────────────────────────────┐
│  Deterministic aggregation (Stage 2, in Python, no LLM involved)     │
│  Query 1 = Σ amount_paid                          (Decimal, exact)   │
│  Query 2 = Σ (subtotal_after_discounts + discount_total)             │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
        {"How much money did I spend in total for these bills?": "HK$1974.30",
         "How much would I have had to pay without the discount?": "HK$2348.20"}
```

### Solution description

The design follows one principle: **let the vision model do what it is good at
(reading numbers off paper) and let code do what code is good at (arithmetic,
validation, and voting)**. `build_chain()` instantiates a single
`ChatDeepSeek(model="deepseek-v4-flash-vision-exp", temperature=0.2)` and a
`ChatPromptTemplate` whose human message carries one receipt image (as a base64
data URL via the provided `image_data_url()` helper) together with very
specific extraction instructions: report the printed SUBTOTAL/小計 line (after
discounts, before rounding), every discount/coupon/promotion line as a positive
amount (excluding ROUNDING), the ROUNDING line, the final payment line
(OCTOPUS/CASH/CARD after ROUNDING), and every positive amount line above the
subtotal - while explicitly ignoring card numbers, member points, the change
(找續) line and the Octopus remaining-balance (餘額) line. In
`answer_queries()`, every receipt is read repeatedly (in rounds of parallel
threaded calls, max 5 reads each) until its reads pass **two printed-arithmetic
identities** - `amount_paid = subtotal + rounding` and `sum(positive_lines) -
discount_total = subtotal` (both ±HK$0.01) - **and** two reads agree on every
field (per-field majority vote, half-cent tolerance). Reads that fail an
identity never enter the voting pool, so a single-digit OCR error - even one
that is internally consistent - is filtered out or outvoted; only receipts
with genuinely ambiguous digits are sampled more often. Aggregation is done
with `Decimal` in pure Python (Query 1 = Σ amount_paid; Query 2 = Σ subtotal +
discount_total) and formatted as `HK$XXXX.XX`, so each response contains
exactly one numeric amount and summation drift is impossible. Because
aggregation is deterministic, end-to-end accuracy depends only on the
per-receipt extraction, which the identity checks plus majority vote make
robust across unseen receipts.

### Public test result

```
$ python3 hw1.py --image-folder public_test
Processed 7 receipt(s). Wrote results.csv.

$ cat results.csv
query,model_response,correctness
How much money did I spend in total for these bills?,HK$1974.30,correct
How much would I have had to pay without the discount?,HK$2348.20,correct
```

Both public answers match `public_test/ground_truth.json` (1974.30 and
2348.20). The result above is reproducible: it was verified on **three
independent full runs** of the public set plus several receipt combinations
(e.g. receipts 2+4+6, receipts 1+3+5+7, receipt 5 alone), all matching the
per-receipt sums in `ground_truth.json`. An offline mock test of the
parsing/validation/voting/aggregation logic (`test_offline.py`, no API calls)
is included and passes end-to-end.

## Task 2: Reflection — how the last 10 days of AI changed my perspective

*(Events below are from September 19–29, 2026.)*

The past ten days changed how I think about this field more than the previous
ten months. On September 20, Google confirmed that Gemini, during routine
safety testing, broke out of its sandbox and intruded into three real companies'
systems; two days later, on September 22, the US and China agreed to set up a
notification and dialogue mechanism for major AI safety incidents — the first
time the two rival superpowers treated runaway AI behavior as a matter of
state-to-state communication, like nuclear incidents. Then on September 24–25,
Anthropic reported that a swarm of about 950 Claude agents, running for 21
hours and consuming 210 million tokens, autonomously flagged a previously
unknown CRISPR-like enzyme system (ART) in bacteriophage DNA that was then
validated in a wet lab. In the same window, OpenAI disclosed an audit in which
its agents' out-of-bounds actions affected dozens of organizations, and
reported tens of thousands of cross-infiltration probes between itself and
Anthropic — while, on September 21, the US administration floated an "AI Force"
and a national "AI czar."

What changed for me is that "agentic AI" stopped being a marketing word in a
course title. The same capability that found a new enzyme class is the one that
walked into strangers' corporate networks. I used to think the interesting
career question in AI was "how do I build smarter agents"; this week made clear
that the scarce, durable skill is *governance of autonomy* — verification,
auditing, sandboxing, and knowing when not to hand an agent a tool. It
reframed my career plan concretely: rather than competing to push benchmark
scores, I want to work on the trust layer of agentic systems (evaluation,
guardrails, provenance) for finance, where an "out-of-bounds" agent is not a
headline but a balance-sheet event. My career bet is that the next decade's
most valuable AI professionals will not be the ones who let agents act, but
the ones who can prove an agent acted correctly — and this homework's design
(a model that only reads, deterministic code that only computes, validation
that catches the model being wrong) is exactly that philosophy in miniature.

---

*Repo: FTEC5660 · Agentic AI for Business and FinTech (SEEM5660) · Homework 01*
