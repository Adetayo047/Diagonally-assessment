# Commercial loss ranking: which 100 addresses to inspect next

Senior AI/ML Engineer assessment, Diagonally Ltd.

| File | What it is |
|---|---|
| `score.py` | The scoring script. One command, writes `submission.csv` |
| `submission.csv` | All 2,000 customers ranked, in the required format |
| `analysis.ipynb` | Step-by-step analysis, with charts, behind every decision below |
| `requirements.txt` | Python dependencies |

## 1. How to run

Python 3.10+. The data files are not included; point `--data-dir` at the folder that holds them.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python score.py --data-dir /path/to/data
```

This writes `submission.csv` in about 2 seconds. Useful options:

```bash
python score.py --data-dir /path/to/data --evaluate          # also print precision@k on past inspections
python score.py --readings new_month.csv --customers customers.csv --out next_month.csv
```

To re-run the analysis notebook: `DATA_DIR=/path/to/data jupyter lab analysis.ipynb`. It defaults to the parent folder if `DATA_DIR` is not set.

Scoring needs only `meter_readings.csv` and `customers.csv`. `inspections.csv` is used only by `--evaluate`, and `feeders.csv` only by the notebook. A new month of readings is picked up automatically.

## 2. Approach (plain English)

Nobody can tell from billing data alone who is stealing. What we *can* see is meters behaving in ways honest meters don't. Studying the 100 confirmed theft cases showed that each type leaves its own fingerprint in the monthly bills:

- **Stuck meter:** the bill shows exactly the same number, to the decimal, month after month.
- **Estimated billing (under-billing):** bills are round numbers like 100 or 250 kWh, which means nobody is reading the meter.
- **Tampering:** usage flicks between normal and very low, on and off.
- **Bypass:** usage suddenly halves and becomes erratic.

The script checks every customer for these four patterns. Each customer's score is the strongest pattern found, weighted by how reliable that pattern proved on past inspections. The reason column names the pattern in plain words.

Two things deliberately **don't** raise a score:

- **Empty premises.** Months marked vacant are ignored. Otherwise, empty houses would fill the entire top 100.
- **Normal seasonal dips.** Usage naturally swings about 20% through the year, and that is removed first.

A steady, permanent drop scores low, because that is what people who genuinely cut back look like.

**Is it better than guessing?** On past inspections, using only what was known on the day of each visit, **97 of the top 100 were confirmed theft**, against 28–40 for random picks. The popular "biggest drop in usage" rule scored **0**: its top 100 was entirely empty premises.

## 3. Assumptions

- **Customers already confirmed as theft stay on the list** if their bills still show the pattern. Their usage never recovered after inspection, so the loss wasn't fixed. They are 47 of the top 100; if the utility only wants new addresses, filter them out.
- **`no_access` means "unknown", not "clean".** It is excluded from evaluation.
- **The `vacant` status is correct as recorded.**
- **Round-number bills count as commercial loss** (unmetered, estimated usage), even though the cause may be a billing-process failure rather than theft.
- **Six inspections are dated before the premises was connected.** These are impossible, so I excluded them from evaluation.
- **A drop only counts with at least 3 months of history on each side.**
- **Feeder losses are not used to score individuals.** Across the 12 feeders, feeder loss correlated only weakly with theft rate (0.19), and one number per 170 customers can't separate them.

## 4. Three biggest weaknesses

1. **The accuracy figure comes from the wrong population.** It is measured on 247 customers someone *chose* to inspect (40% were thieves) plus 113 vacant premises. It can't test the hardest case: customers whose usage genuinely fell, because none of them has been inspected since their drop. Precision on the uninspected 1,720 will be lower than 97%, and I don't know by how much.
2. **Quiet theft is under-ranked.** A bypass that takes a *steady* fraction looks exactly like a family using less. It scores 0.35 and falls outside the top 100; 195 uninspected customers look like that. Even noisy bypass drops rank 100–225, behind the near-certain stuck/round/on-off cases. A connection bypassed from day one shows no drop at all and scores zero.
3. **The thresholds are hand-set on one synthetic dataset.** Examples are "3 identical readings", "40% round bills" and "7 on/off switches". The ordering between patterns is a judgement call, and scores around rank 100 are nearly tied (about 0.87), so the exact cut-off is close to arbitrary. Tampering also needs several months to show; at inspection time, 3 of the 25 cases hadn't switched enough times to be caught.

## 5. Time spent

About **5 hours**.

## 6. AI tools used

- **Claude Code** (Anthropic): helped write and debug the pandas code in `score.py` and the analysis notebook, draft and edit the README and notebook text, and check the output format against the brief.
- **My own decisions:** which hypotheses to test, reading the charts to identify each loss type's fingerprint, choosing the thresholds and weights, and designing the evaluation (point-in-time features, vacant premises as hard negatives).
- I have reviewed every line and can explain or change any of it.


## Deployment, and knowing it still works in six months

**Deploy:** run `score.py` monthly as a scheduled job after billing closes. Write the ranked list to wherever inspection planning happens, and keep each month's list and features. The script is deterministic and has no trained model to manage.

**Close the loop:** every inspection result goes back into `inspections.csv`, tagged with the rank and pattern that sent the inspector there. This is the most important step.

**Monitor monthly:**

- **Hit rate** of inspections from the list, per pattern. This is the real measure. If one pattern's hit rate falls, re-tune that detector.
- **Share of customers each detector flags.** A sudden jump, for example in round-number bills, usually means a billing-system change, not a crime wave.
- **Feeder loss after fixes.** Confirmed cases on a feeder should lower its loss; if they don't, the list is missing something.
- **Input checks:** row counts, missing months, new status codes, and duplicate readings. The job should fail loudly rather than score bad data.

**Keep learning:** send about 10% of visits to random or lower-ranked addresses. That is the only way to measure what the list misses and to escape the selection bias described in weakness 1.
