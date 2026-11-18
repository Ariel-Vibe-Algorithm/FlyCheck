# FlyCheck — Step-by-step build plan

Last updated: 2026-11-18. Read [CLAUDE.md](CLAUDE.md) first: it holds the product requirements, the verified data findings (section 4) and the ML rules this plan follows.

The data constraint shapes the whole plan. We use **only existing datasets**: no self-collected archive. The labelled delay data is small, about 761 completed departures in the open archive, and comes from an unusual 2026 period. So the model is a small, transparent and honestly evaluated "relative risk" model. Each step ends in something that runs.

## Timeline at a glance

| Phase | Dates | Milestone | Outcome |
|---|---|---|---|
| 0. Setup | Nov 18–19 | — | Repo, environment, data snapshots, open questions sent |
| 1. Data pipeline | Nov 19–22 | M2 | Clean flight table + tests |
| 2. Analytics dashboard | Nov 22–25 | **M2 (Nov 25)** | Streamlit Explore Data + About pages on real data |
| 3. Congestion + features | Nov 26–Dec 5 | M3 | Congestion profile/score, feature builders + tests |
| 4. Delay model | Dec 5–14 | **M3 (Dec 16)** | Trained, evaluated, saved pipeline; Analyze a Flight page |
| 5. Comparison + guide + extras | Dec 14–20 | Deployment | Compare page, Airport Guide, live-status card, weather, ILS conversion |
| 6. Deploy | Dec 20–23 | **Deploy (Dec 23)** | Public Streamlit Cloud app |
| 7. Polish + demo | Dec 23–Jan 6 | **Demo (Jan 6)** | Tests, limitations, 5-minute demo |

---

## Phase 0 — Setup (Nov 18–19)

1. `git init`, create the GitHub repository, add `.gitignore` (ignore `data/raw/`, `.venv/`, `__pycache__/`).
2. Create the structure from CLAUDE.md section 6 (`src/`, `tests/`, `data/raw/`, `data/processed/`, `models/`).
3. Create `requirements.txt` (pandas, numpy, scikit-learn, joblib, streamlit, plotly, pytest, holidays, pyarrow). Pin versions after the first working run.
4. Add `src/download_data.py` to fetch and record (URL, retrieval date, row count) these sources:
   - OData `flydata_log.csv` → `data/raw/` (CC BY).
   - TLVFlights airline-level CSV export (`/en/flight-statistics/export.csv`). Check its terms on the site first.
   - CBS price index series 121390 (API, JSON).
   - One snapshot of the current data.gov.il flight board. It adds recent schedule days for the congestion profile, and needs no ongoing collection.
5. Data requests, run in parallel with the build (see CLAUDE.md section 4, "Pending larger sources"):
   - ask Rank Air for an academic export;
   - complete the EUROCONTROL ADRR request.

   **Decision point 2026-12-01:** if a larger source has arrived, it replaces or extends the OData data behind the same internal flight schema. If not, continue with the small data, or switch to US BTS data with instructor approval.

   Every loader converts its source into one internal schema: scheduled time, actual time, airline, destination, direction, status. Model and UI code never depend on a specific source.
6. Non-code actions (team):
   - Tell the instructor about the data limitation and scope (small, 2026-only sample) and ask whether it is acceptable.
   - Ask the author of `giladct/flights-delays` for permission to use `tlv_flights.db` for academic evaluation. It has no license, so until the author answers we neither use nor commit it.
   - Confirm team roles and the official M2 rubric.

**Done when:** `python -m src.download_data` reproduces the raw files and prints a metadata summary.

## Phase 1 — Data pipeline (Nov 19–22)

`src/data_processing.py`, with each function small and tested:

1. `load_flydata_log(path)`: read the CSV robustly. Skip and count malformed or truncated lines; at least one exists.
2. `latest_snapshot_per_flight(df)`: one row per flight occurrence (airline + flight number + direction + scheduled time), keeping the last `log_timestamp`.
3. `dedupe_codeshares(df)`: collapse codeshares into physical flights (same scheduled time, destination, updated time and terminal). Keep the list of marketing carriers.
4. `build_departures(df)`:
   - Departures only (`CHAORD == "D"`).
   - Final status classes: `DEPARTED` = completed; `CANCELED` = cancelled; anything else = unresolved (excluded from the target, reported separately).
   - Compute `delay_minutes = CHPTOL - CHSTOL` and `delayed_30`.
   - Keep extreme or negative values, but flag them.
5. Timezone: treat `CHSTOL`/`CHPTOL` as Asia/Jerusalem local time. Spot-check this against the live board before relying on it.
6. Save `data/processed/departures.parquet` and `data/processed/schedule.parquet`. Schedule includes arrivals and departures, for congestion.
7. Tests in `tests/test_data_processing.py`:
   - duplicate snapshots collapse to one row;
   - codeshares are counted once;
   - cancelled and unresolved flights are excluded from the target;
   - delays are computed correctly across midnight;
   - a malformed line is skipped.

**Done when:** `python -m pytest` passes and a summary prints the rows per day, the status counts and the class balance.

## Phase 2 — M2 analytics dashboard (Nov 22–25)

`app.py` as a multipage Streamlit app. Pages for later phases show a clear "coming in M3" message and never fake numbers.

1. **Explore Data**
   - Filters: airline, destination, weekday, hour.
   - Delay rate and delay distribution, always with sample sizes (n).
   - Hourly departures/arrivals and a weekday × hour heatmap.
   - Airline reliability from the TLVFlights aggregates, with source link and coverage dates.
   - CBS monthly travel-price index chart, labelled "context, not a fare forecast".
2. **About & Data**
   - The sources and licenses.
   - Exact coverage days.
   - The airport closure (Feb 11–May 15, 2026).
   - The definition of "actual" time (likely near takeoff, not gate).
   - Known limitations.
3. Use `st.cache_data` for the data loads.

**Done when:** the app runs locally on real processed data, and every chart shows its n and its source.

## Phase 3 — Congestion + feature builders (Nov 26–Dec 5)

`src/congestion.py`:

1. Build an hourly series of scheduled departures from unique physical flights, over covered days only.
2. Window count for an airport arrival time `t`: the number of departures scheduled in `[t+1h, t+4h]`. Handle midnight crossing.
3. Profiles by weekday × hour, with a fallback to the hour-only profile when samples are sparse. Return the fallback level and n.
4. Score 0–100 = the percentile within a fixed training reference distribution. Low / medium / high at the 33rd and 66th percentiles, with cutoffs taken from training data only.
5. Save `models/congestion_profile.parquet` plus metadata.

`src/features.py`. These are pre-booking features only; see CLAUDE.md section 5.

- Scheduled hour, weekday, Israeli holiday / holiday-eve flag (`holidays` package).
- Airline, and airline type (Israeli/foreign, low-cost/full-service), from a manual mapping table in `data/reference/`.
- Destination region and distance, from OurAirports coordinates.
- Historical congestion feature, from the frozen training-period profile.
- Optional: scheduled turnaround gap (the matching inbound flight's scheduled arrival). Only if time allows.

Tests:

- Midnight windows.
- Unknown airline or destination.
- Holiday flag on known dates.
- No feature uses `CHPTOL`, the final status or future records.

## Phase 4 — Delay model, M3 (Dec 5–14)

`src/train.py`:

1. Split by date, never by random row. The completed departures per day are:

   | Jan 16 | Mar 16 | Mar 17 | Jun 25 | Jun 26 | Jul 1 | Jul 2 | Jul 3 |
   |---|---|---|---|---|---|---|---|
   | 11 | 8 | 30 | 239 | 12 | 2 | 259 | 200 |

   **Proposed split:** train on everything up to and including Jul 2 (~561 departures), test on Jul 3 (~200). Choose models with cross-validation grouped by day inside the training set.
2. Models, all inside a `Pipeline` + `ColumnTransformer` with `handle_unknown="ignore"`:
   - (a) a per-airline base-rate baseline;
   - (b) `DummyClassifier`;
   - (c) `LogisticRegression`;
   - (d) `RandomForestClassifier`, with shallow trees because the sample is small.
3. Feature ablation: start with airline + hour + weekday + region, then add one feature at a time. Keep a feature only if it improves the held-out result. Record the ablation table; it is demo material.
4. Metrics:
   - class balance;
   - confusion matrix;
   - precision, recall, F1;
   - ROC-AUC;
   - Brier score or a reliability plot.
5. Save `models/delay_pipeline.joblib` and `models/model_metadata.json` (training dates, n, metrics, feature list).
6. If permission arrives, use the giladct data only as a separate, clearly labelled external check. Its "actual" time definition differs (AeroDataBox), so it is never merged into training.

`src/predict.py`, and the **Analyze a Flight** page:

- **Input:** airline, destination, date, time, and an arrival lead time (default 3 h).
- **Output:**
  - delay probability, with a "relative risk from limited 2026 data" label;
  - congestion level and score;
  - the daily profile, highlighting the arrival and departure times;
  - historical n and the fallback level;
  - "better alternatives": the same destination at other hours or days, ranked by risk and congestion.
- Predictions for unsupported or sparse inputs are suppressed or qualified.

**Done when:** a fresh clone runs `python -m src.train`, then `python -m pytest`, then `streamlit run app.py`, and gets real predictions.

## Phase 5 — Comparison, guide and extras (Dec 14–20)

In priority order. Stop wherever time runs out.

1. **Compare Flights**:
   - add or remove options;
   - a sortable table of risk, congestion and an optional user-entered price;
   - prices converted to ILS with Bank of Israel rates, showing the rate date.

   It never declares a "best" flight.
2. **Airport Guide** (static content, each item with an IAA link and retrieval date):
   - arrive 3 h before (the official guidance);
   - terminals;
   - train and bus links;
   - the parking price table;
   - carry-on rules;
   - lounges and accessibility.
3. **Live status card** on Analyze a Flight, shown only when the flight is on the current data.gov.il board:
   - shows the terminal, check-in counters and zone;
   - is labelled "current status — not a forecast";
   - is cached, with a graceful fallback.
4. **Weather** at TLV and the destination (Open-Meteo, ≤16 days ahead), for display only and **not** a model feature. Attribution: CC BY 4.0.

## Phase 6 — Deploy (Dec 20–23)

1. Pin `requirements.txt`, then test a fresh venv install.
2. Commit only redistributable, compact artifacts: processed OData data (CC BY, with attribution) and the model files. Never commit raw files without a license.
3. Deploy to Streamlit Community Cloud, test the public link, and add the link to the README.

## Phase 7 — Polish and demo (Dec 23–Jan 6)

1. End-to-end tests:
   - unknown categories;
   - midnight and date boundaries;
   - cancellations and duplicate snapshots;
   - no future information in features.
2. README: problem, data sources and licenses, how to run, results, limitations.
3. Five-minute demo:
   - the problem;
   - the data and its honest limitations;
   - the dashboard;
   - a live prediction;
   - the comparison;
   - the evaluation and ablation table;
   - the limitations.

## Explicitly out of scope

- Fare prediction or "best time to buy".
- Security queue / wait times.
- A cancellation model (too few cases; show historical rates only).
- Weather as a model feature.
- A self-collected archive.
- A paid API.
- An LLM agent.
