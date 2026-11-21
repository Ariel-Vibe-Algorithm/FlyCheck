# FlyCheck — Project Context

## 1. Project overview

**Name:** FlyCheck
**Tagline:** Know before you book.
**Type:** Academic MVP — Vibe Coding with Claude, Ariel University, Fall 2026.
**Scope:** Departing international flights from Ben Gurion Airport (TLV), Israel.
**Goal:** Help travelers evaluate flights before purchasing a ticket by estimating delay risk and expected airport congestion around their arrival at the airport.

Travelers can already compare fares, airlines and departure times, but these do not necessarily tell them how reliable a particular flight is or how busy the airport is likely to be. FlyCheck adds historical-data-driven decision support. It is not a booking engine or live flight tracker.

**Project statement:** Travelers departing from TLV lack personalized information about flight delay risk and airport congestion before booking. FlyCheck uses historical flight records, machine learning and interactive visualizations to help them compare options.

## 2. Product requirements

### Target user and user stories

- As a traveler considering a flight, I want to enter its published details and estimate the risk of a delay.
- As a traveler, I want to see how busy the airport is expected to be when I arrive, not just when my flight departs.
- As a traveler comparing flights, I want to compare delay risk, airport congestion and optionally price side by side.
- As a curious user, I want to explore historical patterns by airline, destination, weekday and hour.

### Inputs

Required: airline, destination airport, departure date and scheduled local departure time. Origin is fixed to TLV. Optional: flight number, terminal (if known), ticket price and intended airport arrival lead time (default: 3 hours, configurable). Do not require aircraft type or passenger count. Use airport codes to disambiguate destinations and validate input against supported categories.

### Outputs

1. Estimated probability of a delay of at least 30 minutes, based on historical patterns.
2. Estimated delay in minutes (optional regression extension; do not display until implemented and validated).
3. Estimated airport congestion around the intended airport arrival time, expressed as low / medium / high and optionally a relative score from 0–100.
4. Historical congestion profile across the relevant day, highlighting the airport arrival time and departure time.
5. Historical statistics and sample sizes for the selected route/airline when available.
6. Side-by-side comparison of manually entered flight options, including optional prices in a common currency. Avoid inventing fares or declaring a universal best flight; let users sort by their priorities.

Clearly distinguish a statistical forecast from current flight status. Display dataset coverage dates, limitations and unsupported or sparse routes. No guaranteed predictions or queue-time claims. Do not recommend arriving later than official airline/airport guidance.

## 3. Course requirements

Based on the two supplied course presentations (meetings 1 and 2):

- Real problem, identifiable user, accessible data, simple ML component and Streamlit interface connected to the team's domain.
- M1 (week 3, Nov 4): project definition, GitHub repository, README.md, CLAUDE.md, working app.py skeleton, dataset link and column descriptions, roles, UI sketch, and 3–4 slide / 3-minute pitch. M1 grading: README 40%, team/roles 15%, repository/app 30%, pitch 15%. The detailed M1 instructions on the course website are authoritative and must be checked separately.
- M2 (week 6, Nov 25): data loading, cleaning and interactive analytics dashboard.
- M3 (week 9, Dec 16): train, evaluate, save and integrate an ML model for inference in the app.
- Deployment (week 10, Dec 23): Streamlit Community Cloud.
- Final demo (week 12, Jan 6 2027): five-minute team presentation, defense/questions and peer rating.
- Overall grade: final project 50%, milestones 30% (10% each), demo 20%. Attendance, all milestones and demo participation are passing conditions. Team roles: product, programmer, data, UI/UX. Team members and assignments are not yet specified.
- An AI agent is a bonus, not a required milestone. Prefer a reliable, explainable MVP over additional complexity.

## 4. Data sources: audit results (2026-11-18)

**Team decision:** use only existing datasets. No self-collected archive.

The openly available labelled data is small and comes from an unusual period. Larger sources are being requested (see "Pending larger sources" below), with a **decision point on 2026-12-01**. If no larger source has arrived by then, build the delay model on the small data as a transparent "relative risk" model, or move to US BTS data (needs instructor approval). Tell the instructor about this scope limitation.

### Pending larger sources

- **Rank Air** ([site](https://www.rank-air.com/), [code](https://github.com/Daniel1Maymon/israel-flights-etl)). It has polled the official data.gov.il board every 15 minutes, probably since August 2025, storing one row per flight with actual time and delay. That is the same source and time definition as the OData log. Its public API deliberately blocks bulk dumps, so **do not scrape it**; ask the developer for an academic export.
- **EUROCONTROL Aviation Data Repository for Research (ADRR)** ([page](https://www.eurocontrol.int/dashboard/aviation-data-research)).
  - **Coverage:** commercial flights "in and over Europe", 2015–2025, with filed and actual off-block times. About 76% of TLV departures go to Europe; LLBG inclusion is not yet verified.
  - **Access:** free, research only, via OneSky with an academic email. The student is registering.
  - **Terms (read 2026-11-18):** no sharing or public display of the raw data; credit EUROCONTROL with the Article 6 legend. Whether a public app may show aggregates and predictions is being asked in the request.
  - **Caveat:** filed off-block time may be updated by delay messages, which would understate delay. Verify once the data arrives.
- **Other options:**
  - TLVFlights full data (request).
  - IAA freedom-of-information request (30+ days).
  - AeroDataBox (paid; its terms forbid permanent storage).
  - FlightAware (minimum $100/month).
  - Wayback Machine captures (checked; too sparse).
- **US fallback:**
  - BTS On-Time Performance (US domestic only, millions of flights, delay causes).
  - TSA hourly checkpoint throughput PDFs (not verified).
  - DB1B fares.

### Datasets

| Source | Status | What it gives | License / terms | Use |
|---|---|---|---|---|
| OData flydata log ([page](https://www.odata.org.il/he/dataset/flydata), [CSV](https://www.odata.org.il/dataset/98b4f544-a7c8-4113-87de-7da1861c3a74/resource/d5ee74ca-2f56-4734-933e-02f4934420c2/download/flydata_log.csv)) | Downloaded and audited | Board snapshots logged on only 4 days (2026-01-17, 03-17, 06-26, 07-03). Of 2,844 departure records (about 2,030 physical flights after codeshare dedup), the final status is 761 DEPARTED, 25 CANCELED and 1,238 unresolved. | CC BY. The page calls it an "experiment"; it stopped updating 2026-07-04 | **Primary training and analytics data** |
| data.gov.il live flight board, resource `e83f763b-b7d7-479e-b172-ae981ddc6de5`, via `https://data.gov.il/api/3/action/datastore_search?resource_id=...` | API verified working | Current board only: about 24 h back to 3–4 days ahead, about 1,800 rows (arrivals and departures), same schema | "Other (Open)" | A one-time snapshot for recent schedule density; an optional live-status card. **Not a historical archive** |
| [giladct/flights-delays](https://github.com/giladct/flights-delays) `tlv_flights.db` | Downloaded and profiled | 4,827 departures scheduled 2026-04-30 to 06-04; 2,345 have actual_out and delay_minutes. Comes from AeroDataBox; actual time is probably the gate time | **No license** | Do not use or commit unless the author grants permission. If permission comes, use it only as a separate external check, never merged into training (its time definition differs) |
| [TLVFlights statistics](https://www.tlvflights.com/en/flight-statistics/?range=all) | Researched, not downloaded | About 101k flights (2025-12-10 to 2026-11-17, excluding 2026-02-11 to 05-15). The CSV export is **airline-level aggregates only** | Site says free to use with a link | Airline reliability context in the dashboard; sanity check. Not for training |
| CBS price index series 121390 "Expenditures for travel abroad and local flights" (`https://api.cbs.gov.il/index/data/price?id=121390&format=json&lang=en`) | API verified | Monthly national index since 1983 (base: 2024 average = 100) | Public | Seasonal fare **context only**, not a fare prediction |
| Busy TLV, Fly TLV | Reference only | Estimated congestion. Fly TLV counts departures in the next 3 h; no measured queue times | — | Competitor reference |

**Not available:**

- Per-flight fare history. Amadeus Self-Service shut down on 2026-07-17; Kiwi and Skyscanner are partner-only.
- Security queue or wait-time data.
- Hourly passenger counts.
- Any large per-flight TLV archive on Kaggle or Hugging Face.

### Verified flydata schema and quirks

Columns: `_id, CHOPER, CHFLTN, CHOPERD, CHSTOL, CHPTOL, CHAORD, CHLOC1, CHLOC1D, CHLOC1TH, CHLOC1T, CHLOC1CH, CHLOCCT, CHTERM, CHCINT, CHCKZN, CHRMINE, CHRMINH, log_timestamp, log_action`.

- `CHAORD` is A (arrival) or D (departure).
- `CHRMINE` statuses: ON TIME, NOT FINAL, LANDED, DEPARTED, FINAL, CANCELED, LANDING, DELAYED, EARLY.
- `CHTERM` is 1 or 3. `CHCINT` (check-in counters) and `CHCKZN` (check-in zone) are "null" in 57% of rows.
- The log holds about 225.5k rows, each a repeated snapshot (up to 86 per flight). It has at least one truncated, malformed line, so parse robustly.
- Times are ISO strings without a timezone and appear to be Asia/Jerusalem local time. Spot-check this before relying on it.

Context:

- **Airport closure:** TLV was closed 2026-02-11 to 05-15 (wartime). All available data is from around this abnormal period, so seasonality cannot be learned.
- **What "actual" means:** per TLVFlights, the IAA time is close to takeoff, not gate departure. State this as a limitation.
- **Class balance:** in the OData sample, 56% of completed departures are delayed by 30 minutes or more (median delay 33 min). giladct shows 25%, likely because it uses a different time definition. Do not mix the two.

### Data cleaning and feasibility gate

- Download one reproducible snapshot and record source URL, retrieval date, license, date range and row counts.
- Inspect multiple snapshots of the same flight; identify unique flight occurrences using verified identifiers, direction, airport and service date. Handle codeshares without counting the same physical flight multiple times where possible.
- Preserve a defensible scheduled time and a verified final actual time for each completed flight. Distinguish schedule changes from delays. Exclude cancellations, diversions and unresolved flights from the delay target; report separately.
- Calculate delay_minutes = actual_time - scheduled_time in minutes; delayed_30 = delay_minutes >= 30. Negative values can represent early departures; inspect extreme values rather than blindly deleting them.
- If actual departure cannot be verified, stop and document the blocker. Consider a different historical dataset or an explicitly renamed estimate-revision target with instructor approval; never present estimated times as actual observations.
- Check missingness, duplicate rates, time coverage, airline/route sample sizes and class balance. Verify coverage of arrivals and departures for congestion analysis, including zero-flight hours only where collection coverage is demonstrably complete.

## 5. ML design

### Delay prediction (required)

Use a binary classifier for delayed_30. Features must be available before purchase or derived from historical patterns.

**Candidate features**, each kept only if a chronological ablation shows it helps:

- airline;
- airline type (Israeli/foreign, low-cost/full-service), from a manual mapping;
- destination region and distance (OurAirports);
- scheduled hour and weekday;
- Israeli holiday / holiday-eve flag (Python `holidays` package);
- historical congestion estimate;
- optional terminal;
- optional scheduled turnaround gap.

Month adds nothing with the current data, since the few covered days give no seasonal variation.

**Excluded features:**

- actual time, final status, realized flight counts, future records;
- realized weather (not known at booking). Weather is display-only;
- security events.

**Not modelled:**

- **Cancellations:** too rare (about 1–2%). Show historical rates only.
- **Fares and queue times:** no data.

**Data size:** with about 761 labelled departures, keep the model small (few features, shallow trees). Present predictions as relative risk from limited 2026 data.

**Prediction horizon:** the historical model gives the same answer for any future date and has no live information. A real schedule exists only about 3–4 days ahead, on the live board; it may be displayed separately but must not be fed into the model.

Start with a DummyClassifier or historical base-rate baseline and LogisticRegression; compare with RandomForestClassifier. Use a scikit-learn Pipeline with ColumnTransformer, categorical encoding (handle unknown categories) and numeric missing-value handling. Save the entire fitted pipeline, not just the classifier, with joblib; only load trusted locally generated artifacts.

Use chronological train/validation/test splits, with the same flight occurrence confined to one split. Historical aggregates for training must be computed from earlier observations only (or a dedicated earlier reference period). Fit preprocessing on training data only. For test and live-style inference, use frozen historical features computed from data available before the prediction cutoff. Consider booking lead-time limitations if the archive provides snapshot timestamps.

Report class balance, confusion matrix, precision, recall, F1, ROC-AUC when both classes exist, and probability calibration (Brier score or reliability plot). Compare the base model with a congestion-feature model using the same held-out period. Do not claim improvement without results. Suppress or qualify predictions for unsupported inputs. Optional delay-duration regression can use MAE and a median baseline; define whether the prediction is unconditional or conditional on delay.

### Airport congestion (required product output, ML optional)

MVP: use historical scheduled departures as a proxy for departing passenger activity, and optionally count arrivals separately as context for airside activity. This is NOT a measurement of actual terminal occupancy, security lines or wait times.

- Build a complete hourly time series over verified coverage periods from unique physical flight occurrences.
- For an airport arrival time t, estimate the historical number of departures scheduled in a forward window around t (e.g. t+1 to t+4 hours), reflecting the fact that passengers arrive ahead of departure. Choose and document this assumption; compute across midnight correctly.
- Group historical window counts by month, weekday and hour; use fallback weekday/hour or overall profiles when samples are sparse. Show sample size and fallback level.
- Define a 0–100 score as a percentile of historical window counts in a fixed training reference distribution, with low/medium/high categories (e.g. below 33rd percentile, 33rd–66th, above 66th). This is a relative activity index, not percent capacity. Compute cutoffs from training data only.
- Default arrival time = scheduled departure minus 3 hours. Show relevant time zone and date. Display a daily hourly forecast profile and highlight both times.
- Reuse historically derived congestion features in the delay model, with the same leakage-safe feature generation during training and inference. A future flight's actual schedule density is unknown without live data.
- Optional second model: RandomForestRegressor predicting hourly flight counts from calendar features, benchmarked against historical averages with chronological holdout and MAE.
- Optional passenger estimate: aircraft seating capacity and documented load factor assumptions, only if credible data is available. Label as modeled passenger flow, not measured occupancy. Not required for MVP.

### Model updates

Train offline once on a fixed dataset and save artifacts. Streamlit loads the saved pipeline and congestion profile and performs inference only. No periodic scraping, continuous training, paid API, background worker or real-time integration is required. Clearly communicate that accuracy may degrade as seasonal patterns, airlines and regional conditions change. Manual retraining can be a future feature.

## 6. UI and architecture

Suggested Streamlit pages/tabs:

- **Analyze a Flight:** input form, delay probability, congestion level, historical context and daily congestion chart.
- **Compare Flights:** manually add/remove options and compare risk, congestion and optional prices; sortable table.
- **Explore Data:** filters by airline, route, period and weekday; historical delay rates, delay distribution, airline/route sample sizes, hourly departures/arrivals and weekday/hour heatmap.
- **Airport Guide (static, cosmetic):** official arrival guidance (3 h before international flights), terminals, train/bus links, parking prices, carry-on rules, lounges and accessibility. Each item links to IAA and shows a retrieval date.
- **About & Model:** methodology, data attribution, coverage, evaluation metrics, limitations and training date.

Optional extras, each clearly separated from the forecast:

- a live-status card from the data.gov.il board, with terminal and check-in counters, labelled "current status — not a forecast";
- Open-Meteo weather at TLV and the destination, up to 16 days ahead, display only, CC BY 4.0 attribution;
- conversion of user-entered prices to ILS with Bank of Israel rates.

Use a clean aviation-themed interface, readable labels and consistent units. English is the proposed implementation language; Hebrew localization is optional. Use numeric and text labels in addition to color. Provide clear empty/error states. Never show fake predictions in a production screen; any prototype sample data must be explicitly marked as illustrative.

Stack: Python, Pandas, NumPy, scikit-learn, joblib, Streamlit, Plotly, pytest, Git/GitHub, Streamlit Community Cloud. A local CSV/Parquet snapshot and compact model artifacts are sufficient. No database, accounts, React, custom server or LLM API needed. Pin compatible dependencies after testing. Cache read-only model/data resources appropriately.

Proposed repository (adapt to instructor's provided template):

```text
flycheck/
├── CLAUDE.md
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 # ignored; documented download instructions
│   └── processed/           # compact permitted artifacts or rebuild instructions
├── models/                  # fitted pipeline, metadata, congestion profiles
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── features.py
│   ├── train.py
│   ├── congestion.py
│   └── predict.py
└── tests/
    ├── test_data_processing.py
    ├── test_features.py
    └── test_predictions.py
```

Example commands (after implementation):

```bash
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -r requirements.txt
python -m src.train
python -m pytest
python -m streamlit run app.py
```

Keep large raw files out of Git. Provide reproducible data download/preparation instructions. Deploy only redistributable data/artifacts that fit platform limits; ensure inference works without training during app startup. Never commit credentials.

## 7. Delivery plan and acceptance criteria

The detailed step-by-step plan with dates is in [PLAN.md](PLAN.md). Keep both files in sync.

- [x] Audit candidate archive and document feasibility, license, schema, coverage and actual-time reliability (2026-11-18; see section 4).
- [ ] Create repository, initial README, app skeleton, UI sketch and M1 pitch; confirm team roles and detailed course rubric.
- [x] Implement and test cleaning, deduplication, timestamp handling and hourly historical aggregation (2026-11-18).
- [x] Build M2 dashboard using real data and filters (2026-11-18: `app.py` + `app_pages/`).
- [x] M2 deliverables prepared (2026-11-21):
  - README: M2 summary table, Data Card (5 fields), "What we fixed and why", target histogram (`docs/img/target_histogram.png`), team table to fill.
  - Explore page: insight ("delays build up during the day") and baseline cards.
  - Pitch script and submission checklist: `docs/M2_pitch.md`.
  - Clean-venv install verified; 34 tests passing.
  - Still to do by the team: fill names, a commit from each member, push by 2026-11-25 14:00, submit the link.
- [x] Training/evaluation infrastructure for delay, cancellation and hourly activity (2026-11-18). Files: `src/sources.py`, `features.py`, `estimators.py`, `train.py`, `predict.py`.
  - Chronological split, with the profile computed from training days only.
  - Gate: a model is used only if the 95% paired-bootstrap Brier gain is above 0 AND its ROC-AUC beats the best baseline.
  - Cancellation needs at least 50/10 positives in train/test before a model is trained.
  - Current data: no model beats the airline-rate baseline. Rerun when larger data arrives (README "Adding a larger dataset").
- [ ] Train delay models on the larger dataset when it arrives; report results honestly.
- [ ] Implement congestion profile and integrate both outputs into the app.
- [ ] Add comparison view and transparent limitations; handle missing/unsupported inputs.
- [ ] Test end-to-end inference, midnight/date boundaries, unknown categories, cancellations, duplicated snapshots and absence of future information in features.
- [ ] Deploy on Streamlit Cloud; test fresh installation and public app link.
- [ ] Prepare final five-minute demo: problem, data, historical dashboard, real prediction, flight comparison, evaluation and limitations.

Definition of done: reproducible pipeline, real historical data, at least one evaluated fitted ML model, functional prediction and congestion outputs, clear source attribution, understandable UI, passing relevant tests, deployed app and documented limitations. A simple working project is preferable to ambitious unfinished features. A top grade is a goal, not a guarantee.

## 8. Instructions for Claude

Act as a senior Python/ML developer and teaching partner. Prioritize simplicity and adherence to the course rubric. Help the student understand the code and justify design decisions in the demo. Work in small, testable increments; inspect existing files before editing; do not overwrite student work or add dependencies unnecessarily. Explain important changes and run tests where possible. Never invent data, metrics, model predictions, source availability, or completed work. Keep README and this context aligned with verified findings. Ask only for genuinely blocking decisions; make reversible assumptions explicit.

**Work log (mandatory):** [WORKLOG.md](WORKLOG.md) is the project journal, written in Hebrew. Update it at the end of every work session and whenever a significant decision, finding or change of direction occurs. Keep these parts current:

- the "מצב נוכחי" (current status) block;
- the decisions table, with the reason and the rejected alternatives;
- the open questions;
- a dated entry at the top of the "יומן" (log) section.

Record only verified facts, and mark uncertain items as such. Never put personal details (addresses, phone numbers, emails) in it. Keep CLAUDE.md, PLAN.md and WORKLOG.md consistent.

**Data audit:** done on 2026-11-18 (section 4). **Next task:** follow [PLAN.md](PLAN.md) from Phase 0. Phase 0 sets up the repo, environment and reproducible download script; Phase 1 builds the data pipeline with tests. Do not build model or UI code ahead of the processed data.

## 9. Current status and open decisions

**Settled:**

- **Product name:** FlyCheck (for an academic project; commercial name/trademark availability is not established).
- **Features:** flight delay risk, estimated airport congestion at arrival, historical analysis, flight comparison, and an Airport Guide page.
- **Data:** existing datasets only. OData flydata log (CC BY) is the primary source; TLVFlights aggregates and the CBS price index provide context. The delay target can be constructed (CHPTOL at final status DEPARTED), but only about 761 labelled departures exist.
- **Defaults:** TLV-only departures, 30-minute delay threshold, 3-hour arrival lead time, static dataset, historical congestion proxy, English UI.

**Not done yet:** code, git repository, model, deployment. Only the audit is complete. No accuracy has been established.

**Open:**

- Instructor approval of the small-data scope (and of the US fallback, if it comes to that).
- Rank Air export request (not sent yet).
- EUROCONTROL ADRR access (registration in progress).
- Data decision point: 2026-12-01.
- Permission from the giladct author.
- Team composition and roles.
- The official M2 rubric.
- Whether M1 was submitted.
- Timezone spot-check of CHSTOL.
