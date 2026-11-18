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

## 4. Candidate data sources — verify before implementation

| Source | URL | Intended use |
|---|---|---|
| Historical flydata archive (OData / The Octopus nonprofit) | https://www.odata.org.il/he/dataset/flydata | Primary candidate training and analytics data |
| Candidate CSV download | https://www.odata.org.il/dataset/98b4f544-a7c8-4113-87de-7da1861c3a74/resource/d5ee74ca-2f56-4734-933e-02f4934420c2/download/flydata_log.csv | Download a fixed snapshot if available |
| Israel Airports Authority flight board | https://data.gov.il/datasets/airport_authority/flydata | Reference schema; optional live integration, not required |
| TLV Flights statistics | https://www.tlvflights.com/en/flight-statistics/?range=all | Reference/competitor research |
| Busy TLV | https://busy-tlv.co.il/stats | Reference for congestion analytics |
| Fly TLV | https://fly-tlv.com/airport-status?lang=he | Reference for congestion estimates |

These links and candidate schemas come from the planning discussion; the raw archive has NOT been downloaded or audited. Verify availability, license/redistribution rights, historical coverage, completeness, timestamp meaning and data quality before committing to the project. The official feed reportedly refreshes every 15 minutes; a refreshed flight board is not the same as a historical dataset. Do not assume that a changing estimated flight time is an observed actual departure.

Candidate raw columns (confirm against actual schema): CHOPER airline code; CHOPERD airline name; CHFLTN flight number; CHSTOL scheduled time; CHPTOL updated/possibly actual time; CHAORD arrival/departure; CHLOC1 other airport; CHLOCCT country; CHTERM terminal; CHRMINH status. Inspect snapshot timestamps and status values, if present. Confirm local versus UTC time and handle Asia/Jerusalem daylight saving transitions.

### Data cleaning and feasibility gate

- Download one reproducible snapshot and record source URL, retrieval date, license, date range and row counts.
- Inspect multiple snapshots of the same flight; identify unique flight occurrences using verified identifiers, direction, airport and service date. Handle codeshares without counting the same physical flight multiple times where possible.
- Preserve a defensible scheduled time and a verified final actual time for each completed flight. Distinguish schedule changes from delays. Exclude cancellations, diversions and unresolved flights from the delay target; report separately.
- Calculate delay_minutes = actual_time - scheduled_time in minutes; delayed_30 = delay_minutes >= 30. Negative values can represent early departures; inspect extreme values rather than blindly deleting them.
- If actual departure cannot be verified, stop and document the blocker. Consider a different historical dataset or an explicitly renamed estimate-revision target with instructor approval; never present estimated times as actual observations.
- Check missingness, duplicate rates, time coverage, airline/route sample sizes and class balance. Verify coverage of arrivals and departures for congestion analysis, including zero-flight hours only where collection coverage is demonstrably complete.

## 5. ML design

### Delay prediction (required)

Use a binary classifier for delayed_30. Features must be available before purchase or derived from historical patterns: airline, destination, scheduled hour, weekday, month, optional terminal and historical congestion estimate. Optional Israeli holiday indicator only if a reliable source is available. Do not use actual time, final status, realized flight counts or future records as features.

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
- **About & Model:** methodology, data attribution, coverage, evaluation metrics, limitations and training date.

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

- [ ] Audit candidate archive and document feasibility, license, schema, coverage and actual-time reliability.
- [ ] Create repository, initial README, app skeleton, UI sketch and M1 pitch; confirm team roles and detailed course rubric.
- [ ] Implement and test cleaning, deduplication, timestamp handling and hourly historical aggregation.
- [ ] Build M2 dashboard using real data and filters.
- [ ] Train baseline and candidate delay models; evaluate chronologically; save pipeline and metadata.
- [ ] Implement congestion profile and integrate both outputs into the app.
- [ ] Add comparison view and transparent limitations; handle missing/unsupported inputs.
- [ ] Test end-to-end inference, midnight/date boundaries, unknown categories, cancellations, duplicated snapshots and absence of future information in features.
- [ ] Deploy on Streamlit Cloud; test fresh installation and public app link.
- [ ] Prepare final five-minute demo: problem, data, historical dashboard, real prediction, flight comparison, evaluation and limitations.

Definition of done: reproducible pipeline, real historical data, at least one evaluated fitted ML model, functional prediction and congestion outputs, clear source attribution, understandable UI, passing relevant tests, deployed app and documented limitations. A simple working project is preferable to ambitious unfinished features. A top grade is a goal, not a guarantee.

## 8. Instructions for Claude

Act as a senior Python/ML developer and teaching partner. Prioritize simplicity and adherence to the course rubric. Help the student understand the code and justify design decisions in the demo. Work in small, testable increments; inspect existing files before editing; do not overwrite student work or add dependencies unnecessarily. Explain important changes and run tests where possible. Never invent data, metrics, model predictions, source availability, or completed work. Keep README and this context aligned with verified findings. Ask only for genuinely blocking decisions; make reversible assumptions explicit.

**First task:** audit the historical TLV dataset before designing the final model. Verify the candidate download, inspect schema and sample rows, determine whether completed flight events and actual departure times are recoverable, estimate usable unique flights and coverage, and report a go/no-go decision with specific evidence. Then propose the smallest working M1/M2 implementation. If the archive fails, identify viable alternatives and flag any scope change for approval. Do not start building a full application based on an unverified schema.

## 9. Current status and open decisions

- Confirmed product name: FlyCheck (for an academic project; commercial name/trademark availability is not established).
- Confirmed features: flight delay prediction, estimated airport congestion at arrival, and historical analysis. Flight comparison is a planned feature from the discussion.
- Proposed defaults, not verified results: TLV-only departures, 30-minute delay threshold, three-hour arrival lead time, static dataset, historical congestion proxy, and English UI.
- No dataset downloaded/audited, model trained, accuracy established, code implemented, or deployment completed in this planning discussion.
- Confirm team composition, instructor's full submission instructions, current milestone status, dataset license/schema/coverage, and whether the delay target can be constructed reliably.
