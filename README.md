# FlyCheck ✈️ — Know before you book

FlyCheck helps travelers departing from **Ben Gurion Airport (TLV)** judge a flight *before* buying a ticket. It shows how often similar flights were delayed, and how busy the departure hall usually is when you arrive at the airport.

> Academic MVP. Course: Vibe Coding with Claude, Ariel University, Fall 2026.

## The problem

When you book a flight from TLV you can compare price, airline and departure time. Booking sites don't tell you:

- How often does this airline or departure time leave 30+ minutes late?
- How crowded is the terminal likely to be when I arrive, 3 hours before departure?

FlyCheck answers these questions from historical flight records, a machine-learning model and interactive charts. It is decision support, not a live flight tracker, and it never guarantees what will happen to a specific flight.

## Simplified app design

**[Open the simplified FlyCheck page](https://claude.ai/artifact/Cb6JECwreqzy6hxNtkR1e8)** (public link). The source is in [`docs/app_artifact.html`](docs/app_artifact.html).

It is a simpler, user-facing version of the app design, in a liquid-glass style with airplane animations. Each question gets one plain answer, and the numbers behind it are one tap away. It has four pages:

| Page | What it answers |
|---|---|
| **Check a flight** | "Will it leave on time?": how many in 10 of the airline's flights left 30+ min late. "How crowded will the airport be?": Quiet / Moderate / Busy when you arrive |
| **Compare** | The same two answers for several flights side by side, with an optional price |
| **Airport tips** | When to arrive, terminals, parking, getting there, carry-on, lounges |
| **About the data** | Key numbers and charts; sources, limitations, models and cleaning steps collapsed below |

The delay answer uses the airline's historical rate on the airport board, the best baseline in `models/metrics.json`. With fewer than 10 departures for the airline, it uses the rate of all flights and says so. No model predictions are shown. To rebuild the page from the current processed data, run `python docs/artifact/build.py`. It fills [`docs/artifact/template.html`](docs/artifact/template.html).

## App prototype

**[Open the interactive prototype](https://claude.ai/artifact/P39yWMPVP7GJLMZgTsXeL4?sk=TCwr40dUzM0DzY7Z0TZDjA#analyze)** (public link). The source is also in [`docs/app_prototype.html`](docs/app_prototype.html).

The prototype shows the planned app layout. Its historical numbers are real; no model predictions are shown yet. It has five pages:

| Page | What it does | Status |
|---|---|---|
| **Analyze a flight** | Enter airline, destination, date and time. Shows the historical delay reference and the departure-hall activity at your arrival time, with an hourly chart | Historical parts work; the ML probability comes in M3 |
| **Compare flights** | Add options and sort them by delay history, hall activity or the price you enter | Works on historical data |
| **Explore data** | Delay distribution, delay by hour, airlines, destinations, airline reliability, airfare price index | Real data |
| **Airport guide** | When to arrive, terminals, parking, trains and buses, carry-on rules, lounges | Static content with official links |
| **About & data** | Cleaning funnel, sources, limitations, project status | Real data |

## Project status

| Step | Due | Status |
|---|---|---|
| Data audit (feasibility, schema, coverage) | — | ✅ Done (Nov 18) |
| Data download + cleaning pipeline + tests | — | ✅ Done |
| App prototype | — | ✅ Done |
| **M2**: analytics dashboard in Streamlit | Nov 25 | ✅ Ready to submit (see [M2 summary](#m2-data-dashboard-nov-25)) |
| Baselines + training/evaluation pipeline for 3 targets | — | ✅ Built (34 tests passing) |
| **M3**: delay model trained, evaluated, integrated | Dec 16 | Planned |
| Deployment to Streamlit Community Cloud | Dec 23 | Planned |
| Final demo | Jan 6, 2027 | Planned |

The step-by-step plan is in [PLAN.md](PLAN.md). Decisions and progress are logged in [WORKLOG.md](WORKLOG.md), in Hebrew.

## M2: data dashboard (Nov 25)

| Requirement | What we have | Where |
|---|---|---|
| **Clean data**: 300+ rows, Data Card, what we fixed | 889 labelled departures (5,213 flights in total), one row per physical flight | `data/processed/flights.parquet` · [Data Card](#data-card) · [What we fixed and why](#what-we-fixed-and-why) |
| **EDA page**: 4 metrics, 2 charts, 1 filter | 4 KPIs (departures, share delayed 30+, median delay, cancellations). Charts include the delay histogram, delay by departure time, airlines, and hourly activity. Sidebar filters: airline, destination country, hour, weekday | Streamlit page **Explore data** (`app_pages/explore.py`) |
| **Insight** | Delays build up during the day (below) | Top of **Explore data** |
| **Baseline** | The airline's past delay rate: ROC-AUC 0.64, Brier 0.256 on the latest days | Top of **Explore data** and its **Baselines & models** tab |

**Insight: delays build up during the day.** Departures scheduled before noon left 30+ minutes late 45% of the time (n = 449). After 18:00 the share was 63% (n = 142). This is not only because of which airlines fly in the evening: El Al alone went from 32% (n = 161) to 64% (n = 42). On the two fully covered days (Jun 25 and Jul 2, 2026) the evening share reached 77–79%. For a traveler who cares about delays, a morning flight is the safer bet.

**Baseline M3 must beat.** We tested on the 351 latest departures (Jul 3, Nov 17–18 2026), which were never used for training:

| Baseline | ROC-AUC | Brier score |
|---|---|---|
| Overall rate (52%, the same for every flight) | 0.50 | 0.272 |
| **Airline's past rate (best baseline)** | **0.64** | **0.256** |

A first logistic regression and random forest did not beat the airline rate: their best ROC-AUC was 0.52. The full table is in `models/metrics.json` and on the dashboard.

## Data

| Source | What we use | License / terms |
|---|---|---|
| [Flight board archive](https://www.odata.org.il/he/dataset/flydata) by The Octopus (odata.org.il) | Snapshots of the Israel Airports Authority flight board, with scheduled and actual times | CC BY |
| [data.gov.il flight board](https://data.gov.il/dataset/flydata) | Current board (about 1 day back, 3–4 days ahead): recent schedule and live status | Open (data.gov.il) |
| [TLVFlights statistics](https://www.tlvflights.com/en/flight-statistics/?range=all) | Airline-level reliability (about 101k flights, Dec 2025 – Nov 2026) | Free to use with a link |
| [Israel CBS price index 121390](https://api.cbs.gov.il/index/data/price?id=121390&format=json&lang=en) | Monthly "travel abroad and local flights" index, as price context | Public |

### What the data looks like after cleaning

There are two flight sources:

- **The OData archive** logs the board on only 4 days (Jan 17, Mar 17, Jun 26, Jul 3 2026).
- **One data.gov.il board snapshot** (Nov 18, 2026) adds Nov 17–20.

TLV was closed from Feb 11 to May 15, 2026.

| Step | Rows |
|---|---|
| Raw board rows (225,500 archive + 1,827 snapshot) | 227,327 (3 truncated lines skipped) |
| Unique flight listings (latest snapshot kept) | 7,507 |
| Physical flights (codeshares merged) | 5,213 |
| Departures | 2,595 (931 completed, 27 cancelled, 1,637 without a final status yet) |
| **Labelled departures** (completed, international, trusted delay) | **889** |

Of the 889 labelled departures, 52.1% left 30+ minutes late; the median delay is 31 minutes.

Nothing is deleted silently. Domestic flights (41) and one likely unrecorded schedule change (a flight that "left" 127 min early) are flagged and counted in `data/processed/cleaning_report.json`.

**Larger sources requested:**

- a year-long archive of the same official board (Rank Air);
- EUROCONTROL research flight data.

The pipeline converts every source into one internal flight schema, so a new source only needs a loader.

### Data Card

| Field | |
|---|---|
| **Source and link** | Israel Airports Authority flight board. Archive by The Octopus: [odata.org.il/he/dataset/flydata](https://www.odata.org.il/he/dataset/flydata). One snapshot of the live board: [data.gov.il/dataset/flydata](https://data.gov.il/dataset/flydata). Downloaded Nov 18, 2026 by `src/download_data.py`, which records each file's URL, time, size and SHA-256 in `data/raw/manifest.json`. |
| **Size and format** | Raw: a CSV of 225,500 board snapshot rows × 20 columns, plus a JSON snapshot of 1,827 rows. Cleaned: `data/processed/flights.parquet`, 5,213 rows (one per physical flight) × 25 columns. 889 of them are labelled departures. Main columns: `scheduled`, `airline_code`, `airport_code`, `country`, `hour`, `weekday`, `terminal`, `delay_minutes`, `delayed_30`. |
| **Target variable (y)** | `delayed_30` (binary): 1 if `delay_minutes` ≥ 30. `delay_minutes` = board time − scheduled time, in minutes, Israel local time. 52.1% of labelled departures are 1. Delay median 31 min, mean 47, range −21 to 386. |
| **Missing values and problems** | Each flight appears up to 86 times (one row per board refresh). Codeshares list one plane under several flight numbers. 3 CSV lines are truncated. `delay_minutes` is empty for 64% of flights by design: arrivals, 27 cancellations, and 1,637 departures with no final status when logging stopped. Check-in counter and zone are missing for 57%. One flight "left" 127 min early, which is likely an unrecorded schedule change. |
| **License and biases** | CC BY (attribution required); the data.gov.il board is open. **Biases:** only 10 scheduled days, all around the wartime airport closure (Feb 11 – May 15, 2026), so there is no seasonality. El Al is 33% of labelled departures. On the last logged day of each source, late flights had often not departed when logging stopped, so delayed evening flights are likely under-represented there. The evening share was 0–20% on Jul 3 and Nov 18, against 77–79% on fully covered days. The board's "actual" time is close to takeoff, not gate departure. |

![Histogram of the target variable](docs/img/target_histogram.png)

### What we fixed and why

1. **Kept one row per flight.** The board was logged every few minutes, so one flight appeared up to 86 times. We keep each flight's latest snapshot (227,327 rows → 7,507 listings). Otherwise busy flights would be counted dozens of times, and early "ON TIME" snapshots would hide the final delay.
2. **Merged codeshares.** One plane sold under several flight numbers (for example AZ 807 + LH 5233) becomes one physical flight (7,507 → 5,213). Otherwise routes with many partner airlines would weigh several times more than they should.
3. **Labelled only real outcomes, and flagged instead of deleting.** The delay target uses only completed departures with a real departure time. Cancelled flights (27) and flights with no final status (1,637) are kept but left out of the target. Filling them in would invent on-time flights. Domestic flights (41) and the "127 min early" flight are flagged out of scope and counted in `data/processed/cleaning_report.json`.

We also skip and count the 3 truncated CSV lines instead of failing on them.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

python -m src.download_data      # downloads raw data into data/raw/ (+ manifest.json)
python -m src.data_processing    # builds data/processed/flights.parquet + cleaning_report.json
python -m src.train              # baselines + models for delay, cancellation, airport activity
python -m pytest                 # runs the tests (pipeline, analytics, models, every app page)
python -m streamlit run app.py   # opens the app at http://localhost:8501
```

### Adding a larger dataset

The pipeline is ready for a larger archive; no code changes are needed for the supported formats.

1. Put the files in `data/raw/`:
   - **Rank Air export:** any CSV in `data/raw/rankair/`, with the column names from their database or CSV.
   - **More board snapshots:** `python -m src.download_data` adds `data/raw/govil/board_*.json`.
2. Run `python -m src.data_processing`. Every source is loaded, converted to one schema, and merged; the latest snapshot of each flight wins.
   - Rank Air times may be stored with a timezone. The loader tests both readings against the overlapping archive days and keeps the one that matches. The result is in `cleaning_report.json` → `sources.rankair`.
3. Run `python -m src.train`, then `python -m pytest`.
   - Each model is shown in the app **only if** it clearly beats the best baseline on the held-out latest days.
   - Cancellation gets a model automatically once there are enough cancellations (≥ 50 in train, ≥ 10 in test).
4. Check the data license before committing. `data/processed/flights.parquet` contains every source's rows. If a source may not be redistributed, commit only `models/` and keep the processed file local.

The processed files in `data/processed/` are committed, so the app runs right after installing the requirements. Raw data is not committed; `src/download_data.py` re-downloads it and records the source URL, retrieval time, size and SHA-256 of each file.

## Repository structure

```text
FlyCheck/
├── README.md            this file
├── CLAUDE.md            project context and verified data findings (for the AI assistant)
├── PLAN.md              step-by-step build plan
├── WORKLOG.md           decisions and progress log (Hebrew)
├── requirements.txt     pinned dependencies
├── app.py               Streamlit entry point (navigation)
├── app_pages/           one file per page: analyze, compare, explore, guide, about
├── docs/
│   └── app_prototype.html   interactive app prototype
├── src/
│   ├── download_data.py     reproducible raw-data download
│   ├── sources.py           adapters: OData archive, data.gov.il snapshots, Rank Air export
│   ├── data_processing.py   cleaning, deduplication, delay target, context data
│   ├── analytics.py         hourly activity, congestion index, historical delay rates
│   ├── features.py          booking-time features (no outcome information)
│   ├── estimators.py        airline-rate baseline (sklearn-compatible)
│   ├── train.py             baselines vs models for 3 targets, chronological split, gate
│   └── predict.py           inference for the app (only gated models)
├── models/              metrics.json, hourly profile, fitted pipelines
├── tests/               pipeline, analytics, sources, models and page smoke tests
└── data/
    ├── raw/             git-ignored; rebuilt by download_data.py
    └── processed/       cleaned flights, cleaning report, TLVFlights and price-index tables
```

## Method in brief

- **Delay target:** `delay_minutes` = board time − scheduled time, for departures whose final status is DEPARTED. `delayed_30` = delay ≥ 30 min. Times are Israel local time (checked against the live board on Nov 18, 2026).
- **Three targets, each with baselines and a model** (`python -m src.train`). Evaluation is chronological: the latest days are the test set, and historical aggregates come from the training days only.

  | Target | Baselines | Model | Result on current data |
  |---|---|---|---|
  | Delay 30+ min | majority class, overall rate, airline rate | logistic regression, random forest | Airline rate is best (ROC-AUC 0.64); no model clearly beats it |
  | Cancellation | overall rate, airline rate (archive, TLVFlights) | logistic regression | Baselines only: 26 cancellations, too few for a model |
  | Departures per hour | hourly mean, weekday-hour mean | random forest regressor | Hourly mean is best (MAE 3.4) |

- **When a model is used:** only if it clearly beats the best baseline on the test days. That means the 95% paired-bootstrap interval of the Brier improvement is above zero, and its ROC-AUC is higher. Otherwise the app shows historical rates. Features are only things known before booking: airline, destination, country, weekday, hour, holidays, scheduled activity.
- **Departure-hall activity:** the number of international departures scheduled 1–4 hours after you arrive, ranked against all hours of the day (0–100, low / medium / high). It is a relative index, not a measurement of queues or waiting time.

## Limitations

- The labelled sample is small (889 departures) and comes from an unusual period around the airport closure. It cannot show seasonality.
- The board's "actual" time is close to takeoff, not gate departure.
- FlyCheck does not predict ticket prices or security-queue times; no public data exists for them.
- Historical patterns are not a guarantee about any future flight. Always follow official airline and airport guidance on when to arrive.

## Team

| Name | Role | GitHub |
|---|---|---|
| *to fill* | Product | |
| *to fill* | Programmer | |
| *to fill* | Data | |
| *to fill* | UI/UX | |

Every team member commits to this repository (course requirement).

## Attribution

- Flight board data: Israel Airports Authority via data.gov.il, archived by The Octopus (odata.org.il), CC BY.
- Airline statistics: [tlvflights.com](https://www.tlvflights.com).
- Price index: Israel Central Bureau of Statistics.
