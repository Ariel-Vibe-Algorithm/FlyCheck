"""Build docs/app_artifact.html: the simplified FlyCheck page published as a claude.ai artifact.

Reads the processed data and model metrics, computes the same historical numbers the
Streamlit app shows (via src.analytics), and injects them into docs/artifact/template.html.

    python docs/artifact/build.py

Nothing here is a model prediction; the page shows historical rates only.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import analytics as A  # noqa: E402


def collect() -> dict:
    f = pd.read_parquet(ROOT / 'data/processed/flights.parquet')
    rep = json.load(open(ROOT / 'data/processed/cleaning_report.json'))
    met = json.load(open(ROOT / 'models/metrics.json'))
    prof = pd.read_csv(ROOT / 'models/hourly_profile.csv')['flights_per_hour'].round(2).tolist()
    days = [str(d) for d in A.complete_schedule_days(f)]
    arr_prof = A.hourly_profile(f, 'A').round(2).tolist()
    s = A.labelled(f)
    air = A.group_delay_stats(s, ['airline_code'], min_n=10)
    names = s.groupby('airline_code')['airline_name'].agg(lambda x: x.mode().iloc[0])
    airlines = [{"code": r.airline_code, "name": names[r.airline_code].title(), "n": int(r.n), "rate": round(r.share_delayed_30,3), "med": float(r.median_delay)} for r in air.itertuples()]
    byhour = []
    for lo in range(0, 24, 3):
        sub = s[(s.hour >= lo) & (s.hour < lo+3)]
        byhour.append({"h": lo, "n": int(len(sub)), "rate": round(float(sub.delayed_30.astype(float).mean()),3)})
    bins = [-1e9, 0, 15, 30, 60, 120, 1e9]
    cut = pd.cut(s.delay_minutes, bins, right=False, labels=["early","0-15","15-30","30-60","60-120","120+"])
    dist = {"labels": list(cut.cat.categories), "counts": [int(x) for x in cut.value_counts(sort=False)]}
    dst = A.group_delay_stats(s, ['airport_code'], min_n=1).head(12)
    cities = s.groupby('airport_code')['city'].agg(lambda x: x.mode().iloc[0])
    countries = s.groupby('airport_code')['country'].agg(lambda x: x.mode().iloc[0])
    dests = [{"code": r.airport_code, "city": cities[r.airport_code].title(), "country": countries[r.airport_code].title(), "n": int(r.n), "rate": round(r.share_delayed_30,3)} for r in dst.itertuples()]
    # all destinations for dropdown + lookup
    alld = A.group_delay_stats(s, ['airport_code'], min_n=1)
    destAll = [{"code": r.airport_code, "city": cities[r.airport_code].title(), "n": int(r.n), "rate": round(r.share_delayed_30,3)} for r in alld.itertuples()]
    # all airlines (any n) for lookups
    alla = A.group_delay_stats(s, ['airline_code'], min_n=1)
    airAll = [{"code": r.airline_code, "name": names[r.airline_code].title(), "n": int(r.n), "rate": round(r.share_delayed_30,3)} for r in alla.itertuples()]
    t = pd.read_csv(ROOT / 'data/processed/airline_reliability_tlvflights.csv')
    tot = int(t.total_flights.sum())
    t = t.sort_values('total_flights', ascending=False).head(15)
    tlv = [{"code": r.airline_code, "name": r.airline_name, "n": int(r.total_flights), "ontime": round(r.on_time_rate*100,1), "late30": round(r.late_30_rate,3), "canc": round(r.cancel_rate,3)} for r in t.itertuples()]
    c = pd.read_csv(ROOT / 'data/processed/price_index_travel_abroad.csv'); c['month']=pd.to_datetime(c.month)
    c = c[c.month.dt.year >= 2015]
    cbs = [{"y": int(r.month.year), "m": int(r.month.month), "v": round(r.index_value,1)} for r in c.itertuples()]
    # weekday x hour heatmap of scheduled intl departures on complete days
    i = A.international(f); i = i[(i.direction=='D') & i.service_date.isin(A.complete_schedule_days(f))]
    wk = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    ndays = i.groupby('weekday').service_date.nunique()
    heat = {w: [round(float(((i.weekday==w)&(i.hour==h)).sum())/max(int(ndays.get(w,0)),1),1) for h in range(24)] for w in wk if w in ndays}
    models = {}
    for task, v in met['tasks'].items():
        models[task] = {"split": v['split'], "results": {k: r.get('test', {}) | {"kind": r.get('kind')} for k, r in v['results'].items()},
                        "best_model": v.get('best_model'), "best_baseline": v.get('best_baseline'), "beats": v.get('beats_baseline'),
                        "ready": v.get('ready_for_app'), "reason": v.get('status_reason'), "ci": v.get('brier_gain_ci95')}
    out = dict(report=rep, hourly=prof, hourly_arr=arr_prof, days=days, airlines=airlines, airAll=airAll, byhour=byhour, dist=dist,
               dests=dests, destAll=destAll, tlv=tlv, total_tlv=tot, cbs=cbs, heat=heat, heat_days={k:int(v) for k,v in ndays.items()},
               models=models, trained=met['trained_at_utc'], data_span=met['data'])
    return out


def main() -> None:
    data = json.dumps(collect(), separators=(",", ":"), default=str)
    template = (ROOT / "docs/artifact/template.html").read_text(encoding="utf-8")
    out = ROOT / "docs/app_artifact.html"
    out.write_text(template.replace("__DATA__", data), encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
