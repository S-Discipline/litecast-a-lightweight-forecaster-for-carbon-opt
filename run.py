"""LiteCast reproduction: forecast quality (MAPE, concordance) and carbon-aware
scheduling (carbon optimality vs oracle) for LiteCast and Persistence, across
the real hourly CarbonCast region dataset (2020-2021).

Reads config.yaml for experiment settings. Prints per-region and aggregate
metrics to stdout so each run's log is the evidence channel.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_region, REGIONS  # noqa: E402
from src.forecaster import LiteCast, MixLiteCast, Persistence  # noqa: E402
from src.metrics import mape, concordance_index, carbon_optimality, additional_emissions  # noqa: E402
from src.scheduler import evaluate_schedule  # noqa: E402


def run_region(cfg: dict, region: str, cache_dir: str) -> dict:
    d = load_region(cache_dir, region)
    ci = d["ci"]["carbon_intensity"].astype(float)
    ci = ci[~ci.index.duplicated()].sort_index()
    da = d["da"]

    test_start = pd.Timestamp(cfg["test_start"])
    test_end = pd.Timestamp(cfg["test_end"])
    train_days = cfg["train_days"]
    horizon = cfg["horizon"]
    job_lengths = cfg["job_lengths"]
    slacks = cfg["slacks"]
    modes = cfg["modes"]
    step_days = cfg.get("step_days", 1)
    use_weather = cfg.get("use_weather", True)
    use_demand = cfg.get("use_demand", True)
    use_time = cfg.get("use_time", True)
    model = cfg.get("model", "direct")
    compare_persistence = cfg.get("compare_persistence", True)

    fc_cls = LiteCast if model == "direct" else MixLiteCast
    forecaster = fc_cls(train_days=train_days, horizon=horizon,
                        use_weather=use_weather, use_demand=use_demand,
                        use_time=use_time)
    persistence = Persistence(horizon=horizon)

    origins = pd.date_range(start=test_start, end=test_end, freq=f"{step_days}D")

    mapes = {"litecast": [], "persistence": []}
    cis = {"litecast": [], "persistence": []}
    agg = {"litecast": {}, "persistence": {}}
    keys = [(L, T, m) for L in job_lengths for T in slacks for m in modes]
    for name in agg:
        for k in keys:
            agg[name][k] = {"e_pred": 0.0, "e_oracle": 0.0}
    n_jobs = 0

    for origin in origins:
        hist = ci[ci.index <= origin]
        if len(hist) < train_days * 24:
            continue
        actual = ci[ci.index > origin]
        if len(actual) < horizon:
            continue
        future_idx = pd.date_range(start=hist.index[-1] + pd.Timedelta(hours=1),
                                   periods=horizon, freq="h")
        future_idx = future_idx[future_idx <= actual.index.max()]
        if len(future_idx) == 0:
            continue
        a = actual.reindex(future_idx)

        forecasts = {}
        try:
            forecasts["litecast"] = forecaster.forecast(hist, da).reindex(future_idx)
        except Exception as e:  # noqa: BLE001
            print(f"  [{region}] litecast failed at {origin}: {e}")
            continue
        if compare_persistence:
            forecasts["persistence"] = persistence.forecast(hist, da).reindex(future_idx)

        for name, fc in forecasts.items():
            f = fc.fillna(a).replace([np.inf, -np.inf], np.nan)
            mask = a.notna() & f.notna()
            if mask.sum() < 24:
                continue
            mapes[name].append(mape(a[mask], f[mask]))
            cis[name].append(concordance_index(a[mask], f[mask]))

            for (L, T, m) in keys:
                wf = f[:T + L]
                wa = a[:T + L]
                if len(wa) < T + L:
                    continue
                e_pred, e_oracle = evaluate_schedule(wf, wa, L, T, m)
                agg[name][(L, T, m)]["e_pred"] += e_pred
                agg[name][(L, T, m)]["e_oracle"] += e_oracle
        n_jobs += 1

    out = {"region": region, "n_origins": n_jobs}
    for name in ("litecast", "persistence"):
        out[f"mape_{name}"] = float(np.nanmean(mapes[name])) if mapes[name] else float("nan")
        out[f"concordance_{name}"] = float(np.nanmean(cis[name])) if cis[name] else float("nan")
    out["optimality"] = {}
    out["additional"] = {}
    for name in ("litecast", "persistence"):
        for k in keys:
            eo = agg[name][k]["e_oracle"]
            ep = agg[name][k]["e_pred"]
            out["optimality"][(name, *k)] = carbon_optimality(ep, eo) if eo > 0 else float("nan")
            out["additional"][(name, *k)] = additional_emissions(ep, eo) if eo > 0 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = os.path.expanduser(cfg.get("cache_dir", "~/litecast_data"))
    regions = cfg.get("regions", REGIONS)
    n_workers = min(cfg.get("n_workers", 4), len(regions))

    # pre-download all region data in the main process (workers must not download)
    for r in regions:
        load_region(cache_dir, r)

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_region, cfg, r, cache_dir): r for r in regions}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"[{r}] FAILED: {e}", flush=True)
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x["region"])

    # ---- compact summary block (the evidence contract) ----
    print("\n" + "=" * 78)
    print("LiteCast reproduction — summary")
    print(f"config: {args.config}")
    print(f"model={cfg.get('model')} train_days={cfg.get('train_days')} horizon={cfg.get('horizon')} "
          f"test={cfg.get('test_start')}..{cfg.get('test_end')} step_days={cfg.get('step_days')}")
    print(f"job_lengths={cfg.get('job_lengths')} slacks={cfg.get('slacks')} modes={cfg.get('modes')}")
    print("=" * 78)

    df = pd.DataFrame([{
        "region": r["region"],
        "n_origins": r["n_origins"],
        "MAPE_litecast%": r["mape_litecast"],
        "MAPE_persistence%": r["mape_persistence"],
        "CI_litecast": r["concordance_litecast"],
        "CI_persistence": r["concordance_persistence"],
    } for r in results])
    print("\nForecast quality per region (real 2021 hourly data):")
    print(df.to_string(index=False))

    rows = []
    for r in results:
        for (name, L, T, m), v in r["optimality"].items():
            rows.append({"region": r["region"], "forecaster": name, "L": L, "T": T,
                         "mode": m, "optimality%": v,
                         "additional%": r["additional"][(name, L, T, m)]})
    sched = pd.DataFrame(rows)
    print("\nCarbon optimality (%) — mean across regions:")
    if len(sched) == 0:
        print("  (no scheduling results — check forecast failures)")
        return
    piv = sched.pivot_table(index=["mode", "L", "T"], columns="forecaster",
                            values="optimality%", aggfunc="mean").round(1)
    print(piv.to_string())
    print("\nAdditional emissions vs oracle (%) — mean across regions (0 = oracle):")
    piv2 = sched.pivot_table(index=["mode", "L", "T"], columns="forecaster",
                             values="additional%", aggfunc="mean").round(1)
    print(piv2.to_string())

    print("\nAggregate headline metrics:")
    print(f"  mean MAPE (LiteCast):      {df['MAPE_litecast%'].mean():.1f}%")
    if "MAPE_persistence%" in df and df["MAPE_persistence%"].notna().any():
        print(f"  mean MAPE (Persistence):   {df['MAPE_persistence%'].mean():.1f}%")
    print(f"  mean concordance (LiteCast):      {df['CI_litecast'].mean():.3f}")
    if "CI_persistence" in df and df["CI_persistence"].notna().any():
        print(f"  mean concordance (Persistence):   {df['CI_persistence'].mean():.3f}")
    for mode in cfg.get("modes", []):
        sub = sched[(sched["mode"] == mode) & (sched["forecaster"] == "litecast")]
        for label, mask in [
            ("job length <24H", sub["L"] < 24),
            ("job length <96H", sub["L"] < 96),
            ("job length <168H", sub["L"] < 168),
        ]:
            mean = sub[mask]["optimality%"].mean()
            print(f"  LiteCast {mode} {label}: carbon optimality {mean:.1f}%")
    print("=" * 78)

    out = cfg.get("out_csv")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        sched.to_csv(out, index=False)
        df.to_csv(out.replace(".csv", "_forecast.csv"), index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
