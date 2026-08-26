"""Compare LiteCast against the real CarbonCast CNN-LSTM on the same Jul-Dec 2021
test window. CarbonCast forecasts come from the pretrained models in the
carbonfirst/CarbonCast repository (regenerated locally from saved_second_tier_models).

Claims tested:
  * LiteCast forecast concordance vs CarbonCast (paper: ~10% higher, 15% in a
    high-wind region) and MAPE ranges.
  * Scheduling carbon optimality / additional emissions vs CarbonCast.
"""
from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_region, REGIONS  # noqa: E402
from src.forecaster import LiteCast  # noqa: E402
from src.metrics import mape, concordance_index, additional_emissions  # noqa: E402
from src.scheduler import evaluate_schedule  # noqa: E402


def load_carboncast(region: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join("carboncast_forecasts",
                                  f"{region}_direct_96hr_CI_forecasts_0.csv"),
                     parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    # overlapping 96h rolling windows: each hour appears once per window; keep the
    # last (freshest, shortest-horizon) forecast for each hour
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return df.set_index("datetime").sort_index()


def run_region(cfg: dict, region: str, cache_dir: str) -> dict:
    d = load_region(cache_dir, region)
    ci = d["ci"]["carbon_intensity"].astype(float)
    ci = ci[~ci.index.duplicated()].sort_index()
    da = d["da"]

    train_days = cfg["train_days"]
    horizon = cfg["horizon"]
    job_lengths = cfg["job_lengths"]
    slacks = cfg["slacks"]
    modes = cfg["modes"]
    step_days = cfg.get("step_days", 1)

    cc = load_carboncast(region)
    cc_fc = cc["avg_carbon_intensity_forecast"]
    cc_act = cc["carbon_intensity_actual"]
    test_start = cc_fc.index.min()
    test_end = cc_fc.index.max()

    forecaster = LiteCast(train_days=train_days, horizon=horizon,
                          use_weather=cfg.get("use_weather", True),
                          use_demand=cfg.get("use_demand", True),
                          use_time=cfg.get("use_time", True))

    origins = pd.date_range(start=test_start, end=test_end, freq=f"{step_days}D")

    agg = {}
    for name in ("litecast", "carboncast"):
        agg[name] = {"mape": [], "ci": [], "sched": {}}
        for (L, T, m) in [(L, T, m) for L in job_lengths for T in slacks for m in modes]:
            agg[name]["sched"][(L, T, m)] = {"e_pred": 0.0, "e_oracle": 0.0}

    for origin in origins:
        hist = ci[ci.index <= origin]
        if len(hist) < train_days * 24:
            continue
        # window covered by CarbonCast's rolling 96h forecasts
        common = cc_fc.index[(cc_fc.index > origin) & (cc_fc.index <= origin + pd.Timedelta(hours=96))]
        if len(common) < 24:
            continue
        a = ci.reindex(common).astype(float)
        cc_f = cc_fc.reindex(common).astype(float)

        try:
            lc_f = forecaster.forecast(hist, da).reindex(common).astype(float)
        except Exception as e:  # noqa: BLE001
            print(f"  [{region}] litecast failed at {origin}: {e}", flush=True)
            continue

        for name, fc in (("litecast", lc_f), ("carboncast", cc_f)):
            mask = a.notna() & fc.notna()
            if mask.sum() < 24:
                continue
            agg[name]["mape"].append(mape(a[mask], fc[mask]))
            agg[name]["ci"].append(concordance_index(a[mask], fc[mask]))
            for (L, T, m) in agg[name]["sched"]:
                wf = fc[:T + L]
                wa = a[:T + L]
                if len(wa) < T + L:
                    continue
                e_pred, e_oracle = evaluate_schedule(wf, wa, L, T, m)
                agg[name]["sched"][(L, T, m)]["e_pred"] += e_pred
                agg[name]["sched"][(L, T, m)]["e_oracle"] += e_oracle

    out = {"region": region, "n_origins": len(origins)}
    for name in ("litecast", "carboncast"):
        out[f"mape_{name}"] = float(np.nanmean(agg[name]["mape"])) if agg[name]["mape"] else float("nan")
        out[f"ci_{name}"] = float(np.nanmean(agg[name]["ci"])) if agg[name]["ci"] else float("nan")
    out["additional"] = {}
    for name in ("litecast", "carboncast"):
        for k, v in agg[name]["sched"].items():
            eo = v["e_oracle"]
            out["additional"][(name, *k)] = additional_emissions(v["e_pred"], eo) if eo > 0 else float("nan")
    return out


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cache_dir = os.path.expanduser(cfg.get("cache_dir", "~/litecast_data"))
    regions = cfg.get("regions", REGIONS)

    for r in regions:
        load_region(cache_dir, r)

    results = []
    with ProcessPoolExecutor(max_workers=cfg.get("n_workers", 4)) as ex:
        futs = {ex.submit(run_region, cfg, r, cache_dir): r for r in regions}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"[{r}] FAILED: {e}", flush=True)
    results.sort(key=lambda x: x["region"])

    print("\n" + "=" * 78)
    print("LiteCast vs CarbonCast (real CNN-LSTM) — Jul-Dec 2021, 96h forecasts")
    print(f"config: {cfg.get('job_lengths')} lengths, {cfg.get('slacks')} slacks, {cfg.get('modes')} modes")
    print("=" * 78)

    df = pd.DataFrame([{
        "region": r["region"],
        "n_origins": r["n_origins"],
        "MAPE_litecast%": r["mape_litecast"],
        "MAPE_carboncast%": r["mape_carboncast"],
        "CI_litecast": r["ci_litecast"],
        "CI_carboncast": r["ci_carboncast"],
    } for r in results])
    print("\nForecast quality (Jul-Dec 2021):")
    print(df.round(3).to_string(index=False))

    print("\nConcordance advantage of LiteCast over CarbonCast per region:")
    adv = df["CI_litecast"] - df["CI_carboncast"]
    print(f"  mean ΔCI (LiteCast - CarbonCast): {adv.mean():.3f}")
    print(f"  regions where LiteCast CI higher: {(adv > 0).sum()}/{len(adv)}")
    print(f"  mean CI LiteCast:   {df['CI_litecast'].mean():.3f}")
    print(f"  mean CI CarbonCast: {df['CI_carboncast'].mean():.3f}")

    rows = []
    for r in results:
        for (name, L, T, m), v in r["additional"].items():
            rows.append({"region": r["region"], "forecaster": name, "L": L, "T": T,
                         "mode": m, "additional%": v})
    sched = pd.DataFrame(rows)
    print("\nAdditional emissions vs oracle (%) — mean across regions (0 = oracle):")
    piv = sched.pivot_table(index=["mode", "L", "T"], columns="forecaster",
                            values="additional%", aggfunc="mean").round(1)
    print(piv.to_string())
    print(f"\n  overall mean additional emissions: LiteCast={sched[sched.forecaster=='litecast']['additional%'].mean():.1f}%  "
          f"CarbonCast={sched[sched.forecaster=='carboncast']['additional%'].mean():.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
