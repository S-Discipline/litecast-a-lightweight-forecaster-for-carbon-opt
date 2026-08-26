"""LiteCast heuristic experiment (Algorithm 1).

Compares regular LiteCast scheduling against LiteCast + the dynamic
schedule-update heuristic. The heuristic periodically retrains on newly
available data while a job waits; if the new forecast's concordance improves
and expected emissions are lower, it moves the job's start time.

Claims tested (paper Fig 7/8):
  * LiteCast + Heuristic reaches up to ~97% carbon optimality on average for
    <24H job lengths (90% <96H, 85% <168H).
  * The heuristic reduces additional emissions vs regular LiteCast.
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
from src.metrics import concordance_index, additional_emissions  # noqa: E402
from src.scheduler import schedule_continuous, schedule_interruptible  # noqa: E402


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
    retrain_every = cfg.get("retrain_every", 6)
    only_short = cfg.get("only_short_jobs", True)

    fc = LiteCast(train_days=train_days, horizon=horizon,
                  use_weather=cfg.get("use_weather", True),
                  use_demand=cfg.get("use_demand", True),
                  use_time=cfg.get("use_time", True))

    test_start = pd.Timestamp(cfg["test_start"])
    test_end = pd.Timestamp(cfg["test_end"])
    origins = pd.date_range(start=test_start, end=test_end, freq=f"{step_days}D")
    max_origin = ci.index.max() - pd.Timedelta(hours=horizon)
    origins = origins[origins <= max_origin]
    max_T = max(slacks)

    agg = {}
    for name in ("regular", "heuristic"):
        agg[name] = {}
        for (L, T, m) in [(L, T, m) for L in job_lengths for T in slacks for m in modes]:
            agg[name][(L, T, m)] = {"e_pred": 0.0, "e_oracle": 0.0, "n": 0}

    for origin in origins:
        hist = ci[ci.index <= origin]
        if len(hist) < train_days * 24:
            continue
        future_idx = pd.date_range(start=hist.index[-1] + pd.Timedelta(hours=1),
                                   periods=horizon, freq="h")
        a = ci.reindex(future_idx)
        if a.notna().sum() < 24:
            continue

        try:
            base_fc = fc.forecast(hist, da).reindex(future_idx)
        except Exception as e:  # noqa: BLE001
            print(f"  [{region}] forecast failed at {origin}: {e}", flush=True)
            continue

        for (L, T, m) in agg["regular"]:
            if a[:T + L].notna().sum() < T + L:
                continue
            if m == "continuous":
                h_regular = schedule_continuous(base_fc, L, T)
                h_oracle = schedule_continuous(a, L, T)
                e_pred = a.iloc[h_regular:h_regular + L].sum()
                e_oracle = a.iloc[h_oracle:h_oracle + L].sum()
            else:
                w = min(len(base_fc), T + L)
                order = np.argsort(base_fc.values[:w], kind="stable")
                h_regular = np.sort(order[:L])
                h_oracle = np.sort(np.argsort(a.values[:w], kind="stable")[:L])
                e_pred = a.iloc[h_regular].sum()
                e_oracle = a.iloc[h_oracle].sum()
            agg["regular"][(L, T, m)]["e_pred"] += e_pred
            agg["regular"][(L, T, m)]["e_oracle"] += e_oracle
            agg["regular"][(L, T, m)]["n"] += 1

        # heuristic: retrain once per candidate wait time, then evaluate all jobs
        retrain_times = list(range(1, min(max_T, horizon), retrain_every))
        retrain_fcs = {}
        for t in retrain_times:
            new_hist = ci[ci.index <= origin + pd.Timedelta(hours=t)]
            if len(new_hist) < train_days * 24:
                continue
            try:
                retrain_fcs[t] = fc.forecast(new_hist, da).reindex(future_idx)
            except Exception:  # noqa: BLE001
                continue

        for (L, T, m) in agg["heuristic"]:
            if a[:T + L].notna().sum() < T + L:
                continue
            if only_short and L >= 24:
                # reuse the regular result for long jobs (heuristic marginal)
                agg["heuristic"][(L, T, m)]["e_pred"] += agg["regular"][(L, T, m)]["e_pred"]
                agg["heuristic"][(L, T, m)]["e_oracle"] += agg["regular"][(L, T, m)]["e_oracle"]
                agg["heuristic"][(L, T, m)]["n"] += 1
                continue
            # base schedule: for continuous keep start hour, for interruptible keep the set
            if m == "continuous":
                h = schedule_continuous(base_fc, L, T)
                e_pred = a.iloc[h:h + L].sum()
            else:
                w = min(len(base_fc), T + L)
                order = np.argsort(base_fc.values[:w], kind="stable")
                h = np.sort(order[:L])
                e_pred = a.iloc[h].sum()
            h_oracle = schedule_continuous(a, L, T) if m == "continuous" else \
                np.sort(np.argsort(a.values[:min(len(a), T + L)], kind="stable")[:L])
            e_oracle = a.iloc[h_oracle:h_oracle + L].sum() if m == "continuous" else a.iloc[h_oracle].sum()

            # Algorithm 1: update schedule while job waits
            best_ci = concordance_index(a, base_fc)
            best_fc = base_fc
            chosen = h if m == "interruptible" else None
            for t in retrain_fcs:
                if t >= T + L:
                    continue
                new_fc = retrain_fcs[t]
                new_ci = concordance_index(a, new_fc)
                if new_ci > best_ci:
                    if m == "continuous":
                        cand = schedule_continuous(new_fc, L, T)
                        cur_cost = best_fc.iloc[h:h + L].sum()
                        cand_cost = new_fc.iloc[cand:cand + L].sum()
                        if cand_cost < cur_cost:
                            best_fc = new_fc
                            h = cand
                            best_ci = new_ci
                    else:
                        cand = np.sort(np.argsort(
                            new_fc.values[:min(len(new_fc), T + L)], kind="stable")[:L])
                        cur_cost = best_fc.iloc[chosen].sum()
                        cand_cost = new_fc.iloc[cand].sum()
                        if cand_cost < cur_cost:
                            best_fc = new_fc
                            chosen = cand
                            best_ci = new_ci
            if m == "continuous":
                e_pred_heuristic = a.iloc[h:h + L].sum()
            else:
                e_pred_heuristic = a.iloc[chosen].sum()
            agg["heuristic"][(L, T, m)]["e_pred"] += e_pred_heuristic
            agg["heuristic"][(L, T, m)]["e_oracle"] += e_oracle
            agg["heuristic"][(L, T, m)]["n"] += 1

    out = {"region": region}
    out["additional"] = {}
    out["n_jobs"] = {}
    for name in ("regular", "heuristic"):
        for (L, T, m), v in agg[name].items():
            eo = v["e_oracle"]
            out["additional"][(name, L, T, m)] = additional_emissions(v["e_pred"], eo) if eo > 0 else float("nan")
            out["n_jobs"][(name, L, T, m)] = v["n"]
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

    rows = []
    for r in results:
        for (name, L, T, m), v in r["additional"].items():
            rows.append({"region": r["region"], "forecaster": name, "L": L, "T": T,
                         "mode": m, "additional%": v})
    sched = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("LiteCast regular vs + Heuristic (Algorithm 1)")
    print(f"config: lengths={cfg.get('job_lengths')} slacks={cfg.get('slacks')} "
          f"modes={cfg.get('modes')} retrain_every={cfg.get('retrain_every')}h")
    print("=" * 78)
    print("\nAdditional emissions vs oracle (%) — mean across regions:")
    piv = sched.pivot_table(index=["mode", "L", "T"], columns="forecaster",
                            values="additional%", aggfunc="mean").round(1)
    print(piv.to_string())

    print("\nHeadline aggregate:")
    for mode in cfg.get("modes", []):
        sub = sched[sched["mode"] == mode]
        for name in ("regular", "heuristic"):
            mean_add = sub[sub.forecaster == name]["additional%"].mean()
            print(f"  {mode} {name}: mean additional emissions {mean_add:.1f}% "
                  f"(optimality {100 + mean_add:.1f}%)")
    # <24H jobs headline
    for name in ("regular", "heuristic"):
        sub = sched[sched["forecaster"] == name]
        for label, mask in [("L<24H", sub["L"] < 24), ("L<96H", sub["L"] < 96), ("all", sub["L"] < 168)]:
            mean_add = sub[mask]["additional%"].mean()
            print(f"  {name} {label}: mean additional {mean_add:.1f}% -> carbon optimality {100 + mean_add:.1f}%")
    print("=" * 78)

    out = cfg.get("out_csv")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        sched.to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
