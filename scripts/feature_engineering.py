import pandas as pd
import numpy as np
import os

# ── LOAD ──────────────────────────────────────────────────
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "telemetry_processed.csv"))
df["speed_zone"] = pd.Categorical(df["speed_zone"],
    categories=["Rest","Walk","Jog","Run","Sprint"], ordered=True)

print(f"✅ Loaded {len(df):,} rows")

# ── KPI 1 : Performance Score ─────────────────────────────
# Weighted composite: speed + sprint contribution + low fatigue
df["performance_score"] = (
    (df["speed_kmh"]      / df["speed_kmh"].max())      * 0.35 +
    (df["sprint_flag"]                                 ) * 0.25 +
    (1 - df["fatigue_index"])                            * 0.20 +
    (df["acceleration"].clip(lower=0) /
     df["acceleration"].clip(lower=0).max())             * 0.20
).round(4)

# ── KPI 2 : High Intensity Zone % per player per match ───
hi_zones = ["Run", "Sprint"]
df["is_high_intensity"] = df["speed_zone"].isin(hi_zones).astype(int)

hi_pct = (df.groupby(["player_id","match_id"])["is_high_intensity"]
            .mean()
            .reset_index()
            .rename(columns={"is_high_intensity": "hi_intensity_pct"}))
hi_pct["hi_intensity_pct"] = (hi_pct["hi_intensity_pct"] * 100).round(2)

# ── KPI 3 : Speed Drop-off (fatigue signature) ───────────
# Compare avg speed first 15 mins vs last 15 mins per player per match
early = df[df["minute"] <= 15].groupby(["player_id","match_id"])["speed_kmh"].mean()
late  = df[df["minute"] >= 75].groupby(["player_id","match_id"])["speed_kmh"].mean()
speed_dropoff = ((early - late) / early * 100).reset_index()
speed_dropoff.columns = ["player_id","match_id","speed_dropoff_pct"]
speed_dropoff["speed_dropoff_pct"] = speed_dropoff["speed_dropoff_pct"].round(2)

# ── KPI 4 : Total Distance per player per match ──────────
total_dist = (df.groupby(["player_id","match_id"])["distance_m"]
                .sum()
                .reset_index()
                .rename(columns={"distance_m": "total_distance_m"}))
total_dist["total_distance_km"] = (total_dist["total_distance_m"] / 1000).round(2)

# ── KPI 5 : Sprint Count per player per match ────────────
sprint_count = (df.groupby(["player_id","match_id"])["sprint_flag"]
                  .sum()
                  .reset_index()
                  .rename(columns={"sprint_flag": "sprint_count"}))

# ── KPI 6 : Avg Injury Risk per player per match ─────────
avg_risk = (df.groupby(["player_id","match_id"])["injury_risk"]
              .mean()
              .reset_index()
              .rename(columns={"injury_risk": "avg_injury_risk"}))
avg_risk["avg_injury_risk"] = avg_risk["avg_injury_risk"].round(3)

# ── MERGE all KPIs into match-level summary ───────────────
from functools import reduce
kpi_frames = [hi_pct, speed_dropoff, total_dist, sprint_count, avg_risk]
match_summary = reduce(lambda l, r: pd.merge(l, r, on=["player_id","match_id"]), kpi_frames)

# Add player info back
player_info = df[["player_id","player_name","position","age"]].drop_duplicates()
match_summary = match_summary.merge(player_info, on="player_id")

# ── MERGE performance_score back into main df ─────────────
df_final = df.copy()

# ── SAVE ──────────────────────────────────────────────────
base = os.path.join(os.path.dirname(__file__), "..", "data")

df_final.to_csv(os.path.join(base, "telemetry_featured.csv"), index=False)
match_summary.to_csv(os.path.join(base, "match_kpi_summary.csv"), index=False)

print(f"✅ telemetry_featured.csv saved  — {len(df_final):,} rows")
print(f"✅ match_kpi_summary.csv saved   — {len(match_summary):,} rows")
print(f"\nKPI columns in match summary:")
print(match_summary.columns.tolist())
print(f"\nSample:")
print(match_summary.head(5).to_string())