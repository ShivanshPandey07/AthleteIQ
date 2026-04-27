import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────
PLAYERS = {
    "P01": {"name": "Arjun Sharma",    "position": "Forward",    "age": 24},
    "P02": {"name": "Rohit Verma",     "position": "Midfielder",  "age": 27},
    "P03": {"name": "Dev Patel",       "position": "Defender",    "age": 22},
    "P04": {"name": "Karan Singh",     "position": "Forward",    "age": 26},
    "P05": {"name": "Nikhil Rao",      "position": "Midfielder",  "age": 29},
    "P06": {"name": "Amit Joshi",      "position": "Defender",    "age": 23},
    "P07": {"name": "Siddharth Nair",  "position": "Goalkeeper",  "age": 28},
    "P08": {"name": "Vikas Mehta",     "position": "Forward",    "age": 25},
    "P09": {"name": "Rahul Das",       "position": "Midfielder",  "age": 30},
    "P10": {"name": "Pranav Iyer",     "position": "Defender",    "age": 21},
}

MATCHES = [f"M{str(i).zfill(2)}" for i in range(1, 16)]  # 15 matches
MATCH_DURATION = 90   # minutes
SAMPLE_RATE    = 1    # one row per minute per player

POSITION_SPEED = {
    "Forward":    {"base": 7.2, "std": 1.8},
    "Midfielder": {"base": 6.8, "std": 1.5},
    "Defender":   {"base": 5.9, "std": 1.4},
    "Goalkeeper": {"base": 2.1, "std": 0.8},
}

# ── HELPERS ───────────────────────────────────────────────
def fatigue_curve(minute):
    """Returns a 0-1 multiplier: starts at 1, dips after 45, recovers slightly."""
    if minute <= 45:
        return 1.0 - 0.003 * minute
    else:
        recovery = 0.05 * np.exp(-0.05 * (minute - 45))
        return 0.865 - 0.004 * (minute - 45) + recovery

def simulate_player_match(player_id, match_id, player_info):
    pos   = player_info["position"]
    age   = player_info["age"]
    speed_cfg = POSITION_SPEED[pos]

    age_penalty = max(0, (age - 26) * 0.02)   # older → slightly slower
    injury_risk_base = 0.1 + age_penalty

    rows = []
    for minute in range(1, MATCH_DURATION + 1):
        f = fatigue_curve(minute)

        speed = max(0, np.random.normal(
            speed_cfg["base"] * f - age_penalty,
            speed_cfg["std"]
        ))

        heart_rate = int(np.clip(
            np.random.normal(140 + speed * 5, 8) * (1 + (1 - f) * 0.15),
            60, 200
        ))

        acceleration = np.round(np.random.normal(0.8 * f, 0.4), 2)
        distance_m   = np.round(speed * 60, 1)        # metres per minute
        sprint_flag  = int(speed > 7.0)

        # Injury risk: high workload + high fatigue + age
        injury_risk = np.clip(
            injury_risk_base + (1 - f) * 0.3 + (speed / 12) * 0.2
            + np.random.normal(0, 0.05),
            0, 1
        )

        rows.append({
            "player_id":      player_id,
            "player_name":    player_info["name"],
            "position":       pos,
            "age":            age,
            "match_id":       match_id,
            "minute":         minute,
            "speed_kmh":      round(speed, 2),
            "heart_rate_bpm": heart_rate,
            "acceleration":   acceleration,
            "distance_m":     distance_m,
            "sprint_flag":    sprint_flag,
            "fatigue_index":  round(1 - f, 3),
            "injury_risk":    round(injury_risk, 3),
        })
    return rows

# ── MAIN ──────────────────────────────────────────────────
all_rows = []
for match_id in MATCHES:
    for player_id, info in PLAYERS.items():
        all_rows.extend(simulate_player_match(player_id, match_id, info))

df = pd.DataFrame(all_rows)

# Derived columns
df["speed_zone"] = pd.cut(
    df["speed_kmh"],
    bins=[-1, 2, 4, 6, 8, 30],
    labels=["Rest", "Walk", "Jog", "Run", "Sprint"]
)
df["half"] = df["minute"].apply(lambda m: "First" if m <= 45 else "Second")

# Save
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "telemetry_raw.csv")
df.to_csv(out_path, index=False)
print(f"✅ Saved {len(df):,} rows → data/telemetry_raw.csv")
print(df.head())
print("\nColumn summary:")
print(df.dtypes)