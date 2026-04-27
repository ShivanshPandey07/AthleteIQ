import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import shap
import os
import warnings
warnings.filterwarnings("ignore")

# ── LOAD ──────────────────────────────────────────────────
base = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(base, "match_kpi_summary.csv"))

print(f"✅ Loaded match_kpi_summary — {len(df)} rows")

# ── TARGET : High Risk Label ──────────────────────────────
# Top 30% injury risk = high risk (label = 1)
threshold = df["avg_injury_risk"].quantile(0.70)
df["high_risk"] = (df["avg_injury_risk"] >= threshold).astype(int)
print(f"Threshold : {threshold:.3f}")
print(f"High risk : {df['high_risk'].sum()} / {len(df)} players")

# ── FEATURES ──────────────────────────────────────────────
le = LabelEncoder()
df["position_enc"] = le.fit_transform(df["position"])

FEATURES = [
    "hi_intensity_pct",
    "speed_dropoff_pct",
    "total_distance_km",
    "sprint_count",
    "age",
    "position_enc"
]

X = df[FEATURES]
y = df["high_risk"]

# ── TRAIN / TEST SPLIT ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nTrain size : {len(X_train)} | Test size : {len(X_test)}")

# ── MODEL ─────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=4,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ── EVALUATION ────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

print(f"\n{'='*45}")
print(f"  ROC-AUC  (test)  : {roc_auc_score(y_test, y_proba):.3f}")
print(f"  ROC-AUC  (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"{'='*45}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Low Risk","High Risk"]))

# ── PLOT 1 : Confusion Matrix ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Risk","High Risk"],
            yticklabels=["Low Risk","High Risk"],
            linewidths=0.5, ax=ax)
ax.set_title("Confusion matrix — injury risk model",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(base, "..", "reports", "plot_07_confusion_matrix.png"),
            bbox_inches="tight", dpi=130)
plt.show()
print("✅ plot_07 saved")

# ── PLOT 2 : Feature Importance ───────────────────────────
importance_df = pd.DataFrame({
    "feature":   FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(importance_df["feature"], importance_df["importance"],
               color="#378ADD", edgecolor="none", height=0.55)
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
ax.set_title("Feature importance — Random Forest",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Importance score")
ax.set_xlim(0, importance_df["importance"].max() * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(base, "..", "reports", "plot_08_feature_importance.png"),
            bbox_inches="tight", dpi=130)
plt.show()
print("✅ plot_08 saved")

# ── PLOT 3 : SHAP values ──────────────────────────────────
explainer  = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification shap_values is a list — take class 1
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(9, 5))
shap.summary_plot(sv, X_test, feature_names=FEATURES,
                  plot_type="dot", show=False)
plt.title("SHAP summary — what drives injury risk",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(base, "..", "reports", "plot_09_shap_summary.png"),
            bbox_inches="tight", dpi=130)
plt.show()
print("✅ plot_09 saved")

# ── SAVE predictions back to CSV ─────────────────────────
df["risk_probability"] = model.predict_proba(X)[:, 1].round(3)
df["risk_label"]       = model.predict(X)
df.to_csv(os.path.join(base, "match_kpi_with_predictions.csv"), index=False)
print(f"\n✅ match_kpi_with_predictions.csv saved — {len(df)} rows")
print("\nTop 5 highest risk predictions:")
print(df.sort_values("risk_probability", ascending=False)
        [["player_name","match_id","avg_injury_risk","risk_probability","risk_label"]]
        .head(5).to_string())