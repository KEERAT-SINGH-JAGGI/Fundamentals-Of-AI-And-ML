import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("   MOBILE ADDICTION PREDICTION – ML Project")
print("=" * 60)

print("\n[1] Loading Dataset …")
df = pd.read_csv("/Users/keeratsinghjaggi/Documents/python/aiml project/mobile_addiction_data.csv")
print(f"    Shape          : {df.shape}")
print(f"    Columns        : {list(df.columns)}")
print(f"\n    First 5 rows:\n{df.head()}")

print("\n[2] Exploratory Data Analysis …")
print("\n  Basic Statistics:")
print(df.describe())
print("\n  Missing values per column:")
print(df.isnull().sum())
print(f"\n  Class distribution:\n{df['Addicted'].value_counts()}")
print("  (0 = Not Addicted, 1 = Addicted)")

print("\n[3] Data Preprocessing …")
if df["Anxiety_Without_Phone"].dtype == object:
    le = LabelEncoder()
    df["Anxiety_Without_Phone"] = le.fit_transform(df["Anxiety_Without_Phone"])
    print("    Encoded 'Anxiety_Without_Phone': Yes→1, No→0")
else:
    print("    'Anxiety_Without_Phone' is already numeric – no encoding needed.")

X = df.drop(columns=["Addicted"])
y = df["Addicted"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("    Feature scaling (StandardScaler) applied.")
print(f"    Features used: {list(X.columns)}")

print("\n[4] Train-Test Split …")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

print("\n[5] Training Models …")
lr_model = LogisticRegression(random_state=42, max_iter=200)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("    Logistic Regression → trained ✔")
print("    Decision Tree       → trained ✔")

print("\n[6] Model Evaluation …")
acc_lr = accuracy_score(y_test, y_pred_lr) * 100
acc_dt = accuracy_score(y_test, y_pred_dt) * 100

print(f"\n  Logistic Regression Accuracy : {acc_lr:.2f}%")
print(f"  Decision Tree       Accuracy : {acc_dt:.2f}%")

print("\n  ── Logistic Regression Report ──")
print(
    classification_report(
        y_test, y_pred_lr, target_names=["Not Addicted", "Addicted"]
    )
)

print("  ── Decision Tree Report ──")
print(
    classification_report(
        y_test, y_pred_dt, target_names=["Not Addicted", "Addicted"]
    )
)

print("  ── Decision Tree Rules (top levels) ──")
feature_names = list(X.columns)
tree_rules = export_text(dt_model, feature_names=feature_names, max_depth=3)
print(tree_rules)

print("\n[7] Generating Graphs …")
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#4CAF50", "#F44336"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Graph 1 – Class Distribution", fontsize=14, fontweight="bold")
counts = df["Addicted"].value_counts()
labels = ["Not Addicted (0)", "Addicted (1)"]

axes[0].pie(
    counts,
    labels=labels,
    autopct="%1.1f%%",
    colors=COLORS,
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
axes[0].set_title("Pie Chart")

sns.countplot(
    x="Addicted", data=df, palette=COLORS, ax=axes[1], hue="Addicted", legend=False
)
axes[1].set_xticklabels(["Not Addicted", "Addicted"])
axes[1].set_title("Bar Chart")
axes[1].set_xlabel("Addiction Status")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("outputs/graph1_class_distribution.png", dpi=150)
plt.close()
print("    ✔ Graph 1 saved: class distribution")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Graph 2 – Feature Distributions", fontsize=14, fontweight="bold")
axes = axes.flatten()

for idx, col in enumerate(X.columns):
    axes[idx].hist(df[col], bins=15, color="#5C6BC0", edgecolor="white", alpha=0.85)
    axes[idx].set_title(col.replace("_", " "), fontsize=10)
    axes[idx].set_xlabel("Value")
    axes[idx].set_ylabel("Frequency")

if len(X.columns) < len(axes):
    axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/graph2_feature_distributions.png", dpi=150)
plt.close()
print("    ✔ Graph 2 saved: feature distributions")

fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    linewidths=0.5,
    ax=ax,
    square=True,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Graph 3 – Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/graph3_correlation_heatmap.png", dpi=150)
plt.close()
print("    ✔ Graph 3 saved: correlation heatmap")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    "Graph 4 – Feature vs Addiction Status (Boxplots)", fontsize=14, fontweight="bold"
)
axes = axes.flatten()

box_features = [
    "Screen_Time_hrs",
    "Social_Media_hrs",
    "Gaming_Time_hrs",
    "Study_Time_hrs",
    "Sleep_Time_hrs",
    "Phone_Pickups_day",
]

for idx, feat in enumerate(box_features):
    sns.boxplot(
        x="Addicted",
        y=feat,
        data=df,
        palette=COLORS,
        ax=axes[idx],
        hue="Addicted",
        legend=False,
    )
    axes[idx].set_xticklabels(["Not Addicted", "Addicted"])
    axes[idx].set_title(feat.replace("_", " "), fontsize=10)
    axes[idx].set_xlabel("")

plt.tight_layout()
plt.savefig("outputs/graph4_boxplots.png", dpi=150)
plt.close()
print("    ✔ Graph 4 saved: boxplots")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Graph 5 – Confusion Matrices", fontsize=14, fontweight="bold")

for ax, y_pred_model, title in zip(
    axes, [y_pred_lr, y_pred_dt], ["Logistic Regression", "Decision Tree"]
):
    cm = confusion_matrix(y_test, y_pred_model)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not Addicted", "Addicted"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"{title}\nAccuracy: {accuracy_score(y_test, y_pred_model) * 100:.1f}%",
        fontsize=11,
    )

plt.tight_layout()
plt.savefig("outputs/graph5_confusion_matrices.png", dpi=150)
plt.close()
print("    ✔ Graph 5 saved: confusion matrices")

importances = dt_model.feature_importances_
feat_imp_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = [
    "#F44336" if i == feat_imp_df["Importance"].idxmax() else "#5C6BC0"
    for i in feat_imp_df.index
]
ax.barh(
    feat_imp_df["Feature"],
    feat_imp_df["Importance"],
    color=colors_bar,
    edgecolor="white",
)
ax.set_title(
    "Graph 6 – Feature Importance (Decision Tree)", fontsize=13, fontweight="bold"
)
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/graph6_feature_importance.png", dpi=150)
plt.close()
print("    ✔ Graph 6 saved: feature importance")

fig, ax = plt.subplots(figsize=(7, 5))
models = ["Logistic Regression", "Decision Tree"]
accs = [acc_lr, acc_dt]
bar_clr = ["#42A5F5", "#FFA726"]

bars = ax.bar(models, accs, color=bar_clr, edgecolor="white", width=0.45, zorder=3)
ax.set_ylim(0, 110)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Graph 7 – Model Accuracy Comparison", fontsize=13, fontweight="bold")
ax.grid(axis="y", zorder=0, alpha=0.4)

for bar, acc in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("outputs/graph7_model_accuracy_comparison.png", dpi=150)
plt.close()
print("    ✔ Graph 7 saved: model accuracy comparison")

fig, ax = plt.subplots(figsize=(8, 6))
for label_val, label_name, clr in [
    (0, "Not Addicted", "#4CAF50"),
    (1, "Addicted", "#F44336"),
]:
    subset = df[df["Addicted"] == label_val]
    ax.scatter(
        subset["Screen_Time_hrs"],
        subset["Social_Media_hrs"],
        label=label_name,
        color=clr,
        alpha=0.7,
        edgecolors="white",
        s=60,
    )

ax.set_xlabel("Daily Screen Time (hrs)")
ax.set_ylabel("Social Media Usage (hrs)")
ax.set_title("Graph 8 – Screen Time vs Social Media Usage", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/graph8_scatter_screentime_socialmedia.png", dpi=150)
plt.close()
print("    ✔ Graph 8 saved: scatter plot")

print("\n[8] Predicting on New Sample Inputs …")
print("-" * 48)

sample_inputs = {
    "Student A (Likely Addicted)": [9.5, 5.0, 3.0, 1.5, 5.0, 120, 1],
    "Student B (Likely Not Addicted)": [3.0, 1.0, 0.5, 6.0, 8.0, 25, 0],
    "Student C (Borderline)": [5.5, 3.0, 2.0, 3.0, 6.5, 65, 1],
}

for student, values in sample_inputs.items():
    arr = np.array(values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = lr_model.predict(arr_scaled)[0]
    prob = lr_model.predict_proba(arr_scaled)[0][1] * 100
    result = "ADDICTED" if pred == 1 else "NOT ADDICTED"
    print(f"  {student}")
    print(f"    Prediction : {result}  |  Confidence: {prob:.1f}%\n")

print("=" * 60)
print("  PROJECT SUMMARY")
print("=" * 60)
print(f"  Dataset size          : {len(df)} rows × {len(df.columns)} columns")
print(f"  Train/Test split      : 75% / 25%  ({len(X_train)}/{len(X_test)} samples)")
print(f"  Logistic Regression   : {acc_lr:.2f}% accuracy")
print(f"  Decision Tree         : {acc_dt:.2f}% accuracy")
print(
    f"  Best model            : {'Logistic Regression' if acc_lr >= acc_dt else 'Decision Tree'}"
)
print(f"\n  All graphs saved to   : outputs/")
print("=" * 60)
print("  Project execution complete!")
print("=" * 60)
