
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="HR Attrition Intelligence", layout="wide")

st.title("🧭 HR Attrition Intelligence — Streamlit Dashboard")

with st.expander("ℹ️ About this app", expanded=False):
    st.write(
        "This dashboard helps HR leaders explore retention risks, train three ML models "
        "(Decision Tree, Random Forest, Gradient Boosting) on attrition, and predict on new uploads. "
        "Nulls are handled expertly with imputers: median for numeric, most-frequent for categorical."
    )

# ---------------------------
# Helpers
# ---------------------------
DEFAULT_FILE = "hr_sample_attrition.csv"

@st.cache_data
def load_data(path_or_buf):
    df = pd.read_csv(path_or_buf)
    return df

def binarize_y(series):
    if series.dtype == "O":
        return series.map({"Yes":1, "No":0}).astype("Int64").astype(int)
    return series.astype(int)

def build_preprocessor(X):
    cat_cols = [c for c in X.columns if X[c].dtype == "O"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), num_cols)
    ])
    return pre, cat_cols, num_cols

def metrics_table(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    }

def get_feature_names(pre, X):
    # Safely reconstruct feature names after ColumnTransformer + OneHotEncoder
    cat_cols = [c for c in X.columns if X[c].dtype == "O"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    cat_transformer = pre.named_transformers_["cat"].named_steps["oh"]
    cat_features = list(cat_transformer.get_feature_names_out(cat_cols))
    all_features = cat_features + num_cols
    return all_features

# ---------------------------
# Sidebar: data source
# ---------------------------
st.sidebar.header("📁 Data Source")
use_sample = st.sidebar.toggle("Use included sample dataset", value=True)
if use_sample:
    src = DEFAULT_FILE
    st.sidebar.success("Using included sample file.")
else:
    up = st.sidebar.file_uploader("Upload your HR CSV", type=["csv"])
    if up is None:
        st.sidebar.info("Upload a CSV or toggle sample dataset.")
        st.stop()
    src = up

df = load_data(src)

required_base = {"Attrition", "JobRole", "JobSatisfaction"}
if not required_base.issubset(df.columns):
    st.error("Dataset must include at least these columns: Attrition, JobRole, JobSatisfaction.")
    st.stop()

# ---------------------------
# Filters (apply to all charts)
# ---------------------------
st.sidebar.header("🔎 Filters")
job_roles = sorted(df["JobRole"].dropna().unique().tolist())
selected_roles = st.sidebar.multiselect("Job Role", job_roles, default=job_roles)

# Handle potential non-integer satisfaction scales by coercing to numeric then finding range
sat_series = pd.to_numeric(df["JobSatisfaction"], errors="coerce")
sat_min = int(np.nanmin(sat_series))
sat_max = int(np.nanmax(sat_series))
sat_sel = st.sidebar.slider("Job Satisfaction (min–max)", min_value=sat_min, max_value=sat_max, value=(sat_min, sat_max))

fdf = df.copy()
fdf = fdf[fdf["JobRole"].isin(selected_roles)]
fdf = fdf[(pd.to_numeric(fdf["JobSatisfaction"], errors="coerce") >= sat_sel[0]) &
          (pd.to_numeric(fdf["JobSatisfaction"], errors="coerce") <= sat_sel[1])]

st.caption(f"Filtered rows: {len(fdf):,} / {len(df):,}")

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Insights Dashboard", "🤖 Modeling (3 Algorithms)", "📤 Predict on New Data"])

# ---------------------------
# Tab 1: Insights (5 complex charts)
# ---------------------------
with tab1:
    st.subheader("Actionable Retention Insights")

    # Prepare attrition as 0/1 for rates
    fdf["_Attr01"] = fdf["Attrition"].map({"Yes":1,"No":0})

    c1, c2 = st.columns(2)
    with c1:
        # 1) Sorted bar: Attrition rate by Job Role
        rate_by_role = (fdf.groupby("JobRole")["_Attr01"].mean()
                          .sort_values(ascending=False).reset_index())
        fig1 = px.bar(rate_by_role, x="JobRole", y="_Attr01",
                      title="Attrition Rate by Job Role", text_auto=".1%")
        fig1.update_layout(xaxis_title="", yaxis_title="Attrition Rate", bargap=0.3)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        # 2) Satisfaction vs Income colored by Attrition with OLS line
        if {"JobSatisfaction","MonthlyIncome","Attrition"}.issubset(fdf.columns):
            fig2 = px.scatter(
                fdf, x="JobSatisfaction", y="MonthlyIncome", color="Attrition",
                trendline="ols", title="Job Satisfaction vs Monthly Income (by Attrition)"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # 3) Heatmap of correlations (numeric only)
    num_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
    if len(num_cols) >= 2:
        corr = fdf[num_cols].corr().round(2)
        fig3 = px.imshow(corr, text_auto=True, title="Correlation Heatmap — Numeric Features")
        st.plotly_chart(fig3, use_container_width=True)

    # 4) Stacked bar: OverTime*Attrition within top JobRoles
    if {"OverTime","Attrition"}.issubset(fdf.columns) and not rate_by_role.empty:
        top_roles = rate_by_role["JobRole"].head(6).tolist()
        sdf = fdf[fdf["JobRole"].isin(top_roles)]
        agg = (sdf.groupby(["JobRole","OverTime","Attrition"]).size()
                 .reset_index(name="Count"))
        fig4 = px.bar(agg, x="JobRole", y="Count", color="Attrition", barmode="stack",
                      facet_col="OverTime", title="OverTime & Attrition by Top Job Roles")
        st.plotly_chart(fig4, use_container_width=True)

    # 5) Tenure cohort line: Attrition rate across tenure buckets
    if "YearsAtCompany" in fdf.columns:
        cuts = pd.cut(pd.to_numeric(fdf["YearsAtCompany"], errors="coerce"),
                      bins=[-1,1,3,5,8,12,40],
                      labels=["<=1","2-3","4-5","6-8","9-12","13+"])
        tmp = fdf.assign(Tenure=cuts)
        rate = tmp.groupby("Tenure")["_Attr01"].mean().reset_index()
        fig5 = px.line(rate, x="Tenure", y="_Attr01", markers=True,
                       title="Attrition Rate by Tenure Cohorts")
        fig5.update_layout(yaxis_title="Attrition Rate")
        st.plotly_chart(fig5, use_container_width=True)

# ---------------------------
# Tab 2: Modeling
# ---------------------------
with tab2:
    st.subheader("Train & Evaluate — Decision Tree / Random Forest / Gradient Boosting")
    st.write("Click **Run Models** to train on the filtered dataset (with imputation) and view metrics, ROC, confusion matrices, and feature importances.")
    run = st.button("▶️ Run Models")

    if run:
        # Drop obvious ID-like columns
        drop_like = [c for c in fdf.columns if c.lower().endswith("id")]
        base = fdf.drop(columns=drop_like) if drop_like else fdf.copy()

        if "Attrition" not in base.columns:
            st.error("No 'Attrition' column found after filtering/cleaning.")
        else:
            y = binarize_y(base["Attrition"])
            X = base.drop(columns=["Attrition"])

            pre, cat_cols, num_cols = build_preprocessor(X)
            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            rows = []
            roc_fig = go.Figure()
            cm_cols = st.columns(3)
            fi_cols = st.columns(3)

            for i, (name, est) in enumerate(models.items()):
                pipe = Pipeline([("pre", pre), ("clf", est)])
                pipe.fit(X_train, y_train)

                # Predictions
                yhat = pipe.predict(X_test)
                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    ypr = pipe.predict_proba(X_test)[:,1]
                else:
                    dec = pipe.decision_function(X_test)
                    ypr = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

                # Metrics
                m = metrics_table(y_test, yhat, ypr)
                m["Model"] = name

                # 5-fold CV Accuracy (stratified)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
                m["CV_Accuracy_Mean"] = float(np.mean(cv_scores))
                m["CV_Accuracy_SD"] = float(np.std(cv_scores))
                rows.append(m)

                # ROC
                fpr, tpr, _ = roc_curve(y_test, ypr)
                aucv = auc(fpr, tpr)
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={aucv:.3f})"))

                # Confusion Matrix
                cm = confusion_matrix(y_test, yhat)
                with cm_cols[i]:
                    cm_fig = px.imshow(cm, text_auto=True, title=f"{name} — Confusion Matrix (Test)")
                    cm_fig.update_xaxes(title_text="Predicted")
                    cm_fig.update_yaxes(title_text="Actual")
                    st.plotly_chart(cm_fig, use_container_width=True)

                # Feature Importances (if available)
                try:
                    # Refit preprocessor to training set to get feature names
                    pre.fit(X_train)
                    feat_names = get_feature_names(pre, X_train)
                    clf = pipe.named_steps["clf"]
                    if hasattr(clf, "feature_importances_"):
                        importances = pd.DataFrame({
                            "feature": feat_names,
                            "importance": clf.feature_importances_
                        }).sort_values("importance", ascending=False).head(15)
                        with fi_cols[i]:
                            fi_fig = px.bar(importances, x="importance", y="feature", orientation="h",
                                            title=f"{name} — Top 15 Feature Importances")
                            st.plotly_chart(fi_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute feature importances for {name}: {e}")

            # Metrics table
            resdf = pd.DataFrame(rows)[["Model","Accuracy","Precision","Recall","F1","AUC","CV_Accuracy_Mean","CV_Accuracy_SD"]]
            st.dataframe(resdf.style.format({c:"{:.3f}" for c in resdf.columns if c != "Model"}), use_container_width=True)

            # ROC plot
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
            roc_fig.update_layout(title="ROC Curves — Test Set", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(roc_fig, use_container_width=True)

# ---------------------------
# Tab 3: Predict on New Data
# ---------------------------
with tab3:
    st.subheader("Upload, Score, and Download Predictions")
    st.write("Upload a dataset with the same schema (feature columns). If it contains an 'Attrition' column, it will be ignored during prediction.")

    pu = st.file_uploader("Upload CSV to score", type=["csv"])
    if pu is not None:
        new_df = pd.read_csv(pu)

        # Train a robust model on *current filtered* data first
        base = fdf.copy()
        # Drop typical ID-like fields
        drop_like = [c for c in base.columns if c.lower().endswith("id")]
        base = base.drop(columns=drop_like) if drop_like else base

        y = binarize_y(base["Attrition"])
        X = base.drop(columns=["Attrition"])

        pre, cat_cols, num_cols = build_preprocessor(X)
        model = GradientBoostingClassifier(random_state=42)  # strong default
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(X, y)

        # Align / handle unseen columns by relying on OneHotEncoder(handle_unknown="ignore")
        pred_df = new_df.copy()
        if "Attrition" in pred_df.columns:
            pred_df = pred_df.drop(columns=["Attrition"])

        # Predict proba if available
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            proba = pipe.predict_proba(pred_df)[:,1]
        else:
            dec = pipe.decision_function(pred_df)
            proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

        label = np.where(proba >= 0.5, "Yes", "No")
        out = new_df.copy()
        out["Attrition_Pred"] = label
        out["Attrition_Prob"] = proba.round(4)

        st.dataframe(out.head(50), use_container_width=True)

        # Download button
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions as CSV", data=csv_bytes, file_name="predictions_with_labels.csv", mime="text/csv")
