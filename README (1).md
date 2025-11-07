
# HR Attrition Intelligence — Streamlit App

A ready-to-deploy Streamlit dashboard to explore retention insights, train 3 ML models (Decision Tree, Random Forest, Gradient Boosting),
and predict attrition on new uploads. Nulls are handled with **median (numeric)** and **most-frequent (categorical)** imputers.

## Files
- `app.py` — main Streamlit app
- `requirements.txt` — lightweight dependencies (no pinned versions)
- `hr_sample_attrition.csv` — sample dataset
- `README.md` — this file

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Create a new GitHub repo and upload these four files to the repo **root** (no folders).
2. On Streamlit Cloud, select the repo and set the entry point to `app.py`.
3. Optional: Add `SECRETS` if your data source needs credentials (not required for the sample).

## Usage
- Use the sidebar to switch between the included sample dataset and your own CSV.
- Apply **Job Role** multiselect and **Job Satisfaction** slider filters; they affect all charts.
- **Modeling tab**: click **Run Models** to train DT/RF/GBT with imputers, view metrics (incl. 5-fold CV), ROC, confusion matrices, and feature importances.
- **Predict tab**: upload a dataset to get `Attrition_Pred` and `Attrition_Prob`, and download the labeled file.

## Notes
- Expect required base columns: `Attrition`, `JobRole`, `JobSatisfaction`. Additional columns improve modeling.
- ID-like columns ending with `id` or `ID` are auto-dropped for modeling.
- Categorical handling uses `OneHotEncoder(handle_unknown="ignore")` for robust scoring on new data.
