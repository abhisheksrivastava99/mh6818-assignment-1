# MH6818 Assignment 1

Streamlit app and supporting assets for an AI credit decisioning system built for the MH6818 FinTech assignment. The app scores applications, applies decision rules, and shows fairness monitoring outputs using the model and local data files stored in this repository.

## Entrypoint
- Live link: https://amh6818-assignment-1-abhishek-srivastava.streamlit.app/
- Streamlit app: `app.py`
- Default branch for deployment: `main`

## Run Locally

1. Activate the virtual environment if needed.
2. Start the app:

```bash
venv/bin/streamlit run app.py
```

## Deployment Coordinates

- GitHub repository: `mh6818-assignment-1`
- Streamlit Community Cloud branch: `main`
- Streamlit entrypoint file: `app.py`

## Required Runtime Assets

The app expects these files to be present in the repository at runtime:

- `credit_default_threshold_xgb_bundle.joblib`
- `fairness_summary.json`
- `data/new-applications.csv`

