# Arsenal Match Intelligence

## Run with Anaconda

```bash
conda create -n arsenal-dashboard python=3.11 -y
conda activate arsenal-dashboard
pip install -r requirements_arsenal.txt
streamlit run arsenal_dashboard.py
```

The command opens a local browser dashboard. Enter a **newly rotated** football-data.org API key in the sidebar. Choose one or more competitions, select the seasons, and click **Download and train**. Downloads are cached in `arsenal_data/`. New completed results are fetched when **Refresh current season** is enabled and you retrain.

## Data access

Older seasons can return HTTP 403. That is a subscription restriction, not a Python bug. The dashboard stops requesting older seasons for that competition after a 403. Historical CSVs from another source would require a separate ingestion/normalisation pipeline; this version does **not** pretend to bypass API access.

## Limits

Only completed 90-minute matches without extra-time/shootout outcomes are included. Opponent selection includes only teams actually in downloaded results. Cup fixtures and team form are only available when the API permits the chosen competitions. Probabilities are experimental and not guaranteed to be calibrated.
