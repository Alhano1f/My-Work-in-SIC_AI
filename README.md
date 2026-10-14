# House Prices | EDA, Data Cleaning & Preprocessing

Reproducible exploratory data analysis and leakage-aware data preparation for Indian house listings.

## What is included
- `notebooks/01_house_prices_eda.ipynb`: executable walkthrough of EDA, cleaning, preprocessing.
- `src/process.py`: reusable data-cleaning and train-only preprocessing code.
- `reports/EDA_REPORT.md` and `reports/figures/`: findings and plots.
- `data/house_prices_cleaned.csv`: cleaned/engineered columns with original listings retained.

## Run locally

```bash
pip install -r requirements.txt
python src/process.py
jupyter notebook notebooks/01_house_prices_eda.ipynb
```

Put the original dataset at `data/house_prices.csv` before running the script. **The raw file is intentionally excluded from Git (about 102 MB).** The cleaned CSV also stays out of Git if it is too large to upload normally; get it from the downloadable project package.

## Processing approach
Prices are converted from `Lac`/`Cr` into INR. Area units are normalized where recognized. Unusable and unspecified numeric values stay missing in cleaned data. Suspicious duplicates are flagged, not removed, because no reliable property identifier exists. For model preparation, missing-value imputation, numeric scaling and one-hot encoding are fitted on training data only. Price-per-square-foot is not used as a predictor because it could leak the target.

See [full EDA report](reports/EDA_REPORT.md) and [processing summary](reports/processing_summary.txt).
