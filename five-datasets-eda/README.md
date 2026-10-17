# 📊 Five Datasets — EDA, Audit & Corrective EDA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alhano1f/My-Work-in-SIC_AI/blob/main/five-datasets-eda/five_datasets_eda.ipynb)

Week 2 project from the SIC AI course: exploratory data analysis of five CSV datasets, followed by a reproducibility audit and a corrective EDA pass.

## 📁 Datasets

| Dataset | File | Rows | Columns | Missing cells | Duplicate rows |
|---------|------|-----:|--------:|--------------:|---------------:|
| Boston | `data_boston.csv` | 506 | 14 | 0 | 0 |
| Census | `data_census.csv` | 281 | 6 | 0 | 1 |
| Coffee Shop | `data_coffeeshop.csv` | 43,182 | 3 | 57 | 11,747 (27.2%) |
| Sales | `data_sales.csv` | 43 | 5 | 0 | 0 |
| Student List | `data_studentlist.csv` | 17 | 8 | 0 | 0 |

## 🔍 What the notebook does

1. **Load the data:** reads the five CSV files and defines one shared table style.
2. **Exploratory analysis:** one card per dataset with size, missing values, duplicates, a column overview and descriptive statistics.
3. **Audit (loading & reproducibility):** re-reads every CSV from disk and recomputes each statistic with NumPy to confirm the pandas results. It also flags Census `Population` and `Households` as numbers stored as text.
4. **Corrective EDA:** fixes what the audit found, covering data types, mode, frequencies, outliers, correlations, group comparisons and preprocessing notes. It works on copies, and a final integrity check confirms the original DataFrames still match the files on disk.

## 🚀 How to run

1. Open `five_datasets_eda.ipynb` in Google Colab (badge above) or Jupyter.
2. Upload the five CSV files listed above. The notebook reads them from `/content/`, Colab's default folder.
3. Run all cells from top to bottom.

## 🧰 Tech stack

Python · pandas · NumPy · Jupyter / Google Colab
