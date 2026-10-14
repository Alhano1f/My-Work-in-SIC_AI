"""Reproducible, leakage-aware House Prices dataset cleaning and preprocessing."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'house_prices.csv'

AREA_FACTORS = {'sqft': 1, 'sqyrd': 9, 'sqm': 10.76391041671, 'acre': 43560,
                'hectare': 107639.1041671, 'cent': 435.6, 'marla': 272.25,
                'kanal': 5445, 'ground': 2400}

def price_inr(value):
    if pd.isna(value):
        return np.nan
    v = re.sub(r'[,₹\s]', '', str(value)).lower()
    m = re.fullmatch(r'(\d+(?:\.\d+)?)(cr|crore|lac|lakh|lakhs|lacs|k|thousand)?', v)
    if not m:
        return np.nan
    return float(m[1]) * {'cr':1e7, 'crore':1e7, 'lac':1e5, 'lakh':1e5,
                           'lakhs':1e5,'lacs':1e5, 'k':1e3, 'thousand':1e3,
                           None:1}[m[2]]

def area_sqft(value):
    if pd.isna(value): return np.nan
    m = re.fullmatch(r'\s*([\d,.]+)\s*([a-zA-Z]+)\s*', str(value))
    if not m: return np.nan
    unit = m[2].lower()
    if unit not in AREA_FACTORS: return np.nan
    return float(m[1].replace(',','')) * AREA_FACTORS[unit]

def count_value(value):
    if pd.isna(value): return np.nan
    s = str(value).strip()
    m = re.fullmatch(r'(\d+)(?:\.0)?', s)
    return float(m[1]) if m else np.nan # '> 10' must not be mistaken for exactly 10

def clean(df):
    out = df.copy()
    out.columns = out.columns.str.strip()
    out = out.drop(columns=['Plot Area','Dimensions','Index'], errors='ignore')
    out['possible_duplicate_listing'] = out.duplicated(keep=False)
    out = out.reset_index(drop=True)
    out['price_inr'] = out['Amount(in rupees)'].map(price_inr)
    out['price_per_sqft_inr'] = pd.to_numeric(out['Price (in rupees)'],errors='coerce')
    for col in ['Carpet Area','Super Area']:
        out[col.lower().replace(' ','_')+'_sqft'] = out[col].map(area_sqft)
    for col in ['Bathroom','Balcony']:
        out[col.lower()+'_count'] = out[col].map(count_value)
    out['bedrooms'] = pd.to_numeric(out['Title'].str.extract(r'\b(\d+)\s*BHK\b',flags=re.I,expand=False),errors='coerce')
    floors = out['Floor'].astype('string').str.extract(r'(?i)^\s*(\d+)\s+out of\s+(\d+)\s*$')
    out['floor_number'] = pd.to_numeric(floors[0],errors='coerce')
    out['total_floors'] = pd.to_numeric(floors[1],errors='coerce')
    out['parking_count'] = pd.to_numeric(out['Car Parking'].astype('string').str.extract(r'\b(\d+)\s+(?:Covered|Open)\b',flags=re.I,expand=False),errors='coerce')
    out['location'] = out['location'].astype('string').str.strip().str.lower()
    for col in ['Furnishing','Transaction','Ownership','facing','overlooking']:
        out[col] = out[col].astype('string').str.strip().replace('',pd.NA)
    # Treat high-cardinality/free text as exploratory only. In particular,
    # never include the rupees/sqft listing price in the feature matrix.
    return out

def main():
    if not RAW.exists(): raise FileNotFoundError(f'Put original CSV at {RAW}')
    df = pd.read_csv(RAW, low_memory=False)
    before = len(df)
    cleaned = clean(df)
    (ROOT/'data').mkdir(exist_ok=True)
    cleaned.to_csv(ROOT/'data'/'house_prices_cleaned.csv',index=False)
    report = [f'Input rows: {before:,}',f'After duplicate check: {len(cleaned):,}',
              f'Rows with positive numeric price: {(cleaned.price_inr>0).sum():,}',
              f'Price conversion missing: {cleaned.price_inr.isna().sum():,}',
              f'Numeric price median: ₹{cleaned.price_inr.median():,.0f}',
              f'Potential duplicate rows excluding the Index: {cleaned.possible_duplicate_listing.sum():,}',
              'No rows dropped for possible duplicates without a stable listing ID.',
              'Unknown area units left missing, not guessed.',
              'Missing target prices excluded from modeling, retained in cleaned file.',
              'No missing input values globally imputed: imputation/scaling fit on train only.']
    feature_cols = ['location','Furnishing','Transaction','Ownership','facing',
                    'carpet_area_sqft','super_area_sqft','bathroom_count',
                    'balcony_count','bedrooms','floor_number','total_floors','parking_count']
    model = cleaned.loc[cleaned.price_inr.gt(0),feature_cols+['price_inr']].copy()
    X = model[feature_cols].copy()
    for c in ['location','Furnishing','Transaction','Ownership','facing']:
        X[c] = X[c].astype('object').where(X[c].notna(), np.nan)
    y = model.price_inr
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in feature_cols if c not in numeric]
    transformer = ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),numeric),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Unknown')),('one_hot',OneHotEncoder(handle_unknown='ignore',min_frequency=100))]),categorical)
    ],sparse_threshold=0.3)
    Xt = transformer.fit_transform(X_train)
    Xv = transformer.transform(X_test)
    joblib.dump(transformer, ROOT/'data'/'preprocessor.joblib')
    np.savez_compressed(ROOT/'data'/'split_indices.npz', train_indices=X_train.index.to_numpy(),test_indices=X_test.index.to_numpy())
    report += [f'Training rows: {len(X_train):,}',f'Test rows: {len(X_test):,}',
               f'Preprocessed feature count: {Xt.shape[1]:,}',
               f'Prepared train/test shapes: {Xt.shape}, {Xv.shape}']
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports'/'processing_summary.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report))
    return cleaned, X_train

if __name__ == '__main__': main()
