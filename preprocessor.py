import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

def money_to_int(money_str):
    if pd.isna(money_str):
        return np.nan # or return None if you want to keep NaNs

    money_str = str(money_str).strip()  # convert to string and strip spaces
    
    multiplier = 1
    number = None
    
    if money_str.endswith('M'):
        multiplier = 1_000_000
        number = money_str.lstrip('$').rstrip('M')
    elif money_str.endswith('K'):
        multiplier = 1_000
        number = money_str.lstrip('$').rstrip('K')
    else:
        # Handles cases with no suffix, just numbers with or without $
        number = money_str.lstrip('$')
    
    try:
        return int(float(number) * multiplier)
    except ValueError:
        # In case the string is not a valid number
        return np.nan


def preproccess(df : pd.DataFrame) -> pd.DataFrame:
    df=df.drop(["temp_ranking","previous_ranking"],axis=1)
    df["city"] = df["city"].notnull().astype(int)
    df=df.drop(["valuation_as_of"],axis=1)
    df=df.drop(["Contact Data"],axis=1)
    df["state"] = df["state"].notnull().astype(int)
    df["country"] = df["country"].notnull().astype(int)
    df["founded"] = df["founded"].notnull().astype(int)
    df["valuation"]=df["valuation"].fillna(0)
    df["url"] = df["url"].notnull().astype(int)
    df["linkedin_url"] = df["linkedin_url"].notnull().astype(int)
    df["total_funding"]=df["total_funding"].apply(money_to_int)
    df["total_funding"]=df["total_funding"].fillna(df["total_funding"].mean())
    df = df[df["valuation"]< df["valuation"].mean() + 2*df["valuation"].std()]
    df["valuation"]=scaler.fit_transform(df[["valuation"]])
    df["current_employees"]=scaler.fit_transform(df[["current_employees"]])
    df["employee_growth"]=scaler.fit_transform(df[["employee_growth"]])
    df["total_funding"]=scaler.fit_transform(df[["total_funding"]])
    df=df[["city",'country','state','linkedin_url','url','founded','current_employees','employee_growth','total_funding','valuation']]

    return df
