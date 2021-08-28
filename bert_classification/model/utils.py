import json
import pandas as pd
from typing import Tuple, Dict

def load_model_config(model_config_path: str):
    with open(model_config_path, mode='r', encoding='utf-8') as file:
        return json.load(file)

def load_codes_mapping(codes_path: str) -> Dict[int, str]:
    with open(codes_path, mode='r', encoding='utf-8') as file:
        return {int(k):v for k,v in json.load(file).items()}

def text_to_df(input_dataset: str) -> pd.DataFrame:
    df = pd.read_csv(input_dataset, sep=',')
    df['intent_junior'] = pd.Categorical(df['intent_junior'])
    df['codes'] = df['intent_junior'].cat.codes.apply(str)
    return df

def make_pds(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = text_to_df(data_path)
    return make_pd(df.copy(), 'train'), make_pd(df.copy(), 'valid')

def make_pd(df: pd.DataFrame, split: str) -> pd.DataFrame:
    new_df = pd.DataFrame()
    new_df['sentence'] = df[df['split'] == split]['text']
    new_df['label'] = df[df['split'] == split]['codes'].apply(int)
    return new_df
