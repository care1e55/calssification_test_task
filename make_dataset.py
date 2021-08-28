import pandas as pd
import numpy as np
from argparse import ArgumentParser
import os
import re
import nltk
from nltk.corpus import stopwords
import string
import json

# TODO: logging, docstring


def filter_stopwords_and_punkt(tokens):
    return list(filter(
        lambda word: 
            word.lower() not in stopwords.words("russian") 
            and not any(punct in word for punct in string.punctuation)
        , tokens
    ))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].str.split().apply(filter_stopwords_and_punkt)
    df['text'] = df['text'].str.join(" ")
    return df


def processed_for_fasttext(dataset: pd.DataFrame, output_dir: str, output_file_name: str):
    file_path = os.path.join(output_dir, output_file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)
    with open(file_path, 'w+') as outfile:
        for i in range(len(dataset)):
            outfile.write(
                '__label__' + dataset['codes'][i]
                + ' ' + 
                dataset['text'][i] + '\n'
            )


def text_to_df(input_dataset: str) -> pd.DataFrame:
    df = pd.read_csv(input_dataset, sep=',')
    df['intent_junior'] = pd.Categorical(df['intent_junior'])
    df['codes'] = df['intent_junior'].cat.codes.apply(str)
    return df


def make_fasttext_dataset(df: pd.DataFrame, output_dir: str):
    train_dataset = df[df['split'] == 'train'].reset_index()[['codes', 'text']]
    valid_dataset = df[df['split'] == 'valid'].reset_index()[['codes', 'text']]
    processed_for_fasttext(train_dataset, output_dir, 'train.txt')
    processed_for_fasttext(valid_dataset, output_dir, 'valid.txt')    


def make_allennlp_dataset(df, name):
    with open(os.path.join(args.output_dir, name), 'w', encoding='utf-8') as outfile:
        for text, code in zip(df['text'].tolist(), df['codes'].tolist()):
            outfile.write(f"{{'text': {text}, 'label': {code}}}")
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--input_dataset', 
        default="junior_no_test.csv", 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--output_dir', 
        default="data", 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--type', 
        default="allennlp", 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--clean', 
        default=False, 
        required=False, 
        type=bool
    )
    parser.add_argument(
        '--augment', 
        default=False, 
        required=False, 
        type=bool
    )

    args = parser.parse_args()
    df = text_to_df(args.input_dataset)
    if args.clean:
        df = clean(df)
    # make_fasttext_dataset(df, args.output_dir)
    make_allennlp_dataset(df[df['split'] == 'train'], 'train.txt')
    make_allennlp_dataset(df[df['split'] == 'valid'], 'valid.txt')
