from argparse import ArgumentParser
from loguru import logger
from typing import Iterable, Tuple, List, Dict
import fasttext
import os
import json

# TODO: docstring

# TODO: config from JSON file
def load_codes_mapping(codes_path: str):
    with open(codes_path, mode='r', encoding='utf-8') as file:
        return {int(k):v for k,v in json.load(file).items()}

def load_model(model_path: str):
    return fasttext.load_model(model_path)

def predict_one(sentence: str, classifier) -> str:
    code = classifier.predict(sentence.strip('\n'))[0][0].replace('__label__', '')
    return code_to_intent[int(code)]
    
def get_preds(input_path: str, classifier):
    for sentence in open(input_path):
        yield predict_one(sentence, classifier)
        
def dump_preds(input_path: str, output_path: str, preds: Iterable, include_source: bool = False):
    if os.path.isfile(output_path):
        os.remove(output_path)
    with open(output_path, 'w+') as outfile:
        for source, pred in zip(open(input_path), preds):
            source = source.strip('\n')
            logger.info(f"{source}: {pred}")
            outfile.write(f"{source} \t {pred}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--input_path', 
        default="fasttext/data/test.txt", 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--output_path', 
        default="fasttext/data/preds.txt", 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--model_path', 
        default="fasttext/model.fasstext", 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--intent_codes_path', 
        default="fasttext/config/codes.json",
        required=False, 
        type=str
    )

    args = parser.parse_args()
    code_to_intent = load_codes_mapping(args.codes_path)
    classifier = load_model(args.model_path)
    preds = get_preds(args.input_path, classifier)
    dump_preds(args.input_path, args.output_path, preds)
