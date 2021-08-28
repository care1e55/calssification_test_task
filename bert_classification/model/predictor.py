from typing import Any, Dict, Iterable
import transformers
import torch
import os
from loguru import logger
from .bert_for_classification import BertForSequenceClassification
from .utils import load_codes_mapping, load_model_config


# TODO: docstring
class Predictor():
    def __init__(self, codes_path, model_config, model_path) -> None:
        self.code_to_ = load_codes_mapping(codes_path)
        self.settings = load_model_config(model_config)
        self.classifier = self.load_model(model_path, self.settings)
    
    def load_model(self, model_path, settings: Dict[str, str]) -> torch.nn.Module:
        model = BertForSequenceClassification(
            settings['pretrained_model_name'], 
            num_classes = settings['num_classes']
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def get_preds_generator(self, input_path: str, classifier: torch.nn.Module) -> Iterable:
        tokenizer = transformers.BertTokenizer.from_pretrained(self.settings['pretrained_model_name'])
        for line in open(input_path):
            encodded = tokenizer.encode_plus(line, max_length=128, pad_to_max_length=True)
            encodded = {'input_ids': encodded['input_ids'], 'attention_mask': encodded['attention_mask']}
            code = classifier(
                torch.tensor(encodded['input_ids']).unsqueeze(0), 
                torch.tensor(encodded['attention_mask']).unsqueeze(0)
            ).argmax().tolist()
            yield self.code_to_[code]

    def dump_preds(self, input_path: str, output_path: str, preds: Iterable, include_source: bool = False):
        if os.path.isfile(output_path):
            os.remove(output_path)
        with open(output_path, 'w+') as outfile:
            for source, pred in zip(open(input_path), preds):
                source = source.strip('\n')
                logger.info(f"{source}: {pred}")
                outfile.write(f"{source}\t{pred}")

    def predict_file(self, input_path, output_path):
        preds_generator = self.get_preds_generator(input_path, self.classifier)
        self.dump_preds(input_path, output_path, preds_generator)
        