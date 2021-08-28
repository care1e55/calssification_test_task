from typing import Dict, Tuple
from argparse import ArgumentParser
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from catalyst.dl import SupervisedRunner
from catalyst.callbacks.metrics.classification import PrecisionRecallF1SupportCallback
from catalyst.callbacks.scheduler import SchedulerCallback
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from catalyst.utils import set_global_seed, prepare_cudnn
from .utils import make_pds, load_model_config
from .classification_dataset import TextClassificationDataset
from .focal_loss import FocalLoss
from .bert_for_classification import BertForSequenceClassification


# TODO: docstrings
class Trainer():
    def __init__(self, model_config_path) -> None:
        set_global_seed(42)
        prepare_cudnn(True)
        self.settings = load_model_config(model_config_path)
        self.dataloaders = self.initilize_dataloaders(self.settings)
        self.model = BertForSequenceClassification(
            self.settings['pretrained_model_name'], 
            num_classes = self.settings['num_classes']
        )
        self.warmup_steps, self.t_total, self.optimizer_grouped_parameters = self.initilize_learning_params(self.dataloaders, self.settings, self.model)

    def initilize_dataloader(self, df: pd.DataFrame, tokenizer: BertTokenizer, settings) -> DataLoader:
        dataset = TextClassificationDataset(df, tokenizer, settings['max_seq_length'])
        sampler = RandomSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=settings['batch_size'])

    def initilize_dataloaders(self, settings: Dict[str, str]) -> Dict[str, DataLoader]:
        train_pd, valid_pd = make_pds(settings['data_path'])
        tokenizer = BertTokenizer.from_pretrained(settings['pretrained_model_name'])
        return {
            "train": self.initilize_dataloader(train_pd, tokenizer, settings),
            "valid": self.initilize_dataloader(valid_pd, tokenizer, settings),
            "test": self.initilize_dataloader(valid_pd, tokenizer, settings)    
        }

    def initilize_learning_params(self, dataloaders: Dict[str, DataLoader], settings: str, model) -> Tuple:
        warmup_steps = len(dataloaders['train']) // 2
        t_total = len(dataloaders['train']) * settings['epochs']
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters()], "weight_decay": 0.0},
        ]
        return warmup_steps, t_total, optimizer_grouped_parameters

    def train(self):
        # TODO: switch between losses
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = FocalLoss()
        optimizer = AdamW(self.optimizer_grouped_parameters, lr=self.settings['lr'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warmup_steps, 
            num_training_steps=self.t_total
        )

        train_val_loaders = {
            "train": self.dataloaders['train'],
            "valid": self.dataloaders['valid']
        }

        runner = SupervisedRunner(
            input_key=(
                "input_ids",
                "attention_mask",
            )
        )

        runner.train(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=train_val_loaders,
            callbacks=[
                AccuracyCallback(
                    input_key="logits", 
                    target_key="targets", 
                    num_classes=self.settings['num_classes']
                ),
                PrecisionRecallF1SupportCallback(
                    input_key="logits", 
                    target_key="targets", 
                    num_classes=self.settings['num_classes']
                ),
                SchedulerCallback(mode='batch'),
            ],
            logdir=self.settings['log_dir'],
            num_epochs=self.settings['epochs'],
            verbose=False,
        )

        torch.save(self.model.state_dict(), self.settings['out_model_path'])
