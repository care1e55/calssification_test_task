from argparse import ArgumentParser
import fasttext
import optuna
import sklearn.metrics
from typing import Tuple, List, Dict
from loguru import logger

# TODO: docstring

def objective(trial) -> float:
    lr = trial.suggest_float('lr', 0.002, 1.0, log=True)
    epoch = trial.suggest_int('epoch', 1, 20)
    ws = trial.suggest_int('ws', 1, 5)
    wordNgrams = trial.suggest_int('wordNgrams', 1, 5)
    classifier = fasttext.train_supervised(
        args.train_path, 
        lr = lr, 
        epoch = epoch, 
        ws = ws,
        wordNgrams = wordNgrams
    ) 
    return get_f1_macro(source, target, classifier)


def get_source_target(dataset: List[str]) -> Tuple[List[str], List[str]]:
    source, target = [], []
    for row in dataset:
        try:
            splitted = row.split(maxsplit=1)
            source.append(splitted[1].replace('\n', ''))
            target.append(splitted[0].replace('__label__', ''))
        except:
            logger.debug(f'Row not processed: {splitted}')
    return target, source


def get_f1_macro(X: List[str], y_true: List[str], classifier) -> float:
    y_pred = [pred[0].replace('__label__', '') for pred in classifier.predict(X)[0]]
    return sklearn.metrics.f1_score(y_true, y_pred, average='macro')


def get_preds(X: List[str], classifier):
    return [pred[0].replace('__label__', '') for pred in classifier.predict(X)[0]]


def get_metrics(y_pred: List[str], y_true: List[str]) -> Dict[str, float]:
    return {
        'f0.5': sklearn.metrics.fbeta_score(y_true, y_pred, average='macro', beta=0.5),
        'f1': sklearn.metrics.fbeta_score(y_true, y_pred, average='macro', beta=1),
        'f2': sklearn.metrics.fbeta_score(y_true, y_pred, average='macro', beta=2),
        'recall': sklearn.metrics.recall_score(y_true, y_pred, average='macro'),
        'precision': sklearn.metrics.precision_score(y_true, y_pred, average='macro'),
        'acuracy': sklearn.metrics.accuracy_score(y_true, y_pred)
    }


def log_metrics(metrics: Dict[str, float]):
    return "\n".join([ f"{k}: {v}" for k, v in metrics.items() ])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--train_path', 
        default="data/train.txt", 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--valid_path', 
        default="data/valid.txt", 
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
        '--n_trials', 
        default=10, 
        required=False, 
        type=int
    )

    args = parser.parse_args()

    target, source = get_source_target(open(args.valid_path).readlines())
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, args.n_trials)

    logger.info('F0.5_macro: {}'.format(study.best_trial.value))
    logger.info("Best hyperparameters: {}".format(study.best_trial.params))
    classifier = fasttext.train_supervised(
        args.train_path, 
        lr = study.best_trial.params['lr'], 
        epoch = study.best_trial.params['epoch'], 
        ws = study.best_trial.params['ws'],
        wordNgrams = study.best_trial.params['wordNgrams']
    )
    preds = get_preds(source, classifier)
    logger.info('\n' + log_metrics(get_metrics(preds, target)))
    classifier.save_model('model.fasstext')
    logger.info('Model saved at model.fasstext')
