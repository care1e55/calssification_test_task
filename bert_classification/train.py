from argparse import ArgumentParser
from model.trainer import Trainer

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--data_path', 
        default="data/train.txt", 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--model_config', 
        default="config/model_config.json", 
        required=False, 
        type=str
    )
    args = parser.parse_args()
    trainer = Trainer(args.model_config)
    trainer.train()
