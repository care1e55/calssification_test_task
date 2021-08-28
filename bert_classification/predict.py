from model.predictor import Predictor
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--input_path', 
        default='test.txt', 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--output_path', 
        default='bert_preds.csv', 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--model_path', 
        default="bert_for_classification.model", 
        required=False, 
        type=str
    )
    parser.add_argument(
        '--intent_codes_path', 
        default="config/intent_codes.json", 
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
    predictor = Predictor(args.intent_codes_path, args.model_config, args.model_path)
    predictor.predict_file(args.input_path, args.output_path)
