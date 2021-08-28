from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
import json


def _get_predictor(self, inference_config: Dict, model_path: str) -> Predictor:
    """Instantiate AllenNLP Predictor with `model_path` and `inference_config`."""
    archive = load_archive(
        model_path, overrides=json.dumps(inference_config), cuda_device=self._cuda_device
    )
    return Predictor.from_archive(archive, "seq2seq")
