from collections import defaultdict
from pathlib import Path
import time
from typing import Any, Literal, Sequence
import bert_score
import bert_score.utils
import torch
from transformers import AutoModel, AutoTokenizer
from lib.shared.metrics.metric import Metric


class QaBertScore:
    def __init__(self, 
                 prefer_cuda: bool,
                 model_type: str = "bert-base-uncased") -> None:
        super().__init__()
        self.model_type = model_type
        self.device = "cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.model = self.get_model(model_type)
        self.model.to(self.device)

    def compute(self, gold: str, prediction: str) -> float:
        idf_dict: defaultdict = defaultdict(lambda: 1.0)
        idf_dict[self.tokenizer.sep_token_id] = 0
        idf_dict[self.tokenizer.cls_token_id] = 0
        result: torch.Tensor = bert_score.bert_cos_score_idf(self.model, [gold], [prediction],
                                                             self.tokenizer, idf_dict, device=self.device).cpu()
        return float(result[..., 2])
    
    @staticmethod
    def get_model(model_type):
        # Modified from bert_score.get_model
        num_layers: int = bert_score.utils.model2layers[model_type]
        
        model: Any
        if "t5" in model_type:
            from transformers import T5EncoderModel
            model = T5EncoderModel.from_pretrained(model_type)
        else:
            model = AutoModel.from_pretrained(model_type)
        model.eval()

        if hasattr(model, "decoder") and hasattr(model, "encoder"):
            model = model.encoder

        if hasattr(model, "n_layers"):  # xlm
            assert (
                0 <= num_layers <= model.n_layers
            ), f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, "layer"):  # xlnet
            assert (
                0 <= num_layers <= len(model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = torch.nn.ModuleList(
                [layer for layer in model.layer[:num_layers]]
            )
        elif hasattr(model, "encoder"):  # albert
            if hasattr(model.encoder, "albert_layer_groups"):
                assert (
                    0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):  # t5
                assert (
                    0 <= num_layers <= len(model.encoder.block)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.block)} for {model_type}"
                model.encoder.block = torch.nn.ModuleList(
                    [layer for layer in model.encoder.block[:num_layers]]
                )
            else:  # bert, roberta
                assert (
                    0 <= num_layers <= len(model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = torch.nn.ModuleList(
                    [layer for layer in model.encoder.layer[:num_layers]]
                )
        elif hasattr(model, "transformer"):  # bert, roberta
            assert (
                0 <= num_layers <= len(model.transformer.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = torch.nn.ModuleList(
                [layer for layer in model.transformer.layer[:num_layers]]
            )
        elif hasattr(model, "layers"):  # bart
            assert (
                0 <= num_layers <= len(model.layers)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layers)} for {model_type}"
            model.layers = torch.nn.ModuleList(
                [layer for layer in model.layers[:num_layers]]
            )
        else:
            raise ValueError("Not supported")
        
        return model


def _test():
    print(QaBertScore(False).compute("Hello World", "Hi Peter"))
    print(bert_score.score(["Hi Peter"], ["Hello World"], 
                           model_type="bert-base-uncased", lang="en", device="cpu"))


if __name__ == "__main__":
    _test()
