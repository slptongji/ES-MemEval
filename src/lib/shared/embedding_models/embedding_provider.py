from langchain_huggingface import HuggingFaceEmbeddings
import torch


class EmbeddingProvider:
    _bge_m3: dict[str, HuggingFaceEmbeddings] = {}

    @classmethod
    def bge_m3(cls, prefer_cuda: bool) -> HuggingFaceEmbeddings:
        device: str = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
        if device not in cls._bge_m3:
            cls._bge_m3[device] = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", 
                                                        model_kwargs={"device": device})
        return cls._bge_m3[device]
    
    _all_minilm_l6_v2: dict[str, HuggingFaceEmbeddings] = {}
    
    @classmethod
    def all_minilm_l6_v2(cls, prefer_cuda: bool) -> HuggingFaceEmbeddings:
        device: str = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
        if device not in cls._all_minilm_l6_v2:
            cls._all_minilm_l6_v2[device] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                model_kwargs={"device": device})
        return cls._all_minilm_l6_v2[device]
