from config import *
from model import *
from flashrag.retriever import DenseRetriever
from flashrag.config import Config
from flashrag.utils import get_dataset
# import torch
# import torch_npu

def mp_rag(cfg, test_data):
    pipeline = MP_RAG(cfg,
                                  max_fusion_rounds=cfg["max_fusion_rounds"])
    result = pipeline.run(test_data)

    return result

def prunerag(cfg, test_data):
    pipeline = FlashRAGPruneRAG(cfg)
    result = pipeline.run(test_data)

    return result


def main(cfg):
    all_splits = get_dataset(cfg)
    test_data = all_splits["dev"]

    func_map = {
        "Naive Gen": naive_gen,
        "Naive RAG": naive_rag,
        "FLARE": flare,
        "Iter-RetGen": iterretgen,
        "IRCoT": ircot,
        "Self-Ask": self_ask,
        "SuRe": sure,
        "MAD": mad,
        "Self-RAG": selfrag,
        "Ret-Robust": retrobust,
        "MP_RAG": mp_rag,
        "MA-RAG": ma_rag,
        "PruneRAG": prunerag
    }

    func = func_map[cfg["method_name"]]
    func(cfg, test_data)


if __name__ == "__main__":
    cfg = init_cfg()
    print("显卡测试shibai")
    main(cfg)

