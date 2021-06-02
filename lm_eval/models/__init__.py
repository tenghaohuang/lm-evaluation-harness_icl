from . import gpt2
from . import gpt3
from . import t5
from . import dummy
from . import dec_only_t5

MODEL_REGISTRY = {
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "deconlyt5": dec_only_t5.DecoderOnlyT5LM,
    "t5": t5.T5LM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
