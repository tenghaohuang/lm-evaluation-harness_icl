import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm
import numpy as np
import math
from IPython import embed
class T5LM():
    MAX_GEN_TOKS = 256
    MAX_INP_LENGTH = 512
    VOCAB_SIZE = 32128
    EOT_TOKEN_ID = 1

    def __init__(self, device='cuda', parallelize=False, pretrained='t5', batch_size=1):

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.t5 = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained, low_cpu_mem_usage=True)
        self.t5.eval()

        if parallelize == "True":
            print(parallelize)
            self.t5.parallelize()
            self.device = torch.device('cuda:0')
        else:
            self.t5.to(self.device)

        self.tokenizer = transformers.T5TokenizerFast.from_pretrained(pretrained)
        self.max_length = self.MAX_INP_LENGTH

        self.batch_size = int(batch_size)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config={}):
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    # def loglikelihood(self, requests):
    #     res = []
    #     for chunk in tqdm(utils.chunks(requests, self.batch_size), total=math.ceil(len(requests) / self.batch_size)):
    #
    #         inputs, targets = zip(*chunk)
    #         # embed()
    #         inputs_tok = self.tokenizer(
    #             list(inputs),
    #             max_length=self.max_length,
    #             padding=True,
    #             # truncation=True,
    #             add_special_tokens=False,
    #             return_tensors="pt"
    #         ).to(self.device)
    #
    #         for key in inputs_tok:
    #             inputs_tok[key] = inputs_tok[key][:, -(self.max_length - 1):]
    #
    #         targets_tok = self.tokenizer(
    #             list(targets),
    #             max_length=self.MAX_GEN_TOKS,
    #             padding=True,
    #             # truncation=True,
    #             add_special_tokens=False,
    #             return_tensors="pt"
    #         ).to(self.device)
    #
    #         for key in targets_tok:
    #             targets_tok[key] = targets_tok[key][:, -(self.max_length - 1):]
    #
    #         with torch.no_grad():
    #             outputs = self.t5(
    #                 **inputs_tok,
    #                 labels=targets_tok["input_ids"]
    #             )
    #
    #         log_softmaxes = F.log_softmax(outputs.logits, dim=-1)
    #
    #         output_iterator = zip(
    #             chunk,
    #             log_softmaxes,
    #             targets_tok["input_ids"],
    #             targets_tok["attention_mask"],
    #         )
    #         for cache_key, log_softmax, target_tok, target_mask in output_iterator:
    #             length = target_mask.sum()
    #             log_softmax = log_softmax[:length]
    #             target_tok = target_tok[:length]
    #             greedy_tokens = log_softmax.argmax(dim=-1)
    #             max_equal = (greedy_tokens == target_tok).all()
    #             target_logits = torch.gather(log_softmax, 1, target_tok.unsqueeze(-1)).squeeze(-1)
    #             answer = (float(target_logits.sum()), bool(max_equal))
    #
    #             if cache_key is not None:
    #                 self.cache_hook.add_partial("loglikelihood", cache_key, answer)
    #
    #             res.append(answer)
    #
    #     return res


    def loglikelihood(self, requests):
        res = []
        print("trace")
        print(self.batch_size)
        print(math.ceil(len(requests)/self.batch_size))
        for chunk in tqdm(utils.chunks(requests, self.batch_size), total=math.ceil(len(requests)/self.batch_size)):

            '''
            input_batch: batch_size * num_shots
            targets_batch: batch_size

            for num:
                inputs = inputs_batch[num]
                targets = [targets_batch[num]]*len(inputs)
            '''
            embed()
            inputs_batch, targets_batch = zip(*chunk)
            for num in range(len(inputs_batch)):
                sub_res = []
                inputs = inputs_batch[num]
                targets = [targets_batch[num]]*len(inputs)
                # embed()
                inputs_tok = self.tokenizer(
                    inputs,
                    max_length=self.max_length,
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt"
                    ).to(self.device)

                for key in inputs_tok:
                    inputs_tok[key] = inputs_tok[key][:, -(self.max_length - 1) :]

                targets_tok = self.tokenizer(
                    targets,
                    max_length=self.MAX_GEN_TOKS,
                    padding=True,
                    # truncation=True,
                    add_special_tokens=False,
                    return_tensors="pt"
                    ).to(self.device)

                for key in targets_tok:
                    targets_tok[key] = targets_tok[key][:, -(self.max_length - 1) :]

                with torch.no_grad():
                    outputs = self.t5(
                        **inputs_tok,
                        labels=targets_tok["input_ids"]
                        )

                log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

                output_iterator = zip(
                    chunk,
                    log_softmaxes,
                    targets_tok["input_ids"],
                    targets_tok["attention_mask"],
                )
                for cache_key, log_softmax, target_tok, target_mask in output_iterator:
                    length = target_mask.sum()
                    log_softmax = log_softmax[:length]
                    target_tok = target_tok[:length]
                    greedy_tokens = log_softmax.argmax(dim=-1)
                    max_equal = (greedy_tokens == target_tok).all()
                    target_logits = torch.gather(log_softmax, 1, target_tok.unsqueeze(-1)).squeeze(-1)
                    answer = (float(target_logits.sum()), bool(max_equal))
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                    sub_res.append(answer)

                # logits_score = sum([x[0] for x in sub_res])
                assert(type(sub_res) == tuple)
                # embed()
                res.append(sub_res)
                # print(sub_res)
                # embed()
        return res
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def greedy_until(self, requests):
        
        res = []

        for context, until in tqdm(requests):
            if isinstance(until, str): until = [until]

            context_enc = self.tokenizer(context, return_tensors="pt").to(self.device).input_ids

            primary_until = self.tokenizer.encode(until[0])

            cont = self.t5.generate(
                context_enc,
                max_length=self.MAX_GEN_TOKS,
                eos_token_id=primary_until,
                do_sample=False
            )

            s = self.tokenizer.decode(cont[0].tolist())
            
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            
            res.append(s)
        
        return res
