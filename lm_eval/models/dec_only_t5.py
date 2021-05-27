from transformers import T5TokenizerFast
import torch
import torch.nn.functional as F
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm

### I very much dislike this solution. TODO: fix this abomination for jay-z repo
import os
import sys
from pathlib import Path
path = Path(os.path.realpath(__file__))
workfolder = str(path.parent.parent.parent.parent)
sys.path.append(workfolder)
###
from models.decoder_only_t5 import DecoderOnlyT5LMHeadModel

class DecoderOnlyT5LM(LM):
    MAX_GEN_TOKS = 256

    def __init__(self, pretrained, device=0, batch_size=1):
        super().__init__()
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.decoder = DecoderOnlyT5LMHeadModel.from_pretrained(pretrained).to(self.device)
        self.decoder.eval()

        self.tokenizer = T5TokenizerFast.from_pretrained('t5-small')
        self.max_length = 1024

        assert self.tokenizer.encode('hello\n\nhello', add_special_tokens=False) == [21820, 21820]

        self.batch_size = batch_size

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config={}):
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.tokenizer.eos_token_id]
            else:
                # "\n" are removed inside T5Tokenizer, by passing this by introducing custom separators
                context = context.replace("\n===\n\n", " === ")
                context = context.replace("\n\n"," == ")
                context_enc = self.tokenizer.encode(context, add_special_tokens=False)

            continuation_enc = self.tokenizer.encode(continuation, add_special_tokens=False)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer.encode(string),
                    prefix_token=self.tokenizer.eos_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]
                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []
        with torch.no_grad():

            def _collate(x):
                # the negative sign on len(toks) sorts descending - this has a few advantages:
                # - time estimates will always be over not underestimates, which is more useful for planning
                # - to know the size of a batch when going through the list, you know the first one is always the batch padded context length.
                #   this is useful to simplify the batching logic and more importantly to make automatic adaptive batches much much easier to implement
                # - any OOMs will happen right away rather than near the end

                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps = self.tokenizer.pad({"input_ids": [ (t[1]+t[2])[-self.max_length:][:-1] for t in chunk]}, return_tensors="pt")
                inps = {k: v.to(self.device) for k, v in inps.items()}
                contlens = [t[2] for t in chunk]
                inplens = inps["attention_mask"].sum(dim=-1).tolist()

                outputs = self._model_call(inps)
                multi_logits = F.log_softmax(outputs, dim=-1).cpu()  # [batch, seq, vocab]

                for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps["input_ids"], inplens, contlens):
                    contlen = len(cont_toks)
                    logits = logits[inplen-contlen:inplen].unsqueeze(0) # [1, seq, vocab]
                    greedy_tokens = logits.argmax(dim=-1)
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0) # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1) # [1, seq]
                    answer = (float(logits.sum()), bool(max_equal))

                    # partial caching
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                    res.append(answer)

        return reord.get_original(res)

    def _model_call(self, inputs):
        """
        inputs: a dictionary torch tensors to pass to the model

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits retuned from the model
        """
        return self.decoder(**inputs)[0][:, :, :self.tokenizer.vocab_size]

    def greedy_until(self, requests):
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])

        reord = utils.Reorderer(requests, _collate)

        for context, until in tqdm(reord.get_reordered()):
            if isinstance(until, str):
                until = [until]

            context_enc = torch.tensor([self.tokenizer.encode(context)[self.MAX_GEN_TOKS - self.max_length:]]).to(self.device)
            primary_until, = self.tokenizer.encode(until[0])
            cont = self.decoder.generate(
                context_enc,
                max_length=context_enc.shape[1] + self.MAX_GEN_TOKS,
                eos_token_id=primary_until,
                do_sample=False
            )

            s = self.tokenizer.decode(cont[0].tolist()[context_enc.shape[1]:])
            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            res.append(s)

        return reord.get_original(res)
