import collections
import itertools
import random
import lm_eval.metrics
from IPython import embed
import numpy as np
from tqdm import tqdm
import math
import torch.nn.functional as F

import torch
def evaluate( lm, task_dict, provide_description, num_fewshot, limit, bootstrap_iters=100000):
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces
    print("doing emsemble")
    task_dict_items = [(name, task) for name, task in task_dict.items() if(task.has_validation_docs() or task.has_test_docs())]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    # if we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger memory,
    # we can always modify this plumbing to support that, but i didn't want to include it just yet because overengineering is bad
    # (or we could make it write the requests to disk and then read them back out again - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable

    docs = {}

    # get lists of each type of requeste
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        #default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
        elif task.has_validation_docs():
            task_doc_func = task.validation_docs

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        print("total_num",len(task_docs))
        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            docs[(task_name, doc_id)] = doc

            ctx = task.fewshot_context_ensemble(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
                rnd=rnd
            )
            reqs = task.construct_requests(doc, ctx)
            # embed()
            if not isinstance(reqs, (list, tuple)): reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.type].append((i, task_name, doc, doc_id))
            # embed()
            # exit()
    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def get_inference(requests,lm):
        res = []
        for request in tqdm(requests):

            '''
            input_batch: batch_size * num_shots
            targets_batch: batch_size

            for num:
                inputs = inputs_batch[num]
                targets = [targets_batch[num]]*len(inputs)
            '''
            sub_res = []
            inputs = request[0]
            targets = [request[1]]*len(inputs)
            # embed()
            inputs_tok = lm.tokenizer(
                inputs,
                max_length=lm.max_length,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt"
                ).to(lm.device)

            for key in inputs_tok:
                inputs_tok[key] = inputs_tok[key][:, -(lm.max_length - 1) :]

            targets_tok = lm.tokenizer(
                targets,
                max_length=lm.MAX_GEN_TOKS,
                padding=True,
                # truncation=True,
                add_special_tokens=False,
                return_tensors="pt"
                ).to(lm.device)

            for key in targets_tok:
                targets_tok[key] = targets_tok[key][:, -(lm.max_length - 1) :]

            with torch.no_grad():
                outputs = lm.t5(
                    **inputs_tok,
                    labels=targets_tok["input_ids"]
                    )

            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                log_softmaxes,
                targets_tok["input_ids"],
                targets_tok["attention_mask"],
            )
            for log_softmax, target_tok, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tok = target_tok[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tok).all()
                target_logits = torch.gather(log_softmax, 1, target_tok.unsqueeze(-1)).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                sub_res.append(answer)

            # logits_score = sum([x[0] for x in sub_res])
            # assert(type(sub_res) == tuple)
            # embed()
            res.append(sub_res)
        return res

    # execute each type of request
    print("total_num",len(requests))
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple seperate LM requests for multiple Requests differing
        # only in index. We could implement some kind of caching, but that would be more of a bandaid
        # solution. we could also implement some kind of autogrouping here; they should end up next to each other.

        print("Running", reqtype, "requests")
        # reqs_parts = chunks(reqs,10000)
        # resps = []
        # for req_part in reqs_parts:
        #     resps+=getattr(lm, reqtype)([req.args for req in req_part])

        # resps = getattr(lm, reqtype)([req.args for req in reqs])

        resps = get_inference([req.args for req in reqs],lm)
        # embed()
        before = len(resps)

        resps = [x for x in resps if type(x) == list]
        after = len(resps)
        assert(before == after)
        # resps = [x if req.index is None else x[req.index] for x, req in zip(resps, reqs)]


        # embed()
        overall = []

        for resps_l, req in zip(resps, reqs):
            # embed()
            tmp = []
            for x in resps_l:
                if req.index is None:
                    tmp.append(x)
                else:
                    tmp.append(x[req.index])
            overall.append(np.array(tmp))
        # embed()
        resps = np.array(overall)


        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))
    
    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        # embed()
        requests.sort(key=lambda x: x[0])
        requests = np.array([x[1] for x in requests])

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]
        # embed()
        metrics = task.process_results(doc, requests)

        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)
    
    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        results[task_name][metric] = task.aggregation()[metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this
        stderr = lm_eval.metrics.stderr_for_metric(task.aggregation()[metric], bootstrap_iters=min(bootstrap_iters, 1000) if metric in ["bleu", "chrf", "ter"] else bootstrap_iters)
        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)
    
    return {
        "results": results,
        "versions": versions
    }
