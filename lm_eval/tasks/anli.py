import numpy as np
from lm_eval.base import rf
from ..metrics import mean
from .common import HFTask


class ANLIBase(HFTask):
    VERSION = 0
    DATASET_PATH = "anli"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.data["train_r" + str(self.SPLIT)])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["dev_r" + str(self.SPLIT)]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test_r" + str(self.SPLIT)]

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc):
        # OA does this a bit weirdly: they prepend "anli 1:  anli 1:  " to the beginning
        # of the prompt (yes, repeating it!). also, " True, False, or Neither?" is directly
        # appended onto the question, with no "Answer:" or even a newline. Do we *really*
        # want to do it exactly as OA did?
        return doc["premise"] + "\nQuestion: " + doc["hypothesis"] + " True, False, or Neither?\nAnswer:"

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " " + ["True", "Neither", "False"][doc["label"]]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc["label"]
        # pred = np.argmax(results)
        pred = results >= results.max(axis=0)
        acc = 1.0 if np.argmax(pred.sum(axis=1)) == gold else 0.0

        return {"acc": acc}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True}


class ANLIRound1(ANLIBase):
    SPLIT = 1


class ANLIRound2(ANLIBase):
    SPLIT = 2


class ANLIRound3(ANLIBase):
    SPLIT = 3
