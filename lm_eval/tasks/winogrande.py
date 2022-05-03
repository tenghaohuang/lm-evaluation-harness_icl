import numpy as np
from .common import HFTask
from lm_eval.base import rf
from ..metrics import mean

"""
This evaluation of Winogrande uses partial evaluation as described by
Trinh & Le in Simple Method for Commonsense Reasoning (2018).
Reference: https://arxiv.org/abs/1806.02847
"""


class Winogrande(HFTask):
    VERSION = 0
    DATASET_PATH = "winogrande"
    DATASET_NAME = "winogrande_xl"

    answer_to_num = {"1": 0, "2": 1}

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def doc_to_text(self, doc):
        return self.partial_context(doc, doc["option" + doc["answer"]])

    def fewshot_description(self):
        # TODO: redo description
        return "Winograd schema sentence including a either a ___ blank with a missing word, making the pronoun ambiguous, or the same with the word filled in."

    @classmethod
    def partial_context(cls, doc, option):
        # Substitute the pronoun in the sentence with the specified option
        # and ignore everything after.
        pronoun_loc = doc["sentence"].index("_")
        return doc["sentence"][:pronoun_loc] + option

    def doc_to_target(self, doc):
        return self.partial_target(doc)

    @classmethod
    def partial_target(cls, doc):
        # The target is everything after the document specified pronoun.
        pronoun_loc = doc["sentence"].index("_") + 1
        return " " + doc["sentence"][pronoun_loc:].strip()

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
        target = self.partial_target(doc)
        lls = []
        for option in [doc["option1"], doc["option2"]]:
            partial_ctx = self.partial_context(doc, option)
            full_ctx = self.append_context(ctx, partial_ctx)
            lls.append(rf.loglikelihood(full_ctx, target)[0])
        return lls

    @classmethod
    def append_context(cls, ctx, partial_ctx):
        ctx = ctx.split("\n\n")  # Each fewshot context is on its own new line.
        ctx.pop()  # Remove the correct context put in by `doc_to_text`.
        return "\n\n".join([*ctx, partial_ctx]) if ctx else partial_ctx

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # return {
        #     "acc": np.argmax(results) == self.answer_to_num[doc["answer"]]
        # }
        pred = results >= results.max(axis=0)
        acc = 1.0 if np.argmax(pred.sum(axis=1)) == self.answer_to_num[doc["answer"]] else 0.0
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
