import torch
from transformers import LogitsProcessor

class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    changed from huggingface (original version multiple by penalty)
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.
    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    changed based on discussion on github (https://github.com/huggingface/transformers/pull/2303)
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty >= 0):
            raise ValueError(f"`penalty` has to be a non-negasitive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        score = score - self.penalty

        scores.scatter_(1, input_ids, score)
        return scores
