import warnings
from abc import ABC, abstractmethod

class TextTooShortError(Exception):
    pass

class BaseMessageModel(ABC):
    def __init__(self, seed, message_code_len=10):
        self.seed = seed
        self.message_code_len = message_code_len

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value
        self.set_random_state()

    def set_random_state(self):
        warnings.warn("set_random_state is not implemented for this processor")

    @abstractmethod
    def get_log_probs_for_vocab(self, input_ids, cur_message, scores_shape, publich_lm_probs = None):
        '''
        :param input_ids: torch.LongTensor of shape [batch_size, seq_len]
        :param messages: int
        :param scores_shape: tuple[int, int]
        '''
        pass

    @abstractmethod
    def get_log_probs_for_messages(self, input_ids, messages, publich_lm_probs = None):
        '''
        :param input_ids: torch.LongTensor of shape [seq_len,]
        :param messages: list of int
        :param scores_shape: tuple[int, int]
        '''
        pass
