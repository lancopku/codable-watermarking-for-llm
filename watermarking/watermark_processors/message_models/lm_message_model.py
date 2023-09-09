import warnings
from typing import Union, Optional, List

import numpy as np
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
import torch

from .base_message_model_fast import BaseMessageModelFast, TextTooShortError
from ...utils.hash_fn import Hash1, random_shuffle
from ...utils.random_utils import np_temp_random

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]


class LMPredictions:
    def __init__(self, public_lm_probs: torch.Tensor, delta, top_k: int = -1, cache_A=True):
        '''
        :param public_lm_probs: torch.FloatTensor of shape [batcs_size, vocab_size]
        '''
        assert public_lm_probs.dim() == 2

        self.public_lm_probs = public_lm_probs
        self.top_k = top_k
        if self.top_k > 0:
            raise NotImplementedError
        self.vocab_size = public_lm_probs.shape[-1]
        self.cache_A = cache_A
        self.delta = delta

    @property
    def topk_indices(self) -> torch.Tensor:
        if self.top_k == -1:
            raise ValueError("topk is not set")
        if hasattr(self, "_topk_indices"):
            return self._topk_indices
        self._topk_indices = torch.topk(self.public_lm_probs, self.top_k, dim=-1).indices
        return self._topk_indices

    @property
    def topk_public_lm_probs(self) -> torch.Tensor:
        if self.top_k == -1:
            raise ValueError("topk is not set")
        if hasattr(self, "_topk_public_lm_probs"):
            return self._topk_public_lm_probs
        self._topk_public_lm_probs = torch.torch.gather(self.public_lm_probs, -1, self.topk_indices)
        self._topk_public_lm_probs = self._topk_public_lm_probs / self._topk_public_lm_probs.sum(-1,
                                                                                                 keepdim=True)
        return self._topk_public_lm_probs

    def As_for_message(self, batch_idx: int, seeds: torch.Tensor) -> List[torch.Tensor]:
        '''
        :param batch_idx: int
        :param seeds: torch.LongTensor of shape [num_seeds]
        :param x_cur_idx: int or None

        :return: As, List[torch.FloatTensor] of shape [num_seeds, vocab_size]
        or [x_cur_id in A for A in As], torch.FloatTensor of shape [vocab_size]
        '''
        warnings.warn('Deprecated, use P_message_single instead')
        assert seeds.dim() == 1
        As = [None] * len(seeds)
        if self.cache_A:
            if not hasattr(self, "_A_for_message_dict"):
                self._A_for_message_dict = {}
            if batch_idx not in self._A_for_message_dict:
                self._A_for_message_dict[batch_idx] = {}
            for i, seed in enumerate(seeds):
                if int(seed) in self._A_for_message_dict[batch_idx]:
                    As[i] = self._A_for_message_dict[batch_idx][int(seed)]
        if self.cache_A:
            uncached_seeds_idxs = [i for i, seed in enumerate(seeds) if As[i] is None]
            shuffle_idxs = random_shuffle(key=seeds[uncached_seeds_idxs],
                                          n=self.top_k if self.top_k > 0 else self.vocab_size,
                                          device=self.public_lm_probs.device)
        else:
            shuffle_idxs = random_shuffle(key=seeds,
                                          n=self.top_k if self.top_k > 0 else self.vocab_size,
                                          device=self.public_lm_probs.device)
            uncached_seeds_idxs = range(len(seeds))

        if shuffle_idxs.dim() == 1:
            shuffle_idxs = shuffle_idxs.unsqueeze(0)
        single_lm_probs = self.topk_public_lm_probs[batch_idx:batch_idx + 1] if self.top_k > 0 \
            else self.public_lm_probs[batch_idx:batch_idx + 1]
        single_lm_probs = single_lm_probs.expand(shuffle_idxs.shape[0], -1)
        shuffled_lm_probs = single_lm_probs.gather(-1, shuffle_idxs)

        shuffled_lm_probs_cum_sum = shuffled_lm_probs.cumsum(-1)
        # select the first half of the shuffled tokens
        split_idxs = (shuffled_lm_probs_cum_sum <= 0.5).sum(-1) + 1
        for i, uncached_seeds_idx in enumerate(uncached_seeds_idxs):
            split_idx = split_idxs[i]
            if self.top_k < 0:
                A = shuffle_idxs[i, :split_idx]
            else:
                A = self.topk_indices[batch_idx][shuffle_idxs[i, :split_idx]]
            if self.cache_A:
                self._A_for_message_dict[batch_idx][int(seeds[uncached_seeds_idx])] = A
            As[uncached_seeds_idx] = A

        return As

    def P_message_single(self, seeds: torch.Tensor) -> torch.Tensor:
        '''
        :param batch_idx: int
        :param seeds: torch.LongTensor of shape [bsz]

        :return: P_message, torch.FloatTensor of shape [bsz, vocab_size]
        '''
        single_lm_probs = self.topk_public_lm_probs if self.top_k > 0 \
            else self.public_lm_probs

        # print(seeds.shape)
        assert seeds.dim() == 1 and seeds.numel() == single_lm_probs.shape[0]

        shuffle_idxs = random_shuffle(key=seeds,
                                      n=self.top_k if self.top_k > 0 else self.vocab_size,
                                      device=self.public_lm_probs.device)

        if shuffle_idxs.dim() == 1:
            shuffle_idxs = shuffle_idxs.unsqueeze(0)
        shuffled_lm_probs = single_lm_probs.gather(-1, shuffle_idxs)

        shuffled_lm_probs_black_mask = (shuffled_lm_probs.cumsum(-1) < 0.5).float() * (-self.delta)
        # TO DO here we change single_lm_probs in-place!
        p_message = single_lm_probs.scatter_(dim=-1, index=shuffle_idxs,
                                             src=shuffled_lm_probs_black_mask)

        return p_message

    def is_in_As_for_message(self, batch_idx: int, seeds: torch.Tensor,
                             x_cur_idx: int) -> \
            Union[List[torch.Tensor], torch.Tensor]:
        '''
        :param batch_idx: int
        :param seeds: torch.LongTensor of shape [num_seeds]
        :param x_cur_idx: int or None

        :return: [x_cur_id in A for A in As], torch.FloatTensor of shape [vocab_size]
        '''
        assert seeds.dim() == 1

        shuffle_idxs = random_shuffle(key=seeds,
                                      n=self.top_k if self.top_k > 0 else self.vocab_size,
                                      device=self.public_lm_probs.device)

        if shuffle_idxs.dim() == 1:
            shuffle_idxs = shuffle_idxs.unsqueeze(0)

        single_lm_probs = self.topk_public_lm_probs[batch_idx:batch_idx + 1] if self.top_k > 0 \
            else self.public_lm_probs[batch_idx:batch_idx + 1]
        single_lm_probs = single_lm_probs.expand(shuffle_idxs.shape[0], -1)
        shuffled_lm_probs = single_lm_probs.gather(-1, shuffle_idxs)

        eq_mask = torch.eq(shuffle_idxs, x_cur_idx)
        left_mask = 1 - eq_mask.cumsum(-1) + eq_mask
        isinAs = (shuffled_lm_probs * left_mask).sum(-1) >= 0.5

        # left_mask = 1 - eq_mask.cumsum(-1)
        # isinAs = (shuffled_lm_probs * left_mask).sum(-1) < 0.5
        return isinAs

    def clear_cache(self):
        if hasattr(self, "_A_for_message_dict"):
            del self._A_for_message_dict


class SeedCacher:
    def __init__(self):
        self._cache = []
        self._curs = []

    def add(self, seeds):
        self._cache.extend(seeds)

    def add_cur(self,cur):
        self._curs.extend(cur)

    def clear(self):
        self._cache = []
        self._curs = []

    def data(self):
        return self._cache, self._curs


class LMMessageModel(BaseMessageModelFast):
    def __init__(self, tokenizer: HfTokenizer, lm_model, lm_tokenizer: HfTokenizer,
                 delta, lm_prefix_len, seed=42, lm_topk=1000, message_code_len=10,
                 random_permutation_num=100, hash_prefix_len=1, hash_fn=Hash1):
        super().__init__(seed, message_code_len)
        self.tokenizer = tokenizer
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.delta = delta
        self.lm_prefix_len = lm_prefix_len
        self.message_code_len = message_code_len
        self.lm_top_k = lm_topk
        if self.lm_top_k > 0:
            raise NotImplementedError
        self.hash_prefix_len = hash_prefix_len
        self.hash_fn = hash_fn
        self.random_permutation_num = random_permutation_num
        self.tokenizer_id_2_lm_tokenizer_id_list = self.construct_tokenizer_id_2_lm_tokenizer_id_list()
        self.random_permutation_idxs = torch.arange(self.random_permutation_num).unsqueeze(0)
        self.max_vocab_size = 100000
        self.seed_cacher = SeedCacher()

    @property
    def random_delta(self):
        return self.delta * 1.

    @property
    def device(self):
        return self.lm_model.device

    def set_random_state(self):
        pass

    def construct_tokenizer_id_2_lm_tokenizer_id_list(self):
        tokenizer_id_2_lm_tokenizer_id_list = torch.zeros(len(self.tokenizer.vocab),
                                                          dtype=torch.long)
        for v, i in self.tokenizer.vocab.items():
            tokenizer_id_2_lm_tokenizer_id_list[i] = self.lm_tokenizer.convert_tokens_to_ids(v)
        return tokenizer_id_2_lm_tokenizer_id_list.to(self.lm_model.device)

    def check_x_prefix_x_cur_messages(self, x_prefix: Optional[torch.Tensor] = None,
                                      x_cur: Optional[torch.Tensor] = None,
                                      messages: Optional[torch.Tensor] = None):
        if x_prefix is not None:
            assert x_prefix.dim() == 2
        if x_cur is not None:
            assert x_cur.dim() == 2
        if messages is not None:
            assert messages.dim() == 1
        if x_prefix is not None and x_cur is not None:
            assert x_prefix.shape[0] == x_cur.shape[0]

    def filter(self, texts):
        warnings.warn("filter is Deprecated.")

        filtered_bools = torch.full((len(texts),), False, dtype=torch.bool, device=self.device)
        filtered_texts = []
        for i, text in enumerate(texts):
            tokens = self.lm_tokenizer.tokenize(text, add_special_tokens=False)
            if len(tokens) < self.lm_prefix_len:
                filtered_bools[i] = False
                continue
            else:
                filtered_bools[i] = True
                filtered_texts.append(text)
        return filtered_texts, filtered_bools

    def get_lm_predictions(self, strings, return_log_softmax=False, input_lm_token_ids = False):
        ori_pad_side = self.lm_tokenizer.padding_side
        self.lm_tokenizer.padding_side = 'left'
        encodings = self.lm_tokenizer(strings, return_tensors="pt", padding=True,
                                      add_special_tokens=False)
        self.lm_tokenizer.padding_side = ori_pad_side
        if encodings['input_ids'].shape[1] < self.lm_prefix_len:
            # raise TextTooShortError
            warnings.warn("TextTooShortError")
        encodings['input_ids'] = encodings['input_ids'][:, -self.lm_prefix_len:]
        encodings['attention_mask'] = encodings['attention_mask'][:, -self.lm_prefix_len:]
        encodings = encodings.to(self.lm_model.device)
        with torch.no_grad():
            logits = self.lm_model(**encodings).logits
        final_pos = (encodings['input_ids'] != self.lm_tokenizer.pad_token_id).sum(dim=1) - 1
        logits = logits[torch.arange(logits.shape[0]), final_pos]
        if return_log_softmax:
            logit_or_probs = torch.log_softmax(logits, dim=-1)
        else:
            logit_or_probs = torch.softmax(logits, dim=-1)
        # if input_ids.shape[1] < self.lm_prefix_len:
        #     raise TextTooShortError
        # input_ids = input_ids[:, -self.lm_prefix_len:]
        # if not input_lm_token_ids:
        #     strings = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # else:
        #     strings = self.lm_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # encodings = self.lm_tokenizer(strings, return_tensors="pt", padding=True)
        # encodings = encodings.to(self.lm_model.device)
        # with torch.no_grad():
        #     logits = self.lm_model(**encodings).logits
        # final_pos = (encodings['input_ids'] != self.lm_tokenizer.pad_token_id).sum(dim=1) - 1
        # logits = logits[torch.arange(logits.shape[0]), final_pos]
        # # if not keep_lm:
        # #     logits = logits[:, self.tokenizer_id_2_lm_tokenizer_id_list]
        #
        # if return_log_softmax:
        #     logit_or_probs = torch.log_softmax(logits, dim=-1)
        # else:
        #     logit_or_probs = torch.softmax(logits, dim=-1)
        # For test
        # return torch.full_like(logit_or_probs, fill_value=1. / logit_or_probs.shape[-1],
        #                        device=self.device)

        return logit_or_probs

    def get_x_prefix_seed_element(self, x_prefix, transform: bool):
        self.check_x_prefix_x_cur_messages(x_prefix=x_prefix)
        if x_prefix.shape[1] < self.hash_prefix_len:
            raise TextTooShortError
        x_prefix = x_prefix[:, -self.hash_prefix_len:]
        if transform:
            x_prefix = self.tokenizer_id_2_lm_tokenizer_id_list[x_prefix]
        seed_elements = [tuple(_.tolist()) for _ in x_prefix]
        # print(seed_elements)
        return seed_elements

    def get_x_prefix_seeds(self, x_prefix, transform: bool):
        '''
        :param x_prefix: [batch_size, seq_len]

        :return: seeds: list of [batch_size]
        '''
        self.check_x_prefix_x_cur_messages(x_prefix=x_prefix)
        seed_elements = self.get_x_prefix_seed_element(x_prefix, transform=transform)
        seeds = torch.tensor([hash(seed_element + (self.seed,)) for seed_element in seed_elements],
                             device=x_prefix.device)

        self.seed_cacher.add(seeds.tolist())

        return seeds

    def get_x_prefix_A_seeds(self, x_prefix_seeds: torch.Tensor,
                             random_permutation_idxs: torch.Tensor = None):
        '''
        :param x_prefix: [batch_size, seq_len]
        :param x_prefix_seeds: [batch_size]
        :param random_permutation_idxs: [batch_size, random_permutation_num] or [1, random_permutation_num]

        :return: seeds: list of [batch_size, self.random_permutation_num]
        '''


        assert x_prefix_seeds.dim() == 1
        if random_permutation_idxs is not None:
            assert random_permutation_idxs.dim() == 2
            assert random_permutation_idxs.shape[0] == 1 or x_prefix_seeds.shape[0] == \
                   random_permutation_idxs.shape[0]

        if random_permutation_idxs is None:
            self.random_permutation_idxs = self.random_permutation_idxs.to(x_prefix_seeds.device)
            random_permutation_idxs = self.random_permutation_idxs
        else:
            random_permutation_idxs = random_permutation_idxs.to(x_prefix_seeds.device)
        seeds = self.hash_fn(
            x_prefix_seeds.unsqueeze(1) * self.random_permutation_num + random_permutation_idxs)
        return seeds

    def get_x_prefix_and_message_seeds(self, messages: torch.Tensor,
                                       x_prefix_seeds: torch.Tensor):
        '''
        :param message_or_messages: [message_num]

        :return: [batch_size, message_num]
        '''
        seeds = (x_prefix_seeds % (2 ** (32 - self.message_code_len))).unsqueeze(1)
        seeds = seeds * (2 ** self.message_code_len) + messages.unsqueeze(0)
        seeds = self.hash_fn(seeds)
        return seeds

    def get_x_prefix_and_message_and_x_cur_seeds(self, x_prefix: torch.Tensor,
                                                 messages: torch.Tensor, x_cur: torch.Tensor,
                                                 x_prefix_seeds: torch.Tensor,
                                                 x_prefix_and_message_seeds: torch.Tensor = None):
        '''
        :param x_prefix: [batch_size, seq_len]
        :param message_or_messages: [message_num]
        :param x_cur: [batch_size, vocab_size]

        return: [batch_size, message_num, vocab_size]
        '''
        assert x_cur.dim() == 2 and x_prefix.dim() == 2
        if x_prefix_and_message_seeds is None:
            x_prefix_and_message_seeds = self.get_x_prefix_and_message_seeds(messages,
                                                                             x_prefix_seeds)
        x_prefix_and_message_and_x_cur_seeds = self.hash_fn(x_prefix_and_message_seeds.unsqueeze(
            -1) * self.max_vocab_size + x_cur.unsqueeze(1))
        return x_prefix_and_message_and_x_cur_seeds

    def cal_log_P_with_single_message(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                                      messages: torch.Tensor,
                                      lm_predictions: Optional[torch.Tensor],
                                      transform_x_cur: bool = True) -> torch.Tensor:
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num]
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''
        assert messages.numel() == 1
        self.check_x_prefix_x_cur_messages(x_prefix=x_prefix, x_cur=x_cur, messages=messages)

        if transform_x_cur:
            x_cur = self.tokenizer_id_2_lm_tokenizer_id_list[x_cur]

        x_prefix_seeds = self.get_x_prefix_seeds(x_prefix, transform=True)
        x_prefix_and_message_seeds = self.get_x_prefix_and_message_seeds(
            messages,
            x_prefix_seeds=x_prefix_seeds)


        lm_predictions = LMPredictions(lm_predictions, delta=self.delta, top_k=self.lm_top_k,
                                       cache_A=False)
        x_prefix_and_message_idxs = x_prefix_and_message_seeds % self.random_permutation_num
        x_prefix_and_A_seeds = self.get_x_prefix_A_seeds(x_prefix_seeds=x_prefix_seeds,
                                                         random_permutation_idxs=x_prefix_and_message_idxs)

        # x_prefix_and_message_and_x_cur_seeds = self.get_x_prefix_and_message_and_x_cur_seeds(
        #     x_prefix, messages, x_cur, x_prefix_seeds=x_prefix_seeds,
        #     x_prefix_and_message_seeds=x_prefix_and_message_seeds)
        # x_prefix_and_message_and_x_cur_idxs = x_prefix_and_message_and_x_cur_seeds % 2

        if self.lm_top_k > 0:
            # log_Ps = x_prefix_and_message_and_x_cur_idxs.float() * self.random_delta
            raise NotImplementedError
        else:
            assert x_prefix_and_A_seeds.shape[1] == 1
            all_log_Ps = lm_predictions.P_message_single(seeds=x_prefix_and_A_seeds[:, 0])
            # print(all_log_Ps[1, :10])
            log_Ps = all_log_Ps.gather(dim=1, index=x_cur)
            log_Ps = log_Ps.unsqueeze(1)
            # log_Ps = torch.zeros((x_prefix.shape[0], 1, x_cur.shape[1]), device=x_cur.device)
            # for batch_idx in range(x_prefix.shape[0]):
            #     As = lm_predictions.As_for_message(batch_idx=batch_idx,
            #                                        seeds=x_prefix_and_A_seeds[batch_idx])
            #     is_inA = torch.isin(x_cur[batch_idx], As[0])
            #     log_Ps[batch_idx, 0, is_inA] = self.delta

        return log_Ps

    def cal_log_P_with_single_x_cur(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                                    messages: torch.Tensor,
                                    lm_predictions: Optional[torch.Tensor]):
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num]
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''
        assert x_cur.shape[1] == 1

        x_prefix_seeds = self.get_x_prefix_seeds(x_prefix, transform=False)

        lm_predictions = LMPredictions(lm_predictions, delta=self.delta, top_k=self.lm_top_k,
                                       cache_A=False)
        x_prefix_and_A_seeds = self.get_x_prefix_A_seeds(x_prefix_seeds=x_prefix_seeds)
        x_prefix_and_message_seeds = self.get_x_prefix_and_message_seeds(messages,
                                                                         x_prefix_seeds=x_prefix_seeds)
        x_prefix_and_message_idxs = x_prefix_and_message_seeds % self.random_permutation_num
        log_Ps = []
        for batch_idx in range(x_prefix.shape[0]):
            if self.lm_top_k < 0:
                x_cur_idx = int(x_cur[batch_idx, 0])
            else:
                x_cur_idx = \
                    torch.where(lm_predictions.topk_indices[batch_idx] == x_cur[batch_idx, 0])[0]
                if len(x_cur_idx) > 0:
                    x_cur_idx = int(x_cur_idx[0])
                else:
                    x_cur_idx = None
            if x_cur_idx is not None:
                in_A_s = lm_predictions.is_in_As_for_message(batch_idx=batch_idx,
                                                             seeds=x_prefix_and_A_seeds[batch_idx],
                                                             x_cur_idx=x_cur_idx)
                is_in_certain_A = torch.gather(in_A_s, 0, x_prefix_and_message_idxs[batch_idx])
                single_log_Ps = is_in_certain_A.float() * self.delta
                single_log_Ps = single_log_Ps.reshape(1, -1, 1)
                # print('in top_k')
            else:
                raise NotImplementedError
                x_prefix_and_message_and_x_cur_seeds = self.get_x_prefix_and_message_and_x_cur_seeds(
                    x_prefix[batch_idx:batch_idx + 1], messages,
                    x_cur[batch_idx:batch_idx + 1],
                    x_prefix_seeds=x_prefix_seeds[batch_idx:batch_idx + 1])
                x_prefix_and_message_and_x_cur_idxs = x_prefix_and_message_and_x_cur_seeds % 2
                single_log_Ps = x_prefix_and_message_and_x_cur_idxs.float() * self.random_delta
                # print(single_log_Ps.shape)
            log_Ps.append(single_log_Ps)
        log_Ps = torch.cat(log_Ps, dim=0)
        return log_Ps

    def cal_log_Ps(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                   messages: torch.Tensor,
                   lm_predictions: torch.Tensor):
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num]
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''

        # check input shape
        self.check_x_prefix_x_cur_messages(x_prefix, x_cur, messages)
        assert x_prefix.shape[0] == lm_predictions.shape[0]

        if messages.numel() == 1:
            # cal P(x_prefix, x_cur, message) for all possible x_cur
            log_Ps = self.cal_log_P_with_single_message(x_prefix, x_cur,
                                                        messages,
                                                        lm_predictions)

        elif x_cur.shape[1] == 1:
            log_Ps = self.cal_log_P_with_single_x_cur(x_prefix, x_cur, messages, lm_predictions)
        else:
            raise NotImplementedError

        return log_Ps
