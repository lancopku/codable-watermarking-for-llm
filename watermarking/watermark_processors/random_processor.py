import torch
from tqdm import tqdm

from .base_processor import WmProcessorBase
from ..utils.hash_fn import Hash1


class WmProcessorRandom(WmProcessorBase):
    def __init__(self, message, tokenizer, delta=1.0, start_length=0, seed=42, message_code_len=10,
                 top_k=1000, encode_ratio=10., hash_fn=Hash1, device='cuda:0'):
        super().__init__(seed)
        self.message = message
        self.message_code_len = message_code_len
        self.start_length = start_length
        self.hash_fn = hash_fn
        self.top_k = top_k
        self.lm_prefix_len = 1
        self.encode_ratio = encode_ratio
        self.encode_len = int(self.message_code_len * self.encode_ratio)
        self.delta = delta
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.get_vocab().values())

    def set_random_state(self):
        pass

    def random_num(self, prefix_x, x, time):
        h = (prefix_x + x * self.vocab_size) + time * self.vocab_size * self.vocab_size
        return int(self.hash_fn(h))

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] < self.start_length + self.lm_prefix_len:
            return scores
        input_ids = input_ids[:, self.start_length:]
        if self.top_k > 0:
            possible_tokens = scores.topk(self.top_k, dim=-1)[1]
        else:
            possible_tokens = torch.arange(scores.shape[1], device=self.device).unsqueeze(0).repeat(
                input_ids.shape[0], 1)
        message_len = input_ids.shape[1] - self.lm_prefix_len
        cur_message = self.message[message_len // self.encode_len]
        value = self.hash_fn(input_ids[:, -1:] + possible_tokens * self.vocab_size +
                             cur_message * self.vocab_size * self.vocab_size)
        value = value % 2
        for b_idx in range(input_ids.shape[0]):
            scores[b_idx][possible_tokens[b_idx][value[b_idx] == 1]] += 0.5 * self.delta
            scores[b_idx][possible_tokens[b_idx][value[b_idx] == 0]] -= 0.5 * self.delta
        return scores

    def get_seeds(self, input_ids, confidence=None, threshold=None, start_length=0):
        assert start_length >= 0
        seeds = input_ids[:-1] + input_ids[1:] * self.vocab_size
        if confidence is not None:
            confidence = confidence.detach().clone()
            if len(confidence) == len(input_ids):
                confidence = confidence[1:]
            confidence[:start_length] = 0.
            seeds = seeds[confidence < threshold]
        return seeds.to(self.device)

    def decode(self, text, messages=None, disable_tqdm=False):
        if messages is None:
            messages = torch.arange(2 ** self.message_code_len).to(self.device)
        token_ids = self.tokenizer(text, return_tensors='pt')['input_ids'][0].to(self.device)
        seeds = self.get_seeds(token_ids)
        messages_seeds = messages * self.vocab_size * self.vocab_size
        cnts = []
        batch_size = 1
        for i in tqdm(range(0, len(seeds), batch_size), disable=disable_tqdm):
            cnt = self.hash_fn(messages_seeds.unsqueeze(-1) + seeds[i:i + batch_size].unsqueeze(0)).sum(
                -1) % 2
            cnts.append(cnt.unsqueeze(0))
        cnts = torch.cat(cnts, dim=0)
        decoded_messages = []
        for i in range(0, cnts.shape[0], self.encode_len):
            decoded_message = messages[torch.argmax(cnts[i:i + self.encode_len].sum(0))]
            decoded_messages.append(decoded_message)
        decoded_messages = [int(_) for _ in decoded_messages]
        cnts = cnts * self.delta - 0.5 * self.delta
        return decoded_messages, cnts.cpu()
