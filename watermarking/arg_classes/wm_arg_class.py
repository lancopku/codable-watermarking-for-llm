import os
from dataclasses import dataclass, field
from typing import Optional, List

from watermarking.utils.load_local import load_local_model_or_tokenizer
from watermarking.watermark_processors.message_model_processor import WmProcessorMessageModel
from watermarking.watermark_processors.message_models.lm_message_model import LMMessageModel
from watermarking.watermark_processors.message_models.random_message_model import RandomMessageModel
from watermarking.watermark_processors.random_message_model_processor import \
    WmProcessorRandomMessageModel

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



@dataclass
class WmBaseArgs:
    temperature: float = 1.0
    model_name: str = "facebook/opt-1.3b"
    sample_num: int = 100
    sample_seed: int = 42
    seed: int = 42
    num_beams: int = 4
    delta: float = 1.5
    repeat_penalty: float = 1.5
    message: List[int] = field(default_factory=lambda: [42, 43, 46, 47, 48, 49, 50, 51, 52, 53])
    prompt_length: int = 300
    generated_length: int = 200
    message_code_len: int = 20
    encode_ratio: float = 10.
    device: str = 'cuda:0'
    root_path: str = os.path.join(ROOT_PATH, 'my_watermark_result')
    wm_strategy: str = "base"

    def __post_init__(self):
        if self.message_code_len == 5:
            self.message = [10, 11, 14, 15, 16, 17, 18, 19]

    @property
    def sub_root_path(self):
        return self.wm_strategy

    @property
    def save_file_name(self):
        model_name = self.model_name.replace('/', '-')
        s = (
            f'{model_name}_{self.delta}_'
            f'{self.repeat_penalty}_{self.prompt_length}_{self.generated_length}'
            f'_{self.sample_num}_{self.sample_seed}_{self.seed}'
            f'_{self.message_code_len}_{self.encode_ratio}'
            f'_{self.num_beams}_{self.temperature}')
        return s

    @property
    def special_file_name(self):
        return ''

    @property
    def complete_save_file_path(self):
        complete_save_file_name = self.save_file_name + self.special_file_name + '.json'
        complete_save_file_path = os.path.join(self.root_path, self.sub_root_path,
                                               complete_save_file_name)
        return complete_save_file_path

    def load_result(self):
        import json
        with open(self.complete_save_file_path, 'r') as f:
            result = json.load(f)
        return result

    def get_decoder(self):
        raise NotImplementedError

    @property
    def decoder(self):
        if hasattr(self, '_decoder'):
            return self._decoder
        else:
            self._decoder = self.get_decoder()
            return self._decoder


@dataclass
class WmNoneArgs(WmBaseArgs):
    wm_strategy: str = "none"


@dataclass
class WmRandomArgs(WmBaseArgs):
    prefix_len: int = 1
    top_k: int = 1000
    wm_strategy: str = "random_7_8"
    lm_model_name: str = 'gpt2'

    @property
    def special_file_name(self):
        s = (f'_{self.prefix_len}_{self.top_k}'
             f'_{self.lm_model_name}')
        return s

    def _load_and_tokenizer(self):
        if self.lm_model_name == 'same':
            raise NotImplementedError
        else:
            lm_tokenizer = load_local_model_or_tokenizer(self.lm_model_name.split('/')[-1],
                                                         'tokenizer')
            return lm_tokenizer

    def get_decoder(self, lm_tokenizer=None):
        if lm_tokenizer is None:
            lm_tokenizer = self._load_and_tokenizer()
        lm_message_model = RandomMessageModel(tokenizer=lm_tokenizer,
                                              lm_tokenizer=lm_tokenizer,
                                              delta=self.delta,
                                              message_code_len=self.message_code_len,
                                              device=self.device,
                                              )

        watermark_processor = WmProcessorRandomMessageModel(message_model=lm_message_model,
                                                            tokenizer=lm_tokenizer,
                                                            encode_ratio=self.encode_ratio,
                                                            message=self.message,
                                                            top_k=self.top_k,
                                                            )
        return watermark_processor

@dataclass
class WmRandomNoHashArgs(WmRandomArgs):
    wm_strategy: str = "random_7_8_no_hash"

@dataclass
class WmLMArgs(WmBaseArgs):
    lm_prefix_len: int = 10
    lm_top_k: int = -1
    lm_model_name: str = 'gpt2'
    wm_strategy: str = "lm_new_7_10"
    message_model_strategy: str = 'vanilla'
    random_permutation_num: int = 100
    max_confidence_lbd: float = 0.5
    top_k: int = 1000

    @property
    def special_file_name(self):
        s = (
            f'_{self.lm_prefix_len}_{self.lm_top_k}'
            f'_{self.random_permutation_num}'
            f'_{self.message_model_strategy}'
            f'_{self.max_confidence_lbd}'
            f'_{self.lm_model_name}'
            f'_{self.top_k}'
        )
        return s

    def _load_model_and_tokenizer(self):
        if self.lm_model_name == 'same':
            raise NotImplementedError
        else:
            lm_model = load_local_model_or_tokenizer(self.lm_model_name.split('/')[-1], 'model')
            lm_tokenizer = load_local_model_or_tokenizer(self.lm_model_name.split('/')[-1],
                                                         'tokenizer')
            lm_model = lm_model.to(self.device)
            return lm_model, lm_tokenizer

    def get_decoder(self, lm_tokenizer=None, lm_model=None) -> WmProcessorMessageModel:
        if lm_tokenizer is None or lm_model is None:
            lm_model, lm_tokenizer = self._load_model_and_tokenizer()
        lm_message_model = LMMessageModel(tokenizer=lm_tokenizer,
                                          lm_model=lm_model,
                                          lm_tokenizer=lm_tokenizer,
                                          delta=self.delta,
                                          lm_prefix_len=self.lm_prefix_len,
                                          lm_topk=self.lm_top_k,
                                          message_code_len=self.message_code_len,
                                          random_permutation_num=self.random_permutation_num)

        watermark_processor = WmProcessorMessageModel(message_model=lm_message_model,
                                                      tokenizer=lm_tokenizer,
                                                      encode_ratio=self.encode_ratio,
                                                      max_confidence_lbd=self.max_confidence_lbd,
                                                      strategy=self.message_model_strategy,
                                                      message=self.message,
                                                      top_k=self.top_k,
                                                      )
        return watermark_processor


@dataclass
class WmLMNewArgs(WmLMArgs):
    wm_strategy: str = "lm_no_hash_7_10"

    def get_decoder(self, lm_tokenizer=None, lm_model=None) -> WmProcessorMessageModel:
        decoder = super().get_decoder(lm_tokenizer, lm_model)
        import torch
        decoder.message_model.get_x_prefix_seeds = lambda x, transform: torch.ones(x.shape[0],
                                                                                   device=x.device,
                                                                                   dtype=torch.long)
        return decoder


@dataclass
class WmAnalysisArgs:
    analysis_file_name: str
    oracle_model_name: str = "facebook/opt-2.7b"
    device: str = 'cuda:0'
    args: Optional[WmBaseArgs] = None

    @property
    def analysis_save_file_name(self):
        return self.analysis_file_name
