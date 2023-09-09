import json
import os.path
import time
from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from datasets import load_dataset, load_from_disk
from .watermark_processor import RepetitionPenaltyLogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from .utils.load_local import load_local_model_or_tokenizer
from .arg_classes.wm_arg_class import WmBaseArgs
from .watermark_processors.message_model_processor import WmProcessorMessageModel
from .watermark_processors.message_models.lm_message_model import LMMessageModel

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class HumanWrittenDecodeArgs:
    wm_args: Optional[WmBaseArgs] = None
    decode_sample_seed: int = 42
    decode_sample_num: int = 500
    decode_device: str = 'cuda:0'

    @property
    def complete_save_file_path(self):
        s = self.wm_args.complete_save_file_path
        decode_s = (f'_{self.decode_sample_seed}'
                    f'_{self.decode_sample_num}')
        s = s.replace('.json', f'{decode_s}_human_written.json')
        return s


def truncate(d, max_length=200, start_pos=0):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, start_pos:start_pos + max_length]
    return d


def main(args: HumanWrittenDecodeArgs):
    if os.path.exists(args.complete_save_file_path+ "NEW_DECODE_TIME_TEST"):
        print(f'{args.complete_save_file_path} already exists, skip.')
        return

    if args.wm_args.model_name in ['facebook/opt-1.3b', 'facebook/opt-2.7b', 'alpaca-native']:
        tokenizer = load_local_model_or_tokenizer(args.wm_args.model_name.split('/')[-1],
                                                  'tokenizer')
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.wm_args.model_name)
    else:
        raise NotImplementedError(f'model_name: {args.wm_args.model_name}')

    c4_sliced_and_filted = load_from_disk(
        os.path.join(ROOT_PATH, 'c4-train.00000-of-00512_sliced'))
    c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(
        seed=args.decode_sample_seed).select(
        range(args.decode_sample_num))

    results = {'text': [],
               'human_written_text': [],
               'decoded_message': [],
               'acc': [],
               'confidence': []}

    decoder = args.wm_args.decoder


    for text in c4_sliced_and_filted['text']:
        tokenized_input = tokenizer(text, return_tensors='pt')
        tokenized_input = truncate(tokenized_input, max_length=args.wm_args.generated_length,
                                   start_pos=args.wm_args.prompt_length)

        human_written_text = tokenizer.decode(tokenized_input['input_ids'][0])

        results['text'].append(text)
        results['human_written_text'].append(human_written_text)

    start_time = time.time()

    for human_written_text in results['human_written_text']:
        decoded_message, other_information = decoder.decode(human_written_text,
                                                            disable_tqdm=True)

    end_time = time.time()

    os.makedirs(os.path.dirname(args.complete_save_file_path), exist_ok=True)

    args_dict = {k: v for k, v in args.__dict__.items() if k != 'wm_args'}
    results['args'] = args_dict
    results['args']['wm_args'] = {k: v for k, v in args.wm_args.__dict__.items() if
                                  k not in {'_decoder'} and isinstance(v, (
                                  str, int, float, bool, list))}
    results['time'] = end_time - start_time
    print(f'TIME:', end_time - start_time)
    with open(args.complete_save_file_path + "NEW_DECODE_TIME_TEST", 'w') as f:
        json.dump(results, f, indent=4)
