import json
import os.path
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from .experiments.watermark import compute_ppl_single
from .utils.load_local import load_local_model_or_tokenizer
from .arg_classes.wm_arg_class import WmAnalysisArgs
from .watermark_processors.message_model_processor import WmProcessorMessageModel
from .watermark_processors.message_models.lm_message_model import LMMessageModel


def check_ppl_in(results):
    if 'analysis' in results and 'ppls' in results['analysis']:
        return True
    return False


def check_decoded_results_in(results):
    if 'analysis' in results and 'confidence' in results['analysis']:
        return True
    return False


def main(args: WmAnalysisArgs):
    if not os.path.exists(args.analysis_file_name):
        raise ValueError(f'analysis_file_name: {args.analysis_file_name} does not exist!')

    with open(args.analysis_file_name, 'r') as f:
        results = json.load(f)

    texts = results['text']
    prefix_and_output_texts = results['prefix_and_output_text']
    output_texts = results['output_text']

    analysis_results = results.get('analysis', {})

    if check_decoded_results_in(results) and check_ppl_in(results):
        warnings.warn('decoded_results and ppls already in results. Skipping analysis.')
        return

    if not check_ppl_in(results):
        if args.oracle_model_name == 'facebook/opt-2.7b':
            oracle_tokenizer = load_local_model_or_tokenizer(args.oracle_model_name.split('/')[-1],
                                                             'tokenizer')
            if oracle_tokenizer is None:
                oracle_tokenizer = AutoTokenizer.from_pretrained(args.oracle_model_name)
            oracle_model = load_local_model_or_tokenizer(args.oracle_model_name.split('/')[-1],
                                                         'model')
            if oracle_model is None:
                oracle_model = AutoModelForCausalLM.from_pretrained(args.oracle_model_name)
        else:
            raise NotImplementedError(f'oracle_model_name: {args.oracle_model_name}')
        oracle_model = oracle_model.to(args.device)
        losses, ppls = [], []

        for text, prefix_and_output_text, output_text in tqdm(
                zip(texts, prefix_and_output_texts, output_texts)):
            loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
                                           output_text=output_text,
                                           oracle_model_name=args.oracle_model_name,
                                           oracle_model=oracle_model,
                                           oracle_tokenizer=oracle_tokenizer)
            losses.append(loss)
            ppls.append(ppl)
        analysis_results['losses'] = losses
        analysis_results['ppls'] = ppls

    confidences = []
    decoded_messages = []
    accs = []

    if not check_decoded_results_in(results):
        try:
            decoder = args.args.decoder
        except NotImplementedError:
            decoder = None
        if decoder:
            for output_text in tqdm(output_texts):
                decoded_message, other_information = decoder.decode(output_text,
                                                                    disable_tqdm=True)

                confidence = other_information[1]
                available_message_num = args.args.generated_length // (
                    int(args.args.message_code_len * args.args.encode_ratio))
                acc = decoded_message[:available_message_num] == args.args.message[
                                                                 :available_message_num]

                confidences.append(confidence)

                decoded_messages.append(decoded_message)
                accs.append(acc)

        analysis_results['confidence'] = confidences
        analysis_results['decoded_message'] = decoded_messages
        analysis_results['acc'] = accs

    analysis_results['args'] = {k:v for k,v in vars(args).items() if k != 'args'}
    results['analysis'] = analysis_results

    with open(args.analysis_save_file_name, 'w') as f:
        json.dump(results, f, indent=4)
