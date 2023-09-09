import json

from tqdm import tqdm

import watermarking.watermark_processor
from watermarking.utils.load_local import load_local_model_or_tokenizer
from datasets import load_dataset, load_from_disk
from watermarking.watermark_processors.message_models.lm_message_model import LMMessageModel
from watermarking.watermark_processors.message_model_processor import WmProcessorMessageModel
from watermarking.arg_classes.wm_arg_class import WmLMArgs
import random

from watermarking.attacker.substition_attack import SubstitutionAttacker

MODEL_NAME = 'roberta-large'
REPLACEMENT_PROPORTION = 0.05
ATTEMPT_RATE = 3.0
LOGIT_THRESHOLD = 1.0

config = {
    'model_name': MODEL_NAME,
    'replacement_proportion': REPLACEMENT_PROPORTION,
    'attempt_rate': ATTEMPT_RATE,
    'logit_threshold': LOGIT_THRESHOLD,
}


def main(args: WmLMArgs):
    results = args.load_result()
    tokenizer = load_local_model_or_tokenizer(args.model_name, 'tokenizer')
    lm_tokenizer = load_local_model_or_tokenizer(args.lm_model_name, 'tokenizer')
    lm_model = load_local_model_or_tokenizer(args.lm_model_name, 'model')
    lm_model = lm_model.to(args.device)
    lm_message_model = LMMessageModel(tokenizer=tokenizer, lm_model=lm_model,
                                      lm_tokenizer=lm_tokenizer,
                                      delta=1., lm_prefix_len=10, lm_topk=-1, message_code_len=20,
                                      random_permutation_num=100)
    wm_precessor_message_model = WmProcessorMessageModel(message_model=lm_message_model,
                                                         tokenizer=tokenizer,
                                                         encode_ratio=10, max_confidence_lbd=0.5,
                                                         strategy='max_confidence',
                                                         message=[42, 43, 46, 47, 48, 49, 50, 51,
                                                                  52, 53])

    random.seed(args.sample_seed)

    substitution_attacker = SubstitutionAttacker(model_name=MODEL_NAME,
                                                 replacement_proportion=REPLACEMENT_PROPORTION,
                                                 attempt_rate=ATTEMPT_RATE,
                                                 logit_threshold=LOGIT_THRESHOLD,
                                                 device=args.device)

    x_wms = results['output_text']

    decoded_results = []
    substituted_texts = []
    substituted_text_other_informations = []
    corrected = []
    for i in tqdm(range(len(x_wms))):
        substituted_text, other_information = substitution_attacker.attack(x_wms[i])
        substituted_texts.append(substituted_text)
        substituted_text_other_informations.append(other_information)
        y = wm_precessor_message_model.decode(substituted_text, disable_tqdm=True)
        corrected.append(y[0] == wm_precessor_message_model.message.tolist()[:len(y)])
        decoded_results.append([y[0], y[1][1]])
    successful_rate = sum(corrected) / len(corrected)

    if f'sub-analysis-{REPLACEMENT_PROPORTION}-{ATTEMPT_RATE}-{LOGIT_THRESHOLD}-{MODEL_NAME}' not in results:
        analysis_results = {}
        analysis_results['successful_rate'] = successful_rate
        analysis_results['decoded_results'] = decoded_results
        analysis_results['config'] = config
        analysis_results['substituted_texts'] = substituted_texts
        analysis_results[
            'substituted_text_other_informations'] = substituted_text_other_informations
        results[
            f'sub-analysis-{REPLACEMENT_PROPORTION}-{ATTEMPT_RATE}-{LOGIT_THRESHOLD}-{MODEL_NAME}'] = analysis_results

    # print(analysis_results)
    with open(args.complete_save_file_path, 'w') as f:
        json.dump(results, f, indent=4)
