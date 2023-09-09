import json
import os.path
import random
import warnings
from dataclasses import dataclass

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .arg_classes.wm_arg_class import WmBaseArgs
from .attacker.substition_attack import SubstitutionAttacker
from .experiments.watermark import compute_ppl_single
from .utils.load_local import load_local_model_or_tokenizer


@dataclass
class SubstitutionArgs:
    sub_model_name = 'roberta-large'
    sub_replacement_proportion: float = 0.1
    sub_attempt_rate: float = 3.0
    sub_logit_threshold: float = 1.0
    oracle_model_name = 'facebook/opt-2.7b'
    sub_normalize: bool = True

    @property
    def sub_result_name(self):
        s = (f'sub-analysis-{self.sub_replacement_proportion}-{self.sub_attempt_rate}'
             f'-{self.sub_logit_threshold}-{self.sub_model_name}')
        if self.sub_normalize:
            s+= '-normalized'
        return s


def check_sub_message_in(results, sub_args: SubstitutionArgs):
    if sub_args.sub_result_name in results and 'decoded_results' in results[
        sub_args.sub_result_name]:
        return True
    return False


def check_sub_ppl_in(results, sub_args: SubstitutionArgs):
    if sub_args.sub_result_name in results and 'sub_ppls' in results[
        sub_args.sub_result_name]:
        return True
    return False


def main(args: WmBaseArgs, sub_args: SubstitutionArgs):
    if not os.path.exists(args.complete_save_file_path):
        raise ValueError(f'analysis_file_name: {args.complete_save_file_path} does not exist!')

    with open(args.complete_save_file_path, 'r') as f:
        results = json.load(f)

    if check_sub_message_in(results, sub_args) and check_sub_ppl_in(results, sub_args):
        warnings.warn('substitution already in results. Skipping analysis.')
        return

    sub_analysis_results = results.get(sub_args.sub_result_name, {})

    if not check_sub_message_in(results, sub_args):
        random.seed(args.sample_seed)

        decoder = args.decoder
        substitution_attacker = SubstitutionAttacker(model_name=sub_args.sub_model_name,
                                                     replacement_proportion=sub_args.sub_replacement_proportion,
                                                     attempt_rate=sub_args.sub_attempt_rate,
                                                     logit_threshold=sub_args.sub_logit_threshold,
                                                     device=args.device,
                                                     normalize=sub_args.sub_normalize)

        x_wms = results['output_text']

        decoded_results = []
        substituted_texts = []
        substituted_text_other_informations = []
        corrected = []
        for i in tqdm(range(len(x_wms))):
            substituted_text, other_information = substitution_attacker.attack(x_wms[i])
            substituted_texts.append(substituted_text)
            substituted_text_other_informations.append(other_information)
            y = decoder.decode(substituted_text, disable_tqdm=True)
            corrected.append(y[0] == decoder.message.tolist()[:len(y)])
            decoded_results.append([y[0], y[1][1]])
        successful_rate = sum(corrected) / len(corrected)

        sub_analysis_results['successful_rate'] = successful_rate
        sub_analysis_results['decoded_results'] = decoded_results
        sub_analysis_results['sub_args'] = sub_args.__dict__
        sub_analysis_results['substituted_texts'] = substituted_texts
        sub_analysis_results[
            'substituted_text_other_informations'] = substituted_text_other_informations

    if not check_sub_ppl_in(results, sub_args):
        if sub_args.oracle_model_name == 'facebook/opt-2.7b':
            oracle_tokenizer = load_local_model_or_tokenizer(
                sub_args.oracle_model_name.split('/')[-1],
                'tokenizer')
            if oracle_tokenizer is None:
                oracle_tokenizer = AutoTokenizer.from_pretrained(sub_args.oracle_model_name)
            oracle_model = load_local_model_or_tokenizer(sub_args.oracle_model_name.split('/')[-1],
                                                         'model')
            if oracle_model is None:
                oracle_model = AutoModelForCausalLM.from_pretrained(sub_args.oracle_model_name)
        else:
            raise NotImplementedError(f'oracle_model_name: {sub_args.oracle_model_name}')
        oracle_model = oracle_model.to(args.device)

        losses, ppls = [], []

        prefix_and_output_texts = results['prefix_and_output_text']

        for substituted_text, prefix_and_output_text in tqdm(zip(
                sub_analysis_results['substituted_texts'], prefix_and_output_texts)):
            loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
                                           output_text=substituted_text,
                                           oracle_model_name=sub_args.oracle_model_name,
                                           oracle_model=oracle_model,
                                           oracle_tokenizer=oracle_tokenizer)
            losses.append(loss)
            ppls.append(ppl)
        sub_analysis_results['sub_losses'] = losses
        sub_analysis_results['sub_ppls'] = ppls

    results[sub_args.sub_result_name] = sub_analysis_results

    with open(args.complete_save_file_path, 'w') as f:
        json.dump(results, f, indent=4)
