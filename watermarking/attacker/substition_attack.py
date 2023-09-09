import os.path
import warnings

import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer
import random


class SubstitutionAttacker:
    def __init__(self, model_name='roberta-base', device='cuda:0', replacement_proportion=0.05,
                 attempt_rate=3., logit_threshold=1., normalize = False):
        warnings.warn("Do not forget to set global random seed!")

        if not 'roberta' in model_name:
            raise ValueError('Only roberta models are supported')
        local_model_name = os.path.join('~/llm-ckpts', model_name)
        try:
            self.model = RobertaForMaskedLM.from_pretrained(local_model_name).to(device)
        except:
            self.model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(local_model_name)
        except:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        self.model.eval()
        self.device = device
        self.replacement_proportion = replacement_proportion
        self.attempt_proportion = attempt_rate * replacement_proportion
        self.logit_threshold = logit_threshold
        self.normalize = normalize

    def attack(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        len_text = len(tokenized_text)

        total_attempts = int(len_text * self.attempt_proportion)
        successful_replacements = int(len_text * self.replacement_proportion)

        attempts = 0
        successes = 0

        tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        replaced_ids = tokenized_ids.copy()

        while attempts < total_attempts and successes < successful_replacements:
            token_index = random.randint(0, len_text - 1)
            original_token_id = tokenized_ids[token_index]
            current_token_id = replaced_ids[token_index]

            # if the token has already been replaced, skip
            if current_token_id != original_token_id:
                continue
            else:
                attempts += 1

            replaced_ids[token_index] = self.tokenizer.mask_token_id
            indexed_tokens = torch.tensor([replaced_ids]).to(self.device)
            mask_token_index = torch.where(indexed_tokens[0] == self.tokenizer.mask_token_id)[0]

            with torch.no_grad():
                predictions = self.model(indexed_tokens)[0]

            logits = predictions[0, mask_token_index, :]
            logits[:, original_token_id] = float('-inf')
            logits[:, self.tokenizer.eos_token_id] = float('-inf') # prevent the model from generating <eos>
            new_token_ids = logits[0].argsort(descending=True)

            replacement_flag = False
            for new_token_id in new_token_ids:
                if self.normalize:
                    token = self.tokenizer.decode([new_token_id])
                    old_token = self.tokenizer.decode([original_token_id])
                    if token.strip().lower() == old_token.strip().lower():
                        continue
                if logits[0, new_token_id] - logits[
                    0, original_token_id] > - self.logit_threshold:
                    replaced_ids[token_index] = new_token_id
                    successes += 1
                    replacement_flag = True
                    break
                else:
                    break
            if replacement_flag == False:
                replaced_ids[token_index] = current_token_id

        return self.tokenizer.decode(replaced_ids), successes / len(tokenized_text)
