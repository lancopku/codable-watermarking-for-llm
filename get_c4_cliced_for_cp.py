from datasets import load_dataset
from functools import partial
from watermarking.utils.text_tools import truncate

from transformers import AutoTokenizer

c4 = load_dataset('json',data_files='../c4/realnewslike/c4-train.00102-of-00512.json')

def filter(sample, tokenizer,min_length = 1000):
    # 500: 300 prompt length + 200 output length
    return len(tokenizer.encode(sample['text']))>min_length

def apply_truncate(example,max_length = 1000):
    tokenized_text = tokenizer.encode(example['text'],add_special_tokens=False)
    truncated_text = tokenized_text[:max_length]
    truncated_text = tokenizer.decode(truncated_text)
    example['truncated_text'] = truncated_text
    return example

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
c4_filtered  = c4.filter(partial(filter,tokenizer=tokenizer))
c4_filtered = c4_filtered.map(apply_truncate)

c4_filtered.save_to_disk('./c4-train.00102-of-00512_sliced_for_cp')