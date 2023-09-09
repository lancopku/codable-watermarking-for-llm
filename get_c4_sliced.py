from datasets import load_dataset
from functools import partial

from transformers import AutoTokenizer

c4 = load_dataset('json',data_files='./c4/realnewslike/c4-train.00000-of-00512.json')

def filter(sample, tokenizer,min_length = 500):
    # 500: 300 prompt length + 200 output length
    return len(tokenizer.encode(sample['text']))>min_length

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
c4_filtered  = c4.filter(partial(filter,tokenizer=tokenizer))
c4_filtered.save_to_disk('./c4-train.00000-of-00512_sliced')