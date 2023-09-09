from watermarking.wm_random import main, WmRandomArgs
from transformers import HfArgumentParser

parser = HfArgumentParser((WmRandomArgs,))
args: WmRandomArgs
args, = parser.parse_args_into_dataclasses()
main(args)