from watermarking.wm_random_speed_test import main, WmRandomArgs
from transformers import HfArgumentParser

parser = HfArgumentParser((WmRandomArgs,))
args: WmRandomArgs
args, = parser.parse_args_into_dataclasses()
main(args)