from watermarking.wm_lm import main, WmLMArgs
from transformers import HfArgumentParser

parser = HfArgumentParser((WmLMArgs,))
args: WmLMArgs
args, = parser.parse_args_into_dataclasses()
main(args)