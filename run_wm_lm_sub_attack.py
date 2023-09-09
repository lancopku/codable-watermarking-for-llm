from watermarking.wm_sub_attack import WmLMArgs, main
from transformers import HfArgumentParser

parser = HfArgumentParser((WmLMArgs,))
args: WmLMArgs
args, = parser.parse_args_into_dataclasses()
main(args)


