from watermarking.wm_cp_attack_selected import WmLMArgs, main
from transformers import HfArgumentParser

parser = HfArgumentParser((WmLMArgs,))
args: WmLMArgs
args, = parser.parse_args_into_dataclasses()
main(args)