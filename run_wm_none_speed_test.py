from watermarking.wm_none_speed_test import main, WmNoneArgs
from transformers import HfArgumentParser

parser = HfArgumentParser((WmNoneArgs,))
args: WmNoneArgs
args, = parser.parse_args_into_dataclasses()
main(args)