from watermarking.wm_lm import WmLMArgs
from transformers import HfArgumentParser
from watermarking.wm_sub_analysis import main, SubstitutionArgs

parser = HfArgumentParser((WmLMArgs, SubstitutionArgs))
args: WmLMArgs
sub_args: SubstitutionArgs
args, sub_args = parser.parse_args_into_dataclasses()

main(args, sub_args)
