from watermarking.wm_random import WmRandomArgs
from transformers import HfArgumentParser
from watermarking.wm_sub_analysis import main, SubstitutionArgs

parser = HfArgumentParser((WmRandomArgs, SubstitutionArgs))
args: WmRandomArgs
sub_args: SubstitutionArgs
args, sub_args = parser.parse_args_into_dataclasses()

main(args, sub_args)
