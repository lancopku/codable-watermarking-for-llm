from watermarking.wm_random import WmRandomArgs
from transformers import HfArgumentParser
from watermarking.wm_cp_analysis import main, CopyPasteArgs

parser = HfArgumentParser((WmRandomArgs, CopyPasteArgs))
args: WmRandomArgs
cp_args: CopyPasteArgs
args, cp_args = parser.parse_args_into_dataclasses()

main(args, cp_args)
