from watermarking.wm_lm import WmLMArgs
from transformers import HfArgumentParser
from watermarking.wm_cp_analysis import main, CopyPasteArgs

parser = HfArgumentParser((WmLMArgs, CopyPasteArgs))
args: WmLMArgs
cp_args: CopyPasteArgs
args, cp_args = parser.parse_args_into_dataclasses()

main(args, cp_args)
