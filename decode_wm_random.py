from watermarking.arg_classes.wm_arg_class import WmRandomArgs
from watermarking.wm_decode import main, HumanWrittenDecodeArgs
from transformers import HfArgumentParser

parser = HfArgumentParser((WmRandomArgs, HumanWrittenDecodeArgs))
args: WmRandomArgs
human_written_args: HumanWrittenDecodeArgs
args, human_written_args = parser.parse_args_into_dataclasses()
human_written_args.wm_args = args
main(human_written_args)