from watermarking.wm_cp_attack import WmLMArgs, main, RunControlArgs
from transformers import HfArgumentParser



parser = HfArgumentParser((WmLMArgs,RunControlArgs))
args: WmLMArgs
control_args: RunControlArgs
args, control_args = parser.parse_args_into_dataclasses()
main(args, control_args)