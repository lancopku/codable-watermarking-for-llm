from watermarking.wm_none import WmNoneArgs
from watermarking.arg_classes.wm_arg_class import WmAnalysisArgs
from transformers import HfArgumentParser
from watermarking.wm_analysis import main

oracle_model_name = 'facebook/opt-2.7b'
parser = HfArgumentParser((WmNoneArgs,))
args: WmNoneArgs
args, = parser.parse_args_into_dataclasses()
save_file_path = args.complete_save_file_path

analysis_args = WmAnalysisArgs(analysis_file_name=save_file_path, device=args.device,
                      oracle_model_name=oracle_model_name, args=args)
main(analysis_args)
