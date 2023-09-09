from dataclasses import dataclass
from ..arg_classes.wm_arg_class import WmNoneArgs, WmRandomArgs, WmLMNewArgs, WmLMArgs


str_to_class = {
    WmNoneArgs.wm_strategy: WmNoneArgs,
    WmRandomArgs.wm_strategy: WmRandomArgs,
    WmLMNewArgs.wm_strategy: WmLMNewArgs,
    WmLMArgs.wm_strategy: WmLMArgs,
}

def check_no_post_init(cls):
    if cls.__class__ == dataclass:
        raise Exception(f"{cls.__class__} is not dataclass")
    if "__post_init__" in cls.__dict__:
        raise Exception(f"{cls.__class__} rewrites __post_init__")
    return True


def _load_args_from_dict(args_dict, args_class):
    # check post_init to be null
    check_no_post_init(args_class)
    class_instance = args_class(**args_dict)
    return class_instance

def load_args_from_dict(args_dict):
    wm_strategy = args_dict['wm_strategy']
    arg_class = str_to_class[wm_strategy]
    return _load_args_from_dict(args_dict, arg_class)

