import numpy as np


def np_temp_random(seed: int):
    def np_temp_random_inner(func):
        def np_temp_random_innner_inner(*args, **kwargs):
            ori_state = np.random.get_state()
            np.random.seed(seed)
            result = func(*args, **kwargs)
            np.random.set_state(ori_state)
            return result

        return np_temp_random_innner_inner

    return np_temp_random_inner