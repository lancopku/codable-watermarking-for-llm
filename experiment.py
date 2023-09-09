from watermarking.utils.experiment_utils import Sh_run

sample_seeds = [42]

encode_ratios = [10., 5.]
repeat_panalities = [1.5]
deltas = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]
new_deltas = [1.0, 1.2, 1.5, 2.0, 3.0]

new_encode_ratios = [10.]
sample_num = 500


hypers = [
    {
        'expr': 'python run_wm_lm.py',
        'model_name': ['facebook/opt-1.3b'],
        'repeat_penalty': repeat_panalities,
        'temperature': [1.0],
        'sample_seed': sample_seeds,
        'num_beams': [4],
        'generated_length': [200],
        'sample_num': [sample_num],
        'lm_prefix_len': [10],
        'lm_top_k': [-1],
        'message_code_len': [20],
        'random_permutation_num': [100],
        'max_confidence_lbd': [0.5],
        'encode_ratio': encode_ratios,
        'delta': deltas,
        'message_model_strategy': ['vanilla'],
        'lm_model_name': ['gpt2-large', 'gpt2-medium_', 'gpt2-xl']
    },
]
run = Sh_run(hyperparameter_lists=hypers,
             gpu_list=[0,1,2,3,4,5])

run.run()

print(f'assigned tasks on gpus {run.gpu_list}')

print(f'running tasks {run.experiment_strs}')
