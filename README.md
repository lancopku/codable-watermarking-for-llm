# llm-watermarking
Repository for Towards Codable Watermarking for Large Language Models



### Dataset

We use a slice of c4 dataset. Please download c4/realnewslike/c4-train.00000-of-00512.json.gz from huggingface https://huggingface.co/datasets/c4, unzip it, and then run ./get_c4_sliced.py. and ./get_c4_cliced_for_cp.py



### Models

The models, opt-1.3b(2.7b) / gpt2, will be automatically downloaded via huggingface transformers package when code is run. 

In case the model downloading process fails, you can do the following:

Download facebook/opt-1.3b(2.7b) to ./llm-ckpts/opt-1.3b or ./llm-ckpts/opt-1.3b (or adjust the 'ROOT_PATH_LIST' in ./watermarking/utils/load_local.py)

Download gpt2 to ./llm-ckpts/gpt2



### **Demos**

./wm.ipynb provides a basic demo that shows the encoding and decoding of watermarking.



### **Experiments**

Run ./run_wm_none.py or ./run_wm_random.py or ./run_wm_lm.py to **generate no-watermark text or Vanilla-Marking or Balance-Marking**



Run ./analysis_wm_none.py or  ./analysis_wm_random.py or ./analysis_wm_lm.py  to **get the text quality, successful rate and other information about watermarked text (in json format)**



Run run_wm_lm(random)_cp_attack.py to **get the cp-attacked results for copy-paste attack**

Run run_wm_lm(random)_sub_attack.py to **get the substitution-attacked results for substitution attack**



Run the file with 'speed_test' to **get the time cost of watermarking.**



You can use ./experiment.py to generate a gpu_sh.sh file and run this shell file to perform multiple experiments.

You can use ./append_analysis.py after run ./experiment.py to get a shell file including /analysis_wm_*.py



### Other details

We use `LMMessageModel.cal_log_Ps`  to cal $P(t_{:i},t_{i+1},m)$:

```
log_Ps = cal_log_Ps(t_prefix, t_cur,messages,lm_predictions)
# log_Ps[i,j,k] = P(x_prefix[i],t_cur[i,j],messages[k])
# lm_predictions[i,j] = P_LM(t_prefix[i],vocab[j])
```



