# Open World Image Classification Challenge


[**Workshop Webpage**](http://www.cs.cmu.edu/~shuk/open-world-vision.html "open world vision")
[**EvalAI Challenge Server**](https://eval.ai/web/challenges/challenge-page/1041/overview "open world vision")


![alt text](http://www.cs.cmu.edu/~shuk/open-world-vision_files/logo.png)


This github repository is a complimentary resource for the Open World Image Classification Challenge. From the jupyter notebook files, you can see how the challenge dataset is organized, how the evaluation is doen, and how the required csv file is generated for benchmarking on EvalAI. 



## Installation
You may want to install necessary packages. Run `pip install -r requirements.txt` in your python environment. Note that we only tested the code on python >= 3.6.


## Training demo
We provide a demo of how to train a quick-n-dirty model with notebook [demo_train.ipynb](https://github.com/pi-umd/open_world_vision/blob/main/demo_train.ipynb). We hope this demo is helpful in understanding the data. This is by no means a suggestion how to train a good model.


## Evaluation
You should be able to run the following command line to evaluate. Please also refer to [demo_evaluation.ipynb](https://github.com/pi-umd/open_world_vision/blob/main/demo_evaluation.ipynb) for details related to evaluation and metrics. Core functions are in [eval.py](https://github.com/pi-umd/open_world_vision/blob/main/eval.py) and [eval_server.py](https://github.com/pi-umd/open_world_vision/blob/main/eval_server.py)

`python eval_server.py --pred_file ./results/pred_0.4_1.csv --gt_file ./labels_valsets/eval_full_data_0.4_1.csv --out_dir . --model_name test`


## Generating the csv file for benchmarking
The challenge server at EvalAI requires participants to upload a csv file that lists results on the test-set. Please also refer to [demo_testset.ipynb](https://github.com/pi-umd/open_world_vision/blob/main/demo_testset.ipynb) for how to do so.


## Questions?
Should you have technical questions, please create an issue here. 