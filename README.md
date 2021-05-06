# Open World Image Classification Challenge


[**Webpage**](http://www.cs.cmu.edu/~shuk/open-world-vision.html "open world vision"): 


![alt text](http://www.cs.cmu.edu/~shuk/open-world-vision_files/logo.png)


## todo
(1) notebook demo, (2) guideline/code for evaluation, (3) guideline for csv file generation, (4) logistics


## Installation

**Works with python >= 3.6**

run `pip install -r requirements.txt` in your python environment. **Make sure that you do this in an environment you are fine making alterations to.**

NOTE: This won't install `accimage`. Please install it separately, if you want to use it.

##Usage

### Loading a dataset

* Refer to the class `EvalDataset` in `data_loader.py` and modify it according to your data structure.
    * The version of the class provided in the repo supports following 2 data formats
        * `.txt` file containing absolute path of image in each line
        * `.csv` file (comma separated), containing absolute path of images in first column
            * We do not assume the presence of an index column. You can turn this on by instantiating `EvalDataset` with `index=True`. In this case, we will assume image paths to be the second column.
            * We assume the presence of a header. You can turn this off by instantiating `EvalDataset` with `header=False`
    * We support the `accimage` loader if it is present, and you want to use it. Instantiate `EvalDataset` with `accimage=True` to enable it.
    


### Generating model outputs

* Refer to `run.py`
    * Regarding the model files:  
        * Model class should be present as `model` key
        * Weights should be present as `state_dict`
    * If required, edit lines `92-94` according to your model. Make other changes, if required.
    * run the file with following arguments 
      
    | Parameter                 | Default       | Description   |	
    | :------------------------ |:-------------:| :-------------|
    | --input_file | | path to a .txt/.csv file in the format described in loading dataset|
    | --exp_name | | unique name for this run of the evaluation|
    | --j | 4| # of workers for data loader. Make sure to take care of this if you change data loader |
    | --b | 32 | batch size|
    | --out_dir | | directory to be used to save the results. We will save a `','` separated csv which will be named by the `exp_name`|
    | --accimage | False | use this if accimage module is available on the environment, and you want to use it |

    * Please add other options related to dataloader if you don't want to use the default values.
    
    

### Evaluating the model

* Refer to `eval.py`
    * Calculates the `top-k` accuracy over all the images and each class for a list of k values
    * run the file with following arguments
    
    | Parameter                 | Default       | Description   |	
    | :------------------------ |:-------------:| :-------------|
    | --pred_file | | path to a .txt/.csv file in the format described in loading dataset|path to csv file containing predicted labels. First column should contain image name and rest of the columns should contain predicted probabilities for each of the class_id in the ascending order of class_ids
    | --gt_file | | path to csv file containing ground-truth labels. First column should contain image name and second column should contain the ground truth class_id|
    | --k_vals | 1| space separated list of k(s) in top-k evaluation |
