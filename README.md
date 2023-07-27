# SubCDR
Source code and data for "A subcomponent-guided deep learning method for interpretable cancer drug response prediction"

![Framework of SubCDR](https://github.com/liuxuan666/SubCDR/blob/main/Subcdr.png)  

# Requirements
* Python >= 3.7
* PyTorch >= 1.5
* PyTorch Geometry >= 1.6
* hickle >= 3.4
* DeepChem >= 2.4
* RDkit >= 2020.09

# Usage
* python main_cv.py \<parameters\>  #---Regression task with 5-fold CV
* python main_independent.py \<parameters\> #---Independent testing with 9(traing):1(testing) split of the dataset
* python main_classify.py \<parameters\> #---Binary classification task with IC50 values
