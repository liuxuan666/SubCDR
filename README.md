# SubCDR
Source code and data for "A interpretable subcomponent-guided deep learning method for cancer drug response prediction"

![Framework of SubCDR](https://github.com/liuxuan666/SubCDR/blob/main/Subcdr.png)  

# Requirements
* Python >= 3.6
* PyTorch >= 1.4
* PyTorch Geometry >= 1.6
* hickle >= 3.4
* DeepChem >= 2.4
* RDkit >= 2020.09

# Usage
* python main_cv.py \<parameters\>
* python main_independent.py \<parameters\>
* python main_classify.py \<parameters\>
  
# Case study (predicted unmeasured CDRs)
As GDSC v2 database only measured IC50 values of part cell line and drug combinations. We applied SubCDR to predicted IC50 values for unmeasured combinations. The predicted results can be find at data/unmeasured_combinations_results.xlsx
