# COVID-19-Deep-Learning
tensorflow projects for diagnosis and prognostic estimation of COVID-19

> This is a deep-learning framework for identificating high-risk COVID-19 patients and estimating how long the patient can be cured。

### requirments
- Anaconda python 3.7.3 Win10
- Tensorflow 2.0.0 with GPU

## network architecture (see achitecture.pptx)
![framework overview and data set](tf_covid19_care/images/framework_overview_dataset.PNG)
![netwok architecture](tf_covid19_care/images/architecture.PNG)

[pretrained model and 50 data subjects for evaluation](https://pan.baidu.com/s/1ybZmR6LbXXFDVDoLKkSdlA)
# password for download:8vst
#after download, unzip the checkpoint.zip, then put all directories of weight files as well as the files for normalization (feature_minv.npy and the feature_maxv.npy) to tf_covid19_care/checkpoints

#if you have any problem, please feel free to ask questions via sending email to wjcy19870122@sjtu.edu.cn
## Training

``` bash
(1) prepare your data (see the 50 data subjects for examples).
(2) cd trainers and run the file: run_train.bat.
Note: you may need to modify the configs/cfgs.py file:changing cfg.data_set to the directory of your dataset.
```
##  Evaluation
``` bash
(1) cd tests and run the run_test.bat file.
(2) run the compute_metrics.py file to obtain the results.
```
## Results
![average day error of prediction](tf_covid19_care/images/result_examples1.PNG)
#The average error between predicted recovery day and true recovery day. The results demonstrate that treatment schemes have significant impact on the predictions.

![significant features for the predictions](tf_covid19_care/images/result_examples2.PNG)
#The top 10 features with a significant impact on model prediction. AM: Albumin, HG: Hemoglobin, TP: Total Protein, α-HBDH: Alpha-hydroxybutyrate Dehydrogenase, CRP: C-reactive Protein, EPC: Expectoration, SK: Shock, PA: Poor Appetite, PS: Poor Spirits, CGH: Cough, WK: Weakness, CCBD: Chest Congestion/Breathing Difficulty, ARDS: Acute Respiratory Distress Syndrome, LDH: Lactate Dehydrogenase, DB: Diabetes. These features are significant on both prediction tasks (a and b: severity-level grading; c, d, e and f: recovery-time regression). However, the impact ranking is different among these features. The p-values calculated from multi-variable linear classification/regression analyses demonstrate that some features especially the symptoms (see the red p-values) are non-significant for linear analysis methods. 

![examples of prediction](tf_covid19_care/images/result_examples.PNG)
#Visualization of the predicted probability distribution for four patients. The days need for a patient to be cured can be estimated by the day with the maximum probability (see the vertical dashed lines). Besides, the cumulative incidence function (e.g., P(t≤7)=17.1%) can also be calculated to assess the risk of patients. The top-3 features are shown to explain the decision made by the model. The dead patient #4 can be easily identified by observing the shape of the curve.



## TODO

- [ ]  address imbalance cured-days distribution problem
- [ ]  evluation on multi-center data


