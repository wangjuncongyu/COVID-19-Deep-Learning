# COVID-19-Deep-Learning
tensorflow projects for diagnosis and prognostic estimation of COVID-19

> This is a deep-learning framework for identificating high-risk COVID-19 patients and estimating how long the patient can be cured。

### requirments
- Anaconda python 3.7.3 Win10
- Tensorflow 2.0.0 with GPU

## network architecture (see achitecture.pptx)
![netwok architecture](tf_covid19_care/images/architecture.PNG)

[pretrained model and 50 data subjects for evaluation](https://pan.baidu.com/s/1ybZmR6LbXXFDVDoLKkSdlA)
# password for download:8vst
#after download, unzip the checkpoint.zip, then put all directories of weight files as well as the files for normalization (feature_minv.npy and the feature_maxv.npy) to tf_covid19_care/checkpoints

#if you have any problem, please feel free to ask questions via sending email to wjcy19870122@sjtu.edu.cn
## Training

``` bash
# run bat batch to start training 
(1)
cd trainers
run_train.bat



# run all tests
npm test
```

## TODO

- [ ]  address imbalance cured-days distribution problem
- [ ]  evluation on multi-center data


