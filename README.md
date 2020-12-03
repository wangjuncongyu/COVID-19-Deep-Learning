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
cd trainers
run_train.bat

# serve with hot reload at localhost:8080
npm run dev

# build for production with minification
npm run build

# run unit tests
npm run unit

# run e2e tests
npm run e2e

# run all tests
npm test
```

## TODO

- [ ]  实现音乐播放器的播放模式调整
- [ ]  搜索模块

## 鸣谢

此网站提供的API: [https://api.imjad.cn/cloudmusic/](https://api.imjad.cn/cloudmusic/) ,此接口的说明请到这里[查看](https://api.imjad.cn/cloudmusic/index.html)

歌单列表部分为自己提供，在别一个项目中[MusicApi](https://github.com/javaSwing/MusicAPI)


