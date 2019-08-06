# Description
> This repository implements facenet using tensorflow2.0-keras api and in eager-mode supporting different backbones and different loss types. The project is still undergo. 

# How to use
> + [Install tensorflow2.0-beta1 version.](https://tensorflow.google.cn/install/pip)
> + Anaconda virtual enviroment is recommended.
> + I use tfrecords format data to create input-pipeline.To create tfrecords format training data, run the script datasets.py by:

> 
```bash
  bash scripts/tfrcd_data_gen.sh
```

# TODO
- [x] Provide train code in vggface2 datasets using SoftmaxLoss.
- [x] To implement  CenterLoss, LSoftmaxLoss, L2SoftmaxLoss,  AMSoftmaxLoss,  ASoftmaxLoss, ArcFaceSoftmaxLoss.
- [ ] Provide test code in LFW datasets.
- [ ] Train some models using different backbone and diffrent loss fun in secord.
- [ ] Compare and analysis different loss type.  
- [ ] Refactor and clean the code.


