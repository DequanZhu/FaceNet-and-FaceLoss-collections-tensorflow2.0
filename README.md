# Description
> This repository implements facenet using tensorflow2.0-keras api and in eager-mode .The project is still undergo.The code is just veritify in cpu mode and gpu-run support will come soon.

# How to use
> + [Install tensorflow2.0-aplha version.](https://tensorflow.google.cn/install/pip)
> + Anaconda virtual enviroment is recommended.
> + I use tfrecords foramt data to create input-pipeline.To create tfrecords format training data, run the script datasets.py by:

> 
```bash
  python datasets.py --train_datasets ~/your/datasets/dir ~/dest/dir --nrof_imgs_per_file 50000
```

# TODO
- [x] To implement  CenterLoss, LSoftmaxLoss, L2SoftmaxLoss,  AMSoftmaxLoss,  ASoftmaxLoss, ArcFaceSoftmaxLoss.
- [ ] Train on single GPU and distribute-training.
- [ ] Test and compare different loss.  
- [ ] Clean the code.


