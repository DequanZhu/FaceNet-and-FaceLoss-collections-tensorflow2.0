# Description
> This repository implements facenet using tensorflow2.0-keras api and in eager-mode .The project is still undergo.The code is just veritify in cpu mode and gpu-run support will come soon.

# How to use
> + [Install tensorflow2.0-beta1 version.](https://tensorflow.google.cn/install/pip)
> + Anaconda virtual enviroment is recommended.
> + I use tfrecords foramt data to create input-pipeline.To create tfrecords format training data, run the script datasets.py by:

> 
```bash
  python datasets.py --src_dir ~/your/datasets/dir --dest_dir ~/your/dest/dir --nrof_imgs_per_file 50000
```

# TODO
- [x] Provide train code in vggface2 datasets using softmax.
- [x] To implement  CenterLoss, LSoftmaxLoss, L2SoftmaxLoss,  AMSoftmaxLoss,  ASoftmaxLoss, ArcFaceSoftmaxLoss.
- [ ] Provide test code which use LFW datasets.
- [ ] Train some models using different backbone and diffrent loss fun in two.
- [ ] Compare and analysis different loss type.  
- [ ] Refactor and clean the code.


