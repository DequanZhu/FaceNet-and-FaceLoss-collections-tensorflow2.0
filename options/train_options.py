### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import argparse
import os

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--loss_type', type=str, 
                            choices=['ArcFace','Center','AMface','SphereFace','OrignalSoftmax','LAFace'], 
                            default='ArcFace',
                            help='the margin is m in formula cos(theta-m) ')
        self.parser.add_argument('--backbone', type=str, default='Resnet50')
        self.parser.add_argument('--restore', action='store_true',
                            help='Whether to restart training from checkpoint ')
        self.parser.add_argument('--epoches', type=int, default=20,
                            help='The number of epochs to run')
        self.parser.add_argument('--nrof_classes', type=int, default=9277,
                            help='The number of identities')
        self.parser.add_argument('--batch_size', type=int,
                            default=32, help='The size of batch')
        self.parser.add_argument('--image_size', type=int,
                            default=160, help='The size of input image')
        self.parser.add_argument('--embedding_size', type=int,
                            default=128, help='The size of feature to embedding')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/',
                            help='Directory name to save the checkpoints')
        self.parser.add_argument('--train_log_dir', type=str, default='./logs/',
                            help='Directory name to save training logs')
        self.parser.add_argument('--datasets', type=str, default='../data/train_tfrcd/',
                            help='Directory name to load training data')
        self.parser.add_argument('--split_ratio', type=float, default=0.9,
                            help='The ratio of training data for split data')
        self.parser.add_argument('--learning_rate', type=float, default=2e-3,
                            help='Initial learning rate. If set to a negative value a learning rate ')

        opt = self.parser.parse_args()
        if opt.loss_type=='ArcFace' or opt.loss_type=='AMFace':
            self.parser.add_argument('--margin', type=float, default=0.3, help='the margin is m in formula cos(theta-m) ')
            self.parser.add_argument('--feature_scale', type=float, default=0.3, help='the feature s in formula s*(cos(theta-m)) ')

        if opt.loss_type=='CenterLoss':
            self.parser.add_argument('--margin', type=int, default=3, help='the margin to original theta ')
            self.parser.add_argument('--beta', type=float, default=0.5, help='the weight of center_loss to cross_entory loss')

        if opt.loss_type=='SphereFace' or opt.loss_type=='LAFace':
            self.parser.add_argument('--alpha', type=float, default=0.95, help='control how much the centers to update ')
            self.parser.add_argument('--beta', type=int, default=1000, help='used to control simulated annealing learning strategy')
            self.parser.add_argument('--decay', type=float, default=0.99, help='the decay ratio of beta')
            self.parser.add_argument('--beta_min', type=int, default=5, help='the min value of beta and is no longer to reduce')
            self.parser.add_argument('--feature_norm', type=bool, action='store_true', help='whether to norm the feature so that the sum of square equals to 1')
        self.initialized = True
        # return self.parser.parse_args(argv)

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt=self.parser.parse_args()
        checkpoint_dir=opt.checkpoint_dir
        opt.checkpoint_dir=os.path.join(checkpoint_dir, opt.backbone, opt.loss_type)
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')


        
        # save to the disk        
        # opt.checkpoint_dir = os.path.join(self.opt.checkpoint_dir, self.opt.backbone)
        if not os.path.exists(opt.checkpoint_dir):
            os.makedirs(opt.checkpoint_dir)
        if save and not self.opt.restore:
            file_name = os.path.join(opt.checkpoint_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt


if __name__=='__main__':
    opt=TrainOptions().parse()
    print(opt)