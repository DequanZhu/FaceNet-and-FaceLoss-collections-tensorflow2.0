import argparse
import os
import sys
import pandas as pd

class TrainOptions():
    def __init__(self,argv):
        self.argv=argv
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # Common parameters for training the model
        self.parser.add_argument('--loss_type', type=str, 
                                choices=['OrignalSoftmax', 'L2Softmax','LSoftmax', 'AMSoftmax','ArcFaceSoftmax','CenterLoss','ASoftmax',], 
                                default='OrignalSoftmax',
                                help='the margin is m in formula cos(theta-m) ')
        self.parser.add_argument('--backbone', type=str, default='Resnet50',
                                help='The base network for extracting face features ')
        self.parser.add_argument('--restore', action='store_true',
                                help='Whether to restart training from checkpoint ')
        self.parser.add_argument('--max_epoch', type=int, default=50,
                                help='The max number of epochs to run')
        self.parser.add_argument('--nrof_classes', type=int, default=None,
                                help='The number of identities')
        self.parser.add_argument('--batch_size', type=int,
                                default=32, help='The num of one batch samples')
        self.parser.add_argument('--image_size', type=int,
                                default=160, help='The size of input face image')
        self.parser.add_argument('--embedding_size', type=int,
                                default=128, help='The size of the extracted feature ')
        self.parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint/',
                                help='Directory where to save the checkpoints')
        self.parser.add_argument('--log_dir', type=str, default='../logs/',
                                help='Directory where to save training log information ')
        self.parser.add_argument('--datasets', type=str, default='/home/zdq/vgg_tfrcd/',
                                help='Directory where to load train and validate tfrecord format data')
        self.parser.add_argument('--learning_rate', type=float, default=2e-3,
                                help='Initial learning rate. If set to a negative value a learning rate ')


        # Parameters for LSoftmax, ArcFaceSoftmax, AMSoftmax
        self.parser.add_argument('--margin', type=float, default=0.3,
                                    help='The margin is m in ArcFaceSoftmax formula s*cos(theta-m) or in AMSoftmax formula s*(cos(theta)-m).')
        self.parser.add_argument('--feature_scale', type=float, default=1, 
                                help='The feature scales s in ArcFaceSoftmax formula s*cos(theta-m) or in AMSoftmax formula s*(cos(theta)-m) ')


        # Parameters for L2Softmax
        self.parser.add_argument('--l2_feature_scale', type=float, default=16.0,     
                                help='The feature length ')


        # Parameters for CenterLoss
        self.parser.add_argument('--alpha', type=float, default=0.95,   
                                help='Center update rate for center loss.')
        self.parser.add_argument('--loss_weight', type=float, default=0.5,     
                                help='Center loss factor.')


        # Parameters for ASoftmax
        self.parser.add_argument('--beta', type=int, default=1000, 
                                help='the beta in formula fyi=(beta*ori_softmax_loss+A_softmax)/(1+beta)')
        self.parser.add_argument('--decay', type=float, default=0.99, 
                                help='the decay ratio of beta')
        self.parser.add_argument('--beta_min', type=int, default=0, 
                                help='the min value of beta and after that is no longer to reduce')

        self.initialized = True


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        opt=self.parser.parse_args(self.argv)
        opt.checkpoint_dir=os.path.join(opt.checkpoint_dir, opt.loss_type, opt.backbone)
        if not os.path.exists(opt.checkpoint_dir):
            os.makedirs(opt.checkpoint_dir)
        opt.log_dir=os.path.join(opt.log_dir, opt.loss_type, opt.backbone)
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
        df=pd.read_csv(os.path.join(opt.datasets,'info.csv'))
        opt.nrof_classes = df['class_num'][0]
        args = vars(opt)


        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        # save to the disk        

        if save and not opt.restore:
            file_name = os.path.join(opt.checkpoint_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return opt


if __name__=='__main__':
    opt=TrainOptions(sys.argv[1:]).parse()
    # print(opt)