import tensorflow as tf


class Center_Loss():
    def __init__(self,alpha,nrof_classes,embedding_size):
        super(Center_Loss,self).__init__()
        self.alpha=alpha
        self.centers=tf.zeros(shape=[nrof_classes,embedding_size],dtype=tf.float32)


    def __call__(self,features, label):
        embedding_size = tf.shape(features)[1]
        label=tf.expand_dims(label,axis=1)
        centers_batch = tf.gather_nd(self.centers, label)
        diff = (1 - self.alpha) * (centers_batch - features)
        centers = tf.tensor_scatter_nd_sub(self.centers, label, diff)
        loss = tf.math.reduce_mean(tf.math.square(features - centers_batch))
        return loss

