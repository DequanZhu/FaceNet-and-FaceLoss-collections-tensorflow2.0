import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
import numpy as np
import math


class OriginalSoftmaxLinear(layers.Layer):
    def __init__(self, units):
        super(OriginalSoftmaxLinear, self).__init__()
        self.fc=layers.Dense(units)

    def call(self, embedding, labels):
        logits = self.fc(embedding)
        return logits



class LSoftmaxLinear(layers.Layer):
    def __init__(self, units, input_dim, margin):
        super(LSoftmaxLinear, self).__init__()
        self.m = margin  # m
        self.beta = 1000
        self.beta_min = 0
        self.decay = 0.99
        self.divisor = math.pi / self.m  # pi/m
        # Initialize L-Softmax parameters
        self.weight = self.add_weight(shape=(input_dim, units),
                                      initializer='he_normal',
                                      trainable=True)

    def find_k(self, cos_theta):
        eps = 1e-7
        cos_theta = tf.clip_by_value(cos_theta, -1 + eps, 1 - eps)
        acos = tf.math.acos(cos_theta)
        k = tf.math.floordiv(acos, self.divisor)
        return k

    def call(self, embedding, labels):
        x, w = embedding, self.weight
        w_norm = tf.norm(w, ord='euclidean', axis=0)
        x_norm = tf.norm(x, ord='euclidean', axis=1)
        logits = tf.matmul(x, w)
        indices_m = tf.expand_dims(tf.Variable(range(embedding.shape[0])), axis=1)
        indices_n = tf.expand_dims(labels, axis=1)
        indices = tf.concat([indices_m, indices_n], 1)
        selected_logits = tf.gather_nd(logits, indices)
        w_target_norm = tf.gather(w_norm, labels)
        cos_theta_target = selected_logits / (w_target_norm*x_norm + 1e-10)
        theta_m=self.m*tf.math.acos(cos_theta_target)
        cos_m_theta_target=tf.math.cos(theta_m)
        k = self.find_k(cos_theta_target)
        logit_target_updated = (w_target_norm *
                                x_norm *
                                (((-1) ** k * cos_m_theta_target) - 2 * k))

        beta = max(self.beta, self.beta_min)
        logit_target_updated_beta = (logit_target_updated + beta * selected_logits) / (1 + beta)
        self.beta *= self.decay
        logits = tf.tensor_scatter_nd_update(logits, indices, logit_target_updated_beta)
        return logits




class ASoftmaxLinear(layers.Layer):
    def __init__(self, units, input_dim, margin, beta = 1000, beta_min=0, decay=0.99):
        super(ASoftmaxLinear, self).__init__()
        self.m = margin  # m
        self.beta = beta
        self.beta_min = beta_min
        self.decay = decay
        self.divisor = math.pi / self.m # pi/m
        self.weight = self.add_weight(shape=(input_dim, units),
                                      initializer='he_normal',
                                      trainable=True)


    def find_k(self, cos_theta):
        eps = 1e-7
        cos_theta = tf.clip_by_value(cos_theta, -1 + eps, 1 - eps)
        acos = tf.math.acos(cos_theta)
        k = tf.math.floordiv(acos, self.divisor)
        return k

    def call(self, embedding, labels):
        x, w = embedding, self.weight
        w = tf.math.l2_normalize(w, axis=0)
        x_norm = tf.norm(x, ord='euclidean', axis=1)
        logits = tf.matmul(x, w)
        indices_m = tf.expand_dims(tf.Variable(range(embedding.shape[0])), axis=1)
        indices_n = tf.expand_dims(labels, axis=1)
        indices = tf.concat([indices_m, indices_n], 1)
        selected_logits = tf.gather_nd(logits, indices)
        cos_theta_target = selected_logits / (1.0*x_norm + 1e-10)
        theta_m=self.m*tf.math.acos(cos_theta_target)
        cos_m_theta_target=tf.math.cos(theta_m)
        k = self.find_k(cos_theta_target)
        logit_target_updated = (1.0 *
                                x_norm *
                                (((-1) ** k * cos_m_theta_target) - 2 * k))

        beta = max(self.beta, self.beta_min)
        logit_target_updated_beta = (logit_target_updated + beta * selected_logits) / (1 + beta)
        self.beta *= self.decay
        logits = tf.tensor_scatter_nd_update(logits, indices, logit_target_updated_beta)
        return logits


class CenterLossLinear(layers.Layer):
    def __init__(self, units, input_dim, alpha, beta):
        super(CenterLossLinear, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.centers = tf.zeros(shape=[units, input_dim], dtype=tf.float32)
        self.mse=losses.MeanSquaredError()()
        self.fc=layers.Dense(units)

    def call(self, embedding, labels):
        centers_batch = tf.gather(self.centers, labels)
        diff = (1 - self.alpha) * (centers_batch - embedding)
        centers = tf.tensor_scatter_nd_sub(self.centers, labels, diff)
        center_loss=self.mse()
        logits=self.fc(embedding)
        return logits, center_loss


class L2SoftmaxLinear(layers.Layer):
    def __init__(self, units, input_dim, feature_scale=None):
        super(L2SoftmaxLinear, self).__init__()
        if(feature_scale!=None):
            self.s=feature_scale
        else:
            exp_s=tf.math.exp(self.s)
            self.s=tf.math.floordiv(exp_s,exp_s+units-2)
        self.fc=layers.Dense(units)

    def call(self, embedding, labels):
        embedding = tf.math.l2_normalize(embedding, axis=1)
        embedding*=self.s
        logits=self.fc(embedding)
        return logits



class AMSoftmaxLinear(layers.Layer):
    def __init__(self, units, input_dim, margin, feature_scale=64):
        super(AMSoftmaxLinear, self).__init__()
        self.m = margin  
        self.weight = self.add_weight(shape=(input_dim, units),
                                      initializer='he_normal',
                                      trainable=True)
        self.s = feature_scale

    def __call__(self, embedding, labels):
        x, w = embedding, self.weight
        w = tf.math.l2_normalize(w, axis=0)
        x = tf.math.l2_normalize(x, axis=1)
        logits = tf.matmul(x, w)
        indices_m = tf.expand_dims(tf.Variable(range(embedding.shape[0])), axis=1)
        indices_n = tf.expand_dims(labels, axis=1)
        indices = tf.concat([indices_m, indices_n], 1)
        selected_logits = tf.gather_nd(logits, indices)
        logit_target_updated = self.s*(selected_logits-self.m)
        logits = tf.tensor_scatter_nd_update(logits, indices, logit_target_updated)
        return logits


class ArcFaceSoftmaxLinear(layers.Layer):
    def __init__(self, units, input_dim, margin, feature_scale=64):
        super(ArcFaceSoftmaxLinear, self).__init__()
        self.m = margin  # m
        self.s = feature_scale
        self.cos_m = tf.math.cos(self.m)
        self.sin_m = tf.math.sin(self.m)
        self.threshold = tf.math.cos(math.pi-self.m)
        self.weight = self.add_weight(shape=(input_dim, units),
                                      initializer='he_normal',
                                      trainable=True)

    def __call__(self, embedding, labels):
        x, w = embedding, self.weight
        w = tf.math.l2_normalize(w, axis=0)
        x = tf.math.l2_normalize(x, axis=1)
        logits = tf.matmul(x, w)
        indices_m = tf.expand_dims(tf.Variable(range(embedding.shape[0])), axis=1)
        indices_n = tf.expand_dims(labels, axis=1)
        indices = tf.concat([indices_m, indices_n], 1)
        selected_logits = tf.gather_nd(logits, indices)
        cos_theta = selected_logits
        sin_theta = tf.math.sqrt((1.0-tf.math.square(cos_theta)))
        logit_target = self.s * (cos_theta*self.cos_m-sin_theta*self.sin_m)
        keep_val = self.s*(cos_theta - self.m*self.sin_m)
        logit_target_updated = tf.where(cos_theta > self.threshold, logit_target, keep_val)
        logit_target_updated = self.s*(selected_logits-self.m)
        logits = tf.tensor_scatter_nd_update(logits, indices, logit_target_updated)
        return logits



if __name__=="__main__":
    Loss=LSoftmaxLinear(5,10,3)
    init = tf.random_normal_initializer()
    input=tf.Variable(initial_value=init(shape=(8, 10),dtype='float32'))
    target=tf.Variable(initial_value=[0,1,1,3,4,2,0,4])
    a=Loss(input,target)
    print(a)