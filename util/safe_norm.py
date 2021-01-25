import tensorflow as tf

def safe_norm(x, epsilon=1e-12, axis=None):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)