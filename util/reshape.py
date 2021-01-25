import tensorflow as tf

def _reshape(y, time, warm_up_length, num_scat, max_cand, prob=False):
    if prob:
        y = tf.split(y, (time-warm_up_length), -1)
        y = tf.stack(y, axis=1)
    y = tf.split(y, num_scat, axis=-1)
    y = tf.stack(y, axis=-2)
    y = tf.split(y, max_cand, axis=-1)
    y = tf.stack(y, axis=-2)
    return y