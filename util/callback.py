import tensorflow as tf

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if any([logs.get(k) > 0.99 for k in logs.keys()]):
            self.model.stop_training = True
            self.reason_to_stop = 'Achieved too high metrics, stopping to avoid overfitting.'
