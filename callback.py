import tensorflow as tf

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.998) or (logs.get('val_accuracy') > 0.95):
            self.model.stop_training = True
