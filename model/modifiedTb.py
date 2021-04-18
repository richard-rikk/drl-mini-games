import os
import tensorflow as tf

from tensorflow.keras.callbacks  import TensorBoard

# This modified tensorboard will not save at every fit only when the target model gets trained.
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer    = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
        self.step = 0
        self.own_stats = {}

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_dir  = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter # pylint: disable=protected-access
        self._val_dir    = os.path.join(self._log_write_dir, 'validation')
        self._val_step   = self.model._test_counter # pylint: disable=protected-access
        

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)


    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass
    
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in self.own_stats.items():
                tf.summary.scalar(name, value, step=index)
                self.writer.flush()
            
            self.step += 1 
            
    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self.own_stats.update(stats)
        self._write_logs(stats, self.step)