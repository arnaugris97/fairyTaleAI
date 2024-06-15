from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

class TensorBoardLogger:
    def __init__(self, base_log_dir='runs/bert_experiment'):
        log_dir = self._get_log_dir(base_log_dir)
        self.writer = SummaryWriter(log_dir)

    def _get_log_dir(self, base_log_dir):
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        return os.path.join(base_log_dir, current_time)

    def log_training_loss(self, loss, epoch, step, total_steps):
        self.writer.add_scalar('Training Loss', loss, epoch * total_steps + step)

    def log_average_training_loss(self, avg_loss, epoch):
        self.writer.add_scalar('Average Training Loss', avg_loss, epoch)

    def log_validation_loss(self, avg_loss, epoch):
        self.writer.add_scalar('Validation Loss', avg_loss, epoch)
    
    def log_validation_accuracy_nsp(self, accuracy, epoch):
        self.writer.add_scalar('Validation Accuracy NSP', accuracy, epoch)
    
    def log_validation_accuracy_mlm(self, accuracy,  epoch, step, total_steps):
        self.writer.add_scalar('Validation Accuracy MLM', accuracy, epoch * total_steps + step)

    def close(self):
        self.writer.close()
