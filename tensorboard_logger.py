from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir='runs/bert_experiment'):
        self.writer = SummaryWriter(log_dir)

    def log_training_loss(self, loss, epoch, step, total_steps):
        self.writer.add_scalar('Training Loss', loss, epoch * total_steps + step)

    def log_average_training_loss(self, avg_loss, epoch):
        self.writer.add_scalar('Average Training Loss', avg_loss, epoch)

    def log_validation_loss(self, avg_loss, epoch):
        self.writer.add_scalar('Validation Loss', avg_loss, epoch)

    def close(self):
        self.writer.close()
