from neurotorch.core.trainer import TrainerDecorator
import tensorboardX
import os
import logging
import time


class LossWriter(TrainerDecorator):
    """
    Logs the loss at each iteration to a Tensorboard log
    """
    def __init__(self, trainer, logger_dir, experiment_name):
        """
        Initializes the Tensorboard writer

        :param trainer: Trainer object that the class wraps
        :param logger_dir: Directory to save Tensorboard logs
        :param experiment_name: The name to mark the experiment
        """
        if not os.path.isdir(logger_dir):
            raise IOError("{} is not a valid directory".format(logger_dir))

        super().__init__(trainer)
        experiment_dir = os.path.join(logger_dir, experiment_name)
        os.mkdir(experiment_dir)
        self.train_writer = tensorboardX.SummaryWriter(os.path.join(experiment_dir,
                                                       "train_log"))
        self.validation_writer = tensorboardX.SummaryWriter(os.path.join(experiment_dir,
                                                            "validation_log"))

        self.iteration = 0

    def log_loss(self, loss: float, duration: float, iteration: int):
        """
        Writes the loss onto the Tensorboard log

        :param loss: The training loss of the model
        :param duration: The time elapsed by the current iteration
        :param iteration: The current iteration of the model
        """
        self.train_writer.add_scalar("Time", duration, iteration)
        self.train_writer.add_scalar("Loss", loss, iteration)

    def run_epoch(self, sample_batch):
        """
        Runs an epoch and saves the parameters onto the Tensorboard log

        :param sample_batch: A batch of input/label samples for training
        """
        start = time.time()
        loss = super().run_epoch(sample_batch)
        end = time.time()

        duration = end - start

        self.iteration += 1

        self.log_loss(loss, duration, self.iteration)


class TrainingLogger(TrainerDecorator):
    """
    Logs the iteration parameters onto a plain text log file
    """
    def __init__(self, trainer, logger_dir=None):
        """
        Initializes the Python Logger

        :param trainer: Trainer object that the class wraps
        :param logger_dir: The directory to save logs
        """
        if logger_dir is not None and not os.path.isdir(logger_dir):
            raise IOError("{} is not a valid directory".format(logger_dir))

        super().__init__(trainer)

        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)

        if logger_dir:
            file_handler = logging.FileHandler(os.path.join(logger_dir,
                                                            "training.log"))
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler)

        self.iteration = 0

    def run_epoch(self, sample_batch):
        """
        Runs an epoch and saves the parameters in a log

        :param sample_batch: A batch of input/label samples for training
        """
        start = time.time()
        loss = super().run_epoch(sample_batch)
        end = time.time()
        duration = end - start

        self.iteration += 1

        self.logger.info("Iteration: {}, Loss: {}, Time: {}".format(self.iteration,
                                                                    loss,
                                                                    duration))


class ImageWriter(TrainerDecorator):
    def __init__(self, trainer, image_dir):
        if not os.path.isdir(image_dir):
            raise IOError("{} is not a valid directory".format(image_dir))
