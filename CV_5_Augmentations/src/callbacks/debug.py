from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision.utils import make_grid
import torch
from clearml import Task

from src.lightning_module import ButterflyModule
from src.transform import cv_image_to_tensor, denormalize, tensor_to_cv_image


# class VisualizeBatch(Callback):
#     def __init__(self, every_n_epochs: int):
#         super().__init__()
#         self.every_n_epochs = every_n_epochs

#     def on_train_epoch_start(self, trainer: Trainer, pl_module: ButterflyModule):  # noqa: WPS210
#         if trainer.current_epoch % self.every_n_epochs == 0:
#             images = next(iter(trainer.train_dataloader))[0]

#             visualizations = []
#             for img in images:
#                 img = denormalize(tensor_to_cv_image(img))
#                 visualizations.append(cv_image_to_tensor(img, normalize=False))
#             grid = make_grid(visualizations, normalize=False)
#             trainer.logger.experiment.add_image(
#                 'Batch preview',
#                 img_tensor=grid,
#                 global_step=trainer.global_step,
#             )


class DisplayDebugImagesCallback(Callback):
    def __init__(self, frequency=5):
        """
        Callback to display debug images every 'frequency' epochs using ClearML.
        :param frequency: Frequency of epochs at which images should be logged.
        """
        super().__init__()
        self.frequency = frequency

    def on_train_epoch_start(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.frequency == 0:
            dataloader = trainer.datamodule.train_dataloader()
            images = next(iter(dataloader))[0]
            
            visualizations = []
            for img in images:
                img = denormalize(tensor_to_cv_image(img))
                visualizations.append(cv_image_to_tensor(img, normalize=False))
            grid = make_grid(visualizations, normalize=False)
            grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()

            # Log images to ClearML
            task = Task.current_task()
            if task is not None:
                task.get_logger().report_image(
                    title='Debug Images',
                    series=f'Epoch {trainer.current_epoch}',
                    iteration=trainer.global_step,
                    image=grid_np
                )

