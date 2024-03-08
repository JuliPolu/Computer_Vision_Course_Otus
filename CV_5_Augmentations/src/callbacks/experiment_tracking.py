import os
import csv
from typing import Dict, List

import torch
from clearml import  Task
from pytorch_lightning import Callback, LightningModule, Trainer
from sklearn.metrics import confusion_matrix
from torch import Tensor


class LogConfusionMatrix(Callback):
    def __init__(self, task: Task = None):
        super().__init__()
        self.task = task if task is not None else Task.current_task()
        self.predicts: List[Tensor] = []
        self.targets: List[Tensor] = []

    def on_test_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Tensor, 
        batch: List[Tensor], 
        batch_idx: int
    ) -> None:
        if outputs is not None:
            pred = outputs.argmax(dim=1) if outputs.ndim > 1 else outputs  # Modify as needed
            self.predicts.append(pred)
        self.targets.append(batch[1])

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        targets = torch.cat(self.targets, dim=0).detach().cpu().numpy()
        predicts = torch.cat(self.predicts, dim=0).detach().cpu().numpy()
        cf_matrix = confusion_matrix(targets, predicts)

        # Log the confusion matrix
        self.task.get_logger().report_confusion_matrix(
            title='Confusion Matrix',
            series='Confusion Matrix',
            iteration=trainer.current_epoch,
            matrix=cf_matrix,
        )


class SaveMetricsCallback(Callback):
    def __init__(self, filepath: str, filename: str):
        super().__init__()
        self.filepath = filepath
        self.filename = filename

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        full_path = os.path.join(self.filepath, self.filename)
        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Value'])
            for key, value in metrics.items():
                writer.writerow([key, value.item() if isinstance(value, torch.Tensor) else value])
