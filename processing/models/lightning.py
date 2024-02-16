from torch.optim import Adam

import lightning as L

from .metrics import Dice, Accuracy


class TrainingModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-1, dropout=0.2):
        super().__init__()

        self.learning_rate = learning_rate

        self.model = model(3, 1, dropout=dropout)
        self.loss = Dice()
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def run_step(self, batch):
        x, y = batch['image'], batch['mask']
        y_pred = self.forward(x)
        loss = 1 - self.loss(y, y_pred)
        accuracy = self.accuracy(y, y_pred).detach()

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.run_step(batch)

        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.run_step(batch)

        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.run_step(batch)

        self.log('test_accuracy', accuracy, prog_bar=True)
        self.log('test_loss', loss, prog_bar=True)

        return loss

    def predict_step(self, batch):
        x, y = batch['image'], batch['mask']
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
