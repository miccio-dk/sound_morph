import torch

from pytorch_lightning import LightningModule

class CvaeBase(LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        losses = self._step(batch, batch_idx)
        for loss_name, loss in losses.items():
            self.log(f'train_{loss_name}', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self._step(batch, batch_idx)
        for loss_name, loss in losses.items():
            self.log(f'val_{loss_name}', loss)

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx)
    
    def _step(self, batch, batch_idx):
        x_true, labels = batch
        # extract c
        c = torch.stack([labels[lbl] for lbl in self.configs['c_labels']], dim=-1).float()
        # run entire model
        x_rec, mean, log_var, z = self._shared_eval(x_true, c)
        # calculate loss
        losses = self._loss_function(x_true, x_rec, mean, log_var, z)
        return losses
    
    def _reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.configs['lr'])
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.configs['lr_scheduler']),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]
    