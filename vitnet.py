import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics.functional as metrics
import numpy as np
import numpy as np

class Model(pl.LightningModule):
    def __init__(self, architecture='vit_tiny_patch16_224', loss_func=nn.L1Loss(), augment_technique='acm', 
                 # In case augment_technique is set to 'acm'
                 # whether to roll the image or to apply the mask where it was taken
                 roll_masks = True, 
                 # total number of masks gathered with data loading
                 nb_masks = 6, 
                 # minimum number of masks to apply
                 min_masks_to_apply = 0, 
                 # maximum number of masks to apply
                 max_masks_to_apply = 0, 
                 # number of images to keep intact in each batch
                 skip_augment_probability = 1/7,
                 ):
        super().__init__()
        self.loss = 1000
        self.pc = 0
        self.loss_func = loss_func

        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.model.norm = nn.Identity()
        self.model.pre_legits = nn.Identity()
        self.model.head = nn.Sequential(nn.Linear(192,128), nn.Linear(128,2))

        self.save_imgs = True
        self.augment_technique = augment_technique
        self.nb_masks = nb_masks
        self.min_masks_to_apply = min_masks_to_apply
        self.max_masks_to_apply = max_masks_to_apply
        assert self.max_masks_to_apply <= self.nb_masks, "cannot apply more masks than there are masks"
        assert self.nb_masks >= 0 and self.min_masks_to_apply >= 0 and max_masks_to_apply >= 0, 'cannot have negative number of masks'
        self.roll_masks = roll_masks
        self.skip_augment_probability = skip_augment_probability

        self.lr = 1e-3
        self.lr_patience = 5
        self.lr_min = 1e-7

        self.labels_p = []
        self.labels_gt = []

        self.tr_loss = []
        self.vl_loss = []
        self.ts_loss = []
        self.tr_mae = []
        self.vl_mae = []
        self.ts_mae = []

        self.test_step_outputs = []
        self.train_step_outputs = []
        self.val_step_outputs = []

    def forward(self, images):
        images = self.model(images)
        return images

    def calculate_centers_of_mass(self, masks):
        B, _, H, W = masks.shape
        
        # Create coordinate grids for Y and X
        y_grid = torch.arange(H).view(1, H, 1).expand(B, H, W).to(masks.device)
        x_grid = torch.arange(W).view(1, 1, W).expand(B, H, W).to(masks.device)

        # Calculate weighted coordinates
        weighted_y = y_grid * (masks.squeeze(1))
        weighted_x = x_grid * (masks.squeeze(1))
        
        # Sum of weights (mass) for each image in the batch
        mass = masks.sum(dim=(2, 3))  # Shape: (B,)
        
        # Handle cases where mass is zero (to avoid division by zero)
        mass = torch.where(mass == 0, torch.ones_like(mass), mass)
        mass = mass.squeeze()

        # Calculate centers of mass
        center_of_mass_y = weighted_y.sum(dim=(1, 2)) / mass
        center_of_mass_x = weighted_x.sum(dim=(1, 2)) / mass

        # Combine results
        return torch.stack((center_of_mass_y, center_of_mass_x), dim=1)

    def anatomical_cutmix_multi_zone(self, images, labels, masks, nb_extracted_zones, min_coverage_proportion = 0.015):
        batch_size = images.shape[0]
        
        if np.random.random() < self.skip_augment_probability or nb_extracted_zones == 0 or nb_extracted_zones > self.nb_masks:
            return images, labels
        
        # Shuffle the batch indices
        indices_shuffled = torch.randperm(batch_size)
        images_shuffled = images[indices_shuffled]
        labels_shuffled = labels[indices_shuffled]
        masks_shuffled = masks[indices_shuffled]

        # Select multiple mask channels per image
        random_indices = torch.stack([torch.randperm(self.nb_masks)[:nb_extracted_zones] for _ in range(batch_size)], dim=0) # selectes nb_extracted_zones DIFFERENT zones
        # random_indices = torch.randint(masks.shape[1], (batch_size, nb_extracted_zones), device=masks.device) #selects nb_extracted_zones zones
        batch_indices = torch.arange(batch_size, device=masks.device).unsqueeze(1)

        # Gather selected masks
        source_masks = masks[batch_indices, random_indices].sum(dim=1) > 0  # Combine multiple masks per image
        destination_masks = masks_shuffled[batch_indices, random_indices].sum(dim=1) > 0

        # Compute coverage proportion for each mask
        coverage = source_masks.view(batch_size, -1).sum(dim=1) / source_masks[0].numel()

        # Create a condition to filter out small coverage areas
        valid_mask = (coverage >= min_coverage_proportion)

        if valid_mask.any():
            # Select valid images
            valid_indices = valid_mask.nonzero().squeeze()

            # Get the masks and images for valid indices
            device = masks.device
            valid_indices = valid_indices.to(device)
            indices_shuffled = indices_shuffled.to(device)
            valid_images = images[valid_indices]
            valid_src_masks = source_masks[valid_indices]
            valid_dst_masks = destination_masks[valid_indices]

            # # Useful for small batch sizes -> reduced to 1
            # if valid_images.ndimension() == 3:
            #     valid_mask = valid_images.unsqueeze(0)
            # if valid_src_masks.ndimension() == 3:
            #     valid_src_masks = valid_src_masks.unsqueeze(0)
            # if valid_dst_masks.ndimension() == 3:
            #     valid_dst_masks = valid_dst_masks.unsqueeze(0)

            # Apply translation and mask combination
            extracted_zones = valid_images * valid_src_masks # shape (B_valid,C,H,W)
            
            if not self.roll_masks:
                # Apply CutMix on valid indices
                images_shuffled[valid_indices] = images_shuffled[valid_indices] * (~source_masks[valid_indices]) + extracted_zones
            else:
                # Compute translation vectors using centers of mass
                src_com = self.calculate_centers_of_mass(valid_src_masks)
                dst_com = self.calculate_centers_of_mass(valid_dst_masks)
                
                is_zero = torch.all(dst_com == torch.tensor([0,0], dtype=torch.int, device=device), dim=1)
                translation_vectors = torch.where(is_zero.unsqueeze(1), torch.tensor([0,0], dtype=torch.int, device=device), dst_com - src_com)
                # translation_vectors = torch.where(dst_com == 0, torch.zeros_like(dst_com), dst_com - src_com)
                translation_vectors = translation_vectors.round().to(torch.int64)
                # Useful in case batch size is reduced to 1
                if extracted_zones.ndimension() == 3:
                    extracted_zones = extracted_zones.unsqueeze(0)
                translated_batch = torch.zeros_like(extracted_zones)
                translated_masks = torch.zeros_like(valid_src_masks)
                for i in range(extracted_zones.shape[0]):
                    delta = translation_vectors[i].tolist()
                    # print(f'delta {delta}')
                    translated_batch[i] = torch.roll(extracted_zones[i], shifts=(delta[0],delta[1]), dims=(1, 2))
                    translated_masks[i] = torch.roll(valid_src_masks[i], shifts=(delta[0],delta[1]), dims=(1, 2))

                # Apply CutMix on valid indices
                images_shuffled[valid_indices] = images_shuffled[valid_indices] * (~translated_masks) + translated_batch

            coverage = nb_extracted_zones/6
            labels_shuffled[valid_indices] = (
                labels[valid_indices] * coverage +
                labels_shuffled[valid_indices] * (1 - coverage)
            )

        return images_shuffled, labels_shuffled
    
    def training_step(self, batch, batch_idx):
        images, labels, masks = batch
        if self.augment_technique == 'acm':
            np.random.randint(self.max_masks_to_apply+1-self.min_masks_to_apply)+self.min_masks_to_apply
            nb_extracted_zones = np.random.randint(self.nb_masks+1)+self.min_masks_to_apply
            images, labels = self.anatomical_cutmix_multi_zone(images,labels,masks, nb_extracted_zones=nb_extracted_zones)
        elif self.augment_technique == 'cm':
            images , targets_a, targets_b, lam ,box= self.cutmix_data(images, labels)
            labels= lam*targets_a+(1-lam)*targets_b
            
        output = self.forward(images)
        output=output[:,0]+output[:,1]
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output, labels)
        pc = metrics.pearson_corrcoef(output, labels)
        self.train_step_outputs.append({'loss': loss, 'mae': mae, "pc":pc})
        return {'loss': loss, 'mae': mae, "pc":pc}

    def validation_step(self, batch, batch_idx):
        images, labels, _ = batch
        output = self.forward(images)
        output=output[:,0]+output[:,1]
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output, labels)
        pc = metrics.pearson_corrcoef(output, labels)
        self.val_step_outputs.append({'loss': loss, 'mae': mae, "pc":pc})
        return {'loss': loss, 'mae': mae, "pc":pc}

    def test_step(self, batch, batch_idx):
        images, labels, _ = batch
        output = self.forward(images)
        output=output[:,0]+output[:,1]
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output, labels)
        pc = metrics.pearson_corrcoef(output, labels)
        self.labels_p = self.labels_p + output.squeeze().tolist()
        self.labels_gt = self.labels_gt + labels.squeeze().tolist()
        self.test_step_outputs.append({"loss": loss, "mae": mae, "pc":pc})
        return {"loss": loss, "mae": mae, "pc":pc}

    def on_train_epoch_end(self):
        loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        mae = torch.stack([x['mae'] for x in self.train_step_outputs]).mean()
        pc = torch.stack([x['pc'] for x in self.train_step_outputs]).mean()
        self.tr_loss.append(loss)
        self.tr_mae.append(mae)
        self.log('Loss/Train', loss, prog_bar=True, on_epoch = True, sync_dist=True)
        self.log('MAE/Train', mae, prog_bar=True, on_epoch = True, sync_dist=True)
        self.log('PC/Train', pc, prog_bar=True, on_epoch = True, sync_dist=True)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        mae = torch.stack([x['mae'] for x in self.val_step_outputs]).mean()
        pc = torch.stack([x['pc'] for x in self.val_step_outputs]).mean()
        self.loss = loss
        self.pc = pc
        self.vl_loss.append(loss)
        self.vl_mae.append(mae)
        self.log('Loss/Val', loss, prog_bar=True, on_epoch = True, sync_dist=True)
        self.log('MAE/Val', mae, prog_bar=True, on_epoch = True, sync_dist=True)
        self.log('PC/Val', pc, prog_bar=True, on_epoch = True, sync_dist=True)
        self.val_step_outputs.clear()

    def on_test_epoch_end(self):
        loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        mae = torch.stack([x['mae'] for x in self.test_step_outputs]).mean()
        pc = torch.stack([x['pc'] for x in self.test_step_outputs]).mean()
        mae_sdv = torch.stack([x['mae'] for x in self.test_step_outputs]).std()
        self.ts_loss.append(loss)
        self.ts_mae.append(mae)
        self.log('Loss/Test', loss, prog_bar=True, on_epoch = True, sync_dist=True)
        self.log('PC/Test', pc, prog_bar=True, on_epoch = True, sync_dist=True)
        self.log('MAE/Test', mae, prog_bar=True, on_epoch = True, sync_dist=True)
        self.log('MAE_SDV/Test',mae_sdv, prog_bar=True, on_epoch = True, sync_dist=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience, min_lr=self.lr_min)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'Loss/Val'}#, "interval": 'epoch'}

    def rand_bbox(self,size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

       # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_data(self,x, y, alpha= 1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        y_a = y
        y_b = y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        box=bbx1, bby1, bbx2, bby2
        return x, y_a, y_b, lam,box