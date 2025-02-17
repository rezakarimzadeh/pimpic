import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .unet3d import Encoder3D

class projection_head(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),# affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),# affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.normalize(x)
        return x
    
class EncoderPiM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        projected_dim = config['projected_dim']
        self.eps = 1e-6
        
        self.model = Encoder3D(config)
        self.projection_head = projection_head(in_dim=self.get_projection_head_dims(), hidden_dim=projected_dim*2, 
                                               out_dim=projected_dim)
        
    def get_projection_head_dims(self):
        [x,y,z] = self.config['roi_size']
        ph_in_dim = (x//16)*(y//16)*(z//16) + (x//8)*(y//8)*(z//8) + (x//4)*(y//4)*(z//4)
        return ph_in_dim
    
    def forward(self, x):
        x, enc_features = self.model(x)
        [f1, f2, f3, f4, f5] = enc_features
        b_size = f5.shape[0]
        avg_pooled = torch.cat([f.mean(dim=1).view(b_size, -1) for f in [f3,f4,f5]], dim=1)
        projected_features = self.projection_head(avg_pooled)
        return x, enc_features, projected_features

    def pim_loss(self, projected_features, intersection_matrix):
        similarity_matrix = torch.matmul(projected_features, projected_features.T)  # (num_atoms, num_atoms)
        loss = - torch.log(1 - torch.clamp(torch.abs(similarity_matrix - intersection_matrix), 0, 1) + self.eps).mean() 
        return loss
    
    def training_step(self, batch, batch_idx):
        patches, intersection_matrix, start_stop_coords, img_shape = batch
        _, _, projected_features = self(patches)
        loss = self.pim_loss(projected_features, intersection_matrix)
        self.log("train_loss_pim", loss, on_epoch=True, on_step=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer
      

class ContrastiveLoss(nn.Module):
    def __init__(self, tau=1.0, normalize=False):
        """
        Contrastive loss function for inputs of shape [batch, 4, features] where
        the second dimension represents (p1, p2, n1, n2).

        Args:
            tau (float): Temperature scaling parameter.
            normalize (bool): If True, normalize vectors to unit norm before computing similarity.
        """
        super(ContrastiveLoss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, x):
        """
        Compute the contrastive loss for inputs [batch, 4, features].

        Args:
            x (torch.Tensor): Tensor of shape [batch, 4, features], where the second dimension
                              contains (p1, p2, n1, n2).

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        batch_size, num_samples, feature_dim = x.shape
        assert num_samples == 4, "Input tensor must have 4 samples per batch (p1, p2, n1, n2)."
        # Split the input into positive and negative pairs
        p1, p2, n1, n2 = x[:, 0, :], x[:, 1, :], x[:, 2, :], x[:, 3, :]
        # Normalize features if required
        if self.normalize:
            p1 = nn.functional.normalize(p1, dim=-1)
            p2 = nn.functional.normalize(p2, dim=-1)
            n1 = nn.functional.normalize(n1, dim=-1)
            n2 = nn.functional.normalize(n2, dim=-1)

        # Compute similarity scores
        sim_p = torch.exp(torch.sum(p1 * p2, dim=-1) / self.tau)  # Positive pair similarity
        sim_n1 = torch.exp(torch.sum(p1 * n1, dim=-1) / self.tau)  # Negative pair 1 similarity
        sim_n2 = torch.exp(torch.sum(p1 * n2, dim=-1) / self.tau)  # Negative pair 2 similarity

        sim_n21 = torch.exp(torch.sum(p2 * n1, dim=-1) / self.tau)  # Negative pair 1 similarity
        sim_n22 = torch.exp(torch.sum(p2 * n2, dim=-1) / self.tau)  # Negative pair 2 similarity
        # Denominator: sum of all similarities (excluding self-similarity)
        denom = 2*sim_p + sim_n1 + sim_n2 + sim_n21 + sim_n22
        # Contrastive loss for positive pair (p1, p2)
        loss = -torch.log(2*sim_p / denom)

        # Average over the batch
        return loss.mean()


class EncoderPiC(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        projected_dim = config['projected_dim']
        self.encoder = Encoder3D(config)
        self.projection_head_IC = projection_head(in_dim=config["base_features"]*7, hidden_dim=projected_dim, 
                                               out_dim=projected_dim)
        self.cl_loss_fn = ContrastiveLoss()
        self.eps = 1e-9

    def forward(self, x):
        x, enc_features  = self.encoder(x)
        return x, enc_features
 
    def get_intersection_coords(self, tensor, intersection_matrix):
        """
        Compute the intersections between bounding boxes in a batch.

        Args:
            tensor (torch.Tensor): Tensor of shape [b, 2, 3], where b is the batch size.
                                Each bounding box is represented as [[start_x, start_y, start_z],
                                                                    [end_x, end_y, end_z]].

        Returns:
            List[Tuple[int, int, torch.Tensor, torch.Tensor]]: A list of tuples where each tuple contains:
                                                - The indices of the pair of intersecting bounding boxes.
                                                - The intersection start and end coordinates relative to box 1's origin.
                                                - The intersection start and end coordinates relative to box 2's origin.
        """
        batch_size = tensor.size(0)
        intersections = []

        # Iterate over all pairs of bounding boxes
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if intersection_matrix[i,j] < 0.01:
                    # print('less than 1 percent')
                    continue
                # Compute the start and end of the intersection
                start_intersection = torch.max(tensor[i, 0], tensor[j, 0])  # Max of start coordinates (x, y, z)
                end_intersection = torch.min(tensor[i, 1], tensor[j, 1])    # Min of end coordinates (x, y, z)

                # Check if there is an intersection in all dimensions (x, y, z)
                if torch.all(end_intersection > start_intersection):
                    # Transform to the new coordinate system for both boxes
                    intersection_box1 = torch.stack([start_intersection - tensor[i, 0],
                                                    end_intersection - tensor[i, 0]], dim=0)
                    intersection_box2 = torch.stack([start_intersection - tensor[j, 0],
                                                    end_intersection - tensor[j, 0]], dim=0)

                    intersections.append((i, j, intersection_box1, intersection_box2))

        return intersections
    
    def compute_average_features_with_scaling(self, feature_maps, intersections, scaling_factor):
        """
        Compute the average feature values for intersected and non-intersected regions, keeping channel dimensions,
        with a scaling coefficient applied to coordinates.

        Args:
            feature_maps (torch.Tensor): Tensor of shape [b, c, x, y, z], where b is the batch size,
                                        c is the number of channels, and x, y, z are spatial dimensions.
            intersections (List[Tuple[int, int, torch.Tensor, torch.Tensor]]): Intersection details.
            scaling_factor (float): Scaling factor for coordinates.

        Returns:
            torch.Tensor: Output tensor of shape [num_of_pairs, 4, c].
        """
        batch_size, channels, x, y, z = feature_maps.shape
        
        # Apply scaling and round up coordinates
        def scale_coords_start(coords):
            return (coords.float() / scaling_factor).floor().long()
        def scale_coords_end(coords):
            return (coords.float() / scaling_factor).ceil().long()
        
        outputs = []
        for intersection in intersections:
            b0, b1, box1_coords, box2_coords = intersection
            
            start1, end1 = scale_coords_start(box1_coords[0]), scale_coords_end(box1_coords[1])
            start2, end2 = scale_coords_start(box2_coords[0]), scale_coords_end(box2_coords[1])
            # Extract intersection regions
            intersection_box1 = feature_maps[b0, :, start1[0]:end1[0], start1[1]:end1[1], start1[2]:end1[2]]
            intersection_box2 = feature_maps[b1, :, start2[0]:end2[0], start2[1]:end2[1], start2[2]:end2[2]]

            # Compute sums and counts for intersections
            intersection_sum_box1 = intersection_box1.sum(dim=(1, 2, 3))
            intersection_count_box1 = intersection_box1.numel() / channels

            intersection_sum_box2 = intersection_box2.sum(dim=(1, 2, 3))
            intersection_count_box2 = intersection_box2.numel() / channels

            # Compute averages for intersections
            avg_intersection_box1 = intersection_sum_box1 / (intersection_count_box1+self.eps)
            avg_intersection_box2 = intersection_sum_box2 / (intersection_count_box2+self.eps)

            # Compute non-intersected regions
            def compute_non_intersection(b, start, end):
                non_intersection_mask = torch.ones((x, y, z), dtype=torch.bool, device=feature_maps.device)
                non_intersection_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = False
                non_intersection_box = feature_maps[b, :, non_intersection_mask]
                non_intersection_sum = non_intersection_box.sum(dim=1)
                non_intersection_count = non_intersection_box.numel() / channels
                return non_intersection_sum / (non_intersection_count+self.eps)

            avg_non_intersection_box1 = compute_non_intersection(b0, start1, end1)
            avg_non_intersection_box2 = compute_non_intersection(b1, start2, end2)
            # Stack results for this pair
            outputs.append(torch.stack([
                avg_intersection_box1,
                avg_intersection_box2,
                avg_non_intersection_box1,
                avg_non_intersection_box2
            ], dim=0))

        # Concatenate results for all pairs
        if outputs:
            return torch.stack(outputs, dim=0)
        else:
            return None
        
    def perform_projection_head_IC(self, input_tensor):
        batchsize, num_groups, features = input_tensor.shape
        input_reshaped = input_tensor.view(-1, features)  
        output_reshaped = self.projection_head_IC(input_reshaped) 
        output_tensor = output_reshaped.view(batchsize, num_groups, -1)  
        return output_tensor

    def get_projected_features(self, start_stop_coords, enc_features, intersection_matrix):
        # [f1, f2, f3, f4, f5]  = enc_features
        intersection_coords = self.get_intersection_coords(start_stop_coords, intersection_matrix)
        feature_pyramid = []
        for i, dfi in enumerate(enc_features):
            if i<3:
                avg_intersection = self.compute_average_features_with_scaling(feature_maps=dfi, 
                                                                                intersections=intersection_coords,
                                                                                scaling_factor=2**(i))
                if avg_intersection is not None:
                    feature_pyramid.append(avg_intersection)
        if feature_pyramid:
            cat_feature_pyramid = torch.cat(feature_pyramid, dim=-1)
            projected_featuers = self.perform_projection_head_IC(cat_feature_pyramid)
            return projected_featuers
        else:
            return None
    
    def pic_loss(self, start_stop_coords, enc_features, intersection_matrix):
        enc_loss = 0
        projected_features = self.get_projected_features(start_stop_coords, enc_features, intersection_matrix)
        if projected_features is not None:
            enc_loss = self.cl_loss_fn(projected_features)
        return enc_loss
    
    def training_step(self, batch, batch_idx):
        patches, intersection_matrix, start_stop_coords, img_shape = batch
        x, enc_features = self(patches)
        enc_loss = self.pic_loss(start_stop_coords, enc_features, intersection_matrix)
        self.log("train_loss_pic", enc_loss, on_epoch=True, on_step=True)
        return enc_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer
    

class EncoderPiMPiC(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        projected_dim = config['projected_dim']
        self.eps = 1e-6
        
        self.encoder = EncoderPiC(config)
        self.projection_head_IP = projection_head(in_dim=self.get_projection_head_dims(), hidden_dim=projected_dim*2, 
                                               out_dim=projected_dim)
        
    def get_projection_head_dims(self):
        [x,y,z] = self.config['roi_size']
        ph_in_dim = (x//16)*(y//16)*(z//16) + (x//8)*(y//8)*(z//8) + (x//4)*(y//4)*(z//4)
        return ph_in_dim
    
    def forward(self, x):
        x, enc_features = self.encoder(x)
        return x, enc_features
    
    def get_intersection_prediction_projections(self, enc_features):
        [f1, f2, f3, f4, f5] = enc_features
        b_size = f5.shape[0]
        avg_pooled = torch.cat([f.mean(dim=1).view(b_size, -1) for f in [f3,f4,f5]], dim=1)
        projected_features = self.projection_head_IP(avg_pooled)
        return projected_features
    
    def pim_loss(self, projected_features, intersection_matrix):
        similarity_matrix = torch.matmul(projected_features, projected_features.T)  # (num_atoms, num_atoms)
        loss = - torch.log(1 - torch.clamp(torch.abs(similarity_matrix - intersection_matrix), 0, 1) + self.eps).mean() 
        return loss
    
    def training_step(self, batch, batch_idx):
        patches, intersection_matrix, start_stop_coords, img_shape = batch
        x, enc_features = self(patches)
        intersection_prediction_projection = self.get_intersection_prediction_projections(enc_features)
        pim_loss_val = self.pim_loss(intersection_prediction_projection, intersection_matrix)
        pic_loss_val = self.encoder.pic_loss(start_stop_coords, enc_features, intersection_matrix)
        loss = pim_loss_val + pic_loss_val
        self.log("train_loss_pim", pim_loss_val, on_epoch=True, on_step=True)
        self.log("train_loss_pic", pic_loss_val, on_epoch=True, on_step=True)
        self.log("train_loss_pimpic", loss, on_epoch=True, on_step=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer
