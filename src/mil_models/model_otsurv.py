import torch
from torch import nn

from .components import process_surv, replace_nan
from .otsurv_component.ot_attn import OT_Attn


class OTSurv(nn.Module):
    """
    OTSurv: A Novel Multiple Instance Learning Framework for Survival Prediction 
    with Heterogeneity-aware Optimal Transport.

    Args:
        num_classes (int): Number of output classes for survival prediction
        patch_dim (int): Input feature dimension of patches
        hidden_dim (int): Hidden dimension for feature encoding
        num_prototypes (int): Number of learnable prototype embeddings for OT aggregation
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(
            self,
            num_classes=1,
            patch_dim=1024,
            hidden_dim=256,
            num_prototypes=16,
            dropout_rate=0.25
            ):
        super().__init__()
        
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Optimal transport attention mechanism for heterogeneity-aware aggregation
        self.ot_attn = OT_Attn(impl="hot")
        
        # Learnable prototype embeddings for optimal transport
        self.prototype_embeddings = nn.Embedding(num_prototypes, hidden_dim)
        
        # Linear layer for prototype aggregation
        self.linear = nn.Linear(num_prototypes, 1)
        
        # Final classifier for survival prediction
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward_no_loss(self, wsi_features, return_attn=False):
        """
        Forward pass without computing loss.
        
        Args:
            wsi_features: List of WSI Features. The last two elements are:
                   - wsi_features[-2]: iterations_per_epoch
                   - wsi_features[-1]: iterations
            return_attn: Whether to return attention weights

        """
        *features_list, iterations_per_epoch, iterations = wsi_features

        h_path_list = []
        Attn_OT_list = []
        for features in features_list:
            # Encode patch features
            encoded_features = self.patch_encoder(features)
            encoded_features = replace_nan(encoded_features)

            # Apply optimal transport attention
            # encoded_features:[Ni, D]
            # self.prototype_embeddings.weight:[K, D]
            # Attn_OT:[Ni, K]
            Attn_OT, _ = self.ot_attn(encoded_features, self.prototype_embeddings.weight, iterations, iterations_per_epoch)
            aggregated_feature = torch.mm(Attn_OT.T, encoded_features)
            
            h_path_list.append(aggregated_feature.unsqueeze(0))
            Attn_OT_list.append(Attn_OT)

        # h_path: [B, K, D]
        h_path = torch.cat(h_path_list, dim=0)

        # h_path: [B, D]
        h_path = self.linear(h_path.transpose(-1, -2)).squeeze(-1)

        # logits: [B, 1]
        logits = self.classifier(h_path)

        out = {'logits': logits}
        if return_attn:
            out['Attn_OT'] = Attn_OT_list

        return out

    def forward(self, x_path, return_attn=False, label=None, censorship=None, loss_fn=None):
        out = self.forward_no_loss(x_path, return_attn)
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        results_dict.update(out)

        return results_dict, log_dict