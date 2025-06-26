import torch

from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss


def process_surv(logits, label, censorship, loss_fn=None):
    results_dict = {'logits': logits}
    log_dict = {}

    if loss_fn is not None and label is not None:
        if isinstance(loss_fn, NLLSurvLoss):
            surv_loss_dict = loss_fn(logits=logits, times=label, censorships=censorship)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).unsqueeze(dim=1)
            results_dict.update({'hazards': hazards,
                                    'survival': survival,
                                    'risk': risk})
        elif isinstance(loss_fn, CoxLoss):
            # logits is log risk
            surv_loss_dict = loss_fn(logits=logits, times=label, censorships=censorship)
            risk = torch.exp(logits)
            results_dict['risk'] = risk

        elif isinstance(loss_fn, SurvRankingLoss):
                surv_loss_dict = loss_fn(z=logits, times=label, censorships=censorship)
                results_dict['risk'] = logits

        loss = surv_loss_dict['loss']
        log_dict['surv_loss'] = surv_loss_dict['loss'].item()
        log_dict.update(
            {k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})
        results_dict.update({'loss': loss})

    return results_dict, log_dict


def replace_nan(features):
    """
    Replace NaN values in feature tensors with appropriate values.
    
    Args:
        features (torch.Tensor): Input feature tensor of shape (batch_size, feature_dim)
        
    Returns:
        torch.Tensor: Feature tensor with NaN values replaced
    """
    if not torch.isnan(features).any():
        return features
        
    # Create a copy to avoid modifying the original tensor
    cleaned_features = features.clone()
    
    for i in range(features.size(0)):
        sample_features = features[i]
        
        if torch.isnan(sample_features).all():
            # If all features are NaN, use zero as default
            replacement_value = 0.0
        else:
            # Use mean of non-NaN values as replacement
            replacement_value = torch.nanmean(sample_features).item()
        
        # Replace NaN values with the computed replacement value
        cleaned_features[i] = torch.nan_to_num(sample_features, nan=replacement_value)
    
    return cleaned_features