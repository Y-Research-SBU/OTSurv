from mil_models.model_otsurv import OTSurv


def create_survival_model(args):
    if args.loss_fn == 'nll':
        num_classes = args.n_label_bins
    elif args.loss_fn == 'cox':
        num_classes = 1
    elif args.loss_fn == 'rank':
        num_classes = 1

    if args.model_type == 'otsurv':
        model = OTSurv(
            num_classes=num_classes,
            patch_dim=args.feat_dim,
            hidden_dim=256,
            num_prototypes=16)   
          
    return model


