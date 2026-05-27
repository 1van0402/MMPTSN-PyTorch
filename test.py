"""Model evaluation utilities for generating predictions on the test set."""

import torch

def predicted(test_loader, model):
    """Evaluate the model on a test loader and return ground-truth labels and predicted labels."""

    with torch.no_grad():
        model.eval()
        all_labels, all_predictions = [], []

        for spectral_data, spatial_data, sar_data, phe_data, labels in test_loader:
            spectral_data = spectral_data.cuda()
            spatial_data  = spatial_data.cuda()
            sar_data      = sar_data.cuda()
            phe_data      = phe_data.cuda()
            labels        = labels.cuda()

            logits = model(spectral_data, spatial_data, sar_data, phe_data)
            predicted_tmp = torch.argmax(logits, 1)
            all_predictions.append(predicted_tmp.cpu())
            all_labels.append(labels.cpu())

        all_labels = torch.cat(all_labels, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)

        return all_labels, all_predictions