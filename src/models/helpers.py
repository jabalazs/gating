import warnings

import numpy as np

from tqdm import tqdm
from sklearn.metrics import classification_report

from .. import config


def evaluate(model, batches, silent=False):
    model.eval()
    num_batches = batches.num_batches
    outputs = []
    true_labels = []
    for batch_index in range(num_batches):
        batch = batches[batch_index]
        out = model(batch)
        outputs.append(out["logits"].cpu().detach().numpy())
        true_labels.extend(batch["labels"])

    output = np.vstack(outputs)
    pred_labels = output.argmax(axis=1)
    true_labels = np.array(true_labels)

    num_correct = (pred_labels == true_labels).sum()
    num_total = len(pred_labels)
    accuracy = num_correct / num_total

    if not silent:
        # We do this for silencing sklearn's warning when calculating the f1 score of a
        # class with 0 predictions
        warnings.filterwarnings("ignore")
        tqdm.write(
            classification_report(
                true_labels, pred_labels, target_names=config.LABELS
            )
        )
        warnings.filterwarnings("default")
        tqdm.write(f"\nAccuracy: {accuracy:.3f}\n")

    pred_labels = pred_labels.tolist()
    pred_labels = [config.ID2LABEL[label] for label in pred_labels]
    ret_dict = {"accuracy": accuracy, "labels": pred_labels, "output": output}
    return ret_dict
