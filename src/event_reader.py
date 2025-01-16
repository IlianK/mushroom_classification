import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

########## LIST & EXTRACT METRICS ###########

# List scalar tags
def list_scalar_tags(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    scalar_tags = event_acc.Tags()['scalars']
    
    print("Available Scalar Tags:")
    for tag in scalar_tags:
        print(tag)
    return scalar_tags


# Extract all metrics 
def extract_all_metrics(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    scalar_tags = event_acc.Tags()['scalars']
    
    metrics = {}
    for tag in scalar_tags:
        steps = [x.step for x in event_acc.Scalars(tag)]
        values = [x.value for x in event_acc.Scalars(tag)]
        metrics[tag] = {'steps': steps, 'values': values}
    
    return metrics


# Extract loss and accuracy 
def extract_loss_and_accuracy(metrics):
    return {
        "train_loss": metrics.get("Train/Loss", {}),
        "val_loss": metrics.get("Validation/Loss", {}),
        "train_accuracy": metrics.get("Train/Accuracy", {}),
        "val_accuracy": metrics.get("Validation/Accuracy", {}),
        "train_batch_loss": metrics.get("Train/BatchLoss", {}),
        "val_batch_loss": metrics.get("Validation/BatchLoss", {}),
        "train_batch_accuracy": metrics.get("Train/BatchAccuracy", {}),
        "val_batch_accuracy": metrics.get("Validation/BatchAccuracy", {}),
    }


########### PLOTTING ###########

# Plot a specific scalar metric
def plot_scalar_metric(log_dir, scalar_tag, title=None, save_path=None):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    scalar_values = [x.value for x in event_acc.Scalars(scalar_tag)]
    steps = [x.step for x in event_acc.Scalars(scalar_tag)]
    
    title = title or f'{scalar_tag} vs. Steps'
    plt.figure(figsize=(10, 6))
    plt.plot(steps, scalar_values, label=scalar_tag)
    plt.xlabel('Steps')
    plt.ylabel(scalar_tag)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"plot_{title.replace(' ', '_').lower()}.png"  
        plt.savefig(os.path.join(save_path, filename))
    plt.show()


# Plot accuracy or loss
def plot_metric(metrics, use_batch_steps, metric_type, save_path=None):
    train_metric = metrics.get(f"train_{metric_type}", {})
    val_metric = metrics.get(f"val_{metric_type}", {})
    train_batch_metric = metrics.get(f"train_batch_{metric_type}", {})
    val_batch_metric = metrics.get(f"val_batch_{metric_type}", {})
    
    plt.figure(figsize=(10, 5))
    steps_label = "Steps" if use_batch_steps else "Epoch"
    
    if use_batch_steps and "steps" in train_batch_metric and "values" in train_batch_metric:
        plt.plot(train_batch_metric["steps"], train_batch_metric["values"], label=f"Training {steps_label} {metric_type.capitalize()}", color='blue', linestyle='-')
    if not use_batch_steps and "steps" in train_metric and "values" in train_metric:
        plt.plot(train_metric["steps"], train_metric["values"], label=f"Training {steps_label} {metric_type.capitalize()}", color='blue', linestyle='--')
    
    if use_batch_steps and "steps" in val_batch_metric and "values" in val_batch_metric:
        plt.plot(val_batch_metric["steps"], val_batch_metric["values"], label=f"Validation {steps_label} {metric_type.capitalize()}", linestyle='--', color='red')
    if not use_batch_steps and "steps" in val_metric and "values" in val_metric:
        plt.plot(val_metric["steps"], val_metric["values"], label=f"Validation {steps_label} {metric_type.capitalize()}", linestyle='-.', color='red')

    plt.xlabel(steps_label)
    plt.ylabel(metric_type.capitalize())
    plt.title(f"Training and Validation {metric_type.capitalize()}")
    plt.legend()
    plt.grid()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"plot_{metric_type}.png"))
    plt.show()


# Plot loss
def plot_loss(loss_metrics, use_batch_steps=True, save_path=None):
    plot_metric(loss_metrics, use_batch_steps, "loss", save_path)


# Plot accuracy
def plot_accuracy(acc_metrics, use_batch_steps=True, save_path=None):
    plot_metric(acc_metrics, use_batch_steps, "accuracy", save_path)


# COmpare all models based on their event files
def compare_models(log_dir_base, model_types, use_batch_steps=False, save_path=None):
    """
    Compare all models based on their event files.

    Parameters:
        log_dir_base (str): Base directory containing model logs.
        model_types (list): List of model type subdirectories to compare.
        use_batch_steps (bool): Whether to use batch steps or epoch steps.
        save_path (str): Path to save comparison plots (optional).
    """
    all_losses = {}
    all_accuracies = {}

    for model_type in model_types:
        log_dir = os.path.join(log_dir_base, model_type, 'log')
        try:
            metrics = extract_all_metrics(log_dir)
            loss_accuracy_metrics = extract_loss_and_accuracy(metrics)

            if use_batch_steps:
                all_losses[model_type] = {
                    "train": loss_accuracy_metrics.get("train_batch_loss", {}),
                    "val": loss_accuracy_metrics.get("val_batch_loss", {}),
                }
                all_accuracies[model_type] = {
                    "train": loss_accuracy_metrics.get("train_batch_accuracy", {}),
                    "val": loss_accuracy_metrics.get("val_batch_accuracy", {}),
                }
            else:
                all_losses[model_type] = {
                    "train": loss_accuracy_metrics.get("train_loss", {}),
                    "val": loss_accuracy_metrics.get("val_loss", {}),
                }
                all_accuracies[model_type] = {
                    "train": loss_accuracy_metrics.get("train_accuracy", {}),
                    "val": loss_accuracy_metrics.get("val_accuracy", {}),
                }
        except Exception as e:
            print(f"Error processing model '{model_type}': {e}")
            continue

    # Plot Loss Comparison
    plt.figure(figsize=(12, 6))
    for model_type, losses in all_losses.items():
        if "steps" in losses["train"] and "values" in losses["train"]:
            plt.plot(losses["train"]["steps"], losses["train"]["values"], label=f'{model_type} Train Loss', linestyle='--')
        if "steps" in losses["val"] and "values" in losses["val"]:
            plt.plot(losses["val"]["steps"], losses["val"]["values"], label=f'{model_type} Validation Loss', linestyle='-.')

    steps_label = "Batch Steps" if use_batch_steps else "Epoch"
    plt.xlabel(steps_label)
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Models")
    plt.legend()
    plt.grid()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "comparison_loss.png"))
    plt.show()

    # Plot Accuracy Comparison
    plt.figure(figsize=(12, 6))
    for model_type, accuracies in all_accuracies.items():
        if "steps" in accuracies["train"] and "values" in accuracies["train"]:
            plt.plot(accuracies["train"]["steps"], accuracies["train"]["values"], label=f'{model_type} Train Accuracy', linestyle='--')
        if "steps" in accuracies["val"] and "values" in accuracies["val"]:
            plt.plot(accuracies["val"]["steps"], accuracies["val"]["values"], label=f'{model_type} Validation Accuracy', linestyle='-.')

    plt.xlabel(steps_label)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Models")
    plt.legend()
    plt.grid()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "comparison_accuracy.png"))
    plt.show()
