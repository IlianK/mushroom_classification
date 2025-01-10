import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# List scalar tags
def list_scalar_tags(log_dir):
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    scalar_tags = event_acc.Tags()['scalars']
    
    print("Available Scalar Tags:")
    for tag in scalar_tags:
        print(tag)
    return scalar_tags


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
        "val_accuracy": metrics.get("Validation/Accuracy", {})
    }


# Plot loss
def plot_loss(loss_metrics, save_path=None):
    train_loss = loss_metrics.get("train_loss", {})
    val_loss = loss_metrics.get("val_loss", {})
    
    plt.figure(figsize=(10, 5))
    if "steps" in train_loss and "values" in train_loss:
        plt.plot(train_loss["steps"], train_loss["values"], label="Training Loss", color='blue')
    if "steps" in val_loss and "values" in val_loss:
        plt.plot(val_loss["steps"], val_loss["values"], label="Validation Loss", linestyle='--', color='red')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "plot_loss.png"))
    plt.show()


# Plot accuracy
def plot_accuracy(acc_metrics, save_path=None):
    train_accuracy = acc_metrics.get("train_accuracy", {})
    val_accuracy = acc_metrics.get("val_accuracy", {})
    
    plt.figure(figsize=(10, 5))
    if "steps" in train_accuracy and "values" in train_accuracy:
        plt.plot(train_accuracy["steps"], train_accuracy["values"], label="Training Accuracy", color='blue')
    if "steps" in val_accuracy and "values" in val_accuracy:
        plt.plot(val_accuracy["steps"], val_accuracy["values"], label="Validation Accuracy", linestyle='--', color='red')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "plot_accuracy.png"))
    plt.show()


# Compare models
def compare_models(log_dir_base, model_types, save_path=None):  
    all_losses = {}
    all_accuracies = {}
    
    for model_type in model_types:
        log_dir = os.path.join(log_dir_base, model_type, 'log')
        try:
            metrics = extract_all_metrics(log_dir)
            loss_accuracy_metrics = extract_loss_and_accuracy(metrics)
            
            # Collect loss data
            all_losses[model_type] = {
                "train": loss_accuracy_metrics.get("train_loss", {}),
                "val": loss_accuracy_metrics.get("val_loss", {})
            }
            
            # Collect accuracy data
            all_accuracies[model_type] = {
                "train": loss_accuracy_metrics.get("train_accuracy", {}),
                "val": loss_accuracy_metrics.get("val_accuracy", {})
            }
        except Exception as e:
            print(f"Error processing model '{model_type}': {e}")
            continue
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    for model_type, losses in all_losses.items():
        if "steps" in losses["train"] and "values" in losses["train"]:
            plt.plot(losses["train"]["steps"], losses["train"]["values"], label=f'{model_type} Train Loss', linestyle='-')
        if "steps" in losses["val"] and "values" in losses["val"]:
            plt.plot(losses["val"]["steps"], losses["val"]["values"], label=f'{model_type} Validation Loss', linestyle='--')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Models")
    plt.legend()
    plt.grid()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "comparison_loss.png"))
    plt.show()
    
    # Plot accuracies
    plt.figure(figsize=(12, 6))
    for model_type, accuracies in all_accuracies.items():
        if "steps" in accuracies["train"] and "values" in accuracies["train"]:
            plt.plot(accuracies["train"]["steps"], accuracies["train"]["values"], label=f'{model_type} Train Accuracy', linestyle='-')
        if "steps" in accuracies["val"] and "values" in accuracies["val"]:
            plt.plot(accuracies["val"]["steps"], accuracies["val"]["values"], label=f'{model_type} Validation Accuracy', linestyle='--')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Models")
    plt.legend()
    plt.grid()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "comparison_accuracy.png"))
    plt.show()
