import numpy as np
from transformers import TrainerCallback, Trainer
import time
import json
from torch.utils.data import DataLoader
from sklearn.metrics import brier_score_loss, mean_squared_error, mean_absolute_error, r2_score, precision_recall_fscore_support, accuracy_score, matthews_corrcoef
import torch
from scipy.stats import pearsonr, spearmanr
import os
import pandas as pd
from typing import List, Optional, Dict
from datasets import Dataset

def compute_mcc(labels, preds):
    """
    Computes the Matthews Correlation Coefficient (MCC) in different scenarios:
      1) Binary classification (labels and preds are 1D, up to 2 classes)
      2) Multi-class classification (labels and preds are 1D, more than 2 classes)
      3) Multi-label classification (labels and preds are 2D; per-label MCC is averaged)
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    if labels.ndim == 1:
        # Binary or multi-class single-label scenario: compute MCC once
        return matthews_corrcoef(labels, preds)

    elif labels.ndim == 2:
        # Multi-label scenario: compute MCC for each label separately, then average
        num_labels = labels.shape[1]
        mcc_per_label_weighted = []
        mcc_per_label_absolute = []
        
        y_true_flat = labels.reshape(-1)
        y_pred_flat = preds.reshape(-1)
        mcc_micro = matthews_corrcoef(y_true_flat, y_pred_flat)
        for c in range(num_labels):
            mcc_c_ = matthews_corrcoef(labels[:, c], preds[:, c])
            mcc_per_label_weighted.append(mcc_c_)
            mcc_per_label_absolute.append(abs(mcc_c_))
        return float(np.mean(mcc_per_label_weighted)), float(np.mean(mcc_per_label_absolute)), mcc_micro

    else:
        # Catch-all (unlikely scenario): return 0 or raise an error
        return 0.0

def compute_metrics_classification(pred):
    labels = pred.label_ids
    preds = pred.predictions

    # For multi-label classification (e.g., GoEmotions)
    if len(preds.shape) > 1 and preds.shape[1] > 1:  # Multilabel or multiclass logits
        # Convert logits to probabilities and apply a threshold for multi-label
        preds = (preds > 0.5).astype(int) if labels.shape == preds.shape else preds.argmax(axis=-1)
    
    # Ensure labels and preds are in the same format
    if len(labels.shape) > 1:  # Multilabel case
        if len(preds.shape) == 1:  # Convert labels to flat format if preds are flat
            labels = labels.argmax(axis=1)
        else:  # Otherwise, keep as multilabel
            pass
    else:  # Single-label case
        preds = preds.argmax(axis=-1) if len(preds.shape) > 1 else preds

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    #mcc_weighted, mcc_absolute, mcc_micro = compute_mcc(labels, preds) 
    mcc = compute_mcc(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        #'mcc_weighted': mcc_weighted,
        #'mcc_absolute': mcc_absolute,
        #'mcc_micro': mcc_micro
        'mcc': mcc
    }

def compute_metrics_regression(pred):
    labels = pred.label_ids
    preds = pred.predictions

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().ravel()
    else:
        labels = labels.ravel()

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy().ravel()
    else:
        preds = preds.ravel()

    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    pearson = float(pearsonr(labels, preds)[0])
    spearman = float(spearmanr(labels, preds)[0])

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


def softmax(logits, axis=-1):
    """Numpy softmax for multi-class probabilities."""
    exp = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)

def sigmoid(logits):
    """Numpy sigmoid for multi-label or binary single-logit."""
    return 1 / (1 + np.exp(-logits))

def compute_ece(y_true, y_proba, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE) for classification.
    For multi-class or multi-label, this will be computed per class and averaged.
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    # We'll handle shape consistency outside this function.
    # Here we assume y_true, y_proba are 1D for binary or 2D (N, #classes) for multi-class/label.

    if y_proba.ndim == 1:
        # Binary classification (prob of positive class)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1

        ece = 0.0
        for b in range(n_bins):
            bin_mask = (bin_indices == b)
            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                avg_confidence = np.mean(y_proba[bin_mask])
                avg_accuracy = np.mean(y_true[bin_mask])
                ece += (bin_size / len(y_true)) * abs(avg_confidence - avg_accuracy)
        return ece
    else:
        # Multi-class or multi-label shape: (num_samples, num_classes)
        # We'll compute ECE for each column independently, then average
        num_classes = y_proba.shape[1]
        ece_values = []
        for c in range(num_classes):
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_proba[:, c], bins) - 1
            ece_c = 0.0
            for b in range(n_bins):
                bin_mask = (bin_indices == b)
                bin_size = np.sum(bin_mask)
                if bin_size > 0:
                    avg_confidence = np.mean(y_proba[bin_mask, c])
                    avg_accuracy = np.mean(y_true[bin_mask, c])
                    ece_c += (bin_size / len(y_proba)) * abs(avg_confidence - avg_accuracy)
            ece_values.append(ece_c)
        return float(np.mean(ece_values))

class GeneralCalibrationCallback(TrainerCallback):
    """
    A generalized TrainerCallback that handles:
      - Binary classification
      - Multi-class classification
      - Multi-label classification
      
    It computes Brier Score & ECE for classification tasks (binary, multi-class, multi-label).
    Logs them in HF Trainer metrics at each evaluation.
    """
    def __init__(self, output_dir:str, n_bins=10):
        super().__init__()
        self.n_bins = n_bins
        self.output_dir = output_dir
        self.eval_results = []
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"[Callback] on_evaluate called at step {state.global_step}")
        if not metrics:
            return control

        # Retrieve your predictions
        logits = metrics.get("eval_predictions", None)
        labels = metrics.get("eval_label_ids", None)

        if logits is None or labels is None:
            print("[Callback] No predictions found in metrics!")
            return control
        
        # Decide which scenario we're in:
        # 1) Regression: typically predictions/logits are just floats, shape (N,) or (N,1)
        # 2) Binary classification: logits could be shape (N,) or (N,2)
        # 3) Multi-class classification: shape (N, C) with C>2
        # 4) Multi-label classification: shape (N, C) + labels are also shape (N, C) in {0,1}
        
        # We also check the data type / shape of labels to see if they're 0/1 or continuous.
        labels = np.array(labels)
        if labels.ndim == 2 and not np.issubdtype(labels.dtype, np.floating) and set(np.unique(labels)).issubset({0,1}):
            # Could be multi-label classification if more than 1 column
            # or binary classification if shape is (N, 1)
            if labels.shape[1] == 1:
                # Might be a special case for binary classification, flatten
                labels = labels.ravel()
        
        # Heuristic approach to detect task type:
        task_type = self._detect_task_type(logits, labels)

        if task_type == "binary":
            # Binary classification
            # Convert logits to probabilities
            if logits.ndim == 2 and logits.shape[1] == 2:
                probs = softmax(logits, axis=1)[:, 1]  # take positive class prob
            elif logits.ndim == 1:
                probs = sigmoid(logits)
            else:
                # Unexpected shape
                return

            # Brier Score
            brier = brier_score_loss(labels, probs)
            
            # ECE
            ece_val = compute_ece(labels, probs, n_bins=self.n_bins)
            self._log_metric(state, control, kwargs, {"brier_score": brier, "ece": ece_val})
            self.eval_results.append({"step": state.global_step, "brier_score": float(brier), "ece": float(ece_val)})
        
        elif task_type == "multiclass":
            # Multi-class classification: shape (N, C), C>2
            # Convert logits to prob distribution
            probs = softmax(logits, axis=1)
            # For Brier Score in multi-class, we treat labels as one-hot, then compute MSE
            # Brier Score (multi-class generalization) = mean of (y_proba - y_onehot)^2 across classes
            num_classes = probs.shape[1]
            y_onehot = np.eye(num_classes)[labels]  # shape (N, C)

            brier_vals = (probs - y_onehot)**2  # element-wise
            brier_score_mc = np.mean(np.sum(brier_vals, axis=1))  # average over samples

            # ECE for multi-class: compute ECE per class and average
            ece_val = compute_ece(y_onehot, probs, n_bins=self.n_bins)
            self._log_metric(state, control, kwargs, {"brier_score": brier_score_mc, "ece": ece_val})
            self.eval_results.append({"step": state.global_step, "brier_score": float(brier_score_mc), "ece": float(ece_val)})
            
        elif task_type == "multilabel":
            # Multi-label classification: shape (N, C), labels also shape (N, C)
            # Convert logits to probabilities with sigmoid
            probs = sigmoid(logits)
            # Brier Score for multi-label: average of MSE across all classes
            brier_scores = []
            ece_scores = []

            for c in range(probs.shape[1]):
                brier_c = brier_score_loss(labels[:, c], probs[:, c])
                ece_c = compute_ece(labels[:, c], probs[:, c], n_bins=self.n_bins)
                brier_scores.append(brier_c)
                ece_scores.append(ece_c)

            brier_ml = float(np.mean(brier_scores))
            ece_ml = float(np.mean(ece_scores))
            self._log_metric(state, control, kwargs, {"brier_score": brier_ml, "ece": ece_ml})
            self.eval_results.append({"epoch": state.epoch if state.epoch is not None else None, 
                                      "brier_score": brier_ml, 
                                      "ece": ece_ml})
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            df_calib = pd.DataFrame(self.eval_results)
            df_calib.to_json(os.path.join(self.output_dir, "calibration_metrics.json"), orient="records")
        
        return control

    def _detect_task_type(self, logits, labels):
        """
        Simple heuristic to identify the task type based on shapes and label distribution.
        """
        # Classification (labels are integer):
        # Distinguish between binary, multiclass, or multi-label
        if labels.ndim == 1:
            # Could be binary or multi-class
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 2:
                # Binary classification
                return "binary"
            else:
                return "multiclass"
        elif labels.ndim == 2:
            # Possibly multi-label: each sample has a vector of 0/1
            # If each row has more than one 1 or 0 present, it's multi-label
            if set(np.unique(labels)).issubset({0,1}):
                # multi-label
                return "multilabel"
            else:
                # if they're integers > 1 or inconsistent, might be some unusual format
                return "multiclass"
        else:
            # Catch-all
            return "multiclass"

    def _log_metric(self, state, control, kwargs, metrics_dict):
        """Helper to log metrics in Trainer format."""
        if state.is_local_process_zero:
            control.should_log = True
            if "logs" not in kwargs:
                kwargs["logs"] = {}
            kwargs["logs"].update(metrics_dict)


class GradientStatsCallback(TrainerCallback):
    """
    A TrainerCallback to compute mean and variance of gradients 
    across all model parameters at each training step.
    """
    def __init__(self, store_stats: bool, output_dir: str):
        """
        Args:
            store_stats (bool): Whether to store the computed stats 
                                in memory for later inspection.
        """
        self.store_stats = store_stats
        self.output_dir = output_dir
        self.grad_means = []  # Store gradient means per step
        self.grad_vars = []   # Store gradient variances per step

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        This hook is called at the end of every optimizer step. 
        We compute the mean/variance of the gradients here.
        """
        #print("on_step_end called: step =", state.global_step)
        if model is None:
            return

        # Collect gradients from all parameters (on CPU).
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                # Copy gradient to CPU and flatten
                print("Grad shape:", param.grad.shape, "Mean:", param.grad.detach().cpu().float().mean())
                grad_list.append(param.grad.detach().cpu().view(-1).numpy())

        if not grad_list:
            return

        # Concatenate all parameter gradients into one array
        all_grads = np.concatenate(grad_list, axis=0)

        # Compute statistics
        grad_mean = float(all_grads.mean())
        grad_var = float(all_grads.var())

        # Optionally store for later
        if self.store_stats:
            self.grad_means.append(grad_mean)
            self.grad_vars.append(grad_var)

        # Optionally log the stats using the Trainer's logger
        # The trainer will show these under the metrics in console or in logs
        if state.is_local_process_zero:  # Only log once in multi-GPU setup
            # Using the internal logger:
            control.should_log = True  # ensures logs get recorded this step
            kwargs["logs"] = kwargs.get("logs", {})
            kwargs["logs"].update({
                "grad_mean": grad_mean,
                "grad_variance": grad_var
            })

        return control
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
    # This method is called once training is done
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            df_grad = pd.DataFrame({
                "grad_mean": self.grad_means,
                "grad_variance": self.grad_vars
            })
            df_grad.to_json(os.path.join(self.output_dir, "gradient_stats.json"), orient="records")


class SaveEvaluationResultsCallback(TrainerCallback):
    def __init__(self, benchmark, task, experiment, implementation, seed):
        super().__init__()
        self.benchmark = benchmark
        self.task = task
        self.experiment = experiment
        self.implementation = implementation
        self.seed = seed
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        with open(f"results_hf/{self.implementation}_BERT_FINAL_{self.experiment}_seed_{self.seed}/{self.benchmark}/{self.task}/eval_metrics_epoch_{state.epoch}.txt", 'w') as f:
            f.write(str(metrics))

class SaveLossesCallback(TrainerCallback):
    "A callback that saves the training loss at each logging step"
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            # We'll only save the losses from the first process to avoid writing the same file multiple times
            with open(f'{self.output_dir}/losses.json', 'a') as f:
                json.dump(logs, f)
                f.write('\n')


class AlphaWeightsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        optimizer = kwargs['optimizer']
        
        # Print alpha_weights before optimizer step
        alpha_weights_before = [param.clone() for param in model.classifier.alpha_weights]

        # Update parameters
        optimizer.step()

        # Print alpha_weights after optimizer step
        alpha_weights_after = model.classifier.alpha_weights

        # Calculate and print the sum of the differences
        sum_of_differences = sum((after - before).abs().sum().item() for before, after in zip(alpha_weights_before, alpha_weights_after))
        print("Sum of the differences between alpha_weights before and after optimizer step:", sum_of_differences)


class TimeCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.start_time_training = None
        self.end_time_training = None
        self.start_time__training_step = None
        self.end_time_training_step = None
        self.start_time_validation_step = None
        self.end_time_validation_step = None
        self.output_dir = output_dir
        self.is_evaluating = False
        self.data = {"training_times": [], "training_step_times": [], "validation_step_times": []}

    def on_step_begin(self, args, state, control, **kwargs):
        if self.is_evaluating:
            self.start_time_validation_step = time.time()
        else:
            self.start_time_training_step = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.is_evaluating:
            self.end_time_validation_step = time.time()
            self.data["validation_step_times"].append(self.end_time_validation_step - self.start_time_validation_step)
        else:
            self.end_time_training_step = time.time()
            self.data["training_step_times"].append(self.end_time_training_step - self.start_time_training_step)
        self._save_data()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time_training = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.end_time_training = time.time()
        self.data["training_times"].append(self.end_time_training - self.start_time_training)
        self._save_data()
        self.is_evaluating = True

    def on_evaluate(self, args, state, control, **kwargs):
        self.is_evaluating = False

    
    def _save_data(self):
        with open(f"{self.output_dir}/time.json", 'w') as f:
            json.dump(self.data, f)

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        # Get the original DataLoader
        train_dataloader = super().get_train_dataloader()
        # Get the batch size from TrainingArguments
        batch_size = self.args.per_device_train_batch_size * self.args.n_gpu
        # Create a new DataLoader with drop_last=True
        return DataLoader(train_dataloader.dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class RollingMetricsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Rolling totals (sum of values since start of training)
        self._total_loss_scalar = 0.0
        self._total_grad_mean   = 0.0
        self._total_grad_var    = 0.0
        self._total_grad_norm   = 0.0
        self._total_lr          = 0.0

        # Values at last log boundary
        self._logging_loss_scalar   = 0.0
        self._logging_grad_mean     = 0.0
        self._logging_grad_var      = 0.0
        self._logging_grad_norm     = 0.0
        self._logging_lr            = 0.0

        # For calculating how many steps pass between logs
        self._globalstep_last_logged = 0
        
    def training_step(self, model, inputs):
        """
        1) Runs HF's standard forward/backward pass (super().training_step)
        2) Captures raw_grad_mean, raw_grad_var, grad_norm, learning_rate
        3) Accumulates them
        4) Every self.args.logging_steps, logs the *rolling averages*.
        """
        # 1) Let HF do forward + backward + partial logging
        loss = super().training_step(model, inputs)
        loss_val = loss.item()

        # 2) Compute your custom metrics now that param.grad is available
        raw_grad_mean, raw_grad_var = self._compute_raw_grad_stats(model)
        grad_norm = self._compute_grad_norm(model)
        lr = self._get_current_lr()  # typically first param group

        # 3) Accumulate them in total
        self._total_loss_scalar += loss_val
        self._total_grad_mean   += raw_grad_mean
        self._total_grad_var    += raw_grad_var
        self._total_grad_norm   += grad_norm
        self._total_lr          += lr

        # 4) Check if it's time to log (every logging_steps)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            steps_diff = (self.state.global_step + 1) - self._globalstep_last_logged

            # Rolling average since last log
            rolling_loss = ((self._total_loss_scalar - self._logging_loss_scalar) 
                            / steps_diff)
            rolling_grad_mean = ((self._total_grad_mean - self._logging_grad_mean) 
                                 / steps_diff)
            rolling_grad_var  = ((self._total_grad_var - self._logging_grad_var) 
                                 / steps_diff)
            rolling_grad_norm = ((self._total_grad_norm - self._logging_grad_norm) 
                                 / steps_diff)
            rolling_lr = ((self._total_lr - self._logging_lr) 
                          / steps_diff)

            # Log them all as one dictionary
            self.log({
                "rolling_loss": rolling_loss,
                "rolling_raw_grad_mean": rolling_grad_mean,
                "rolling_raw_grad_var": rolling_grad_var,
                "rolling_grad_norm": rolling_grad_norm,
                "rolling_lr": rolling_lr,
                "epoch": float(self.state.epoch),
            })

            # Update references so next logging interval calculates the delta properly
            self._logging_loss_scalar   = self._total_loss_scalar
            self._logging_grad_mean     = self._total_grad_mean
            self._logging_grad_var      = self._total_grad_var
            self._logging_grad_norm     = self._total_grad_norm
            self._logging_lr            = self._total_lr
            self._globalstep_last_logged = self.state.global_step + 1

        return loss

    def _compute_raw_grad_stats(self, model):
        """
        Summation-based raw grad mean + var across all parameters.
        Minimizes CPU copies by doing sums on GPU, then one .item().
        """
        total_sum = 0.0
        total_sq_sum = 0.0
        total_elems = 0

        for p in model.parameters():
            if p.grad is not None:
                g = p.grad
                total_sum += g.sum()
                total_sq_sum += (g * g).sum()
                total_elems += g.numel()

        if total_elems == 0:
            return 0.0, 0.0

        mean_val = (total_sum / total_elems).item()
        var_val  = (total_sq_sum / total_elems - mean_val**2).item()
        return mean_val, var_val

    def _compute_grad_norm(self, model):
        """
        Hugging Face typically logs grad_norm as the L2 norm of all param grads:
          grad_norm = sqrt( sum(grad^2) for all params )
        We'll replicate that logic.
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()  # L2 norm
                total_norm += param_norm**2
        return total_norm**0.5

    def _get_current_lr(self):
        """
        Typically, HF logs learning_rate from the first param group,
        or you can average across all param_groups if desired.
        """
        # We'll just read the first group for simplicity
        if self.optimizer and len(self.optimizer.param_groups) > 0:
            return self.optimizer.param_groups[0]["lr"]
        return 0.0

    def evaluate(
    self,
    eval_dataset: Optional[Dataset] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "eval"
) -> Dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No eval_dataset was provided for evaluation.")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # 1) Run standard HF evaluation loop
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=(True if self.compute_metrics is None else None),
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.log(output.metrics)

        # 2) If you want the raw predictions for calibration:
        #    use `predict(...)` on the same dataset
        pred_output = self.predict(eval_dataset)  # returns predictions + label_ids
        print("DEBUG => pred_output.label_ids shape:", None if pred_output.label_ids is None else pred_output.label_ids.shape)
        
        
        # 3) Put them into `metrics`
        #    (but be careful if "predictions" or "logits" is large -- watch memory!)
        output.metrics["eval_predictions"] = pred_output.predictions
        output.metrics["eval_label_ids"] = pred_output.label_ids

        # 4) Fire the callback in the standard way.  No extra named args.
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics=output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics