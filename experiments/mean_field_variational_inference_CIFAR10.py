"""Carry out MFVI for a small ResNet on CIFAR-10."""

# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tqdm
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt
import posteriors
from optree import tree_map
import hydra
import time
from omegaconf import DictConfig, OmegaConf
from datetime import datetime


from src.data import ImageLoader
from src.abi.utils import lppd, pointwise_lppd
from src.abi.utils import Task

from src.config import DataConfig
from src.posteriors_methods.torch_models import TinyResNet, TinyResNetConfigTorch
    

def predict_with_samples(model, x_data, param_samples, device):
    """Make predictions with multiple param samples on a single batch."""

    with torch.no_grad():
        # if leaves in param_samples are one-dimensional, calculate logits without vmap
        if tree_map(lambda x: x.ndim, param_samples)['core.conv1.weight'] == 4:
            logits = torch.func.functional_call(model, param_samples, x_data)
        else:
            def model_func(p, x):
                return torch.func.functional_call(model, p, x)
           
            logits = torch.vmap(model_func, in_dims=(0, None))(param_samples, x_data)
        
        return logits

def evaluate_in_batches(model, loader, param_samples, device):
    all_predictions = []
    all_true_classes = []
    
    for batch in tqdm.tqdm(loader):
        x_batch, y_batch = batch
        predictions = predict_with_samples(model, x_batch, param_samples, device)
        all_predictions.append(predictions)
        all_true_classes.append(y_batch)
    
    if predictions.ndim == 2:
        all_predictions = torch.cat(all_predictions, dim=0)
    else:
        all_predictions = torch.cat(all_predictions, dim=1)
    all_true_classes = torch.cat(all_true_classes)
    
    return all_predictions, all_true_classes


@hydra.main(version_base=None, config_path="configs", config_name="mfvi_cifar10_benchmark")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Create results directory
    os.makedirs(cfg.output.save_dir, exist_ok=True)

    # Check for GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Optional: Print GPU info
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Store results for different seeds
    all_results = []
    
    # Iterate over seeds
    for seed in cfg.seeds:
        print(f"\n=== Running with seed {seed} ===")

        start_time_seed = time.time()
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        data_load_start = time.time()
        
        # Setup data loading
        config_data = DataConfig(
            path=cfg.data.path,
            source=cfg.data.source,
            data_type=cfg.data.data_type,
            task=cfg.data.task,
            normalize=cfg.data.normalize,
            train_split=cfg.data.train_split,
            valid_split=cfg.data.valid_split,
            test_split=cfg.data.test_split,
            target_column=None,
            features=None,
            datapoint_limit=None,
        )
        
        loader = ImageLoader(
            config_data,
            rng=jax.random.key(seed)
        )
        loader.load_data()
        
        # Create a dataset that stores individual data points
        class PosteriorsDataset(torch.utils.data.Dataset):
            def __init__(self, loader_fn):
                self.data = []
                # Collect all individual data points
                for batch in loader_fn():
                    x_batch, y_batch = batch
                    x_tensor = torch.tensor(np.array(x_batch), dtype=torch.float32)
                    y_tensor = torch.tensor(np.array(y_batch), dtype=torch.int64)
                    for i in range(len(x_tensor)):
                        self.data.append((x_tensor[i], y_tensor[i]))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                x, y = self.data[idx]
                x = x.to(device)
                y = y.to(device)
                return x, y

        # Create JAX iterators
        train_iter_fn = partial(loader.iter_image, split="train", batch_size=loader.len_train)  # Get all training data at once
        val_iter_fn = partial(loader.iter_image, split="valid", batch_size=loader.len_valid)    # Get all validation data at once
        test_iter_fn = partial(loader.iter_image, split="test", batch_size=loader.len_test)     # Get all test data at once

        # Create PyTorch Datasets and DataLoaders
        train_dataset = PosteriorsDataset(train_iter_fn)
        val_dataset = PosteriorsDataset(val_iter_fn)
        test_dataset = PosteriorsDataset(test_iter_fn)

        # Let DataLoader create new batches on each iteration
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
        
        num_data = loader.len_train

        data_load_time = time.time() - data_load_start

        # Initialize model based on config
        activation = F.relu if cfg.model.activation == "relu" else F.sigmoid
        model_config = TinyResNetConfigTorch(out_dim=cfg.model.out_dim, activation=activation)
        model = TinyResNet(model_config).to(device)
        
        params_init = dict(model.named_parameters())
        model_fn = posteriors.utils.model_to_function(model)
        
        # Define log posterior
        def log_posterior(params, batch):
            x_tensor, y_tensor = batch
            # x_tensor = torch.tensor(np.array(batch_x), dtype=torch.float32).to(device)
            # y_tensor = torch.tensor(np.array(batch_y), dtype=torch.int64).to(device)
            
            logits = model_fn(params, x_tensor)
            log_probs = -nn.functional.cross_entropy(logits, y_tensor)
            log_prior = posteriors.diag_normal_log_prob(params) / num_data
            log_post = log_probs + log_prior
            return log_post, (log_probs, log_prior)
        
        training_start = time.time()
            
        # Run VI training
        vi_transform = posteriors.vi.diag.build(
            log_posterior,
            optimizer=torchopt.adam(
                lr=cfg.training.learning_rate, 
                weight_decay=cfg.training.weight_decay, 
                maximize=cfg.training.maximize
            ),
            init_log_sds=cfg.training.init_log_sds,
            temperature=cfg.training.temperature_scale / num_data
        )
        
        vi_state = vi_transform.init(params_init)
        nelbos = []
        
        # Training loop
        for epoch in range(cfg.training.n_epochs):
            for i, batch in enumerate(train_loader):
                vi_state, aux = vi_transform.update(vi_state, batch, inplace=True)
                current_nelbo = vi_state.nelbo.item()
                nelbos.append(current_nelbo)
                
                if (i + 1) % cfg.output.log_interval == 0:
                    print(f"Seed {seed}, Epoch {epoch+1}/{cfg.training.n_epochs}, "
                          f"Iteration {i+1}, NELBO {current_nelbo:.4f}")
                    print(f"Log Lik: {aux[0].item():.4f}, Log Prior: {aux[1].item():.4f}, "
                          f"Log Post: {aux[0].item() + aux[1].item():.4f}")
                    
                if i >= cfg.training.max_iter:
                    break
            if i >= cfg.training.max_iter:
                break

            for j, batch in enumerate(val_loader):
                val_preds = model_fn(vi_state.params, batch[0])
                val_loss = -nn.functional.cross_entropy(val_preds, batch[1])
                val_acc = (val_preds.argmax(dim=1) == batch[1]).float().mean()
                print(f"========= Validation Epoch {epoch} ==========")
                print(f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")
                if j == 0:
                    break

        training_time = time.time() - training_start

        evaluation_start = time.time()
        
        # Get parameter mean and sample for prediction
        param_mean = vi_state.params
        param_samples = posteriors.vi.diag.sample(vi_state, (cfg.evaluation.n_samples,))

        # Get predictions using mean parameter sample
        all_preds, all_true_classes = evaluate_in_batches(model, test_loader, param_mean, device)
        # Move to cpu
        all_preds = all_preds.cpu()
        all_true_classes = all_true_classes.cpu()
        all_preds_classes = torch.argmax(all_preds, dim=1)
        mean_accuracy = np.mean(all_true_classes.numpy() == all_preds_classes.numpy())
        
        # Get predictions using all parameter samples
        all_preds_samples, all_true_classes = evaluate_in_batches(model, test_loader, param_samples, device)
        # Move to CPU
        all_preds_samples = all_preds_samples.cpu()
        all_true_classes = all_true_classes.cpu()
        all_preds_samples_mean = torch.mean(all_preds_samples, dim=0)
        all_preds_samples_classes = torch.argmax(all_preds_samples_mean, dim=1)
        sample_accuracy = np.mean(all_true_classes.numpy() == all_preds_samples_classes.numpy())

        # Calculate LPPD
        lppd_res = lppd(
                    pointwise_lppd(np.array(all_preds_samples)[np.newaxis, ...], np.array(all_true_classes), Task.CLASSIFICATION)
                )
        
        evaluation_time = time.time() - evaluation_start

        # Calculate total time for this seed
        seed_total_time = time.time() - start_time_seed

        # Store results
        all_results.append({
            'seed': seed,
            'mean_accuracy': mean_accuracy,
            'sample_accuracy': sample_accuracy,
            'final_nelbo': nelbos[-1] if nelbos else None,
            'lppd': lppd_res,
            'data_load_time': data_load_time,
            'training_time': training_time,
            'evaluation_time': evaluation_time,
            'total_seed_time': seed_total_time
        })
        print(f"\n=== Results for seed {seed} ===")
        print(f"Mean Accuracy: {mean_accuracy:.6f}")
        print(f"Sample Accuracy: {sample_accuracy:.6f}")
        print(f"Final NELBO: {nelbos[-1] if nelbos else None}")
        print(f"LPPD: {lppd_res}")
        print(f"Data Load Time: {data_load_time:.2f}s")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Evaluation Time: {evaluation_time:.2f}s")
        print(f"Total Time for Seed {seed}: {seed_total_time:.2f}s")
        print(f"=== End of results for seed {seed} ===")

    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(cfg.output.save_dir, f'seed_results_{current_time}.csv'), index=False)
    print(f"\nResults saved to {os.path.join(cfg.output.save_dir, 'seed_results.csv')}")

if __name__ == "__main__":
    main()


# %%
