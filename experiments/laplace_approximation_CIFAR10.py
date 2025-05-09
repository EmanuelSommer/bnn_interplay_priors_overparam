"""Carry out the Laplace approximation for a small ResNet on CIFAR-10."""

# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jax
import numpy as np
import pandas as pd
import tqdm
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt
import posteriors
import hydra
import time
from optree import tree_map
from omegaconf import DictConfig, OmegaConf
from src.posteriors_methods.torch_models import TinyResNet, TinyResNetConfigTorch

from src.data import ImageLoader
from src.abi.utils import Task
from src.config import DataConfig
from datetime import datetime

from src.abi.utils import (
    pointwise_lppd,
    lppd,
)

@hydra.main(version_base=None, config_path="../src/posteriors_methods/conf_laplace", config_name="config")
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

        seed_start_time = time.time()

        print(f"\n=== Running with seed {seed} ===")
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
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

        class PosteriorsDataset(torch.utils.data.Dataset):
            def __init__(self, loader_fn):
                self.data = []
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

        # Create PyTorch Datasets and DataLoaders that correspond to the JAX iterators
        train_dataset = PosteriorsDataset(train_iter_fn)
        val_dataset = PosteriorsDataset(val_iter_fn)
        test_dataset = PosteriorsDataset(test_iter_fn)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
                
        # Initialize model based on config
        activation = F.relu if cfg.model.activation == "relu" else F.sigmoid
        model_config = TinyResNetConfigTorch(out_dim=cfg.model.out_dim, activation=activation)
        model = TinyResNet(model_config).to(device)
        
        params_init = dict(model.named_parameters())
        model_fn = posteriors.utils.model_to_function(model)

        def cross_entropy_loss(params, batch):
            """Cross entropy loss function."""
            x_tensor, y_tensor = batch
            logits = model_fn(params, x_tensor)
            loss = nn.functional.cross_entropy(logits, y_tensor)
            return loss, loss
        
        def outer_log_lik_laplace(logits, batch):
            """A function that takes the output of `forward` and batch
            then returns the log likelihood of the model output,
            with no auxiliary information."""
            _, labels = batch
            log_likelihood = -nn.functional.cross_entropy(logits, labels)
            return log_likelihood

        def model_func(p, x):
                return torch.func.functional_call(model, p, x)

        def forward_laplace(params, batch):
            """Forward function for Laplace approximation."""
            x_batch, _ = batch
            logits = model_func(params, x_batch)
            return logits, torch.tensor([])
        
        if cfg.map.train_map:

            map_start_time = time.time()
            
            if cfg.map.use_functional:
                map_transform = posteriors.torchopt.build(
                    loss_fn=cross_entropy_loss,
                    optimizer = torchopt.adamw(
                        lr=cfg.map.learning_rate, 
                        weight_decay=cfg.map.weight_decay, 
                        maximize=cfg.map.maximize
                    )
                )

                # Initialize the transform
                map_state = map_transform.init(params_init)

                # Train MAP model
                best_val_loss = float('-inf') if cfg.map.maximize else float('inf')
                patience_counter = 0

                print("Training MAP estimate using functional approach...")
                for epoch in range(cfg.map.n_epochs):
                    # Training epoch
                    for i, batch in enumerate(train_loader):
                        map_state, aux = map_transform.update(map_state, batch, inplace=True)
                        
                        if (i + 1) % cfg.output.log_interval == 0:
                            print(f"Epoch {epoch+1}/{cfg.map.n_epochs}, Batch {i+1}, CE Loss: {aux.item():.4f}")
                    
                    val_losses = []

                    # Validation epoch
                    with torch.no_grad():
                        for i_val, batch_val in enumerate(val_loader):
                            x_val, y_val = batch_val
                            logits = torch.func.functional_call(model, map_state.params, x_val)
                            val_loss = nn.functional.cross_entropy(logits, y_val)
                            val_losses.append(val_loss.item())
                            
                        avg_val_loss = sum(val_losses) / len(val_losses)
                        print(f"Epoch {epoch+1}: Avg Val Loss: {avg_val_loss:.4f}")
                    
                    model.train()
                    
                    # Update best state if current average validation loss is better
                    is_better = avg_val_loss < best_val_loss if not cfg.map.maximize else avg_val_loss > best_val_loss
                    if is_better:
                        best_val_loss = avg_val_loss
                        best_state = map_state
                        best_epoch = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        print(f"Patience counter: {patience_counter}/{cfg.map.patience}")
                        
                    if patience_counter >= cfg.map.patience:
                        print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
                        map_state = best_state  # Roll back to best state
                        break
                    
                optimized_params = map_state.params
            
            else:
                # Traditional non-functional PyTorch training
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=cfg.map.learning_rate,
                    weight_decay=cfg.map.weight_decay
                )
                
                # Train MAP model
                best_val_loss = float('inf')
                patience_counter = 0
                best_state_dict = None
                
                print("Training MAP estimate using traditional PyTorch...")
                for epoch in range(cfg.map.n_epochs):
                    # Training epoch
                    model.train()
                    for i, batch in enumerate(train_loader):
                        x_train, y_train = batch
                        optimizer.zero_grad()
                        logits = model(x_train)
                        loss = nn.functional.cross_entropy(logits, y_train)
                        loss.backward()
                        optimizer.step()
                        
                        if (i + 1) % cfg.output.log_interval == 0:
                            print(f"Epoch {epoch+1}/{cfg.map.n_epochs}, Batch {i+1}, CE Loss: {loss.item():.4f}")
                    
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for i_val, batch_val in enumerate(val_loader):
                            x_val, y_val = batch_val
                            logits = model(x_val)
                            val_loss = nn.functional.cross_entropy(logits, y_val)
                            val_losses.append(val_loss.item())
                        
                        avg_val_loss = sum(val_losses) / len(val_losses)
                        print(f"Epoch {epoch+1}: Avg Val Loss: {avg_val_loss:.4f}")
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                        best_epoch = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        print(f"Patience counter: {patience_counter}/{cfg.map.patience}")
                    
                    if patience_counter >= cfg.map.patience:
                        print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
                        model.load_state_dict(best_state_dict)  # Roll back to best state
                        break
                
                optimized_params = dict(model.named_parameters())

            map_training_time = time.time() - map_start_time

        else: # if warmstart is available

            def npz_to_torch_state_dict(npz_file_path):
                # Load parameters from the npz file
                params = np.load(npz_file_path)
                
                # Create a torch model state_dict based on the provided JAX mapping
                # Transpose a convolution kernel from [kH, kW, inC, outC] to [outC, inC, kH, kW]
                def transpose_conv(kernel):
                    return kernel.transpose(3, 2, 0, 1)
                
                # Transpose a dense kernel from [in_features, out_features] to [out_features, in_features]
                def transpose_dense(kernel):
                    return kernel.T
                
                state_dict = {
                    'core.conv1.weight': torch.tensor(transpose_conv(params['core.Conv_0.kernel.npy'])),
                    'core.block.conv1.weight': torch.tensor(transpose_conv(params['core.BasicBlock_0.Conv_0.kernel.npy'])),
                    'core.block.conv2.weight': torch.tensor(transpose_conv(params['core.BasicBlock_0.Conv_1.kernel.npy'])),
                    'core.fc.weight': torch.tensor(transpose_dense(params['core.Dense_0.kernel.npy'])),
                    'core.fc.bias': torch.tensor(params['core.Dense_0.bias.npy']),
                }
                
                return state_dict
        
            # load warmstart from disk
            optimized_params = npz_to_torch_state_dict(cfg.map.warmstart_path)

        
        X_test_batch, y_test_batch = next(iter(test_loader))

        def model_func(p, x):
                return torch.func.functional_call(model, p, x)

        mean_logits = model_func(optimized_params, X_test_batch)
        acc_mean = np.mean(torch.argmax(mean_logits, dim=1).cpu().numpy() == y_test_batch.cpu().numpy())
        print(f"Mean accuracy on test batch: {acc_mean:.4f}")

        laplace_start_time = time.time()

        # Train Laplace approximation using the MAP estimate
        laplace_transform = posteriors.laplace.diag_ggn.build(
            forward = forward_laplace,
            outer_log_likelihood=outer_log_lik_laplace,
            init_prec_diag=cfg.laplace.init_prec_diag,

        )

        # Extract last layer parameters (fully connected layer) from optimized params
        last_layer_params = {}
        for name, param in optimized_params.items():
            if 'fc' in name:  # Only keep parameters from the fully connected layer
                last_layer_params[name] = param

        last_layer_state = laplace_transform.init(last_layer_params)  # Use only last layer params

        for i, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc="Laplace training"):
            last_layer_state, _ = laplace_transform.update(last_layer_state, batch, inplace=True)

        param_samples = posteriors.laplace.diag_ggn.sample(
            last_layer_state,
            sample_shape=(cfg.laplace.n_samples, )
        )

        laplace_time = time.time() - laplace_start_time

        eval_start_time = time.time()
        
        def model_features_func(params, x):
            # Extract features by manually calling the layers except for last layer with the provided parameters
            x = F.conv2d(x, params['core.conv1.weight'], None, stride=1, padding=1)
            x = model.core.activation(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            
            identity = x
            x = F.conv2d(x, params['core.block.conv1.weight'], None, stride=1, padding=1)
            x = model.core.activation(x)
            x = F.conv2d(x, params['core.block.conv2.weight'], None, stride=1, padding=1)
            
            if model.core.block.downsample is not None:
                identity = F.conv2d(identity, params['core.block.downsample.weight'],
                                    None,
                                    stride=model.core.block.downsample.stride)
            
            x += identity
            x = model.core.activation(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            
            return x

        def predict_with_samples(model, X_test_tensor, param_samples):
            """Make predictions with multiple param samples on a single batch."""

            with torch.no_grad():
                features = model_features_func(optimized_params, X_test_tensor)

            def predict_with_feature_samples(features, weight, bias):
                """Use sampled parameters on pre-computed features."""
                return torch.nn.functional.linear(features, weight, bias)

            with torch.no_grad():
                # If params_sample has a sample dimension, use vmap
                if tree_map(lambda x: x.ndim, param_samples)['core.fc.weight'] == 3:  # Multiple samples
                    logits = torch.vmap(lambda w, b: predict_with_feature_samples(features, w, b))(
                        param_samples['core.fc.weight'], param_samples['core.fc.bias'])
                else:  # Single sample
                    logits = predict_with_feature_samples(features, param_samples['core.fc.weight'], param_samples['core.fc.bias'])
                
                return logits
            

        def evaluate_in_batches(model, loader, param_samples):
            all_predictions = []
            all_true_classes = []
            
            for batch in tqdm.tqdm(loader):
                x_batch, y_batch = batch
                predictions = predict_with_samples(model, x_batch, param_samples)
                all_predictions.append(predictions)
                all_true_classes.append(y_batch)
            
            if predictions.ndim == 2:
                all_predictions = torch.cat(all_predictions, dim=0)
            else:
                all_predictions = torch.cat(all_predictions, dim=1)
            all_true_classes = torch.cat(all_true_classes)
            
            return all_predictions, all_true_classes


        all_preds_mean, all_true_classes_mean = evaluate_in_batches(model, test_loader, optimized_params)
        all_preds_classes_mean = torch.argmax(all_preds_mean, dim=1)
        accuracy_mean = np.mean(all_true_classes_mean.cpu().numpy() == all_preds_classes_mean.cpu().numpy())
        all_preds_samples, all_true_classes = evaluate_in_batches(model, test_loader, param_samples)
        all_preds_samples_mean = torch.mean(all_preds_samples, dim=0)
        all_preds_classes = torch.argmax(all_preds_samples_mean, dim=1)
        accuracy = np.mean(all_true_classes.cpu().numpy() == all_preds_classes.cpu().numpy())

        lppd_res = lppd(
                    pointwise_lppd(np.array(all_preds_samples.cpu())[np.newaxis, ...], np.array(all_true_classes.cpu()), Task.CLASSIFICATION)
                )
        
        eval_time = time.time() - eval_start_time
        seed_total_time = time.time() - seed_start_time

        all_results.append({
            'seed': seed,
            'accuracy': accuracy,
            'accuracy_mean': accuracy_mean,
            'lppd': lppd_res,
            'map_training_time': map_training_time if cfg.map.train_map else 0,
            'laplace_time': laplace_time,
            'eval_time': eval_time,
            'seed_time': seed_total_time,
        })
        print(f"\n=== Results for seed {seed} ===")
        print(f"Accuracy Sampling: {accuracy:.6f}")
        print(f"Accuracy Mean: {accuracy_mean:.6f}")
        print(f"LPPD: {lppd_res:.6f}")
        print(f"Laplace Approximation Time: {laplace_time:.2f} seconds")
        print(f"Evaluation Time: {eval_time:.2f} seconds")
        print(f"Total Time for Seed {seed}: {seed_total_time:.2f} seconds")
        print(f"=== End of results for seed {seed} ===")

    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(cfg.output.save_dir, f'seed_results_{current_time}.csv'), index=False)
    print(f"\nResults saved to {os.path.join(cfg.output.save_dir, f'seed_results_{current_time}.csv')}")

if __name__ == "__main__":
    main()