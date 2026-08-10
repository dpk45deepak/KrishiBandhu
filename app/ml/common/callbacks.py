"""
Callback system for training lifecycle management.
Follows the Observer pattern for training events.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from loguru import logger
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from .utils import ensure_directory, save_json, Timer, get_memory_usage
from .exceptions import TrainingError


class Callback(ABC):
    """
    Abstract base class for training callbacks.
    """
    
    def on_train_begin(self, **kwargs) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, **kwargs) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, **kwargs) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_early_stopping(self, **kwargs) -> None:
        """Called when early stopping is triggered."""
        pass


class CallbackManager:
    """
    Manages and executes callbacks during training.
    """
    
    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: Callback) -> None:
        """Add a callback to the manager."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callback) -> None:
        """Remove a callback from the manager."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def trigger(self, event: str, **kwargs) -> None:
        """
        Trigger a callback event.
        
        Args:
            event: Event name (on_train_begin, on_epoch_end, etc.)
            **kwargs: Event-specific arguments
        """
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method and callable(method):
                try:
                    method(**kwargs)
                except Exception as e:
                    logger.error(f"Callback {callback.__class__.__name__} failed: {e}")
                    # Don't fail training due to callback errors
    
    def on_train_begin(self, **kwargs) -> None:
        self.trigger('on_train_begin', **kwargs)
    
    def on_train_end(self, **kwargs) -> None:
        self.trigger('on_train_end', **kwargs)
    
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        self.trigger('on_epoch_begin', epoch=epoch, **kwargs)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        self.trigger('on_epoch_end', epoch=epoch, metrics=metrics, **kwargs)
    
    def on_batch_begin(self, batch: int, **kwargs) -> None:
        self.trigger('on_batch_begin', batch=batch, **kwargs)
    
    def on_batch_end(self, batch: int, metrics: Dict[str, float], **kwargs) -> None:
        self.trigger('on_batch_end', batch=batch, metrics=metrics, **kwargs)
    
    def on_early_stopping(self, **kwargs) -> None:
        self.trigger('on_early_stopping', **kwargs)


class TrainingLogger(Callback):
    """
    Callback for logging training progress.
    """
    
    def __init__(self, log_every: int = 1, log_file: Optional[Path] = None):
        self.log_every = log_every
        self.log_file = log_file
        self.history: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.best_metric = float('inf')
        self.best_epoch = 0
    
    def on_train_begin(self, **kwargs) -> None:
        """Initialize logging at the start of training."""
        self.start_time = datetime.now()
        self.history = []
        logger.info("=" * 80)
        logger.info("TRAINING STARTED")
        logger.info("=" * 80)
        
        if self.log_file:
            self.log_file = Path(self.log_file)
            ensure_directory(self.log_file.parent)
            with open(self.log_file, 'w') as f:
                f.write("epoch,train_loss,val_loss,train_metric,val_metric\n")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Log metrics at the end of each epoch."""
        if epoch % self.log_every == 0:
            # Format metrics for display
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            
            # Track best metric
            val_metric = metrics.get('val_metric', metrics.get('val_loss', float('inf')))
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_epoch = epoch
            
            # Log with rich formatting
            logger.info(
                f"Epoch {epoch:3d} | {metric_str} | "
                f"Best: {self.best_metric:.4f} (epoch {self.best_epoch})"
            )
            
            # Save history
            self.history.append({'epoch': epoch, **metrics})
            
            # Write to CSV if log_file is set
            if self.log_file:
                row = [str(epoch)]
                for key in ['train_loss', 'val_loss', 'train_metric', 'val_metric']:
                    row.append(str(metrics.get(key, '')))
                with open(self.log_file, 'a') as f:
                    f.write(','.join(row) + '\n')
    
    def on_train_end(self, **kwargs) -> None:
        """Log final training summary."""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info("=" * 80)
            logger.info("TRAINING COMPLETED")
            logger.info(f"Total time: {elapsed:.2f} seconds")
            logger.info(f"Best metric: {self.best_metric:.4f} at epoch {self.best_epoch}")
            logger.info("=" * 80)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.history


class EarlyStopping(Callback):
    """
    Callback for early stopping.
    Monitors a metric and stops training when it stops improving.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Check for early stopping."""
        if self.monitor not in metrics:
            return
        
        current = metrics[self.monitor]
        
        # Initialize best score
        if self.best_score is None:
            self.best_score = current
            return
        
        # Check if improvement
        is_better = False
        if self.mode == 'min':
            if current < self.best_score - self.min_delta:
                is_better = True
        else:  # max
            if current > self.best_score + self.min_delta:
                is_better = True
        
        if is_better:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                logger.warning(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {self.monitor}: {self.best_score:.4f}"
                )
    
    def on_early_stopping(self, **kwargs) -> None:
        """Called when early stopping is triggered."""
        pass


class ModelCheckpoint(Callback):
    """
    Callback for saving model checkpoints.
    """
    
    def __init__(
        self,
        save_path: Path,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True,
        max_keep: int = 5
    ):
        self.save_path = Path(save_path)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.max_keep = max_keep
        
        self.best_score = None
        self.best_epoch = 0
        self.saved_checkpoints: List[Path] = []
        
        ensure_directory(self.save_path)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Save checkpoint at the end of each epoch."""
        model = kwargs.get('model')
        if model is None:
            return
        
        if self.monitor not in metrics:
            return
        
        current = metrics[self.monitor]
        
        # Determine if we should save
        should_save = not self.save_best_only
        
        if self.best_score is None:
            self.best_score = current
            should_save = True
        else:
            is_better = False
            if self.mode == 'min':
                is_better = current < self.best_score
            else:
                is_better = current > self.best_score
            
            if is_better:
                self.best_score = current
                self.best_epoch = epoch
                should_save = True
        
        if should_save:
            checkpoint_path = self.save_path / f"checkpoint_epoch_{epoch:04d}.pkl"
            
            try:
                # Save the model
                if hasattr(model, 'save'):
                    model.save(checkpoint_path)
                else:
                    # Use pickle as fallback
                    from .utils import save_pickle
                    save_pickle(model.get_model(), checkpoint_path)
                
                self.saved_checkpoints.append(checkpoint_path)
                
                # Keep only max_keep checkpoints
                if len(self.saved_checkpoints) > self.max_keep:
                    old_checkpoint = self.saved_checkpoints.pop(0)
                    if old_checkpoint.exists():
                        old_checkpoint.unlink()
                
                if self.verbose:
                    logger.info(
                        f"Saved checkpoint: {checkpoint_path.name} "
                        f"({self.monitor}: {current:.4f})"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the path to the best checkpoint."""
        if not self.saved_checkpoints:
            return None
        
        # The best checkpoint should be the last one if saving best only
        return self.saved_checkpoints[-1]
    
    def get_checkpoint_by_epoch(self, epoch: int) -> Optional[Path]:
        """Get checkpoint for a specific epoch."""
        for path in self.saved_checkpoints:
            if f"epoch_{epoch:04d}" in str(path):
                return path
        return None


class ProgressBar(Callback):
    """
    Callback for displaying training progress with Rich.
    """
    
    def __init__(self, total_epochs: int, description: str = "Training"):
        self.total_epochs = total_epochs
        self.description = description
        self.progress = None
        self.task_id = None
    
    def on_train_begin(self, **kwargs) -> None:
        """Initialize the progress bar."""
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("• {task.fields[metrics]}"),
            console=None  # Use default console
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            self.description,
            total=self.total_epochs,
            metrics="Starting..."
        )
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Update the progress bar."""
        if self.progress and self.task_id:
            # Format metrics for display
            metric_str = ", ".join([
                f"{k}={v:.4f}" for k, v in list(metrics.items())[:3]
            ])
            self.progress.update(
                self.task_id,
                advance=1,
                metrics=metric_str
            )
    
    def on_train_end(self, **kwargs) -> None:
        """Close the progress bar."""
        if self.progress:
            self.progress.stop()
    
    def on_early_stopping(self, **kwargs) -> None:
        """Stop the progress bar early."""
        if self.progress:
            self.progress.stop()


class MetricsTracker(Callback):
    """
    Callback for tracking and aggregating metrics.
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.best_metrics: Dict[str, float] = {}
        self.current_epoch_metrics: Dict[str, float] = {}
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Track metrics for each epoch."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
            
            # Track best values
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            elif (key.startswith('val_') or key.endswith('_loss')):
                # For validation metrics, we typically want the best
                if value < self.best_metrics[key]:
                    self.best_metrics[key] = value
            else:
                # For other metrics, track separately
                pass
        
        self.current_epoch_metrics = metrics
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history for all metrics."""
        return self.history
    
    def get_best(self, metric: str) -> Optional[float]:
        """Get the best value for a metric."""
        return self.best_metrics.get(metric)
    
    def get_latest(self, metric: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if metric in self.history and self.history[metric]:
            return self.history[metric][-1]
        return None


class LearningRateScheduler(Callback):
    """
    Callback for learning rate scheduling.
    """
    
    def __init__(
        self,
        scheduler: Callable[[int, float], float],
        initial_lr: float = 0.01,
        verbose: bool = True
    ):
        self.scheduler = scheduler
        self.current_lr = initial_lr
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Update learning rate at the beginning of each epoch."""
        model = kwargs.get('model')
        if model is None:
            return
        
        new_lr = self.scheduler(epoch, self.current_lr)
        
        if new_lr != self.current_lr:
            self.current_lr = new_lr
            # Update the model's learning rate
            if hasattr(model, 'set_learning_rate'):
                model.set_learning_rate(new_lr)
            elif hasattr(model, 'model') and hasattr(model.model, 'set_params'):
                # Try to set through underlying model
                try:
                    if 'learning_rate' in model.model.get_params():
                        model.model.set_params(learning_rate=new_lr)
                except:
                    pass
            
            if self.verbose:
                logger.info(f"Learning rate updated to {new_lr:.6f}")


class MemoryMonitor(Callback):
    """
    Callback for monitoring memory usage during training.
    """
    
    def __init__(self, alert_threshold_mb: float = 1024 * 4):  # 4GB
        self.alert_threshold_mb = alert_threshold_mb
        self.peak_memory_mb = 0
        self.alert_triggered = False
    
    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """Check memory usage after each epoch."""
        memory_info = get_memory_usage()
        current_mb = memory_info['rss_mb']
        
        if current_mb > self.peak_memory_mb:
            self.peak_memory_mb = current_mb
        
        if current_mb > self.alert_threshold_mb and not self.alert_triggered:
            logger.warning(
                f"Memory usage exceeded threshold: {current_mb:.1f}MB "
                f"(threshold: {self.alert_threshold_mb:.1f}MB)"
            )
            self.alert_triggered = True
        
        if epoch % 10 == 0:
            logger.debug(f"Memory usage: {current_mb:.1f}MB")
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_memory_mb