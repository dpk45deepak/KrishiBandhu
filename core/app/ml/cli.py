"""
CLI module for AgriMind ML Framework.
Provides command-line interface for training, prediction, evaluation, and more.
"""

import typer
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import pandas as pd

from .common.datasets import DatasetConfig, DatasetLoader
from .common.trainer import ModelTrainer, TrainingConfig
from .common.registry import ModelRegistry
from .common.predictor import ModelPredictor
from .common.persistence import ModelPersistence
from .evaluation.evaluator import ModelEvaluator
from .evaluation.reports import ReportGenerator
from .tuning.tuner import TunerFactory
from .explainability.explain import ModelExplainer

app = typer.Typer(
    name="agrimind-ml",
    help="AgriMind AI Machine Learning Framework CLI",
    add_completion=False
)
console = Console()

# Initialize registry
REGISTRY_PATH = Path("models/registry")
registry = ModelRegistry(REGISTRY_PATH)


@app.command()
def train(
    model_type: str = typer.Argument(..., help="Model type: classification or regression"),
    model_name: str = typer.Argument(..., help="Model name (e.g., random_forest, xgboost)"),
    data_path: Path = typer.Option(..., "--data", "-d", help="Path to training data"),
    target_column: str = typer.Option(..., "--target", "-t", help="Target column name"),
    test_size: float = typer.Option(0.2, "--test-size", help="Test set size"),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs"),
    learning_rate: float = typer.Option(0.01, "--lr", help="Learning rate"),
    validation_split: float = typer.Option(0.1, "--val-split", help="Validation split ratio"),
    save_path: Optional[Path] = typer.Option(None, "--save", "-s", help="Path to save model"),
    params_file: Optional[Path] = typer.Option(None, "--params", help="JSON file with model parameters"),
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Verbose output")
):
    """
    Train a machine learning model.
    """
    console.print(Panel.fit("🤖 AgriMind ML Training", style="bold cyan"))
    
    try:
        # Load parameters from file if provided
        params = {}
        if params_file and params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
        
        # Import model class
        model_class = _get_model_class(model_type, model_name)
        
        # Create model instance with parameters
        model = model_class(**params)
        
        # Load dataset
        dataset_config = DatasetConfig(
            file_path=data_path,
            target_column=target_column,
            test_size=test_size
        )
        dataset_loader = DatasetLoader(dataset_config)
        dataset_loader.load().validate().preprocess()
        
        # Training config
        training_config = TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            validation_split=validation_split,
            verbose=verbose,
            use_progress_bar=True
        )
        
        # Train model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training model...", total=None)
            trainer = ModelTrainer(model, training_config)
            trainer.from_dataset(model, dataset_loader, training_config)
            progress.update(task, completed=True)
        
        # Save model
        if save_path is None:
            save_path = Path("models") / model_type / f"{model_name}.joblib"
        
        ModelPersistence.save_model(model, save_path)
        
        # Register model
        entry = registry.register_model(
            model,
            save_path,
            tags=[model_type, model_name],
            description=f"{model_name} trained on {data_path.name}",
            training_dataset="1.0"
        )
        
        # Display results
        console.print("\n[green]✓ Training completed successfully![/green]")
        console.print(f"[bold]Model ID:[/bold] {entry.model_id}")
        console.print(f"[bold]Model Version:[/bold] {entry.version}")
        console.print(f"[bold]Saved to:[/bold] {save_path}")
        
        if model._metadata:
            table = Table(title="Training Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            for metric, value in model._metadata.metrics.items():
                if metric != 'confusion_matrix':
                    table.add_row(str(metric), f"{value:.4f}")
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def predict(
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Model ID from registry"),
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to model file"),
    data_path: Path = typer.Option(..., "--data", "-d", help="Path to data for prediction"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save predictions"),
    batch_size: int = typer.Option(1000, "--batch-size", help="Batch size for large datasets"),
    probabilities: bool = typer.Option(False, "--probabilities", "-p", help="Output probabilities")
):
    """
    Make predictions using a trained model.
    """
    console.print(Panel.fit("🔮 AgriMind ML Predictions", style="bold cyan"))
    
    try:
        # Load model
        predictor = ModelPredictor()
        if model_id:
            # Load from registry
            model, entry = registry.get_model(model_id=model_id)
            predictor.model = model
            predictor._is_loaded = True
        elif model_path:
            predictor.load_model(model_path)
        else:
            raise ValueError("Either model-id or model-path must be provided")
        
        # Load data
        if data_path.suffix == '.csv':
            X = pd.read_csv(data_path)
        elif data_path.suffix in ['.xlsx', '.xls']:
            X = pd.read_excel(data_path)
        elif data_path.suffix == '.parquet':
            X = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Make predictions
        console.print(f"[yellow]Making predictions on {len(X)} samples...[/yellow]")
        result = predictor.predict_batch(X, batch_size=batch_size, return_proba=probabilities)
        
        # Prepare output
        predictions_df = pd.DataFrame({
            'prediction': result['predictions']
        })
        
        if probabilities and 'probabilities' in result:
            prob_df = pd.DataFrame(
                result['probabilities'],
                columns=[f'class_{i}' for i in range(len(result['probabilities'][0]))]
            )
            predictions_df = pd.concat([predictions_df, prob_df], axis=1)
        
        # Save or display
        if output_path:
            predictions_df.to_csv(output_path, index=False)
            console.print(f"[green]✓ Predictions saved to {output_path}[/green]")
        else:
            console.print("\n[bold]Predictions:[/bold]")
            console.print(predictions_df.head(10).to_string())
            if len(predictions_df) > 10:
                console.print(f"... and {len(predictions_df) - 10} more rows")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Model ID from registry"),
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to model file"),
    test_data: Path = typer.Option(..., "--test", "-t", help="Path to test data"),
    target_column: str = typer.Option(..., "--target", help="Target column name"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for reports")
):
    """
    Evaluate a trained model.
    """
    console.print(Panel.fit("📊 AgriMind ML Evaluation", style="bold cyan"))
    
    try:
        # Load model
        if model_id:
            model, entry = registry.get_model(model_id=model_id)
        elif model_path:
            model = ModelPersistence.load_model(model_path)
        else:
            raise ValueError("Either model-id or model-path must be provided")
        
        # Load test data
        if test_data.suffix == '.csv':
            data = pd.read_csv(test_data)
        elif test_data.suffix in ['.xlsx', '.xls']:
            data = pd.read_excel(test_data)
        else:
            raise ValueError(f"Unsupported file format: {test_data.suffix}")
        
        # Prepare features and target
        X_test = data.drop(columns=[target_column])
        y_test = data[target_column]
        
        # Evaluate
        evaluator = ModelEvaluator(model)
        results = evaluator.evaluate(X_test, y_test)
        
        # Generate report
        if output_dir:
            report_gen = ReportGenerator(output_dir)
            report_gen.generate_report(model, results, X_test, y_test)
            console.print(f"[green]✓ Report saved to {output_dir}[/green]")
        
        # Display results
        table = Table(title="Evaluation Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        metrics = results.get('metrics', {})
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                table.add_row(str(metric), f"{value:.4f}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    model_ids: List[str] = typer.Option(..., "--model-id", "-m", help="Model IDs to compare"),
    test_data: Path = typer.Option(..., "--test", "-t", help="Path to test data"),
    target_column: str = typer.Option(..., "--target", help="Target column name"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for comparison report")
):
    """
    Compare multiple models.
    """
    console.print(Panel.fit("⚖️ AgriMind ML Model Comparison", style="bold cyan"))
    
    try:
        from .evaluation.comparison import ModelComparator
        
        # Load models
        models = []
        for mid in model_ids:
            model, entry = registry.get_model(model_id=mid)
            models.append((model, entry.model_name))
        
        # Load test data
        if test_data.suffix == '.csv':
            data = pd.read_csv(test_data)
        else:
            raise ValueError(f"Unsupported file format: {test_data.suffix}")
        
        X_test = data.drop(columns=[target_column])
        y_test = data[target_column]
        
        # Compare models
        comparator = ModelComparator()
        comparison_results = comparator.compare_models(models, X_test, y_test)
        
        # Display results
        table = Table(title="Model Comparison")
        table.add_column("Metric", style="cyan")
        
        for model_name in comparison_results['model_names']:
            table.add_column(model_name, style="green")
        
        for metric, values in comparison_results['metrics'].items():
            row = [str(metric)]
            for val in values:
                row.append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
            table.add_row(*row)
        
        console.print(table)
        
        # Save results
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            console.print(f"[green]✓ Comparison saved to {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def tune(
    model_type: str = typer.Argument(..., help="Model type: classification or regression"),
    model_name: str = typer.Argument(..., help="Model name"),
    data_path: Path = typer.Option(..., "--data", "-d", help="Path to training data"),
    target_column: str = typer.Option(..., "--target", "-t", help="Target column name"),
    tuner_type: str = typer.Option('optuna', "--tuner", help="Tuner type: grid, random, optuna, bayesian"),
    n_trials: int = typer.Option(100, "--trials", help="Number of tuning trials"),
    metric: str = typer.Option('accuracy', "--metric", help="Metric to optimize"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for results")
):
    """
    Perform hyperparameter tuning.
    """
    console.print(Panel.fit("🔧 AgriMind ML Hyperparameter Tuning", style="bold cyan"))
    
    try:
        # Import model class
        model_class = _get_model_class(model_type, model_name)
        
        # Load data
        dataset_config = DatasetConfig(
            file_path=data_path,
            target_column=target_column
        )
        dataset_loader = DatasetLoader(dataset_config)
        dataset_loader.load().validate().preprocess()
        X, y = dataset_loader.get_data()
        X_train, y_train, X_val, y_val = dataset_loader.split()
        
        # Define parameter space
        param_space = _get_param_space(model_name)
        
        # Create tuner
        tuner = TunerFactory.create(
            tuner_type,
            model_class,
            param_space,
            n_trials=n_trials,
            metric=metric,
            direction='maximize',
            verbose=True
        )
        
        # Run tuning
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Tuning {model_name}...", total=None)
            result = tuner.tune(X_train, y_train, X_val, y_val)
            progress.update(task, completed=True)
        
        # Display results
        console.print("\n[green]✓ Tuning completed![/green]")
        console.print(f"[bold]Best Score:[/bold] {result.best_score:.4f}")
        console.print("[bold]Best Parameters:[/bold]")
        for param, value in result.best_params.items():
            console.print(f"  {param}: {value}")
        
        # Save results
        if output_path:
            tuner.save_results(output_path)
            console.print(f"[green]✓ Results saved to {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def registry_command(
    action: str = typer.Argument(..., help="Action: list, get, delete, update"),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Model ID"),
    model_name: Optional[str] = typer.Option(None, "--name", help="Model name"),
    status: Optional[str] = typer.Option(None, "--status", help="Update status")
):
    """
    Manage the model registry.
    """
    console.print(Panel.fit("📚 AgriMind ML Registry", style="bold cyan"))
    
    try:
        if action == 'list':
            entries = registry.list_models(limit=100)
            table = Table(title="Model Registry")
            table.add_column("Model ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Version", style="blue")
            table.add_column("Status", style="magenta")
            table.add_column("Metrics", style="white")
            
            for entry in entries:
                metrics_str = ", ".join([
                    f"{k}: {v:.3f}" for k, v in list(entry.metrics.items())[:3]
                ])
                table.add_row(
                    entry.model_id[:12],
                    entry.model_name,
                    entry.model_type,
                    entry.version,
                    entry.status,
                    metrics_str
                )
            console.print(table)
            
        elif action == 'get':
            if not model_id:
                raise ValueError("model-id required for get action")
            entry = registry.get_entry(model_id=model_id)
            console.print("[bold]Model Details:[/bold]")
            for key, value in entry.to_dict().items():
                if key not in ['metrics', 'hyperparameters']:
                    console.print(f"  {key}: {value}")
            
            console.print("\n[bold]Metrics:[/bold]")
            for k, v in entry.metrics.items():
                console.print(f"  {k}: {v:.4f}")
            
        elif action == 'update':
            if not model_id or not status:
                raise ValueError("model-id and status required for update action")
            registry.update_status(model_id, status)
            console.print(f"[green]✓ Model {model_id} status updated to {status}[/green]")
            
        elif action == 'delete':
            if not model_id:
                raise ValueError("model-id required for delete action")
            registry.delete_version(model_id, 'latest')
            console.print(f"[green]✓ Model {model_id} deleted[/green]")
            
        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("Available actions: list, get, update, delete")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def _get_model_class(model_type: str, model_name: str):
    """Get model class from model type and name."""
    model_map = {
        'classification': {
            'random_forest': 'RandomForestClassifier',
            'xgboost': 'XGBoostClassifier',
            'lightgbm': 'LightGBMClassifier',
            'catboost': 'CatBoostClassifier',
            'logistic': 'LogisticRegressionClassifier',
            'decision_tree': 'DecisionTreeClassifier',
            'svm': 'SVMClassifier',
            'extra_trees': 'ExtraTreesClassifier',
            'voting': 'VotingEnsemble',
            'stacking': 'StackingEnsemble'
        },
        'regression': {
            'linear': 'LinearRegression',
            'random_forest': 'RandomForestRegressor',
            'xgboost': 'XGBoostRegressor',
            'lightgbm': 'LightGBMRegressor',
            'catboost': 'CatBoostRegressor',
            'elasticnet': 'ElasticNetRegressor',
            'ridge': 'RidgeRegressor',
            'lasso': 'LassoRegressor',
            'svr': 'SVRRegressor'
        }
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    class_name = model_map[model_type].get(model_name)
    if not class_name:
        raise ValueError(f"Unknown model: {model_name} for type {model_type}")
    
    # Import and return the class
    module_path = f"agrimind_ml.{model_type}"
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def _get_param_space(model_name: str) -> Dict[str, Any]:
    """Get parameter space for tuning."""
    param_spaces = {
        'random_forest': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5},
            'max_features': {'type': 'choice', 'values': ['sqrt', 'log2', None]}
        },
        'xgboost': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'learning_rate': {'type': 'loguniform', 'low': 0.001, 'high': 0.3},
            'subsample': {'type': 'uniform', 'low': 0.5, 'high': 1.0},
            'colsample_bytree': {'type': 'uniform', 'low': 0.5, 'high': 1.0}
        },
        'lightgbm': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'max_depth': {'type': 'int', 'low': -1, 'high': 15},
            'learning_rate': {'type': 'loguniform', 'low': 0.001, 'high': 0.3},
            'num_leaves': {'type': 'int', 'low': 10, 'high': 50},
            'subsample': {'type': 'uniform', 'low': 0.5, 'high': 1.0}
        },
        'linear': {
            'fit_intercept': {'type': 'choice', 'values': [True, False]}
        }
    }
    return param_spaces.get(model_name, {'n_estimators': {'type': 'int', 'low': 50, 'high': 200}})


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()