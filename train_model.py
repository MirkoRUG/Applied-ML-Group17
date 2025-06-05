"""
Script to train and save the CatBoost movie score prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

from project_name.models.catboost_model import train_and_save_model


def main():
    """Main function to train and save the model."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train CatBoost movie score prediction model')
    parser.add_argument('--data_path', type=str, default='data.csv',
                       help='Path to the training data CSV file')
    parser.add_argument('--model_path', type=str, default='models/catboost_movie_model.cbm',
                       help='Path to save the trained model')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of boosting iterations')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--depth', type=int, default=6,
                       help='Tree depth')
    
    args = parser.parse_args()
    
    # Custom model parameters
    model_params = {
        'iterations': args.iterations,
        'learning_rate': args.learning_rate,
        'depth': args.depth,
        'loss_function': 'RMSE',
        'verbose': True,
        'random_seed': 42,
        'early_stopping_rounds': 50
    }
    
    print("Starting model training...")
    print(f"Data path: {args.data_path}")
    print(f"Model will be saved to: {args.model_path}")
    print(f"Model parameters: {model_params}")
    
    try:
        # Train and save model
        metrics = train_and_save_model(
            data_path=args.data_path,
            model_path=args.model_path,
            model_params=model_params
        )
        
        print("\nTraining completed successfully!")
        print(f"Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"Training MAE: {metrics['train_mae']:.4f}")
        print(f"Number of features: {metrics['n_features']}")
        print(f"Number of training samples: {metrics['n_samples']}")
        
        # Print top 10 most important features
        print("\nTop 10 most important features:")
        feature_importance = metrics['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature:<20} {importance:.4f}")
        
        # Save training report
        report_path = args.model_path.replace('.cbm', '_training_report.json')
        training_report = {
            'timestamp': datetime.now().isoformat(),
            'data_path': args.data_path,
            'model_path': args.model_path,
            'model_params': model_params,
            'metrics': metrics
        }
        
        with open(report_path, 'w') as f:
            json.dump(training_report, f, indent=2, default=str)
        
        print(f"\nTraining report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
