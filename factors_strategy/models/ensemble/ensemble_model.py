"""
Ensemble Model for Stock Prediction
Combines XGBoost, LightGBM, and deep learning models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble model combining multiple algorithms"""
    
    def __init__(self, config_path: str = "config/model.yaml"):
        """Initialize ensemble model"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.model_weights = {}
        self.feature_importance = {}
        
        # Initialize individual models
        self._initialize_models()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['models']
        
    def _initialize_models(self):
        """Initialize all component models"""
        
        # XGBoost
        if self.config['xgboost']['enabled']:
            self.models['xgboost'] = xgb.XGBClassifier(
                **self.config['xgboost']['parameters'],
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
        # LightGBM
        if self.config['lightgbm']['enabled']:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                **self.config['lightgbm']['parameters'],
                verbose=-1
            )
            
        # Initialize weights from config
        self.model_weights = self.config['ensemble']['weights']
        
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Train all models in the ensemble"""
        
        logger.info("Training ensemble models...")
        scores = {}
        
        # Train XGBoost
        if 'xgboost' in self.models:
            logger.info("Training XGBoost...")
            eval_set = [(X_val, y_val)] if X_val is not None else None
            
            self.models['xgboost'].fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=self.config['xgboost']['parameters']['early_stopping_rounds'],
                verbose=False
            )
            
            # Get feature importance
            self.feature_importance['xgboost'] = pd.DataFrame({
                'feature': feature_names or X_train.columns,
                'importance': self.models['xgboost'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Evaluate
            if X_val is not None:
                val_pred = self.models['xgboost'].predict_proba(X_val)[:, 1]
                scores['xgboost'] = roc_auc_score(y_val, val_pred)
                logger.info(f"XGBoost AUC: {scores['xgboost']:.4f}")
                
        # Train LightGBM
        if 'lightgbm' in self.models:
            logger.info("Training LightGBM...")
            eval_set = [(X_val, y_val)] if X_val is not None else None
            
            self.models['lightgbm'].fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=self.config['lightgbm']['parameters']['early_stopping_rounds'],
                verbose=False
            )
            
            # Get feature importance
            self.feature_importance['lightgbm'] = pd.DataFrame({
                'feature': feature_names or X_train.columns,
                'importance': self.models['lightgbm'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Evaluate
            if X_val is not None:
                val_pred = self.models['lightgbm'].predict_proba(X_val)[:, 1]
                scores['lightgbm'] = roc_auc_score(y_val, val_pred)
                logger.info(f"LightGBM AUC: {scores['lightgbm']:.4f}")
                
        # Calculate ensemble score
        if X_val is not None:
            ensemble_pred = self.predict_proba(X_val)
            scores['ensemble'] = roc_auc_score(y_val, ensemble_pred)
            logger.info(f"Ensemble AUC: {scores['ensemble']:.4f}")
            
        return scores
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from ensemble"""
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                predictions[model_name] = model.predict_proba(X)[:, 1]
            else:
                # For models that don't have predict_proba
                predictions[model_name] = model.predict(X)
                
        # Weighted average ensemble
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 0)
            ensemble_pred += weight * pred
            total_weight += weight
            
        # Normalize by total weight
        if total_weight > 0:
            ensemble_pred /= total_weight
            
        return ensemble_pred
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions from ensemble"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
        
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get aggregated feature importance across models"""
        
        # Aggregate importance scores
        all_features = set()
        for model_imp in self.feature_importance.values():
            all_features.update(model_imp['feature'].tolist())
            
        aggregated = pd.DataFrame({'feature': list(all_features)})
        
        # Add importance from each model
        for model_name, importance_df in self.feature_importance.items():
            imp_dict = dict(zip(importance_df['feature'], importance_df['importance']))
            aggregated[f'{model_name}_importance'] = aggregated['feature'].map(imp_dict).fillna(0)
            
        # Calculate weighted average importance
        aggregated['avg_importance'] = 0
        total_weight = 0
        
        for model_name in self.feature_importance.keys():
            weight = self.model_weights.get(model_name, 0)
            aggregated['avg_importance'] += weight * aggregated[f'{model_name}_importance']
            total_weight += weight
            
        if total_weight > 0:
            aggregated['avg_importance'] /= total_weight
            
        # Sort and return top features
        return aggregated.nlargest(top_n, 'avg_importance')
        
    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.Series, 
                      n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = {model_name: [] for model_name in self.models.keys()}
        cv_scores['ensemble'] = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Cross-validation fold {fold + 1}/{n_splits}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train models on this fold
            fold_scores = self.train(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold
            )
            
            # Store scores
            for model_name, score in fold_scores.items():
                cv_scores[model_name].append(score)
                
        # Calculate mean and std
        cv_results = {}
        for model_name, scores in cv_scores.items():
            cv_results[model_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
            
        return cv_results
        
    def optimize_weights(self, 
                        X_val: pd.DataFrame, 
                        y_val: pd.Series,
                        method: str = 'grid_search') -> Dict[str, float]:
        """Optimize ensemble weights"""
        
        logger.info(f"Optimizing ensemble weights using {method}")
        
        # Get individual model predictions
        predictions = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                predictions[model_name] = model.predict_proba(X_val)[:, 1]
            else:
                predictions[model_name] = model.predict(X_val)
                
        if method == 'grid_search':
            # Grid search over weight combinations
            best_score = -np.inf
            best_weights = self.model_weights.copy()
            
            # Generate weight combinations (sum to 1)
            weight_options = np.arange(0, 1.1, 0.1)
            
            for w1 in weight_options:
                for w2 in weight_options:
                    for w3 in weight_options:
                        for w4 in weight_options:
                            for w5 in weight_options:
                                weights = {
                                    'xgboost': w1,
                                    'lightgbm': w2,
                                    'cnn': w3,
                                    'lstm': w4,
                                    'transformer': w5
                                }
                                
                                # Normalize weights
                                total = sum(weights.values())
                                if total == 0:
                                    continue
                                    
                                weights = {k: v/total for k, v in weights.items()}
                                
                                # Calculate ensemble prediction
                                ensemble_pred = np.zeros(len(X_val))
                                for model_name, pred in predictions.items():
                                    if model_name in weights:
                                        ensemble_pred += weights[model_name] * pred
                                        
                                # Evaluate
                                score = roc_auc_score(y_val, ensemble_pred)
                                
                                if score > best_score:
                                    best_score = score
                                    best_weights = weights.copy()
                                    
            self.model_weights = best_weights
            logger.info(f"Optimized weights: {best_weights}")
            logger.info(f"Best ensemble AUC: {best_score:.4f}")
            
        return self.model_weights
        
    def save_models(self, directory: str):
        """Save all models"""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = save_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
            
        # Save ensemble configuration
        ensemble_config = {
            'model_weights': self.model_weights,
            'feature_importance': {
                name: imp.to_dict() 
                for name, imp in self.feature_importance.items()
            }
        }
        
        config_path = save_dir / "ensemble_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(ensemble_config, f)
            
        logger.info(f"Saved ensemble configuration to {config_path}")
        
    def load_models(self, directory: str):
        """Load saved models"""
        load_dir = Path(directory)
        
        # Load individual models
        for model_name in ['xgboost', 'lightgbm']:
            model_path = load_dir / f"{model_name}_model.pkl"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} from {model_path}")
                
        # Load ensemble configuration
        config_path = load_dir / "ensemble_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                ensemble_config = yaml.safe_load(f)
                
            self.model_weights = ensemble_config['model_weights']
            
            # Restore feature importance
            for name, imp_dict in ensemble_config['feature_importance'].items():
                self.feature_importance[name] = pd.DataFrame(imp_dict)
                
            logger.info("Loaded ensemble configuration")


class StackingEnsemble(EnsembleModel):
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, config_path: str = "config/model.yaml"):
        """Initialize stacking ensemble"""
        super().__init__(config_path)
        
        # Meta-learner
        self.meta_learner = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        
    def train_stacking(self, 
                      X_train: pd.DataFrame, 
                      y_train: pd.Series,
                      X_val: pd.DataFrame,
                      y_val: pd.Series) -> Dict[str, float]:
        """Train stacking ensemble"""
        
        # First, train base models
        base_scores = self.train(X_train, y_train, X_val, y_val)
        
        # Generate meta-features (predictions from base models)
        meta_train = self._generate_meta_features(X_train)
        meta_val = self._generate_meta_features(X_val)
        
        # Train meta-learner
        logger.info("Training meta-learner...")
        self.meta_learner.fit(meta_train, y_train)
        
        # Evaluate
        meta_pred = self.meta_learner.predict_proba(meta_val)[:, 1]
        stacking_score = roc_auc_score(y_val, meta_pred)
        
        logger.info(f"Stacking ensemble AUC: {stacking_score:.4f}")
        
        scores = base_scores.copy()
        scores['stacking'] = stacking_score
        
        return scores
        
    def _generate_meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate meta-features from base model predictions"""
        meta_features = pd.DataFrame()
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                meta_features[f'{model_name}_proba'] = model.predict_proba(X)[:, 1]
            else:
                meta_features[f'{model_name}_pred'] = model.predict(X)
                
        return meta_features
        
    def predict_proba_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from stacking ensemble"""
        meta_features = self._generate_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)[:, 1]