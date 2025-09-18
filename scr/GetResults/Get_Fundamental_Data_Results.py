import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# NUEVAS IMPORTACIONES DEL SISTEMA ML AVANZADO
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
import multiprocessing
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import joblib
from tqdm import tqdm
import time
import traceback
from enum import Enum

# Machine Learning imports adicionales
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.feature_selection import SelectKBest, f_regression

# Statistical imports adicionales
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =====================================================================================
# SISTEMA ML AVANZADO CON ANTI-DATA LEAKAGE (DEL SEGUNDO CÓDIGO)
# =====================================================================================

class ModelType(Enum):
    TICKER = "ticker"
    INDUSTRY = "industry"

@dataclass
class MLValuationConfig:
    """Configuración optimizada para Fair Value"""
    
    # Requerimientos de datos flexibles
    min_tickers_per_industry: int = 3
    min_samples_industry: int = 6
    min_samples_ticker: int = 5
    min_years_per_ticker: int = 1
    
    # Features ESTRICTAMENTE prohibidas para evitar data leakage
    forbidden_features: List[str] = field(default_factory=lambda: [
        # Precios y valoraciones (target leakage directo)
        'Close', 'MarketCap', 'Log_MarketCap', 'P/E', 'P/B', 'P/S', 'P/CF',
        'Enterprise Value', 'EV/EBITDA', 'EV/Sales', 'Log_Enterprise Value',
        'Revenue/Share', 'Cash/Share', 'Assets/Share', 'FCF_Yield', 'Dividend_Yield',
        
        # Metadatos temporales
        'Year', 'Date', 'ticker', 'period', 'fiscal_year',
        
        # Features sospechosas de leakage (correlacionadas con precios)
        'Log_Revenue', 'Log_Total Assets', 'Log_EBITDA', 'Log_Net Income',
        'Comprehensive Income',  # Puede incluir cambios en valoración
        
        # Crecimientos que pueden ser calculados con información futura
        'Revenue_Growth_Next_Y', 'Income_Growth_Next_Y', 'EPS_Growth_Next_Y',
        
        # Per-share metrics que dependen del precio
        'Book Value Per Share', 'Operating Cash Flow Per Share', 'Free Cash Flow Per Share',
        'Earnings Per Share', 'Basic EPS', 'Diluted EPS'
    ])
    
    # Features SEGURAS - fundamentales puros sin transformaciones sospechosas
    safe_fundamental_features: List[str] = field(default_factory=lambda: [
        # Estado de resultados - valores absolutos
        'Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EBITDA', 'EBIT',
        'Cost Of Goods Sold', 'Operating Expenses', 'Research And Development Expenses',
        'Income Taxes', 'Interest Expense',
        
        # Balance general - valores absolutos
        'Total Assets', 'Cash On Hand', 'Total Liabilities', 'Share Holder Equity',
        'Total Current Assets', 'Total Current Liabilities', 'Working Capital',
        'Long Term Debt', 'Inventory', 'Receivables', 'Property, Plant, And Equipment',
        'Goodwill And Intangible Assets', 'Retained Earnings (Accumulated Deficit)',
        
        # Flujo de efectivo
        'Cash Flow From Operating Activities', 'Free Cash Flow', 'Net Cash Flow',
        'Cash Flow From Investing Activities', 'Cash Flow From Financial Activities',
        
        # Ratios independientes del precio
        'ROE - Return On Equity', 'ROA - Return On Assets', 'ROIC', 'ROI - Return On Investment',
        'Operating Margin', 'Net Profit Margin', 'Gross Margin', 'EBIT Margin', 'EBITDA Margin',
        'Current Ratio', 'Quick Ratio', 'Cash Ratio', 'Debt/Equity Ratio', 'Debt/Assets',
        'Interest Coverage', 'Asset Turnover', 'Inventory Turnover Ratio',
        
        # Crecimientos históricos (seguros)
        'Revenue_Growth_1Y', 'Net Income_Growth_1Y', 'EBITDA_Growth_1Y',
        'Operating Income_Growth_1Y', 'Free Cash Flow_Growth_1Y',
        
        # Scores de calidad
        'Piotroski_Score', 'Altman_Z_Score',
        
        # Variables macro (independientes de la empresa)
        'VIX', 'VIX_MA', '10Y_Treasury', '2Y_Treasury', 'AAA_Yield', 'Credit_Spread',
        'GDP_Growth', 'Inflation_Rate', 'unemployment_rate', 'industrial_production',
        'consumer_sentiment', 'CPI', 'm2_usd', 'eur_usd_rate', 'gold_usd'
    ])
    
    # Configuración de features
    max_features_per_model: int = 6  # Reducido para evitar overfitting
    min_correlation_with_target: float = 0.02
    max_feature_correlation: float = 0.80
    
    # Rangos de Fair Value
    range_tolerance_levels: Dict[str, float] = field(default_factory=lambda: {
        'tight': 0.25,      # ±25%
        'moderate': 0.40,   # ±40%
        'wide': 0.60        # ±60%
    })
    
    # Criterios de aceptación para Fair Value
    min_fair_value_capture: float = 0.35      # 35% en rango moderate
    min_directional_sense: float = 0.52       # 52% direccional
    max_extreme_miss_rate: float = 0.40       # Máximo 40% errores extremos
    
    # Targets
    industry_target: str = 'MarketCap'
    ticker_target: str = 'Close'
    
    # Sistema
    random_state: int = 42
    n_jobs: int = -1
    
    # Validación TEMPORAL para evitar leakage
    use_temporal_validation: bool = True
    n_folds: int = 3
    min_samples_for_kfold: int = 8
    
    # Criterios de usabilidad
    min_ticker_usability: float = 0.20
    min_industry_usability: float = 0.15
    
    # Control de outliers y data quality
    max_outlier_ratio: float = 0.1          # Máximo 10% outliers
    outlier_threshold_iqr: float = 3.0      # 3 IQR para outliers
    max_acceptable_mape: float = 5.0        # Máximo 500% MAPE
    max_acceptable_r2: float = 0.95         # Máximo R² para detectar overfitting
    
    # Almacenamiento
    save_models: bool = True
    model_compression_level: int = 3
    log_level: str = "INFO"
    enable_file_logging: bool = True
    
    def to_dict(self):
        """Convertir a dict con tipos JSON-serializables"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                result[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result

def setup_logging(config: MLValuationConfig, log_file: Optional[str] = None):
    """Setup de logging optimizado"""
    logger = logging.getLogger('ml_fair_value')
    
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if config.enable_file_logging and log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger

def convert_to_json_serializable(obj):
    """Convertir objetos numpy a tipos JSON-serializables"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

class DataLeakageDetector:
    """Detector de data leakage y problemas de datos"""
    
    def __init__(self, config: MLValuationConfig):
        self.config = config
        self.logger = logging.getLogger('ml_fair_value.DataLeakageDetector')
    
    def detect_leakage_risk(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
        """Detectar riesgo de data leakage"""
        
        risks = {
            'high_risk_features': [],
            'suspicious_correlations': [],
            'extreme_r2_risk': False,
            'recommendations': []
        }
        
        # 1. Verificar correlaciones extremas con target
        for feature in feature_names:
            if feature in X.columns:
                try:
                    corr = abs(X[feature].corr(y))
                    if pd.notna(corr) and corr > 0.95:  # Correlación sospechosamente alta
                        risks['high_risk_features'].append({
                            'feature': feature,
                            'correlation': float(corr),
                            'risk': 'extreme_correlation'
                        })
                except:
                    continue
        
        # 2. Verificar features con nombres sospechosos
        suspicious_keywords = ['log_', 'per_share', '_yield', 'comprehensive', '_ratio']
        for feature in feature_names:
            for keyword in suspicious_keywords:
                if keyword.lower() in feature.lower():
                    risks['suspicious_correlations'].append({
                        'feature': feature,
                        'reason': f'Contains suspicious keyword: {keyword}'
                    })
        
        # 3. Generar recomendaciones
        if risks['high_risk_features']:
            risks['recommendations'].append("Remove features with extreme correlations (>0.95)")
        
        if risks['suspicious_correlations']:
            risks['recommendations'].append("Review features with suspicious naming patterns")
        
        return risks

class FairValueEvaluator:
    """Evaluador para Fair Value con detección de overfitting"""
    
    def __init__(self, config: MLValuationConfig):
        self.config = config
        self.logger = logging.getLogger('ml_fair_value.FairValueEvaluator')
        self.leakage_detector = DataLeakageDetector(config)
    
    def evaluate_fair_value_quality(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   X: Optional[pd.DataFrame] = None, 
                                   feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluar calidad con detección de problemas"""
        
        # Limpiar datos
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
        if np.sum(mask) < 2:
            return self._get_default_metrics()
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # Detectar y filtrar outliers extremos
        y_true_clean, y_pred_clean = self._filter_extreme_outliers(y_true_clean, y_pred_clean)
        
        if len(y_true_clean) < 2:
            return self._get_default_metrics()
        
        metrics = {}
        
        # 1. Fair Value Range Capture
        for range_name, tolerance in self.config.range_tolerance_levels.items():
            lower_bound = y_pred_clean * (1 - tolerance)
            upper_bound = y_pred_clean * (1 + tolerance)
            
            within_range = (y_true_clean >= lower_bound) & (y_true_clean <= upper_bound)
            capture_rate = np.mean(within_range)
            metrics[f'capture_rate_{range_name}'] = float(capture_rate)
        
        # 2. Directional Sense
        directional_accuracy = self._calculate_directional_sense(y_true_clean, y_pred_clean)
        metrics['directional_sense'] = float(directional_accuracy)
        
        # 3. Extreme Miss Rate
        relative_errors = np.abs(y_pred_clean - y_true_clean) / y_true_clean
        extreme_miss_rate = np.mean(relative_errors > 1.0)
        metrics['extreme_miss_rate'] = float(extreme_miss_rate)
        
        # 4. Consistency Score
        consistency_score = self._calculate_fair_value_consistency(y_true_clean, y_pred_clean)
        metrics['consistency_score'] = float(consistency_score)
        
        # 5. Fair Value Score
        fair_value_score = self._calculate_fair_value_score(metrics)
        metrics['fair_value_score'] = float(fair_value_score)
        
        # 6. Métricas tradicionales con validación
        try:
            r2 = r2_score(y_true_clean, y_pred_clean)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
            
            # VALIDAR MÉTRICAS PARA DETECTAR PROBLEMAS
            if r2 > self.config.max_acceptable_r2:
                self.logger.warning(f"R² sospechosamente alto: {r2:.4f} - posible overfitting o data leakage")
                metrics['overfitting_risk'] = True
            elif r2 < -1.0:
                self.logger.warning(f"R² extremadamente bajo: {r2:.4f} - modelo no confiable")
                metrics['extremely_low_r2'] = True
            else:
                metrics['overfitting_risk'] = False
                metrics['extremely_low_r2'] = False
            
            if mape > self.config.max_acceptable_mape:
                self.logger.warning(f"MAPE muy alto: {mape:.4f} - modelo problemático")
                metrics['high_error_risk'] = True
            else:
                metrics['high_error_risk'] = False
            
            metrics['r2'] = float(min(r2, self.config.max_acceptable_r2))  # Cap R²
            metrics['mae'] = float(mae)
            metrics['mape'] = float(min(mape, self.config.max_acceptable_mape))  # Cap MAPE
            metrics['median_error'] = float(np.median(relative_errors))
            
        except Exception as e:
            self.logger.warning(f"Error calculando métricas tradicionales: {e}")
            metrics.update({
                'r2': -1.0, 'mae': float('inf'), 'mape': float('inf'), 
                'median_error': 1.0, 'overfitting_risk': True, 'high_error_risk': True
            })
        
        # 7. Detectar data leakage si tenemos features
        if X is not None and feature_names is not None:
            leakage_analysis = self.leakage_detector.detect_leakage_risk(X, pd.Series(y_true), feature_names)
            metrics['leakage_risk_detected'] = len(leakage_analysis['high_risk_features']) > 0
        else:
            metrics['leakage_risk_detected'] = False
        
        return metrics
    
    def _filter_extreme_outliers(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filtrar outliers extremos que pueden distorsionar métricas"""
        
        # Filtro 1: Eliminar predicciones extremas vs valores reales
        ratio = y_pred / y_true
        q1, q3 = np.percentile(ratio, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.config.outlier_threshold_iqr * iqr
        upper_bound = q3 + self.config.outlier_threshold_iqr * iqr
        
        mask1 = (ratio >= lower_bound) & (ratio <= upper_bound)
        
        # Filtro 2: Eliminar valores extremos en y_true
        y_true_q1, y_true_q3 = np.percentile(y_true, [5, 95])  # Usar percentiles más conservadores
        mask2 = (y_true >= y_true_q1) & (y_true <= y_true_q3)
        
        # Filtro 3: Eliminar predicciones con órdenes de magnitud incorrectos
        mask3 = (y_pred > y_true * 0.01) & (y_pred < y_true * 100)  # Rango 1:100
        
        # Combinar filtros
        combined_mask = mask1 & mask2 & mask3
        
        outliers_removed = np.sum(~combined_mask)
        outlier_ratio = outliers_removed / len(y_true)
        
        if outlier_ratio > self.config.max_outlier_ratio:
            self.logger.warning(f"Alto ratio de outliers removidos: {outlier_ratio:.2%}")
        
        return y_true[combined_mask], y_pred[combined_mask]
    
    def _calculate_directional_sense(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular sentido direccional"""
        if len(y_true) <= 1:
            return 0.5
        
        try:
            rank_corr, _ = spearmanr(y_true, y_pred)
            if np.isnan(rank_corr):
                return 0.5
            
            directional_score = (rank_corr + 1) / 2
            return max(0.0, min(1.0, directional_score))
            
        except:
            return 0.5
    
    def _calculate_fair_value_consistency(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Medir consistencia del fair value"""
        try:
            if len(y_true) < 3:
                return 0.5
            
            ratios = y_pred / y_true
            ratio_stability = 1.0 / (1.0 + np.std(ratios))
            
            extreme_predictions = np.mean((y_pred > y_true * 2) | (y_pred < y_true * 0.5))
            extreme_penalty = 1.0 - extreme_predictions
            
            consistency = (ratio_stability * 0.6 + extreme_penalty * 0.4)
            return max(0.0, min(1.0, consistency))
            
        except:
            return 0.0
    
    def _calculate_fair_value_score(self, metrics: Dict[str, float]) -> float:
        """Score compuesto para fair value"""
        
        moderate_capture = metrics.get('capture_rate_moderate', 0)
        directional_sense = metrics.get('directional_sense', 0.5)
        extreme_miss_rate = metrics.get('extreme_miss_rate', 1.0)
        consistency = metrics.get('consistency_score', 0)
        
        # Penalizar modelos con riesgo de overfitting
        overfitting_penalty = 0.5 if metrics.get('overfitting_risk', False) else 1.0
        error_penalty = 0.7 if metrics.get('high_error_risk', False) else 1.0
        
        base_score = (
            moderate_capture * 0.40 +
            directional_sense * 0.25 +
            consistency * 0.15
        )
        
        extreme_penalty = max(0, 1 - extreme_miss_rate * 1.2)
        
        final_score = base_score * extreme_penalty * overfitting_penalty * error_penalty
        return max(0.0, min(1.0, final_score))
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Métricas por defecto"""
        return {
            'capture_rate_tight': 0.0,
            'capture_rate_moderate': 0.0,
            'capture_rate_wide': 0.0,
            'directional_sense': 0.5,
            'extreme_miss_rate': 1.0,
            'consistency_score': 0.0,
            'fair_value_score': 0.0,
            'r2': -1.0,
            'mae': 0.0,
            'mape': 1.0,
            'median_error': 1.0,
            'overfitting_risk': True,
            'high_error_risk': True,
            'leakage_risk_detected': False
        }
    
    def is_fair_value_model_useful(self, metrics: Dict[str, float], model_type: ModelType) -> bool:
        """Determinar si un modelo es útil con validaciones adicionales"""
        
        # Rechazar modelos con riesgos altos
        if metrics.get('overfitting_risk', False) or metrics.get('extremely_low_r2', False) or metrics.get('leakage_risk_detected', False):
            self.logger.debug("Modelo rechazado por riesgo de calidad")
            return False
        
        if metrics.get('leakage_risk_detected', False):
            self.logger.debug("Modelo rechazado por riesgo de data leakage")
            return False
        
        fair_value_score = metrics.get('fair_value_score', 0)
        moderate_capture = metrics.get('capture_rate_moderate', 0)
        extreme_miss_rate = metrics.get('extreme_miss_rate', 1.0)
        directional_sense = metrics.get('directional_sense', 0.5)
        
        if model_type == ModelType.TICKER:
            min_usability = self.config.min_ticker_usability
        else:
            min_usability = self.config.min_industry_usability
        
        # Condiciones más estrictas para evitar modelos problemáticos
        useful_conditions = [
            fair_value_score >= min_usability,
            (moderate_capture >= self.config.min_fair_value_capture and 
             extreme_miss_rate <= self.config.max_extreme_miss_rate),
            (directional_sense >= self.config.min_directional_sense and 
             moderate_capture >= 0.25),
            (moderate_capture >= 0.45 and directional_sense >= 0.50)
        ]
        
        is_useful = any(useful_conditions)
        
        self.logger.debug(f"Modelo {model_type.value} útil: {is_useful} "
                         f"(FV_score: {fair_value_score:.3f}, "
                         f"capture: {moderate_capture:.3f}, "
                         f"directional: {directional_sense:.3f})")
        
        return is_useful

class SafeFeatureSelector:
    """Selector de features con de data leakage"""
    
    def __init__(self, config: MLValuationConfig):
        self.config = config
        self.logger = logging.getLogger('ml_fair_value.SafeFeatureSelector')
    
    def select_safe_features(self, X: pd.DataFrame, y: pd.Series, model_type: ModelType) -> List[str]:
        """Selección segura de features sin data leakage"""
        
        # Usar SOLO features seguras predefinidas
        safe_features = [col for col in self.config.safe_fundamental_features 
                        if col in X.columns]
        
        self.logger.debug(f"Features seguras disponibles: {len(safe_features)}")
        
        if len(safe_features) <= self.config.max_features_per_model:
            return safe_features
        
        # Selección por correlación con validación adicional
        feature_scores = {}
        
        for col in safe_features:
            try:
                col_data = X[col].dropna()
                valid_ratio = len(col_data) / len(X)
                
                if valid_ratio >= 0.6:  # Al menos 60% de datos válidos
                    corr = abs(col_data.corr(y.loc[col_data.index]))
                    if pd.notna(corr) and self.config.min_correlation_with_target <= corr <= 0.90:
                        # Score = correlación * ratio de datos válidos
                        feature_scores[col] = corr * valid_ratio
            except Exception as e:
                self.logger.debug(f"Error evaluando feature {col}: {e}")
                continue
        
        if len(feature_scores) < 2:
            return safe_features[:self.config.max_features_per_model]
        
        # Seleccionar evitando multicolinealidad
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        for feature, score in sorted_features:
            if len(selected) >= self.config.max_features_per_model:
                break
            
            # Verificar multicolinealidad
            is_redundant = False
            for sel_feature in selected:
                try:
                    inter_corr = abs(X[feature].corr(X[sel_feature]))
                    if pd.notna(inter_corr) and inter_corr > self.config.max_feature_correlation:
                        is_redundant = True
                        break
                except:
                    continue
            
            if not is_redundant:
                selected.append(feature)
        
        self.logger.debug(f"Features seleccionadas: {len(selected)}")
        return selected

class TemporalFairValueEnsemble(BaseEstimator, RegressorMixin):
    """Ensemble con validación temporal para prevenir data leakage"""
    
    def __init__(self, config: MLValuationConfig, model_type: ModelType):
        self.config = config
        self.model_type = model_type
        self.logger = logging.getLogger(f'ml_fair_value.TemporalFairValueEnsemble.{model_type.value}')
        
        # Modelos base más conservadores
        self.base_models = {
            'ridge': Ridge(alpha=10.0, random_state=config.random_state),  # Más regularización
            'random_forest': RandomForestRegressor(
                n_estimators=30,
                max_depth=4,  # Más shallow para evitar overfitting
                min_samples_split=5,
                min_samples_leaf=3,
                max_features=0.6,
                random_state=config.random_state,
                n_jobs=1
            ),
            'huber': HuberRegressor(epsilon=1.35, alpha=0.1)  # Más regularización
        }
        
        self.fitted_models = {}
        self.model_weights = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        self.evaluator = FairValueEvaluator(config)
        self.training_metrics = {}
    
    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """Entrenar con validación temporal"""
        
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Limpiar datos
        X_clean, y_clean = self._clean_data(X, y)
        
        if len(X_clean) < self.config.min_samples_ticker:
            raise ValueError("Datos insuficientes para entrenamiento")
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Entrenar modelos con validación temporal
        model_scores = {}
        
        for name, model in self.base_models.items():
            try:
                score = self._train_and_validate_temporal(model, X_scaled, y_clean, X_clean, name)
                
                if score > 0.05:  # Umbral muy bajo pero con validaciones de calidad
                    model.fit(X_scaled, y_clean)
                    self.fitted_models[name] = model
                    model_scores[name] = score
                    
                    self.logger.debug(f"Modelo {name} entrenado - Score: {score:.3f}")
                    
            except Exception as e:
                self.logger.debug(f"Error entrenando modelo {name}: {e}")
                continue
        
        if not model_scores:
            # Fallback muy conservador
            try:
                ridge = Ridge(alpha=50.0, random_state=self.config.random_state)
                ridge.fit(X_scaled, y_clean)
                self.fitted_models['ridge'] = ridge
                model_scores['ridge'] = 0.1
                self.logger.warning("Usando modelo Ridge ultra-conservador como fallback")
            except Exception as e:
                raise ValueError(f"Falló incluso el modelo fallback: {e}")
        
        # Calcular pesos
        total_score = sum(model_scores.values())
        if total_score > 0:
            self.model_weights = {name: score/total_score for name, score in model_scores.items()}
        else:
            n_models = len(model_scores)
            self.model_weights = {name: 1.0/n_models for name in model_scores.keys()}
        
        self.logger.info(f"Ensemble temporal entrenado con {len(self.fitted_models)} modelos")
        return self
    
    def _train_and_validate_temporal(self, model, X: np.ndarray, y: np.ndarray, 
                                   X_df: pd.DataFrame, model_name: str) -> float:
        """Validación temporal para evitar data leakage"""
        
        if len(X) >= self.config.min_samples_for_kfold and self.config.use_temporal_validation:
            # Usar TimeSeriesSplit para validación temporal
            tscv = TimeSeriesSplit(n_splits=self.config.n_folds)
            
            y_pred_cv = []
            y_true_cv = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if len(X_train) < 3 or len(X_val) < 1:
                    continue
                
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                pred = model_clone.predict(X_val)
                
                y_pred_cv.extend(pred)
                y_true_cv.extend(y_val)
            
            if len(y_pred_cv) < 2:
                # Fallback a validación simple
                return self._simple_validation(model, X, y, X_df, model_name)
            
            # Evaluar con validaciones de calidad
            metrics = self.evaluator.evaluate_fair_value_quality(
                np.array(y_true_cv), np.array(y_pred_cv), X_df, self.feature_names
            )
            score = metrics['fair_value_score']
            
        else:
            # Para datasets pequeños, validación simple
            score = self._simple_validation(model, X, y, X_df, model_name)
        
        # Guardar métricas
        self.training_metrics[model_name] = metrics if 'metrics' in locals() else {}
        
        return score
    
    def _simple_validation(self, model, X: np.ndarray, y: np.ndarray, 
                          X_df: pd.DataFrame, model_name: str) -> float:
        """Validación simple para datasets pequeños"""
        try:
            model.fit(X, y)
            y_pred = model.predict(X)
            metrics = self.evaluator.evaluate_fair_value_quality(y, y_pred, X_df, self.feature_names)
            return metrics['fair_value_score'] * 0.7  # Penalizar por falta de validación cruzada
        except:
            return 0.0
    
    def predict_range(self, X) -> Dict[str, np.ndarray]:
        """Generar rangos de fair value"""
        
        X_clean = self._clean_data(X, is_prediction=True)
        X_scaled = self.scaler.transform(X_clean)
        
        predictions = []
        weights = []
        
        for name, model in self.fitted_models.items():
            try:
                pred = model.predict(X_scaled)
                predictions.append(pred)
                weights.append(self.model_weights[name])
            except Exception as e:
                self.logger.debug(f"Error predicción {name}: {e}")
                continue
        
        if not predictions:
            n_samples = len(X_clean)
            return {
                'lower': np.full(n_samples, 10.0),
                'mean': np.full(n_samples, 50.0),
                'upper': np.full(n_samples, 200.0)
            }
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        ensemble_mean = np.average(predictions, axis=0, weights=weights)
        
        # Rangos conservadores
        if len(predictions) > 1:
            model_std = np.std(predictions, axis=0)
            uncertainty = np.maximum(model_std, ensemble_mean * 0.30)
        else:
            uncertainty = ensemble_mean * 0.35
        
        moderate_tolerance = self.config.range_tolerance_levels['moderate']
        
        return {
            'lower': ensemble_mean * (1 - moderate_tolerance),
            'mean': ensemble_mean,
            'upper': ensemble_mean * (1 + moderate_tolerance)
        }
    
    def predict(self, X) -> np.ndarray:
        """Predicción simple"""
        ranges = self.predict_range(X)
        return ranges['mean']
    
    def _clean_data(self, X, y=None, is_prediction: bool = False):
        """Limpiar datos con control de outliers"""
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values.copy()
            if y is not None and hasattr(y, 'values'):
                y_values = y.values.copy()
            else:
                y_values = y
        else:
            X_values = X.copy()
            y_values = y.copy() if y is not None else None
        
        X_values = np.array(X_values, dtype=float)
        X_values[np.isinf(X_values)] = np.nan
        
        if y_values is not None and not is_prediction:
            y_values = np.array(y_values, dtype=float)
            y_values[np.isinf(y_values)] = np.nan
        
        # Imputación conservadora
        if not is_prediction:
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X_values)
            self.imputer_ = imputer
            
            if y_values is not None:
                valid_mask = ~np.isnan(y_values)
                X_clean = X_imputed[valid_mask]
                y_clean = y_values[valid_mask]
                
                # Control de outliers en target
                if len(y_clean) > 5:
                    q1, q3 = np.percentile(y_clean, [10, 90])  # Más conservador
                    iqr = q3 - q1
                    lower_bound = q1 - 2 * iqr
                    upper_bound = q3 + 2 * iqr
                    
                    outlier_mask = (y_clean >= lower_bound) & (y_clean <= upper_bound)
                    X_clean = X_clean[outlier_mask]
                    y_clean = y_clean[outlier_mask]
                
                return X_clean, y_clean
            else:
                return X_imputed
        else:
            if hasattr(self, 'imputer_'):
                X_imputed = self.imputer_.transform(X_values)
            else:
                imputer = SimpleImputer(strategy='median')
                X_imputed = imputer.fit_transform(X_values)
            return X_imputed

class MLFairValueSystem:
    """Sistema principal ML Fair Value"""
    
    def __init__(self, config: Optional[MLValuationConfig] = None, base_paths: Optional[Dict] = None):
        self.config = config or MLValuationConfig()
        self.logger = setup_logging(self.config)
        self.evaluator = FairValueEvaluator(self.config)
        self.feature_selector = SafeFeatureSelector(self.config)
        
        # Setup paths
        self._setup_paths(base_paths)
        
        # Storage
        self.ticker_models = {}
        self.industry_models = {}
        self.ticker_to_industry = {}
        self.model_performance_metrics = {}
        self.system_statistics = {}
        self.feature_usage_stats = Counter()
        
        # Data
        self.screener_data = None
        self.available_tickers = []
    
    def _setup_paths(self, base_paths: Optional[Dict] = None):
        """Configurar rutas del sistema"""
        if base_paths:
            self.ROOT = Path(base_paths['root'])
            self.HISTORICAL_PATH = Path(base_paths['historical'])
            self.FORECAST_PATH = Path(base_paths['forecast'])
            self.SCREENER_PATH = Path(base_paths['screener'])
            self.OUTPUT_PATH = Path(base_paths['output'])
        else:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.ROOT = project_root / "data"
            self.HISTORICAL_PATH = self.ROOT / "Fundamental_Data_and_Ratios" / "Anual"
            self.FORECAST_PATH = self.ROOT / "Forecast_Fundamental_Data_and_Ratios" / "Anual"
            self.SCREENER_PATH = self.ROOT / "Ticker_List" / "screener.parquet"
            self.OUTPUT_PATH = self.ROOT / "Results"
        
        self.OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    
    def load_data(self) -> bool:
        """Cargar datos del sistema"""
        self.logger.info("Cargando datos para Fair Value ML...")
        
        try:
            self.screener_data = pd.read_parquet(self.SCREENER_PATH)
            self.logger.info(f"Screener cargado: {len(self.screener_data)} empresas")
            
            available_tickers = self.screener_data['ticker'].unique()
            self.available_tickers = list(available_tickers)
            self.logger.info(f"Tickers a procesar: {len(self.available_tickers)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            return False
    
    def train_ticker_model(self, ticker: str) -> Optional[Dict]:
        """Entrenar modelo de fair value para ticker con validaciones"""
        
        try:
            hist_path = self.HISTORICAL_PATH / f"{ticker}.parquet"
            if not hist_path.exists():
                return None
            
            df = pd.read_parquet(hist_path)
            if len(df) < self.config.min_samples_ticker:
                return None
            
            # Preparar target
            target_col = self.config.ticker_target
            if target_col not in df.columns:
                return None
            
            y = pd.to_numeric(df[target_col], errors='coerce')
            valid_mask = y.notna() & np.isfinite(y) & (y > 0)
            
            df_clean = df[valid_mask].copy()
            y_clean = y[valid_mask]
            
            if len(df_clean) < self.config.min_samples_ticker:
                return None
            
            # USAR SOLO FEATURES SEGURAS
            safe_features = self.feature_selector.select_safe_features(df_clean, y_clean, ModelType.TICKER)
            if len(safe_features) < 2:
                return None
            
            X_selected = df_clean[safe_features]
            
            # Actualizar estadísticas de uso de features
            for feature in safe_features:
                self.feature_usage_stats[feature] += 1
            
            # Entrenar modelo temporal
            model = TemporalFairValueEnsemble(self.config, ModelType.TICKER)
            model.fit(X_selected, y_clean)
            
            # Evaluar con validaciones
            y_pred = model.predict(X_selected)
            fair_value_metrics = self.evaluator.evaluate_fair_value_quality(
                y_clean.values, y_pred, X_selected, safe_features
            )
            
            # Verificar si es útil
            if not self.evaluator.is_fair_value_model_useful(fair_value_metrics, ModelType.TICKER):
                return None
            
            # Convertir métricas a tipos JSON-serializables
            safe_metrics = convert_to_json_serializable(fair_value_metrics)
            
            # Guardar métricas
            model_id = f"ticker_{ticker}"
            self.model_performance_metrics[model_id] = {
                'model_type': 'ticker',
                'ticker': ticker,
                'n_samples': int(len(y_clean)),
                'n_features': int(len(safe_features)),
                'features': safe_features,
                **safe_metrics
            }
            
            return {
                'model': model,
                'features': safe_features,
                'metrics': safe_metrics,
                'n_samples': len(y_clean)
            }
            
        except Exception as e:
            self.logger.debug(f"Error entrenando ticker {ticker}: {e}")
            return None
    
    def train_industry_models(self) -> bool:
        """Entrenar modelos por industria"""
        
        # Agrupar por industria
        industry_groups = {}
        for _, row in self.screener_data.iterrows():
            ticker = row['ticker']
            if ticker in self.available_tickers:
                industry = row.get('zacks_x_ind_desc', 'Unknown')
                if industry != 'Unknown':
                    if industry not in industry_groups:
                        industry_groups[industry] = []
                    industry_groups[industry].append(ticker)
        
        # Filtrar industrias
        industry_groups = {k: v for k, v in industry_groups.items() 
                         if len(v) >= self.config.min_tickers_per_industry}
        
        self.logger.info(f"Entrenando modelos para {len(industry_groups)} industrias")
        
        successful_models = 0
        for industry, tickers in industry_groups.items():
            try:
                model_result = self._train_industry_model(industry, tickers)
                if model_result:
                    self.industry_models[industry] = model_result
                    for ticker in tickers:
                        self.ticker_to_industry[ticker] = industry
                    successful_models += 1
                    
            except Exception as e:
                self.logger.debug(f"Error entrenando industria {industry}: {e}")
                continue
        
        self.logger.info(f"Modelos de industria entrenados: {successful_models}")
        return successful_models > 0
    
    def _train_industry_model(self, industry: str, tickers: List[str]) -> Optional[Dict]:
        """Entrenar modelo para industria con validaciones"""
        
        all_data = []
        all_targets = []
        
        for ticker in tickers:
            try:
                hist_path = self.HISTORICAL_PATH / f"{ticker}.parquet"
                if not hist_path.exists():
                    continue
                
                df = pd.read_parquet(hist_path)
                if len(df) < 2:
                    continue
                
                # Target para industria
                target_col = self.config.industry_target
                if target_col not in df.columns:
                    if 'Close' in df.columns and 'Shares Outstanding' in df.columns:
                        shares = pd.to_numeric(df['Shares Outstanding'], errors='coerce')
                        close = pd.to_numeric(df['Close'], errors='coerce')
                        df[target_col] = close * shares
                    else:
                        continue
                
                y = pd.to_numeric(df[target_col], errors='coerce')
                valid_mask = y.notna() & np.isfinite(y) & (y > 0)
                
                df_clean = df[valid_mask]
                y_clean = y[valid_mask]
                
                if len(df_clean) >= 1:
                    all_data.append(df_clean)
                    all_targets.append(y_clean)
                
            except Exception:
                continue
        
        if len(all_data) < self.config.min_tickers_per_industry:
            return None
        
        try:
            # Features comunes SEGURAS
            common_features = set()
            if all_data:
                common_features = set(all_data[0].columns)
                for df in all_data[1:]:
                    common_features = common_features.intersection(set(df.columns))
            
            safe_available = [col for col in common_features 
                            if col in self.config.safe_fundamental_features]
            
            if len(safe_available) < 2:
                return None
            
            # Concatenar datos
            X_combined = pd.concat([df[safe_available] for df in all_data], ignore_index=True)
            y_combined = pd.concat(all_targets, ignore_index=True)
            
            if len(X_combined) < self.config.min_samples_industry:
                return None
            
            # Seleccionar features seguras
            selected_features = self.feature_selector.select_safe_features(X_combined, y_combined, ModelType.INDUSTRY)
            if len(selected_features) < 2:
                return None
            
            X_selected = X_combined[selected_features]
            
            # Actualizar estadísticas
            for feature in selected_features:
                self.feature_usage_stats[feature] += 1
            
            # Entrenar modelo
            model = TemporalFairValueEnsemble(self.config, ModelType.INDUSTRY)
            model.fit(X_selected, y_combined)
            
            # Evaluar
            y_pred = model.predict(X_selected)
            fair_value_metrics = self.evaluator.evaluate_fair_value_quality(
                y_combined.values, y_pred, X_selected, selected_features
            )
            
            if not self.evaluator.is_fair_value_model_useful(fair_value_metrics, ModelType.INDUSTRY):
                return None
            
            # Convertir métricas
            safe_metrics = convert_to_json_serializable(fair_value_metrics)
            
            # Guardar métricas
            model_id = f"industry_{industry}"
            self.model_performance_metrics[model_id] = {
                'model_type': 'industry',
                'industry': industry,
                'n_samples': int(len(y_combined)),
                'n_features': int(len(selected_features)),
                'features': selected_features,
                'n_companies': int(len(tickers)),
                **safe_metrics
            }
            
            return {
                'model': model,
                'features': selected_features,
                'metrics': safe_metrics,
                'tickers': tickers,
                'n_samples': len(y_combined)
            }
            
        except Exception as e:
            self.logger.debug(f"Error combinando datos industria {industry}: {e}")
            return None
    
    def train_all_models(self) -> bool:
        """Entrenar todos los modelos"""
        
        self.logger.info("Iniciando entrenamiento de modelos Fair Value ML...")
        start_time = datetime.now()
        
        # 1. Entrenar modelos de industria
        self.train_industry_models()
        
        # 2. Entrenar modelos de ticker
        successful_tickers = 0
        
        self.logger.info("Entrenando modelos de ticker individuales...")
        for ticker in tqdm(self.available_tickers, desc="Entrenando tickers"):
            model_result = self.train_ticker_model(ticker)
            if model_result:
                self.ticker_models[ticker] = model_result
                successful_tickers += 1
        
        # 3. Estadísticas
        training_time = datetime.now() - start_time
        self.system_statistics = convert_to_json_serializable({
            'training_duration_seconds': training_time.total_seconds(),
            'total_tickers_attempted': len(self.available_tickers),
            'successful_ticker_models': successful_tickers,
            'success_rate_ticker': successful_tickers / len(self.available_tickers) if self.available_tickers else 0,
            'successful_industry_models': len(self.industry_models),
            'total_successful_models': successful_tickers + len(self.industry_models)
        })
        
        self.logger.info(f"Entrenamiento completado:")
        self.logger.info(f"  - Modelos de ticker: {successful_tickers}/{len(self.available_tickers)} ({self.system_statistics['success_rate_ticker']:.1%})")
        self.logger.info(f"  - Modelos de industria: {len(self.industry_models)}")
        self.logger.info(f"  - Tiempo total: {training_time}")
        
        return successful_tickers > 0 or len(self.industry_models) > 0
    
    def predict_fair_value_ranges(self, ticker: str, forecast_data: pd.DataFrame, 
                                current_shares: Optional[float] = None) -> Dict[str, Any]:
        """Generar predicciones de fair value"""
        
        result = {
            'ticker': ticker,
            'predictions_available': False,
            'ticker_predictions': {},
            'industry_predictions': {},
            'years': [],
            'confidence_info': {}
        }
        
        if forecast_data is None or forecast_data.empty:
            return result
        
        # Obtener años
        years = []
        if 'Year' in forecast_data.columns:
            years = sorted(forecast_data['Year'].dropna().unique())
        elif 'Date' in forecast_data.columns:
            dates = pd.to_datetime(forecast_data['Date'], errors='coerce')
            years = sorted(dates.dt.year.dropna().unique())
        
        if not years:
            return result
        
        result['years'] = [int(y) for y in years]
        result['predictions_available'] = True
        
        # Predicciones con modelo de ticker
        if ticker in self.ticker_models:
            model_info = self.ticker_models[ticker]
            model = model_info['model']
            features = model_info['features']
            
            for year in result['years']:
                try:
                    if 'Year' in forecast_data.columns:
                        year_data = forecast_data[forecast_data['Year'] == year]
                    else:
                        year_mask = pd.to_datetime(forecast_data['Date']).dt.year == year
                        year_data = forecast_data[year_mask]
                    
                    if len(year_data) > 0:
                        available_features = [f for f in features if f in year_data.columns]
                        if len(available_features) >= len(features) * 0.7:  # Requiere 70% de features
                            X_pred = year_data[available_features].iloc[0:1]
                            ranges = model.predict_range(X_pred)
                            
                            result['ticker_predictions'][year] = {
                                'lower': float(ranges['lower'][0]),
                                'mean': float(ranges['mean'][0]),
                                'upper': float(ranges['upper'][0])
                            }
                
                except Exception as e:
                    self.logger.debug(f"Error predicción ticker {ticker} año {year}: {e}")
                    continue
        
        # Predicciones con modelo de industria
        if ticker in self.ticker_to_industry:
            industry = self.ticker_to_industry[ticker]
            if industry in self.industry_models:
                model_info = self.industry_models[industry]
                model = model_info['model']
                features = model_info['features']
                
                for year in result['years']:
                    try:
                        if 'Year' in forecast_data.columns:
                            year_data = forecast_data[forecast_data['Year'] == year]
                        else:
                            year_mask = pd.to_datetime(forecast_data['Date']).dt.year == year
                            year_data = forecast_data[year_mask]
                        
                        if len(year_data) > 0:
                            available_features = [f for f in features if f in year_data.columns]
                            if len(available_features) >= len(features) * 0.7:
                                X_pred = year_data[available_features].iloc[0:1]
                                mc_ranges = model.predict_range(X_pred)
                                
                                if current_shares and current_shares > 0:
                                    price_ranges = {
                                        'lower': float(mc_ranges['lower'][0] / (current_shares*1000000)),
                                        'mean': float(mc_ranges['mean'][0] / (current_shares*1000000)),
                                        'upper': float(mc_ranges['upper'][0] / (current_shares*1000000))
                                    }
                                    
                                    result['industry_predictions'][year] = {
                                        'marketcap_range': {
                                            'lower': float(mc_ranges['lower'][0]),
                                            'mean': float(mc_ranges['mean'][0]),
                                            'upper': float(mc_ranges['upper'][0])
                                        },
                                        'price_range': price_ranges
                                    }
                    
                    except Exception as e:
                        self.logger.debug(f"Error predicción industria {ticker} año {year}: {e}")
                        continue
        
        # Información de confianza
        confidence_info = {}
        if ticker in self.ticker_models:
            ticker_metrics = self.ticker_models[ticker]['metrics']
            confidence_info['ticker_fair_value_score'] = ticker_metrics.get('fair_value_score', 0)
            confidence_info['ticker_capture_rate'] = ticker_metrics.get('capture_rate_moderate', 0)
        
        if ticker in self.ticker_to_industry and self.ticker_to_industry[ticker] in self.industry_models:
            industry_metrics = self.industry_models[self.ticker_to_industry[ticker]]['metrics']
            confidence_info['industry_fair_value_score'] = industry_metrics.get('fair_value_score', 0)
            confidence_info['industry_capture_rate'] = industry_metrics.get('capture_rate_moderate', 0)
        
        result['confidence_info'] = confidence_info
        
        return result
    
    def export_results(self) -> Dict[str, str]:
        """Exportar resultados con conversión JSON segura"""
        
        results = {}
        
        # 1. Métricas de performance
        if self.model_performance_metrics:
            performance_data = []
            for model_id, metrics in self.model_performance_metrics.items():
                row = {'model_id': model_id}
                # Convertir a tipos JSON-serializables
                safe_metrics = convert_to_json_serializable(metrics)
                row.update(safe_metrics)
                performance_data.append(row)
            
            performance_df = pd.DataFrame(performance_data)
            performance_path = self.OUTPUT_PATH / "ML_Model_Performance.parquet"
            performance_df.to_parquet(performance_path, index=False)
            results['performance_report'] = str(performance_path)
        
        # 2. Metadata con conversión JSON segura
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'system_version': 'ML_Fair_Value_v4.0_Anti_Leakage_Integrated',
            'config': self.config.to_dict(),
            'system_statistics': self.system_statistics,
            'model_statistics': {
                'total_ticker_models': len(self.ticker_models),
                'total_industry_models': len(self.industry_models),
                'total_models': len(self.ticker_models) + len(self.industry_models)
            }
        }
        
        # Añadir métricas promedio adicionales solicitadas
        if self.model_performance_metrics:
            r2_scores = [m.get('r2', 0) for m in self.model_performance_metrics.values() if m.get('r2', 0) > -0.5]
            directional_senses = [m.get('directional_sense', 0) for m in self.model_performance_metrics.values()]
            consistency_scores = [m.get('consistency_score', 0) for m in self.model_performance_metrics.values()]
            
            if r2_scores:
                metadata['model_statistics']['avg_r2_score'] = float(np.mean(r2_scores))
            if directional_senses:
                metadata['model_statistics']['avg_directional_sense'] = float(np.mean(directional_senses))
            if consistency_scores:
                metadata['model_statistics']['avg_consistency_score'] = float(np.mean(consistency_scores))
        
        metadata_path = self.OUTPUT_PATH / "ML_System_Metadata.json"
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            results['metadata'] = str(metadata_path)
        except Exception as e:
            self.logger.error(f"Error guardando metadata: {e}")
        
        return results

# =====================================================================================
# CLASE PRINCIPAL INTEGRADA (MANTIENE TODO DEL PRIMER CÓDIGO EXCEPTO ML)
# =====================================================================================

class ProfessionalValuationSystem:
    def __init__(self):
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        
        # Navigate to project root (up 2 levels from scr/GetResults/)
        project_root = script_dir.parent.parent
        
        # Paths using relative navigation
        self.ROOT = project_root / "data"
        self.HISTORICAL_PATH = self.ROOT / "Fundamental_Data_and_Ratios" / "Anual"
        self.PREDICTIONS_PATH = self.ROOT / "Forecast_Fundamental_Data_and_Ratios" / "Anual"
        self.SCREENER_PATH = self.ROOT / "Ticker_List" / "screener.parquet"
        self.MACRO_PATH = self.ROOT / "Macro_Data" / "Macro_Data.parquet"
        self.OUTPUT_PATH = self.ROOT / 'Results' /  "Fundamental_Data_Results.parquet"
        
        # Professional valuation parameters
        self.risk_free_rate = 0.045  # Current 10Y Treasury
        self.market_premium = 0.08   # Historical equity risk premium
        self.terminal_growth = 0.025 # GDP growth proxy
        self.confidence_levels = [0.05, 0.25, 0.50, 0.75, 0.95]  # For ranges
        self.monte_carlo_sims = 1000
        
        # Key financial metrics for valuation
        self.valuation_metrics = [
            'Free Cash Flow', 'EBITDA', 'Net Income', 'Revenue', 
            'Book Value Per Share', 'ROE - Return On Equity', 'ROIC',
            'Debt/EBITDA', 'Interest Coverage', 'Current Ratio',
            'Operating Margin', 'Net Profit Margin', 'Asset Turnover'
        ]
        
        # INTEGRACIÓN DEL SISTEMA ML AVANZADO
        self.ml_config = MLValuationConfig()
        self.ml_system = None
        
        print("=" * 80)
        print("PROFESSIONAL EQUITY VALUATION SYSTEM")
        print("=" * 80)
        print("Methodologies:")
        print("  • DCF (Discounted Cash Flow) Analysis")
        print("  • Comparable Company Analysis (Trading & Transaction Multiples)")
        print("  • Residual Income Valuation")
        print("  • ADVANCED ML FAIR VALUE SYSTEM (Anti-Data Leakage)")
        print("  • Monte Carlo Simulation for Uncertainty")
        print("  • Sector-Specific Valuation Models")
        print("=" * 80)
    
    def load_comprehensive_data(self):
        """Load all data sources with professional data quality checks"""
        print("\n1. LOADING & VALIDATING DATA INFRASTRUCTURE")
        print("-" * 50)
        
        # Load screener for company information
        try:
            self.screener_data = pd.read_parquet(self.SCREENER_PATH)
            print(f"✓ Company database: {len(self.screener_data)} companies")
            
            # Create sector/industry mappings
            self.sector_mapping = self.screener_data.groupby('zacks_x_sector_desc').apply(
                lambda x: x['ticker'].tolist()
            ).to_dict()
            
            self.industry_mapping = self.screener_data.groupby('zacks_x_ind_desc').apply(
                lambda x: x['ticker'].tolist()
            ).to_dict()
            
        except Exception as e:
            print(f"✗ Screener data failed: {e}")
            self.screener_data = pd.DataFrame()
        
        # Load macro data
        try:
            self.macro_data = pd.read_parquet(self.MACRO_PATH)
            self.macro_data['date'] = pd.to_datetime(self.macro_data['date'])
            print(f"✓ Macro data: {len(self.macro_data)} observations")
            
            # Calculate macro indicators
            self.calculate_macro_indicators()
            
        except Exception as e:
            print(f"✗ Macro data failed: {e}")
            self.macro_data = pd.DataFrame()
        
        # Inventory available tickers
        hist_files = set()
        pred_files = set()
        
        if self.HISTORICAL_PATH.exists():
            hist_files = {f.stem for f in self.HISTORICAL_PATH.glob("*.parquet")}
        if self.PREDICTIONS_PATH.exists():
            pred_files = {f.stem for f in self.PREDICTIONS_PATH.glob("*.parquet")}
        
        self.available_tickers = list(hist_files.intersection(pred_files))
        
        print(f"\n✓ Data inventory complete:")
        print(f"  Historical: {len(hist_files)} files")
        print(f"  Forecasts: {len(pred_files)} files")
        print(f"  Complete set: {len(self.available_tickers)} companies")
        
        return len(self.available_tickers) > 0
    
    def calculate_macro_indicators(self):
        """Calculate professional macro indicators"""
        if self.macro_data.empty:
            return
        
        # Economic cycle indicators
        self.macro_data['GDP_Growth'] = self.macro_data.get('GDP', 0).pct_change(4)
        self.macro_data['Inflation_Rate'] = self.macro_data.get('CPI', 0).pct_change(250)
        self.macro_data['Yield_Curve'] = self.macro_data.get('10Y_Treasury', 0) - self.macro_data.get('2Y_Treasury', 0)
        
        # Market regime indicators
        self.macro_data['VIX_MA'] = self.macro_data.get('VIX', 20).rolling(20).mean()
        self.macro_data['Credit_Spread'] = self.macro_data.get('BAA_Yield', 0) - self.macro_data.get('AAA_Yield', 0)
        
        # Latest values for valuation adjustments
        latest_macro = self.macro_data.iloc[-1] if not self.macro_data.empty else {}
        self.current_macro_regime = {
            'gdp_growth': latest_macro.get('GDP_Growth', 0.025),
            'inflation': latest_macro.get('Inflation_Rate', 0.02),
            'yield_curve': latest_macro.get('Yield_Curve', 0.015),
            'market_volatility': latest_macro.get('VIX_MA', 20),
            'credit_conditions': latest_macro.get('Credit_Spread', 0.01)
        }
    
    def load_company_financials(self, ticker):
        """Load and prepare financial data for a company"""
        try:
            # Historical financials
            hist_file = self.HISTORICAL_PATH / f"{ticker}.parquet"
            hist_df = pd.read_parquet(hist_file)
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            hist_df = hist_df.sort_values('Date')
            
            # Forecast financials
            pred_file = self.PREDICTIONS_PATH / f"{ticker}.parquet"
            pred_df = pd.read_parquet(pred_file)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            pred_df = pred_df.sort_values('Date')
            
            # Company info
            company_info = {}
            if not self.screener_data.empty:
                info = self.screener_data[self.screener_data['ticker'] == ticker]
                if not info.empty:
                    company_info = info.iloc[0].to_dict()
            
            return {
                'ticker': ticker,
                'historical': hist_df,
                'forecast': pred_df,
                'info': company_info
            }
            
        except Exception as e:
            return None
    
    def calculate_wacc(self, company_data):
        """Calculate Weighted Average Cost of Capital (WACC)"""
        hist_df = company_data['historical']
        info = company_data['info']
        
        # Get latest financials
        latest = hist_df.iloc[-1]
        
        # Market values
        market_cap = latest.get('MarketCap', 0)
        total_debt = latest.get('Long Term Debt', 0) + latest.get('Total Current Liabilities', 0) * 0.5
        
        # Beta adjustment
        beta = info.get('beta', 1.0)
        if pd.isna(beta) or beta <= 0:
            beta = 1.0
        
        # Adjusted beta (Bloomberg adjustment)
        adjusted_beta = 0.67 * beta + 0.33 * 1.0
        
        # Cost of equity (CAPM)
        cost_of_equity = self.risk_free_rate + adjusted_beta * self.market_premium
        
        # Cost of debt
        interest_expense = abs(latest.get('Total Non-Operating Income/Expense', 0))
        if total_debt > 0 and interest_expense > 0:
            cost_of_debt = interest_expense / total_debt
        else:
            # Use credit spread based estimate
            cost_of_debt = self.risk_free_rate + 0.02  # Default spread
        
        # Tax rate
        pretax_income = latest.get('Pre-Tax Income', 1)
        income_tax = latest.get('Income Taxes', 0)
        tax_rate = min(max(income_tax / pretax_income if pretax_income != 0 else 0.21, 0), 0.35)
        
        # WACC calculation
        total_value = market_cap + total_debt
        if total_value > 0:
            wacc = (market_cap / total_value) * cost_of_equity + \
                   (total_debt / total_value) * cost_of_debt * (1 - tax_rate)
        else:
            wacc = cost_of_equity
        
        # Adjust for company size (size premium)
        if market_cap < 1000000000:  # Small cap
            wacc += 0.03
        elif market_cap < 10000000000:  # Mid cap
            wacc += 0.015
        
        # Adjust for financial health
        debt_equity = latest.get('Debt/Equity Ratio', 0)
        if debt_equity > 2:
            wacc += 0.02
        elif debt_equity > 1:
            wacc += 0.01
        
        return min(max(wacc, 0.05), 0.25)  # Reasonable bounds
    
    def dcf_valuation(self, company_data):
        """Professional DCF valuation with multiple scenarios"""
        hist_df = company_data['historical']
        pred_df = company_data['forecast']
        
        # Get WACC
        wacc = self.calculate_wacc(company_data)
        
        # Extract forecast free cash flows
        fcf_forecast = []
        for i in range(min(3, len(pred_df))):
            fcf = pred_df.iloc[i].get('Free Cash Flow', 0)
            if pd.isna(fcf) or fcf == 0:
                # Estimate from other metrics
                ebitda = pred_df.iloc[i].get('EBITDA', 0)
                capex = pred_df.iloc[i].get('Net Change In Property, Plant, And Equipment', 0)
                nwc_change = pred_df.iloc[i].get('Total Change In Assets/Liabilities', 0)
                tax_rate = 0.21
                fcf = ebitda * (1 - tax_rate) - abs(capex) - nwc_change
            fcf_forecast.append(fcf)
        
        if not fcf_forecast or all(f <= 0 for f in fcf_forecast):
            return None
        
        # Terminal value calculation
        terminal_fcf = fcf_forecast[-1] * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (wacc - self.terminal_growth)
        
        # Discount cash flows
        pv_fcf = sum(fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(fcf_forecast))
        pv_terminal = terminal_value / (1 + wacc) ** len(fcf_forecast)
        
        # Enterprise value
        enterprise_value = pv_fcf + pv_terminal
        
        # Equity value
        latest = hist_df.iloc[-1]
        net_debt = latest.get('Net Debt', 0)
        if pd.isna(net_debt):
            total_debt = latest.get('Long Term Debt', 0)
            cash = latest.get('Cash On Hand', 0)
            net_debt = total_debt - cash
        
        equity_value = enterprise_value - net_debt
        shares = latest.get('Shares Outstanding', 1)
        
        if shares > 0:
            fair_value = equity_value / shares
        else:
            fair_value = 0
        
        # Sensitivity analysis
        wacc_range = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
        growth_range = [self.terminal_growth - 0.01, self.terminal_growth, self.terminal_growth + 0.01]
        
        sensitivity = {}
        for w in wacc_range:
            for g in growth_range:
                if w > g:  # Valid combination
                    tv = terminal_fcf / (w - g)
                    pv = sum(fcf / (1 + w) ** (i + 1) for i, fcf in enumerate(fcf_forecast))
                    pv_t = tv / (1 + w) ** len(fcf_forecast)
                    ev = pv + pv_t
                    price = (ev - net_debt) / shares if shares > 0 else 0
                    sensitivity[f'wacc_{w:.3f}_growth_{g:.3f}'] = price
        
        return {
            'method': 'DCF',
            'fair_value': fair_value,
            'enterprise_value': enterprise_value,
            'wacc_used': wacc,
            'sensitivity': sensitivity
        }
    
    def comparable_valuation(self, company_data):
        """Comparable company analysis using peers - ENHANCED DEBUG VERSION WITH CORRECT NAMES"""
        ticker = company_data['ticker']
        hist_df = company_data['historical']
        info = company_data['info']
        
        print(f"DEBUG: Starting comparable valuation for {ticker}")
        
        # Get peer companies
        sector = info.get('zacks_x_sector_desc', '')
        industry = info.get('zacks_x_ind_desc', '')
        
        print(f"DEBUG: {ticker} - Sector: {sector}, Industry: {industry}")
        
        peers = []
        if industry and industry in self.industry_mapping:
            peers = [p for p in self.industry_mapping[industry] if p != ticker][:20]
            print(f"DEBUG: Found {len(peers)} industry peers for {ticker}")
        elif sector and sector in self.sector_mapping:
            peers = [p for p in self.sector_mapping[sector] if p != ticker][:30]
            print(f"DEBUG: Found {len(peers)} sector peers for {ticker}")
        
        if not peers:
            # Use statistical similarity
            peers = self.find_statistical_peers(company_data)
            print(f"DEBUG: Found {len(peers)} statistical peers for {ticker}")
        
        if not peers:
            print(f"DEBUG: No peers found for {ticker}")
            return None
        
        # Collect peer multiples - FIXED: usar nombres exactos que espera el dashboard
        peer_multiples = {
            'P/E': [],           # Nombres exactos del dashboard
            'EV/EBITDA': [],     
            'P/B': [],           
            'P/S': [],           
            'EV/Sales': []       
        }
        
        peers_processed = 0
        for peer in peers:
            try:
                peer_file = self.HISTORICAL_PATH / f"{peer}.parquet"
                if not peer_file.exists():
                    continue
                    
                peer_hist = pd.read_parquet(peer_file)
                if peer_hist.empty:
                    continue
                    
                peers_processed += 1
                latest = peer_hist.iloc[-1]
                
                # P/E
                pe = latest.get('P/E', np.nan)
                if not pd.isna(pe) and 0 < pe < 100:
                    peer_multiples['P/E'].append(pe)
                
                # EV/EBITDA
                ev_ebitda = latest.get('EV/EBITDA', np.nan)
                if not pd.isna(ev_ebitda) and 0 < ev_ebitda < 50:
                    peer_multiples['EV/EBITDA'].append(ev_ebitda)
                
                # P/B
                pb = latest.get('P/B', np.nan)
                if not pd.isna(pb) and 0 < pb < 20:
                    peer_multiples['P/B'].append(pb)
                
                # P/S
                ps = latest.get('P/S', np.nan)
                if not pd.isna(ps) and 0 < ps < 20:
                    peer_multiples['P/S'].append(ps)
                
                # EV/Sales
                ev_sales = latest.get('EV/Sales', np.nan)
                if not pd.isna(ev_sales) and 0 < ev_sales < 20:
                    peer_multiples['EV/Sales'].append(ev_sales)
                    
            except Exception as e:
                print(f"DEBUG: Error processing peer {peer}: {e}")
                continue
        
        print(f"DEBUG: {ticker} - Processed {peers_processed} peers")
        for multiple, values in peer_multiples.items():
            print(f"DEBUG: {ticker} - {multiple}: {len(values)} values")
        
        # Calculate target multiples (using median and IQR)
        valuations = {}
        latest = hist_df.iloc[-1]
        shares = latest.get('Shares Outstanding', 1)
        
        multiples_calculated = 0
        for multiple, values in peer_multiples.items():
            if len(values) >= 3:  # Need at least 3 peers
                median_multiple = np.median(values)
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                
                print(f"DEBUG: {ticker} - Calculating {multiple} with {len(values)} peers, median: {median_multiple:.2f}")
                
                if multiple == 'P/E':
                    eps = latest.get('EPS - Earnings Per Share', 0)
                    if eps > 0:
                        valuations['P/E'] = {  # FIXED: usar nombre exacto del dashboard
                            'median': median_multiple * eps,
                            'low': q1 * eps,
                            'high': q3 * eps,
                            'multiple': median_multiple
                        }
                        multiples_calculated += 1
                        print(f"DEBUG: {ticker} - P/E calculated: {median_multiple * eps:.2f}")
                
                elif multiple == 'P/B':
                    book_value = latest.get('Book Value Per Share', 0)
                    if book_value > 0:
                        valuations['P/B'] = {  # FIXED: usar nombre exacto del dashboard
                            'median': median_multiple * book_value,
                            'low': q1 * book_value,
                            'high': q3 * book_value,
                            'multiple': median_multiple
                        }
                        multiples_calculated += 1
                        print(f"DEBUG: {ticker} - P/B calculated: {median_multiple * book_value:.2f}")
                
                elif multiple == 'P/S':
                    revenue_per_share = latest.get('Revenue/Share', 0)
                    if revenue_per_share == 0 and 'Revenue' in latest and shares > 0:
                        revenue_per_share = latest['Revenue'] / shares
                    if revenue_per_share > 0:
                        valuations['P/S'] = {  # FIXED: usar nombre exacto del dashboard
                            'median': median_multiple * revenue_per_share,
                            'low': q1 * revenue_per_share,
                            'high': q3 * revenue_per_share,
                            'multiple': median_multiple
                        }
                        multiples_calculated += 1
                        print(f"DEBUG: {ticker} - P/S calculated: {median_multiple * revenue_per_share:.2f}")
                
                elif multiple == 'EV/EBITDA':
                    ebitda = latest.get('EBITDA', 0)
                    net_debt = latest.get('Net Debt', 0)
                    if pd.isna(net_debt):
                        total_debt = latest.get('Long Term Debt', 0)
                        cash = latest.get('Cash On Hand', 0)
                        net_debt = total_debt - cash
                    
                    if ebitda > 0 and shares > 0:
                        ev = median_multiple * ebitda
                        equity_value = ev - net_debt
                        
                        ev_low = q1 * ebitda
                        ev_high = q3 * ebitda
                        
                        valuations['EV/EBITDA'] = {  # FIXED: usar nombre exacto del dashboard
                            'median': equity_value / shares,
                            'low': (ev_low - net_debt) / shares,
                            'high': (ev_high - net_debt) / shares,
                            'multiple': median_multiple
                        }
                        multiples_calculated += 1
                        print(f"DEBUG: {ticker} - EV/EBITDA calculated: {equity_value / shares:.2f}")
                
                elif multiple == 'EV/Sales':
                    revenue = latest.get('Revenue', 0)
                    net_debt = latest.get('Net Debt', 0)
                    if pd.isna(net_debt):
                        total_debt = latest.get('Long Term Debt', 0)
                        cash = latest.get('Cash On Hand', 0)
                        net_debt = total_debt - cash
                    
                    if revenue > 0 and shares > 0:
                        ev = median_multiple * revenue
                        equity_value = ev - net_debt
                        
                        ev_low = q1 * revenue
                        ev_high = q3 * revenue
                        
                        valuations['EV/Sales'] = {  # FIXED: usar nombre exacto del dashboard
                            'median': equity_value / shares,
                            'low': (ev_low - net_debt) / shares,
                            'high': (ev_high - net_debt) / shares,
                            'multiple': median_multiple
                        }
                        multiples_calculated += 1
                        print(f"DEBUG: {ticker} - EV/Sales calculated: {equity_value / shares:.2f}")
        
        print(f"DEBUG: {ticker} - Total multiples calculated: {multiples_calculated}")
        
        if multiples_calculated == 0:
            print(f"DEBUG: No multiples calculated for {ticker}")
            return None
        
        result = {
            'method': 'Comparables',
            'valuations': valuations,
            'peer_count': peers_processed
        }
        
        print(f"DEBUG: {ticker} - Returning comparables result with {len(valuations)} valuations")
        return result
    
    def find_statistical_peers(self, company_data, n_peers=20):
        """Find peers using statistical characteristics"""
        ticker = company_data['ticker']
        hist_df = company_data['historical']
        latest = hist_df.iloc[-1]
        
        # Key characteristics for similarity
        characteristics = {
            'size': np.log1p(latest.get('MarketCap', 0)),
            'profitability': latest.get('ROE - Return On Equity', 0),
            'growth': latest.get('Revenue_Growth_1Y', 0),
            'leverage': latest.get('Debt/Equity Ratio', 0),
            'margin': latest.get('Operating Margin', 0)
        }
        
        similar_companies = []
        
        for other_ticker in self.available_tickers[:200]:  # Limit for performance
            if other_ticker == ticker:
                continue
            
            try:
                other_hist = pd.read_parquet(self.HISTORICAL_PATH / f"{other_ticker}.parquet")
                if not other_hist.empty:
                    other_latest = other_hist.iloc[-1]
                    
                    other_chars = {
                        'size': np.log1p(other_latest.get('MarketCap', 0)),
                        'profitability': other_latest.get('ROE - Return On Equity', 0),
                        'growth': other_latest.get('Revenue_Growth_1Y', 0),
                        'leverage': other_latest.get('Debt/Equity Ratio', 0),
                        'margin': other_latest.get('Operating Margin', 0)
                    }
                    
                    # Calculate distance
                    distance = sum((characteristics[k] - other_chars[k]) ** 2 
                                 for k in characteristics.keys())
                    
                    similar_companies.append((other_ticker, distance))
            except:
                continue
        
        # Sort by similarity and return top N
        similar_companies.sort(key=lambda x: x[1])
        return [ticker for ticker, _ in similar_companies[:n_peers]]
    
    def residual_income_valuation(self, company_data):
        """Residual Income Model valuation"""
        hist_df = company_data['historical']
        pred_df = company_data['forecast']
        
        latest = hist_df.iloc[-1]
        book_value = latest.get('Book Value Per Share', 0)
        
        if book_value <= 0:
            return None
        
        # Cost of equity
        info = company_data['info']
        beta = info.get('beta', 1.0)
        if pd.isna(beta) or beta <= 0:
            beta = 1.0
        
        cost_of_equity = self.risk_free_rate + beta * self.market_premium
        
        # Calculate residual income for forecast years
        residual_incomes = []
        current_book = book_value
        
        for i in range(min(3, len(pred_df))):
            forecast = pred_df.iloc[i]
            
            # Net income per share
            net_income = forecast.get('Net Income', 0)
            shares = forecast.get('Shares Outstanding', latest.get('Shares Outstanding', 1))
            eps = net_income / shares if shares > 0 else 0
            
            # Residual income
            residual_income = eps - (cost_of_equity * current_book)
            residual_incomes.append(residual_income)
            
            # Update book value (assuming retention)
            retention_ratio = 0.6  # Typical retention
            current_book += eps * retention_ratio
        
        # Terminal value
        terminal_ri = residual_incomes[-1] * (1 + self.terminal_growth)
        terminal_value = terminal_ri / (cost_of_equity - self.terminal_growth)
        
        # Present value of residual incomes
        pv_ri = sum(ri / (1 + cost_of_equity) ** (i + 1) 
                   for i, ri in enumerate(residual_incomes))
        pv_terminal = terminal_value / (1 + cost_of_equity) ** len(residual_incomes)
        
        # Fair value
        fair_value = book_value + pv_ri + pv_terminal
        
        return {
            'method': 'Residual_Income',
            'fair_value': fair_value,
            'book_value': book_value,
            'cost_of_equity': cost_of_equity
        }
    
    def monte_carlo_valuation(self, company_data, n_simulations=1000):
        """Monte Carlo simulation for valuation ranges"""
        valuations = []
        
        # Get base case valuations
        dcf_result = self.dcf_valuation(company_data)
        ri_result = self.residual_income_valuation(company_data)
        comp_result = self.comparable_valuation(company_data)
        
        base_values = []
        if dcf_result:
            base_values.append(dcf_result['fair_value'])
        if ri_result:
            base_values.append(ri_result['fair_value'])
        if comp_result and comp_result['valuations']:
            for method_values in comp_result['valuations'].values():
                if 'median' in method_values:
                    base_values.append(method_values['median'])
        
        if not base_values:
            return None
        
        base_value = np.median(base_values)
        
        # Estimate uncertainty based on historical volatility
        hist_df = company_data['historical']
        returns = hist_df['Close'].pct_change().dropna() if 'Close' in hist_df.columns else pd.Series([0.2])
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.3
        
        # Run simulations
        for _ in range(n_simulations):
            # Random factors
            macro_factor = np.random.normal(1.0, 0.1)  # Macro environment
            company_factor = np.random.normal(1.0, volatility/2)  # Company specific
            model_factor = np.random.normal(1.0, 0.05)  # Model uncertainty
            
            # Combine factors
            total_factor = macro_factor * company_factor * model_factor
            
            # Apply to base value
            simulated_value = base_value * total_factor
            valuations.append(max(simulated_value, 0))
        
        # Calculate percentiles
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[f'P{p}'] = np.percentile(valuations, p)
        
        return {
            'method': 'Monte_Carlo',
            'simulations': n_simulations,
            'mean': np.mean(valuations),
            'std': np.std(valuations),
            'percentiles': percentiles,
            'confidence_interval_90': (percentiles['P5'], percentiles['P95']),
            'confidence_interval_50': (percentiles['P25'], percentiles['P75'])
        }
    
    # =====================================================================================
    # INTEGRACIÓN DEL SISTEMA ML AVANZADO COMO REEMPLAZO DEL ML ORIGINAL
    # =====================================================================================
    
    def initialize_advanced_ml_system(self):
        """Inicializar el sistema ML avanzado con anti-data leakage"""
        print("\n2. INITIALIZING ADVANCED ML FAIR VALUE SYSTEM")
        print("-" * 50)
        
        # Configurar paths para el sistema ML
        ml_paths = {
            'root': str(self.ROOT),
            'historical': str(self.HISTORICAL_PATH),
            'forecast': str(self.PREDICTIONS_PATH),
            'screener': str(self.SCREENER_PATH),
            'output': str(self.ROOT / 'Results')
        }
        
        # Inicializar sistema ML con configuración personalizada
        self.ml_system = MLFairValueSystem(self.ml_config, ml_paths)
        
        # Cargar datos en el sistema ML
        if self.ml_system.load_data():
            print("✓ ML system data loaded successfully")
            
            # Entrenar modelos ML
            if self.ml_system.train_all_models():
                print("✓ Advanced ML models trained successfully")
                return True
            else:
                print("✗ ML model training failed")
                return False
        else:
            print("✗ ML system data loading failed")
            return False
    
    def get_ml_fair_value_predictions(self, ticker, forecast_data, current_shares=None):
        """Obtener predicciones ML del sistema avanzado"""
        if not self.ml_system:
            return None
        
        try:
            # Obtener predicciones del sistema ML avanzado
            ml_predictions = self.ml_system.predict_fair_value_ranges(
                ticker, forecast_data, current_shares
            )
            
            if not ml_predictions['predictions_available']:
                return None
            
            # Convertir a formato compatible con el sistema original
            ml_results = {
                'method': 'Advanced_ML_Fair_Value',
                'predictions_by_year': ml_predictions,
                'confidence_info': ml_predictions['confidence_info']
            }
            
            # Generar valores consolidados para integración
            ticker_values = []
            industry_values = []
            
            # Recopilar valores de todos los años
            for year in ml_predictions['years']:
                if year in ml_predictions['ticker_predictions']:
                    pred = ml_predictions['ticker_predictions'][year]
                    ticker_values.extend([pred['lower'], pred['mean'], pred['upper']])
                
                if year in ml_predictions['industry_predictions']:
                    if 'price_range' in ml_predictions['industry_predictions'][year]:
                        pred = ml_predictions['industry_predictions'][year]['price_range']
                        industry_values.extend([pred['lower'], pred['mean'], pred['upper']])
            
            # Calcular valores consolidados
            all_ml_values = ticker_values + industry_values
            
            if all_ml_values:
                # Mapear a bear_case, base_case, bull_case como solicitado
                ml_results['consensus_fair_value'] = np.median(all_ml_values)
                ml_results['consensus_range_lower'] = np.percentile(all_ml_values, 25)
                ml_results['consensus_range_upper'] = np.percentile(all_ml_values, 75)
                
                # Mapeo específico solicitado por el usuario
                ml_results['bear_case'] = ml_results['consensus_range_lower']
                ml_results['base_case'] = ml_results['consensus_fair_value'] 
                ml_results['bull_case'] = ml_results['consensus_range_upper']
                
                # Métricas de confianza
                conf_info = ml_predictions['confidence_info']
                ml_results['ticker_confidence'] = conf_info.get('ticker_fair_value_score', 0)
                ml_results['industry_confidence'] = conf_info.get('industry_fair_value_score', 0)
                ml_results['overall_confidence'] = np.mean([
                    conf_info.get('ticker_fair_value_score', 0),
                    conf_info.get('industry_fair_value_score', 0)
                ])
            else:
                # Si no hay valores ML válidos, retornar valores por defecto
                latest_price = forecast_data['Close'].iloc[-1] if 'Close' in forecast_data.columns else 50.0
                ml_results['consensus_fair_value'] = latest_price
                ml_results['consensus_range_lower'] = latest_price * 0.8
                ml_results['consensus_range_upper'] = latest_price * 1.2
                ml_results['bear_case'] = latest_price * 0.8
                ml_results['base_case'] = latest_price
                ml_results['bull_case'] = latest_price * 1.2
                ml_results['ticker_confidence'] = 0.0
                ml_results['industry_confidence'] = 0.0
                ml_results['overall_confidence'] = 0.0
            
            return ml_results
            
        except Exception as e:
            print(f"Error in ML predictions for {ticker}: {e}")
            return None
    
    def comprehensive_valuation(self, ticker):
        """Perform comprehensive valuation using all methods INCLUDING advanced ML"""
        company_data = self.load_company_financials(ticker)
        
        if not company_data:
            return None
        
        results = {
            'ticker': ticker,
            'company_name': company_data['info'].get('comp_name_2', ticker),
            'sector': company_data['info'].get('zacks_x_sector_desc', 'Unknown'),
            'industry': company_data['info'].get('zacks_x_ind_desc', 'Unknown'),
            'current_price': company_data['historical']['Close'].iloc[-1] if 'Close' in company_data['historical'].columns else 0
        }
        
        # 1. DCF Valuation
        dcf_result = self.dcf_valuation(company_data)
        if dcf_result:
            results['dcf'] = dcf_result
        
        # 2. Comparable Valuation
        comp_result = self.comparable_valuation(company_data)
        if comp_result:
            results['comparables'] = comp_result
        
        # 3. Residual Income Valuation
        ri_result = self.residual_income_valuation(company_data)
        if ri_result:
            results['residual_income'] = ri_result
        
        # 4. ADVANCED ML VALUATION (REEMPLAZA AL ML ORIGINAL)
        if self.ml_system:
            try:
                # Obtener shares outstanding actuales
                latest = company_data['historical'].iloc[-1]
                current_shares = latest.get('Shares Outstanding', None)
                if pd.isna(current_shares) or current_shares <= 0:
                    current_shares = latest.get('Basic Shares Outstanding', None)
                
                # Obtener predicciones ML avanzadas
                ml_result = self.get_ml_fair_value_predictions(
                    ticker, company_data['forecast'], current_shares
                )
                
                if ml_result:
                    results['ml_valuations'] = {
                        'advanced_ml_system': ml_result['base_case'],
                        'ml_bear_case': ml_result['bear_case'],
                        'ml_base_case': ml_result['base_case'],
                        'ml_bull_case': ml_result['bull_case'],
                        'ml_confidence': ml_result['overall_confidence'],
                        'full_ml_data': ml_result  # Para auditabilidad
                    }
            except Exception as e:
                print(f"Advanced ML valuation failed for {ticker}: {e}")
        
        # 5. Monte Carlo Simulation
        mc_result = self.monte_carlo_valuation(company_data, self.monte_carlo_sims)
        if mc_result:
            results['monte_carlo'] = mc_result
        
        # 6. Synthesize final valuation range (MODIFICADO para incluir ML avanzado)
        all_values = []
        
        if 'dcf' in results:
            all_values.append(results['dcf']['fair_value'])
        
        if 'comparables' in results:
            for method_values in results['comparables']['valuations'].values():
                if 'median' in method_values:
                    all_values.append(method_values['median'])
        
        if 'residual_income' in results:
            all_values.append(results['residual_income']['fair_value'])
        
        # INCLUIR VALORES ML AVANZADOS EN LA SÍNTESIS
        if 'ml_valuations' in results:
            ml_vals = results['ml_valuations']
            all_values.extend([
                ml_vals['ml_bear_case'],
                ml_vals['ml_base_case'], 
                ml_vals['ml_bull_case']
            ])
        
        if all_values:
            # Remove outliers using IQR
            q1 = np.percentile(all_values, 25)
            q3 = np.percentile(all_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            clean_values = [v for v in all_values if lower_bound <= v <= upper_bound]
            
            if clean_values:
                results['synthesis'] = {
                    'fair_value_mean': np.mean(clean_values),
                    'fair_value_median': np.median(clean_values),
                    'value_range': {
                        'bear_case': np.percentile(clean_values, 20),
                        'base_case': np.median(clean_values),
                        'bull_case': np.percentile(clean_values, 80)
                    },
                    'confidence_level': len(clean_values) / len(all_values) if all_values else 0,
                    'upside_potential': ((np.median(clean_values) / results['current_price']) - 1) * 100 if results['current_price'] > 0 else 0
                }
        
        return results
    
    def generate_forecast_valuations(self):
        """Generate valuations for all companies with forecasts - WITH PARALLELIZATION"""
        print("\n3. GENERATING COMPREHENSIVE VALUATIONS WITH ADVANCED ML + PARALLELIZATION")
        print("-" * 50)
        
        # Inicializar sistema ML avanzado
        if not self.initialize_advanced_ml_system():
            print("⚠ Advanced ML system failed to initialize, continuing with traditional methods only")
        
        all_valuations = []
        
        # PARALELIZACIÓN AÑADIDA - Usar ThreadPoolExecutor para paralelizar el procesamiento
        total_tickers = len(self.available_tickers)
        print(f"Processing {total_tickers} tickers with parallelization...")
        
        processed = 0
        errors = 0
        
        # Determinar número óptimo de threads
        n_workers = min(multiprocessing.cpu_count(), 8)  # Max 8 threads para evitar saturar I/O
        print(f"Using {n_workers} parallel workers")
        
        # Procesar en lotes para manejar memoria
        batch_size = 500  # Procesar 500 tickers por lote
        
        for i in range(0, total_tickers, batch_size):
            batch_tickers = self.available_tickers[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_tickers-1)//batch_size + 1}")
            
            # Usar ThreadPoolExecutor para paralelización
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Enviar trabajos
                future_to_ticker = {
                    executor.submit(self._robust_comprehensive_valuation, ticker): ticker 
                    for ticker in batch_tickers
                }
                
                # Recopilar resultados
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result(timeout=120)  # 2 minutos timeout por ticker
                        
                        if result:
                            all_valuations.append(result)
                            processed += 1
                        else:
                            errors += 1
                        
                        # Progress indicator
                        if (processed + errors) % 50 == 0:
                            progress = ((processed + errors) / total_tickers) * 100
                            print(f"Progress: {processed + errors}/{total_tickers} ({progress:.1f}%) - Successful: {processed}")
                        
                    except Exception as e:
                        errors += 1
                        print(f"⚠ Error processing {ticker}: {str(e)[:50]}...")
                        continue
            
            # Memory cleanup after each batch
            gc.collect()
        
        print(f"\n✓ Parallel processing completed:")
        print(f"  - Successfully processed: {processed}")
        print(f"  - Errors/Skipped: {errors}")
        print(f"  - Total processed: {processed + errors}")
        
        if not all_valuations:
            print("⚠ No valuations were generated successfully")
            return False
        
        # Create summary DataFrame
        print("Creating summary DataFrame...")
        summary_data = []
        
        for val in all_valuations:
            try:
                row = {
                    'Ticker': val.get('ticker', 'Unknown'),
                    'Company': val.get('company_name', 'Unknown'),
                    'Sector': val.get('sector', 'Unknown'), 
                    'Industry': val.get('industry', 'Unknown'),
                    'Current_Price': self._safe_get_numeric_value(val.get('current_price', 0))
                }
                
                # Add valuation methods safely
                if 'dcf' in val:
                    row['DCF_Value'] = self._safe_get_numeric_value(val['dcf'].get('fair_value', 0))
                
                # COMPARABLES - FIXED: PRECIOS OBJETIVO, NO MÚLTIPLOS
                if 'comparables' in val and isinstance(val['comparables'], dict):
                    print(f"DEBUG: Processing comparables for {val.get('ticker', 'Unknown')}")
                    comp_data = val['comparables'].get('valuations', {})
                    print(f"DEBUG: Comparable data keys: {list(comp_data.keys()) if comp_data else 'None'}")
                    
                    if isinstance(comp_data, dict) and comp_data:
                        comp_values = []  # Para valoraciones (precios objetivo)
                        
                        # FIXED: Nombres exactos que espera el Dashboard (con los PRECIOS OBJETIVO)
                        name_mapping = {
                            'P/E': 'Comp_P/E',
                            'P/B': 'Comp_P/B', 
                            'P/S': 'Comp_P/S',
                            'EV/EBITDA': 'Comp_EV/EBITDA',
                            'EV/Sales': 'Comp_EV/Sales'
                        }
                        
                        # Extraer PRECIOS OBJETIVO (no múltiplos) para el Dashboard
                        for multiple_name, multiple_data in comp_data.items():
                            print(f"DEBUG: Processing multiple {multiple_name}: {multiple_data}")
                            if isinstance(multiple_data, dict) and multiple_name in name_mapping:
                                column_name = name_mapping[multiple_name]
                                
                                # FIXED: Guardar el PRECIO OBJETIVO (median), no el múltiplo
                                if 'median' in multiple_data:
                                    target_price = self._safe_get_numeric_value(multiple_data['median'])
                                    print(f"DEBUG: Target price for {multiple_name}: {target_price}")
                                    if target_price > 0:
                                        row[column_name] = target_price  # Comp_P/E = precio objetivo, no múltiplo
                                        comp_values.append(target_price)
                                        print(f"DEBUG: Added {column_name} = {target_price} (target price)")
                                
                                # OPCIONAL: También guardar el múltiplo si se necesita para referencia
                                if 'multiple' in multiple_data:
                                    multiple_val = self._safe_get_numeric_value(multiple_data['multiple'])
                                    if multiple_val > 0:
                                        row[f'{column_name}_Multiple'] = multiple_val  # Para referencia
                                        print(f"DEBUG: Added {column_name}_Multiple = {multiple_val} (for reference)")
                        
                        # Promedio de todas las valoraciones válidas (precios objetivo)
                        if comp_values:
                            row['Comp_Average'] = np.mean(comp_values)
                            row['Comp_Count'] = len(comp_values)
                            print(f"DEBUG: Added Comp_Average = {np.mean(comp_values)} with {len(comp_values)} target prices")
                        else:
                            print(f"DEBUG: No target prices found")
                        
                        # Información adicional sobre peers
                        peer_count = val['comparables'].get('peer_count', 0)
                        if peer_count > 0:
                            row['Peer_Count'] = peer_count
                            print(f"DEBUG: Added Peer_Count = {peer_count}")
                    else:
                        print(f"DEBUG: No valid comp_data")
                else:
                    print(f"DEBUG: No comparables found for {val.get('ticker', 'Unknown')}")
                
                if 'residual_income' in val:
                    row['RI_Value'] = self._safe_get_numeric_value(val['residual_income'].get('fair_value', 0))
                
                # ML results
                if 'ml_valuations' in val:
                    ml_vals = val['ml_valuations']
                    row['ML_Advanced'] = self._safe_get_numeric_value(ml_vals.get('ml_base_case', 0))
                    row['ML_Bear'] = self._safe_get_numeric_value(ml_vals.get('ml_bear_case', 0))
                    row['ML_Bull'] = self._safe_get_numeric_value(ml_vals.get('ml_bull_case', 0))
                    row['ML_Confidence'] = self._safe_get_numeric_value(ml_vals.get('ml_confidence', 0))
                
                if 'monte_carlo' in val:
                    mc = val['monte_carlo']
                    if isinstance(mc, dict):
                        row['MC_Mean'] = self._safe_get_numeric_value(mc.get('mean', 0))
                        percentiles = mc.get('percentiles', {})
                        if isinstance(percentiles, dict):
                            row['MC_P10'] = self._safe_get_numeric_value(percentiles.get('P10', 0))
                            row['MC_P50'] = self._safe_get_numeric_value(percentiles.get('P50', 0))
                            row['MC_P90'] = self._safe_get_numeric_value(percentiles.get('P90', 0))
                
                if 'synthesis' in val:
                    synth = val['synthesis']
                    if isinstance(synth, dict):
                        row['Fair_Value'] = self._safe_get_numeric_value(synth.get('fair_value_median', 0))
                        value_range = synth.get('value_range', {})
                        if isinstance(value_range, dict):
                            row['Bear_Case'] = self._safe_get_numeric_value(value_range.get('bear_case', 0))
                            row['Base_Case'] = self._safe_get_numeric_value(value_range.get('base_case', 0))
                            row['Bull_Case'] = self._safe_get_numeric_value(value_range.get('bull_case', 0))
                        
                        row['Upside_%'] = self._safe_get_numeric_value(synth.get('upside_potential', 0))
                        row['Confidence'] = self._safe_get_numeric_value(synth.get('confidence_level', 0))
                
                summary_data.append(row)
                
            except Exception as e:
                print(f"⚠ Error creating summary for {val.get('ticker', 'Unknown')}: {str(e)[:30]}")
                continue
        
        if summary_data:
            self.valuation_summary = pd.DataFrame(summary_data)
            self.full_valuations = all_valuations
            print(f"✓ Summary DataFrame created with {len(summary_data)} companies")
            return True
        else:
            print("⚠ Failed to create summary DataFrame")
            return False
    
    def _robust_comprehensive_valuation(self, ticker):
        """Robust comprehensive valuation with data type safety and parallelization support"""
        try:
            company_data = self.load_company_financials(ticker)
            
            if not company_data or not self._validate_company_data(company_data):
                return None
            
            results = {
                'ticker': ticker,
                'company_name': company_data['info'].get('comp_name_2', ticker),
                'sector': company_data['info'].get('zacks_x_sector_desc', 'Unknown'),
                'industry': company_data['info'].get('zacks_x_ind_desc', 'Unknown'),
                'current_price': self._safe_get_numeric(company_data['historical'], 'Close', default=0.0)
            }
            
            # 1. DCF Valuation with error handling
            try:
                dcf_result = self.dcf_valuation(company_data)
                if dcf_result and self._validate_valuation_result(dcf_result):
                    results['dcf'] = dcf_result
            except Exception as e:
                pass  # Skip DCF if it fails
            
            # 2. Comparable Valuation with error handling - ENHANCED DEBUG
            try:
                comp_result = self.comparable_valuation(company_data)
                print(f"DEBUG: {ticker} - Comparable result: {comp_result is not None}")
                if comp_result:
                    print(f"DEBUG: {ticker} - Comparable has {len(comp_result.get('valuations', {}))} valuations")
                if comp_result and self._validate_valuation_result(comp_result):
                    results['comparables'] = comp_result
                    print(f"DEBUG: {ticker} - Comparable result ACCEPTED")
                else:
                    print(f"DEBUG: {ticker} - Comparable result REJECTED")
            except Exception as e:
                print(f"DEBUG: {ticker} - Comparable exception: {e}")
                pass  # Skip comparables if it fails
            
            # 3. Residual Income Valuation with error handling
            try:
                ri_result = self.residual_income_valuation(company_data)
                if ri_result and self._validate_valuation_result(ri_result):
                    results['residual_income'] = ri_result
            except Exception as e:
                pass  # Skip RI if it fails
            
            # 4. ML Valuation with error handling
            if self.ml_system:
                try:
                    latest = company_data['historical'].iloc[-1]
                    current_shares = self._safe_get_numeric_value(latest.get('Shares Outstanding', None))
                    if not current_shares or current_shares <= 0:
                        current_shares = self._safe_get_numeric_value(latest.get('Basic Shares Outstanding', None))
                    
                    ml_result = self.get_ml_fair_value_predictions(
                        ticker, company_data['forecast'], current_shares
                    )
                    
                    if ml_result and self._validate_ml_result(ml_result):
                        results['ml_valuations'] = {
                            'advanced_ml_system': ml_result.get('base_case', 0),
                            'ml_bear_case': ml_result.get('bear_case', 0),
                            'ml_base_case': ml_result.get('base_case', 0),
                            'ml_bull_case': ml_result.get('bull_case', 0),
                            'ml_confidence': ml_result.get('overall_confidence', 0),
                            'full_ml_data': ml_result
                        }
                except Exception as e:
                    pass  # Skip ML if it fails
            
            # 5. Monte Carlo with error handling
            try:
                mc_result = self.monte_carlo_valuation(company_data, self.monte_carlo_sims)
                if mc_result and self._validate_valuation_result(mc_result):
                    results['monte_carlo'] = mc_result
            except Exception as e:
                pass  # Skip MC if it fails
            
            # 6. Synthesize final valuation range
            try:
                all_values = []
                
                if 'dcf' in results:
                    val = self._safe_get_numeric_value(results['dcf'].get('fair_value'))
                    if val and val > 0:
                        all_values.append(val)
                
                if 'comparables' in results:
                    for method_values in results['comparables'].get('valuations', {}).values():
                        val = self._safe_get_numeric_value(method_values.get('median'))
                        if val and val > 0:
                            all_values.append(val)
                
                if 'residual_income' in results:
                    val = self._safe_get_numeric_value(results['residual_income'].get('fair_value'))
                    if val and val > 0:
                        all_values.append(val)
                
                if 'ml_valuations' in results:
                    ml_vals = results['ml_valuations']
                    for key in ['ml_bear_case', 'ml_base_case', 'ml_bull_case']:
                        val = self._safe_get_numeric_value(ml_vals.get(key))
                        if val and val > 0:
                            all_values.append(val)
                
                if len(all_values) >= 2:
                    # Remove outliers using IQR
                    q1 = np.percentile(all_values, 25)
                    q3 = np.percentile(all_values, 75)
                    iqr = q3 - q1
                    lower_bound = max(q1 - 1.5 * iqr, 0)  # Ensure positive
                    upper_bound = q3 + 1.5 * iqr
                    
                    clean_values = [v for v in all_values if lower_bound <= v <= upper_bound]
                    
                    if clean_values:
                        current_price = results.get('current_price', 0)
                        median_value = np.median(clean_values)
                        
                        results['synthesis'] = {
                            'fair_value_mean': float(np.mean(clean_values)),
                            'fair_value_median': float(median_value),
                            'value_range': {
                                'bear_case': float(np.percentile(clean_values, 20)),
                                'base_case': float(median_value),
                                'bull_case': float(np.percentile(clean_values, 80))
                            },
                            'confidence_level': len(clean_values) / len(all_values),
                            'upside_potential': float(((median_value / current_price) - 1) * 100) if current_price > 0 else 0.0
                        }
            except Exception as e:
                pass  # Skip synthesis if it fails
            
            return results if len(results) > 5 else None  # Only return if we have substantial data
            
        except Exception as e:
            return None
    
    def _validate_company_data(self, company_data):
        """Validate company data structure and content"""
        try:
            if not company_data:
                return False
            
            required_keys = ['ticker', 'historical', 'forecast', 'info']
            if not all(key in company_data for key in required_keys):
                return False
            
            hist_df = company_data['historical']
            if hist_df is None or hist_df.empty:
                return False
            
            return True
        except:
            return False
    
    def _validate_valuation_result(self, result):
        """Validate valuation result - FIXED to accept comparables"""
        try:
            if not result or not isinstance(result, dict):
                return False
            
            # Check for fair_value or similar numeric result
            numeric_keys = ['fair_value', 'mean', 'median']
            for key in numeric_keys:
                if key in result:
                    val = self._safe_get_numeric_value(result[key])
                    if val and val > 0 and np.isfinite(val):
                        return True
            
            # FIXED: Special validation for comparables
            if result.get('method') == 'Comparables':
                valuations = result.get('valuations', {})
                if isinstance(valuations, dict) and len(valuations) > 0:
                    # If we have at least one valuation, it's valid
                    for multiple_data in valuations.values():
                        if isinstance(multiple_data, dict) and 'median' in multiple_data:
                            return True
            
            return False
        except:
            return False
    
    def _validate_ml_result(self, ml_result):
        """Validate ML result"""
        try:
            if not ml_result or not isinstance(ml_result, dict):
                return False
            
            # Check for required keys
            required_keys = ['base_case', 'bear_case', 'bull_case']
            for key in required_keys:
                val = self._safe_get_numeric_value(ml_result.get(key))
                if not val or val <= 0 or not np.isfinite(val):
                    return False
            
            return True
        except:
            return False
    
    def _safe_get_numeric(self, df, column, default=0.0):
        """Safely get numeric value from DataFrame"""
        try:
            if df is None or df.empty or column not in df.columns:
                return default
            
            value = df[column].iloc[-1]
            return self._safe_get_numeric_value(value, default)
        except:
            return default
    
    def _safe_get_numeric_value(self, value, default=0.0):
        """Safely convert value to numeric"""
        try:
            if value is None:
                return default
            
            if pd.isna(value):
                return default
            
            if isinstance(value, (int, float)):
                if np.isfinite(value):
                    return float(value)
                else:
                    return default
            
            # Try to convert string to float
            if isinstance(value, str):
                try:
                    converted = float(value)
                    if np.isfinite(converted):
                        return converted
                    else:
                        return default
                except:
                    return default
            
            # Try pandas numeric conversion
            try:
                converted = pd.to_numeric(value, errors='coerce')
                if pd.notna(converted) and np.isfinite(converted):
                    return float(converted)
                else:
                    return default
            except:
                return default
            
        except:
            return default
    
    def export_professional_report(self):
        """Export professional valuation report to Parquet with ML audit trail"""
        print("\n4. GENERATING PROFESSIONAL REPORT WITH ML AUDIT TRAIL")
        print("-" * 50)
        
        # Create the executive summary DataFrame
        exec_summary = self.valuation_summary.copy()
        
        # Add rating based on upside potential
        def get_rating(upside):
            if upside > 30:
                return 'STRONG BUY'
            elif upside > 15:
                return 'BUY'
            elif upside > 5:
                return 'HOLD'
            elif upside > -10:
                return 'REDUCE'
            else:
                return 'SELL'
        
        if 'Upside_%' in exec_summary.columns:
            exec_summary['Rating'] = exec_summary['Upside_%'].apply(get_rating)
            # Sort by upside potential
            exec_summary = exec_summary.sort_values('Upside_%', ascending=False)
        
        # Add metadata columns
        exec_summary['Report_Date'] = datetime.now().strftime('%Y-%m-%d')
        exec_summary['Analysis_Timestamp'] = datetime.now()
        exec_summary['ML_System_Version'] = 'Advanced_Anti_Leakage_v4.0'
        
        # Save main report as parquet file
        exec_summary.to_parquet(self.OUTPUT_PATH, index=False)
        
        # EXPORTAR DATOS ML PARA AUDITABILIDAD
        if self.ml_system:
            try:
                ml_results = self.ml_system.export_results()
                print(f"✓ ML audit trail exported:")
                for desc, path in ml_results.items():
                    print(f"  - {desc}: {path}")
            except Exception as e:
                print(f"⚠ ML audit trail export partially failed: {e}")
        
        print(f"✓ Professional report saved to: {self.OUTPUT_PATH}")
        print(f"✓ Report contains {len(exec_summary)} companies")
        
        return True
    
    def run_complete_analysis(self):
        """Run complete professional valuation analysis with advanced ML"""
        print("\n" + "=" * 80)
        print("STARTING PROFESSIONAL VALUATION ANALYSIS WITH ADVANCED ML")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # Step 1: Load data
        if not self.load_comprehensive_data():
            print("ERROR: Failed to load data")
            return False
        
        # Step 2: Generate valuations (ML system initialized dentro de este método)
        if not self.generate_forecast_valuations():
            print("ERROR: Failed to generate valuations")
            return False
        
        # Step 3: Export report
        if not self.export_professional_report():
            print("ERROR: Failed to export report")
            return False
        
        duration = datetime.now() - start_time
        
        print("\n" + "=" * 80)
        print("PROFESSIONAL VALUATION ANALYSIS COMPLETED")
        print("=" * 80)
        print(f"Duration: {duration}")
        print(f"Companies Analyzed: {len(self.valuation_summary)}")
        
        # Print top features used in ML models
        if self.ml_system and hasattr(self.ml_system, 'feature_usage_stats') and self.ml_system.feature_usage_stats:
            print("\nTOP 15 FEATURES MÁS UTILIZADAS EN LOS MODELOS ML:")
            print("-" * 70)
            
            # Get top 15 most used features
            top_features = self.ml_system.feature_usage_stats.most_common(15)
            total_usage = sum(self.ml_system.feature_usage_stats.values())
            
            for i, (feature, count) in enumerate(top_features, 1):
                percentage = (count / total_usage) * 100 if total_usage > 0 else 0
                print(f"{i:2d}. {feature:<45} {count:4d} modelos ({percentage:5.1f}%)")
            
            print("-" * 70)
            print(f"Total de features utilizadas: {len(self.ml_system.feature_usage_stats)}")
            print(f"Total de usos: {total_usage}")
        
        elif 'Upside_%' in self.valuation_summary.columns:
            # Fallback: mostrar oportunidades de inversión si no hay estadísticas ML
            print("\nTOP 15 INVESTMENT OPPORTUNITIES:")
            print("-" * 50)
            
            # Create Rating column if it doesn't exist
            if 'Rating' not in self.valuation_summary.columns:
                def get_rating(upside):
                    if pd.isna(upside):
                        return 'NEUTRAL'
                    elif upside > 30:
                        return 'STRONG BUY'
                    elif upside > 15:
                        return 'BUY'
                    elif upside > 5:
                        return 'HOLD'
                    elif upside > -10:
                        return 'REDUCE'
                    else:
                        return 'SELL'
                
                self.valuation_summary['Rating'] = self.valuation_summary['Upside_%'].apply(get_rating)
            
            # Select only existing columns
            available_cols = ['Ticker', 'Company', 'Current_Price', 'Fair_Value', 'Upside_%', 'Rating']
            existing_cols = [col for col in available_cols if col in self.valuation_summary.columns]
            
            top_15 = self.valuation_summary.nlargest(15, 'Upside_%')[existing_cols]
            
            for idx, row in top_15.iterrows():
                # Handle missing columns gracefully
                ticker = row.get('Ticker', 'N/A')
                company = str(row.get('Company', 'N/A'))[:30]
                current_price = row.get('Current_Price', 0)
                fair_value = row.get('Fair_Value', 0)
                upside = row.get('Upside_%', 0)
                rating = row.get('Rating', 'N/A')
                
                print(f"{ticker:6} | {company:30} | "
                     f"Current: ${current_price:7.2f} | "
                     f"Fair: ${fair_value:7.2f} | "
                     f"Upside: {upside:6.1f}% | "
                     f"{rating}")
        
        print(f"\n✓ Full report available at: {self.OUTPUT_PATH}")
        
        # Estadísticas del sistema ML
        if self.ml_system:
            print("\nADVANCED ML SYSTEM STATISTICS:")
            print("-" * 50)
            print(f"Ticker models trained: {len(self.ml_system.ticker_models)}")
            print(f"Industry models trained: {len(self.ml_system.industry_models)}")
            if hasattr(self.ml_system, 'system_statistics'):
                stats = self.ml_system.system_statistics
                print(f"ML Success rate: {stats.get('success_rate_ticker', 0):.1%}")
        
        return True


if __name__ == "__main__":
    # Initialize system
    valuation_system = ProfessionalValuationSystem()
    
    # Run analysis
    success = valuation_system.run_complete_analysis()
    
    if success:
        print("\n🎯 READY FOR INVESTMENT DECISION MAKING!")
        print("Professional valuations with advanced ML and uncertainty ranges generated.")
        print("✓ Full audit trail available for ML models")
        print("✓ Anti-data leakage measures implemented")
        print("✓ Temporal validation and feature safety ensured")
        print("✓ Parallelization implemented for faster processing")
        print("✓ Comparable multiples (Comp_P_E, Comp_P_B, Comp_P_S, Comp_EV_EBITDA) fixed and included")
    else:
        print("\n❌ Analysis failed. Please check data availability.")
    
    # Clean memory
    gc.collect()
