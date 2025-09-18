import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from scipy import signal, stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
import warnings
from functools import lru_cache
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import logging
import time
import glob
import talib as ta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================== CONFIGURACIÓN CUANTITATIVA ======================
# PALETA DE COLORES INSTITUCIONAL
UNIFIED_COLORS = {
    'bg': '#0f1419',           # Fondo principal
    'bg_secondary': '#1a1f2e',  # Fondo secundario
    'bg_tertiary': '#242938',   # Fondo para cards
    'grid': '#2a2f3e',         # Líneas de grid
    'text': '#e8eaed',          # Texto principal
    'text_secondary': '#9ca3af', # Texto secundario
    'text_muted': '#6b7280',    # Texto apagado
    'primary': '#00d4ff',       # Azul cyan primario
    'primary_dark': '#0099cc',  # Azul cyan oscuro
    'success': '#00ff88',       # Verde éxito
    'warning': '#ffaa00',       # Amarillo advertencia
    'danger': '#ff3366',        # Rojo peligro
    'support': '#00ff88',       # Verde para soporte
    'resistance': '#ff3366',    # Rojo para resistencia
    'bullish': '#00ff88',       # Verde alcista
    'bearish': '#ff3366',       # Rojo bajista
    'neutral': '#ffaa00',       # Amarillo neutral
    'accent': '#9333ea',        # Morado acento
    'border': '#2a2f3e'         # Bordes
}

@dataclass
class QuantitativeConfig:
    """Configuración para análisis cuantitativo profesional"""
    # Obtener el directorio del archivo actual
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Directorio raíz del proyecto (dos niveles arriba desde scr/Dashboard/)
    PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")

    # Configuración de rutas relativas
    BASE_PATH = os.path.join(PROJECT_ROOT, "data")

    DATA_PATH = os.path.join(BASE_PATH, 'Historical_Prices')
    SCREENER_PATH = os.path.join(BASE_PATH, "Ticker_List", "screener.parquet")
    
    # Parámetros de Machine Learning
    ML_CONFIG = {
        'dbscan_eps': 0.02,
        'dbscan_min_samples': 3,
        'kmeans_clusters': 8,
        'isolation_contamination': 0.1,
        'pca_components': 5,
        'ransac_residual_threshold': 0.01,
        'gmm_components': 3
    }
    
    # Configuración por granularidad
    GRANULARITY_CONFIG = {
        'diario': {'periods': 252, 'sr_levels': 15, 'channel_threshold': 0.65},
        'semanal': {'periods': 156, 'sr_levels': 20, 'channel_threshold': 0.60},
        'mensual': {'periods': None, 'sr_levels': 25, 'channel_threshold': 0.55}
    }
    
    # Umbrales cuantitativos
    SIGNAL_THRESHOLDS = {
        'strong_buy': 0.35, 'buy': 0.15, 'sell': -0.15, 'strong_sell': -0.35
    }
    
    # Pesos para scoring cuantitativo
    FEATURE_WEIGHTS = {
        'technical': 0.25, 'ml_patterns': 0.25, 'support_resistance': 0.20,
        'volume_analysis': 0.15, 'trend_strength': 0.15
    }

# ====================== GESTOR DE DATOS CUANTITATIVO ======================
class QuantitativeDataManager:
    """Gestión profesional de datos con preprocesamiento cuantitativo"""
    
    def __init__(self, config: QuantitativeConfig):
        self.config = config
        self.scaler = RobustScaler()
        self._cache = {}
    
    @lru_cache(maxsize=32)
    def load_and_preprocess(self, ticker: str, granularity: str) -> pd.DataFrame:
        """Carga y preprocesa datos con técnicas cuantitativas"""
        try:
            file_path = os.path.join(self.config.DATA_PATH, f"{ticker}.parquet")
            if not os.path.exists(file_path):
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            if df.empty or len(df) < 50:
                return pd.DataFrame()
            
            df = self._clean_and_resample(df, granularity)
            df = self._extract_quantitative_features(df)
            df = self._apply_data_limits(df, granularity)
            
            logger.info(f"Datos procesados: {ticker} - {len(df)} períodos")
            return df
            
        except Exception as e:
            logger.error(f"Error procesando {ticker}: {e}")
            return pd.DataFrame()
    
    def _clean_and_resample(self, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Limpieza y resampleo profesional"""
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Configurar índice temporal
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')
        
        df = df.sort_index()
        
        # Remover outliers usando IQR robusto
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.05)
                Q3 = df[col].quantile(0.95)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 2*IQR) & (df[col] <= Q3 + 2*IQR)]
        
        # Resamplear según granularidad
        if granularity == 'semanal':
            df = df.resample('W-FRI').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 
                'close': 'last', 'volume': 'sum'
            }).dropna()
        elif granularity == 'mensual':
            df = df.resample('M').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 
                'close': 'last', 'volume': 'sum'
            }).dropna()
        
        return df.dropna()
    
    def _extract_quantitative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracción de características cuantitativas avanzadas"""
        # Precios y retornos
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Características técnicas
        df['rsi'] = self._calculate_rsi(df['close'])
        df['bb_position'] = self._calculate_bb_position(df['close'])
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Características de momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Características de volumen-precio
        df['vwap'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).rolling(10).sum()
        
        # Nuevos indicadores técnicos
        df = self._add_technical_indicators(df)
        
        # Limpiar valores infinitos y NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade indicadores técnicos adicionales usando TA-Lib"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            
            # Estocástico
            stoch_k, stoch_d = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # EMA
            df['ema_12'] = ta.EMA(close, timeperiod=12)
            df['ema_26'] = ta.EMA(close, timeperiod=26)
            
            # Bandas de Bollinger
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # RSI adicional
            df['rsi_7'] = ta.RSI(close, timeperiod=7)
            df['rsi_21'] = ta.RSI(close, timeperiod=21)
            
        except Exception as e:
            logger.error(f"Error añadiendo indicadores técnicos: {e}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI optimizado"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Posición en Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (prices - lower) / (upper - lower).replace(0, 1)
    
    def _apply_data_limits(self, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Aplica límites de datos según granularidad"""
        config = self.config.GRANULARITY_CONFIG[granularity]
        max_periods = config['periods']
        
        if max_periods and len(df) > max_periods:
            return df.tail(max_periods)
        return df

# ====================== ANÁLISIS CUANTITATIVO DE SOPORTE/RESISTENCIA ======================
class QuantitativeSRAnalyzer:
    """Análisis cuantitativo de S/R usando Machine Learning"""
    
    def __init__(self, config: QuantitativeConfig):
        self.config = config
        self.dbscan = DBSCAN(eps=config.ML_CONFIG['dbscan_eps'], 
                            min_samples=config.ML_CONFIG['dbscan_min_samples'])
        self.kmeans = KMeans(n_clusters=config.ML_CONFIG['kmeans_clusters'], random_state=42)
    
    def analyze_support_resistance(self, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Análisis cuantitativo completo de S/R"""
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        
        try:
            # Detectar puntos de interés usando múltiples métodos
            pivot_points = self._detect_pivot_points(df)
            volume_levels = self._detect_volume_clusters(df)
            psychological_levels = self._detect_psychological_levels(df)
            
            # Combinar todos los niveles candidatos
            all_levels = []
            all_levels.extend(pivot_points)
            all_levels.extend(volume_levels) 
            all_levels.extend(psychological_levels)
            
            if not all_levels:
                return pd.DataFrame()
            
            # Clustering de niveles usando DBSCAN
            levels_df = pd.DataFrame(all_levels)
            clustered_levels = self._cluster_levels(levels_df, df['close'].iloc[-1])
            
            # Ranking cuantitativo
            final_levels = self._rank_levels_quantitatively(clustered_levels, df, granularity)
            
            logger.info(f"S/R cuantitativo: {len(final_levels)} niveles detectados")
            return final_levels
            
        except Exception as e:
            logger.error(f"Error en análisis S/R cuantitativo: {e}")
            return pd.DataFrame()
    
    def _detect_pivot_points(self, df: pd.DataFrame) -> List[Dict]:
        """Detección de pivotes usando análisis de extremos locales"""
        levels = []
        
        try:
            # Usar diferentes órdenes para capturar pivotes de diferentes escalas
            for order in [3, 5, 8]:
                if len(df) < order * 3:
                    continue
                    
                highs_idx = signal.argrelextrema(df['high'].values, np.greater, order=order)[0]
                lows_idx = signal.argrelextrema(df['low'].values, np.less, order=order)[0]
                
                for idx in highs_idx:
                    price = df['high'].iloc[idx]
                    volume_confirm = df['volume'].iloc[max(0, idx-2):idx+3].mean()
                    levels.append({
                        'price': price, 'type': 'resistance', 'method': 'pivot',
                        'strength': 1.0 + (order / 10), 'volume_confirmation': volume_confirm
                    })
                
                for idx in lows_idx:
                    price = df['low'].iloc[idx]
                    volume_confirm = df['volume'].iloc[max(0, idx-2):idx+3].mean()
                    levels.append({
                        'price': price, 'type': 'support', 'method': 'pivot',
                        'strength': 1.0 + (order / 10), 'volume_confirmation': volume_confirm
                    })
            
        except Exception as e:
            logger.debug(f"Error en detección de pivotes: {e}")
        
        return levels
    
    def _detect_volume_clusters(self, df: pd.DataFrame) -> List[Dict]:
        """Detección de niveles basada en clustering de volumen"""
        levels = []
        
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return levels
            
            # Preparar datos para clustering
            price_volume_data = np.column_stack([
                df['close'].values, 
                df['volume'].values / df['volume'].max()  # Normalizar volumen
            ])
            
            # Aplicar K-means para encontrar centros de actividad
            kmeans = KMeans(n_clusters=min(8, len(df)//10), random_state=42)
            clusters = kmeans.fit_predict(price_volume_data)
            
            # Analizar cada cluster
            for i in range(kmeans.n_clusters):
                cluster_mask = clusters == i
                cluster_data = df[cluster_mask]
                
                if len(cluster_data) < 3:
                    continue
                
                avg_price = cluster_data['close'].mean()
                total_volume = cluster_data['volume'].sum()
                price_std = cluster_data['close'].std()
                
                # Solo niveles con concentración significativa
                if price_std < avg_price * 0.02 and total_volume > df['volume'].median():
                    current_price = df['close'].iloc[-1]
                    level_type = 'resistance' if avg_price > current_price else 'support'
                    
                    levels.append({
                        'price': avg_price, 'type': level_type, 'method': 'volume_cluster',
                        'strength': min(2.0, total_volume / df['volume'].median()),
                        'volume_confirmation': total_volume
                    })
            
        except Exception as e:
            logger.debug(f"Error en clustering de volumen: {e}")
        
        return levels
    
    def _detect_psychological_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Detección de niveles psicológicos usando análisis estadístico"""
        levels = []
        
        try:
            current_price = df['close'].iloc[-1]
            price_range = df['high'].max() - df['low'].min()
            
            # Niveles redondos basados en la magnitud del precio
            if current_price > 1000:
                increments = [100, 250, 500, 1000]
            elif current_price > 100:
                increments = [10, 25, 50, 100]
            elif current_price > 10:
                increments = [1, 2.5, 5, 10]
            else:
                increments = [0.1, 0.25, 0.5, 1]
            
            for increment in increments:
                base = round(current_price / increment) * increment
                for mult in [-3, -2, -1, 0, 1, 2, 3]:
                    level_price = base + (mult * increment)
                    
                    if level_price > 0 and abs(level_price - current_price) > increment * 0.1:
                        # Verificar relevancia histórica
                        touches = self._count_historical_touches(df, level_price, increment * 0.5)
                        
                        if touches > 0:
                            level_type = 'resistance' if level_price > current_price else 'support'
                            levels.append({
                                'price': level_price, 'type': level_type, 'method': 'psychological',
                                'strength': 0.8 + (touches * 0.2), 'volume_confirmation': df['volume'].median()
                            })
            
        except Exception as e:
            logger.debug(f"Error en niveles psicológicos: {e}")
        
        return levels
    
    def _count_historical_touches(self, df: pd.DataFrame, level: float, tolerance: float) -> int:
        """Cuenta toques históricos de un nivel"""
        try:
            touches = 0
            for _, row in df.iterrows():
                if (abs(row['high'] - level) <= tolerance or 
                    abs(row['low'] - level) <= tolerance):
                    touches += 1
            return min(touches, 10)  # Limitar para evitar sobreponderar
        except:
            return 0
    
    def _cluster_levels(self, levels_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
        """Clustering de niveles usando DBSCAN"""
        try:
            if len(levels_df) < 3:
                return levels_df
            
            # Preparar datos para clustering (precio normalizado por precio actual)
            price_normalized = levels_df['price'].values.reshape(-1, 1) / current_price
            
            # Aplicar DBSCAN
            clusters = self.dbscan.fit_predict(price_normalized)
            levels_df['cluster'] = clusters
            
            # Consolidar clusters
            consolidated = []
            for cluster_id in np.unique(clusters):
                if cluster_id == -1:  # Outliers
                    cluster_data = levels_df[levels_df['cluster'] == cluster_id]
                    for _, level in cluster_data.iterrows():
                        consolidated.append(level.to_dict())
                else:
                    cluster_data = levels_df[levels_df['cluster'] == cluster_id]
                    if len(cluster_data) > 1:
                        # Consolidar cluster
                        consolidated_level = {
                            'price': cluster_data['price'].mean(),
                            'type': cluster_data['type'].mode().iloc[0],
                            'method': 'clustered',
                            'strength': cluster_data['strength'].mean(),
                            'volume_confirmation': cluster_data['volume_confirmation'].sum(),
                            'cluster_size': len(cluster_data)
                        }
                        consolidated.append(consolidated_level)
                    else:
                        consolidated.append(cluster_data.iloc[0].to_dict())
            
            return pd.DataFrame(consolidated)
            
        except Exception as e:
            logger.debug(f"Error en clustering: {e}")
            return levels_df
    
    def _rank_levels_quantitatively(self, levels_df: pd.DataFrame, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Ranking cuantitativo de niveles"""
        if levels_df.empty:
            return levels_df
        
        try:
            current_price = df['close'].iloc[-1]
            max_levels = self.config.GRANULARITY_CONFIG[granularity]['sr_levels']
            
            # Calcular métricas cuantitativas
            levels_df['distance_pct'] = abs(levels_df['price'] - current_price) / current_price
            levels_df['proximity_score'] = 1 / (1 + levels_df['distance_pct'] * 10)
            
            # Score cuantitativo compuesto
            levels_df['quantitative_score'] = (
                levels_df['strength'] * 0.4 +
                levels_df['proximity_score'] * 0.3 +
                (levels_df['volume_confirmation'] / df['volume'].median()) * 0.3
            )
            
            # Filtrar por distancia máxima
            max_distance = {'diario': 0.4, 'semanal': 0.5, 'mensual': 0.6}[granularity]
            levels_df = levels_df[levels_df['distance_pct'] <= max_distance]
            
            # Seleccionar top niveles
            levels_df = levels_df.nlargest(max_levels, 'quantitative_score')
            
            # Añadir categorías de proximidad
            levels_df['proximity'] = levels_df['distance_pct'].apply(
                lambda x: 'immediate' if x < 0.02 else 'near' if x < 0.05 else 'medium' if x < 0.15 else 'far'
            )
            
            return levels_df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error en ranking cuantitativo: {e}")
            return levels_df

# ====================== ANÁLISIS CUANTITATIVO DE PATRONES ======================
class QuantitativePatternAnalyzer:
    """Análisis cuantitativo de patrones usando ML"""
    
    def __init__(self, config: QuantitativeConfig):
        self.config = config
        self.ransac = RANSACRegressor(residual_threshold=config.ML_CONFIG['ransac_residual_threshold'])
    
    def detect_patterns(self, df: pd.DataFrame, granularity: str) -> List[Dict]:
        """Detección cuantitativa de patrones"""
        if df.empty or len(df) < 40:
            return []
        
        try:
            patterns = []
            
            # Detectar tendencias usando RANSAC
            trend_patterns = self._detect_trend_patterns(df)
            patterns.extend(trend_patterns)
            
            # Detectar patrones de reversión usando análisis estadístico
            reversal_patterns = self._detect_reversal_patterns(df)
            patterns.extend(reversal_patterns)
            
            # Filtrar y validar patrones
            validated_patterns = self._validate_patterns_quantitatively(patterns, df, granularity)
            
            logger.info(f"Patrones cuantitativos: {len(validated_patterns)} detectados")
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error en detección cuantitativa de patrones: {e}")
            return []
    
    def _detect_trend_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detección de tendencias usando RANSAC"""
        patterns = []
        
        try:
            # Analizar diferentes ventanas temporales
            for window in [30, 60, 90]:
                if len(df) < window:
                    continue
                
                segment = df.tail(window).copy()
                X = np.arange(len(segment)).reshape(-1, 1)
                y = segment['close'].values
                
                # Aplicar RANSAC para detección robusta de tendencia
                try:
                    self.ransac.fit(X, y)
                    slope = self.ransac.estimator_.coef_[0]
                    r2_score = self.ransac.score(X, y)
                    
                    if r2_score > 0.7:  # Tendencia clara
                        trend_strength = abs(slope) * window / segment['close'].mean()
                        
                        if trend_strength > 0.02:  # Tendencia significativa
                            pattern_type = 'uptrend' if slope > 0 else 'downtrend'
                            
                            patterns.append({
                                'type': pattern_type,
                                'confidence': r2_score,
                                'strength': trend_strength,
                                'start_idx': len(df) - window,
                                'end_idx': len(df) - 1,
                                'slope': slope,
                                'method': 'ransac_trend'
                            })
                
                except Exception:
                    continue
            
        except Exception as e:
            logger.debug(f"Error en detección de tendencias: {e}")
        
        return patterns
    
    def _detect_reversal_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detección de patrones de reversión usando análisis estadístico"""
        patterns = []
        
        try:
            # Detectar dobles y triples usando clustering de extremos
            for lookback in [60, 120, 200]:
                if len(df) < lookback:
                    continue
                
                segment = df.tail(lookback)
                
                # Encontrar máximos y mínimos locales
                highs_idx = signal.argrelextrema(segment['high'].values, np.greater, order=5)[0]
                lows_idx = signal.argrelextrema(segment['low'].values, np.less, order=5)[0]
                
                # Analizar agrupaciones de extremos
                if len(highs_idx) >= 2:
                    high_patterns = self._analyze_extrema_clusters(
                        segment, highs_idx, 'high', 'resistance', lookback
                    )
                    patterns.extend(high_patterns)
                
                if len(lows_idx) >= 2:
                    low_patterns = self._analyze_extrema_clusters(
                        segment, lows_idx, 'low', 'support', lookback
                    )
                    patterns.extend(low_patterns)
            
        except Exception as e:
            logger.debug(f"Error en detección de reversión: {e}")
        
        return patterns
    
    def _analyze_extrema_clusters(self, segment: pd.DataFrame, extrema_idx: np.ndarray, 
                                 price_col: str, pattern_type: str, lookback: int) -> List[Dict]:
        """Análisis de clusters de extremos para patrones de reversión"""
        patterns = []
        
        try:
            if len(extrema_idx) < 2:
                return patterns
            
            extrema_prices = segment[price_col].iloc[extrema_idx].values
            
            # Usar clustering para encontrar niveles similares
            if len(extrema_prices) >= 2:
                price_matrix = extrema_prices.reshape(-1, 1)
                kmeans = KMeans(n_clusters=min(3, len(extrema_prices)), random_state=42)
                clusters = kmeans.fit_predict(price_matrix)
                
                # Analizar cada cluster
                for cluster_id in range(kmeans.n_clusters):
                    cluster_indices = extrema_idx[clusters == cluster_id]
                    cluster_prices = extrema_prices[clusters == cluster_id]
                    
                    if len(cluster_indices) >= 2:
                        price_std = np.std(cluster_prices)
                        price_mean = np.mean(cluster_prices)
                        
                        # Verificar que los precios estén suficientemente agrupados
                        if price_std < price_mean * 0.03:  # Dentro del 3%
                            pattern_name = f"{'double' if len(cluster_indices) == 2 else 'triple'}_{pattern_type}"
                            
                            # Verificar separación temporal
                            time_span = cluster_indices[-1] - cluster_indices[0]
                            if time_span > 10:  # Mínimo 10 períodos
                                confidence = 0.8 - (price_std / price_mean)
                                
                                patterns.append({
                                    'type': pattern_name,
                                    'confidence': confidence,
                                    'price_level': price_mean,
                                    'start_idx': len(segment) - lookback + cluster_indices[0],
                                    'end_idx': len(segment) - lookback + cluster_indices[-1],
                                    'touches': len(cluster_indices),
                                    'method': 'statistical_reversal'
                                })
        
        except Exception as e:
            logger.debug(f"Error en análisis de extremos: {e}")
        
        return patterns
    
    def _validate_patterns_quantitatively(self, patterns: List[Dict], df: pd.DataFrame, granularity: str) -> List[Dict]:
        """Validación cuantitativa de patrones"""
        validated = []
        
        try:
            current_price = df['close'].iloc[-1]
            threshold = self.config.GRANULARITY_CONFIG[granularity]['channel_threshold']
            
            for pattern in patterns:
                # Calcular score de validez
                validity_score = pattern.get('confidence', 0.5)
                
                # Bonificar patrones activos/recientes
                days_from_end = len(df) - 1 - pattern.get('end_idx', 0)
                recency_bonus = max(0, 1 - days_from_end / 30)  # Decae en 30 días
                validity_score *= (1 + recency_bonus)
                
                # Bonificar patrones cerca del precio actual
                if 'price_level' in pattern:
                    distance = abs(pattern['price_level'] - current_price) / current_price
                    proximity_bonus = max(0, 1 - distance * 5)  # Decae con distancia
                    validity_score *= (1 + proximity_bonus * 0.5)
                
                # Filtrar por umbral de calidad
                if validity_score >= threshold:
                    pattern['quantitative_score'] = validity_score
                    pattern['active'] = days_from_end <= 10
                    validated.append(pattern)
            
            # Ordenar por score y limitar
            validated.sort(key=lambda x: x['quantitative_score'], reverse=True)
            return validated[:12]  # Top 12 patrones
            
        except Exception as e:
            logger.error(f"Error en validación cuantitativa: {e}")
            return patterns

# ====================== SISTEMA CUANTITATIVO DE SEÑALES ======================
class QuantitativeSignalSystem:
    """Sistema cuantitativo de señales usando ML ensemble"""
    
    def __init__(self, config: QuantitativeConfig):
        self.config = config
        self.isolation_forest = IsolationForest(contamination=config.ML_CONFIG['isolation_contamination'])
        self.gmm = GaussianMixture(n_components=config.ML_CONFIG['gmm_components'])
        self.pca = PCA(n_components=config.ML_CONFIG['pca_components'])
    
    def generate_quantitative_signals(self, df: pd.DataFrame, sr_levels: pd.DataFrame, 
                                    patterns: List[Dict]) -> Dict:
        """Generación cuantitativa de señales usando ensemble ML"""
        if df.empty or len(df) < 20:
            return self._empty_signal()
        
        try:
            # Extraer features cuantitativos
            features = self._extract_signal_features(df, sr_levels, patterns)
            
            if features is None:
                return self._empty_signal()
            
            # Análisis cuantitativo multi-método
            technical_score = self._analyze_technical_features(features)
            ml_score = self._analyze_ml_features(df, features)
            sr_score = self._analyze_sr_interaction(df, sr_levels)
            volume_score = self._analyze_volume_anomalies(df)
            trend_score = self._analyze_trend_regime(df)
            
            # Score cuantitativo ponderado
            total_score = (
                technical_score * self.config.FEATURE_WEIGHTS['technical'] +
                ml_score * self.config.FEATURE_WEIGHTS['ml_patterns'] +
                sr_score * self.config.FEATURE_WEIGHTS['support_resistance'] +
                volume_score * self.config.FEATURE_WEIGHTS['volume_analysis'] +
                trend_score * self.config.FEATURE_WEIGHTS['trend_strength']
            )
            
            # Clasificar señal
            signal_classification = self._classify_signal(total_score)
            
            # Calcular niveles de trading cuantitativos
            trading_levels = self._calculate_quantitative_levels(df, sr_levels, signal_classification)
            
            return {
                'action': signal_classification['action'],
                'confidence': signal_classification['confidence'],
                'score': total_score,
                'entry_price': trading_levels['entry'],
                'stop_loss': trading_levels['stop_loss'],
                'take_profit': trading_levels['take_profit'],
                'current_price': float(df['close'].iloc[-1]),
                'quantitative_features': features,
                'reasoning': self._generate_quantitative_reasoning(
                    technical_score, ml_score, sr_score, volume_score, trend_score
                )
            }
            
        except Exception as e:
            logger.error(f"Error en señales cuantitativas: {e}")
            return self._empty_signal()
    
    def _extract_signal_features(self, df: pd.DataFrame, sr_levels: pd.DataFrame, patterns: List[Dict]) -> Optional[Dict]:
        """Extracción de features cuantitativos para señales"""
        try:
            last_row = df.iloc[-1]
            
            features = {
                # Features técnicos
                'rsi': last_row.get('rsi', 50),
                'bb_position': last_row.get('bb_position', 0.5),
                'volume_ratio': last_row.get('volume_ratio', 1.0),
                'volatility': last_row.get('volatility', 0.02),
                
                # Features de momentum
                'momentum_5': last_row.get('momentum_5', 0),
                'momentum_10': last_row.get('momentum_10', 0),
                'momentum_20': last_row.get('momentum_20', 0),
                
                # Features de contexto
                'sr_proximity': self._calculate_sr_proximity(df, sr_levels),
                'pattern_strength': self._calculate_pattern_strength(patterns),
                'trend_consistency': self._calculate_trend_consistency(df)
            }
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extrayendo features: {e}")
            return None
    
    def _analyze_technical_features(self, features: Dict) -> float:
        """Análisis cuantitativo de features técnicos"""
        try:
            score = 0.0
            
            # RSI con scoring no lineal
            rsi = features['rsi']
            if rsi < 20:
                score += 0.8
            elif rsi < 30:
                score += 0.6
            elif rsi > 80:
                score -= 0.8
            elif rsi > 70:
                score -= 0.6
            
            # Bollinger Bands position
            bb_pos = features['bb_position']
            if bb_pos < 0.1:
                score += 0.7
            elif bb_pos > 0.9:
                score -= 0.7
            
            # Volume ratio
            vol_ratio = features['volume_ratio']
            if vol_ratio > 2.0:
                score += 0.4 if features['momentum_5'] > 0 else -0.4
            
            # Momentum alignment
            mom_signals = [features['momentum_5'], features['momentum_10'], features['momentum_20']]
            momentum_alignment = sum(1 if m > 0.02 else -1 if m < -0.02 else 0 for m in mom_signals)
            score += momentum_alignment * 0.2
            
            return np.clip(score, -2.0, 2.0)
            
        except Exception as e:
            logger.debug(f"Error en análisis técnico: {e}")
            return 0.0
    
    def _analyze_ml_features(self, df: pd.DataFrame, features: Dict) -> float:
        """Análisis usando ML features"""
        try:
            score = 0.0
            
            # Pattern strength
            pattern_strength = features.get('pattern_strength', 0)
            score += pattern_strength * 0.5
            
            # Trend consistency
            trend_consistency = features.get('trend_consistency', 0)
            score += trend_consistency * 0.3
            
            # Regime detection usando features estadísticos
            if len(df) >= 20:
                recent_returns = df['returns'].tail(20).values
                volatility_regime = np.std(recent_returns) / np.std(df['returns'].tail(60))
                
                # Penalizar en regímenes de alta volatilidad
                if volatility_regime > 1.5:
                    score *= 0.7
                elif volatility_regime < 0.7:
                    score *= 1.2
            
            return np.clip(score, -1.5, 1.5)
            
        except Exception as e:
            logger.debug(f"Error en ML features: {e}")
            return 0.0
    
    def _analyze_sr_interaction(self, df: pd.DataFrame, sr_levels: pd.DataFrame) -> float:
        """Análisis cuantitativo de interacción con S/R"""
        try:
            if sr_levels.empty:
                return 0.0
            
            current_price = df['close'].iloc[-1]
            score = 0.0
            
            # Analizar niveles por proximidad
            for _, level in sr_levels.head(5).iterrows():  # Top 5 niveles
                distance = abs(level['price'] - current_price) / current_price
                strength = level.get('quantitative_score', 1.0)
                
                if distance < 0.02:  # Muy cerca
                    multiplier = strength * 2.0
                    if level['type'] == 'support' and current_price > level['price']:
                        score += multiplier
                    elif level['type'] == 'resistance' and current_price < level['price']:
                        score -= multiplier
                elif distance < 0.05:  # Cerca
                    multiplier = strength * 1.0
                    if level['type'] == 'support':
                        score += multiplier * 0.5
                    else:
                        score -= multiplier * 0.5
            
            return np.clip(score, -2.0, 2.0)
            
        except Exception as e:
            logger.debug(f"Error en análisis S/R: {e}")
            return 0.0
    
    def _analyze_volume_anomalies(self, df: pd.DataFrame) -> float:
        """Análisis de anomalías de volumen usando Isolation Forest"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return 0.0
            
            # Preparar datos de volumen
            volume_features = np.column_stack([
                df['volume'].values,
                df['volume_ratio'].values if 'volume_ratio' in df.columns else np.ones(len(df))
            ])
            
            # Detectar anomalías
            self.isolation_forest.fit(volume_features[-50:])  # Usar últimos 50 períodos
            anomaly_scores = self.isolation_forest.decision_function(volume_features[-5:])  # Últimos 5
            
            # Convertir a score de señal
            recent_volume_score = np.mean(anomaly_scores)
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            # Volumen anómalo + movimiento de precio = señal
            if recent_volume_score > 0.1 and abs(price_change) > 0.02:
                return 0.6 if price_change > 0 else -0.6
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error en análisis de volumen: {e}")
            return 0.0
    
    def _analyze_trend_regime(self, df: pd.DataFrame) -> float:
        """Análisis de régimen de tendencia usando GMM"""
        try:
            if len(df) < 30:
                return 0.0
            
            # Preparar features para GMM
            trend_features = np.column_stack([
                df['returns'].values,
                df['volatility'].values if 'volatility' in df.columns else np.abs(df['returns']).rolling(10).mean().values
            ])
            
            # Ajustar GMM
            self.gmm.fit(trend_features[-100:])  # Últimos 100 períodos
            
            # Predecir régimen actual
            current_features = trend_features[-1:] if len(trend_features) > 0 else np.array([[0, 0.02]])
            current_regime = self.gmm.predict(current_features)[0]
            regime_proba = self.gmm.predict_proba(current_features)[0]
            
            # Interpretar régimen
            confidence = np.max(regime_proba)
            
            # Análisis de momentum del régimen
            recent_returns = df['returns'].tail(10).mean()
            trend_strength = recent_returns / (df['volatility'].tail(10).mean() + 1e-6)
            
            return np.clip(trend_strength * confidence, -1.0, 1.0)
            
        except Exception as e:
            logger.debug(f"Error en análisis de régimen: {e}")
            return 0.0
    
    def _calculate_sr_proximity(self, df: pd.DataFrame, sr_levels: pd.DataFrame) -> float:
        """Calcula proximidad cuantitativa a niveles S/R"""
        try:
            if sr_levels.empty:
                return 0.5
            
            current_price = df['close'].iloc[-1]
            min_distance = float('inf')
            
            for _, level in sr_levels.iterrows():
                distance = abs(level['price'] - current_price) / current_price
                min_distance = min(min_distance, distance)
            
            # Convertir distancia to score (inverso)
            return max(0, 1 - min_distance * 20)
            
        except:
            return 0.5
    
    def _calculate_pattern_strength(self, patterns: List[Dict]) -> float:
        """Calcula fuerza cuantitativa de patrones"""
        try:
            if not patterns:
                return 0.0
            
            active_patterns = [p for p in patterns if p.get('active', False)]
            if not active_patterns:
                return 0.0
            
            total_strength = sum(p.get('quantitative_score', 0.5) for p in active_patterns[:3])
            return min(total_strength / 3, 1.0)
            
        except:
            return 0.0
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calcula consistencia de tendencia"""
        try:
            if len(df) < 20:
                return 0.0
            
            # Múltiples timeframes de momentum
            mom_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
            mom_10 = (df['close'].iloc[-1] - df['close'].iloc[-11]) / df['close'].iloc[-11]
            mom_20 = (df['close'].iloc[-1] - df['close'].iloc[-21]) / df['close'].iloc[-21]
            
            # Consistencia = alineación de signos
            momentums = [mom_5, mom_10, mom_20]
            positive = sum(1 for m in momentums if m > 0.01)
            negative = sum(1 for m in momentums if m < -0.01)
            
            if positive == 3:
                return 1.0
            elif negative == 3:
                return -1.0
            elif positive > negative:
                return 0.5
            elif negative > positive:
                return -0.5
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _classify_signal(self, total_score: float) -> Dict:
        """Clasificación cuantitativa de señales"""
        thresholds = self.config.SIGNAL_THRESHOLDS
        
        if total_score >= thresholds['strong_buy']:
            return {'action': 'COMPRA FUERTE', 'confidence': min(0.95, 0.7 + abs(total_score) * 0.5)}
        elif total_score >= thresholds['buy']:
            return {'action': 'COMPRA', 'confidence': min(0.8, 0.5 + abs(total_score) * 0.8)}
        elif total_score <= thresholds['strong_sell']:
            return {'action': 'VENTA FUERTE', 'confidence': min(0.95, 0.7 + abs(total_score) * 0.5)}
        elif total_score <= thresholds['sell']:
            return {'action': 'VENTA', 'confidence': min(0.8, 0.5 + abs(total_score) * 0.8)}
        else:
            return {'action': 'NEUTRAL', 'confidence': max(0.3, 0.5 - abs(total_score) * 2)}
    
    def _calculate_quantitative_levels(self, df: pd.DataFrame, sr_levels: pd.DataFrame, 
                                     signal_class: Dict) -> Dict:
        """Cálculo cuantitativo de niveles de trading"""
        current_price = df['close'].iloc[-1]
        atr = df['volatility'].tail(20).mean() * current_price if 'volatility' in df.columns else current_price * 0.02
        
        entry = current_price
        stop_loss = None
        take_profit = None
        
        if 'COMPRA' in signal_class['action']:
            # Stop loss: soporte más cercano o ATR-based
            if not sr_levels.empty:
                supports = sr_levels[
                    (sr_levels['type'] == 'support') & 
                    (sr_levels['price'] < current_price * 0.98)
                ]
                if not supports.empty:
                    stop_loss = supports.iloc[0]['price']
            
            if stop_loss is None:
                stop_loss = current_price - atr * 2
            
            # Take profit: resistencia más cercana o risk/reward 2:1
            if not sr_levels.empty:
                resistances = sr_levels[
                    (sr_levels['type'] == 'resistance') & 
                    (sr_levels['price'] > current_price * 1.02)
                ]
                if not resistances.empty:
                    take_profit = resistances.iloc[0]['price']
            
            if take_profit is None:
                risk = current_price - stop_loss
                take_profit = current_price + risk * 2  # R:R 1:2
        
        elif 'VENTA' in signal_class['action']:
            # Stop loss: resistencia más cercana o ATR-based
            if not sr_levels.empty:
                resistances = sr_levels[
                    (sr_levels['type'] == 'resistance') & 
                    (sr_levels['price'] > current_price * 1.02)
                ]
                if not resistances.empty:
                    stop_loss = resistances.iloc[0]['price']
            
            if stop_loss is None:
                stop_loss = current_price + atr * 2
            
            # Take profit: soporte más cercano o risk/reward 2:1
            if not sr_levels.empty:
                supports = sr_levels[
                    (sr_levels['type'] == 'support') & 
                    (sr_levels['price'] < current_price * 0.98)
                ]
                if not supports.empty:
                    take_profit = supports.iloc[0]['price']
            
            if take_profit is None:
                risk = stop_loss - current_price
                take_profit = current_price - risk * 2  # R:R 1:2
        
        return {
            'entry': float(entry),
            'stop_loss': float(stop_loss) if stop_loss else None,
            'take_profit': float(take_profit) if take_profit else None
        }
    
    def _generate_quantitative_reasoning(self, technical: float, ml: float, sr: float, 
                                       volume: float, trend: float) -> str:
        """Genera explicación cuantitativa"""
        components = []
        
        if abs(technical) > 0.3:
            components.append(f"Técnicos: {'Alcista' if technical > 0 else 'Bajista'} ({technical:.2f})")
        
        if abs(ml) > 0.2:
            components.append(f"ML Patterns: {'Positivo' if ml > 0 else 'Negativo'} ({ml:.2f})")
        
        if abs(sr) > 0.3:
            components.append(f"S/R: {'Soporte' if sr > 0 else 'Resistencia'} ({sr:.2f})")
        
        if abs(volume) > 0.2:
            components.append(f"Volumen: {'Anómalo alcista' if volume > 0 else 'Anómalo bajista'}")
        
        if abs(trend) > 0.2:
            components.append(f"Tendencia: {'Consistente+' if trend > 0 else 'Consistente-'}")
        
        return "Análisis cuantitativo: " + "; ".join(components) if components else "Señal neutral"
    
    def _empty_signal(self) -> Dict:
        """Señal vacía"""
        return {
            'action': 'SIN DATOS',
            'confidence': 0,
            'score': 0,
            'entry_price': 0,
            'stop_loss': None,
            'take_profit': None,
            'current_price': 0,
            'quantitative_features': {},
            'reasoning': 'Datos insuficientes para análisis cuantitativo'
        }


# ====================== DETECTOR DE CANALES MEJORADO ======================
class ChannelDetector:
    """Detección profesional de canales de tendencia - VERSIÓN MEJORADA"""
    
    def __init__(self, config: QuantitativeConfig):
        self.config = config
        self.min_touches = 3  # Mínimo de toques para validar un canal
        self.touch_tolerance = 0.02  # Tolerancia de 2% para considerar un toque
        self.max_channels_to_display = 3  # Máximo de canales a mostrar
    
    def detect_channels(self, df: pd.DataFrame, lookback_periods: int = 150) -> List[Dict]:
        """Detección profesional de canales usando OHLC"""
        if len(df) < 30:
            logger.debug("Dataset demasiado pequeño para detectar canales")
            return []
        
        try:
            # Usar datos más recientes pero conservar algo de historia
            max_lookback = min(lookback_periods, len(df))
            recent_data = df.tail(max_lookback).copy()
            
            all_channels = []
            
            # Detección adaptativa con ventanas variables
            channels_parallel = self._adaptive_parallel_detection(recent_data, df)
            channels_triangle = self._adaptive_triangle_detection(recent_data, df)
            channels_wedge = self._detect_wedge_patterns(recent_data, df)
            
            all_channels.extend(channels_parallel)
            all_channels.extend(channels_triangle)
            all_channels.extend(channels_wedge)
            
            # Filtrar y rankear canales
            filtered_channels = self._filter_and_rank_channels(all_channels, df)
            
            # Ajustar canales que han sido rotos
            adjusted_channels = self._adjust_broken_channels(filtered_channels, df)
            
            # Seleccionar solo los más relevantes
            final_channels = self._select_most_relevant(adjusted_channels, df)
            
            logger.info(f"Canales detectados: {len(final_channels)} de {len(all_channels)} candidatos")
            return final_channels
            
        except Exception as e:
            logger.error(f"Error en detección de canales: {e}")
            return []
    
    def _adaptive_parallel_detection(self, segment: pd.DataFrame, full_df: pd.DataFrame) -> List[Dict]:
        """Detección adaptativa de canales paralelos usando OHLC"""
        channels = []
        segment_start_in_full = len(full_df) - len(segment)
        
        # Búsqueda adaptativa con ventanas variables
        min_window = 15
        max_window = min(100, len(segment))
        
        # Buscar desde el final hacia atrás (más reciente primero)
        for end_idx in range(len(segment)-1, min_window, -5):
            for start_idx in range(max(0, end_idx - max_window), end_idx - min_window, 5):
                window_size = end_idx - start_idx
                
                if window_size < min_window:
                    continue
                
                sub_segment = segment.iloc[start_idx:end_idx]
                
                # Encontrar puntos de soporte y resistencia usando OHLC
                upper_points = self._find_resistance_points_ohlc(sub_segment)
                lower_points = self._find_support_points_ohlc(sub_segment)
                
                if len(upper_points) < 2 or len(lower_points) < 2:
                    continue
                
                # Ajustar líneas con RANSAC para robustez
                upper_line = self._fit_line_ransac(upper_points)
                lower_line = self._fit_line_ransac(lower_points)
                
                if upper_line is None or lower_line is None:
                    continue
                
                # Verificar paralelismo
                if self._check_parallelism(upper_line, lower_line):
                    # Validar toques
                    upper_touches = self._count_line_touches(sub_segment, upper_line, 'resistance')
                    lower_touches = self._count_line_touches(sub_segment, lower_line, 'support')
                    
                    if upper_touches >= self.min_touches and lower_touches >= self.min_touches:
                        channel = self._create_channel_dict(
                            'parallel', start_idx + segment_start_in_full,
                            end_idx + segment_start_in_full - 1,
                            upper_line, lower_line, sub_segment, full_df,
                            upper_touches + lower_touches
                        )
                        
                        if channel['quality_score'] > 0.5:
                            channels.append(channel)
        
        return channels
    
    def _adaptive_triangle_detection(self, segment: pd.DataFrame, full_df: pd.DataFrame) -> List[Dict]:
        """Detección adaptativa de triángulos"""
        channels = []
        segment_start_in_full = len(full_df) - len(segment)
        
        min_window = 20
        max_window = min(80, len(segment))
        
        for end_idx in range(len(segment)-1, min_window, -5):
            for start_idx in range(max(0, end_idx - max_window), end_idx - min_window, 5):
                window_size = end_idx - start_idx
                
                if window_size < min_window:
                    continue
                
                sub_segment = segment.iloc[start_idx:end_idx]
                
                upper_points = self._find_resistance_points_ohlc(sub_segment)
                lower_points = self._find_support_points_ohlc(sub_segment)
                
                if len(upper_points) < 2 or len(lower_points) < 2:
                    continue
                
                upper_line = self._fit_line_ransac(upper_points)
                lower_line = self._fit_line_ransac(lower_points)
                
                if upper_line is None or lower_line is None:
                    continue
                
                # Verificar convergencia/divergencia
                convergence_type = self._check_convergence(upper_line, lower_line, window_size)
                
                if convergence_type in ['ascending', 'descending', 'symmetric']:
                    upper_touches = self._count_line_touches(sub_segment, upper_line, 'resistance')
                    lower_touches = self._count_line_touches(sub_segment, lower_line, 'support')
                    
                    if upper_touches >= 2 and lower_touches >= 2:
                        channel = self._create_channel_dict(
                            convergence_type, start_idx + segment_start_in_full,
                            end_idx + segment_start_in_full - 1,
                            upper_line, lower_line, sub_segment, full_df,
                            upper_touches + lower_touches
                        )
                        
                        if channel['quality_score'] > 0.4:
                            channels.append(channel)
        
        return channels
    
    def _detect_wedge_patterns(self, segment: pd.DataFrame, full_df: pd.DataFrame) -> List[Dict]:
        """Detección de cuñas ascendentes y descendentes"""
        channels = []
        segment_start_in_full = len(full_df) - len(segment)
        
        min_window = 25
        max_window = min(70, len(segment))
        
        for window_size in range(min_window, max_window, 10):
            for start_idx in range(0, len(segment) - window_size, 5):
                sub_segment = segment.iloc[start_idx:start_idx + window_size]
                
                upper_points = self._find_resistance_points_ohlc(sub_segment)
                lower_points = self._find_support_points_ohlc(sub_segment)
                
                if len(upper_points) < 3 or len(lower_points) < 3:
                    continue
                
                upper_line = self._fit_line_ransac(upper_points)
                lower_line = self._fit_line_ransac(lower_points)
                
                if upper_line is None or lower_line is None:
                    continue
                
                # Detectar tipo de cuña
                upper_slope = upper_line[0]
                lower_slope = lower_line[0]
                
                wedge_type = None
                if upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope:
                    wedge_type = 'rising_wedge'
                elif upper_slope < 0 and lower_slope < 0 and upper_slope > lower_slope:
                    wedge_type = 'falling_wedge'
                
                if wedge_type:
                    upper_touches = self._count_line_touches(sub_segment, upper_line, 'resistance')
                    lower_touches = self._count_line_touches(sub_segment, lower_line, 'support')
                    
                    if upper_touches >= 2 and lower_touches >= 2:
                        channel = self._create_channel_dict(
                            wedge_type, start_idx + segment_start_in_full,
                            start_idx + window_size + segment_start_in_full - 1,
                            upper_line, lower_line, sub_segment, full_df,
                            upper_touches + lower_touches
                        )
                        
                        if channel['quality_score'] > 0.5:
                            channels.append(channel)
        
        return channels
    
    def _find_resistance_points_ohlc(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """Encuentra puntos de resistencia usando máximos y cierres"""
        points = []
        window = max(3, len(df) // 10)
        
        for i in range(window, len(df) - window):
            # Usar el máximo del período
            current_high = df['high'].iloc[i]
            
            # Verificar si es un máximo local
            if (current_high >= df['high'].iloc[i-window:i].max() and 
                current_high >= df['high'].iloc[i+1:i+window+1].max()):
                # Usar el máximo entre high y close para mayor precisión
                resistance_price = max(current_high, df['close'].iloc[i])
                points.append((i, resistance_price))
        
        return points
    
    def _find_support_points_ohlc(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """Encuentra puntos de soporte usando mínimos y cierres"""
        points = []
        window = max(3, len(df) // 10)
        
        for i in range(window, len(df) - window):
            # Usar el mínimo del período
            current_low = df['low'].iloc[i]
            
            # Verificar si es un mínimo local
            if (current_low <= df['low'].iloc[i-window:i].min() and 
                current_low <= df['low'].iloc[i+1:i+window+1].min()):
                # Usar el mínimo entre low y close para mayor precisión
                support_price = min(current_low, df['close'].iloc[i])
                points.append((i, support_price))
        
        return points
    
    def _fit_line_ransac(self, points: List[Tuple[int, float]], 
                        max_iterations: int = 100) -> Optional[Tuple[float, float]]:
        """Ajusta línea usando RANSAC para mayor robustez"""
        if len(points) < 2:
            return None
        
        try:
            X = np.array([[p[0]] for p in points])
            y = np.array([p[1] for p in points])
            
            # Usar RANSAC para ajuste robusto
            ransac = RANSACRegressor(random_state=42, max_trials=max_iterations)
            ransac.fit(X, y)
            
            # Obtener parámetros de la línea
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            
            return (slope, intercept)
        except:
            # Fallback a regresión simple si RANSAC falla
            try:
                x_vals = np.array([p[0] for p in points])
                y_vals = np.array([p[1] for p in points])
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                return (slope, intercept)
            except:
                return None
    
    def _check_parallelism(self, line1: Tuple[float, float], 
                          line2: Tuple[float, float]) -> bool:
        """Verifica paralelismo con tolerancia adaptativa"""
        slope1, slope2 = line1[0], line2[0]
        
        # Tolerancia relativa basada en la magnitud de las pendientes
        avg_slope = (abs(slope1) + abs(slope2)) / 2
        tolerance = max(0.1, min(0.3, avg_slope * 0.2))
        
        if abs(slope1) < 1e-6 and abs(slope2) < 1e-6:
            return True
        
        slope_diff = abs(slope1 - slope2)
        relative_diff = slope_diff / (max(abs(slope1), abs(slope2)) + 1e-6)
        
        return relative_diff < tolerance
    
    def _check_convergence(self, upper_line: Tuple[float, float], 
                          lower_line: Tuple[float, float],
                          window_size: int) -> str:
        """Determina el tipo de convergencia del canal"""
        upper_slope = upper_line[0]
        lower_slope = lower_line[0]
        
        # Calcular puntos de proyección
        upper_start = upper_line[1]
        upper_end = upper_slope * window_size + upper_line[1]
        lower_start = lower_line[1]
        lower_end = lower_slope * window_size + lower_line[1]
        
        # Calcular anchos del canal
        width_start = upper_start - lower_start
        width_end = upper_end - lower_end
        
        # Determinar tipo basado en convergencia
        convergence_rate = abs(width_end - width_start) / (abs(width_start) + 1e-6)
        
        if convergence_rate < 0.1:  # Casi paralelo
            return 'none'
        elif width_end < width_start * 0.7:  # Convergiendo
            if upper_slope > 0 and lower_slope > 0:
                return 'ascending'
            elif upper_slope < 0 and lower_slope < 0:
                return 'descending'
            else:
                return 'symmetric'
        elif width_end > width_start * 1.3:  # Divergiendo
            return 'expanding'
        else:
            return 'none'
    
    def _count_line_touches(self, df: pd.DataFrame, line: Tuple[float, float], 
                           line_type: str) -> int:
        """Cuenta toques precisos usando OHLC"""
        touches = 0
        slope, intercept = line
        
        for i in range(len(df)):
            expected_price = slope * i + intercept
            
            if line_type == 'resistance':
                # Para resistencia, verificar si el high toca la línea
                actual_price = df['high'].iloc[i]
                # También considerar el close si es mayor
                close_price = df['close'].iloc[i]
                
                if (abs(actual_price - expected_price) / expected_price < self.touch_tolerance or
                    abs(close_price - expected_price) / expected_price < self.touch_tolerance):
                    touches += 1
            else:  # support
                # Para soporte, verificar si el low toca la línea
                actual_price = df['low'].iloc[i]
                # También considerar el close si es menor
                close_price = df['close'].iloc[i]
                
                if (abs(actual_price - expected_price) / expected_price < self.touch_tolerance or
                    abs(close_price - expected_price) / expected_price < self.touch_tolerance):
                    touches += 1
        
        return touches
    
    def _create_channel_dict(self, channel_type: str, start_idx: int, end_idx: int,
                            upper_line: Tuple[float, float], lower_line: Tuple[float, float],
                            sub_segment: pd.DataFrame, full_df: pd.DataFrame,
                            total_touches: int) -> Dict:
        """Crea diccionario de canal con toda la información necesaria"""
        # Calcular valores en los extremos
        window_size = len(sub_segment)
        upper_start_y = upper_line[1]
        upper_end_y = upper_line[0] * (window_size - 1) + upper_line[1]
        lower_start_y = lower_line[1]
        lower_end_y = lower_line[0] * (window_size - 1) + lower_line[1]
        
        # Calcular calidad del canal
        quality_score = self._calculate_channel_quality(
            sub_segment, upper_line, lower_line, total_touches
        )
        
        # Verificar si está roto
        is_broken = self._is_channel_broken_ohlc(sub_segment, upper_line, lower_line)
        
        # Calcular relevancia al precio actual
        current_price = full_df['close'].iloc[-1]
        days_since_end = len(full_df) - 1 - end_idx
        relevance = quality_score * (1.0 / (1.0 + days_since_end * 0.05))
        
        return {
            'type': channel_type,
            'start_absolute_idx': start_idx,
            'end_absolute_idx': end_idx,
            'start_date': full_df.index[start_idx],
            'end_date': full_df.index[end_idx],
            'upper_start_y': upper_start_y,
            'upper_end_y': upper_end_y,
            'lower_start_y': lower_start_y,
            'lower_end_y': lower_end_y,
            'upper_slope': upper_line[0],
            'lower_slope': lower_line[0],
            'quality_score': quality_score,
            'relevance': relevance,
            'broken': is_broken,
            'total_touches': total_touches,
            'window_size': window_size
        }
    
    def _calculate_channel_quality(self, df: pd.DataFrame, 
                                  upper_line: Tuple[float, float],
                                  lower_line: Tuple[float, float],
                                  total_touches: int) -> float:
        """Calcula score de calidad del canal"""
        # Factor de toques
        touch_score = min(1.0, total_touches / (len(df) * 0.3))
        
        # Factor de contención de precios
        containment = 0
        for i in range(len(df)):
            upper_val = upper_line[0] * i + upper_line[1]
            lower_val = lower_line[0] * i + lower_line[1]
            
            # Verificar si el precio está contenido usando OHLC
            if lower_val <= df['low'].iloc[i] and df['high'].iloc[i] <= upper_val:
                containment += 1
        
        containment_score = containment / len(df)
        
        # Factor de ancho consistente
        width_start = upper_line[1] - lower_line[1]
        width_end = (upper_line[0] - lower_line[0]) * len(df) + width_start
        width_consistency = 1.0 - abs(width_end - width_start) / (abs(width_start) + 1e-6)
        width_consistency = max(0, min(1, width_consistency))
        
        # Combinar factores
        quality = (touch_score * 0.4 + containment_score * 0.4 + width_consistency * 0.2)
        
        return quality
    
    def _is_channel_broken_ohlc(self, df: pd.DataFrame, 
                                upper_line: Tuple[float, float],
                                lower_line: Tuple[float, float]) -> bool:
        """Verifica ruptura usando OHLC con criterios estrictos"""
        if len(df) < 5:
            return False
        
        recent_bars = min(5, len(df) // 4)
        tolerance = df['close'].std() * 0.15
        
        for i in range(len(df) - recent_bars, len(df)):
            upper_val = upper_line[0] * i + upper_line[1]
            lower_val = lower_line[0] * i + lower_line[1]
            
            # Ruptura alcista: close por encima de la resistencia
            if df['close'].iloc[i] > upper_val + tolerance:
                # Confirmar con el siguiente período si existe
                if i < len(df) - 1:
                    if df['low'].iloc[i+1] > upper_val:
                        return True
                else:
                    return True
            
            # Ruptura bajista: close por debajo del soporte
            if df['close'].iloc[i] < lower_val - tolerance:
                # Confirmar con el siguiente período si existe
                if i < len(df) - 1:
                    if df['high'].iloc[i+1] < lower_val:
                        return True
                else:
                    return True
        
        return False
    
    def _filter_and_rank_channels(self, channels: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """Filtra y rankea canales por calidad y relevancia"""
        if not channels:
            return []
        
        # Eliminar canales duplicados o muy similares
        filtered = []
        for channel in channels:
            is_duplicate = False
            for kept in filtered:
                if self._are_channels_similar(channel, kept):
                    # Mantener el de mayor calidad
                    if channel['quality_score'] > kept['quality_score']:
                        filtered.remove(kept)
                        filtered.append(channel)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(channel)
        
        # Ordenar por relevancia y calidad
        filtered.sort(key=lambda x: x['relevance'] * x['quality_score'], reverse=True)
        
        return filtered
    
    def _are_channels_similar(self, ch1: Dict, ch2: Dict, threshold: float = 0.7) -> bool:
        """Verifica si dos canales son similares"""
        # Overlap temporal
        time_overlap = (min(ch1['end_absolute_idx'], ch2['end_absolute_idx']) - 
                       max(ch1['start_absolute_idx'], ch2['start_absolute_idx']))
        
        if time_overlap <= 0:
            return False
        
        overlap_ratio = time_overlap / min(
            ch1['window_size'], ch2['window_size']
        )
        
        # Similaridad de pendientes
        upper_slope_diff = abs(ch1['upper_slope'] - ch2['upper_slope'])
        lower_slope_diff = abs(ch1['lower_slope'] - ch2['lower_slope'])
        
        avg_slope = (abs(ch1['upper_slope']) + abs(ch2['upper_slope']) + 
                    abs(ch1['lower_slope']) + abs(ch2['lower_slope'])) / 4 + 1e-6
        
        slope_similarity = 1 - (upper_slope_diff + lower_slope_diff) / (2 * avg_slope)
        
        return overlap_ratio > threshold and slope_similarity > 0.8
    
    def _adjust_broken_channels(self, channels: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """Ajusta canales que han sido rotos para incluir nuevos extremos"""
        adjusted = []
        
        for channel in channels:
            if channel['broken'] and channel['end_absolute_idx'] >= len(df) - 10:
                # Intentar extender el canal si es reciente y está roto
                extended = self._try_extend_channel(channel, df)
                if extended:
                    adjusted.append(extended)
                else:
                    adjusted.append(channel)
            else:
                adjusted.append(channel)
        
        return adjusted
    
    def _try_extend_channel(self, channel: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """Intenta extender un canal roto para incluir nuevos puntos"""
        try:
            start_idx = channel['start_absolute_idx']
            end_idx = min(len(df) - 1, channel['end_absolute_idx'] + 10)
            
            if end_idx <= channel['end_absolute_idx']:
                return None
            
            segment = df.iloc[start_idx:end_idx + 1]
            
            # Re-calcular puntos con el segmento extendido
            upper_points = self._find_resistance_points_ohlc(segment)
            lower_points = self._find_support_points_ohlc(segment)
            
            if len(upper_points) < 2 or len(lower_points) < 2:
                return None
            
            # Re-ajustar líneas
            upper_line = self._fit_line_ransac(upper_points)
            lower_line = self._fit_line_ransac(lower_points)
            
            if upper_line is None or lower_line is None:
                return None
            
            # Crear canal extendido
            total_touches = (self._count_line_touches(segment, upper_line, 'resistance') +
                           self._count_line_touches(segment, lower_line, 'support'))
            
            extended_channel = self._create_channel_dict(
                channel['type'], start_idx, end_idx,
                upper_line, lower_line, segment, df, total_touches
            )
            
            # Solo retornar si mejora la calidad
            if extended_channel['quality_score'] > channel['quality_score'] * 0.8:
                extended_channel['extended'] = True
                return extended_channel
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extendiendo canal: {e}")
            return None
    
    def _select_most_relevant(self, channels: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """Selecciona los canales más relevantes para el precio actual"""
        if not channels:
            return []
        
        current_idx = len(df) - 1
        current_price = df['close'].iloc[-1]
        
        # Separar canales activos y históricos
        active_channels = []
        recent_channels = []
        
        for channel in channels:
            # Canal activo: el precio actual está dentro del rango temporal
            if channel['end_absolute_idx'] >= current_idx - 5:
                # Verificar si el precio está cerca del canal
                projected_upper = (channel['upper_slope'] * (current_idx - channel['start_absolute_idx']) + 
                                 channel['upper_start_y'])
                projected_lower = (channel['lower_slope'] * (current_idx - channel['start_absolute_idx']) + 
                                 channel['lower_start_y'])
                
                price_position = (current_price - projected_lower) / (projected_upper - projected_lower + 1e-6)
                
                # El precio está dentro o cerca del canal proyectado
                if -0.2 <= price_position <= 1.2:
                    active_channels.append(channel)
            # Canal reciente pero no activo
            elif channel['end_absolute_idx'] >= current_idx - 20:
                recent_channels.append(channel)
        
        # Priorizar canales activos
        result = active_channels[:self.max_channels_to_display]
        
        # Si hay espacio, añadir canales recientes
        if len(result) < self.max_channels_to_display:
            remaining_slots = self.max_channels_to_display - len(result)
            result.extend(recent_channels[:remaining_slots])
        
        # Si aún no hay suficientes, tomar los de mejor calidad
        if len(result) < 2 and len(channels) > len(result):
            remaining_slots = 2 - len(result)
            for channel in channels:
                if channel not in result:
                    result.append(channel)
                    remaining_slots -= 1
                    if remaining_slots <= 0:
                        break
        
        return result

# ====================== VISUALIZACIÓN CUANTITATIVA CORREGIDA ======================
class QuantitativeVisualizer:
    """Visualización para análisis cuantitativo - VERSIÓN CORREGIDA"""
    
    def __init__(self):
        self.colors = {
            'bg': UNIFIED_COLORS['bg'],
            'grid': UNIFIED_COLORS['grid'],
            'text': UNIFIED_COLORS['text'],
            'support': UNIFIED_COLORS['support'],
            'resistance': UNIFIED_COLORS['resistance'],
            'bullish': UNIFIED_COLORS['bullish'],
            'bearish': UNIFIED_COLORS['bearish'],
            'neutral': UNIFIED_COLORS['neutral'],
            'macd': '#00d4ff',
            'macd_signal': '#ff3366',
            'macd_histogram': '#9333ea',
            'stoch_k': '#00d4ff',
            'stoch_d': '#9333ea',
            'ema_12': '#ffaa00',
            'ema_26': '#ff3366',
            'bb_upper': '#00d4ff',
            'bb_lower': '#ff3366',
            'channel_parallel': 'rgba(0, 212, 255, 0.3)',
            'channel_triangle': 'rgba(255, 170, 0, 0.3)'
        }
    
    def create_quantitative_chart(self, df: pd.DataFrame, sr_levels: pd.DataFrame,
                                 patterns: List[Dict], signal: Dict,
                                 indicators: List[str], show_sr: bool, 
                                 show_patterns: bool, show_channels: bool,
                                 channels: List[Dict]) -> go.Figure:
        """Crea gráfico cuantitativo profesional"""
        try:
            # Determinar número de filas según indicadores seleccionados
            rows = 1
            row_heights = [0.6]  # Reducimos la altura principal para hacer espacio
            subplot_titles = ['Gráfico de Velas']
            
            if 'Volume' in indicators:
                rows += 1
                row_heights.append(0.1)
                subplot_titles.append('Volumen')

            if 'TransactionalVolume' in indicators:
                rows += 1
                row_heights.append(0.1)
                subplot_titles.append('Volumen Transaccionado')                
            
            if 'RSI' in indicators:
                rows += 1
                row_heights.append(0.1)
                subplot_titles.append('RSI')
            
            if 'MACD' in indicators:
                rows += 1
                row_heights.append(0.1)
                subplot_titles.append('MACD')
            
            if 'Stochastic' in indicators:
                rows += 1
                row_heights.append(0.1)
                subplot_titles.append('Estocástico')
                
            if 'VolumeProfile' in indicators:
                rows += 1
                row_heights.append(0.2)  # Espacio para el perfil de volumen
                subplot_titles.append('Perfil de Volumen')
            
            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=row_heights,
                subplot_titles=subplot_titles
            )
            
            # Gráfico principal
            fig.add_trace(
                go.Candlestick(
                    x=df.index, open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name='Precio',
                    increasing=dict(line=dict(color=self.colors['bullish'])),
                    decreasing=dict(line=dict(color=self.colors['bearish']))
                ), row=1, col=1
            )
            
            # S/R levels
            if show_sr:
                self._add_sr_levels(fig, sr_levels, df)
            
            # Patterns
            if show_patterns:
                self._add_patterns(fig, patterns, df)
            
            # Signal
            if signal.get('action') != 'SIN DATOS':
                self._add_signal_markers(fig, signal, df)
            
            # Canales con coordenadas corregidas
            if show_channels:
                self._add_channels_corrected(fig, channels, df)

            # Indicadores
            self._add_technical_indicators(fig, df, indicators)
            
            # Layout profesional
            fig.update_layout(
                template='plotly_dark',
                height=1000,
                showlegend=True,
                hovermode='x unified',
                xaxis_rangeslider_visible=False,
                paper_bgcolor=self.colors['bg'],
                plot_bgcolor=UNIFIED_COLORS['bg'],
                font=dict(color=UNIFIED_COLORS['text']),
                # Nueva configuración de la leyenda
                legend=dict(
                    orientation="h",  # Horizontal
                    yanchor="bottom",
                    y=1.02,  # Justo encima del gráfico
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0, 0, 0, 0.5)',  # Fondo semitransparente
                    font=dict(size=10)  # Tamaño de fuente reducido
                ),
                margin=dict(t=100)  # Aumentar margen superior para acomodar la leyenda
            )
            if len(indicators) > 3:
                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.12,  # Más arriba para múltiples indicadores
                        xanchor="center",
                        x=0.5,
                        bgcolor='rgba(0, 0, 0, 0.7)',
                        font=dict(size=9),
                        itemsizing='constant'
                    ),
                    margin=dict(t=120)  # Margen superior aún mayor
                )
            return fig
            
        except Exception as e:
            logger.error(f"Error creando gráfico cuantitativo: {e}")
            return go.Figure()
    
    def _add_channels_corrected(self, fig, channels: List[Dict], df: pd.DataFrame):
        """Añade canales detectados al gráfico con coordenadas corregidas"""
        for i, channel in enumerate(channels):
            try:
                # Usar las fechas y valores Y pre-calculados
                start_date = channel['start_date']
                end_date = channel['end_date']
                
                upper_start_y = channel['upper_start_y']
                upper_end_y = channel['upper_end_y']
                lower_start_y = channel['lower_start_y']
                lower_end_y = channel['lower_end_y']
                
                # Color según el tipo de canal
                channel_colors = {
                    'parallel': 'rgba(0, 150, 255, 0.3)',
                    'ascending': 'rgba(255, 165, 0, 0.3)',
                    'descending': 'rgba(255, 100, 100, 0.3)',
                    'symmetric': 'rgba(150, 0, 255, 0.3)',
                    'rising_wedge': 'rgba(255, 50, 50, 0.3)',
                    'falling_wedge': 'rgba(50, 255, 50, 0.3)',
                    'expanding': 'rgba(255, 255, 0, 0.3)'
                }
                
                color = channel_colors.get(channel['type'], 'rgba(128, 128, 128, 0.3)')
                
                # Etiqueta descriptiva
                label = f"{channel['type'].replace('_', ' ').title()} "
                label += f"(Q:{channel['quality_score']:.2f}, T:{channel['total_touches']})"
                
                if channel.get('extended'):
                    label += " [Ext]"
                if channel.get('broken'):
                    label += " [Broken]"
                
                # Área del canal
                fig.add_trace(go.Scatter(
                    x=[start_date, end_date, end_date, start_date],
                    y=[upper_start_y, upper_end_y, lower_end_y, lower_start_y],
                    fill='toself',
                    fillcolor=color,
                    line=dict(width=0),
                    name=label,
                    opacity=0.2 if channel.get('broken') else 0.3,
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{channel['type']}</b><br>" +
                        f"Calidad: {channel['quality_score']:.2f}<br>" +
                        f"Toques: {channel['total_touches']}<br>" +
                        f"Superior: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    )
                ), row=1, col=1)
                
                # Línea superior del canal
                line_style = 'dot' if channel.get('broken') else 'dash'
                fig.add_trace(go.Scatter(
                    x=[start_date, end_date],
                    y=[upper_start_y, upper_end_y],
                    line=dict(
                        color=color.replace('0.3', '0.8'), 
                        width=2, 
                        dash=line_style
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=1)
                
                # Línea inferior del canal
                fig.add_trace(go.Scatter(
                    x=[start_date, end_date],
                    y=[lower_start_y, lower_end_y],
                    line=dict(
                        color=color.replace('0.3', '0.8'), 
                        width=2, 
                        dash=line_style
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=1)
                
                # Solo añadir etiqueta para los primeros 2 canales
                if i < 2:
                    # Posición de la etiqueta
                    label_y = upper_end_y if i == 0 else lower_end_y
                    label_text = channel['type'].replace('_', ' ').title()
                    
                    fig.add_annotation(
                        x=end_date,
                        y=label_y,
                        text=label_text,
                        showarrow=True,
                        arrowhead=2,
                        ax=40,
                        ay=-20 if i == 0 else 20,
                        font=dict(
                            color=color.replace('0.3', '0.9'), 
                            size=10
                        ),
                        row=1, col=1
                    )
                
            except Exception as e:
                logger.debug(f"Error añadiendo canal {channel.get('type', 'unknown')}: {e}")
                continue
    
    def _add_sr_levels(self, fig, sr_levels: pd.DataFrame, df: pd.DataFrame):
        """Añade niveles S/R cuantitativos"""
        if sr_levels.empty:
            return
        
        for _, level in sr_levels.head(10).iterrows():
            color = self.colors['support'] if level['type'] == 'support' else self.colors['resistance']
            
            fig.add_hline(
                y=level['price'],
                line_color=color,
                line_width=2 if level['proximity'] in ['immediate', 'near'] else 1,
                line_dash='solid' if level['proximity'] == 'immediate' else 'dash',
                opacity=0.8,
                row=1, col=1
            )
            
            # Etiqueta cuantitativa
            fig.add_annotation(
                x=df.index[-1], y=level['price'],
                text=f"${level['price']:.2f} ({level.get('quantitative_score', 1):.1f})",
                showarrow=True, arrowcolor=color,
                font=dict(color=color, size=9),
                row=1, col=1
            )
    
    def _add_patterns(self, fig, patterns: List[Dict], df: pd.DataFrame):
        """Añade patrones cuantitativos"""
        for pattern in patterns[:5]:
            if pattern.get('type') in ['uptrend', 'downtrend']:
                # Líneas de tendencia
                start_idx = pattern.get('start_idx', 0)
                end_idx = pattern.get('end_idx', len(df)-1)
                
                if start_idx < len(df) and end_idx < len(df):
                    color = self.colors['bullish'] if 'up' in pattern['type'] else self.colors['bearish']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[df.index[start_idx], df.index[end_idx]],
                            y=[df['close'].iloc[start_idx], df['close'].iloc[end_idx]],
                            mode='lines',
                            name=f"{pattern['type']} ({pattern.get('confidence', 0):.0%})",
                            line=dict(color=color, width=2),
                            opacity=0.8
                        ), row=1, col=1
                    )
            
            elif 'double' in pattern.get('type', '') or 'triple' in pattern.get('type', ''):
                # Niveles de reversión
                price_level = pattern.get('price_level')
                if price_level:
                    color = self.colors['bearish'] if 'resistance' in pattern.get('type', '') else self.colors['bullish']
                    
                    fig.add_hline(
                        y=price_level,
                        line_color=color,
                        line_width=2,
                        line_dash='dot',
                        opacity=0.7,
                        row=1, col=1
                    )
    
    def _add_signal_markers(self, fig, signal: Dict, df: pd.DataFrame):
        """Añade marcadores de señal cuantitativos"""
        last_price = df['close'].iloc[-1]
        last_date = df.index[-1]
        
        action = signal['action']
        
        if 'COMPRA' in action:
            color = '#00ff00' if 'FUERTE' in action else self.colors['bullish']
            symbol = 'triangle-up'
        elif 'VENTA' in action:
            color = '#ff0000' if 'FUERTE' in action else self.colors['bearish']
            symbol = 'triangle-down'
        else:
            color = self.colors['neutral']
            symbol = 'circle'
        
        fig.add_trace(
            go.Scatter(
                x=[last_date], y=[last_price],
                mode='markers',
                marker=dict(size=15, color=color, symbol=symbol, line=dict(color='white', width=2)),
                name=f'{action} ({signal.get("confidence", 0):.0%})',
                showlegend=True
            ), row=1, col=1
        )
        
        # Niveles de trading
        if signal.get('stop_loss'):
            fig.add_hline(y=signal['stop_loss'], line_color='red', line_dash='dash', row=1, col=1)
        
        if signal.get('take_profit'):
            fig.add_hline(y=signal['take_profit'], line_color='green', line_dash='dash', row=1, col=1)
    
    def _add_volume_profile(self, fig, df: pd.DataFrame, row: int):
        """Añade perfil de volumen al gráfico"""
        try:
            # Calcular el perfil de volumen
            price_range = df['high'].max() - df['low'].min()
            num_bins = 50  # Número de bins para el perfil
            bin_size = price_range / num_bins
            
            # Crear bins de precio
            bins = np.linspace(df['low'].min(), df['high'].max(), num_bins + 1)
            volume_profile = np.zeros(num_bins)
            
            # Calcular volumen por bin
            for i in range(len(df)):
                low = df['low'].iloc[i]
                high = df['high'].iloc[i]
                volume = df['volume'].iloc[i]
                
                # Distribuir el volumen entre los bins que toca esta vela
                low_bin = max(0, min(num_bins - 1, int((low - df['low'].min()) / bin_size)))
                high_bin = max(0, min(num_bins - 1, int((high - df['low'].min()) / bin_size)))
                
                bins_touched = high_bin - low_bin + 1
                if bins_touched > 0:
                    volume_per_bin = volume / bins_touched
                    for bin_idx in range(low_bin, high_bin + 1):
                        volume_profile[bin_idx] += volume_per_bin
            
            # Precio medio de cada bin
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Crear gráfico de perfil de volumen
            fig.add_trace(
                go.Bar(
                    x=volume_profile,
                    y=bin_centers,
                    orientation='h',
                    name='Perfil de Volumen',
                    marker_color='rgba(100, 100, 200, 0.6)',
                    hovertemplate='Volumen: %{x}<br>Precio: %{y:.2f}<extra></extra>'
                ), row=row, col=1
            )
            
            # Ajustar ejes
            fig.update_yaxes(range=[df['low'].min(), df['high'].max()], row=row, col=1)
            fig.update_xaxes(title_text="Volumen", row=row, col=1)
            
        except Exception as e:
            logger.error(f"Error añadiendo perfil de volumen: {e}")    

    def _add_technical_indicators(self, fig, df: pd.DataFrame, indicators: List[str]):
        """Añade indicadores técnicos"""
        current_row = 1
        
        # Indicadores en el gráfico principal
        if 'SMA_20' in indicators and 'close' in df.columns:
            sma20 = df['close'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=sma20, mode='lines', name='SMA 20',
                          line=dict(color='orange', width=1)), row=current_row, col=1
            )
        
        if 'SMA_50' in indicators and 'close' in df.columns:
            sma50 = df['close'].rolling(50).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=sma50, mode='lines', name='SMA 50',
                          line=dict(color='cyan', width=1)), row=current_row, col=1
            )
        
        # EMA - Asegurar cálculo si no existe
        if 'EMA_12' in indicators:
            # Calcular EMA 12 si no existe
            if 'ema_12' not in df.columns or df['ema_12'].isna().all():
                try:
                    # Calcular EMA 12 manualmente
                    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
                except Exception as e:
                    print(f"Error calculando EMA 12: {e}")
                    # Si falla, usar un cálculo simple
                    df['ema_12'] = df['close'].rolling(window=12).mean()
            
            if not df['ema_12'].isna().all():
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['ema_12'], mode='lines', name='EMA 12',
                            line=dict(color=self.colors['ema_12'], width=2)), row=current_row, col=1
                )
        
        if 'EMA_26' in indicators:
            # Calcular EMA 26 si no existe
            if 'ema_26' not in df.columns or df['ema_26'].isna().all():
                try:
                    # Calcular EMA 26 manualmente
                    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
                except Exception as e:
                    print(f"Error calculando EMA 26: {e}")
                    # Si falla, usar un cálculo simple
                    df['ema_26'] = df['close'].rolling(window=26).mean()
            
            if not df['ema_26'].isna().all():
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['ema_26'], mode='lines', name='EMA 26',
                            line=dict(color=self.colors['ema_26'], width=2)), row=current_row, col=1
                )

        if 'BollingerBands' in indicators and 'close' in df.columns:
            window = 20  # Periodo estándar
            # Calcular Bollinger Bands con TA-Lib
            upper_band, middle_band, lower_band = ta.BBANDS(
                df['close'].values,
                timeperiod=window,
                nbdevup=2,
                nbdevdn=2,
                matype=0  # 0 = SMA
            )

            # Añadir al DataFrame si lo necesitas
            df['bb_upper'] = upper_band
            df['bb_middle'] = middle_band
            df['bb_lower'] = lower_band

            # Añadir trazas al gráfico
            fig.add_trace(
                go.Scatter(x=df.index, y=upper_band, mode='lines', name='BB Superior',
                        line=dict(color='lightgray', width=1)), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=middle_band, mode='lines', name='BB Media',
                        line=dict(color='deepskyblue', width=2)), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=lower_band, mode='lines', name='BB Inferior',
                        line=dict(color='lightgray', width=1)), row=1, col=1
            )
        
        # Volumen (si está seleccionado)
        if 'Volume' in indicators and 'volume' in df.columns:
            current_row += 1
            colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
                     for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volumen',
                      marker_color=colors, opacity=0.7), row=current_row, col=1
            )
        # Volumen monetario (si está seleccionado)
        if 'TransactionalVolume' in indicators and 'volume' in df.columns and 'close' in df.columns:
            current_row += 1
            colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
                    for i in range(len(df))]

            monetary_volume = df['volume'] * df['close']  # Valor transaccionado

            fig.add_trace(
                go.Bar(x=df.index, y=monetary_volume, name='Volumen Monetario',
                    marker_color=colors, opacity=0.7),
                row=current_row, col=1
            )
        
        # RSI (si está seleccionado)
        if 'RSI' in indicators and 'rsi' in df.columns:
            current_row += 1
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], mode='lines', name='RSI',
                          line=dict(color='yellow', width=2)), row=current_row, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=current_row, col=1)
        
        # MACD (si está seleccionado)
        if 'MACD' in indicators:
            # Calcular MACD si no existe en el DataFrame
            if 'macd' not in df.columns or df['macd'].isna().all():
                try:
                    exp1 = df['close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = exp1 - exp2
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_histogram'] = df['macd'] - df['macd_signal']
                except:
                    pass
            
            if 'macd' in df.columns and not df['macd'].isna().all():
                current_row += 1
                
                # MACD line
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['macd'], mode='lines', name='MACD',
                            line=dict(color=self.colors['macd'], width=2)), row=current_row, col=1
                )
                
                # Signal line
                if 'macd_signal' in df.columns and not df['macd_signal'].isna().all():
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['macd_signal'], mode='lines', name='Señal MACD',
                                line=dict(color=self.colors['macd_signal'], width=2)), row=current_row, col=1
                    )
                
                # Histogram
                if 'macd_histogram' in df.columns and not df['macd_histogram'].isna().all():
                    colors_histogram = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
                    fig.add_trace(
                        go.Bar(x=df.index, y=df['macd_histogram'], name='Histograma MACD',
                            marker_color=colors_histogram, opacity=0.7), row=current_row, col=1
                    )
                
                fig.add_hline(y=0, line_color="white", opacity=0.5, row=current_row, col=1)
        
        # Estocástico (si está seleccionado)
        if 'Stochastic' in indicators:
            # Calcular Estocástico si no existe en el DataFrame
            if 'stoch_k' not in df.columns or df['stoch_k'].isna().all():
                try:
                    # Cálculo básico del Estocástico
                    low_min = df['low'].rolling(window=14).min()
                    high_max = df['high'].rolling(window=14).max()
                    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
                    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
                except:
                    pass
            
            if 'stoch_k' in df.columns and not df['stoch_k'].isna().all():
                current_row += 1
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['stoch_k'], mode='lines', name='Estocástico %K',
                            line=dict(color=self.colors['stoch_k'], width=2)), row=current_row, col=1
                )
                
                if 'stoch_d' in df.columns and not df['stoch_d'].isna().all():
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['stoch_d'], mode='lines', name='Estocástico %D',
                                line=dict(color=self.colors['stoch_d'], width=2)), row=current_row, col=1
                    )
                
                fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7, row=current_row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.7, row=current_row, col=1)

        # Perfil de Volumen (si está seleccionado)
        if 'VolumeProfile' in indicators:
            current_row += 1
            self._add_volume_profile(fig, df, current_row)

# ====================== DASHBOARD CUANTITATIVO ======================
class QuantitativeTradingDashboard:
    """Dashboard cuantitativo profesional"""
    
    def __init__(self):
        self.config = QuantitativeConfig()
        self.data_manager = QuantitativeDataManager(self.config)
        self.sr_analyzer = QuantitativeSRAnalyzer(self.config)
        self.pattern_analyzer = QuantitativePatternAnalyzer(self.config)
        self.signal_system = QuantitativeSignalSystem(self.config)
        self.visualizer = QuantitativeVisualizer()
        self.channel_detector = ChannelDetector(self.config)
        
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Layout del dashboard cuantitativo"""
        
        # Obtener tickers
        try:
            if os.path.exists(self.config.SCREENER_PATH):
                screener_df = pd.read_parquet(self.config.SCREENER_PATH)
                tickers = screener_df['ticker'].dropna().unique()[:6000]
            else:
                files = glob.glob(os.path.join(self.config.DATA_PATH, "*.parquet"))
                tickers = [os.path.basename(f).replace('.parquet', '') for f in files[:6000]]
        except:
            tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        ticker_options = [{'label': ticker, 'value': ticker} for ticker in tickers]
        
        # Aplicar estilos personalizados inline
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>Technical Trading System</title>
                {%favicon%}
                {%css%}
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Rajdhani:wght@300;400;500;600;700&display=swap" rel="stylesheet">
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body {
                        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
                        background: #0f1419;
                        color: #e8eaed;
                        line-height: 1.6;
                    }
                    
                    .container-fluid {
                        background: #0f1419 !important;
                        padding: 20px;
                    }
                    
                    .card {
                        background: #1a1f2e !important;
                        border: 1px solid #2a2f3e !important;
                        border-radius: 8px;
                        color: #e8eaed !important;
                    }
                    
                    .card-header {
                        background: #00d4ff !important;
                        border-bottom: 1px solid #2a2f3e !important;
                        color: #00d4ff !important;
                    }
                    
                    .btn-success {
                        background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
                        border: none !important;
                        font-weight: 500;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }
                    
                    .btn-success:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3) !important;
                    }
                    
                    /* Dropdowns */
                    .Select-control {
                        background: #242938 !important;
                        border: 1px solid #2a2f3e !important;
                        color: #e8eaed !important;
                    }
                    
                    .Select-menu-outer {
                        background: #242938 !important;
                        border: 1px solid #2a2f3e !important;
                    }
                    
                    .Select-option {
                        background: #242938 !important;
                        color: #e8eaed !important;
                    }
                    
                    .Select-option:hover {
                        background: #00d4ff !important;
                        color: #0f1419 !important;
                    }
                    
                    .Select-value-label {
                        color: white !important;
                    }

                    /* Estilo para el texto de las opciones en el menú desplegable */
                    .VirtualizedSelectOption {
                        color: white !important;
                    }

                    /* Estilo para el input de búsqueda */
                    .Select-input input {
                        color: white !important;
                    }

                    /* Estilo para el placeholder */
                    .Select-placeholder {
                        color: #e8eaed !important;
                    }

                    /* Labels */
                    label {
                        color: #9ca3af !important;
                        font-weight: 500 !important;
                        text-transform: uppercase !important;
                        font-size: 0.8rem !important;
                        letter-spacing: 1px !important;
                    }

                    /* Footer */
                    .footer {
                        text-align: center;
                        padding: 30px;
                        margin-top: 40px;
                        border-top: 1px solid #2a2f3e;
                    }                    

                    /* Scrollbar */
                    ::-webkit-scrollbar {
                        width: 10px;
                        height: 10px;
                    }
                    
                    ::-webkit-scrollbar-track {
                        background: #1a1f2e;
                        border-radius: 4px;
                    }
                    
                    ::-webkit-scrollbar-thumb {
                        background: #00d4ff;
                        border-radius: 4px;
                    }
                    
                    ::-webkit-scrollbar-thumb:hover {
                        background: #0099cc;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        self.app.layout = html.Div([
            # Header principal
            html.Div([
                html.H1("TECHNICAL TRADING SYSTEM", 
                        style={
                            'color': UNIFIED_COLORS['primary'],
                            'fontSize': '2.8rem',
                            'fontWeight': '300',
                            'margin': '0',
                            'textAlign': 'center',
                            'letterSpacing': '4px',
                            'textTransform': 'uppercase',
                            'fontFamily': 'Rajdhani, monospace'
                        }),
                html.Div("MACHINE LEARNING • ADVANCED ANALYTICS • ALGORITHMIC SIGNALS", 
                        style={
                            'color': UNIFIED_COLORS['text_secondary'],
                            'fontSize': '0.9rem',
                            'textAlign': 'center',
                            'marginTop': '8px',
                            'letterSpacing': '2px',
                            'fontWeight': '400'
                        })
            ], style={
                'padding': '30px',
                'borderBottom': f'1px solid {UNIFIED_COLORS["primary"]}',
                'marginBottom': '25px',
                'background': f'linear-gradient(180deg, {UNIFIED_COLORS["bg_secondary"]} 0%, {UNIFIED_COLORS["bg"]} 100%)'
            }),
            
            # Controles
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Ticker:", style={'fontWeight': 'bold'}),
                                    dcc.Dropdown(
                                        id='ticker-dropdown',
                                        options=ticker_options,
                                        value=ticker_options[0]['value'] if ticker_options else 'AAPL',
                                        searchable=True,
                                        style={'color': 'white'}
                                    )
                                ], md=3),
                                
                                dbc.Col([
                                    dbc.Label("Granularidad:", style={'fontWeight': 'bold'}),
                                    dcc.Dropdown(
                                        id='granularity',
                                        options=[
                                            {'label': 'Diario', 'value': 'diario'},
                                            {'label': 'Semanal', 'value': 'semanal'},
                                            {'label': 'Mensual', 'value': 'mensual'}
                                        ],
                                        value='diario',
                                        clearable=False,
                                        style={'color': 'white'}
                                    )
                                ], md=2),
                                
                                dbc.Col([
                                    dbc.Label("Indicadores:", style={'fontWeight': 'bold'}),
                                    dbc.Checklist(
                                        id='indicators',
                                        options=[
                                            {'label': ' SMA 20', 'value': 'SMA_20'},
                                            {'label': ' SMA 50', 'value': 'SMA_50'},
                                            {'label': ' EMA 12', 'value': 'EMA_12'},
                                            {'label': ' EMA 26', 'value': 'EMA_26'},
                                            {'label': ' Bollinger Bands', 'value': 'BollingerBands'},
                                            {'label': ' RSI', 'value': 'RSI'},
                                            {'label': ' MACD', 'value': 'MACD'},
                                            {'label': ' Estocástico', 'value': 'Stochastic'},
                                            {'label': ' Volumen', 'value': 'Volume'},
                                            {'label': ' Volumen Transaccionado', 'value': 'TransactionalVolume'},
                                            {'label': ' Volume Profile', 'value': 'VolumeProfile'}  # Añadido aquí
                                        ],
                                        value=['Volume', 'RSI'],
                                        inline=True
                                    )
                                ], md=4),
                              
                                dbc.Col([
                                    dbc.Label("Visualización:", style={'fontWeight': 'bold'}),
                                    dbc.Checklist(
                                        id='visualization-options',
                                        options=[
                                            {'label': ' Soportes/Resistencias', 'value': 'show_sr'},
                                            {'label': ' Patrones ML', 'value': 'show_patterns'},
                                            {'label': ' Canales', 'value': 'show_channels'} 
                                        ],
                                        value=['show_sr'],
                                        inline=False
                                    )
                                ], md=2),

                                dbc.Col([
                                    dbc.Button("ANALIZAR", id='analyze-btn',
                                            color="success", size="lg",
                                            style={
                                                'height': '50px',
                                                'width': '150px',
                                                'fontWeight': 'bold',
                                                'fontSize': '0.9rem'
                                            })
                                ], md=1, className="d-flex align-items-center")
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Panel de señal
            dbc.Card([
                html.Div(
                    dbc.CardHeader(html.H5("Señal Cuantitativa")),
                    id="signal-header",
                    n_clicks=0,
                    style={"cursor": "pointer"}
                ),
                dbc.Collapse(
                    id="signal-collapse",
                    is_open=True,
                    children=[
                        dbc.CardBody([
                            html.Div(id='signal-panel')
                        ])
                    ]
                )
            ], className="dashboard-card"),
            
            # Gráfico
            dbc.Row([
                dbc.Col([
                    dcc.Loading([
                        dcc.Graph(id='main-chart', style={'height': '1000px'})  # Aumentada la altura
                    ])
                ])
            ]),
            
            # Tablas
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Niveles S/R Cuantitativos")),
                        dbc.CardBody([html.Div(id='sr-table')])
                    ])
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Patrones ML")),
                        dbc.CardBody([html.Div(id='patterns-table')])
                    ])
                ], md=6)
            ], className="mt-4")
            
        ])
        
        # Footer profesional
        html.Div([
            html.Hr(style={
                'border': f'1px solid {UNIFIED_COLORS["border"]}', 
                'margin': '40px 0 20px 0',
                'opacity': '0.5'
            }),
            html.Div([
                html.P("QUANTITATIVE ANALYTICS PLATFORM", style={
                    'fontSize': '0.9rem',
                    'fontWeight': '600',
                    'letterSpacing': '1px',
                    'margin': '0',
                    'color': UNIFIED_COLORS['primary']
                }),
                html.P("Developed by Alejandro Moreno • Institutional Investment Solutions", style={
                    'fontSize': '0.8rem',
                    'fontWeight': '400',
                    'margin': '5px 0 0 0',
                    'color': UNIFIED_COLORS['text_secondary']
                })
            ], style={'textAlign': 'center', 'padding': '20px'})
        ], style={'background': UNIFIED_COLORS['bg'], 'minHeight': '100vh'})


    def setup_callbacks(self):
        """Callbacks del dashboard"""
        
        @self.app.callback(
            [Output('main-chart', 'figure'),
             Output('signal-panel', 'children'),
             Output('sr-table', 'children'),
             Output('patterns-table', 'children')],
            [Input('analyze-btn', 'n_clicks')],
            [State('ticker-dropdown', 'value'),
             State('granularity', 'value'),
             State('indicators', 'value'),
             State('visualization-options', 'value')]
        )
        
        def analyze_ticker(n_clicks, ticker, granularity, indicators, visualization_options):
            if not ticker:
                return go.Figure(), "Selecciona un ticker", "", ""
            
            try:
                # Cargar datos
                df = self.data_manager.load_and_preprocess(ticker, granularity)
                
                if df.empty:
                    return go.Figure(), f"No hay datos para {ticker}", "", ""
                
                # Análisis cuantitativo
                sr_levels = self.sr_analyzer.analyze_support_resistance(df, granularity)
                patterns = self.pattern_analyzer.detect_patterns(df, granularity)
                signal = self.signal_system.generate_quantitative_signals(df, sr_levels, patterns)
                
                # Detección de canales (nuevo)
                show_channels = 'show_channels' in visualization_options
                channels = self.channel_detector.detect_channels(df) if show_channels else []                

                # Crear visualizaciones
                show_sr = 'show_sr' in visualization_options
                show_patterns = 'show_patterns' in visualization_options
                
                fig = self.visualizer.create_quantitative_chart(
                    df, sr_levels, patterns, signal, indicators, 
                    show_sr, show_patterns, show_channels, channels  # Nuevo parámetro
                )

                signal_panel = self._create_signal_panel(signal, ticker)
                sr_table = self._create_sr_table(sr_levels, df['close'].iloc[-1] if not df.empty else 0)
                patterns_table = self._create_patterns_table(patterns)
                
                return fig, signal_panel, sr_table, patterns_table
                
            except Exception as e:
                logger.error(f"Error en análisis: {e}")
                return go.Figure(), f"Error: {str(e)}", "", ""
            
        @self.app.callback(
            Output("signal-collapse", "is_open"),
            Input("signal-header", "n_clicks"),
            State("signal-collapse", "is_open")
        )
        def toggle_signal_panel(n_clicks, is_open):
            if n_clicks:
                return not is_open
            return is_open

    def _create_signal_panel(self, signal: Dict, ticker: str):
        """Panel de señal cuantitativa"""
        if signal['action'] == 'SIN DATOS':
            return dbc.Alert("Sin datos suficientes", color="warning")
        
        action = signal['action']
        confidence = signal.get('confidence', 0)
        score = signal.get('score', 0)
        
        # Color según acción
        if 'COMPRA' in action:
            color = 'success'
            icon = '📈'
        elif 'VENTA' in action:
            color = 'danger'
            icon = '📉'
        else:
            color = 'warning'
            icon = '⚡'
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{icon} {action}", className=f"text-{color}"),
                        html.P(f"Confianza: {confidence:.0%}"),
                        html.P(f"Score Cuantitativo: {score:.3f}"),
                        html.P(f"Precio Actual: ${signal.get('current_price', 0):.2f}")
                    ])
                ], color=color, outline=True)
            ], md=4),
            
            dbc.Col([
                html.H6("Niveles de Trading:"),
                html.P(f"📍 Entrada: ${signal.get('entry_price', 0):.2f}"),
                html.P(f"🛑 Stop Loss: ${signal.get('stop_loss', 0):.2f}" if signal.get('stop_loss') else "🛑 Stop Loss: N/A"),
                html.P(f"💰 Take Profit: ${signal.get('take_profit', 0):.2f}" if signal.get('take_profit') else "💰 Take Profit: N/A")
            ], md=4),
            
            dbc.Col([
                html.H6("Análisis Cuantitativo:"),
                html.P(signal.get('reasoning', 'N/A'), style={'fontSize': '12px'})
            ], md=4)
        ])
    
    def _create_sr_table(self, sr_levels: pd.DataFrame, current_price: float):
        """Tabla de S/R cuantitativos con estilos mejorados"""
        if sr_levels.empty:
            return "No se detectaron niveles S/R"
        
        # Ordenar por precio descendente
        sr_levels = sr_levels.sort_values('price', ascending=False)
        
        # Encontrar niveles adyacentes al precio actual
        above_current = sr_levels[sr_levels['price'] > current_price]
        below_current = sr_levels[sr_levels['price'] < current_price]
        
        support_text = "N/A"
        resistance_text = "N/A"
        
        if not below_current.empty:
            closest_below = below_current.iloc[0]
            below_text = f"{closest_below['price']:.2f}"
        if not above_current.empty:
            closest_above = above_current.iloc[-1]
            above_text = f"{closest_above['price']:.2f}"
        
        range_info = html.Div([
            html.P(f"Precio actual de ${current_price:.2f} entre {below_text} y {above_text}",
                  style={'fontWeight': 'bold', 'marginBottom': '10px'})
        ])
        
        # Calcular score máximo para normalizar
        max_score = sr_levels['quantitative_score'].max() if not sr_levels.empty else 1
        
        rows = []
        for _, level in sr_levels.iterrows():
            # Determinar color base
            if level['type'] == 'support':
                base_color = (0, 255, 0)  # Verde para soportes
            else:
                base_color = (255, 0, 0)  # Rojo para resistencias
            
            # Calcular intensidad basada en el score (normalizado)
            intensity = 0.3 + 0.7 * (level['quantitative_score'] / max_score)
            color = f'rgba({base_color[0]}, {base_color[1]}, {base_color[2]}, {intensity})'
            
            rows.append(
                html.Tr([
                    html.Td(f"${level['price']:.2f}"),
                    html.Td(level['type'].title()),
                    html.Td(f"{level.get('quantitative_score', 0):.2f}"),
                    html.Td(level.get('proximity', 'N/A').title())
                ], style={'color': color, 'fontWeight': 'bold'})
            )
        
        table = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Precio"),
                    html.Th("Tipo"),
                    html.Th("Score"),
                    html.Th("Proximidad")
                ])
            ]),
            html.Tbody(rows)
        ], bordered=True, hover=True, size="sm")
        
        return html.Div([range_info, table])
    
    def _create_patterns_table(self, patterns: List[Dict]):
        """Tabla de patrones ML"""
        if not patterns:
            return "No se detectaron patrones"
        
        rows = []
        for pattern in patterns[:8]:
            rows.append(
                html.Tr([
                    html.Td(pattern.get('type', 'N/A').replace('_', ' ').title()),
                    html.Td(f"{pattern.get('confidence', 0):.0%}"),
                    html.Td(f"{pattern.get('quantitative_score', 0):.2f}"),
                    html.Td(f"{pattern.get('price_level', 0):.2f}"),
                    html.Td("✅" if pattern.get('active') else "historic")
                ])
            )
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Patrón"),
                    html.Th("Confianza"),
                    html.Th("Score ML"),
                    html.Th("Price"),
                    html.Th("Estado")
                ])
            ]),
            html.Tbody(rows)
        ], bordered=True, hover=True, size="sm")
    
    def run(self, debug=False, port=8051):
        """Ejecuta el dashboard cuantitativo"""
        print("=" * 80)
        print("📊 TECHNICAL TRADING SYSTEM")
        print("=" * 80)
        print("🤖 CARACTERÍSTICAS PROFESIONALES:")
        print("   • Machine Learning: DBSCAN, K-means, Isolation Forest")
        print("   • Análisis Estadístico: RANSAC, GMM, PCA")
        print("   • S/R Cuantitativos: Clustering + Análisis de volumen")
        print("   • Patrones ML: Detección automática de tendencias")
        print("   • Señales Ensemble: Múltiples algoritmos combinados")
        print("   • Indicadores Técnicos: SMA, EMA, Bollinger, RSI, MACD, Estocástico")
        print("   • Canales de Tendencia: Detección automática de canales paralelos y triangulares")
        print("👨‍💻 Developed by Alejandro Moreno")
        print("=" * 80)
        print(f"🌐 Dashboard disponible en: http://localhost:{port}")
        print("=" * 80)
        
        self.app.run(debug=False, host="0.0.0.0", port=8051)

# ====================== MAIN ======================
if __name__ == "__main__":
    try:
        dashboard = QuantitativeTradingDashboard()
        dashboard.app.run(debug=False, host="0.0.0.0", port=8051)
    except Exception as e:
        print(f"❌ Error: {e}")

        input("Presiona Enter para salir...")
