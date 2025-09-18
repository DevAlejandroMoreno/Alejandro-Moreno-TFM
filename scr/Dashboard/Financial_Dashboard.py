import dash
from dash import dcc, html, Input, Output, callback, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import concurrent.futures
from functools import lru_cache
import logging
from scipy import stats
import random

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Obtener el directorio del archivo actual
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio raíz del proyecto (dos niveles arriba desde scr/Dashboard/)
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")

# Configuración de rutas relativas
BASE_PATH = os.path.join(PROJECT_ROOT, "data")
ANNUAL_DATA_PATH = os.path.join(BASE_PATH, "Fundamental_Data_and_Ratios", "Anual")
QUARTERLY_DATA_PATH = os.path.join(BASE_PATH, "Fundamental_Data_and_Ratios", "Trimestral")
SCREENER_PATH = os.path.join(BASE_PATH, "Ticker_List", "screener.parquet")

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
    'accent': '#9333ea',        # Morado acento
    'border': '#2a2f3e'         # Bordes
}

# Diccionario con explicaciones de métricas para inversores profesionales
METRIC_INFO = {
    'Revenue': {'desc': 'Total income from operations. Higher is better - indicates company growth and market demand.', 'better': 'Higher'},
    'Net Income': {'desc': 'Bottom-line profit after all expenses. Higher is better - shows profitability and efficiency.', 'better': 'Higher'},
    'EBITDA': {'desc': 'Earnings before interest, taxes, depreciation, amortization. Higher indicates operational strength.', 'better': 'Higher'},
    'Operating Income': {'desc': 'Profit from core business operations. Higher shows operational efficiency and profitability.', 'better': 'Higher'},
    'Total Assets': {'desc': 'All company resources and investments. Growth indicates expansion, but efficiency matters more.', 'better': 'Growth'},
    'Share Holder Equity': {'desc': 'Shareholders ownership value. Higher indicates stronger balance sheet and lower leverage.', 'better': 'Higher'},
    'Long Term Debt': {'desc': 'Debt obligations beyond one year. Lower is generally better, but depends on industry and interest rates.', 'better': 'Lower'},
    'Cash On Hand': {'desc': 'Liquid assets available. Higher provides flexibility but excess cash may indicate lack of investment opportunities.', 'better': 'Moderate'},
    'Shares Outstanding': {'desc': 'Total shares issued by company. Decreasing may indicate share buybacks, which can boost EPS.', 'better': 'Stable/Decreasing'},
    'EPS - Earnings Per Share': {'desc': 'Net income divided by shares outstanding. Higher indicates greater profitability per share.', 'better': 'Higher'},
    'Revenue/Share': {'desc': 'Revenue per share outstanding. Higher indicates company generates more sales per share.', 'better': 'Higher'},
    'Cash/Share': {'desc': 'Cash per share outstanding. Higher provides more safety but may indicate inefficient capital allocation.', 'better': 'Moderate'},
    'Gross Margin': {'desc': 'Gross profit as % of revenue. Higher indicates pricing power and operational efficiency.', 'better': 'Higher'},
    'Operating Margin': {'desc': 'Operating income as % of revenue. Higher shows better cost control and operational efficiency.', 'better': 'Higher'},
    'Net Profit Margin': {'desc': 'Net income as % of revenue. Higher indicates overall profitability and financial efficiency.', 'better': 'Higher'},
    'EBITDA Margin': {'desc': 'EBITDA as % of revenue. Higher shows strong operational performance before financing decisions.', 'better': 'Higher'},
    'P/E': {'desc': 'Price-to-earnings ratio. Lower may indicate undervaluation, but very low could signal problems.', 'better': 'Moderate'},
    'P/B': {'desc': 'Price-to-book ratio. Lower may indicate undervaluation relative to assets.', 'better': 'Lower'},
    'P/S': {'desc': 'Price-to-sales ratio. Lower may indicate undervaluation relative to revenue generation.', 'better': 'Lower'},
    'EV/EBITDA': {'desc': 'Enterprise value to EBITDA. Lower indicates potentially undervalued company.', 'better': 'Lower'},
    'Book Value Per Share': {'desc': 'Equity per share. Higher indicates more tangible value backing each share.', 'better': 'Higher'},
    'MarketCap': {'desc': 'Total market value of all shares. Indicates company size and market perception.', 'better': 'Growth'},
    'Debt/EBITDA': {'desc': 'Debt relative to earnings. Lower indicates better ability to service debt. Above 4x often concerning.', 'better': 'Lower'},
    'Debt/Assets': {'desc': 'Debt as % of total assets. Lower indicates stronger balance sheet and less leverage risk.', 'better': 'Lower'},
    'Debt/Equity Ratio': {'desc': 'Total debt relative to equity. Lower indicates less financial leverage and risk.', 'better': 'Lower'},
    'Net Debt': {'desc': 'Total debt minus cash. Lower/negative is better - indicates net cash position.', 'better': 'Lower'},
    'Current Ratio': {'desc': 'Current assets / current liabilities. Above 1.0 indicates ability to pay short-term obligations.', 'better': '>1.0'},
    'Working Capital': {'desc': 'Current assets minus current liabilities. Positive value suggests short-term financial health, but excessive levels may indicate inefficient capital use.','better': 'Balanced'},
    'Cash Ratio': {'desc': 'Cash / current liabilities. Higher indicates better liquidity but may suggest inefficient cash use.', 'better': 'Moderate'},
    'ROE - Return On Equity': {'desc': 'Net income / shareholders equity. Higher indicates more efficient use of shareholder capital.', 'better': 'Higher'},
    'ROA - Return On Assets': {'desc': 'Net income / total assets. Higher shows more efficient use of company assets.', 'better': 'Higher'},
    'ROI - Return On Investment': {'desc': 'Return relative to investment. Higher indicates better capital efficiency and profitability.', 'better': 'Higher'},
    'Asset Turnover': {'desc': 'Revenue / total assets. Higher indicates more efficient asset utilization.', 'better': 'Higher'},
    'Inventory Turnover Ratio': {'desc': 'How quickly inventory is sold. Higher indicates efficient inventory management.', 'better': 'Higher'},
    'Receiveable Turnover': {'desc': 'How quickly receivables are collected. Higher indicates efficient credit management.', 'better': 'Higher'},
    'Days Sales In Receivables': {'desc': 'Average days to collect receivables. Lower indicates faster cash collection.', 'better': 'Lower'},
    'Operating Cash Flow Per Share': {'desc': 'Operating cash flow per share. Higher indicates strong cash generation per share.', 'better': 'Higher'},
    'Free Cash Flow Per Share': {'desc': 'Free cash flow per share. Higher indicates more cash available for dividends, buybacks, growth.', 'better': 'Higher'}
}

def format_large_number(value):
    """Format large numbers with T, B, M, K suffixes - FIXED VERSION"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    try:
        abs_value = abs(float(value))
        if abs_value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif abs_value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs_value >= 1e6:
            return f"${value/1e6:.1f}M"
        elif abs_value >= 1e3:
            return f"${value/1e3:.0f}K"
        else:
            return f"${value:.0f}"
    except:
        return "N/A"

def format_shares_number(value):
    """Format share numbers without $ symbol - FIXED VERSION"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    try:
        abs_value = abs(float(value))
        if abs_value >= 1e9:
            return f"{value/1e9:.2f}B"
        elif abs_value >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs_value >= 1e3:
            return f"{value/1e3:.0f}K"
        else:
            return f"{value:.0f}"
    except:
        return "N/A"

class ProfessionalInvestmentAnalyzer:
    def __init__(self):
        self.screener_data = None
        self.industry_cache = {}
        self.ticker_cache = {}
        self.load_screener_data()
        
    def load_screener_data(self):
        """Carga los datos del screener con información de empresas"""
        try:
            if os.path.exists(SCREENER_PATH):
                self.screener_data = pd.read_parquet(SCREENER_PATH)
                logging.info(f"Screener data loaded: {len(self.screener_data)} companies")
            else:
                self.screener_data = pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading screener data: {e}")
            self.screener_data = pd.DataFrame()

    @lru_cache(maxsize=300)
    def load_ticker_data(self, ticker, period='annual'):
        """Carga datos de un ticker específico con cache optimizado"""
        try:
            path = ANNUAL_DATA_PATH if period == 'annual' else QUARTERLY_DATA_PATH
            file_path = os.path.join(path, f"{ticker}.parquet")
            
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                return self.calculate_professional_metrics(df)
            else:
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading ticker {ticker}: {e}")
            return pd.DataFrame()

    def calculate_professional_metrics(self, df):
        """Calcula métricas profesionales adicionales"""
        metrics = df.copy()
        
        try:
            # Ratios de Valoración Fundamentales
            if 'Close' in df.columns and 'EPS - Earnings Per Share' in df.columns:
                metrics['P/E'] = df['Close'] / df['EPS - Earnings Per Share'].replace(0, np.nan)
            if 'MarketCap' in df.columns and 'Revenue' in df.columns:
                metrics['P/S'] = df['MarketCap'] / df['Revenue'].replace(0, np.nan)
            if 'MarketCap' in df.columns and 'Share Holder Equity' in df.columns:
                metrics['P/B'] = df['MarketCap'] / df['Share Holder Equity'].replace(0, np.nan)
            if 'MarketCap' in df.columns and 'EBITDA' in df.columns:
                metrics['EV/EBITDA'] = df['MarketCap'] / df['EBITDA'].replace(0, np.nan)

            # Métricas de Deuda Profesionales
            if 'Long Term Debt' in df.columns and 'EBITDA' in df.columns:
                metrics['Debt/EBITDA'] = df['Long Term Debt'] / df['EBITDA'].replace(0, np.nan)
            if 'Long Term Debt' in df.columns and 'Total Assets' in df.columns:
                metrics['Debt/Assets'] = (df['Long Term Debt'] / df['Total Assets']) * 100
            if 'Long Term Debt' in df.columns and 'Cash On Hand' in df.columns:
                metrics['Net Debt'] = df['Long Term Debt'] - df['Cash On Hand']
            
            # Interest Coverage Ratio (métrica profesional clave)
            if 'Operating Income' in df.columns and 'Income Taxes' in df.columns:
                interest_expense = df.get('Total Non-Operating Income/Expense', 0).abs()
                metrics['Interest Coverage'] = df['Operating Income'] / interest_expense.replace(0, np.nan)
            
            # Working Capital Metrics
            if 'Total Current Assets' in df.columns and 'Total Current Liabilities' in df.columns:
                metrics['Working Capital'] = df['Total Current Assets'] - df['Total Current Liabilities']
                metrics['Working Capital Ratio'] = metrics['Working Capital'] / df['Revenue'].replace(0, np.nan) * 100
            
            # Free Cash Flow (crítico para inversores)
            if 'Cash Flow From Operating Activities' in df.columns and 'Net Change In Property, Plant, And Equipment' in df.columns:
                capex = df['Net Change In Property, Plant, And Equipment'].abs()
                metrics['Free Cash Flow'] = df['Cash Flow From Operating Activities'] - capex
                if 'Shares Outstanding' in df.columns:
                    metrics['Free Cash Flow Per Share'] = metrics['Free Cash Flow'] / df['Shares Outstanding'].replace(0, np.nan)

            # Métricas Per Share Profesionales
            if 'Revenue' in df.columns and 'Shares Outstanding' in df.columns:
                metrics['Revenue/Share'] = df['Revenue'] / df['Shares Outstanding'].replace(0, np.nan)
            if 'Cash On Hand' in df.columns and 'Shares Outstanding' in df.columns:
                metrics['Cash/Share'] = df['Cash On Hand'] / df['Shares Outstanding'].replace(0, np.nan)
            if 'Total Assets' in df.columns and 'Shares Outstanding' in df.columns:
                metrics['Assets/Share'] = df['Total Assets'] / df['Shares Outstanding'].replace(0, np.nan)

            # Dividend Yield si hay dividendos
            if 'Common Stock Dividends Paid' in df.columns and 'Shares Outstanding' in df.columns and 'Close' in df.columns:
                annual_dividend = df['Common Stock Dividends Paid'].abs() / df['Shares Outstanding'].replace(0, np.nan)
                metrics['Dividend Yield'] = (annual_dividend / df['Close']) * 100

            # Métricas de Eficiencia Avanzadas
            if 'Revenue' in df.columns and 'Total Assets' in df.columns:
                prev_assets = df['Total Assets'].shift(1)
                avg_assets = (df['Total Assets'] + prev_assets) / 2
                metrics['Asset Turnover (Avg)'] = df['Revenue'] / avg_assets.replace(0, np.nan)

        except Exception as e:
            logging.error(f"Error calculating professional metrics: {e}")
        
        return metrics

    def predict_metric_advanced(self, series, periods=3):
        """Predicción ultra-conservadora para evitar valores disparatados"""
        clean_series = series.dropna()
        if len(clean_series) < 4:
            return [np.nan] * periods
            
        try:
            # Análisis estadístico de la serie
            last_value = clean_series.iloc[-1]
            series_mean = clean_series.mean()
            series_std = clean_series.std()
            
            # Calcular crecimiento histórico promedio (más robusto)
            growth_rates = clean_series.pct_change().dropna()
            if len(growth_rates) > 0:
                # Filtrar crecimientos extremos
                growth_rates = growth_rates[abs(growth_rates) < 0.5]  # Max 50% growth per period
                avg_growth = growth_rates.median()  # Usar mediana en lugar de media
            else:
                avg_growth = 0.02
            
            # Limitar crecimiento anual a rangos realistas
            if avg_growth > 0.25:  # Max 25% growth per year
                avg_growth = 0.25
            elif avg_growth < -0.15:  # Max 15% decline per year  
                avg_growth = -0.15
                
            # Calcular tendencia lineal solo si es estable
            X = np.arange(len(clean_series)).reshape(-1, 1)
            y = clean_series.values
            model = LinearRegression()
            model.fit(X, y)
            r_squared = model.score(X, y)
            
            predictions = []
            current_value = last_value
            
            for i in range(periods):
                if r_squared > 0.7:  # Solo usar regresión si correlación es fuerte
                    future_X = np.array([[len(clean_series) + i]])
                    pred = model.predict(future_X)[0]
                    
                    # Validación extrema: no más de 30% cambio vs último valor
                    max_change = abs(last_value * 0.3)
                    if abs(pred - last_value) > max_change:
                        pred = current_value * (1 + avg_growth)
                else:
                    # Usar crecimiento promedio histórico
                    pred = current_value * (1 + avg_growth)
                
                # Validación final: valor debe estar dentro de 3 desviaciones estándar
                if series_std > 0:
                    if pred > series_mean + 3 * series_std:
                        pred = series_mean + 2 * series_std
                    elif pred < series_mean - 3 * series_std:
                        pred = series_mean - 2 * series_std
                
                # No permitir valores negativos para métricas que no pueden serlo
                if last_value > 0 and pred < 0:
                    pred = current_value * 0.95  # Pequeña disminución
                
                predictions.append(pred)
                current_value = pred
                
            return predictions
            
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            # Fallback ultra-conservador
            last_value = clean_series.iloc[-1]
            conservative_growth = 0.02  # 2% anual
            return [last_value * (1 + conservative_growth)**i for i in range(1, periods + 1)]

    def get_industry_benchmark(self, sector, industry, period='annual', years_back=999):
        """Calcula benchmark de industria con fechas alineadas - MEJORADO Y CORREGIDO"""
        if not sector or self.screener_data.empty:
            return pd.DataFrame()
            
        cache_key = f"{industry}_{sector}_{period}_{years_back}"
        if cache_key in self.industry_cache:
            return self.industry_cache[cache_key]
            
        try:
            # Primero intentar conseguir empresas de la misma industria
            industry_companies = self.screener_data[
                (self.screener_data['zacks_x_ind_desc'] == industry) &
                (self.screener_data['market_val'].notna())
            ].copy()
            
            # Si no hay suficientes de la industria, usar sector
            if len(industry_companies) < 8:  # Aumentamos el mínimo
                sector_companies = self.screener_data[
                    (self.screener_data['zacks_x_sector_desc'] == sector) &
                    (self.screener_data['market_val'].notna())
                ].copy()
                
                # Combinar industria + sector, priorizando industria
                combined_companies = pd.concat([industry_companies, sector_companies]).drop_duplicates(subset=['ticker'])
            else:
                combined_companies = industry_companies
            
            # Ordenar por market cap (usar las más grandes)
            combined_companies = combined_companies.sort_values('market_val', ascending=False)
            
            # Tomar las top 15-20 empresas por capitalización para mejor benchmark
            top_companies = combined_companies.head(20)['ticker'].tolist()
            
            benchmark_data = []
            successful_loads = 0
            
            for ticker in top_companies:
                if successful_loads >= 15:  # Máximo 15 empresas para benchmark sólido
                    break
                    
                df = self.load_ticker_data(ticker, period)
                if not df.empty and len(df) >= 3:
                    # CORREGIDO: Usar el período completo especificado
                    if years_back != 999:
                        cutoff_date = df['Date'].max() - timedelta(days=365*years_back)
                        df_period = df[df['Date'] >= cutoff_date].copy()
                    else:
                        df_period = df.copy()
                    
                    if len(df_period) >= 2:
                        df_period['ticker'] = ticker
                        benchmark_data.append(df_period)
                        successful_loads += 1
            
            if len(benchmark_data) >= 5:  # Mínimo 5 empresas para benchmark confiable
                combined_df = pd.concat(benchmark_data, ignore_index=True)
                
                # MEJORA: Alinear fechas mejor - usar trimestres para anuales
                if period == 'annual':
                    # Para datos anuales, usar Q4 (diciembre) como referencia estándar
                    combined_df['Year'] = combined_df['Date'].dt.year
                    combined_df['Quarter'] = combined_df['Date'].dt.quarter
                    
                    # Priorizar Q4, luego Q3, luego Q2, luego Q1
                    combined_df['Priority'] = combined_df['Quarter'].map({4: 1, 3: 2, 2: 3, 1: 4})
                    combined_df = combined_df.sort_values(['Year', 'Priority']).groupby(['Year', 'ticker']).first().reset_index()
                    
                    # Crear fecha consistente (31 de diciembre de cada año)
                    combined_df['Date'] = pd.to_datetime(combined_df['Year'], format='%Y') + pd.DateOffset(months=11, days=30)
                else:
                    # Para trimestrales, mantener fechas originales pero alineadas por quarter
                    combined_df['YearQuarter'] = combined_df['Date'].dt.to_period('Q')
                
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col not in ['Year', 'Quarter', 'Priority']]
                
                if len(numeric_cols) > 0:
                    if period == 'annual':
                        # CORREGIDO: Usar mediana por año, excluyendo outliers
                        benchmark_stats = combined_df.groupby('Year')[numeric_cols].apply(
                            lambda x: x.quantile(0.5)  # Mediana robusta
                        ).reset_index()
                        
                        # Añadir fechas
                        benchmark_stats['Date'] = pd.to_datetime(benchmark_stats['Year'], format='%Y') + pd.DateOffset(months=11, days=30)
                    else:
                        benchmark_stats = combined_df.groupby('YearQuarter')[numeric_cols].median().reset_index()
                        benchmark_stats['Date'] = benchmark_stats['YearQuarter'].dt.to_timestamp()
                    
                    # Limpiar outliers extremos en el benchmark
                    for col in numeric_cols:
                        if col in benchmark_stats.columns:
                            Q1 = benchmark_stats[col].quantile(0.1)
                            Q3 = benchmark_stats[col].quantile(0.9)
                            IQR = Q3 - Q1
                            benchmark_stats[col] = benchmark_stats[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                    
                    self.industry_cache[cache_key] = benchmark_stats
                    return benchmark_stats
                
        except Exception as e:
            logging.error(f"Error calculating industry benchmark: {e}")
        
        return pd.DataFrame()

    def get_sector_competitors(self, ticker):
        """Obtiene empresas competidoras MEJORADO para mayor precisión"""
        if self.screener_data.empty:
            return []
            
        try:
            company_data = self.screener_data[self.screener_data['ticker'] == ticker]
            if company_data.empty:
                return []
                
            company_info = company_data.iloc[0]
            sector = company_info.get('zacks_x_sector_desc')
            industry = company_info.get('zacks_x_ind_desc')
            company_market_cap = company_info.get('market_val', 0)
            
            if not sector:
                return []
            
            # MEJORADO: Criterios más estrictos para competidores
            # 1. Misma industria específica (prioridad máxima)
            same_industry = self.screener_data[
                (self.screener_data['zacks_x_ind_desc'] == industry) & 
                (self.screener_data['ticker'] != ticker) &
                (self.screener_data['market_val'].notna()) &
                (self.screener_data['market_val'] > company_market_cap * 0.1)  # Mínimo 10% del market cap
            ].copy()
            
            # Filtrar por tamaño similar (dentro de 2 órdenes de magnitud)
            if company_market_cap > 0:
                same_industry = same_industry[
                    (same_industry['market_val'] >= company_market_cap * 0.1) &
                    (same_industry['market_val'] <= company_market_cap * 10)
                ]
            
            # 2. Mismo sector pero diferente industria (secundario)
            same_sector_diff_industry = self.screener_data[
                (self.screener_data['zacks_x_sector_desc'] == sector) & 
                (self.screener_data['zacks_x_ind_desc'] != industry) &
                (self.screener_data['ticker'] != ticker) &
                (self.screener_data['market_val'].notna()) &
                (self.screener_data['market_val'] > company_market_cap * 0.2)  # Más restrictivo
            ].copy()
            
            if company_market_cap > 0:
                same_sector_diff_industry = same_sector_diff_industry[
                    (same_sector_diff_industry['market_val'] >= company_market_cap * 0.2) &
                    (same_sector_diff_industry['market_val'] <= company_market_cap * 5)
                ]
            
            # Combinar con prioridades
            # Tomar top 6 de misma industria
            top_same_industry = same_industry.nlargest(6, 'market_val')
            
            # Si no hay suficientes, completar con mismo sector
            remaining_slots = max(0, 8 - len(top_same_industry))
            if remaining_slots > 0:
                top_same_sector = same_sector_diff_industry.nlargest(remaining_slots, 'market_val')
                competitors = pd.concat([top_same_industry, top_same_sector])
            else:
                competitors = top_same_industry
            
            # Ordenar por market cap descendente
            competitors = competitors.sort_values('market_val', ascending=False)
            
            return competitors['ticker'].head(8).tolist()
            
        except Exception as e:
            logging.error(f"Error getting competitors: {e}")
            return []

    def get_available_tickers(self, period='annual'):
        """Lista de tickers disponibles"""
        path = ANNUAL_DATA_PATH if period == 'annual' else QUARTERLY_DATA_PATH
        try:
            if os.path.exists(path):
                files = [f.replace('.parquet', '') for f in os.listdir(path) if f.endswith('.parquet')]
                return sorted(files)
        except:
            pass
        return []

    def get_available_industries(self):
        """Obtiene lista de industrias únicas disponibles"""
        if self.screener_data.empty:
            return []
        
        try:
            industries = self.screener_data['zacks_x_ind_desc'].dropna().unique().tolist()
            return sorted(industries)
        except:
            return []

    def get_tickers_by_industry(self, industry):
        """Obtiene tickers filtrados por industria"""
        if self.screener_data.empty or not industry:
            return []
        
        try:
            industry_companies = self.screener_data[
                self.screener_data['zacks_x_ind_desc'] == industry
            ]
            return industry_companies['ticker'].tolist()
        except:
            return []

    def get_tickers_by_market_cap(self, market_cap_filter, available_tickers):
        """Filtra tickers por capitalización bursátil"""
        if self.screener_data.empty or not market_cap_filter or market_cap_filter == 'all':
            return available_tickers
        
        try:
            filtered_df = self.screener_data[
                self.screener_data['ticker'].isin(available_tickers) & 
                self.screener_data['market_val'].notna()
            ].copy()
            
            if market_cap_filter == 'mega_cap':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 200e9]
            elif market_cap_filter == 'large_cap':
                filtered_df = filtered_df[(filtered_df['market_val']*1000000 >= 10e9) & (filtered_df['market_val']*1000000 < 200e9)]
            elif market_cap_filter == 'mid_cap':
                filtered_df = filtered_df[(filtered_df['market_val']*1000000 >= 2e9) & (filtered_df['market_val']*1000000 < 10e9)]
            elif market_cap_filter == 'small_cap':
                filtered_df = filtered_df[(filtered_df['market_val']*1000000 >= 300e6) & (filtered_df['market_val']*1000000 < 2e9)]
            elif market_cap_filter == 'micro_cap':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 < 300e6]
            elif market_cap_filter == 'above_100b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 100e9]
            elif market_cap_filter == 'above_50b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 50e9]
            elif market_cap_filter == 'above_30b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 30e9]
            elif market_cap_filter == 'above_25b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 25e9]
            elif market_cap_filter == 'above_20b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 20e9]
            elif market_cap_filter == 'above_15b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 15e9]
            elif market_cap_filter == 'above_10b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 10e9]
            elif market_cap_filter == 'above_5b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 5e9]
            elif market_cap_filter == 'above_1b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 >= 1e9]
            elif market_cap_filter == 'below_50b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 < 50e9]
            elif market_cap_filter == 'below_25b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 < 25e9]
            elif market_cap_filter == 'below_10b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 < 10e9]
            elif market_cap_filter == 'below_5b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 < 5e9]
            elif market_cap_filter == 'below_1b':
                filtered_df = filtered_df[filtered_df['market_val']*1000000 < 1e9]
            
            return sorted(filtered_df['ticker'].tolist())
        except:
            return available_tickers

# Inicializar analizador
analyzer = ProfessionalInvestmentAnalyzer()

# Colores para gráficos
COLOR_SEQUENCE = ['#00d4ff', '#00ff88', '#ffaa00', '#ff3366', '#9333ea', '#00cc99', '#ff6699', '#66ccff']

app = dash.Dash(__name__, 
                external_stylesheets=[
                    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
                ],
                suppress_callback_exceptions=True)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Fundamental Analysis Platform</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Rajdhani:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
                background: #0f1419;
                color: #e8eaed;
                line-height: 1.6;
            }
            .main-container { max-width: 1900px; margin: 0 auto; padding: 20px; }
            
            /* Header principal */
            .header { 
                background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
                color: #e8eaed; 
                padding: 30px; 
                border-radius: 8px; 
                margin-bottom: 25px;
                border: 1px solid #2a2f3e;
            }
            .main-title {
                color: #00d4ff;
                font-size: 2.8rem;
                font-weight: 300;
                margin: 0;
                text-align: center;
                letter-spacing: 4px;
                text-transform: uppercase;
                font-family: 'Rajdhani', monospace;
            }
            .subtitle {
                color: #9ca3af;
                font-size: 0.9rem;
                text-align: center;
                margin-top: 8px;
                letter-spacing: 2px;
                font-weight: 400;
            }
            
            /* Controles */
            .controls-container { 
                background: #1a1f2e; 
                padding: 25px; 
                border-radius: 8px; 
                margin-bottom: 25px;
                border: 1px solid #2a2f3e;
            }
            .controls-grid { 
                display: grid; 
                grid-template-columns: 1fr 1fr 120px 100px 2fr; 
                gap: 15px; 
                align-items: end;
            }
            .control-group label { 
                font-weight: 500; 
                margin-bottom: 8px; 
                display: block; 
                color: #9ca3af; 
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            /* Botones */
            .update-button {
                background: linear-gradient(135deg, #00d4ff, #0099cc);
                color: #0f1419;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 0.9rem;
                cursor: pointer;
                transition: all 0.3s;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-top: 15px;
            }
            .update-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
            }
            
            /* Sugerencias de competidores */
            .competitors-suggest {
                background: #2d3748;
                padding: 15px;
                border-radius: 6px;
                margin-top: 15px;
                border: 1px solid #00d4ff;
                border-left: 3px solid #00d4ff;
                color: #e2e8f0;
            }
            
            /* Checkbox de benchmark */
            .benchmark-checkbox {
                background: #242938;
                padding: 15px;
                border-radius: 6px;
                margin-top: 15px;
                border: 1px solid #2a2f3e;
            }
            
            /* Contenido principal */
            .main-content { 
                background: #1a1f2e; 
                border-radius: 8px; 
                border: 1px solid #2a2f3e;
                overflow: hidden;
            }
            
            /* Overview de la empresa */
            .company-overview { 
                padding: 30px; 
                border-bottom: 1px solid #2a2f3e; 
                background: #242938;
            }
            .company-title { 
                font-size: 2.2em; 
                font-weight: 300; 
                color: #00d4ff; 
                margin-bottom: 10px; 
                letter-spacing: 1px;
            }
            .company-meta { 
                color: #9ca3af; 
                margin-bottom: 25px; 
                font-size: 0.9rem;
                font-weight: 400;
                letter-spacing: 0.5px;
            }
            
            /* Métricas overview */
            .overview-metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); 
                gap: 15px; 
            }
            .metric-card { 
                background: #1a1f2e; 
                padding: 18px; 
                border-radius: 6px; 
                text-align: center;
                border: 1px solid #2a2f3e;
                transition: all 0.3s;
            }
            .metric-card:hover {
                border-color: #00d4ff;
                transform: translateY(-2px);
            }
            .metric-value { 
                font-size: 1.8em; 
                font-weight: 600; 
                color: #00d4ff; 
                margin-bottom: 5px; 
            }
            .metric-label { 
                color: #9ca3af; 
                font-size: 0.75rem; 
                font-weight: 500; 
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            /* Tabs */
            .tabs-container { 
                background: #242938; 
                border-bottom: 1px solid #2a2f3e; 
            }
            .tab-content { 
                padding: 30px;
                background: #1a1f2e;
            }
            .dash-tab {
                color: #9ca3af !important;
                border-bottom: 2px solid transparent !important;
                padding: 12px 24px !important;
                font-weight: 500 !important;
                background-color: #242938 !important;
            }

            .dash-tab--selected {
                color: #00d4ff !important;
                border-bottom: 2px solid #00d4ff !important;
                background-color: #242938 !important;
                font-weight: 600 !important;
            }
            
            /* Secciones */
            .section-header { 
                margin-bottom: 35px; 
            }
            .section-title { 
                font-size: 1.8em; 
                font-weight: 300; 
                color: #00d4ff; 
                margin-bottom: 10px; 
                letter-spacing: 1px;
            }
            .section-description { 
                color: #9ca3af; 
                font-size: 0.9rem; 
                line-height: 1.6;
            }
            
            /* Gráficos */
            .charts-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); 
                gap: 25px; 
            }
            .chart-container { 
                background: #242938; 
                border-radius: 8px; 
                padding: 20px;
                border: 1px solid #2a2f3e;
                position: relative;
            }
            
            /* Info buttons para métricas */
            .chart-info-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                width: 28px;
                height: 28px;
                border-radius: 50%;
                background: #00d4ff;
                color: #0f1419;
                border: none;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                opacity: 0.8;
                transition: all 0.3s;
                z-index: 100;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .chart-info-btn:hover {
                opacity: 1;
                transform: scale(1.1);
                background: #00ff88;
            }
            
            /* Tooltips de métricas */
            .metric-tooltip {
                background: #242938;
                color: #e8eaed;
                padding: 15px;
                border-radius: 6px;
                font-size: 0.85rem;
                box-shadow: 0 8px 30px rgba(0,0,0,0.5);
                line-height: 1.6;
                border: 1px solid #00d4ff;
                max-width: 320px;
                z-index: 1001;
            }
            
            /* Tablas comparativas */
            .comparison-table { 
                background: #1a1f2e; 
                border-radius: 8px; 
                overflow: hidden;
                border: 1px solid #2a2f3e;
            }
            .category-group { 
                margin-bottom: 45px; 
            }
            .category-title { 
                font-size: 1.4em; 
                font-weight: 500; 
                color: #0f1419; 
                margin-bottom: 20px;
                padding: 15px 20px;
                background: linear-gradient(135deg, #00d4ff, #0099cc);
                border-radius: 6px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            /* Tickers clickeables */
            .clickable-ticker {
                cursor: pointer;
                color: #00d4ff;
                font-weight: 600;
                text-decoration: none;
                transition: all 0.2s;
            }
            .clickable-ticker:hover {
                color: #00ff88;
                text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
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
            .Select-control {
                background: #242938 !important;
                border: 1px solid #2a2f3e !important;
                color: #ffffff !important;  /* Cambiado a blanco */
            }
            .Select-placeholder {
                color: #ffffff !important;
            }
            .Select-value-label {
                color: #ffffff !important;
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
            /* Cajas de colores para explicaciones */
            .performance-colors-box {
                background: #242938;
                padding: 15px;
                border-radius: 6px;
                margin-top: 15px;
                border: 1px solid #00d4ff;
                border-left: 3px solid #00d4ff;
                color: #e2e8f0;
                font-size: 13px;
                line-height: 1.6;
            }


            /* Footer */
            .footer {
                text-align: center;
                padding: 30px;
                margin-top: 40px;
                border-top: 1px solid #2a2f3e;
            }
            
            /* Responsive */
            @media (max-width: 1200px) {
                .controls-grid {
                    grid-template-columns: 1fr 1fr;
                    grid-template-rows: auto auto auto;
                    gap: 15px;
                }
                .charts-grid {
                    grid-template-columns: 1fr;
                }
            }
            
            #company-overview-header {
                cursor: pointer;
                transition: background 0.3s;
            }

            #company-overview-header:hover {
                background: rgba(0, 212, 255, 0.08);
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

def get_company_name(ticker):
    """Obtiene el nombre de la empresa del screener data"""
    try:
        if not analyzer.screener_data.empty:
            company_data = analyzer.screener_data[analyzer.screener_data['ticker'] == ticker]
            if not company_data.empty:
                name = company_data.iloc[0].get('comp_name_2', company_data.iloc[0].get('comp_name', ''))
                return f"({name[:30]}...)" if len(name) > 30 else f"({name})"
        return ""
    except:
        return ""

# Layout
app.layout = html.Div([
    html.Div([
        # Header principal
        html.Div([
            html.H1("FUNDAMENTAL ANALYSIS PLATFORM", className='main-title'),
            html.P("INSTITUTIONAL-GRADE VALUATION • PREDICTIVE MODELING • INDUSTRY BENCHMARKING", 
                  className='subtitle')
        ], className='header'),
        
        # Controles
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Industry Filter"),
                    dcc.Dropdown(
                        id='industry-filter',
                        options=[{'label': 'All Industries', 'value': 'all'}] + 
                               [{'label': industry, 'value': industry} for industry in analyzer.get_available_industries()],
                        value='all',
                        placeholder="Filter by industry...",
                        style={'background': UNIFIED_COLORS['bg_tertiary'], 'color': UNIFIED_COLORS['text']}
                    )
                ], className='filter-industry'),
                
                html.Div([
                    html.Label("Market Cap Filter"),
                    dcc.Dropdown(
                        id='market-cap-filter',
                        options=[
                            {'label': 'All Cap Sizes', 'value': 'all'},
                            {'label': 'Mega Cap (>$200B)', 'value': 'mega_cap'},
                            {'label': 'Large Cap ($10B-$200B)', 'value': 'large_cap'},
                            {'label': 'Mid Cap ($2B-$10B)', 'value': 'mid_cap'},
                            {'label': 'Small Cap ($300M-$2B)', 'value': 'small_cap'},
                            {'label': 'Micro Cap (<$300M)', 'value': 'micro_cap'}
                        ],
                        value='all',
                        placeholder="Filter by market cap...",
                        style={'background': UNIFIED_COLORS['bg_tertiary'], 'color': UNIFIED_COLORS['text']}
                    )
                ], className='filter-market-cap'),
                
                html.Div([
                    html.Label("Reporting Period"),
                    dcc.Dropdown(
                        id='period-select',
                        options=[
                            {'label': 'Annual', 'value': 'annual'},
                            {'label': 'Quarterly', 'value': 'quarterly'}
                        ],
                        value='annual',
                        style={'background': UNIFIED_COLORS['bg_tertiary'], 'color': UNIFIED_COLORS['text']}
                    )
                ], className='filter-period'),
                
                html.Div([
                    html.Label("Analysis Period"),
                    dcc.Dropdown(
                        id='time-range',
                        options=[
                            {'label': '5Y', 'value': 5},
                            {'label': '10Y', 'value': 10},
                            {'label': 'Full', 'value': 999}
                        ],
                        value=10,
                        style={'background': UNIFIED_COLORS['bg_tertiary'], 'color': UNIFIED_COLORS['text']}
                    )
                ], className='filter-time'),
                
                html.Div([
                    html.Label("Companies Pool"),
                    dcc.Dropdown(
                        id='companies-pool',
                        options=[{'label': t, 'value': t} for t in analyzer.get_available_tickers()],
                        placeholder="Select companies for comparative analysis...",
                        multi=True,
                        style={'background': UNIFIED_COLORS['bg_tertiary'], 'color': UNIFIED_COLORS['text']}
                    )
                ], className='filter-companies')
            ], className='controls-grid'),
            
            # Update Button
            html.Div([
                html.Button("UPDATE ANALYSIS", id='update-button', className='update-button', n_clicks=0),
                
                html.Div([
                    dcc.Checklist(
                        id='show-benchmark',
                        options=[{'label': ' Compare against industry median performance', 'value': 'show'}],
                        value=['show'],
                        style={'margin': '0', 'display': 'inline-block'},
                        inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                        labelStyle={'fontSize': '0.85rem', 'fontWeight': '500', 'color': UNIFIED_COLORS['text_secondary']}
                    )
                ], style={'marginLeft': '20px', 'display': 'inline-block', 'verticalAlign': 'middle'})
            ], style={'marginTop': '15px', 'display': 'flex', 'alignItems': 'center'}),

            # Sugerencias de competidores
            html.Div(id='competitors-suggestion')
        ], className='controls-container'),

        # Store para el estado del colapso
        dcc.Store(id="overview-collapsed", data=False),

        # Contenido principal
        html.Div([
            html.Div(id='company-overview-section'),

            html.Div([
                dcc.Tabs(id="analysis-tabs", value='fundamentals', children=[
                    dcc.Tab(label='Growth & Fundamentals', value='fundamentals', className='dash-tab', selected_className='dash-tab--selected'),
                    dcc.Tab(label='Valuation Metrics', value='valuation', className='dash-tab', selected_className='dash-tab--selected'),
                    dcc.Tab(label='Financial Health', value='health', className='dash-tab', selected_className='dash-tab--selected'),
                    dcc.Tab(label='Operational Efficiency', value='efficiency', className='dash-tab', selected_className='dash-tab--selected'),
                    dcc.Tab(label='Comparative Analysis', value='comparative', className='dash-tab', selected_className='dash-tab--selected')
                ])
            ], className='tabs-container'),
            
            html.Div(id='analysis-content')
        ], className='main-content'),
        
        # Footer profesional
        html.Div([
            html.Hr(style={
                'border': f'1px solid {UNIFIED_COLORS["border"]}',
                'margin': '40px 0 20px 0',
                'opacity': '0.5'
            }),
            html.Div([
                html.P("FUNDAMENTAL ANALYSIS PLATFORM", style={
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
            ])
        ], className='footer')
        
    ], className='main-container')
])

# Callback para actualizar opciones de companies pool cuando cambian los filtros
@app.callback(
    Output('companies-pool', 'options'),
    [Input('industry-filter', 'value'),
     Input('market-cap-filter', 'value'),
     Input('period-select', 'value'),
     Input('update-button', 'n_clicks')]
)
def update_companies_options(industry_filter, market_cap_filter, period, n_clicks):
    # Obtener todos los tickers disponibles para el período seleccionado
    available_tickers = analyzer.get_available_tickers(period)
    
    # Aplicar filtro de industria
    if industry_filter and industry_filter != 'all':
        industry_tickers = analyzer.get_tickers_by_industry(industry_filter)
        available_tickers = [t for t in available_tickers if t in industry_tickers]
    
    # Aplicar filtro de market cap
    filtered_tickers = analyzer.get_tickers_by_market_cap(market_cap_filter, available_tickers)
    
    return [{'label': f"{ticker} {get_company_name(ticker)}", 'value': ticker} for ticker in filtered_tickers]

# Callback para autocompletar el pool cuando se selecciona industrias (ahora múltiples)
@app.callback(
    Output('companies-pool', 'value'),
    [Input('industry-filter', 'value')],
    [State('companies-pool', 'value'),
     State('period-select', 'value')]
)
def auto_populate_from_industry(industry_filter, current_selection, period):
    # Si se seleccionan industrias específicas, añadir algunos tickers representativos
    if industry_filter and 'all' not in industry_filter:
        all_industry_tickers = []
        for industry in industry_filter:
            industry_tickers = analyzer.get_tickers_by_industry(industry)
            all_industry_tickers.extend(industry_tickers)
        
        # Remover duplicados manteniendo el orden
        all_industry_tickers = list(dict.fromkeys(all_industry_tickers))
        available_tickers = analyzer.get_available_tickers(period)
        valid_tickers = [t for t in all_industry_tickers if t in available_tickers]
        
        # Si no hay selección previa, tomar los primeros 8 tickers de todas las industrias
        if not current_selection:
            return valid_tickers[:8]
        else:
            # Mantener selección actual y añadir nuevos si no están ya incluidos
            new_selection = list(current_selection) if current_selection else []
            for ticker in valid_tickers[:8]:
                if ticker not in new_selection:
                    new_selection.append(ticker)
                if len(new_selection) >= 12:  # Limitar a 12 máximo para múltiples industrias
                    break
            return new_selection
    
    # Si se selecciona "all" o no hay selección, mantener selección actual
    return current_selection if current_selection else []

@app.callback(
    Output('competitors-suggestion', 'children'),
    [Input('companies-pool', 'value')]
)
def suggest_competitors(selected_companies):
    if not selected_companies or len(selected_companies) == 0:
        return html.Div()
    
    primary_ticker = selected_companies[0]  # Primera empresa = primary
    
    try:
        competitors = analyzer.get_sector_competitors(primary_ticker)
        if competitors:
            return html.Div([
                html.Div([
                    html.Strong(f"Recommended Peers for {primary_ticker}: ", 
                            style={'color': '#e8eaed', 'fontSize': '14px'}),
                    html.Span(" • ".join(competitors[:6]), 
                            style={'color': '#9ca3af', 'fontSize': '13px'})
                ])
            ], className='competitors-suggest')
    except:
        pass
    
    return html.Div()

# Callback para mostrar/ocultar overview metrics
@app.callback(
    Output("company-overview-metrics", "style"),
    Input("company-overview-header", "n_clicks"),
    State("overview-collapsed", "data"),
    prevent_initial_call=True
)
def toggle_overview(n_clicks, collapsed):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
        
    # Alternar el estado
    new_collapsed = not collapsed
    
    # Devolver el estilo apropiado
    if new_collapsed:
        return {"display": "none"}
    else:
        return {"display": "block"}

# Callback adicional para actualizar el store
@app.callback(
    Output("overview-collapsed", "data"),
    Input("company-overview-header", "n_clicks"),
    State("overview-collapsed", "data"),
    prevent_initial_call=True
)
def update_collapse_state(n_clicks, collapsed):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    return not collapsed

@app.callback(
    Output('company-overview-section', 'children'),
    [Input('companies-pool', 'value'),
     Input('period-select', 'value'),
     Input('update-button', 'n_clicks')]
)
def update_company_overview(selected_companies, period, n_clicks):
    if not selected_companies or len(selected_companies) == 0:
        return html.Div()

    primary_ticker = selected_companies[0]

    try:
        df = analyzer.load_ticker_data(primary_ticker, period)
        if df.empty:
            return html.Div("No data available")

        # Company info
        company_info = {}
        if not analyzer.screener_data.empty:
            comp_data = analyzer.screener_data[analyzer.screener_data['ticker'] == primary_ticker]
            if not comp_data.empty:
                company_info = comp_data.iloc[0].to_dict()

        latest = df.iloc[-1]

        # Growth trends
        revenue_growth, income_growth = 0, 0
        if len(df) >= 3:
            if 'Revenue' in df.columns:
                revenue_growth = df['Revenue'].pct_change().tail(3).mean() * 100
            if 'Net Income' in df.columns:
                income_growth = df['Net Income'].pct_change().tail(3).mean() * 100

        company_name = company_info.get('comp_name_2', company_info.get('comp_name', primary_ticker))

        return html.Div([
            # Cabecera clickeable
            html.Div([
                html.H2(f"{company_name} ({primary_ticker})", className='company-title'),
                html.Div(
                    f"Sector: {company_info.get('zacks_x_sector_desc', 'N/A')} • "
                    f"Industry: {company_info.get('zacks_x_ind_desc', 'N/A')} • "
                    f"Country: {company_info.get('country_code', 'N/A')}",
                    className='company-meta'
                ),
            ], id="company-overview-header", n_clicks=0, style={"cursor": "pointer"}),

            # Métricas colapsables
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(format_large_number(latest.get('MarketCap', 0)), className='metric-value'),
                        html.Div("Market Cap", className='metric-label')
                    ], className='metric-card'),

                    html.Div([
                        html.Div(format_large_number(latest.get('Revenue', 0)), className='metric-value'),
                        html.Div("Revenue TTM", className='metric-label')
                    ], className='metric-card'),

                    html.Div([
                        html.Div(f"{latest.get('P/E', 0):.1f}x", className='metric-value'),
                        html.Div("P/E Ratio", className='metric-label')
                    ], className='metric-card'),

                    html.Div([
                        html.Div(format_large_number(latest.get('Long Term Debt', 0)), className='metric-value'),
                        html.Div("Total Debt", className='metric-label')
                    ], className='metric-card'),

                    html.Div([
                        html.Div(format_shares_number(latest.get('Shares Outstanding', 0)), className='metric-value'),
                        html.Div("Shares Outstanding", className='metric-label')
                    ], className='metric-card'),

                    html.Div([
                        html.Div(f"{latest.get('ROE - Return On Equity', 0):.1f}%", className='metric-value'),
                        html.Div("Return on Equity", className='metric-label')
                    ], className='metric-card'),

                    html.Div([
                        html.Div(f"{revenue_growth:.1f}%", className='metric-value'),
                        html.Div("Revenue Growth (3Y)", className='metric-label')
                    ], className='metric-card'),

                    html.Div([
                        html.Div(f"{company_info.get('beta', 0):.2f}", className='metric-value'),
                        html.Div("Beta (Volatility)", className='metric-label')
                    ], className='metric-card')
                ], className='overview-metrics')
            ], id="company-overview-metrics", style={"display": "block"})
        ], className='company-overview')

    except Exception as e:
        return html.Div(f"Error: {str(e)}")

@app.callback(
    Output('analysis-content', 'children'),
    [Input('analysis-tabs', 'value'),
     Input('companies-pool', 'value'),
     Input('period-select', 'value'),
     Input('time-range', 'value'),
     Input('show-benchmark', 'value'),
     Input('update-button', 'n_clicks')]
)
def update_analysis_content(active_tab, selected_companies, period, years, show_benchmark, n_clicks):
    if not selected_companies or len(selected_companies) == 0:
        return html.Div("Select companies to begin institutional-grade analysis", 
                       style={'padding': '50px', 'textAlign': 'center', 'color': '#718096'})
    
    try:
        # Primera empresa = primary, resto = comparison
        primary_ticker = selected_companies[0]
        comparison_tickers = selected_companies[1:] if len(selected_companies) > 1 else []
        
        # Load main company data
        main_df = analyzer.load_ticker_data(primary_ticker, period)
        if main_df.empty:
            return html.Div("No data available for selected company")
        
        # Filter by time range
        if years != 999:
            cutoff_date = main_df['Date'].max() - timedelta(days=365*years)
            main_df = main_df[main_df['Date'] >= cutoff_date]
        
        # Load comparison data
        comparison_data = {}
        if comparison_tickers:
            for ticker in comparison_tickers:
                comp_df = analyzer.load_ticker_data(ticker, period)
                if not comp_df.empty:
                    if years != 999:
                        cutoff_date = comp_df['Date'].max() - timedelta(days=365*years)
                        comp_df = comp_df[comp_df['Date'] >= cutoff_date]
                    comparison_data[ticker] = comp_df
        
        # Load industry benchmark
        benchmark_df = pd.DataFrame()
        if 'show' in show_benchmark:
            sector = None
            industry = None
            if not analyzer.screener_data.empty:
                comp_info = analyzer.screener_data[analyzer.screener_data['ticker'] == primary_ticker]
                if not comp_info.empty:
                    sector = comp_info.iloc[0].get('zacks_x_sector_desc')
                    industry = comp_info.iloc[0].get('zacks_x_ind_desc')
            
            if sector and industry:
                benchmark_df = analyzer.get_industry_benchmark(sector, industry, period, years)
        
        # Generate content based on active tab
        if active_tab == 'fundamentals':
            return create_fundamentals_analysis(main_df, primary_ticker, comparison_data, benchmark_df)
        elif active_tab == 'valuation':
            return create_valuation_analysis(main_df, primary_ticker, comparison_data, benchmark_df)
        elif active_tab == 'health':
            return create_health_analysis(main_df, primary_ticker, comparison_data, benchmark_df)
        elif active_tab == 'efficiency':
            return create_efficiency_analysis(main_df, primary_ticker, comparison_data, benchmark_df)
        elif active_tab == 'comparative':
            return create_comparative_analysis(main_df, primary_ticker, comparison_data, benchmark_df)
        
    except Exception as e:
        return html.Div(f"Error loading analysis: {str(e)}", style={'padding': '20px'})

def create_professional_chart_with_info(main_df, main_ticker, metric, title, comparison_data=None, benchmark_df=None, format_type='number'):
    """Creates professional charts with predictions - FORMATO CORREGIDO CON ESCALADO REAL"""
    fig = go.Figure()
    
    try:
        # Determinar escala antes de crear trazas
        scale_factor = 1
        scale_suffix = ""
        
        if format_type == 'millions':
            all_values = []
            if not main_df.empty and metric in main_df.columns:
                all_values.extend(main_df[metric].dropna().tolist())
            if comparison_data:
                for ticker, df in comparison_data.items():
                    if metric in df.columns:
                        all_values.extend(df[metric].dropna().tolist())
            if not benchmark_df.empty and metric in benchmark_df.columns:
                all_values.extend(benchmark_df[metric].dropna().tolist())
            
            if all_values:
                max_val = max(abs(v) for v in all_values if pd.notna(v))
                
                if max_val >= 1e12:  # Trillones
                    scale_factor = 1e12
                    scale_suffix = " (Trillions $)"
                elif max_val >= 1e9:  # Billones  
                    scale_factor = 1e9
                    scale_suffix = " (Billions $)"
                elif max_val >= 1e6:  # Millones
                    scale_factor = 1e6
                    scale_suffix = " (Millions $)"
                elif max_val >= 1e3:  # Miles
                    scale_factor = 1e3
                    scale_suffix = " (Thousands $)"

        # Main company historical data
        main_clean = main_df[['Date', metric]].dropna()
        if not main_clean.empty:
            # Aplicar escalado
            y_values = main_clean[metric] / scale_factor if scale_factor > 1 else main_clean[metric]
            
            fig.add_trace(go.Scatter(
                x=main_clean['Date'],
                y=y_values,
                mode='lines+markers',
                name=main_ticker,
                line=dict(color=COLOR_SEQUENCE[0], width=3),
                marker=dict(size=5),
                hovertemplate=f'{main_ticker}<br>%{{x}}<br>{title}: %{{y}}<extra></extra>'
            ))
            
            # Main company predictions
            predictions = analyzer.predict_metric_advanced(main_clean[metric], periods=3)
            if not any(pd.isna(predictions)):
                last_date = main_clean['Date'].iloc[-1]
                future_dates = [last_date + timedelta(days=365*(i+1)) for i in range(3)]
                
                pred_x = [last_date] + future_dates
                scaled_predictions = [p / scale_factor if scale_factor > 1 else p for p in predictions]
                pred_y = [y_values.iloc[-1]] + scaled_predictions
                
                fig.add_trace(go.Scatter(
                    x=pred_x,
                    y=pred_y,
                    mode='lines+markers',
                    name=main_ticker,
                    line=dict(color=COLOR_SEQUENCE[0], width=2, dash='dash'),
                    marker=dict(size=4, symbol='diamond'),
                    showlegend=False,
                    hovertemplate=f'{main_ticker} (Forecast)<br>%{{x}}<br>{title}: %{{y}}<extra></extra>'
                ))
        
        # Comparison companies
        if comparison_data:
            for i, (ticker, df) in enumerate(comparison_data.items()):
                if metric in df.columns:
                    comp_clean = df[['Date', metric]].dropna()
                    if not comp_clean.empty:
                        color_idx = (i + 1) % len(COLOR_SEQUENCE)
                        
                        # Aplicar escalado
                        y_values = comp_clean[metric] / scale_factor if scale_factor > 1 else comp_clean[metric]
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=comp_clean['Date'],
                            y=y_values,
                            mode='lines+markers',
                            name=ticker,
                            line=dict(color=COLOR_SEQUENCE[color_idx], width=2.5),
                            marker=dict(size=4),
                            hovertemplate=f'{ticker}<br>%{{x}}<br>{title}: %{{y}}<extra></extra>'
                        ))
                        
                        # Predictions for comparison companies
                        predictions = analyzer.predict_metric_advanced(comp_clean[metric], periods=3)
                        if not any(pd.isna(predictions)):
                            last_date = comp_clean['Date'].iloc[-1]
                            future_dates = [last_date + timedelta(days=365*(i+1)) for i in range(3)]
                            
                            scaled_predictions = [p / scale_factor if scale_factor > 1 else p for p in predictions]
                            pred_x = [last_date] + future_dates
                            pred_y = [y_values.iloc[-1]] + scaled_predictions
                            
                            fig.add_trace(go.Scatter(
                                x=pred_x,
                                y=pred_y,
                                mode='lines+markers',
                                name=ticker,
                                line=dict(color=COLOR_SEQUENCE[color_idx], width=2, dash='dash'),
                                marker=dict(size=3, symbol='diamond'),
                                showlegend=False,
                                hovertemplate=f'{ticker} (Forecast)<br>%{{x}}<br>{title}: %{{y}}<extra></extra>'
                            ))
        
        # Industry benchmark
        if not benchmark_df.empty and metric in benchmark_df.columns:
            benchmark_clean = benchmark_df[['Date', metric]].dropna()
            if len(benchmark_clean) >= 3:
                # Aplicar escalado
                y_values = benchmark_clean[metric] / scale_factor if scale_factor > 1 else benchmark_clean[metric]
                
                fig.add_trace(go.Scatter(
                    x=benchmark_clean['Date'],
                    y=y_values,
                    mode='lines',
                    name='Industry',
                    line=dict(color='#95a5a6', width=2, dash='dot'),
                    opacity=0.8,
                    hovertemplate=f'Industry Benchmark<br>%{{x}}<br>{title}: %{{y}}<extra></extra>'
                ))
        
        # Configuración del eje Y
        yaxis_config = {'title': title + scale_suffix}
        
        if format_type == 'percentage':
            yaxis_config['tickformat'] = '.1f'
            yaxis_config['ticksuffix'] = '%'
        elif format_type == 'ratio':
            yaxis_config['tickformat'] = '.2f'
            yaxis_config['ticksuffix'] = 'x'
        elif format_type == 'currency':
            yaxis_config['tickformat'] = '$.2f'
        else:
            yaxis_config['tickformat'] = ',.1f'
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=UNIFIED_COLORS['text'], family='Inter')),
            xaxis_title="Period",
            yaxis_title=title,
            template='plotly_dark',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="left", 
                x=0, 
                font=dict(size=11, color=UNIFIED_COLORS['text_secondary'])
            ),
            hovermode='x unified',
            margin=dict(l=60, r=30, t=60, b=50),
            plot_bgcolor=UNIFIED_COLORS['bg'],
            paper_bgcolor=UNIFIED_COLORS['bg_tertiary'],
            font=dict(color=UNIFIED_COLORS['text'])
        )
        
        fig.update_xaxes(gridcolor=UNIFIED_COLORS['grid'], zerolinecolor=UNIFIED_COLORS['grid'])
        fig.update_yaxes(gridcolor=UNIFIED_COLORS['grid'], zerolinecolor=UNIFIED_COLORS['grid'])

    except Exception as e:
        fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    return fig

def create_chart_with_info_tooltip(main_df, main_ticker, metric, title, comparison_data=None, benchmark_df=None, format_type='number'):
    """Create chart with working info tooltip - SIMPLIFICADO"""
    
    fig = create_professional_chart_with_info(main_df, main_ticker, metric, title, comparison_data, benchmark_df, format_type)
    
    # Get metric info
    metric_info = METRIC_INFO.get(metric, {'desc': 'Financial metric for analysis.', 'better': 'Varies'})
    
    # ID único más simple
    chart_id = f"chart_{hash(f'{metric}_{title}') % 100000}"
    
    return html.Div([
        html.Div([
            html.Button("ⓘ", 
                       id={'type': 'info-btn', 'index': chart_id},
                       className='chart-info-btn',
                       n_clicks=0),
            html.Div([
                html.Strong(f"{title}"),
                html.Br(),
                html.Span(metric_info['desc'], style={'fontSize': '12px'}),
                html.Br(), html.Br(),
                html.Strong("Better: ", style={'color': '#4299e1'}),
                html.Span(metric_info['better'], style={'fontSize': '12px', 'fontWeight': '600'})
            ], 
            id={'type': 'tooltip', 'index': chart_id},
            className='metric-tooltip',
            style={'display': 'none'})
        ], style={'position': 'relative'}),
        dcc.Graph(figure=fig, config={'displayModeBar': False})
    ], className='chart-container')

# Callback para mostrar/ocultar tooltips
@app.callback(
    [Output({'type': 'tooltip', 'index': dash.dependencies.MATCH}, 'style')],
    [Input({'type': 'info-btn', 'index': dash.dependencies.MATCH}, 'n_clicks')],
    [State({'type': 'tooltip', 'index': dash.dependencies.MATCH}, 'style')],
    prevent_initial_call=True
)
def toggle_info_tooltip(n_clicks, current_style):
    if n_clicks and n_clicks > 0:
        if current_style and current_style.get('display') == 'block':
            return [{'display': 'none'}]
        else:
            return [{'display': 'block', 'position': 'absolute', 'top': '35px', 'right': '5px', 
                    'zIndex': 1000, 'maxWidth': '300px'}]
    return [{'display': 'none'}]

def create_fundamentals_analysis(main_df, main_ticker, comparison_data, benchmark_df):
    """Growth and fundamental metrics analysis with working tooltips"""
    
    fundamental_groups = [
        {
            'title': 'Revenue & Profitability Growth',
            'metrics': [
                ('Revenue', 'Revenue', 'millions'),
                ('Net Income', 'Net Income', 'millions'),
                ('EBITDA', 'EBITDA', 'millions'),
                ('Operating Income', 'Operating Income', 'millions')
            ]
        },
        {
            'title': 'Balance Sheet Strength', 
            'metrics': [
                ('Total Assets', 'Total Assets', 'millions'),
                ('Share Holder Equity', 'Shareholders Equity', 'millions'),
                ('Long Term Debt', 'Long-Term Debt', 'millions'),
                ('Cash On Hand', 'Cash & Equivalents', 'millions')
            ]
        },
        {
            'title': 'Per-Share Metrics',
            'metrics': [
                ('Shares Outstanding', 'Shares Outstanding (M)', 'number'),
                ('EPS - Earnings Per Share', 'Earnings Per Share', 'currency'),
                ('Revenue/Share', 'Revenue per Share', 'currency'),
                ('Cash/Share', 'Cash per Share', 'currency')
            ]
        },
        {
            'title': 'Profitability Margins',
            'metrics': [
                ('Gross Margin', 'Gross Margin', 'percentage'),
                ('Operating Margin', 'Operating Margin', 'percentage'),
                ('Net Profit Margin', 'Net Margin', 'percentage'),
                ('EBITDA Margin', 'EBITDA Margin', 'percentage')
            ]
        }
    ]
    
    content = []
    
    for group in fundamental_groups:
        charts = []
        for metric, title, format_type in group['metrics']:
            if metric in main_df.columns:
                chart_with_info = create_chart_with_info_tooltip(main_df, main_ticker, metric, title, comparison_data, benchmark_df, format_type)
                charts.append(chart_with_info)
        
        if charts:
            content.append(
                html.Div([
                    html.H3(group['title'], className='category-title'),
                    html.Div(charts, className='charts-grid')
                ], className='category-group')
            )
    
    return html.Div([
        html.Div([
            html.H2("Growth & Fundamentals Analysis", className='section-title'),
            html.P("Comprehensive analysis of revenue growth, profitability trends, balance sheet evolution with 3-year forecasts and industry benchmarking.", className='section-description')
        ], className='section-header'),
        html.Div(content)
    ], className='tab-content')

def create_valuation_analysis(main_df, main_ticker, comparison_data, benchmark_df):
    """Valuation metrics analysis"""
    
    valuation_metrics = [
        ('P/E', 'Price-to-Earnings Ratio', 'ratio'),
        ('P/B', 'Price-to-Book Ratio', 'ratio'),
        ('P/S', 'Price-to-Sales Ratio', 'ratio'),
        ('EV/EBITDA', 'EV/EBITDA Multiple', 'ratio'),
        ('Book Value Per Share', 'Book Value per Share', 'currency'),
        ('MarketCap', 'Market Capitalization', 'millions')
    ]
    
    charts = []
    for metric, title, format_type in valuation_metrics:
        if metric in main_df.columns:
            chart_with_info = create_chart_with_info_tooltip(main_df, main_ticker, metric, title, comparison_data, benchmark_df, format_type)
            charts.append(chart_with_info)
    
    return html.Div([
        html.Div([
            html.H2("Valuation Metrics Analysis", className='section-title'),
            html.P("Assess valuation relative to fundamentals, peer companies, and historical levels to identify potential over/undervaluation.", className='section-description')
        ], className='section-header'),
        html.Div(charts, className='charts-grid')
    ], className='tab-content')

def create_health_analysis(main_df, main_ticker, comparison_data, benchmark_df):
    """Financial health and debt analysis"""
    
    health_groups = [
        {
            'title': 'Debt Management & Leverage',
            'metrics': [
                ('Debt/EBITDA', 'Debt-to-EBITDA Ratio', 'ratio'),
                ('Debt/Assets', 'Debt-to-Assets Ratio', 'percentage'),
                ('Debt/Equity Ratio', 'Debt-to-Equity Ratio', 'ratio'),
                ('Net Debt', 'Net Debt Position', 'millions')
            ]
        },
        {
            'title': 'Liquidity & Working Capital', 
            'metrics': [
                ('Current Ratio', 'Current Ratio', 'ratio'),
                ('Cash Ratio', 'Cash Ratio', 'ratio'),
                ('Working Capital', 'Working Capital', 'millions'),
                ('Cash On Hand', 'Cash Holdings', 'millions')
            ]
        },
        {
            'title': 'Returns & Capital Efficiency',
            'metrics': [
                ('ROE - Return On Equity', 'Return on Equity', 'percentage'),
                ('ROA - Return On Assets', 'Return on Assets', 'percentage'),
                ('ROI - Return On Investment', 'Return on Investment', 'percentage'),
                ('Free Cash Flow', 'Free Cash Flow', 'millions')
            ]
        }
    ]
    
    content = []
    
    for group in health_groups:
        charts = []
        for metric, title, format_type in group['metrics']:
            if metric in main_df.columns:
                chart_with_info = create_chart_with_info_tooltip(main_df, main_ticker, metric, title, comparison_data, benchmark_df, format_type)
                charts.append(chart_with_info)
        
        if charts:
            content.append(
                html.Div([
                    html.H3(group['title'], className='category-title'),
                    html.Div(charts, className='charts-grid')
                ], className='category-group')
            )
    
    return html.Div([
        html.Div([
            html.H2("Financial Health Analysis", className='section-title'),
            html.P("Monitor debt levels, liquidity position, capital efficiency, and overall financial strength relative to industry standards.", className='section-description')
        ], className='section-header'),
        html.Div(content)
    ], className='tab-content')

def create_efficiency_analysis(main_df, main_ticker, comparison_data, benchmark_df):
    """Operational efficiency analysis"""
    
    efficiency_metrics = [
        ('Asset Turnover', 'Asset Turnover Ratio', 'ratio'),
        ('Inventory Turnover Ratio', 'Inventory Turnover', 'ratio'),
        ('Receiveable Turnover', 'Receivables Turnover', 'ratio'),
        ('Days Sales In Receivables', 'Days Sales Outstanding', 'number'),
        ('Operating Cash Flow Per Share', 'Operating Cash Flow/Share', 'currency'),
        ('Free Cash Flow Per Share', 'Free Cash Flow/Share', 'currency')
    ]
    
    charts = []
    for metric, title, format_type in efficiency_metrics:
        if metric in main_df.columns:
            chart_with_info = create_chart_with_info_tooltip(main_df, main_ticker, metric, title, comparison_data, benchmark_df, format_type)
            charts.append(chart_with_info)
    
    return html.Div([
        html.Div([
            html.H2("Operational Efficiency Analysis", className='section-title'),
            html.P("Evaluate operational efficiency, working capital management, and cash generation capabilities relative to industry peers.", className='section-description')
        ], className='section-header'),
        html.Div(charts, className='charts-grid')
    ], className='tab-content')

def create_comparative_analysis(main_df, main_ticker, comparison_data, benchmark_df):
    """Comparative table analysis - SIMPLIFICADO SIN CLICKS"""
    
    if main_df.empty:
        return html.Div("Insufficient data for comparison")
    
    try:
        latest_main = main_df.iloc[-1]
        
        # Key metrics for comparison - MÉTRICAS EXPANDIDAS
        comparison_metrics = [
            ('Revenue', 'millions'), ('Net Income', 'millions'), ('EBITDA', 'millions'),
            ('Total Assets', 'millions'), ('Share Holder Equity', 'millions'),
            ('P/E', 'ratio'), ('P/B', 'ratio'), ('P/S', 'ratio'),
            ('ROE - Return On Equity', 'percentage'), ('ROA - Return On Assets', 'percentage'),
            ('Current Ratio', 'ratio'), ('Debt/Equity Ratio', 'ratio'), ('Debt/Assets', 'percentage'),
            ('Gross Margin', 'percentage'), ('Operating Margin', 'percentage'), ('Net Profit Margin', 'percentage'),
            ('Long Term Debt', 'millions'), ('Total Current Liabilities', 'millions'), 
            ('Total Liabilities', 'millions'), ('Net Debt', 'millions'),
            ('Shares Outstanding', 'shares'), ('Cash On Hand', 'millions'),
            ('MarketCap', 'millions'), ('Free Cash Flow', 'millions')
        ]
        
        table_data = []
        
        for metric, format_type in comparison_metrics:
            if metric in main_df.columns:
                row = {'Metric': metric}
                
                # Main company
                main_val = latest_main.get(metric, np.nan)
                row[main_ticker] = format_value(main_val, format_type)
                
                # Comparison companies
                if comparison_data:
                    for comp_ticker, comp_df in comparison_data.items():
                        if not comp_df.empty and metric in comp_df.columns:
                            comp_val = comp_df.iloc[-1].get(metric, np.nan)
                            row[comp_ticker] = format_value(comp_val, format_type)
                        else:
                            row[comp_ticker] = "N/A"
                
                # Industry benchmark
                if not benchmark_df.empty and metric in benchmark_df.columns:
                    benchmark_vals = benchmark_df[metric].dropna()
                    if len(benchmark_vals) > 0:
                        # Usar la mediana del benchmark (más robusto)
                        benchmark_val = benchmark_vals.median()
                        row['Industry'] = format_value(benchmark_val, format_type)
                    else:
                        row['Industry'] = "N/A"
                else:
                    row['Industry'] = "N/A"
                
                table_data.append(row)
        
        # Create table columns
        columns = ['Metric', main_ticker]
        if comparison_data:
            columns.extend(list(comparison_data.keys()))
        columns.append('Industry')
        
        # CREAR TABLA SIMPLE SIN FUNCIONALIDAD DE CLICKS
        comparison_table = dash_table.DataTable(
            id='comparison-table',
            data=table_data,
            columns=[{"name": col, "id": col} for col in columns],

            # Unificamos estilos de tabla
            style_table={
                'overflowX': 'auto',
                'border': f'2px solid {UNIFIED_COLORS["primary"]}',
                'borderRadius': '8px',
                'marginTop': '20px'
            },

            # Solo un style_cell
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'fontFamily': 'Inter',
                'fontSize': '13px',
                'color': UNIFIED_COLORS['text'],
                'backgroundColor': UNIFIED_COLORS.get('background', '#242938'),
                'border': f'1px solid {UNIFIED_COLORS["primary_dark"]}'
            },

            # Solo un style_header
            style_header={
                'backgroundColor': UNIFIED_COLORS['primary_dark'],
                'color': 'white',
                'fontWeight': '600',
                'fontSize': '14px',
                'textAlign': 'center',
                'border': f'1px solid {UNIFIED_COLORS["primary"]}'
            },

            # Estilo global para celdas de datos (filas)
            style_data={
                'border': f'1px solid {UNIFIED_COLORS["primary_dark"]}'
            },

            # Condicionales: filas alternadas
            style_data_conditional=[
                # Filas alternadas
                {
                    'if': {'row_index': i},
                    'backgroundColor': '#242938' if i % 2 == 0 else '#2a2f3e'
                } for i in range(len(table_data))
            ]
        )

        return html.Div([
            html.Div([
                html.H2("Comparative Analysis", className='section-title'),
                html.P("Comprehensive side-by-side comparison of key financial metrics across peer companies and industry benchmarks.", className='section-description')
            ], className='section-header'),
            
            html.Div([
                html.Div(id='comparison-table-container', children=[comparison_table])
            ], className='comparison-table'),
            
            # CHECKBOX PROFESIONAL PARA COLORES
            html.Div(
                [
                    dcc.Checklist(
                        id='color-toggle-checkbox',
                        options=[{'label': ' Enable Performance Colors', 'value': 'colors'}],
                        value=[],
                        style={'margin': '15px 0'},
                        inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                        labelStyle={
                            'fontSize': '14px',
                            'fontWeight': '500',
                            'color': '#e8eaed'
                        }
                    ),
                    html.P(
                        [
                            html.Span("Green: ", style={'color': '#38a169', 'fontWeight': '600'}),
                            "Better Performance  •  ",
                            html.Span("Red: ",   style={'color': '#e53e3e', 'fontWeight': '600'}),
                            "Worse Performance  •  Color intensity shows relative strength"
                        ],
                        style={
                            'fontSize': '12px',
                            'color': '#7ca0ac',
                            'fontStyle': 'italic',
                            'margin': '5px 0 15px 25px'
                        }
                    )
                ],
                style={
                    'background':    '#2d3748',
                    'padding':       '15px',
                    'borderRadius':  '8px',
                    'border':        '1px solid #01d0fa',
                    'marginTop':     '20px'
                }
            )    
        ], className='tab-content')
        
    except Exception as e:
        return html.Div(f"Error creating comparison: {str(e)}")

# Callback para colores de rendimiento
@app.callback(
    Output('comparison-table', 'style_data_conditional'),
    [Input('color-toggle-checkbox', 'value')],
    [State('comparison-table', 'data'),
     State('companies-pool', 'value')]
)
def update_table_colors(color_enabled, table_data, selected_companies):
    """Maneja colores de rendimiento en la tabla"""
    
    if not table_data or not selected_companies:
        return []
    
    main_ticker = selected_companies[0]  # Primera empresa = primary
    comparison_tickers = selected_companies[1:] if len(selected_companies) > 1 else []
    
    # Estilos base
    base_styles = [
        {
            'if': {'row_index': i},
            'backgroundColor': '#1e293b' if i % 2 == 0 else '#334155',
            'color': '#e8eaed'
        } for i in range(len(table_data))
    ]
    
    # Añadir estilos de color si está habilitado
    if 'colors' in color_enabled and len(table_data) > 1:
        color_styles = calculate_performance_colors(table_data, main_ticker, comparison_tickers)
        return base_styles + color_styles
    
    return base_styles

def calculate_performance_colors(table_data, main_ticker, comparison_tickers):
    """Calcula colores de rendimiento para métricas comparables"""
    color_styles = []
    
    # Métricas que son comparables para colores (solo ratios, margins, percentajes)
    comparable_for_colors = {
        'P/E': 'lower', 'P/B': 'lower', 'P/S': 'lower',
        'ROE - Return On Equity': 'higher',
        'ROA - Return On Assets': 'higher',
        'ROI - Return On Investment': 'higher',
        'Current Ratio': 'higher', 'Debt/Equity Ratio': 'lower',
        'Debt/Assets': 'lower',
        'Gross Margin': 'higher',
        'Operating Margin': 'higher',
        'Net Profit Margin': 'higher',
        'EBITDA Margin': 'higher',
        'Asset Turnover': 'higher'
    }
    
    for i, row in enumerate(table_data):
        metric = row['Metric']
        
        if metric in comparable_for_colors:
            better_direction = comparable_for_colors[metric]
            
            # Extraer valores numéricos de las celdas
            metric_values = {}
            all_columns = [main_ticker] + (comparison_tickers if comparison_tickers else []) + ['Industry']
            
            for col in all_columns:
                if col in row:
                    value_str = row[col]
                    # Extraer número de la string formateada
                    try:
                        if value_str and value_str != "N/A":
                            # Remover formateado y extraer número
                            clean_val = value_str.replace('$', '').replace(',', '').replace('%', '').replace('x', '')
                            if 'B' in clean_val:
                                clean_val = float(clean_val.replace('B', '')) * 1e9
                            elif 'M' in clean_val:
                                clean_val = float(clean_val.replace('M', '')) * 1e6
                            elif 'K' in clean_val:
                                clean_val = float(clean_val.replace('K', '')) * 1e3
                            else:
                                clean_val = float(clean_val)
                            metric_values[col] = clean_val
                    except:
                        continue
            
            if len(metric_values) >= 2:
                values_list = list(metric_values.values())
                min_val = min(values_list)
                max_val = max(values_list)
                
                if max_val != min_val:
                    for col in metric_values.keys():
                        if col != 'Industry':  # No colorear Industry
                            value = metric_values[col]
                            
                            # Calcular intensidad del color (0-1)
                            if better_direction == 'higher':
                                intensity = (value - min_val) / (max_val - min_val)
                                if intensity >= 0.7:
                                    color = f"rgba(34, 197, 94, {0.3 + intensity * 0.4})"  # Verde
                                elif intensity >= 0.4:
                                    color = f"rgba(34, 197, 94, {0.2 + intensity * 0.3})"  
                                else:
                                    color = f"rgba(239, 68, 68, {0.2 + (1-intensity) * 0.3})"  # Rojo
                            else:
                                intensity = 1 - (value - min_val) / (max_val - min_val)
                                if intensity >= 0.7:
                                    color = f"rgba(34, 197, 94, {0.3 + intensity * 0.4})"  # Verde
                                elif intensity >= 0.4:
                                    color = f"rgba(34, 197, 94, {0.2 + intensity * 0.3})"  
                                else:
                                    color = f"rgba(239, 68, 68, {0.2 + (1-intensity) * 0.3})"  # Rojo
                            
                            color_styles.append({
                                'if': {'row_index': i, 'column_id': col},
                                'backgroundColor': color,
                                'color': '#1f2937',
                                'fontWeight': '600'
                            })
    
    return color_styles

def format_value(value, format_type):
    """Format values for display in tables - CORREGIDO"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    try:
        if format_type == 'millions':
            return format_large_number(value)  # Usa el formato T/B/M/K corregido
        elif format_type == 'percentage':
            return f"{value:.1f}%"
        elif format_type == 'ratio':
            return f"{value:.2f}x"
        elif format_type == 'currency':
            return f"${value:.2f}"
        elif format_type == 'shares':  # CORREGIDO: formato para shares sin $
            return format_shares_number(value)
        else:
            return f"{value:.2f}"
    except:
        return "N/A"

if __name__ == '__main__':

    app.run(debug=False, host="0.0.0.0", port=8052)
