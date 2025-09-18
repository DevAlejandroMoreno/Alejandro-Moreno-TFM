"""
Professional Investment Hub Dashboard - V12.0 OPTIMIZED
Sistema integrado de análisis fundamental y técnico para inversores profesionales
Versión optimizada con Market Cap y mejoras de rendimiento
"""

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, ALL
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import subprocess
import webbrowser
import json
import logging
from typing import Dict, List, Tuple, Optional
import socket
import netifaces
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== CONFIGURACIÓN PROFESIONAL ======================
class Config:
    """Configuración centralizada del sistema - Estilo profesional unificado"""

    # Obtener la ruta absoluta del directorio raíz del proyecto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Rutas de datos
    VALUATIONS_PATH = os.path.join(project_root, "data", "Results", "Fundamental_Data_Results.parquet")
    TECHNICAL_PATH = os.path.join(project_root, "data", "Results", "Technical_Data_Results.parquet")
    SCREENER_PATH = os.path.join(project_root, 'data', 'Ticker_List', 'screener.parquet')

    # Configuración de visualización
    MAX_TABLE_ROWS = 1000
    DEFAULT_PAGE_SIZE = 50
    
    # PALETA DE COLORES UNIFICADA
    COLORS = {
        'bg': '#0f1419',           # Fondo principal muy oscuro
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
    
    # Mapeo de señales a colores
    SIGNAL_COLORS = {
        'COMPRA FUERTE': '#00ff88',
        'COMPRA': '#00cc66',
        'NEUTRAL': '#ffaa00',
        'VENTA': '#ff6666',
        'VENTA FUERTE': '#ff3366',
        'SIN DATOS': '#6b7280'
    }

# ====================== GESTIÓN DE DATOS MEJORADA ======================
class DataManager:
    """Gestor centralizado de datos con manejo robusto de errores"""
    
    def __init__(self):
        self.fundamental_data = None
        self.technical_data = None
        self.screener_data = None
        self.combined_data = pd.DataFrame()
        self.detailed_data = pd.DataFrame()
        self.load_all_data()
    
    def load_all_data(self):
        """Carga todos los datos necesarios con manejo de errores"""
        try:
            # Cargar datos fundamentales
            if os.path.exists(Config.VALUATIONS_PATH):
                self.fundamental_data = pd.read_parquet(Config.VALUATIONS_PATH)
                logger.info(f"✓ Datos fundamentales cargados: {len(self.fundamental_data)} empresas")
            else:
                logger.warning(f"Archivo de valoraciones no encontrado: {Config.VALUATIONS_PATH}")
                self.fundamental_data = pd.DataFrame()
            
            # Cargar datos técnicos
            if os.path.exists(Config.TECHNICAL_PATH):
                self.technical_data = pd.read_parquet(Config.TECHNICAL_PATH)
                # Limpiar nombres de columnas
                self.technical_data.columns = [col.strip() for col in self.technical_data.columns]
                logger.info(f"✓ Datos técnicos cargados: {len(self.technical_data)} empresas")
            else:
                logger.warning(f"Archivo técnico no encontrado: {Config.TECHNICAL_PATH}")
                self.technical_data = pd.DataFrame()
            
            # Cargar screener para información adicional
            if os.path.exists(Config.SCREENER_PATH):
                self.screener_data = pd.read_parquet(Config.SCREENER_PATH)
                logger.info(f"✓ Screener cargado: {len(self.screener_data)} empresas")
            else:
                logger.warning(f"Archivo screener no encontrado: {Config.SCREENER_PATH}")
                self.screener_data = pd.DataFrame()
            
            # Combinar datos
            self.combine_data()
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            self.combined_data = pd.DataFrame()
    
    def combine_data(self):
        """Combina todos los datasets en uno principal con validación"""
        try:
            # Inicializar con DataFrame vacío si no hay datos
            if self.fundamental_data is None or self.fundamental_data.empty:
                if self.technical_data is None or self.technical_data.empty:
                    logger.warning("No hay datos para combinar")
                    self.combined_data = pd.DataFrame()
                    return
                else:
                    # Solo usar datos técnicos
                    self.combined_data = self._prepare_technical_data()
                    return
            
            # Preparar datos fundamentales - mantener solo columnas necesarias
            fund_cols = ['Ticker', 'Sector', 'Industry', 
                        'Bear_Case', 'Base_Case', 'Bull_Case', 'DCF_Value',
                        'Comp_P/E', 'Comp_EV/EBITDA', 'Comp_P/B', 'Comp_P/S',
                        'Comp_Average', 'RI_Value', 'ML_Ensemble', 'MC_Mean',
                        'MC_P10', 'MC_P50', 'MC_P90', 'Fair_Value']
            
            # Verificar columnas existentes
            fund_cols_available = [col for col in fund_cols if col in self.fundamental_data.columns]
            fund_df = self.fundamental_data[fund_cols_available].copy()
            
            # Preparar datos técnicos si existen
            if self.technical_data is not None and not self.technical_data.empty:
                tech_df = self._prepare_technical_data()
                
                # Combinar datasets
                self.combined_data = pd.merge(
                    fund_df, 
                    tech_df, 
                    on='Ticker', 
                    how='outer',
                    suffixes=('', '_tech')
                )
            else:
                self.combined_data = fund_df
            
            # Añadir información del screener si está disponible
            if self.screener_data is not None and not self.screener_data.empty:
                self._add_screener_data()
            
            # Respaldar Current_Price si falta
            if 'Current_Price' not in self.combined_data.columns or self.combined_data['Current_Price'].isnull().all():
                # Si no hay Current_Price del screener, intentar usar Tech_Price de datos técnicos
                if 'Tech_Price' in self.combined_data.columns:
                    self.combined_data['Current_Price'] = self.combined_data['Tech_Price']
                # Si tampoco hay Tech_Price, mantener el Current_Price original de fundamental
                elif 'Current_Price' in self.fundamental_data.columns:
                    self.combined_data['Current_Price'] = self.fundamental_data['Current_Price']

            # Filtrar empresas sin datos esenciales
            self._filter_incomplete_data()
            
            # Calcular métricas adicionales - FORMATO NUEVO
            self._calculate_metrics()
            
            # Crear dataset detallado
            self._create_detailed_data()
            
            # Limpiar datos
            self.combined_data = self.combined_data.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"✓ Datos combinados: {len(self.combined_data)} empresas")
            
        except Exception as e:
            logger.error(f"Error combinando datos: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.combined_data = pd.DataFrame()
    
    def _filter_incomplete_data(self):
        """Filtra empresas sin datos esenciales"""
        if self.combined_data.empty:
            return
            
        # Filtrar empresas sin Industry, Company o Current_Price
        essential_cols = ['Industry', 'Company', 'Current_Price']
        available_cols = [col for col in essential_cols if col in self.combined_data.columns]
        
        if available_cols:
            self.combined_data = self.combined_data.dropna(subset=available_cols)
            logger.info(f"✓ Datos filtrados: {len(self.combined_data)} empresas con información completa")
    
    def _prepare_technical_data(self):
        """Prepara datos técnicos con validación de columnas"""
        tech_df = self.technical_data.copy()
        
        # Mapeo de columnas técnicas
        tech_rename = {
            'Precio Actual': 'Tech_Price',
            'Diario': 'Signal_Daily',
            'Semanal': 'Signal_Weekly', 
            'Mensual': 'Signal_Monthly',
            'Stop Loss - Diario': 'SL_Daily',
            'Take Profit - Diario': 'TP_Daily',
            'Stop Loss - Semanal': 'SL_Weekly',
            'Take Profit - Semanal': 'TP_Weekly',
            'Stop Loss - Mensual': 'SL_Monthly',
            'Take Profit - Mensual': 'TP_Monthly',
            'Risk:Reward - Diario': 'RR_Daily',
            'Risk:Reward - Semanal': 'RR_Weekly',
            'Risk:Reward - Mensual': 'RR_Monthly',
            'Risk:Reward - Media': 'RR_Average'
        }
        
        # Renombrar solo columnas que existen
        rename_dict = {old: new for old, new in tech_rename.items() if old in tech_df.columns}
        tech_df = tech_df.rename(columns=rename_dict)
        
        # Seleccionar columnas técnicas relevantes que existen
        tech_cols = ['Ticker', 'Tech_Price', 'Signal_Daily', 'Signal_Weekly', 
                    'Signal_Monthly', 'SL_Daily', 'TP_Daily', 'SL_Weekly',
                    'TP_Weekly', 'SL_Monthly', 'TP_Monthly', 'RR_Daily',
                    'RR_Weekly', 'RR_Monthly', 'RR_Average']
        tech_cols_available = [col for col in tech_cols if col in tech_df.columns]
        
        return tech_df[tech_cols_available]
    
    def _add_screener_data(self):
        """Añade datos del screener con validación"""
        try:
            screener_cols_map = {
                'ticker': 'Ticker',
                'comp_name_2': 'Company',
                'market_val': 'Market_Cap_Raw',
                'last_close': 'Current_Price',                
                'pe_ratio': 'PE',
                'peg_ratio': 'PEG',
                'price_to_book': 'PB',
                'beta': 'Beta',
                'avg_volume': 'Avg_Volume'
            }
            
            # Seleccionar solo columnas disponibles
            available_cols = [col for col in screener_cols_map.keys() if col in self.screener_data.columns]
            if available_cols:
                screener_subset = self.screener_data[available_cols].copy()
                screener_subset = screener_subset.rename(columns=screener_cols_map)
                
                # Calcular Market Cap (multiplicar por 1000) y formatear
                if 'Market_Cap_Raw' in screener_subset.columns:
                    screener_subset['Market_Cap_Num'] = screener_subset['Market_Cap_Raw'] / 1000
                    screener_subset['Market_Cap'] = screener_subset['Market_Cap_Num'].apply(self._format_market_cap)
                    screener_subset.drop('Market_Cap_Raw', axis=1, inplace=True)
                
                # Si ya existe Company en combined_data, reemplazar con la del screener
                if 'Company' in self.combined_data.columns:
                    self.combined_data.drop('Company', axis=1, inplace=True)
                
                self.combined_data = pd.merge(
                    self.combined_data,
                    screener_subset,
                    on='Ticker',
                    how='left'
                )
        except Exception as e:
            logger.warning(f"Error añadiendo datos del screener: {e}")
    
    def _format_market_cap(self, value):
        """Formatea el Market Cap con sufijos B/M/K"""
        try:
            if pd.isna(value) or value == 0:
                return "N/A"
            
            value = float(value)
            abs_value = abs(value)
            
            if abs_value >= 1e9:  # Billones
                return f"${value/1e9:.2f}B"
            elif abs_value >= 1e6:  # Millones
                return f"${value/1e6:.2f}M"
            elif abs_value >= 1e3:  # Miles
                return f"${value/1e3:.2f}K"
            else:
                return f"${value:.2f}"
        except:
            return "N/A"
    
    def _calculate_metrics(self):
        """Calcula métricas adicionales - Versión mejorada"""
        try:
            df = self.combined_data
            
            # Asegurar que las columnas de precio son numéricas
            if 'Current_Price' in df.columns:
                df['Current_Price'] = pd.to_numeric(df['Current_Price'], errors='coerce')
            
            # NUEVO FORMATO: Crear columnas combinadas con precio y porcentaje
            price_columns = {
                'Bear_Case': 'Bear_Price_Combined',
                'Base_Case': 'Base_Price_Combined', 
                'Bull_Case': 'Bull_Price_Combined'
            }
            
            for source_col, target_col in price_columns.items():
                if source_col in df.columns:
                    # Asegurar que la columna fuente es numérica
                    df[source_col] = pd.to_numeric(df[source_col], errors='coerce')
                    
                    # Crear columna combinada
                    df[target_col] = df.apply(
                        lambda row: self._format_price_with_percentage(
                            row[source_col], row['Current_Price']
                        ) if (pd.notna(row[source_col]) and 
                            pd.notna(row['Current_Price']) and 
                            row['Current_Price'] != 0)
                        else 'N/A', 
                        axis=1
                    )
            
            # Resto del código para scores técnicos...
            signal_map = {'COMPRA FUERTE': 2, 'COMPRA': 1, 'NEUTRAL': 0, 
                        'VENTA': -1, 'VENTA FUERTE': -2}
            
            score_cols = []
            for col in ['Signal_Daily', 'Signal_Weekly', 'Signal_Monthly']:
                if col in df.columns:
                    score_col = f'{col}_Score'
                    df[score_col] = df[col].map(signal_map).fillna(0)
                    score_cols.append(score_col)
            
            if score_cols:
                df['Technical_Score'] = df[score_cols].mean(axis=1)
            
            if 'Bear_Case' in df.columns and 'Bull_Case' in df.columns:
                df['Target_Range'] = df.apply(
                    lambda x: f"${x['Bear_Case']:.0f}-${x['Bull_Case']:.0f}" 
                    if pd.notna(x['Bear_Case']) and pd.notna(x['Bull_Case']) else 'N/A', 
                    axis=1
                )
                
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _format_price_with_percentage(self, target_price, current_price):
        """Formatea el precio objetivo con el porcentaje de diferencia - Versión mejorada"""
        try:
            if (pd.isna(target_price) or pd.isna(current_price) or 
                current_price == 0 or target_price == 0):
                return 'N/A'
            
            # Calcular diferencia porcentual
            ratio = target_price / current_price
            diff_pct = ratio * 100 - 100

            
            # Asegurar formato consistente
            sign = "+" if diff_pct >= 0 else ""
            
            # Formatear el resultado con formato consistente
            return f"{sign}{diff_pct:.1f}% (${target_price:.2f})"
        except:
            return 'N/A'
    
    def _create_detailed_data(self):
        """Crea un dataset con toda la información detallada"""
        self.detailed_data = self.combined_data.copy()
    
    def get_filtered_data(self, filters: Dict) -> pd.DataFrame:
        """Aplica filtros a los datos con validación"""
        if self.combined_data.empty:
            return pd.DataFrame()
        
        df = self.combined_data.copy()
        
        try:
            # Filtro por sector
            if filters.get('sectors') and 'Sector' in df.columns:
                df = df[df['Sector'].isin(filters['sectors'])]
            
            # Filtro por industria
            if filters.get('industries') and 'Industry' in df.columns:
                df = df[df['Industry'].isin(filters['industries'])]
            
            # Filtro por señales técnicas
            for timeframe in ['Daily', 'Weekly', 'Monthly']:
                signal_key = f'signal_{timeframe.lower()}'
                col_name = f'Signal_{timeframe}'
                if filters.get(signal_key) and col_name in df.columns:
                    df = df[df[col_name].isin(filters[signal_key])]
            
            # Filtro por tickers específicos
            if filters.get('tickers') and 'Ticker' in df.columns:
                df = df[df['Ticker'].isin(filters['tickers'])]
            
        except Exception as e:
            logger.error(f"Error aplicando filtros: {e}")
        
        return df

# ====================== COMPONENTES UI MEJORADOS ======================
class UIComponents:
    """Componentes de interfaz profesionales con estilo institucional unificado"""
    
    @staticmethod
    def create_control_panel(data_manager, server_ip):
        """Panel de controles con estilo institucional"""
        df = data_manager.combined_data
        
        # Opciones para dropdowns
        sectors = sorted(df['Sector'].dropna().unique()) if 'Sector' in df.columns else []
        tickers = sorted(df['Ticker'].dropna().unique()) if 'Ticker' in df.columns else []
        
        return html.Div([
            # TÍTULO PRINCIPAL INSTITUCIONAL
            html.Div([
                html.H1("QUANTITATIVE ANALYTICS PLATFORM", style={
                    'color': Config.COLORS['primary'],
                    'fontSize': '2.8rem',
                    'fontWeight': '300',
                    'margin': '0',
                    'textAlign': 'center',
                    'letterSpacing': '4px',
                    'textTransform': 'uppercase',
                    'fontFamily': 'Rajdhani, monospace'
                }),
                html.Div("INSTITUTIONAL-GRADE INVESTMENT ANALYSIS SYSTEM", style={
                    'color': Config.COLORS['text_secondary'],
                    'fontSize': '0.9rem',
                    'textAlign': 'center',
                    'marginTop': '8px',
                    'letterSpacing': '2px',
                    'fontWeight': '400'
                })
            ], style={
                'padding': '30px',
                'borderBottom': f'1px solid {Config.COLORS["primary"]}',
                'marginBottom': '25px',
                'background': f'linear-gradient(180deg, {Config.COLORS["bg_secondary"]} 0%, {Config.COLORS["bg"]} 100%)'
            }),
            
            html.Div([
                # Fila 1: Filtros principales
                html.Div([
                    html.Div([
                        html.Label("SECTOR FILTER", style={
                            'fontSize': '0.8rem',
                            'fontWeight': 'bold',
                            'color': Config.COLORS['text_secondary'],
                            'marginBottom': '8px',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px'
                        }),
                        dcc.Dropdown(
                            id='filter-sector',
                            options=[{'label': s, 'value': s} for s in sectors],
                            multi=True,
                            placeholder="Select sectors...",
                            style={'fontSize': '0.9rem'},
                            className='custom-dropdown'
                        )
                    ], style={'flex': '1', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("INDUSTRY FILTER", style={
                            'fontSize': '0.8rem',
                            'fontWeight': 'bold',
                            'color': Config.COLORS['text_secondary'],
                            'marginBottom': '8px',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px'
                        }),
                        dcc.Dropdown(
                            id='filter-industry',
                            options=[],
                            multi=True,
                            placeholder="Select industries...",
                            style={'fontSize': '0.9rem'},
                            className='custom-dropdown'
                        )
                    ], style={'flex': '1', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("SIGNAL FILTER", style={
                            'fontSize': '0.8rem',
                            'fontWeight': 'bold',
                            'color': Config.COLORS['text_secondary'],
                            'marginBottom': '8px',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px'
                        }),
                        dcc.Dropdown(
                            id='filter-signals',
                            options=[
                                {'label': 'Strong Buy', 'value': 'COMPRA FUERTE'},
                                {'label': 'Buy', 'value': 'COMPRA'},
                                {'label': 'Neutral', 'value': 'NEUTRAL'},
                                {'label': 'Sell', 'value': 'VENTA'},
                                {'label': 'Strong Sell', 'value': 'VENTA FUERTE'}
                            ],
                            multi=True,
                            placeholder="Filter by signals...",
                            style={'fontSize': '0.9rem'},
                            className='custom-dropdown'
                        )
                    ], style={'flex': '1', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("TICKER SEARCH", style={
                            'fontSize': '0.8rem',
                            'fontWeight': 'bold',
                            'color': Config.COLORS['text_secondary'],
                            'marginBottom': '8px',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px'
                        }),
                        dcc.Dropdown(
                            id='filter-tickers',
                            options=[{'label': t, 'value': t} for t in tickers],
                            multi=True,
                            placeholder="Search tickers...",
                            style={'fontSize': '0.9rem'},
                            className='custom-dropdown'
                        )
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'marginBottom': '24px', 'gap': '16px'}),
                
                # Fila 2: Búsqueda y acciones
                html.Div([
                    html.Div([
                        # Botón de Fundamental Analysis - Ahora es un enlace
                        html.A(
                            "FUNDAMENTAL ANALYSIS",
                            href=f"http://{server_ip}:8052",
                            target="_blank",
                            style={
                                'background': f'linear-gradient(135deg, #00d4ff, #0099cc)',
                                'border': 'none',
                                'color': 'white',
                                'padding': '12px 24px',
                                'borderRadius': '8px',
                                'fontWeight': 'bold',
                                'fontSize': '0.9rem',
                                'cursor': 'pointer',
                                'marginRight': '12px',
                                'textTransform': 'uppercase',
                                'letterSpacing': '0.5px',
                                'boxShadow': '0 4px 16px rgba(59, 130, 246, 0.3)',
                                'transition': 'all 0.3s ease',
                                'textDecoration': 'none',
                                'display': 'inline-block'
                            }
                        ),
                        # Botón de Technical Analysis - Ahora es un enlace
                        html.A(
                            "TECHNICAL ANALYSIS",
                            href=f"http://{server_ip}:8051", 
                            target="_blank",
                            style={
                                'background': f'linear-gradient(135deg, #00d4ff, #0099cc)',
                                'border': 'none',
                                'color': 'white',
                                'padding': '12px 24px',
                                'borderRadius': '8px',
                                'fontWeight': 'bold',
                                'fontSize': '0.9rem',
                                'cursor': 'pointer',
                                'textTransform': 'uppercase',
                                'letterSpacing': '0.5px',
                                'boxShadow': '0 4px 16px rgba(16, 185, 129, 0.3)',
                                'transition': 'all 0.3s ease',
                                'textDecoration': 'none',
                                'display': 'inline-block'
                            }
                        )
                    ], style={'flex': '1', 'display': 'flex', 'alignItems': 'flex-end', 'marginBottom': '5px'}),
                    
                    html.Div([
                        html.Button("APPLY FILTERS", 
                                   id='btn-apply-filters',
                                   style={
                                       'background': f'linear-gradient(135deg, #00d4ff, #0099cc)',
                                       'border': 'none',
                                       'color': 'white',
                                       'padding': '12px 28px',
                                       'borderRadius': '8px',
                                       'fontWeight': 'bold',
                                       'fontSize': '0.9rem',
                                       'cursor': 'pointer',
                                       'marginRight': '12px',
                                       'textTransform': 'uppercase',
                                       'letterSpacing': '0.5px',
                                       'boxShadow': '0 4px 16px rgba(16, 185, 129, 0.3)'
                                   }),
                        html.Button("RESET", 
                                   id='btn-reset-filters',
                                   style={
                                       'background': Config.COLORS['bg_tertiary'],
                                       'border': f'2px solid {Config.COLORS["border"]}',
                                       'color': Config.COLORS['text'],
                                       'padding': '12px 28px',
                                       'borderRadius': '8px',
                                       'fontWeight': 'bold',
                                       'fontSize': '0.9rem',
                                       'cursor': 'pointer',
                                       'textTransform': 'uppercase',
                                       'letterSpacing': '0.5px'
                                   })
                    ], style={'flex': '1', 'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'flex-end', 'marginBottom': '5px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], style={
                'padding': '0 30px 30px 30px'
            }),
            
            # Panel de peers recomendados
            html.Div(id='competitors-suggestion', style={'marginBottom': '20px'})
        ], style={
            'background': Config.COLORS['bg_secondary'],
            'borderRadius': '8px',
            'marginBottom': '20px',
            'border': f'1px solid {Config.COLORS["border"]}'
        })
    
    @staticmethod
    def create_main_table(data):
        """Crea la tabla principal con nuevas columnas combinadas"""
        if data.empty:
            return html.Div("No data available. Please check data sources.", 
                          style={
                              'padding': '60px', 
                              'textAlign': 'center', 
                              'color': Config.COLORS['text_secondary'],
                              'fontSize': '1.1rem'
                          })
        
        # Configurar columnas con NUEVOS NOMBRES
        columns_config = []
        
        # Columnas esenciales
        if 'Ticker' in data.columns:
            columns_config.append({'name': 'Ticker', 'id': 'Ticker', 'type': 'text'})
        if 'Company' in data.columns:
            columns_config.append({'name': 'Company', 'id': 'Company', 'type': 'text'})
        if 'Industry' in data.columns:
            columns_config.append({'name': 'Industry', 'id': 'Industry', 'type': 'text'})
        
        # Precio actual
        if 'Current_Price' in data.columns:
            columns_config.append({'name': 'Price', 'id': 'Current_Price', 'type': 'numeric', 'format': {'specifier': '$.2f'}})
        
        # NUEVAS COLUMNAS COMBINADAS CON NOMBRES CAMBIADOS
        if 'Bear_Price_Combined' in data.columns:
            columns_config.append({'name': 'Lower Price', 'id': 'Bear_Price_Combined', 'type': 'text'})
        if 'Base_Price_Combined' in data.columns:
            columns_config.append({'name': 'Average Price', 'id': 'Base_Price_Combined', 'type': 'text'})
        if 'Bull_Price_Combined' in data.columns:
            columns_config.append({'name': 'Higher Price', 'id': 'Bull_Price_Combined', 'type': 'text'})
        
        # Market Cap (nueva columna)
        if 'Market_Cap_Num' in data.columns:
            columns_config.append({'name': 'Market Cap (B)', 'id': 'Market_Cap_Num', 'type': 'numeric', 'format': {'specifier': '.2f'}})

        # Columnas técnicas
        if 'Signal_Daily' in data.columns:
            columns_config.append({'name': 'Daily', 'id': 'Signal_Daily', 'type': 'text'})
        if 'Signal_Weekly' in data.columns:
            columns_config.append({'name': 'Weekly', 'id': 'Signal_Weekly', 'type': 'text'})
        if 'Signal_Monthly' in data.columns:
            columns_config.append({'name': 'Monthly', 'id': 'Signal_Monthly', 'type': 'text'})
        if 'RR_Average' in data.columns:
            columns_config.append({'name': 'Risk:Reward', 'id': 'RR_Average', 'type': 'numeric', 'format': {'specifier': '.2f'}})
        
        # Preparar datos
        table_data = data.head(Config.MAX_TABLE_ROWS).to_dict('records')
        
        # Estilos condicionales mejorados
        style_conditions = UIComponents._get_table_style_conditions()
        
        return html.Div([
            dash_table.DataTable(
                id='main-table',
                data=table_data,
                columns=columns_config,
                
                # Configuración de tabla
                page_current=0,
                page_size=Config.DEFAULT_PAGE_SIZE,
                page_action='native',
                
                sort_action='native',
                sort_mode='multi',
                
                filter_action='native',
                
                # HEADER STICKY
                fixed_rows={'headers': True},
                
                # Estilos mejorados
                style_cell={
                    'backgroundColor': Config.COLORS['bg_tertiary'],
                    'color': Config.COLORS['text'],
                    'border': f'1px solid {Config.COLORS["border"]}',
                    'fontSize': '0.85rem',
                    'textAlign': 'center',
                    'padding': '12px 8px',
                    'fontFamily': 'Inter, system-ui, sans-serif'
                },
                style_header={
                    'backgroundColor': Config.COLORS['bg_secondary'],
                    'color': Config.COLORS['primary'],
                    'fontWeight': '700',
                    'fontSize': '0.8rem',
                    'textAlign': 'center',
                    'textTransform': 'uppercase',
                    'letterSpacing': '0.5px',
                    'padding': '16px 8px',
                    'borderBottom': f'2px solid {Config.COLORS["primary"]}',
                    'position': 'sticky',
                    'top': 0,
                    'zIndex': 1
                },
                style_data_conditional=style_conditions,
                
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Ticker'},
                        'textAlign': 'left',
                        'fontWeight': 'bold',
                        'color': Config.COLORS['primary'],
                        'fontSize': '0.9rem'
                    },
                    {
                        'if': {'column_id': 'Company'},
                        'textAlign': 'left',
                        'maxWidth': '200px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'fontSize': '0.8rem'
                    },
                    {
                        'if': {'column_id': ['Bear_Price_Combined', 'Base_Price_Combined', 'Bull_Price_Combined']},
                        'fontWeight': 'bold',
                        'fontSize': '0.85rem',
                        'minWidth': '140px'
                    },
                    {
                        'if': {'column_id': 'Market_Cap_Num'},
                        'fontWeight': 'bold',
                        'fontSize': '0.85rem',
                        'minWidth': '100px'
                    }
                ],
                
                style_table={
                    'overflowX': 'auto',
                    'overflowY': 'auto',
                    'maxHeight': '800px',
                    'scrollbar-width': 'thin',  # Para Firefox
                    'scrollbar-color': '#00d4ff #1a1f2e'  # Para Firefox
                }
            )
        ], style={
            'padding': '24px',
            'background': f'linear-gradient(135deg, {Config.COLORS["bg_secondary"]} 0%, {Config.COLORS["bg_tertiary"]} 100%)',
            'borderRadius': '16px',
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.3)',
            'border': f'1px solid {Config.COLORS["border"]}'
        })
    
    @staticmethod
    def create_comparative_panel(data):
        """Panel comparativo removido"""
        return html.Div()
    
    @staticmethod
    def create_technical_details_panel(data):
        """Crea un panel con detalles técnicos completos"""
        if data.empty:
            return html.Div()
        
        # Obtener precio actual para comparación
        current_price = data['Current_Price'].iloc[0] if 'Current_Price' in data.columns else 0
        
        # Mapeo de nombres más descriptivos
        metric_names = {
            'SL_Daily': 'Stop Loss - Daily',
            'TP_Daily': 'Take Profit - Daily',
            'SL_Weekly': 'Stop Loss - Weekly',
            'TP_Weekly': 'Take Profit - Weekly',
            'SL_Monthly': 'Stop Loss - Monthly',
            'TP_Monthly': 'Take Profit - Monthly',
            'RR_Daily': 'Risk:Reward - Daily',
            'RR_Weekly': 'Risk:Reward - Weekly',
            'RR_Monthly': 'Risk:Reward - Monthly'
        }
        
        # Categorías con colores
        categories = {
            'Risk Management': {
                'color': Config.COLORS['danger'],
                'metrics': ['SL_Daily', 'TP_Daily', 'SL_Weekly', 'TP_Weekly', 'SL_Monthly', 'TP_Monthly']
            },
            'Performance Metrics': {
                'color': Config.COLORS['success'],
                'metrics': ['RR_Daily', 'RR_Weekly', 'RR_Monthly']
            }
        }
        
        # Crear secciones por categoría
        sections = []
        
        for category, config in categories.items():
            section_data = []
            
            for col in config['metrics']:
                if col in data.columns and not pd.isna(data[col].iloc[0]):
                    display_name = metric_names.get(col, col.replace('_', ' '))
                    value = data[col].iloc[0]
                    
                    # Formatear valor según tipo
                    if 'RR' in col:
                        formatted_value = f"{value:.2f}:1"
                    else:
                        formatted_value = f"${value:.2f}"
                    
                    section_data.append({
                        'Metric': display_name,
                        'Value': formatted_value
                    })
            
            if section_data:
                sections.append(
                    html.Div([
                        html.H6([
                            html.Span(category, style={
                                'color': config['color'],
                                'fontSize': '1rem',
                                'fontWeight': '600'
                            })
                        ], style={
                            'marginBottom': '12px',
                            'paddingBottom': '8px',
                            'borderBottom': f'2px solid {config["color"]}',
                            'textAlign': 'center'
                        }),
                        dash_table.DataTable(
                            data=section_data,
                            columns=[
                                {'name': 'Metric', 'id': 'Metric', 'type': 'text'},
                                {'name': 'Value', 'id': 'Value', 'type': 'text'}
                            ],
                            style_cell={
                                'backgroundColor': Config.COLORS['bg_tertiary'],
                                'color': Config.COLORS['text'],
                                'border': f'1px solid {Config.COLORS["border"]}',
                                'fontSize': '0.85rem',
                                'padding': '12px 8px',
                                'textAlign': 'left'
                            },
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': 'Value'},
                                    'width': '120px',
                                    'minWidth': '120px',
                                    'maxWidth': '120px',
                                    'textAlign': 'right',
                                    'fontWeight': 'bold',
                                    'color': config['color']
                                }
                            ],
                            style_header={
                                'backgroundColor': Config.COLORS['bg_secondary'],
                                'color': config['color'],
                                'fontWeight': 'bold',
                                'fontSize': '0.8rem',
                                'textTransform': 'uppercase'
                            },
                            style_table={'overflowX': 'auto'}
                        )
                    ], style={
                        'flex': '1',
                        'margin': '0 10px',
                        'minWidth': '300px',
                        'padding': '15px',
                        'background': f'linear-gradient(135deg, {Config.COLORS["bg_tertiary"]} 0%, {Config.COLORS["bg_secondary"]} 100%)',
                        'borderRadius': '12px',
                        'border': f'1px solid {config["color"]}',
                        'boxShadow': f'0 4px 16px {config["color"]}20'
                    })
                )
        
        return html.Div([
            html.Div([
                html.H4("Complete Technical Analysis", style={
                    'color': Config.COLORS['primary'],
                    'marginBottom': '8px',
                    'fontSize': '1.4rem',
                    'fontWeight': '700',
                    'textAlign': 'center'
                }),
                html.P(f"Current Price: ${current_price:.2f}", style={
                    'color': Config.COLORS['warning'],
                    'fontSize': '1.1rem',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'marginBottom': '20px',
                    'padding': '8px 16px',
                    'background': Config.COLORS['bg_secondary'],
                    'borderRadius': '20px',
                    'border': f'2px solid {Config.COLORS["warning"]}',
                    'display': 'inline-block'
                }),
                html.Div(sections, style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'center',
                    'gap': '20px',
                    'marginTop': '20px'
                })
            ])
        ], style={
            'padding': '24px',
            'background': f'linear-gradient(135deg, {Config.COLORS["bg_secondary"]} 0%, {Config.COLORS["bg_tertiary"]} 100%)',
            'borderRadius': '16px',
            'marginTop': '24px',
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.3)',
            'border': f'1px solid {Config.COLORS["border"]}'
        })
    
    @staticmethod
    def create_fundamental_details_panel(data):
        """Crea un panel con detalles fundamentales completos"""
        if data.empty:
            return html.Div()
        
        # Obtener precio actual para comparación
        current_price = data['Current_Price'].iloc[0] if 'Current_Price' in data.columns else 0
        
        # Categorías de análisis fundamental con colores
        categories = {
            'Valuation Models': {
                'color': Config.COLORS['primary'],
                'metrics': ['DCF_Value', 'Comp_Average', 'RI_Value', 'ML_Ensemble', 'Fair_Value']
            },
            'Comparable Analysis': {
                'color': Config.COLORS['success'],
                'metrics': ['Comp_P/E', 'Comp_EV/EBITDA', 'Comp_P/B', 'Comp_P/S']
            },
            'Monte Carlo Analysis': {
                'color': Config.COLORS['accent'],
                'metrics': ['MC_Mean', 'MC_P10', 'MC_P50', 'MC_P90']
            }
        }
        
        # Nombres más descriptivos
        metric_names = {
            'DCF_Value': 'DCF Valuation',
            'Comp_P/E': 'P/E Multiple',
            'Comp_EV/EBITDA': 'EV/EBITDA Multiple',
            'Comp_P/B': 'P/B Multiple',
            'Comp_P/S': 'P/S Multiple',
            'Comp_Average': 'Comparables Average',
            'RI_Value': 'Residual Income Model',
            'ML_Ensemble': 'ML Ensemble Model',
            'MC_Mean': 'Monte Carlo Mean',
            'MC_P10': 'Monte Carlo P10',
            'MC_P50': 'Monte Carlo Median',
            'MC_P90': 'Monte Carlo P90',
            'Fair_Value': 'Fair Value Estimate'
        }
        
        # Crear secciones por categoría
        sections = []
        
        for category, config in categories.items():
            section_data = []
            
            for metric in config['metrics']:
                if metric in data.columns and not pd.isna(data[metric].iloc[0]):
                    display_name = metric_names.get(metric, metric.replace('_', ' '))
                    value = data[metric].iloc[0]
                    
                    # Calcular diferencia porcentual respecto al precio actual
                    if current_price > 0:
                        diff_pct = ((value - current_price) / current_price) * 100
                        if diff_pct > 0:
                            diff_color = Config.COLORS['bullish']
                            diff_text = f"+{diff_pct:.1f}%"
                        else:
                            diff_color = Config.COLORS['bearish']
                            diff_text = f"{diff_pct:.1f}%"
                    else:
                        diff_color = Config.COLORS['text_secondary']
                        diff_text = "N/A"
                    
                    section_data.append({
                        'Metric': display_name,
                        'Value': f"${value:.2f} ({diff_text})"
                    })
            
            if section_data:
                sections.append(
                    html.Div([
                        html.H6([
                            html.Span(category, style={
                                'color': config['color'],
                                'fontSize': '1rem',
                                'fontWeight': '600'
                            })
                        ], style={
                            'marginBottom': '12px',
                            'paddingBottom': '8px',
                            'borderBottom': f'2px solid {config["color"]}',
                            'textAlign': 'center'
                        }),
                        dash_table.DataTable(
                            data=section_data,
                            columns=[
                                {'name': 'Metric', 'id': 'Metric', 'type': 'text'},
                                {'name': 'Value', 'id': 'Value', 'type': 'text'}
                            ],
                            style_cell={
                                'backgroundColor': Config.COLORS['bg_tertiary'],
                                'color': Config.COLORS['text'],
                                'border': f'1px solid {Config.COLORS["border"]}',
                                'fontSize': '0.85rem',
                                'padding': '12px 8px',
                                'textAlign': 'left'
                            },
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': 'Value'},
                                    'width': '140px',
                                    'minWidth': '140px',
                                    'maxWidth': '140px',
                                    'textAlign': 'right',
                                    'fontWeight': 'bold',
                                    'color': config['color']
                                }
                            ],
                            style_header={
                                'backgroundColor': Config.COLORS['bg_secondary'],
                                'color': config['color'],
                                'fontWeight': 'bold',
                                'fontSize': '0.8rem',
                                'textTransform': 'uppercase'
                            },
                            style_table={'overflowX': 'auto'}
                        )
                    ], style={
                        'flex': '1',
                        'margin': '0 10px',
                        'minWidth': '320px',
                        'padding': '15px',
                        'background': f'linear-gradient(135deg, {Config.COLORS["bg_tertiary"]} 0%, {Config.COLORS["bg_secondary"]} 100%)',
                        'borderRadius': '12px',
                        'border': f'1px solid {config["color"]}',
                        'boxShadow': f'0 4px 16px {config["color"]}20'
                    })
                )
        
        return html.Div([
            html.Div([
                html.H4("Complete Fundamental Analysis", style={
                    'color': Config.COLORS['success'],
                    'marginBottom': '8px',
                    'fontSize': '1.4rem',
                    'fontWeight': '700',
                    'textAlign': 'center'
                }),
                html.P(f"Current Price: ${current_price:.2f}", style={
                    'color': Config.COLORS['warning'],
                    'fontSize': '1.1rem',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'marginBottom': '20px',
                    'padding': '8px 16px',
                    'background': Config.COLORS['bg_secondary'],
                    'borderRadius': '20px',
                    'border': f'2px solid {Config.COLORS["warning"]}',
                    'display': 'inline-block'
                }),
                html.Div(sections, style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'center',
                    'gap': '20px',
                    'marginTop': '20px'
                })
            ])
        ], style={
            'padding': '24px',
            'background': f'linear-gradient(135deg, {Config.COLORS["bg_secondary"]} 0%, {Config.COLORS["bg_tertiary"]} 100%)',
            'borderRadius': '16px',
            'marginTop': '24px',
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.3)',
            'border': f'1px solid {Config.COLORS["border"]}'
        })
    
    @staticmethod
    def _get_table_style_conditions():
        """Retorna condiciones de estilo para la tabla - Versión simplificada"""
        conditions = []
        
        # Estilos para señales técnicas
        signal_columns = ['Signal_Daily', 'Signal_Weekly', 'Signal_Monthly']
        for col in signal_columns:
            for signal, color in Config.SIGNAL_COLORS.items():
                conditions.append({
                    'if': {
                        'filter_query': f'{{{col}}} = "{signal}"',
                        'column_id': col
                    },
                    'color': color,
                    'fontWeight': 'bold'
                })
        
        # Estilos simplificados para las columnas de precios combinadas
        price_columns = ['Bear_Price_Combined', 'Base_Price_Combined', 'Bull_Price_Combined']
        for col in price_columns:
            # Para valores positivos
            conditions.append({
                'if': {
                    'column_id': col,
                    'filter_query': f'{{{col}}} contains "+"'
                },
                'color': Config.COLORS['bullish'],
                'fontWeight': 'bold'
            })
            
            # Para valores negativos
            conditions.append({
                'if': {
                    'column_id': col,
                    'filter_query': f'{{{col}}} contains "-"'
                },
                'color': Config.COLORS['bearish'],
                'fontWeight': 'bold'
            })
        
        return conditions

# ====================== DASHBOARD PRINCIPAL MEJORADO ======================
class ProfessionalInvestmentHub:
    """Dashboard principal de inversión profesional con estilo unificado"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.ui = UIComponents()
        self.server_ip = self._get_server_ip()  # Obtener IP al inicializar
        self.app = dash.Dash(
            __name__,
            suppress_callback_exceptions=True
        )
        self.app.title = "Quantitative Analytics Platform"
        self.setup_layout()
        self.setup_callbacks()
    
    def _get_server_ip(self):
        """Obtiene la dirección IP del servidor para acceso en red local"""
        try:
            # Obtener todas las interfaces de red
            interfaces = netifaces.interfaces()
            
            for interface in interfaces:
                # Ignorar interfaces virtuales y de loopback
                if interface.startswith(('lo', 'docker', 'br-', 'veth')):
                    continue
                    
                addresses = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addresses:
                    for link in addresses[netifaces.AF_INET]:
                        ip = link['addr']
                        # Filtrar direcciones IP privadas (redes locales)
                        if ip.startswith(('192.168.', '10.', '172.')):
                            return ip
                            
            # Fallback: obtener la IP mediante conexión externa
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
            
        except Exception as e:
            logger.error(f"Error obteniendo IP del servidor: {e}")
            return "127.0.0.1"
    
    def setup_layout(self):
        """Configura el layout del dashboard con estilo institucional"""
        self.app.layout = html.Div([
            # Stores para datos
            dcc.Store(id='filtered-data-store'),
            dcc.Store(id='selected-ticker-store'),
            dcc.Store(id='selected-row-store'),
            
            # Container principal
            html.Div([
                # Panel de control con título principal
                self.ui.create_control_panel(self.data_manager, self.server_ip),
                
                # Tabla principal
                html.Div(id='main-table-container'),
                
                # Panel comparativo (oculto)
                html.Div(id='comparative-panel', style={'display': 'none'}),
                
                # Panel de detalles técnicos
                html.Div(id='technical-details-panel', style={'display': 'none'}),
                
                # Panel de detalles fundamentales
                html.Div(id='fundamental-details-panel', style={'display': 'none'}),
                
                # Footer profesional
                html.Div([
                    html.Hr(style={
                        'border': f'1px solid {Config.COLORS["border"]}', 
                        'margin': '40px 0 20px 0',
                        'opacity': '0.5'
                    }),
                    html.Div([
                        html.P("QUANTITATIVE ANALYTICS PLATFORM", style={
                            'fontSize': '0.9rem',
                            'fontWeight': '600',
                            'letterSpacing': '1px',
                            'margin': '0',
                            'color': Config.COLORS['primary']
                        }),
                        html.P("Developed by Alejandro Moreno • Institutional Investment Solutions", style={
                            'fontSize': '0.8rem',
                            'fontWeight': '400',
                            'margin': '5px 0 0 0',
                            'color': Config.COLORS['text_secondary']
                        })
                    ], style={'textAlign': 'center', 'padding': '20px'})
                ])
                
            ], style={
                'maxWidth': '1920px',
                'margin': '0 auto',
                'padding': '20px'
            })
        ], style={
            'background': Config.COLORS['bg'],
            'minHeight': '100vh',
            'fontFamily': "'Inter', 'Segoe UI', system-ui, sans-serif"
        })
    
    def setup_callbacks(self):
        """Configura los callbacks del dashboard"""
        
        # Sugerir competidores - solo para el primer ticker
        @self.app.callback(
            Output('competitors-suggestion', 'children'),
            [Input('filter-tickers', 'value')]
        )
        def suggest_competitors(tickers):
            """Muestra peers recomendados solo para el primer ticker seleccionado"""
            if not tickers or len(tickers) == 0:
                return html.Div()
            
            # Solo usar el primer ticker
            first_ticker = tickers[0]
            
            try:
                # Obtener peers del primer ticker
                if not self.data_manager.screener_data.empty:
                    company_data = self.data_manager.screener_data[
                        self.data_manager.screener_data['ticker'] == first_ticker
                    ]
                    
                    if not company_data.empty:
                        sector = company_data.iloc[0].get('zacks_x_sector_desc', '')
                        industry = company_data.iloc[0].get('zacks_x_ind_desc', '')
                        
                        if sector and industry:
                            # Buscar empresas similares
                            peers = self.data_manager.screener_data[
                                (self.data_manager.screener_data['zacks_x_ind_desc'] == industry) &
                                (self.data_manager.screener_data['ticker'] != first_ticker) &
                                (self.data_manager.screener_data['market_val'].notna())
                            ].nlargest(5, 'market_val')['ticker'].tolist()
                            
                            if peers:
                                return html.Div([
                                    html.P([
                                        html.Span(f"Recommended peers for {first_ticker}: ", style={
                                            'color': Config.COLORS['text_secondary'],
                                            'fontSize': '0.8rem',
                                            'fontWeight': '500'
                                        }),
                                        html.Span(" • ".join(peers), style={
                                            'color': Config.COLORS['text_muted'],
                                            'fontSize': '0.8rem',
                                            'fontWeight': '400'
                                        })
                                    ], style={'margin': '0', 'padding': '8px 0'})
                                ], style={
                                    'borderTop': f'1px solid {Config.COLORS["border"]}',
                                    'marginTop': '16px',
                                    'paddingTop': '8px',
                                    'opacity': '0.7'
                                })
            except Exception as e:
                logger.debug(f"Error generando peers recomendados: {e}")
            
            return html.Div()
        
        # Actualizar industrias según sector
        @self.app.callback(
            Output('filter-industry', 'options'),
            Input('filter-sector', 'value')
        )
        def update_industries(sectors):
            if self.data_manager.combined_data.empty or 'Industry' not in self.data_manager.combined_data.columns:
                return []
            
            if not sectors:
                df = self.data_manager.combined_data
            else:
                df = self.data_manager.combined_data[
                    self.data_manager.combined_data['Sector'].isin(sectors)
                ]
            
            industries = sorted(df['Industry'].dropna().unique())
            return [{'label': i, 'value': i} for i in industries]
        
        # Aplicar filtros y actualizar tabla
        @self.app.callback(
            [Output('filtered-data-store', 'data'),
             Output('main-table-container', 'children'),
             Output('comparative-panel', 'children'),
             Output('comparative-panel', 'style')],
            [Input('btn-apply-filters', 'n_clicks'),
             Input('btn-reset-filters', 'n_clicks')],
            [State('filter-sector', 'value'),
             State('filter-industry', 'value'),
             State('filter-signals', 'value'),
             State('filter-tickers', 'value')]
        )
        def apply_filters(apply_clicks, reset_clicks, sectors, industries, signals, tickers):
            
            ctx = callback_context
            
            # Reset o carga inicial
            if not ctx.triggered or (ctx.triggered[0]['prop_id'] == 'btn-reset-filters.n_clicks'):
                filtered_data = self.data_manager.combined_data
            else:
                # Aplicar filtros
                filters = {}
                
                if sectors:
                    filters['sectors'] = sectors
                if industries:
                    filters['industries'] = industries
                if tickers:
                    filters['tickers'] = tickers
                
                # Aplicar filtros de señales
                if signals:
                    filters['signal_daily'] = signals
                    filters['signal_weekly'] = signals
                    filters['signal_monthly'] = signals
                
                filtered_data = self.data_manager.get_filtered_data(filters)
            
            # Actualizar componentes
            main_table = self.ui.create_main_table(filtered_data)
            
            # Panel comparativo removido
            comparative_panel = html.Div()
            panel_style = {'display': 'none'}
            
            # Retornar datos serializados de forma segura
            try:
                json_data = filtered_data.to_json(date_format='iso', default_handler=str) if not filtered_data.empty else '{}'
            except:
                json_data = '{}'
            
            return json_data, main_table, comparative_panel, panel_style
        
        # CALLBACK CORREGIDO - Usa derived_virtual_data para filtros internos
        @self.app.callback(
            [Output('technical-details-panel', 'children'),
            Output('technical-details-panel', 'style'),
            Output('fundamental-details-panel', 'children'),
            Output('fundamental-details-panel', 'style')],
            [Input('main-table', 'active_cell')],
            [State('main-table', 'derived_virtual_data'),
            State('filtered-data-store', 'data')]
        )
        def show_details(active_cell, table_visible_data, filtered_data_json):
            if active_cell and table_visible_data and filtered_data_json:
                try:
                    # Obtener la fila seleccionada de los datos VISIBLES (después de filtros internos)
                    selected_row = table_visible_data[active_cell['row']]
                    ticker = selected_row['Ticker']
                    clicked_column = active_cell.get('column_id', '')
                    
                    # Cargar los datos filtrados completos
                    filtered_data = pd.read_json(filtered_data_json)
                    
                    # Buscar el ticker en los datos filtrados completos
                    company_data = filtered_data[filtered_data['Ticker'] == ticker]
                    
                    if not company_data.empty:
                        # LÓGICA CON NUEVOS NOMBRES DE COLUMNAS
                        
                        # Complete Fundamental Details: Bear Price, Base Price, Bull Price
                        fundamental_trigger_columns = ['Bear_Price_Combined', 'Base_Price_Combined', 'Bull_Price_Combined']
                        show_fundamental = clicked_column in fundamental_trigger_columns
                        
                        # Complete Technical Details: Daily, Weekly, Monthly, Risk:Reward  
                        technical_trigger_columns = ['Signal_Daily', 'Signal_Weekly', 'Signal_Monthly', 'RR_Average']
                        show_technical = clicked_column in technical_trigger_columns
                        
                        # Crear paneles según la columna clickeada
                        technical_details = None
                        technical_style = {'display': 'none'}
                        fundamental_details = None
                        fundamental_style = {'display': 'none'}
                        
                        if show_technical:
                            technical_details = self.ui.create_technical_details_panel(company_data)
                            technical_style = {'display': 'block'}
                        
                        if show_fundamental:
                            fundamental_details = self.ui.create_fundamental_details_panel(company_data)
                            fundamental_style = {'display': 'block'}
                        
                        return technical_details, technical_style, fundamental_details, fundamental_style
                
                except Exception as e:
                    logger.error(f"Error showing details: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            return None, {'display': 'none'}, None, {'display': 'none'}
    
    def run(self, debug=False, port=8050):
        """Ejecuta el dashboard con información mejorada"""
        print("\n" + "="*100)
        print(" INSTITUTIONAL QUANTITATIVE ANALYSIS - UNIFIED PLATFORM ".center(100, "="))
        print("="*100)
        print("\n🚀 INSTITUTIONAL-GRADE INVESTMENT ANALYSIS PLATFORM")
        print("📊 Integrated fundamental & technical analysis with unified navigation")
        print("🎯 Real-time screening, opportunity identification & peer analysis")
        print("💼 Professional-grade tools for institutional investors")
        print(f"\n🔗 INTEGRATED DASHBOARDS:")
        print(f"   • Main Hub: http://{self.server_ip}:{port}")
        print(f"   • Fundamental Analysis: http://{self.server_ip}:8052")
        print(f"   • Technical Analysis: http://{self.server_ip}:8051")
        print("\n📌 NAVIGATION:")
        print("   • Use FUNDAMENTAL/TECHNICAL buttons for specialized analysis")
        print("   • Click Bear/Base/Bull Price columns for fundamental details")
        print("   • Click Daily/Weekly/Monthly/Risk columns for technical details")
        print("\n" + "="*100 + "\n")
        
        # Aplicar estilos CSS personalizados
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>Quantitative Analytics Platform</title>
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
                    
                    /* Scrollbar personalizada */
                    ::-webkit-scrollbar {
                        width: 12px !important;
                        height: 12px !important;
                    }
                    
                    ::-webkit-scrollbar-track {
                        background: #1a1f2e !important;
                        border-radius: 4px !important;
                    }
                    
                    ::-webkit-scrollbar-thumb {
                        background: #00d4ff !important;
                        border-radius: 4px !important;
                    }
                    
                    ::-webkit-scrollbar-thumb:hover {
                        background: #0099cc !important;
                    }
                    /* Para Firefox */
                    * {
                        scrollbar-width: thin !important;
                        scrollbar-color: #00d4ff #1a1f2e !important;
                    }
                                            
                    /* Animaciones suaves */
                    button {
                        transition: all 0.3s ease;
                    }
                    
                    button:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
                    }
                    
                    /* Cards y contenedores */
                    .dash-table-container {
                        background: #1a1f2e;
                        border-radius: 8px;
                        padding: 20px;
                        border: 1px solid #2a2f3e;
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
        
        
        self.app.run(debug=debug, port=port, host='0.0.0.0')

# ====================== MAIN ======================
if __name__ == "__main__":
    hub = ProfessionalInvestmentHub()
    # Escuchar en todas las interfaces para que sea accesible externamente (LAN)
    hub.app.run(debug=False, host="0.0.0.0", port=8050)