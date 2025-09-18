import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import glob
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio padre al path para importar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Importar las clases necesarias del código original
from Dashboard.Technical_Dashboard import QuantitativeConfig, QuantitativeDataManager, QuantitativeSRAnalyzer, QuantitativePatternAnalyzer, QuantitativeSignalSystem

# Calcular número de workers como núcleos disponibles - 1
num_workers = max(1, os.cpu_count() - 1)

class TechnicalDataGenerator:
    def __init__(self):
        self.config = QuantitativeConfig()
        self.data_manager = QuantitativeDataManager(self.config)
        self.sr_analyzer = QuantitativeSRAnalyzer(self.config)
        self.pattern_analyzer = QuantitativePatternAnalyzer(self.config)
        self.signal_system = QuantitativeSignalSystem(self.config)
        
        # Ruta relativa para el archivo de salida
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.output_path = os.path.join(project_root, "data", "Results", "Technical_Data_Results.parquet")
        
    def get_all_tickers(self):
        """Obtiene todos los tickers disponibles"""
        try:
            if os.path.exists(self.config.SCREENER_PATH):
                screener_df = pd.read_parquet(self.config.SCREENER_PATH)
                tickers = screener_df['ticker'].dropna().unique()
            else:
                files = glob.glob(os.path.join(self.config.DATA_PATH, "*.parquet"))
                tickers = [os.path.basename(f).replace('.parquet', '') for f in files]
            return tickers
        
        except Exception as e:
            print(f"Error obteniendo tickers: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'INTC']
    
    def get_daily_price(self, ticker):
        """Obtiene el precio actual de la granularidad diaria"""
        try:
            df = self.data_manager.load_and_preprocess(ticker, 'diario')
            if df.empty:
                # Si no hay datos diarios, intentar con semanal
                df = self.data_manager.load_and_preprocess(ticker, 'semanal')
                if df.empty:
                    # Si no hay semanal, intentar con mensual
                    df = self.data_manager.load_and_preprocess(ticker, 'mensual')
                    if df.empty:
                        return np.nan
            return df['close'].iloc[-1]
        except Exception as e:
            print(f"Error obteniendo precio para {ticker}: {e}")
            return np.nan
    
    def calculate_risk_reward(self, signal, current_price, stop_loss, take_profit):
        """Calcula el ratio Risk:Reward según el tipo de señal"""
        if pd.isna(current_price) or pd.isna(stop_loss) or pd.isna(take_profit):
            return np.nan
            
        if signal in ['COMPRA', 'COMPRA FUERTE']:
            risk = current_price - stop_loss
            reward = take_profit - current_price
        elif signal in ['VENTA', 'VENTA FUERTE']:
            risk = stop_loss - current_price
            reward = current_price - take_profit
        else:
            return np.nan
            
        if risk <= 0:
            return np.nan
            
        return reward / risk

    def process_ticker_granularity(self, ticker, granularity):
        """Procesa un ticker para una granularidad específica"""
        try:
            df = self.data_manager.load_and_preprocess(ticker, granularity)
            if df.empty:
                return None
                
            sr_levels = self.sr_analyzer.analyze_support_resistance(df, granularity)
            patterns = self.pattern_analyzer.detect_patterns(df, granularity)
            signal = self.signal_system.generate_quantitative_signals(df, sr_levels, patterns)
            
            # Para señales neutrales, establecer stop_loss y take_profit como NaN
            stop_loss = signal['stop_loss'] if signal['action'] not in ['NEUTRAL', 'SIN DATOS'] else np.nan
            take_profit = signal['take_profit'] if signal['action'] not in ['NEUTRAL', 'SIN DATOS'] else np.nan
            
            return {
                'ticker': ticker,
                'granularity': granularity,
                'senal': signal['action'],
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        except Exception as e:
            print(f"Error procesando {ticker} ({granularity}): {e}")
            return None
    
    def process_ticker(self, ticker):
        """Procesa un ticker para todas las granularidades"""
        results = {}
        results['ticker'] = ticker
        
        # Obtener precio actual de la granularidad diaria
        current_price = self.get_daily_price(ticker)
        results['precio_actual'] = current_price
        
        for granularity in ['diario', 'semanal', 'mensual']:
            result = self.process_ticker_granularity(ticker, granularity)
            
            if result:
                results[f'{granularity}'] = result['senal']
                results[f'stop_loss_{granularity}'] = result['stop_loss']
                results[f'take_profit_{granularity}'] = result['take_profit']
                
                # Calcular Risk:Reward
                rr = self.calculate_risk_reward(
                    result['senal'],
                    current_price,
                    result['stop_loss'],
                    result['take_profit']
                )
                results[f'risk_reward_{granularity}'] = rr
            else:
                results[f'{granularity}'] = 'SIN DATOS'
                results[f'stop_loss_{granularity}'] = np.nan
                results[f'take_profit_{granularity}'] = np.nan
                results[f'risk_reward_{granularity}'] = np.nan
        
        # Calcular media de Risk:Reward
        valid_rr = []
        for granularity in ['diario', 'semanal', 'mensual']:
            rr_val = results[f'risk_reward_{granularity}']
            if not pd.isna(rr_val):
                valid_rr.append(rr_val)
                
        results['risk_reward_media'] = np.mean(valid_rr) if valid_rr else np.nan
        
        return results
    
    def generate_technical_data(self):
        """Genera todos los datos técnicos usando procesamiento paralelo"""
        tickers = self.get_all_tickers()
        all_results = []
        
        # Usar ProcessPoolExecutor para paralelización
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_ticker = {
                executor.submit(self.process_ticker, ticker): ticker 
                for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    print(f"Procesado: {ticker}")
                except Exception as e:
                    print(f"Error procesando {ticker}: {e}")
        
        # Crear DataFrame con los resultados
        df = pd.DataFrame(all_results)
        
        # Reorganizar columnas - primero las columnas principales, luego los Risk:Reward al final
        main_columns = ['ticker', 'precio_actual']
        for granularity in ['diario', 'semanal', 'mensual']:
            main_columns.extend([
                granularity, 
                f'stop_loss_{granularity}', 
                f'take_profit_{granularity}'
            ])
        
        # Columnas de Risk:Reward al final
        rr_columns = []
        for granularity in ['diario', 'semanal', 'mensual']:
            rr_columns.append(f'risk_reward_{granularity}')
        rr_columns.append('risk_reward_media')
        
        # Combinar todas las columnas
        all_columns = main_columns + rr_columns
        
        df = df[all_columns]
        
        # Renombrar columnas al formato solicitado
        new_column_names = [
            'Ticker', 'Precio Actual', 
            'Diario', 'Stop Loss - Diario', 'Take Profit - Diario',
            'Semanal', 'Stop Loss - Semanal', 'Take Profit - Semanal', 
            'Mensual', 'Stop Loss - Mensual', 'Take Profit - Mensual',
            'Risk:Reward - Diario', 'Risk:Reward - Semanal', 'Risk:Reward - Mensual',
            'Risk:Reward - Media'
        ]
        
        df.columns = new_column_names
        
        # Guardar como parquet
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_parquet(self.output_path, index=False)
        print(f"Datos guardados en: {self.output_path}")
        
        return df

# Ejecutar la generación de datos
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Iniciando generación de datos técnicos: {start_time}")
    
    generator = TechnicalDataGenerator()
    result_df = generator.generate_technical_data()
    
    end_time = datetime.now()
    print(f"Proceso completado: {end_time}")
    print(f"Tiempo total: {end_time - start_time}")
    print(f"Tickers procesados: {len(result_df)}")