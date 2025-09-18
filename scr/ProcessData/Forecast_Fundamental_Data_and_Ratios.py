import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
import gc
import os

warnings.filterwarnings('ignore')

# Rutas relativas - desde scr/ProcessData/ hacia data/
# Obtenemos el directorio del script actual y navegamos hacia data/
SCRIPT_DIR = Path(__file__).parent  # scr/ProcessData/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # project_root/
ROOT = PROJECT_ROOT / "data"

SOURCE_ROOT = ROOT / "Fundamental_Data_and_Ratios"
PREDICTIONS_ROOT = ROOT / "Forecast_Fundamental_Data_and_Ratios"

# Subcarpetas
dirs = {
    "Anual": SOURCE_ROOT / "Anual",
    "Trimestral": SOURCE_ROOT / "Trimestral",
}

# Crear directorios de destino
for sub in dirs:
    (PREDICTIONS_ROOT / sub).mkdir(parents=True, exist_ok=True)

# Configuración conservadora para Windows
MAX_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1


def predict_metric_seasonal_quarterly(series, periods=12):
    """
    Predicción para datos trimestrales con manejo de estacionalidad
    12 trimestres = 3 años completos
    """
    clean_series = series.dropna()
    if len(clean_series) < 4:
        return [np.nan] * periods
    
    try:
        values = clean_series.values
        n = len(values)
        
        if n < 8:  # Menos de 2 años
            last_val = values[-1] if len(values) > 0 else 0
            growth = 0.02 if last_val > 0 else 0
            return [last_val * (1 + growth)**((i+1)/4) for i in range(periods)]
        
        # Análisis estacional básico
        # Calcular promedios por posición trimestral
        quarterly_ratios = np.ones(4)
        
        if n >= 8:  # Al menos 2 años de datos
            # Obtener últimos 2-3 años para análisis estacional
            recent_data = values[-16:] if n >= 16 else values[-12:] if n >= 12 else values[-8:]
            
            # Calcular ratios estacionales
            quarters = [[] for _ in range(4)]
            for i, val in enumerate(recent_data):
                if val > 0:
                    quarter_idx = i % 4
                    quarters[quarter_idx].append(val)
            
            # Ratios vs promedio
            overall_mean = np.mean([v for q in quarters for v in q])
            if overall_mean > 0:
                for q in range(4):
                    if quarters[q]:
                        quarterly_ratios[q] = np.mean(quarters[q]) / overall_mean
        
        # Tendencia general usando regresión
        X = np.arange(n).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, values)
        
        # Crecimiento tendencial por trimestre
        slope = model.coef_[0] if hasattr(model, 'coef_') else 0
        base_growth = slope / np.mean(values) if np.mean(values) > 0 else 0.02
        base_growth = np.clip(base_growth, -0.05, 0.08)  # Limitar crecimiento trimestral
        
        # Generar predicciones
        predictions = []
        last_value = values[-1]
        
        for i in range(periods):
            # Trimestre correspondiente
            quarter_idx = (n + i) % 4
            
            # Crecimiento base (tendencial)
            periods_ahead = (i + 1)
            trend_factor = 1 + (base_growth * periods_ahead)
            
            # Aplicar patrón estacional
            seasonal_factor = quarterly_ratios[quarter_idx]
            
            # Predicción combinada
            prediction = last_value * trend_factor * seasonal_factor
            
            # Validaciones de seguridad
            if prediction <= 0 and last_value > 0:
                prediction = last_value * (0.98 ** (i + 1))
            elif prediction > last_value * 3:  # No más de 3x
                prediction = last_value * (1.05 ** periods_ahead)
            
            predictions.append(prediction)
        
        return predictions
        
    except Exception:
        # Fallback conservador
        last_val = values[-1] if len(values) > 0 else 0
        return [last_val * (1.02 ** ((i+1)/4)) for i in range(periods)]


def predict_metric_annual_safe(series, periods=3):
    """Predicción segura para datos anuales"""
    clean_series = series.dropna()
    if len(clean_series) < 3:
        return [np.nan] * periods
    
    try:
        values = clean_series.values
        n = len(values)
        last_value = values[-1]
        
        # Crecimiento histórico
        if n >= 2:
            growth_rates = []
            for i in range(1, min(n, 6)):  # Últimos 5 crecimientos max
                if values[-i-1] > 0:
                    growth = (values[-i] / values[-i-1]) - 1
                    if -0.3 < growth < 0.5:  # Filtrar extremos
                        growth_rates.append(growth)
            
            avg_growth = np.median(growth_rates) if growth_rates else 0.02
            avg_growth = np.clip(avg_growth, -0.15, 0.25)
        else:
            avg_growth = 0.02
        
        # Regresión lineal si hay suficientes datos
        use_regression = False
        if n >= 5:
            X = np.arange(n).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, values)
            r_squared = model.score(X, values)
            
            if r_squared > 0.6:
                use_regression = True
        
        # Generar predicciones
        predictions = []
        for i in range(periods):
            if use_regression:
                pred = model.predict([[n + i]])[0]
                # Validar que no sea demasiado extremo
                if abs(pred - last_value) > abs(last_value * 0.4):
                    pred = last_value * ((1 + avg_growth) ** (i + 1))
            else:
                pred = last_value * ((1 + avg_growth) ** (i + 1))
            
            # Validaciones finales
            if last_value > 0 and pred < 0:
                pred = last_value * (0.95 ** (i + 1))
            
            predictions.append(pred)
        
        return predictions
        
    except Exception:
        last_val = values[-1] if len(values) > 0 else 0
        return [last_val * (1.02 ** i) for i in range(1, periods + 1)]


def create_predictions_safe(source_file: Path, dest_file: Path, period_type: str):
    """Versión segura y estable de creación de predicciones con preservación de estructura"""
    ticker = source_file.stem
    
    try:
        # Carga segura
        df = pd.read_parquet(source_file, engine="pyarrow")
        if df.empty:
            return f"{ticker} - Empty file"
        
        # Preparación
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        if len(df) == 0:
            return f"{ticker} - No valid dates"
        
        last_date = df['Date'].iloc[-1]
        
        # Identificar todas las columnas para mantener estructura
        excluded_cols = ['Date', 'ticker']
        
        # Columnas numéricas con datos para predicción
        numeric_columns_with_data = []
        # Todas las demás columnas (para mantener estructura)
        all_other_columns = []
        
        for col in df.columns:
            if col not in excluded_cols:
                all_other_columns.append(col)
                # Solo predecir si es numérica Y tiene datos
                if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
                    numeric_columns_with_data.append(col)
        
        # Información de debug
        empty_columns = [col for col in all_other_columns if col not in numeric_columns_with_data]
        if empty_columns:
            # No es un error, solo información
            pass
        
        # Determinar períodos y función de predicción
        if period_type == 'annual':
            periods = 3
            predict_func = predict_metric_annual_safe
            days_increment = 365
        else:  # quarterly - 12 trimestres para 3 años
            periods = 12
            predict_func = predict_metric_seasonal_quarterly
            days_increment = 90
        
        # Generar fechas futuras
        future_dates = []
        for i in range(1, periods + 1):
            future_date = last_date + timedelta(days=days_increment * i)
            future_dates.append(future_date)
        
        # Crear estructura de datos de predicciones
        predictions_data = {
            'Date': future_dates,
            'ticker': [ticker] * periods
        }
        
        # Calcular predicciones para todas las columnas (manteniendo estructura)
        successful_predictions = 0
        for col in all_other_columns:
            if col in numeric_columns_with_data:
                # Columna con datos - aplicar predicción
                try:
                    predictions = predict_func(df[col], periods)
                    predictions_data[col] = predictions
                    successful_predictions += 1
                except Exception as e:
                    predictions_data[col] = [np.nan] * periods
            else:
                # Columna vacía o no numérica - mantener como NaN
                predictions_data[col] = [np.nan] * periods
        
        # Crear DataFrame
        pred_df = pd.DataFrame(predictions_data)
        
        # Metadata básica
        pred_df['prediction_type'] = period_type
        pred_df['base_date'] = last_date
        pred_df['created_at'] = datetime.now()
        
        if period_type == 'quarterly':
            pred_df['quarter_ahead'] = list(range(1, 13))
            pred_df['year_ahead'] = [(i-1)//4 + 1 for i in range(1, 13)]
        else:
            pred_df['year_ahead'] = list(range(1, 4))
        
        # Guardar
        pred_df.to_parquet(dest_file, engine="pyarrow", index=False)
        
        # Limpieza
        del df, pred_df
        gc.collect()
        
        return f"{ticker} - Success ({successful_predictions}/{len(all_other_columns)} metrics predicted, {periods} periods)"
        
    except Exception as e:
        return f"{ticker} - Error: {str(e)[:60]}"


def process_directory_stable(source_dir: Path, dest_dir: Path, period_type: str):
    """Procesamiento estable con ThreadPoolExecutor"""
    if not source_dir.exists():
        return 0, 0
    
    parquet_files = list(source_dir.glob("*.parquet"))
    if not parquet_files:
        return 0, 0
    
    total_files = len(parquet_files)
    period_desc = "3 years" if period_type == "annual" else "3 years (12 quarters)"
    
    print(f"Processing {total_files} files with {MAX_WORKERS} threads...")
    print(f"Predicting: {period_desc}")
    print(f"Output directory: {dest_dir}")
    
    successful = 0
    errors = 0
    
    # ThreadPoolExecutor es más estable en Windows
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Enviar trabajos
        future_to_file = {}
        for source_file in parquet_files:
            dest_file = dest_dir / f"{source_file.stem}.parquet"
            future = executor.submit(create_predictions_safe, source_file, dest_file, period_type)
            future_to_file[future] = source_file.stem
        
        # Procesar resultados
        completed = 0
        for future in as_completed(future_to_file):
            ticker = future_to_file[future]
            completed += 1
            
            try:
                result = future.result()
                if "Success" in result:
                    successful += 1
                    if completed % 100 == 0 or successful % 50 == 0:
                        print(f"   Progress: {completed}/{total_files} - Last: {result}")
                else:
                    errors += 1
                    if "Error" in result:
                        print(f"   {result}")
                        
            except Exception as e:
                errors += 1
                print(f"   {ticker} - Unexpected error: {str(e)[:50]}")
            
            # Progress report
            if completed % 500 == 0:
                print(f"   CHECKPOINT: {completed}/{total_files} files processed")
                print(f"   Success: {successful}, Errors: {errors}")
                gc.collect()  # Limpieza periódica
    
    return successful, errors


def main():
    """Función principal estable"""
    print("STABLE FINANCIAL PREDICTIONS GENERATOR")
    print(f"Windows-Optimized | {MAX_WORKERS} Threads | Seasonal Forecasting")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Root: {ROOT}")
    print("=" * 80)
    
    start_time = datetime.now()
    total_successful = 0
    total_errors = 0
    
    for period_name, source_dir in dirs.items():
        if not source_dir.exists():
            print(f"Directory not found: {source_dir}")
            continue
            
        dest_dir = PREDICTIONS_ROOT / period_name
        dest_dir.mkdir(exist_ok=True)
        
        print(f"\n{period_name.upper()} DATA PROCESSING")
        print("-" * 40)
        
        period_type = 'annual' if period_name == 'Anual' else 'quarterly'
        
        try:
            successful, errors = process_directory_stable(source_dir, dest_dir, period_type)
            total_successful += successful
            total_errors += errors
            
            print(f"COMPLETED {period_name}: {successful} successful, {errors} errors")
            
        except Exception as e:
            print(f"DIRECTORY ERROR {period_name}: {str(e)}")
            total_errors += 1
    
    end_time = datetime.now()
    duration = end_time - start_time
    total_files = total_successful + total_errors
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Duration: {duration}")
    print(f"Files Processed: {total_files}")
    print(f"Successful: {total_successful}")
    print(f"Errors: {total_errors}")
    
    if total_files > 0:
        print(f"Success Rate: {(total_successful/total_files)*100:.1f}%")
        print(f"Speed: {total_files/duration.total_seconds():.1f} files/second")
    
    if total_successful > 0:
        print(f"\nFEATURES:")
        print(f"- Annual: 3-year forecasts with trend analysis")
        print(f"- Quarterly: 12-quarter forecasts with seasonal patterns")
        print(f"- Files saved to: {PREDICTIONS_ROOT}")
        print(f"- Conservative methodology for stability")
        print(f"- Complete structure preservation (all columns maintained)")


if __name__ == "__main__":
    main()