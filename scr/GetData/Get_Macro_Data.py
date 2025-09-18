import yfinance as yf
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.base.tsa_model import ValueWarning
import numpy as np


# Configuración FRED
USE_FRED = True
FRED_API_KEY = "a4fc0b370ee5f473f45757f776effd3e"

if USE_FRED:
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

# Definir project_root y ruta de salida
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_FILE = os.path.join(project_root, 'data', 'Macro_Data', 'Macro_Data.parquet')

def forecast_series_to_today(series, order=(1,1,1)) -> pd.Series:
    """Forecasts a time series to today's date using ARIMA"""
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:,0]
        else:
            raise ValueError("Se necesita 1 sola columna")
    
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    
    # Set frequency explicitly to avoid warnings
    if len(series) > 1:
        inferred_freq = pd.infer_freq(series.index[:min(10, len(series))])
        if inferred_freq:
            series = series.asfreq(inferred_freq)
        else:
            series = series.asfreq("D")
    else:
        series = series.asfreq("D")
    
    last_date = series.index.max()
    today = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))
    
    if last_date >= today:
        return series

    future_dates = pd.date_range(last_date + pd.Timedelta(1,"D"), today, freq="D")
    
    # Handle series with insufficient data for ARIMA
    if len(series.dropna()) < 10:
        # Use forward fill for short series
        fc = pd.Series([series.iloc[-1]] * len(future_dates), 
                      index=future_dates, name=series.name)
        return pd.concat([series, fc])
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            clean_series = series.dropna()
            model = ARIMA(
                clean_series,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(method_kwargs={"warn_convergence": False, "maxiter": 50})
        
        fc = pd.Series(model.forecast(len(future_dates)),
                      index=future_dates,
                      name=series.name)
    except:
        # Fallback to forward fill if ARIMA fails
        last_value = series.iloc[-1] if not pd.isna(series.iloc[-1]) else series.median()
        fc = pd.Series([last_value] * len(future_dates), 
                      index=future_dates, name=series.name)
    
    return pd.concat([series, fc])


def download_yahoo_macro(start="2000-01-01", end=None) -> pd.DataFrame:
    """Download macro data from Yahoo Finance"""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    tickers = {
        "VIX": "^VIX",           # Volatility Index
        "10Y_Treasury": "^TNX",   # 10-Year Treasury
        "interest_rate": "^IRX",  # 3-Month Treasury
        "gold_usd": "GC=F",       # Gold futures
        "eur_usd_rate": "EURUSD=X" # EUR/USD exchange rate
    }

    macro_data = {}
    for label, ticker in tickers.items():
        try:
            print(f"Descargando {label} ({ticker})...")
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            if df.empty or "Adj Close" not in df.columns:
                print(f"⚠️ No datos para {label} ({ticker}).")
                continue
            
            df = df[["Adj Close"]].rename(columns={"Adj Close": label})
            # Fix deprecated fillna methods
            df = df.ffill().bfill()
            macro_data[label] = df
        except Exception as e:
            print(f"❌ Error descargando {label}: {e}")
            continue

    if not macro_data:
        raise ValueError("No se descargaron datos válidos de Yahoo Finance.")

    df_all = pd.concat(macro_data.values(), axis=1, join="outer")
    df_all.columns = macro_data.keys()
    df_all.index.name = "date"
    return df_all.reset_index()


def calculate_growth_rates_before_resampling(data, label):
    """Calculate growth rates at original frequency before resampling to daily"""
    if data.empty:
        return data
    
    if label == "GDP":
        # Calculate quarterly growth rate (annualized)
        growth_rate = data.pct_change(4) * 100  # 4 quarters = 1 year
        return growth_rate
    
    elif label == "CPI":
        # Calculate annual inflation rate
        inflation_rate = data.pct_change(12) * 100  # 12 months = 1 year
        return inflation_rate
    
    return data


def add_fred_economic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive economic data from FRED"""
    print("Descargando datos económicos desde FRED...")
    
    fred_series = {
        # GDP and Growth
        "GDP": "GDP",                    # Gross Domestic Product
        
        # Inflation and Price Indices
        "CPI": "CPIAUCSL",              # Consumer Price Index
        "inflation": "CPALTT01USM657N",  # CPI Annual Rate
        
        # Interest Rates and Yield Curve
        "2Y_Treasury": "GS2",           # 2-Year Treasury
        "10Y_Treasury_FRED": "GS10",    # 10-Year Treasury (backup)
        
        # Corporate Bond Yields for Credit Spreads
        "AAA_Yield": "BAMLC0A1CAAA",   # AAA Corporate Bond Yield
        "BAA_Yield": "BAMLC0A4CBBB",   # BAA Corporate Bond Yield
        
        # Money Supply and Employment
        "m2_usd": "M2SL",              # M2 Money Supply
        "unemployment_rate": "UNRATE",  # Unemployment Rate
        
        # Additional Economic Indicators
        "fed_funds_rate": "FEDFUNDS",   # Federal Funds Rate
        "industrial_production": "INDPRO", # Industrial Production Index
        "consumer_sentiment": "UMCSENT"     # Consumer Sentiment
    }
    
    fred_data = {}
    base_date = df['date'].min() if 'date' in df.columns else "2000-01-01"
    
    for label, series_id in fred_series.items():
        try:
            print(f"  Descargando {label} ({series_id})...")
            data = fred.get_series(series_id, start=base_date)
            
            if data.empty:
                print(f"    ⚠️ No datos para {label}")
                continue
            
            # Handle different frequencies and calculate proper growth rates
            if label == "GDP":
                # Calculate quarterly growth BEFORE resampling
                gdp_growth = calculate_growth_rates_before_resampling(data, "GDP")
                gdp_growth.name = "GDP_Growth"
                
                # Resample both GDP and GDP_Growth to daily
                data_daily = data.resample("D").ffill()
                gdp_growth_daily = gdp_growth.resample("D").ffill()
                
                data_daily.name = label
                fred_data[label] = forecast_series_to_today(data_daily, order=(1,1,1))
                fred_data["GDP_Growth"] = forecast_series_to_today(gdp_growth_daily, order=(1,1,1))
                
            elif label == "CPI":
                # Calculate annual inflation BEFORE resampling
                cpi_inflation = calculate_growth_rates_before_resampling(data, "CPI")
                cpi_inflation.name = "Inflation_Rate"
                
                # Resample both CPI and Inflation_Rate to daily
                data_daily = data.resample("D").ffill()
                inflation_daily = cpi_inflation.resample("D").ffill()
                
                data_daily.name = label
                fred_data[label] = forecast_series_to_today(data_daily, order=(1,1,1))
                fred_data["Inflation_Rate"] = forecast_series_to_today(inflation_daily, order=(1,1,1))
                
            elif label in ["inflation", "unemployment_rate", "m2_usd", "consumer_sentiment"]:
                # Monthly data - forward fill to daily
                data_daily = data.resample("D").ffill()
                if label in ["inflation", "unemployment_rate"]:
                    data_daily = data_daily / 100  # Convert percentages
                data_daily.name = label
                fred_data[label] = forecast_series_to_today(data_daily, order=(1,1,1))
                
            else:
                # Daily or business daily data
                data_daily = data.resample("D").ffill()
                if label in ["2Y_Treasury", "10Y_Treasury_FRED", "AAA_Yield", "BAA_Yield", "fed_funds_rate"]:
                    data_daily = data_daily / 100  # Convert percentages to decimals
                data_daily.name = label
                fred_data[label] = forecast_series_to_today(data_daily, order=(1,1,1))
            
        except Exception as e:
            print(f"    ❌ Error con {label}: {e}")
            continue
    
    if not fred_data:
        print("⚠️ No se pudieron descargar datos de FRED")
        return df
    
    # Merge with existing data
    df_with_date_index = df.set_index("date")
    
    for label, series in fred_data.items():
        df_with_date_index = df_with_date_index.join(series, how="left")
    
    return df_with_date_index.reset_index()


def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing data using various strategies"""
    print("Rellenando datos faltantes...")
    
    # Fix deprecated fillna methods
    df = df.ffill().bfill()
    
    # Handle remaining missing values with interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    
    # Fill any remaining missing values with column medians
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Rellenado {col} con mediana: {median_val:.4f}")
    
    return df


def create_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived indicators that will be needed for macro calculations"""
    print("Creando indicadores derivados...")
    
    # Use backup Treasury data if primary is missing
    if '10Y_Treasury' not in df.columns and '10Y_Treasury_FRED' in df.columns:
        df['10Y_Treasury'] = df['10Y_Treasury_FRED']
    
    # Create VIX moving average
    if 'VIX' in df.columns:
        df['VIX_MA'] = df['VIX'].rolling(window=20, min_periods=1).mean()
    
    # Note: GDP_Growth and Inflation_Rate are now calculated in add_fred_economic_data()
    # with proper frequency handling
    
    # Create Yield Curve (10Y - 2Y spread)
    if '10Y_Treasury' in df.columns and '2Y_Treasury' in df.columns:
        df['Yield_Curve'] = df['10Y_Treasury'] - df['2Y_Treasury']
    
    # Create Credit Spread (BAA - AAA)
    if 'BAA_Yield' in df.columns and 'AAA_Yield' in df.columns:
        df['Credit_Spread'] = df['BAA_Yield'] - df['AAA_Yield']
    
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the final dataset"""
    print("Validando datos finales...")
    
    # Remove any rows where date is invalid
    df = df.dropna(subset=['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])
    
    # Basic data quality checks
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Remove extreme outliers (beyond 5 standard deviations)
        if col in ['GDP_Growth', 'Inflation_Rate']:
            # More lenient for growth rates (can be volatile)
            threshold = 10  # Standard deviations
        else:
            threshold = 5
            
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            outlier_mask = np.abs(df[col] - mean_val) > threshold * std_val
            if outlier_mask.any():
                print(f"  Removiendo {outlier_mask.sum()} outliers extremos de {col}")
                df.loc[outlier_mask, col] = np.nan
    
    # Fill any new missing values created by outlier removal
    df = fill_missing_data(df)
    
    print(f"Datos validados: {len(df)} filas, {len(df.columns)-1} indicadores")
    print(f"Período: {df['date'].min()} a {df['date'].max()}")
    
    return df


def main():
    """Main execution function"""
    print("🚀 Iniciando descarga de datos macroeconómicos...")
    
    # Suppress statsmodels warnings globally
    warnings.filterwarnings("ignore", category=ValueWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="No supported index is available")
    warnings.filterwarnings("ignore", message=".*supported index.*")
    
    try:
        # Step 1: Download Yahoo Finance data
        print("\n📊 Paso 1: Descargando datos de Yahoo Finance...")
        df_macro = download_yahoo_macro()
        
        # Step 2: Add FRED data if available
        if USE_FRED and FRED_API_KEY != "Insertar aquí":
            print("\n🏦 Paso 2: Añadiendo datos de FRED...")
            df_macro = add_fred_economic_data(df_macro)
        else:
            print("\n⚠️ Paso 2: Saltando FRED (API key no configurada)")
            # Add empty columns for missing FRED data
            fred_columns = ["GDP", "CPI", "2Y_Treasury", "AAA_Yield", "BAA_Yield", 
                          "m2_usd", "unemployment_rate", "fed_funds_rate", 
                          "industrial_production", "consumer_sentiment", 
                          "GDP_Growth", "Inflation_Rate"]
            for col in fred_columns:
                if col not in df_macro.columns:
                    df_macro[col] = np.nan
        
        # Step 3: Fill missing data
        print("\n🔧 Paso 3: Procesando datos faltantes...")
        df_macro = fill_missing_data(df_macro)
        
        # Step 4: Create derived indicators
        print("\n📈 Paso 4: Creando indicadores derivados...")
        df_macro = create_derived_indicators(df_macro)
        
        # Step 5: Validate and clean
        print("\n✅ Paso 5: Validando datos...")
        df_macro = validate_data(df_macro)
        
        # Step 6: Save to parquet file
        print(f"\n💾 Paso 6: Guardando en formato parquet: {OUTPUT_FILE}...")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_macro.to_parquet(OUTPUT_FILE, index=False)
        
        # Summary
        print(f"\n🎉 ¡Completado exitosamente!")
        print(f"Archivo parquet guardado: {OUTPUT_FILE}")
        print(f"Registros: {len(df_macro)}")
        print(f"Columnas: {list(df_macro.columns)}")
        
        # Show sample of recent data with actual values
        print(f"\n📋 Muestra de datos recientes:")
        columns_to_show = ['date', 'VIX', 'GDP_Growth', 'Inflation_Rate', 'Yield_Curve', 'Credit_Spread']
        available_cols = [col for col in columns_to_show if col in df_macro.columns]
        recent_data = df_macro.tail(3)[available_cols].round(4)
        print(recent_data.to_string(index=False))
        
        # Show data ranges for key indicators
        print(f"\n📊 Rangos de datos clave:")
        key_indicators = ['GDP_Growth', 'Inflation_Rate', 'Yield_Curve', 'VIX', 'Credit_Spread']
        for col in key_indicators:
            if col in df_macro.columns:
                non_null_data = df_macro[col].dropna()
                if len(non_null_data) > 0:
                    print(f"  {col}: min={non_null_data.min():.4f}, max={non_null_data.max():.4f}, "
                          f"mean={non_null_data.mean():.4f}")
        
    except Exception as e:
        print(f"\n❌ Error en el proceso principal: {e}")
        raise


if __name__ == "__main__":
    main()