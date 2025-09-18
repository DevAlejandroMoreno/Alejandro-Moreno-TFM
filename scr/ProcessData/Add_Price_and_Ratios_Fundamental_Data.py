import pandas as pd
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# Rutas base usando rutas relativas
ROOT = Path(__file__).parent.parent.parent / "data"
MACROROOT = ROOT / "Fundamental_Data"
MARKETROOT = ROOT / "Historical_Prices"
DESTROOT = ROOT / "Fundamental_Data_and_Ratios"
MACRODATAPATH = ROOT / "Macro_Data" / "Macro_Data.parquet"

# Subcarpetas a procesar
dirs = {
    "Anual": MACROROOT / "Anual",
    "Trimestral": MACROROOT / "Trimestral",
}

# Aseguramos directorios de destino
for sub in dirs:
    (DESTROOT / sub).mkdir(parents=True, exist_ok=True)


def check_macro_data_availability():
    """Verifica si los datos macro están disponibles para el proceso principal"""
    try:
        if not MACRODATAPATH.exists():
            print(f"⚠️  Macro data file not found: {MACRODATAPATH}")
            return False
            
        print(f"📊 Loading macro data from: {MACRODATAPATH}")
        df_macro = pd.read_parquet(MACRODATAPATH, engine="pyarrow")
        
        print(f"✅ Loaded {len(df_macro)} macro records from {df_macro['date'].min()} to {df_macro['date'].max()}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR loading macro data: {str(e)}")
        return False


def load_macro_data_local():
    """Carga los datos macro localmente para cada worker process"""
    try:
        if not MACRODATAPATH.exists():
            return None
            
        df_macro = pd.read_parquet(MACRODATAPATH, engine="pyarrow")
        df_macro["date"] = pd.to_datetime(df_macro["date"])
        df_macro = df_macro.sort_values("date").reset_index(drop=True)
        
        # Añadir columnas derivadas para agrupación
        df_macro["year"] = df_macro["date"].dt.year
        df_macro["quarter"] = df_macro["date"].dt.quarter
        df_macro["year_quarter"] = df_macro["year"].astype(str) + "_Q" + df_macro["quarter"].astype(str)
        
        return df_macro
        
    except Exception as e:
        return None


def aggregate_macro_data(df_macro: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Agrega los datos macroeconómicos según la frecuencia especificada
    frequency: 'annual' o 'quarterly'
    """
    if df_macro is None or df_macro.empty:
        return pd.DataFrame()
    
    # Definir columnas para diferentes tipos de agregación
    # Estas métricas se promedian (ratios, tasas, índices)
    avg_columns = [
        'VIX', '10Y_Treasury', 'interest_rate', 'gold_usd', 'eur_usd_rate',
        'CPI', 'Inflation_Rate', 'inflation', '2Y_Treasury', '10Y_Treasury_FRED',
        'AAA_Yield', 'BAA_Yield', 'unemployment_rate', 'fed_funds_rate',
        'consumer_sentiment', 'VIX_MA', 'Yield_Curve', 'Credit_Spread'
    ]
    
    # Estas métricas toman el último valor del período (stocks/niveles)
    last_columns = [
        'GDP', 'GDP_Growth', 'm2_usd', 'industrial_production'
    ]
    
    # Verificar qué columnas existen realmente en los datos
    available_avg_cols = [col for col in avg_columns if col in df_macro.columns]
    available_last_cols = [col for col in last_columns if col in df_macro.columns]
    
    if frequency == 'annual':
        group_col = 'year'
        date_col = 'year'
    else:  # quarterly
        group_col = 'year_quarter'
        date_col = 'year_quarter'
    
    # Preparar diccionario de agregación
    agg_dict = {}
    
    # Añadir promedios
    for col in available_avg_cols:
        agg_dict[col] = 'mean'
    
    # Añadir últimos valores
    for col in available_last_cols:
        agg_dict[col] = 'last'
    
    # También queremos la fecha más reciente del grupo para referencia
    agg_dict['date'] = 'last'
    
    if not agg_dict:
        return pd.DataFrame()
    
    # Realizar la agregación
    df_agg = df_macro.groupby(group_col).agg(agg_dict).reset_index()
    
    # Renombrar la columna de agrupación
    df_agg.rename(columns={group_col: date_col}, inplace=True)
    
    return df_agg


def merge_macro_with_fundamental(df_fundamental: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Hace merge de los datos fundamentales con los datos macro agregados
    """
    # Cargar datos macro localmente en cada worker process
    macro_data = load_macro_data_local()
    
    if macro_data is None or macro_data.empty:
        return df_fundamental
    
    # Agregar datos macro según frecuencia
    df_macro_agg = aggregate_macro_data(macro_data, frequency)
    
    if df_macro_agg.empty:
        return df_fundamental
    
    # Preparar columna de merge en datos fundamentales
    df_fundamental = df_fundamental.copy()
    
    if frequency == 'annual':
        df_fundamental['year'] = df_fundamental['Date'].dt.year
        merge_cols = ['year']
        
    else:  # quarterly
        df_fundamental['year'] = df_fundamental['Date'].dt.year
        df_fundamental['quarter'] = df_fundamental['Date'].dt.quarter
        df_fundamental['year_quarter'] = df_fundamental['year'].astype(str) + "_Q" + df_fundamental['quarter'].astype(str)
        merge_cols = ['year_quarter']
    
    # Hacer el merge
    df_merged = df_fundamental.merge(
        df_macro_agg.drop(columns=['date'], errors='ignore'),  # Evitar duplicar columna date
        on=merge_cols,
        how='left'
    )
    
    # Limpiar columnas auxiliares
    cols_to_drop = ['year', 'quarter', 'year_quarter']
    df_merged = df_merged.drop(columns=[col for col in cols_to_drop if col in df_merged.columns], errors='ignore')
    
    # Contar cuántas filas tienen datos macro (para debugging)
    macro_cols = [col for col in df_macro_agg.columns if col not in merge_cols + ['date']]
    macro_data_available = len(macro_cols) > 0 and not df_merged[macro_cols[0]].isna().all() if macro_cols else False
    
    return df_merged


def calculate_professional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todas las métricas profesionales de inversión de una vez
    para optimizar el rendimiento del dashboard
    """
    df = df.copy()
    
    # =============================================
    # 1. MÉTRICAS FUNDAMENTALES BÁSICAS
    # =============================================
    
    # Working Capital (si no existe)
    if 'Working Capital' not in df.columns:
        if 'Total Current Assets' in df.columns and 'Total Current Liabilities' in df.columns:
            df['Working Capital'] = df['Total Current Assets'] - df['Total Current Liabilities']
    
    # Free Cash Flow (versión mejorada)
    if 'Free Cash Flow' not in df.columns:
        if 'Cash Flow From Operating Activities' in df.columns:
            df['Free Cash Flow'] = df['Cash Flow From Operating Activities'].fillna(0)
            
            # Restar CapEx si está disponible
            if 'Net Change In Property, Plant, And Equipment' in df.columns:
                capex = df['Net Change In Property, Plant, And Equipment'].abs()
                df['Free Cash Flow'] = df['Free Cash Flow'] - capex
    
    # Net Debt
    if 'Net Debt' not in df.columns:
        if 'Long Term Debt' in df.columns and 'Cash On Hand' in df.columns:
            df['Net Debt'] = df['Long Term Debt'].fillna(0) - df['Cash On Hand'].fillna(0)
    
    # =============================================
    # 2. RATIOS DE DEUDA AVANZADOS
    # =============================================
    
    # Debt/EBITDA
    if 'Debt/EBITDA' not in df.columns:
        if 'Long Term Debt' in df.columns and 'EBITDA' in df.columns:
            df['Debt/EBITDA'] = df['Long Term Debt'] / df['EBITDA'].replace(0, np.nan)
    
    # Debt/Assets
    if 'Debt/Assets' not in df.columns:
        if 'Long Term Debt' in df.columns and 'Total Assets' in df.columns:
            df['Debt/Assets'] = (df['Long Term Debt'] / df['Total Assets'].replace(0, np.nan)) * 100
    
    # Interest Coverage Ratio
    if 'Interest Coverage' not in df.columns:
        if 'Operating Income' in df.columns and 'Total Non-Operating Income/Expense' in df.columns:
            interest_expense = df['Total Non-Operating Income/Expense'].abs()
            df['Interest Coverage'] = df['Operating Income'] / interest_expense.replace(0, np.nan)
    
    # =============================================
    # 3. RATIOS DE LIQUIDEZ ADICIONALES
    # =============================================
    
    # Quick Ratio (Acid Test)
    if 'Quick Ratio' not in df.columns:
        if all(col in df.columns for col in ['Total Current Assets', 'Inventory', 'Total Current Liabilities']):
            quick_assets = df['Total Current Assets'] - df['Inventory'].fillna(0)
            df['Quick Ratio'] = quick_assets / df['Total Current Liabilities'].replace(0, np.nan)
    
    # Cash Ratio
    if 'Cash Ratio' not in df.columns:
        if 'Cash On Hand' in df.columns and 'Total Current Liabilities' in df.columns:
            df['Cash Ratio'] = df['Cash On Hand'] / df['Total Current Liabilities'].replace(0, np.nan)
    
    # =============================================
    # 4. MÉTRICAS PER SHARE
    # =============================================
    
    shares_col = 'Shares Outstanding' if 'Shares Outstanding' in df.columns else 'Basic Shares Outstanding'
    
    if shares_col in df.columns:
        # Revenue per Share
        if 'Revenue/Share' not in df.columns and 'Revenue' in df.columns:
            df['Revenue/Share'] = df['Revenue'] / df[shares_col].replace(0, np.nan)
        
        # Cash per Share  
        if 'Cash/Share' not in df.columns and 'Cash On Hand' in df.columns:
            df['Cash/Share'] = df['Cash On Hand'] / df[shares_col].replace(0, np.nan)
        
        # Free Cash Flow per Share
        if 'Free Cash Flow Per Share' not in df.columns and 'Free Cash Flow' in df.columns:
            df['Free Cash Flow Per Share'] = df['Free Cash Flow'] / df[shares_col].replace(0, np.nan)
        
        # Assets per Share
        if 'Assets/Share' not in df.columns and 'Total Assets' in df.columns:
            df['Assets/Share'] = df['Total Assets'] / df[shares_col].replace(0, np.nan)
    
    # =============================================
    # 5. RATIOS DE VALORACIÓN
    # =============================================
    
    if 'Close' in df.columns and 'MarketCap' in df.columns:
        # P/E Ratio
        if 'P/E' not in df.columns and 'EPS - Earnings Per Share' in df.columns:
            df['P/E'] = df['Close'] / df['EPS - Earnings Per Share'].replace(0, np.nan)
        
        # P/B Ratio
        if 'P/B' not in df.columns and 'Book Value Per Share' in df.columns:
            df['P/B'] = df['Close'] / df['Book Value Per Share'].replace(0, np.nan)
        elif 'P/B' not in df.columns and 'Share Holder Equity' in df.columns and shares_col in df.columns:
            book_value_per_share = df['Share Holder Equity'] / df[shares_col].replace(0, np.nan)
            df['P/B'] = df['Close'] / book_value_per_share.replace(0, np.nan)
        
        # P/S Ratio
        if 'P/S' not in df.columns and 'Revenue' in df.columns:
            df['P/S'] = df['MarketCap'] / df['Revenue'].replace(0, np.nan)
        
        # P/CF Ratio
        if 'P/CF' not in df.columns and 'Cash Flow From Operating Activities' in df.columns:
            df['P/CF'] = df['MarketCap'] / df['Cash Flow From Operating Activities'].replace(0, np.nan)
        
        # Enterprise Value
        if 'Enterprise Value' not in df.columns:
            ev = df['MarketCap'].fillna(0)
            if 'Long Term Debt' in df.columns:
                ev = ev + df['Long Term Debt'].fillna(0)
            if 'Cash On Hand' in df.columns:
                ev = ev - df['Cash On Hand'].fillna(0)
            df['Enterprise Value'] = ev
        
        # EV/EBITDA
        if 'EV/EBITDA' not in df.columns and 'EBITDA' in df.columns and 'Enterprise Value' in df.columns:
            df['EV/EBITDA'] = df['Enterprise Value'] / df['EBITDA'].replace(0, np.nan)
        
        # EV/Sales
        if 'EV/Sales' not in df.columns and 'Revenue' in df.columns and 'Enterprise Value' in df.columns:
            df['EV/Sales'] = df['Enterprise Value'] / df['Revenue'].replace(0, np.nan)
    
    # =============================================
    # 6. RATIOS DE RENTABILIDAD ADICIONALES
    # =============================================
    
    # Return on Invested Capital (ROIC)
    if 'ROIC' not in df.columns:
        if all(col in df.columns for col in ['EBIT', 'Income Taxes', 'Total Assets', 'Total Current Liabilities']):
            nopat = df['EBIT'] - df['Income Taxes'].fillna(0)  # Net Operating Profit After Tax
            invested_capital = df['Total Assets'] - df['Total Current Liabilities']
            df['ROIC'] = (nopat / invested_capital.replace(0, np.nan)) * 100
    
    # =============================================
    # 7. FEATURES DE MOMENTUM (CRECIMIENTO)
    # =============================================
    
    # Ordenar por fecha para cálculos temporales
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    momentum_metrics = ['Revenue', 'Net Income', 'EBITDA', 'Free Cash Flow', 'Operating Income']
    
    for metric in momentum_metrics:
        if metric in df.columns:
            # Crecimiento YoY
            growth_col = f'{metric}_Growth_1Y'
            if growth_col not in df.columns:
                df[growth_col] = df[metric].pct_change(1) * 100
                # Limpiar valores extremos
                df[growth_col] = np.clip(df[growth_col], -200, 500)  # Limitar entre -200% y +500%
            
            # CAGR 3 años (si hay suficientes datos)
            cagr_col = f'{metric}_CAGR_3Y'
            if cagr_col not in df.columns and len(df) >= 4:
                df[cagr_col] = ((df[metric] / df[metric].shift(3)) ** (1/3) - 1) * 100
                df[cagr_col] = np.clip(df[cagr_col], -50, 200)  # Limitar CAGR
    
    # =============================================
    # 8. FEATURES DE CALIDAD (PIOTROSKI & ALTMAN)
    # =============================================
    
    # Piotroski F-Score simplificado
    if 'Piotroski_Score' not in df.columns:
        f_score = pd.Series(0, index=df.index)
        
        # ROA positivo
        if 'ROA - Return On Assets' in df.columns:
            f_score += (df['ROA - Return On Assets'] > 0).astype(int)
        
        # Free Cash Flow positivo
        if 'Free Cash Flow' in df.columns:
            f_score += (df['Free Cash Flow'] > 0).astype(int)
        
        # ROA mejorando
        if 'ROA - Return On Assets' in df.columns and len(df) > 1:
            roa_improving = df['ROA - Return On Assets'] > df['ROA - Return On Assets'].shift(1)
            f_score += roa_improving.fillna(0).astype(int)
        
        # Debt/Equity mejorando (disminuyendo)
        if 'Debt/Equity Ratio' in df.columns and len(df) > 1:
            debt_improving = df['Debt/Equity Ratio'] < df['Debt/Equity Ratio'].shift(1)
            f_score += debt_improving.fillna(0).astype(int)
        
        # Current Ratio mejorando
        if 'Current Ratio' in df.columns and len(df) > 1:
            current_improving = df['Current Ratio'] > df['Current Ratio'].shift(1)
            f_score += current_improving.fillna(0).astype(int)
        
        # Shares Outstanding no aumentando (buybacks)
        if shares_col in df.columns and len(df) > 1:
            shares_not_increasing = df[shares_col] <= df[shares_col].shift(1)
            f_score += shares_not_increasing.fillna(0).astype(int)
        
        df['Piotroski_Score'] = np.clip(f_score, 0, 6)  # Score de 0-6
    
    # Altman Z-Score
    if 'Altman_Z_Score' not in df.columns:
        if all(col in df.columns for col in ['Working Capital', 'Total Assets', 'Share Holder Equity', 'EBIT', 'MarketCap', 'Revenue']):
            # Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
            # A = Working Capital / Total Assets
            # B = Retained Earnings / Total Assets  
            # C = EBIT / Total Assets
            # D = Market Cap / Total Liabilities
            # E = Sales / Total Assets
            
            a = df['Working Capital'] / df['Total Assets'].replace(0, np.nan)
            b = df['Share Holder Equity'] / df['Total Assets'].replace(0, np.nan)  # Proxy for retained earnings
            c = df['EBIT'] / df['Total Assets'].replace(0, np.nan)
            d = df['MarketCap'] / df['Total Liabilities'].replace(0, np.nan)
            e = df['Revenue'] / df['Total Assets'].replace(0, np.nan)
            
            df['Altman_Z_Score'] = (1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e).fillna(np.nan)
    
    # =============================================
    # 9. MÉTRICAS DE EFICIENCIA ADICIONALES
    # =============================================
    
    # Days Sales Outstanding (DSO) - mejorado
    if 'Days Sales Outstanding' not in df.columns:
        if 'Days Sales In Receivables' in df.columns:
            df['Days Sales Outstanding'] = df['Days Sales In Receivables']
        elif 'Receivables' in df.columns and 'Revenue' in df.columns:
            df['Days Sales Outstanding'] = (df['Receivables'] / df['Revenue'].replace(0, np.nan)) * 365
    
    # Days Payable Outstanding (DPO)
    if 'Days Payable Outstanding' not in df.columns:
        if 'Accounts Payable' in df.columns and 'Cost Of Goods Sold' in df.columns:
            df['Days Payable Outstanding'] = (df['Accounts Payable'] / df['Cost Of Goods Sold'].replace(0, np.nan)) * 365
    
    # Cash Conversion Cycle
    if 'Cash Conversion Cycle' not in df.columns:
        components = ['Days Sales In Receivables', 'Days Sales Outstanding']
        dso_col = None
        for comp in components:
            if comp in df.columns:
                dso_col = comp
                break
        
        if dso_col and 'Inventory Turnover Ratio' in df.columns:
            days_inventory = 365 / df['Inventory Turnover Ratio'].replace(0, np.nan)
            dpo = df.get('Days Payable Outstanding', 0)
            df['Cash Conversion Cycle'] = df[dso_col] + days_inventory - dpo
    
    # =============================================
    # 10. LOGARITMOS PARA MACHINE LEARNING
    # =============================================
    
    size_metrics = ['Revenue', 'Total Assets', 'MarketCap', 'Enterprise Value']
    for metric in size_metrics:
        log_col = f'Log_{metric}'
        if metric in df.columns and log_col not in df.columns:
            # CORRECCIÓN: Manejar valores NA/NaN antes de aplicar log1p
            valid_mask = df[metric].notna() & (df[metric] > 0)
            df[log_col] = pd.NA  # Inicializar con NA
            if valid_mask.any():
                df.loc[valid_mask, log_col] = np.log1p(df.loc[valid_mask, metric].abs())
    
    # =============================================
    # 11. YIELD Y RATIOS DE DISTRIBUCIÓN
    # =============================================
    
    # Free Cash Flow Yield
    if 'FCF_Yield' not in df.columns:
        if 'Free Cash Flow' in df.columns and 'MarketCap' in df.columns:
            df['FCF_Yield'] = (df['Free Cash Flow'] / df['MarketCap'].replace(0, np.nan)) * 100
    
    # Dividend Yield (si hay dividendos)
    if 'Dividend_Yield' not in df.columns:
        if 'Common Stock Dividends Paid' in df.columns and shares_col in df.columns and 'Close' in df.columns:
            annual_dividend = df['Common Stock Dividends Paid'].abs() / df[shares_col].replace(0, np.nan)
            df['Dividend_Yield'] = (annual_dividend / df['Close'].replace(0, np.nan)) * 100
    
    # Payout Ratio
    if 'Payout_Ratio' not in df.columns:
        if 'Common Stock Dividends Paid' in df.columns and 'Net Income' in df.columns:
            df['Payout_Ratio'] = (df['Common Stock Dividends Paid'].abs() / df['Net Income'].replace(0, np.nan)) * 100
    
    return df


def procesar_parquet(path_src: Path, path_dest: Path, frequency: str):
    """Procesa un archivo parquet añadiendo precios, métricas profesionales y datos macro"""
    ticker = path_src.stem
    
    try:
        # Cargar datos fundamentales
        df = pd.read_parquet(path_src, engine="pyarrow")
        df["Date"] = pd.to_datetime(df["Date"])

        # Añadir columna con el ticker
        df["ticker"] = ticker

        # Añadir precios de mercado (SIMPLIFICADO - volver al original pero corregido)
        market_file = MARKETROOT / f"{ticker}.parquet"
        if market_file.exists():
            try:
                dm = pd.read_parquet(market_file, engine="pyarrow")
                dm["Date"] = pd.to_datetime(dm["date"])
                
                # CORRECCIÓN: Usar "close" (minúscula) y renombrar después
                if "close" in dm.columns:
                    dm = dm[["Date", "close"]].sort_values("Date")
                    dm.rename(columns={"close": "Close"}, inplace=True)
                    
                    # merge_asof con direction='forward'
                    df = df.sort_values("Date")
                    df = pd.merge_asof(
                        df, dm, on="Date", direction="forward", allow_exact_matches=True
                    )
                else:
                    df["Close"] = pd.NA
            except:
                df["Close"] = pd.NA
        else:
            df["Close"] = pd.NA
        
        # Calcular MarketCap (código original)
        if "Shares Outstanding" in df.columns:
            df["MarketCap"] = df["Shares Outstanding"] * df["Close"]
        elif "Basic Shares Outstanding" in df.columns:
            df["MarketCap"] = df["Basic Shares Outstanding"] * df["Close"]
        else:
            df["MarketCap"] = pd.NA
        
        # Calcular TODAS las métricas profesionales
        df = calculate_professional_metrics(df)
        
        # AÑADIR DATOS MACROECONÓMICOS
        df_with_macro = merge_macro_with_fundamental(df, frequency)
        
        # Verificar si se añadieron datos macro
        macro_cols = ['VIX', 'GDP', 'CPI', 'unemployment_rate', '10Y_Treasury']
        macro_available = any(col in df_with_macro.columns and not df_with_macro[col].isna().all() 
                             for col in macro_cols if col in df_with_macro.columns)
        
        df = df_with_macro
        
        # REORDENAR COLUMNAS: Date primero, Ticker segundo, luego el resto
        columnas = df.columns.tolist()
        # Asegurarse de que Date y Ticker estén en las primeras posiciones
        columnas_ordenadas = ['Date', 'ticker'] + [col for col in columnas if col not in ['Date', 'ticker']]
        df = df[columnas_ordenadas]

        # Guardar resultado
        df.to_parquet(path_dest / path_src.name, engine="pyarrow", index=False)
        
        # Reporte
        price_available = not df["Close"].isna().all()
        
        status_symbols = []
        status_symbols.append("✓" if price_available else "⚠")
        status_symbols.append("📊" if macro_available else "📊?")
        status = " ".join(status_symbols)
        
        return f"{ticker} {status} ({len(df)} records, {len([c for c in df.columns if c not in ['Date']])} total metrics)"
        
    except Exception as e:
        return f"{ticker} ✗ ERROR: {str(e)}"


if __name__ == "__main__":
    print("🚀 Starting Enhanced Professional Metrics Processing with Macro Data...")
    print("📊 This will add 40+ institutional-grade financial metrics")
    print("💹 Improved market data integration with forward-looking prices")
    print("🌍 NEW: Macro-economic data integration (VIX, rates, GDP, etc.)")
    print("=" * 80)
    
    # Verificar que las rutas existen
    print(f"📂 ROOT directory: {ROOT.resolve()}")
    print(f"📂 MACROROOT directory: {MACROROOT.resolve()}")
    print(f"📂 MARKETROOT directory: {MARKETROOT.resolve()}")
    print(f"📂 DESTROOT directory: {DESTROOT.resolve()}")
    print(f"📊 MACRODATA file: {MACRODATAPATH.resolve()}")
    
    if not ROOT.exists():
        print(f"❌ ERROR: ROOT directory does not exist: {ROOT.resolve()}")
        exit(1)
    
    # Verificar disponibilidad de datos macro al inicio
    macro_available = check_macro_data_availability()
    if not macro_available:
        print("⚠️  Continuing without macro data...")
    else:
        print("🎯 Macro data verified and available for workers")
    
    total_files = 0
    for sub, src_dir in dirs.items():
        if src_dir.exists():
            total_files += len(list(src_dir.glob("*.parquet")))
    
    print(f"📁 Processing {total_files} files across {len(dirs)} directories...")
    
    # Configuración óptima de workers
    cpu_count = os.cpu_count()
    max_workers = min(cpu_count * 2, total_files, 32)
    
    print(f"💻 Using {max_workers} workers (CPU cores: {cpu_count})")
    
    processed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for sub, src_dir in dirs.items():
            if not src_dir.exists():
                print(f"⚠️  Directory not found: {src_dir}")
                continue
                
            dst_dir = DESTROOT / sub
            frequency = 'annual' if sub == 'Anual' else 'quarterly'
            print(f"📂 Processing {sub} directory (frequency: {frequency})...")
            
            for src_file in src_dir.glob("*.parquet"):
                futures.append(
                    executor.submit(procesar_parquet, src_file, dst_dir, frequency)
                )
        
        for f in as_completed(futures):
            result = f.result()
            print(f"   {result}")
            processed += 1
            
            if processed % 50 == 0:
                print(f"   📈 Progress: {processed}/{total_files} files processed")
    
    print("=" * 80)
    print("🎉 ENHANCED PROCESSING WITH MACRO DATA COMPLETE!")
    print()
    print("✅ Added Professional Metrics:")
    print("   📊 Valuation: P/E, P/B, P/S, EV/EBITDA, EV/Sales, FCF Yield")
    print("   💰 Debt Analysis: Debt/EBITDA, Debt/Assets, Net Debt, Interest Coverage")
    print("   💧 Liquidity: Quick Ratio, Cash Ratio, Working Capital")
    print("   📈 Growth: YoY Growth, 3Y CAGR for key metrics")
    print("   🏆 Quality: Piotroski Score, Altman Z-Score")
    print("   ⚡ Efficiency: Cash Conversion Cycle, DSO, DPO")
    print("   💎 Per-Share: Revenue/Share, FCF/Share, Assets/Share")
    print("   📐 Advanced: ROIC, Enterprise Value, Dividend Yield")
    print()
    print("🌍 Added Macro-Economic Indicators:")
    print("   📈 Market: VIX, VIX_MA, Yield_Curve, Credit_Spread")
    print("   💰 Rates: 10Y_Treasury, 2Y_Treasury, fed_funds_rate, AAA/BAA_Yield")
    print("   🏛️  Economic: GDP, GDP_Growth, CPI, Inflation_Rate")
    print("   👷 Employment: unemployment_rate, industrial_production")
    print("   🌍 FX/Commodities: eur_usd_rate, gold_usd")
    print("   💵 Monetary: m2_usd, consumer_sentiment")
    print()
    print("📱 All metrics are pre-calculated and cached in parquet files.")
    print("📊 Macro data aggregated: Annual for Anual/, Quarterly for Trimestral/")