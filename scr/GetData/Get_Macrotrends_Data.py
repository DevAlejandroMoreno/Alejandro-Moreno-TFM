import requests
from bs4 import BeautifulSoup
import re
import json
import pandas as pd
import time
import random
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import importlib.util
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://www.macrotrends.net"
SCREENER_URL = f"{BASE_URL}/stocks/stock-screener"

# Definir rutas relativas desde la ubicación del script
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent

CACHE_DIR = script_dir / 'Macrotrends_Cache'
OUTPUT_ROOT = project_root / 'data' / 'Fundamental_Data'
SCREENER_PATH = project_root / 'data' / 'Ticker_List' / 'screener.parquet'
ANNUAL_DIR = OUTPUT_ROOT / 'Anual'
QUARTERLY_DIR = OUTPUT_ROOT / 'Trimestral'

# Crear directorios si no existen
for d in [CACHE_DIR, ANNUAL_DIR, QUARTERLY_DIR]: 
    d.mkdir(parents=True, exist_ok=True)

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/115.0.5790.102 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0'
]
STATEMENT_KEYS = ['income-statement', 'balance-sheet', 'cash-flow-statement', 'financial-ratios']
PER_SHARE = ['Basic EPS', 'EPS - Earnings Per Share']
RATIOS = ['Current Ratio','Long-term Debt / Capital','Debt/Equity Ratio','Gross Margin',
          'Operating Margin','EBIT Margin','EBITDA Margin','Pre-Tax Profit Margin',
          'Net Profit Margin','Asset Turnover','Inventory Turnover Ratio','Receiveable Turnover',
          'Days Sales In Receivables','ROE - Return On Equity','Return On Tangible Equity',
          'ROA - Return On Assets','ROI - Return On Investment','Book Value Per Share',
          'Operating Cash Flow Per Share','Free Cash Flow Per Share']

PROGRESS_FILE = CACHE_DIR / 'progress.json'
session = requests.Session()

def import_screener_module():
    """Importar dinámicamente el módulo Get_Screener_Macrotrends"""
    screener_module_path = script_dir / 'Get_Screener_Macrotrends.py'
    
    if not screener_module_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {screener_module_path}")
    
    spec = importlib.util.spec_from_file_location("Get_Screener_Macrotrends", screener_module_path)
    screener_module = importlib.util.module_from_spec(spec)
    
    # Añadir el módulo al sys.modules para evitar problemas de importación
    sys.modules["Get_Screener_Macrotrends"] = screener_module
    spec.loader.exec_module(screener_module)
    
    return screener_module

def create_screener_file():
    """Crear el archivo screener usando el módulo Get_Screener_Macrotrends"""
    logger.info("El archivo screener.parquet no existe. Creándolo...")
    
    try:
        screener_module = import_screener_module()
        success = screener_module.main()
        
        if success:
            logger.info("Archivo screener.parquet creado exitosamente")
            return True
        else:
            logger.error("Error al crear el archivo screener.parquet")
            return False
            
    except Exception as e:
        logger.error(f"Error al importar o ejecutar Get_Screener_Macrotrends: {e}")
        return False

def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress))

def load_progress():
    return json.loads(PROGRESS_FILE.read_text()) if PROGRESS_FILE.exists() else {}

def get_with_retry(url, retries=5):
    for i in range(1, retries+1):
        session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        try:
            r = session.get(url, timeout=30)
            if r.status_code==429: raise requests.HTTPError()
            r.raise_for_status(); return r
        except Exception:
            if i<retries: time.sleep(min(60,5*2**(i-1))+random.random()); continue
            raise

def load_screener_data():
    """Cargar datos del screener, creándolo si no existe"""
    if SCREENER_PATH.exists(): 
        logger.info("Cargando screener.parquet existente...")
        return pd.read_parquet(SCREENER_PATH)
    else:
        logger.info("Archivo screener.parquet no encontrado")
        success = create_screener_file()
        
        if success and SCREENER_PATH.exists():
            logger.info("Cargando screener.parquet recién creado...")
            return pd.read_parquet(SCREENER_PATH)
        else:
            raise FileNotFoundError("No se pudo crear o cargar el archivo screener.parquet")

def fetch_statement_data(ticker,slug,stmt,freq):
    url = f"{BASE_URL}/stocks/charts/{ticker}/{slug}/{stmt}?freq={freq}"
    r = get_with_retry(url)
    s = BeautifulSoup(r.text,'html.parser').find('script',string=re.compile(r'var originalData'))
    m = re.search(r'var originalData = (\[.*?\]);',s.string,re.S)
    if not m: raise ValueError('no data')
    data = json.loads(m.group(1)); recs=[]
    for e in data:
        rec={k:e[k] for k in e if k not in ['field_name','popup_icon']}
        rec['Field']=re.sub(r'<.*?>','',e.get('field_name','')); recs.append(rec)
    df=pd.DataFrame(recs)
    for c in df.columns:
        if c!='Field': df[c]=pd.to_numeric(df[c].astype(str).str.replace(',',''),errors='coerce')
    return df

def process_ticker(ticker,slug):
    done = progress.get('done',[])
    if ticker in done: return
    for freq,outdir in [('A',ANNUAL_DIR),('Q',QUARTERLY_DIR)]:
        # if output exists skip
        if (outdir/f"{ticker}.parquet").exists(): continue
        dfs=[]
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures={ex.submit(fetch_statement_data,ticker,slug,st,freq):st for st in STATEMENT_KEYS}
            for f,st in futures.items():
                try: dfs.append(f.result())
                except ValueError: print(f"{ticker} {st} no data, skipping"); continue
                except: print(f"{ticker} {st} error, skip"); continue
        # even if dfs empty, mark done to avoid reattempts
        if dfs:
            df_all=pd.concat(dfs,axis=0)
            df_pivot=df_all.set_index('Field').T.reset_index().rename(columns={'index':'Date'})
            df_pivot['Date']=pd.to_datetime(df_pivot['Date'],errors='coerce')
            for c in df_pivot.columns:
                if c=='Date' or c in PER_SHARE or c in RATIOS: continue
                df_pivot[c]*=1e6
            df_pivot.to_parquet(outdir/f"{ticker}.parquet",index=False)
    progress.setdefault('done',[]).append(ticker); save_progress(progress)

def main():
    """Función principal que puede ser llamada desde otros archivos"""
    logger.info("=== Iniciando extracción de datos financieros ===")
    
    try:
        # Cargar datos del screener
        screener_df = load_screener_data()
        logger.info(f"Cargados {len(screener_df)} tickers del screener")
        
        # Filtrar tickers con industria "Unclassified" si existe la columna
        if 'zacks_x_ind_desc' in screener_df.columns:
            unclassified_count = len(screener_df[screener_df['zacks_x_ind_desc'] == 'Unclassified'])
            screener_df = screener_df[screener_df['zacks_x_ind_desc'] != 'Unclassified']
            logger.info(f"Filtrados {unclassified_count} tickers con industria 'Unclassified'")
            logger.info(f"Tickers a procesar: {len(screener_df)}")
        else:
            logger.info("Columna 'zacks_x_ind_desc' no encontrada, procesando todos los tickers")
        
        global progress
        progress = load_progress()
        all_tickers = screener_df['ticker'].tolist()
        
        if set(progress.get('done',[]))>=set(all_tickers): 
            progress = {}; save_progress(progress)
        
        # Procesar cada ticker
        for t,slug in tqdm(zip(screener_df['ticker'],screener_df['comp_name']),total=len(screener_df)):
            if not t or not slug: continue
            process_ticker(t,slug); time.sleep(random.uniform(1,3))
        
        logger.info('=== Proceso completado exitosamente ===')
        return True
        
    except Exception as e:
        logger.error(f"Error en el proceso principal: {e}")
        return False
    
    finally:
        if session:
            session.close()

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)