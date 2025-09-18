import pandas as pd
import yfinance as yf
import requests
import random
import time
import re
import logging
import multiprocessing
import os

from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ----------------------------------------
# Configuración de rutas relativas
# ----------------------------------------
# Obtener la ruta absoluta del directorio raíz del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Construir la ruta completa al archivo .parquet
SCREENER_PARQUET = os.path.join(project_root, 'data', 'Ticker_List', 'screener.parquet')
OUTPUT_DIR = Path(os.path.join(project_root, 'data', 'Historical_Prices'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------
# Configuración conservadora para evitar bloqueos
# ----------------------------------------
START_DATE       = pd.to_datetime("2008-01-01")
USERSLEEP_MIN    = 1    # Más lento para evitar bloqueos
USERSLEEP_MAX    = 3   # Más lento para evitar bloqueos
BACKOFF_INITIAL  = 60  # 3 minutos de backoff inicial
MAX_RETRIES      = 3    # Más intentos
COOLDOWN_SEC     = 3600 # 2 horas de cooldown
MAX_WORKERS      = max(1, multiprocessing.cpu_count() - 1)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/120.0"
]

ACCEPT_LANGS = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.8",
    "es-ES,es;q=0.7,en;q=0.6",
    "de-DE,de;q=0.7,en;q=0.6"
]

# Logging mejorado
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Variables globales para manejo de estado
last_fetch_times = {}
temp_failed_tickers = []  # Para reintentar después
perm_failed_tickers = set()  # Fallos permanentes

# ----------------------------------------
# Funciones Auxiliares Mejoradas
# ----------------------------------------
def sanitize_ticker(ticker: str, for_filename: bool = False) -> str:
    """Sanitiza ticker y convierte puntos a guiones para Yahoo Finance, pero mantiene puntos para nombres de archivo"""
    if not ticker or pd.isna(ticker):
        return ""
    
    # Limpiar y normalizar
    clean_ticker = re.sub(r'[^A-Za-z0-9\.-]', '', str(ticker).upper().strip().lstrip('$'))
    
    if for_filename:
        # Para el nombre de archivo, mantener los puntos
        return clean_ticker
    else:
        # Para Yahoo Finance, convertir puntos a guiones
        return clean_ticker.replace('.', '-')

def get_last_trading_day() -> pd.Timestamp:
    """Obtiene el último día de trading"""
    today = pd.Timestamp.today().normalize()
    # Si es fin de semana, retroceder al viernes
    while today.weekday() >= 5:  # 5=Saturday, 6=Sunday
        today -= pd.Timedelta(days=1)
    return today

def create_conservative_session() -> requests.Session:
    """Crea una sesión muy conservadora para evitar detección"""
    sess = requests.Session()
    
    # Headers más naturales y variados
    sess.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': random.choice(ACCEPT_LANGS),
        'Accept-Encoding': 'gzip, deflate, br',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Upgrade-Insecure-Requests': '1',
        'Connection': 'keep-alive',
    })
    
    # Estrategia de reintentos muy conservadora
    retry_strategy = Retry(
        total=2,  # Pocos reintentos automáticos
        backoff_factor=2.0,  # Backoff más lento
        status_forcelist=[500, 502, 503, 504, 520, 521, 522, 524],  # No incluir 429 ni 404
        raise_on_status=False,
        respect_retry_after_header=True
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    sess.mount("http://", HTTPAdapter(max_retries=retry_strategy))
    
    return sess

def download_ticker_data(ticker_yahoo: str, ticker_file: str, start: pd.Timestamp, end: pd.Timestamp, is_retry: bool = False) -> str:
    """
    Descarga datos para un ticker específico
    ticker_yahoo: ticker para consultar Yahoo Finance (con guiones)
    ticker_file: ticker para el nombre del archivo (con puntos)
    Returns: 'success', 'temp_fail', 'perm_fail'
    """
    
    if ticker_yahoo in perm_failed_tickers:
        logger.debug(f"Omitiendo {ticker_yahoo} (fallo permanente previo)")
        return 'perm_fail'
    
    # Gestión estricta de cooldown
    current_time = datetime.now()
    if ticker_yahoo in last_fetch_times:
        elapsed = (current_time - last_fetch_times[ticker_yahoo]).total_seconds()
        if elapsed < COOLDOWN_SEC:
            wait_time = COOLDOWN_SEC - elapsed
            logger.info(f"Cooldown activo para {ticker_yahoo}: esperando {wait_time/60:.1f} minutos")
            time.sleep(wait_time)
    
    # Crear sesión conservadora
    session = create_conservative_session()
    yf.utils.requests = session
    
    try:
        ticker_obj = yf.Ticker(ticker_yahoo)
        output_file = OUTPUT_DIR / f"{ticker_file}.parquet"
        
        # Cargar datos existentes si los hay
        existing_data = pd.DataFrame()
        if output_file.exists():
            try:
                existing_data = pd.read_parquet(output_file)
                existing_data['date'] = pd.to_datetime(existing_data['date'])
            except Exception as e:
                logger.warning(f"Error leyendo datos existentes de {ticker_yahoo}: {e}")
        
        # Intentos de descarga con manejo robusto de errores
        new_data = pd.DataFrame()
        attempt = 0
        backoff_time = BACKOFF_INITIAL
        consecutive_404s = 0
        had_temp_errors = False
        
        while attempt < MAX_RETRIES and new_data.empty:
            try:
                attempt += 1
                retry_msg = " [REINTENTO]" if is_retry else ""
                logger.debug(f"Descargando {ticker_yahoo} (intento {attempt}/{MAX_RETRIES}){retry_msg}")
                
                # Pausa antes de cada intento para evitar ser detectado
                pre_request_delay = random.uniform(2, 6)
                time.sleep(pre_request_delay)
                
                new_data = ticker_obj.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=False,
                    actions=False,
                    timeout=20
                )
                
                if not new_data.empty:
                    logger.debug(f"Datos obtenidos para {ticker_yahoo}: {len(new_data)} filas")
                    break
                else:
                    logger.debug(f"Sin datos para {ticker_yahoo} pero sin error explícito")
                
            except Exception as error:
                error_msg = str(error).lower()
                
                if any(term in error_msg for term in ['rate limit', 'too many requests', '429']):
                    had_temp_errors = True
                    wait_time = backoff_time + random.uniform(30, 90)
                    logger.warning(f"Rate limit detectado para {ticker_yahoo}. Esperando {wait_time/60:.1f} min")
                    time.sleep(wait_time)
                    backoff_time *= 2
                    
                elif any(term in error_msg for term in ['timeout', 'connection', 'network']):
                    had_temp_errors = True
                    wait_time = random.uniform(30, 120)
                    logger.warning(f"Error de conexión en {ticker_yahoo}. Esperando {wait_time:.0f}s")
                    time.sleep(wait_time)
                    
                elif any(term in error_msg for term in ['404', 'not found']):
                    consecutive_404s += 1
                    if consecutive_404s >= 2:  # Más estricto que antes
                        logger.info(f"Ticker {ticker_yahoo} no encontrado (404 x{consecutive_404s})")
                        perm_failed_tickers.add(ticker_yahoo)
                        return 'perm_fail'
                    else:
                        # Podría ser 404 temporal por rate limiting
                        had_temp_errors = True
                        wait_time = backoff_time + random.uniform(60, 180)
                        logger.warning(f"404 posiblemente temporal en {ticker_yahoo}. Esperando {wait_time/60:.1f} min")
                        time.sleep(wait_time)
                        backoff_time *= 1.5
                        
                elif 'delisted' in error_msg:
                    logger.info(f"Ticker {ticker_yahoo} está delisted")
                    perm_failed_tickers.add(ticker_yahoo)
                    return 'perm_fail'
                    
                else:
                    had_temp_errors = True
                    wait_time = random.uniform(20, 60)
                    logger.warning(f"Error desconocido en {ticker_yahoo}: {error}. Esperando {wait_time:.0f}s")
                    time.sleep(wait_time)
        
        # Evaluar resultado de los intentos
        if new_data.empty:
            if had_temp_errors and not is_retry:
                logger.warning(f"Fallos temporales en {ticker_yahoo}, programando para reintento")
                return 'temp_fail'
            else:
                logger.warning(f"No se pudieron obtener datos para {ticker_yahoo} tras {MAX_RETRIES} intentos")
                perm_failed_tickers.add(ticker_yahoo)
                last_fetch_times[ticker_yahoo] = datetime.now()
                return 'perm_fail'
        
        # Procesamiento y limpieza de datos
        if new_data.index.tz is not None:
            new_data.index = new_data.index.tz_localize(None)
        
        new_data = new_data.reset_index()
        
        # Verificar columnas esenciales
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_columns if col not in new_data.columns]
        if missing:
            logger.error(f"Faltan columnas críticas en {ticker_yahoo}: {missing}")
            perm_failed_tickers.add(ticker_yahoo)
            return 'perm_fail'
        
        # Añadir Adj Close si no existe
        if 'Adj Close' not in new_data.columns:
            new_data['Adj Close'] = new_data['Close']
        
        # Renombrar columnas a formato estándar
        new_data = new_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        new_data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }, inplace=True)
        
        # Combinar con datos existentes
        if not existing_data.empty:
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset='date').sort_values('date')
        else:
            combined_data = new_data.sort_values('date')
        
        # Guardar datos
        combined_data.to_parquet(output_file, index=False, compression='snappy')
        
        success_prefix = "✓ [REINTENTO]" if is_retry else "✓"
        logger.info(f"{success_prefix} {ticker_yahoo}: {len(new_data)} nuevas filas, total: {len(combined_data)}")
        
        # Actualizar tiempo del último fetch
        last_fetch_times[ticker_yahoo] = datetime.now()
        
        # Sleep post-descarga con variabilidad para evitar detección
        post_sleep = random.uniform(USERSLEEP_MIN, USERSLEEP_MAX)
        # Ocasionalmente hacer pausas más largas (comportamiento humano)
        if random.random() < 0.15:  # 15% de probabilidad
            post_sleep += random.uniform(10, 30)
        time.sleep(post_sleep)
        
        return 'success'
        
    except Exception as critical_error:
        logger.error(f"Error crítico no manejado en {ticker_yahoo}: {critical_error}")
        if not is_retry:
            return 'temp_fail'
        else:
            perm_failed_tickers.add(ticker_yahoo)
            return 'perm_fail'
    
    finally:
        session.close()

def main():
    """Función principal con sistema de dos pasadas y filtrado por industria"""
    
    start_time = datetime.now()
    logger.info("=== INICIANDO DESCARGA DE DATOS HISTÓRICOS ===")
    
    # Verificar espacio disponible en disco
    try:
        import shutil
        free_gb = shutil.disk_usage(OUTPUT_DIR.parent).free / (1024**3)
        logger.info(f"Espacio libre en disco: {free_gb:.1f} GB")
        if free_gb < 2:
            logger.warning("⚠️ Espacio en disco limitado detectado")
    except:
        logger.debug("No se pudo verificar espacio en disco")
    
    # Verificar archivo screener
    if not os.path.exists(SCREENER_PARQUET):
        logger.error(f"Archivo screener no encontrado: {SCREENER_PARQUET}")
        return
    
    # Cargar y filtrar tickers
    try:
        screener_df = pd.read_parquet(SCREENER_PARQUET)
        logger.info(f"Screener cargado: {len(screener_df)} registros")
        
        if 'ticker' not in screener_df.columns:
            logger.error("Columna 'ticker' no encontrada en screener")
            return
        
        # Filtrar por industria si la columna existe
        industry_column = 'zacks_x_ind_desc'
        if industry_column in screener_df.columns:
            pre_filter_count = len(screener_df)
            
            # Filtrar: excluir 'Unclassified' y valores nulos
            filtered_df = screener_df[
                (screener_df[industry_column].notna()) &
                (screener_df[industry_column] != 'Unclassified')
            ].copy()
            
            post_filter_count = len(filtered_df)
            excluded_count = pre_filter_count - post_filter_count
            
            logger.info(f"Filtro por industria aplicado:")
            logger.info(f"  - Total original: {pre_filter_count}")
            logger.info(f"  - Después del filtro: {post_filter_count}")
            logger.info(f"  - Excluidos (Unclassified/NaN): {excluded_count}")
        else:
            logger.warning(f"Columna '{industry_column}' no encontrada - procesando todos los tickers")
            filtered_df = screener_df
        
    except Exception as e:
        logger.error(f"Error al cargar screener: {e}")
        return
    
    # Procesar lista de tickers
    raw_ticker_list = filtered_df['ticker'].dropna().unique()
    clean_tickers_yahoo = []  # Para consultas a Yahoo
    clean_tickers_file = []   # Para nombres de archivo
    
    for raw_ticker in raw_ticker_list:
        cleaned_yahoo = sanitize_ticker(raw_ticker, for_filename=False)
        cleaned_file = sanitize_ticker(raw_ticker, for_filename=True)
        if cleaned_yahoo and cleaned_file and len(cleaned_yahoo) >= 1:
            clean_tickers_yahoo.append(cleaned_yahoo)
            clean_tickers_file.append(cleaned_file)
    
    # Eliminar duplicados y aleatorizar orden
    ticker_pairs = list(zip(clean_tickers_yahoo, clean_tickers_file))
    random.shuffle(ticker_pairs)
    clean_tickers_yahoo, clean_tickers_file = zip(*ticker_pairs)
    clean_tickers_yahoo = list(clean_tickers_yahoo)
    clean_tickers_file = list(clean_tickers_file)
    
    logger.info(f"Tickers a procesar: {len(clean_tickers_yahoo)} (orden aleatorizado)")
    
    if not clean_tickers_yahoo:
        logger.error("No se encontraron tickers válidos para procesar")
        return
    
    # Configurar período de descarga
    end_date = get_last_trading_day()
    date_range_days = (end_date - START_DATE).days
    
    logger.info(f"Período de datos: {START_DATE.date()} hasta {end_date.date()} ({date_range_days} días)")
    logger.info(f"Configuración: {MAX_WORKERS} workers, sleep {USERSLEEP_MIN}-{USERSLEEP_MAX}s")
    
    # Estimación de tiempo
    avg_sleep = (USERSLEEP_MIN + USERSLEEP_MAX) / 2
    estimated_minutes = (len(clean_tickers_yahoo) * avg_sleep) / 60 / MAX_WORKERS
    logger.info(f"Tiempo estimado: ~{estimated_minutes:.0f} minutos")
    
    # PRIMERA PASADA: Procesamiento inicial
    success_count = 0
    temp_fail_count = 0
    perm_fail_count = 0
    temp_failed_tickers_yahoo = []   # Para reintentar después
    temp_failed_tickers_file = []    # Para reintentar después
    
    logger.info("=== PRIMERA PASADA: Procesamiento inicial ===")
    first_pass_start = datetime.now()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(download_ticker_data, ticker_yahoo, ticker_file, START_DATE, end_date, False): (ticker_yahoo, ticker_file)
            for ticker_yahoo, ticker_file in zip(clean_tickers_yahoo, clean_tickers_file)
        }
        
        with tqdm(total=len(clean_tickers_yahoo), desc='Primera pasada', ncols=100) as progress_bar:
            for future in as_completed(future_to_ticker):
                ticker_yahoo, ticker_file = future_to_ticker[future]
                try:
                    result = future.result()
                    if result == 'success':
                        success_count += 1
                    elif result == 'temp_fail':
                        temp_fail_count += 1
                        temp_failed_tickers_yahoo.append(ticker_yahoo)
                        temp_failed_tickers_file.append(ticker_file)
                    else:  # 'perm_fail'
                        perm_fail_count += 1
                        
                except Exception as e:
                    logger.error(f"Error en future para {ticker_yahoo}: {e}")
                    temp_fail_count += 1
                    temp_failed_tickers_yahoo.append(ticker_yahoo)
                    temp_failed_tickers_file.append(ticker_file)
                
                # Actualizar barra de progreso
                completed = success_count + temp_fail_count + perm_fail_count
                progress_pct = (completed / len(clean_tickers_yahoo)) * 100
                progress_bar.set_postfix({
                    'OK': success_count,
                    'TEMP': temp_fail_count,
                    'PERM': perm_fail_count,
                    'Progress': f"{progress_pct:.0f}%"
                })
                progress_bar.update(1)
    
    first_pass_duration = datetime.now() - first_pass_start
    logger.info(f"Primera pasada completada en {first_pass_duration}")
    logger.info(f"Resultados: {success_count} éxito, {temp_fail_count} temporal, {perm_fail_count} permanente")
    
    # SEGUNDA PASADA: Reintentos de fallos temporales
    if temp_failed_tickers_yahoo:
        logger.info(f"=== SEGUNDA PASADA: Reintentando {len(temp_failed_tickers_yahoo)} tickers ===")
        
        # Pausa estratégica entre pasadas
        cooldown_minutes = 10
        logger.info(f"Pausa estratégica de {cooldown_minutes} minutos para resetear rate limits...")
        time.sleep(cooldown_minutes * 60)
        
        # Aleatorizar orden de reintentos
        temp_pairs = list(zip(temp_failed_tickers_yahoo, temp_failed_tickers_file))
        random.shuffle(temp_pairs)
        temp_failed_tickers_yahoo, temp_failed_tickers_file = zip(*temp_pairs)
        temp_failed_tickers_yahoo = list(temp_failed_tickers_yahoo)
        temp_failed_tickers_file = list(temp_failed_tickers_file)
        
        second_pass_start = datetime.now()
        
        # Solo 1 worker para reintentos (más conservador)
        with ThreadPoolExecutor(max_workers=1) as executor:
            retry_futures = {
                executor.submit(download_ticker_data, ticker_yahoo, ticker_file, START_DATE, end_date, True): (ticker_yahoo, ticker_file)
                for ticker_yahoo, ticker_file in zip(temp_failed_tickers_yahoo, temp_failed_tickers_file)
            }
            
            with tqdm(total=len(temp_failed_tickers_yahoo), desc='Reintentos', ncols=100) as retry_progress:
                for future in as_completed(retry_futures):
                    ticker_yahoo, ticker_file = retry_futures[future]
                    try:
                        result = future.result()
                        if result == 'success':
                            success_count += 1
                            temp_fail_count -= 1
                        else:  # Se convierte en fallo permanente
                            perm_fail_count += 1
                            temp_fail_count -= 1
                    except Exception as e:
                        logger.error(f"Error en reintento para {ticker_yahoo}: {e}")
                        perm_fail_count += 1
                        temp_fail_count -= 1
                    
                    retry_progress.set_postfix({
                        'OK_Total': success_count,
                        'FAIL_Total': perm_fail_count,
                        'Remaining': temp_fail_count
                    })
                    retry_progress.update(1)
        
        second_pass_duration = datetime.now() - second_pass_start
        logger.info(f"Segunda pasada completada en {second_pass_duration}")
    
    # RESUMEN FINAL
    total_duration = datetime.now() - start_time
    total_processed = len(clean_tickers_yahoo)
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    avg_time_per_ticker = total_duration.total_seconds() / total_processed
    
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"Duración total: {total_duration}")
    logger.info(f"Tickers procesados: {total_processed}")
    logger.info(f"Descargas exitosas: {success_count}")
    logger.info(f"Fallos temporales finales: {temp_fail_count}")
    logger.info(f"Fallos permanentes: {perm_fail_count}")
    logger.info(f"Tasa de éxito: {success_rate:.1f}%")
    logger.info(f"Tiempo promedio por ticker: {avg_time_per_ticker:.1f}s")
    
    # Guardar lista de fallos permanentes
    if perm_failed_tickers:
        failures_file = OUTPUT_DIR / "failed_tickers.txt"
        try:
            with open(failures_file, 'w', encoding='utf-8') as f:
                f.write("# Tickers que fallaron permanentemente\n")
                f.write(f"# Generado: {datetime.now()}\n\n")
                for ticker in sorted(perm_failed_tickers):
                    f.write(f"{ticker}\n")
            logger.info(f"Lista de fallos guardada en: {failures_file}")
        except Exception as e:
            logger.warning(f"No se pudo guardar lista de fallos: {e}")
        
        # Mostrar algunos ejemplos en el log
        sample_failures = sorted(list(perm_failed_tickers))[:10]
        logger.info(f"Ejemplos de fallos permanentes: {sample_failures}")
        if len(perm_failed_tickers) > 10:
            logger.info(f"... y {len(perm_failed_tickers) - 10} más (ver {failures_file})")
    
    # Evaluación final del desempeño
    if success_rate == 100:
        logger.info("🎉 ¡Descarga 100% exitosa!")
    elif success_rate >= 98:
        logger.info("✅ Excelente tasa de éxito")
    elif success_rate >= 95:
        logger.info("✅ Muy buena tasa de éxito")
    elif success_rate >= 85:
        logger.info("⚠️ Buena tasa de éxito, algunos fallos menores")
    else:
        logger.warning("❌ Tasa de éxito baja, revisar configuración")
    
    logger.info(f"Archivos de datos guardados en: {OUTPUT_DIR}")
    logger.info(f"Configuración final: {MAX_WORKERS} workers, sleep {USERSLEEP_MIN}-{USERSLEEP_MAX}s")
    logger.info("=== FIN DEL PROCESO ===")

if __name__ == '__main__':
    main()