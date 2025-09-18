import requests
from bs4 import BeautifulSoup
import re
import json
import pandas as pd
import logging
from pathlib import Path
import os


# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
URL = "https://www.macrotrends.net/stocks/stock-screener"

# Obtener la ruta absoluta del directorio raíz del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Construir la ruta completa al archivo .parquet
OUTPUT_FILE = os.path.join(project_root, 'data', 'Ticker_List', 'screener.parquet')


# Headers actualizados
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

def create_session():
    """Crear sesión de requests con headers"""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session

def fetch_page_content(session, url):
    """Obtener contenido de la página"""
    logger.info(f"Obteniendo datos de: {url}")
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        logger.info(f"Respuesta exitosa. Status: {response.status_code}")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al obtener la página: {e}")
        raise

def extract_json_data(html_content):
    """Extraer datos JSON del HTML"""
    logger.info("Parseando contenido HTML...")
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Buscar script con originalData
    script_tag = soup.find('script', string=re.compile(r'var\s+originalData\s*='))
    
    if not script_tag:
        logger.error("No se encontró el script con 'originalData'")
        raise ValueError("Script con 'originalData' no encontrado")
    
    script_text = script_tag.string
    logger.info("Script encontrado, extrayendo datos JSON...")
    
    # Patrones para extraer JSON
    patterns = [
        r'var\s+originalData\s*=\s*(\[.*?\]);',
        r'originalData\s*=\s*(\[.*?\]);',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, script_text, re.DOTALL)
        if match:
            logger.info("Datos JSON extraídos exitosamente")
            return match.group(1)
    
    logger.error("No se pudo extraer originalData")
    raise ValueError("No se pudo extraer 'originalData' del script")

def parse_and_validate_data(json_string):
    """Parsear y validar datos JSON"""
    try:
        data = json.loads(json_string)
        
        if not isinstance(data, list) or not data:
            raise ValueError("Los datos no son una lista válida o están vacíos")
        
        logger.info(f"Se cargaron {len(data)} registros")
        logger.info(f"Columnas disponibles: {list(data[0].keys()) if data else 'N/A'}")
        
        return data
    
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON: {e}")
        raise

def optimize_dataframe_types(df):
    """Optimizar tipos de datos del DataFrame de forma simple"""
    logger.info("Optimizando tipos de datos...")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Intentar convertir columnas numéricas que están como string
            try:
                # Limpiar valores comunes que impiden conversión numérica
                cleaned = df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
                numeric_values = pd.to_numeric(cleaned, errors='coerce')
                
                # Si más del 50% son números válidos, convertir
                if numeric_values.notna().sum() / len(df) > 0.5:
                    df[col] = numeric_values
                    logger.debug(f"Columna '{col}' convertida a numérica")
            except:
                continue
    
    return df

def save_to_parquet(data, filename):
    """Guardar datos en formato Parquet"""
    logger.info("Convirtiendo datos a DataFrame...")
    
    try:
        df = pd.DataFrame(data)
        
        if df.empty:
            raise ValueError("DataFrame está vacío")
        
        # Optimizar tipos de datos
        df = optimize_dataframe_types(df)
        
        # Crear directorio si no existe
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar con compresión
        df.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
        
        # Información del archivo
        file_size = Path(filename).stat().st_size / (1024 * 1024)
        logger.info(f"Archivo guardado: '{filename}' ({file_size:.2f} MB)")
        logger.info(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        
        return True
    
    except Exception as e:
        logger.error(f"Error al guardar archivo: {e}")
        return False

def main():
    """Función principal"""
    logger.info("=== Iniciando Stock Screener Scraper ===")
    
    session = None
    try:
        # 1. Crear sesión
        session = create_session()
        
        # 2. Obtener contenido web
        html_content = fetch_page_content(session, URL)
        
        # 3. Extraer datos JSON
        json_string = extract_json_data(html_content)
        
        # 4. Parsear y validar
        data = parse_and_validate_data(json_string)
        
        # 5. Guardar en Parquet
        success = save_to_parquet(data, OUTPUT_FILE)
        
        if success:
            logger.info("=== Proceso completado exitosamente! ===")
        else:
            logger.error("=== Error al completar el proceso ===")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error general: {e}")
        return False
    
    finally:
        if session:
            session.close()

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)