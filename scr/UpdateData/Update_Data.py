import subprocess
import os
import sys

# === Configuración de rutas ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GETDATA_DIR = os.path.join(BASE_DIR, "GetData")
PROCESSDATA_DIR = os.path.join(BASE_DIR, "ProcessData")
GETRESULTS_DIR = os.path.join(BASE_DIR, "GetResults")

# Scripts organizados por categorías
SCRIPTS = {
    "screener": os.path.join(GETDATA_DIR, "Get_Screener_Macrotrends.py"),
    "yahoo": os.path.join(GETDATA_DIR, "Get_Yahoo_Prices.py"),
    "macro": os.path.join(GETDATA_DIR, "Get_Macro_Data.py"),
    "macrotrends": os.path.join(GETDATA_DIR, "Get_Macrotrends_Data.py"),
    "technical_results": os.path.join(GETRESULTS_DIR, "Get_Technical_Data_Results.py"),
    "process_add": os.path.join(PROCESSDATA_DIR, "Add_Price_and_Ratios_Fundamental_Data.py"),
    "process_forecast": os.path.join(PROCESSDATA_DIR, "Forecast_Fundamental_Data_and_Ratios.py"),
    "fundamental_results": os.path.join(GETRESULTS_DIR, "Get_Fundamental_Data_Results.py"),
}

# === Función para ejecutar scripts ===
def run_script(script_path):
    """Ejecuta un script de Python externo."""
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"✅ Ejecutado correctamente: {os.path.basename(script_path)}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar {script_path}: {e}")

# === Opciones de actualización ===
def actualizar_listado():
    run_script(SCRIPTS["screener"])

def actualizar_precios():
    actualizar_listado()
    run_script(SCRIPTS["yahoo"])

def actualizar_fundamentales_macro():
    actualizar_listado()
    run_script(SCRIPTS["macro"])
    run_script(SCRIPTS["macrotrends"])

def actualizar_tecnicos():
    actualizar_precios()
    run_script(SCRIPTS["technical_results"])

def actualizar_datos_procesados_fundamentales():
    actualizar_listado()
    run_script(SCRIPTS["yahoo"])
    run_script(SCRIPTS["macro"])
    run_script(SCRIPTS["macrotrends"])
    run_script(SCRIPTS["process_add"])
    run_script(SCRIPTS["process_forecast"])
    run_script(SCRIPTS["fundamental_results"])

def actualizar_todo():
    run_script(SCRIPTS["screener"])
    run_script(SCRIPTS["yahoo"])
    run_script(SCRIPTS["macro"])
    run_script(SCRIPTS["technical_results"])
    run_script(SCRIPTS["macrotrends"])
    run_script(SCRIPTS["process_add"])
    run_script(SCRIPTS["process_forecast"])
    run_script(SCRIPTS["fundamental_results"])

# === Menú principal ===
def main():
    print("\n=== Actualización de Datos para Dashboard ===")
    print("⚠️  IMPORTANTE: No es necesario ejecutar las opciones iniciales si seleccionas actualizar datos Fundamentales o Técnicos, ya que estas opciones incluyen esas actualizaciones automáticamente.\n")
    print("1. Actualizar listado de tickers y cotización de hoy")
    print("2. Actualizar precios de cotización históricos")
    print("3. Actualizar datos fundamentales y datos macro")
    print("4. Actualizar Datos y Resultados Técnicos")
    print("5. Actualizar Datos y Resultados Fundamentales")
    print("6. Actualizar TODO")
    print("0. Salir")

    opcion = input("\nSeleccione una opción: ").strip()

    if opcion == "1":
        actualizar_listado()
    elif opcion == "2":
        actualizar_precios()
    elif opcion == "3":
        actualizar_fundamentales_macro()
    elif opcion == "4":
        actualizar_tecnicos()
    elif opcion == "5":
        actualizar_datos_procesados_fundamentales()
    elif opcion == "6":
        actualizar_todo()
    elif opcion == "0":
        print("Saliendo...")
    else:
        print("⚠️ Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()
