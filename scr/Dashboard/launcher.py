import subprocess
import time
import webbrowser
import os
import socket

# Paths relativos a los dashboards desde launcher.py
dashboard_dir = os.path.dirname(__file__)

PRINCIPAL_PATH = os.path.join(dashboard_dir, 'Principal_Dashboard.py')
TECHNICAL_PATH = os.path.join(dashboard_dir, 'Technical_Dashboard.py')
FINANCIAL_PATH = os.path.join(dashboard_dir, 'Financial_Dashboard.py')
def get_local_ip():
    """Obtiene la IP local (ej. 192.168.0.xx)"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def launch_dashboards():
    procs = []
    # Lanzar cada dashboard en su puerto correspondiente
    procs.append(subprocess.Popen(["python", PRINCIPAL_PATH]))
    procs.append(subprocess.Popen(["python", TECHNICAL_PATH]))
    procs.append(subprocess.Popen(["python", FINANCIAL_PATH]))

    # Esperar a que los servidores arranquen
    time.sleep(5)

    # Obtener IP local de la máquina
    local_ip = get_local_ip()

    # Abrir el dashboard principal en el navegador (usando IP local, accesible en LAN)
    webbrowser.open_new(f"http://{local_ip}:8050")

    return procs

if __name__ == "__main__":
    processes = launch_dashboards()
    print("✅ Dashboards iniciados en:")
    print(" - Principal:   http://<IP_LOCAL>:8050")
    print(" - Técnico:     http://<IP_LOCAL>:8051")
    print(" - Fundamental: http://<IP_LOCAL>:8052")

    try:
        # Mantener el launcher vivo para no cerrar los procesos
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹ Cerrando dashboards...")
        for p in processes:
            p.terminate()
