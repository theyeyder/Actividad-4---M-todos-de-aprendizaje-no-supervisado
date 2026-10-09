import csv
import math
import random
from pathlib import Path


# ============================================================
# CONFIGURACION
# ============================================================

VARIABLES = [
    "pasajeros",
    "numero_paradas",
    "transbordos",
    "tiempo_estimado",
    "hora_pico"
]

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_PROYECTO / "datos" / "datos_transporte_no_supervisado.csv"


# ============================================================
# 1. CARGAR EL DATASET
# ============================================================

def cargar_datos(ruta):
    datos = []

    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)

        columnas_esperadas = {
            "id_viaje",
            "pasajeros",
            "numero_paradas",
            "transbordos",
            "tiempo_estimado",
            "hora_pico"
        }

        if not lector.fieldnames:
            raise ValueError("El archivo CSV no contiene encabezados.")

        faltantes = columnas_esperadas - set(lector.fieldnames)
        if faltantes:
            raise ValueError(
                "Faltan columnas en el dataset: " + ", ".join(sorted(faltantes))
            )

        for fila in lector:
            datos.append({
                "id_viaje": int(fila["id_viaje"]),
                "pasajeros": int(fila["pasajeros"]),
                "numero_paradas": int(fila["numero_paradas"]),
                "transbordos": int(fila["transbordos"]),
                "tiempo_estimado": int(fila["tiempo_estimado"]),
                "hora_pico": int(fila["hora_pico"])
            })

    if len(datos) < 3:
        raise ValueError("El dataset debe contener al menos 3 registros.")

    return datos


# ============================================================
# 2. SELECCIONAR LAS VARIABLES
# ============================================================

def seleccionar_variables(datos):
    matriz = []

    for fila in datos:
        matriz.append([
            float(fila[variable])
            for variable in VARIABLES
        ])

    return matriz


# ============================================================
# 3. NORMALIZAR LOS DATOS
# ============================================================

def calcular_media(valores):
    return sum(valores) / len(valores)


def calcular_desviacion(valores, media):
    varianza = sum((x - media) ** 2 for x in valores) / len(valores)
    return math.sqrt(varianza)


def normalizar_datos(matriz):
    columnas = list(zip(*matriz))
    medias = []
    desviaciones = []

    for columna in columnas:
        media = calcular_media(columna)
        desviacion = calcular_desviacion(columna, media)

        if desviacion == 0:
            desviacion = 1.0

        medias.append(media)
        desviaciones.append(desviacion)

    matriz_normalizada = []

    for fila in matriz:
        nueva_fila = []

        for i, valor in enumerate(fila):
            normalizado = (valor - medias[i]) / desviaciones[i]
            nueva_fila.append(normalizado)

        matriz_normalizada.append(nueva_fila)

    return matriz_normalizada, medias, desviaciones


def normalizar_nuevo_registro(fila, medias, desviaciones):
    return [
        (fila[i] - medias[i]) / desviaciones[i]
        for i in range(len(fila))
    ]


# ============================================================
# 4. CALCULAR DISTANCIAS
# ============================================================

def distancia_euclidiana(a, b):
    return math.sqrt(
        sum(
            (x - y) ** 2
            for x, y in zip(a, b)
        )
    )


# ============================================================
# 5. INICIALIZAR LOS CENTROIDES
# ============================================================

def inicializar_centroides_kmeans_pp(matriz, k, semilla=42):
    if k <= 0:
        raise ValueError("El numero de clusters debe ser mayor que cero.")

    if k > len(matriz):
        raise ValueError("No puede haber mas clusters que registros.")

    aleatorio = random.Random(semilla)
    centroides = [list(aleatorio.choice(matriz))]

    while len(centroides) < k:
        distancias_cuadradas = []

        for punto in matriz:
            distancia_minima = min(
                distancia_euclidiana(punto, centroide) ** 2
                for centroide in centroides
            )
            distancias_cuadradas.append(distancia_minima)

        total = sum(distancias_cuadradas)

        if total == 0:
            for punto in matriz:
                if list(punto) not in centroides:
                    centroides.append(list(punto))
                    break
            continue

        objetivo = aleatorio.random() * total
        acumulado = 0.0

        for punto, peso in zip(matriz, distancias_cuadradas):
            acumulado += peso
            if acumulado >= objetivo:
                centroides.append(list(punto))
                break

    return centroides


