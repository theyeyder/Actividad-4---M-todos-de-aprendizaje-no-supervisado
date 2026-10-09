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


# ============================================================
# 6. ALGORITMO K-MEANS
# ============================================================

def asignar_clusters(matriz, centroides):
    asignaciones = []

    for punto in matriz:
        distancias = [
            distancia_euclidiana(punto, centroide)
            for centroide in centroides
        ]

        cluster = distancias.index(min(distancias))
        asignaciones.append(cluster)

    return asignaciones


def recalcular_centroides(matriz, asignaciones, k):
    dimension = len(matriz[0])
    nuevos = []

    for cluster in range(k):
        puntos = [
            punto
            for punto, grupo in zip(matriz, asignaciones)
            if grupo == cluster
        ]

        if not puntos:
            nuevos.append([0.0] * dimension)
            continue

        centroide = []
        for columna in range(dimension):
            centroide.append(
                sum(p[columna] for p in puntos) / len(puntos)
            )

        nuevos.append(centroide)

    return nuevos


def kmeans(matriz, k=3, max_iteraciones=100, tolerancia=1e-6, semilla=42):
    centroides = inicializar_centroides_kmeans_pp(
        matriz,
        k,
        semilla
    )

    asignaciones = None

    for iteracion in range(1, max_iteraciones + 1):
        nuevas_asignaciones = asignar_clusters(matriz, centroides)
        nuevos_centroides = recalcular_centroides(
            matriz,
            nuevas_asignaciones,
            k
        )

        movimiento = max(
            distancia_euclidiana(a, b)
            for a, b in zip(centroides, nuevos_centroides)
        )

        centroides = nuevos_centroides
        asignaciones = nuevas_asignaciones

        if movimiento <= tolerancia:
            return asignaciones, centroides, iteracion

    return asignaciones, centroides, max_iteraciones


# ============================================================
# 7. ASIGNAR VIAJES A CLUSTERS
# ============================================================

def calcular_inercia(matriz, asignaciones, centroides):
    return sum(
        distancia_euclidiana(
            punto,
            centroides[cluster]
        ) ** 2
        for punto, cluster in zip(matriz, asignaciones)
    )


def calcular_silhouette(matriz, asignaciones, k):
    if len(set(asignaciones)) < 2:
        return 0.0

    valores = []

    for indice, punto in enumerate(matriz):
        cluster_actual = asignaciones[indice]

        mismo_cluster = [
            j for j, cluster in enumerate(asignaciones)
            if cluster == cluster_actual and j != indice
        ]

        if mismo_cluster:
            a = sum(
                distancia_euclidiana(punto, matriz[j])
                for j in mismo_cluster
            ) / len(mismo_cluster)
        else:
            a = 0.0

        distancias_otros = []

        for otro_cluster in range(k):
            if otro_cluster == cluster_actual:
                continue

            indices_otro = [
                j for j, cluster in enumerate(asignaciones)
                if cluster == otro_cluster
            ]

            if indices_otro:
                promedio = sum(
                    distancia_euclidiana(punto, matriz[j])
                    for j in indices_otro
                ) / len(indices_otro)
                distancias_otros.append(promedio)

        if not distancias_otros:
            valores.append(0.0)
            continue

        b = min(distancias_otros)
        denominador = max(a, b)
        s = (b - a) / denominador if denominador > 0 else 0.0
        valores.append(s)

    return sum(valores) / len(valores)


# ============================================================
# 8. ANALIZAR LOS GRUPOS
# ============================================================

def analizar_grupos(datos, asignaciones, k):
    resumenes = []

    for cluster in range(k):
        registros = [
            fila
            for fila, grupo in zip(datos, asignaciones)
            if grupo == cluster
        ]

        resumen = {
            "cluster": cluster,
            "cantidad": len(registros)
        }

        for variable in VARIABLES:
            resumen[variable] = (
                sum(fila[variable] for fila in registros) / len(registros)
                if registros else 0.0
            )

        resumenes.append(resumen)

    # Nombre interpretativo segun promedio de pasajeros + tiempo.
    orden = sorted(
        resumenes,
        key=lambda r: (r["pasajeros"] + r["tiempo_estimado"])
    )

    nombres = [
        "Baja demanda",
        "Demanda media",
        "Alta demanda"
    ]

    for posicion, resumen in enumerate(orden):
        if k == 3:
            resumen["nombre"] = nombres[posicion]
        else:
            resumen["nombre"] = f"Grupo {posicion + 1}"

    return resumenes


# ============================================================
# 9. PRESENTAR LOS RESULTADOS
# ============================================================

def mostrar_muestra(datos, cantidad=5):
    print("\nPrimeros registros del dataset:")
    print("ID | Pasajeros | Paradas | Transbordos | Tiempo | Hora pico")
    print("-" * 64)

    for fila in datos[:cantidad]:
        print(
            f"{fila['id_viaje']:>2} | "
            f"{fila['pasajeros']:>9} | "
            f"{fila['numero_paradas']:>7} | "
            f"{fila['transbordos']:>10} | "
            f"{fila['tiempo_estimado']:>6} | "
            f"{fila['hora_pico']:>9}"
        )


def mostrar_resultados(datos, asignaciones, resumenes, inercia, silhouette, iteraciones):
    print("\n============================================")
    print("RESULTADOS DEL AGRUPAMIENTO K-MEANS")
    print("============================================")
    print(f"Iteraciones hasta converger: {iteraciones}")
    print(f"Inercia: {inercia:.4f}")
    print(f"Coeficiente silhouette: {silhouette:.4f}")

    print("\nRESUMEN DE LOS CLUSTERS")
    print("-" * 82)

    for resumen in sorted(resumenes, key=lambda r: r['nombre']):
        print(
            f"Cluster {resumen['cluster'] + 1} - {resumen['nombre']}: "
            f"{resumen['cantidad']} viajes | "
            f"pasajeros={resumen['pasajeros']:.1f} | "
            f"paradas={resumen['numero_paradas']:.1f} | "
            f"transbordos={resumen['transbordos']:.1f} | "
            f"tiempo={resumen['tiempo_estimado']:.1f} | "
            f"hora_pico={resumen['hora_pico']:.2f}"
        )

    print("\nASIGNACION DE LOS PRIMEROS 15 VIAJES")
    print("-" * 55)

    nombres_por_cluster = {
        resumen['cluster']: resumen['nombre']
        for resumen in resumenes
    }

    for fila, cluster in list(zip(datos, asignaciones))[:15]:
        print(
            f"Viaje {fila['id_viaje']:>2} -> "
            f"Cluster {cluster + 1} ({nombres_por_cluster[cluster]})"
        )


def leer_entero(mensaje, minimo, maximo):
    while True:
        try:
            valor = int(input(mensaje))
            if minimo <= valor <= maximo:
                return valor
            print(f"Ingrese un valor entre {minimo} y {maximo}.")
        except ValueError:
            print("Debe ingresar un numero entero.")


def asignar_nuevo_viaje(centroides, medias, desviaciones, nombres_por_cluster):
    print("\n============================================")
    print("ASIGNAR UN NUEVO VIAJE A UN CLUSTER")
    print("============================================")

    nuevo = [
        leer_entero("Cantidad de pasajeros (1-120): ", 1, 120),
        leer_entero("Numero de paradas (1-10): ", 1, 10),
        leer_entero("Numero de transbordos (0-5): ", 0, 5),
        leer_entero("Tiempo estimado en minutos (1-120): ", 1, 120),
        leer_entero("Hora pico? 1=Si / 0=No: ", 0, 1)
    ]

    normalizado = normalizar_nuevo_registro(
        nuevo,
        medias,
        desviaciones
    )

    distancias = [
        distancia_euclidiana(normalizado, centroide)
        for centroide in centroides
    ]

    cluster = distancias.index(min(distancias))

    print(
        f"\nEl nuevo viaje pertenece al Cluster {cluster + 1}: "
        f"{nombres_por_cluster[cluster]}"
    )


# ============================================================
# 10. PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("\n====================================================")
    print(" APRENDIZAJE NO SUPERVISADO - K-MEANS")
    print(" SISTEMA DE TRANSPORTE")
    print("====================================================")

    try:
        datos = cargar_datos(RUTA_DATOS)
    except (FileNotFoundError, ValueError) as error:
        print("\nERROR AL CARGAR LOS DATOS:")
        print(error)
        return

    print(f"\nArchivo de datos: {RUTA_DATOS}")
    print(f"Registros cargados: {len(datos)}")
    mostrar_muestra(datos)

    matriz = seleccionar_variables(datos)
    matriz_normalizada, medias, desviaciones = normalizar_datos(matriz)

    k = 3
    asignaciones, centroides, iteraciones = kmeans(
        matriz_normalizada,
        k=k,
        max_iteraciones=100,
        semilla=42
    )

    inercia = calcular_inercia(
        matriz_normalizada,
        asignaciones,
        centroides
    )

    silhouette = calcular_silhouette(
        matriz_normalizada,
        asignaciones,
        k
    )

    resumenes = analizar_grupos(
        datos,
        asignaciones,
        k
    )

    mostrar_resultados(
        datos,
        asignaciones,
        resumenes,
        inercia,
        silhouette,
        iteraciones
    )

    nombres_por_cluster = {
        resumen['cluster']: resumen['nombre']
        for resumen in resumenes
    }

    while True:
        print("\n¿Que desea hacer?")
        print("1. Asignar un nuevo viaje a un cluster")
        print("2. Salir")

        opcion = leer_entero(
            "Seleccione una opcion: ",
            1,
            2
        )

        if opcion == 1:
            asignar_nuevo_viaje(
                centroides,
                medias,
                desviaciones,
                nombres_por_cluster
            )
        else:
            print("\nGracias por utilizar el modelo no supervisado.")
            break


if __name__ == "__main__":
    main()
