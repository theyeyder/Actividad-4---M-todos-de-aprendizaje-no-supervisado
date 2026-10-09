# Actividad 4 - Metodos de aprendizaje no supervisado

## Proyecto
Modelo de aprendizaje no supervisado para identificar patrones de viajes en un sistema de transporte mediante K-Means.

## Objetivo
Agrupar viajes similares sin utilizar una variable objetivo o etiqueta previa. El modelo analiza pasajeros, numero de paradas, transbordos, tiempo estimado y hora pico.

## Importante
Esta version utiliza unicamente la biblioteca estandar de Python. No requiere `pandas`, `scikit-learn` ni instalaciones con `pip`.

## Estructura
- `datos/datos_transporte_no_supervisado.csv`: dataset simulado de 72 viajes.
- `src/modelo_no_supervisado.py`: implementacion de K-Means.
- `documentos/Descripcion_Datos.pdf`: descripcion de la fuente y variables.
- `documentos/Pruebas_Modelo.pdf`: evidencias y resultados de prueba.
- `documentos/Guion_Video.pdf`: apoyo para el video de maximo 10 minutos.
- `Entrega_Actividad4.pdf`: plantilla del PDF final con enlaces.
- `integrantes.txt`: integrantes del equipo.

## Ejecucion
Desde la carpeta raiz del proyecto:

```powershell
py .\src\modelo_no_supervisado.py
```

Tambien puede usarse:

```powershell
python .\src\modelo_no_supervisado.py
```

## Resultado de referencia
Con el dataset incluido y semilla 42:
- Registros: 72
- Clusters: 3
- Iteraciones: 5
- Inercia: 112.6500
- Silhouette: 0.4188

## Git
El tutor debe agregarse como colaborador. El historial del repositorio debe evidenciar aportes reales de cada integrante del equipo.
