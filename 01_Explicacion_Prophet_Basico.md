# Clase 1 — Prophet básico: ventas de cerveza, vino y licor

Guía para exponer. El notebook es **`01_Prophet_Basico_BeerWineLiquor.ipynb`**.

**Datos:** `Data/BeerWineLiquor.csv`, **actualizados hasta agosto de 2026**. Es la misma serie que usa el video, descargada hoy de la Reserva Federal de EE. UU. (FRED, serie `MRTSSM4453USN`, fuente: Oficina del Censo de EE. UU.).

**Cómo ejecutarlo:** ábrelo en VS Code y dale a **Run All**. Ya está todo instalado y el notebook viene ejecutado. **Todos los números de esta guía son los que salieron de verdad.**

En cada paso tienes el código explicado, qué sale, cómo leerlo y **qué decir**.

---

## 0. Antes de empezar

### Diferencias con el video

| | Video | Yo |
|---|---|---|
| Datos | 1992 → **2018** (324 meses) | 1992 → **agosto 2026** (416 meses) |
| Librería | `fbprophet` | `prophet` (el nombre actual) |
| Pronóstico | 2019–2020 | **sep 2026 → ago 2028** |

> **Ojo:** los valores antiguos no son exactamente iguales a los del video. Por ejemplo, enero de 1992 es **1 414** y en el video era **1 509**. La Oficina del Censo **revisa y corrige** sus datos con el tiempo, así que la versión de hoy es la más precisa.

### ⚠️ Cambio en la librería

```python
from fbprophet import Prophet   # forma VIEJA (la del video). En una instalación nueva da error.
from prophet import Prophet     # forma ACTUAL.
```

**Qué decir:** "El video es de hace unos años. La librería se cambió de nombre: antes era `fbprophet` y ahora es `prophet`. Además actualicé los datos hasta 2026, así que vamos a ver también qué pasó en la pandemia."

### ¿Qué es Prophet? (para abrir)

Prophet es una librería creada por Facebook (hoy Meta) para **pronosticar series de tiempo**: datos medidos en el tiempo, como ventas por mes.

Prophet ve la serie como **una suma de piezas**:

```
Predicción = Tendencia + Estacionalidad + Ruido
```

- **Tendencia:** hacia dónde va la serie a largo plazo.
- **Estacionalidad:** patrones que se repiten. Aquí, cada diciembre se vende muchísimo más.
- **Ruido:** lo que no se puede explicar.

**Qué decir:** "Prophet piensa los datos como piezas de Lego: una pieza es la tendencia de fondo y otra es el patrón que se repite cada año. Las junta y así arma la predicción."

---

## Paso 1: Importar las librerías

```python
import pandas as pd
from prophet import Prophet         # en el video: from fbprophet import Prophet
```

- `pandas` sirve para manejar tablas (como un Excel dentro de Python). Apodo: `pd`.
- `Prophet` es el modelo. Va con **P mayúscula**.

**Qué sale:** nada. Si no hay error, está todo bien.

**Qué decir:** "Primero traemos las herramientas: pandas para la tabla y Prophet para pronosticar."

---

## Paso 2: Leer los datos

```python
df = pd.read_csv('Data/BeerWineLiquor.csv')
df.head()
```

- `read_csv` abre el archivo y lo guarda en `df` (*data frame*, una tabla).
- `head()` muestra las **primeras 5 filas**.

**Qué sale:**

|   | observation_date | MRTSSM4453USN |
|---|---|---|
| 0 | 1992-01-01 | 1414 |
| 1 | 1992-02-01 | 1444 |
| 2 | 1992-03-01 | 1496 |
| 3 | 1992-04-01 | 1569 |
| 4 | 1992-05-01 | 1707 |

**Cómo leerlo:**

- `observation_date` es la fecha. En el video la columna se llamaba `date` y el formato era `1/1/1992`. Ahora FRED la entrega como `1992-01-01`.
- `MRTSSM4453USN` es el **código** de la serie en FRED (en el video la columna se llamaba `beer`). Son las **ventas del mes en tiendas de cerveza, vino y licor de EE. UU., en millones de dólares**.

### ¿Por qué "un dato por mes, siempre el día 1"?

El valor de `1992-12-01` **no** es lo vendido el 1 de diciembre: es el **total de todo diciembre**. Se anota con el día 1 solo para marcar "este mes". Esto importa en tres momentos de la clase:

1. **Paso 8:** hay que decirle a Prophet `freq='MS'` (meses), porque él piensa en días por defecto.
2. **Paso 7:** Prophet apaga los patrones semanales y diarios, porque con un dato por mes no puede verlos.
3. **Paso 14:** en el gráfico anual **solo vale el día 1 de cada mes**.

**Qué decir:** "Son las ventas mensuales de las tiendas de cerveza, vino y licor de EE. UU., en millones de dólares, desde 1992 hasta agosto de 2026. Hay un dato por mes, fechado el día 1. Ojo: ese número no es lo que se vendió el día 1, es el total de todo el mes."

---

## Paso 3: Revisar los tipos de datos

```python
df.info()
```

**Qué sale:**

```
RangeIndex: 416 entries, 0 to 415
 #   Column            Non-Null Count  Dtype
 0   observation_date  416 non-null    str
 1   MRTSSM4453USN     416 non-null    int64
```

**Cómo leerlo:**

- **416 filas** = 416 meses (de enero 1992 a agosto 2026). En el video eran 324.
- **416 non-null**: no hay datos vacíos.
- La fecha es **`str`** (texto). En el video dice **`object`**, que es lo mismo en una versión vieja de pandas. **Python la ve como texto, no como fecha.**
- Las ventas son **`int64`**, números enteros. Esa está bien.

**Qué decir:** "Python cree que la fecha es solo texto, como si fuera un nombre. Prophet necesita una fecha de verdad. Eso lo arreglamos ahora."

---

## Paso 4: Renombrar las columnas a `ds` y `y` (obligatorio)

```python
df.columns = ['ds', 'y']
df.head()
```

Prophet **exige** estos nombres exactos:

| Nombre | Qué tiene que tener |
|---|---|
| `ds` | Las **fechas** |
| `y` | Lo que queremos **predecir** |

**¿Por qué?** El profesor explica que dentro de Prophet esos nombres están **escritos fijos** (*hard coded*) en muchos cálculos.

**Qué decir:** "Esta es una manía de Prophet: las columnas tienen que llamarse exactamente `ds` y `y`. Esto lo van a hacer siempre. Además nos sirve para quitar ese código feo, `MRTSSM4453USN`."

---

## Paso 5: Convertir la fecha en fecha de verdad

```python
df['ds'] = pd.to_datetime(df['ds'])
df.head()
```

`to_datetime` convierte el texto en una **fecha real** y la guarda otra vez en `ds`.

**Qué sale:** se ve igual (`1992-01-01`), porque el texto ya venía en ese formato. Lo que cambia es el **tipo**, y eso lo comprobamos en el paso 6.

**Lo que dice el profesor:** son datos de **inicio de mes** (*month start*). Prophet espera datos **diarios** por defecto, pero puede trabajar con mensuales si le indicamos la frecuencia (paso 8).

---

## Paso 6: Confirmar que quedó bien

```python
df.info()
```

```
 0   ds      416 non-null    datetime64[us]
 1   y       416 non-null    int64
```

**Cómo leerlo:** `ds` ya dice **`datetime64`**: es una **fecha**. El `[us]` es la precisión interna (en el video sale `[ns]`); no importa.

**Diferencia con ARIMA:** el **índice sigue siendo 0, 1, 2…**. La fecha **no** va de índice, es una columna más. Con *statsmodels* (ARIMA) hacíamos lo contrario.

**Qué decir:** "Listo: los datos tienen el formato de Prophet. Dos requisitos: columnas `ds` y `y`, y `ds` tipo fecha. Y a diferencia de ARIMA, la fecha queda como una columna normal."

---

## Paso 7: Crear y entrenar el modelo

```python
m = Prophet()
m.fit(df)
```

- `Prophet()` crea un modelo **en blanco** y lo guarda en `m`.
- `m.fit(df)` lo **entrena** con los 416 meses.

**Qué sale:**

```
cmdstanpy - INFO - Chain [1] start processing
cmdstanpy - INFO - Chain [1] done processing
<prophet.forecaster.Prophet at 0x...>
```

**Cómo leerlo:**

- **No son errores.** `cmdstanpy` es el motor matemático de Prophet. Solo avisa que empezó y terminó.
- En el video salen además `Disabling weekly seasonality` y `Disabling daily seasonality`: Prophet **no encontró patrones semanales ni diarios y los apagó**. Con un dato por mes, no hay semanas ni días que analizar. En la versión nueva a veces no se muestran, pero **igual los apaga**: lo comprobamos en el paso 14.

**Dato extra:** si sabes que hay un patrón semanal y Prophet no lo detecta, se puede forzar con `Prophet(weekly_seasonality=True)`.

**Qué decir:** "Con dos líneas ya entrenamos el modelo. Los mensajes no son errores. Hoy entrenamos con todos los datos; en la próxima clase evaluamos qué tan bueno es."

---

## Paso 8: Crear la tabla del futuro (24 meses)

```python
future = m.make_future_dataframe(periods=24, freq='MS')
future.head()
future.tail()
```

Crea una tabla **solo con fechas**: todas las que ya teníamos **más 24 meses nuevos**, vacíos. Es el **molde** que Prophet va a llenar.

| Parámetro | Qué significa |
|---|---|
| `periods=24` | 24 pasos hacia adelante. Cada paso es **una fila** = un mes. Son **2 años**. |
| `freq='MS'` | ***Month Start***, inicio de mes. Sin esto, Prophet crearía 24 **días**. |

**Qué sale:** `future.head()` empieza en **1992-01-01** y `future.tail()` termina en **2028-08-01**.

| | ds |
|---|---|
| 435 | 2028-04-01 |
| 436 | 2028-05-01 |
| 437 | 2028-06-01 |
| 438 | 2028-07-01 |
| 439 | 2028-08-01 |

**Comparación con ARIMA:** antes poníamos la frecuencia en el índice (`df.index.freq = 'MS'`). Con Prophet se pone **aquí**.

**Ejemplo simple:** es como un cuaderno donde anotaste las ventas hasta agosto de 2026 y ahora escribes las fechas de los próximos 24 meses con la casilla **vacía**, para que Prophet la llene.

**Qué decir:** "Esta tabla es el molde vacío del futuro. Ojo con `freq='MS'`: si se lo olvidan, Prophet predice días en vez de meses."

---

## Paso 9: Comprobar que se agregaron los 24 meses

```python
df.tail()        # los datos reales terminan en 2026-08-01
len(df)          # 416
len(future)      # 440
```

- `df.tail()` termina en **2026-08-01** (ventas de 5 637).
- `len(df)` = **416** y `len(future)` = **440**.

**416 + 24 = 440.** Cuadra. (En el video: 324 + 24 = 348.)

**Qué decir:** "Comprobamos con una suma: teníamos 416 meses, pedimos 24 más y la tabla del futuro tiene 440."

---

## Paso 10: Hacer la predicción

```python
forecast = m.predict(future)
forecast.head()
```

`predict` **llena la tabla** con las predicciones y las guarda en `forecast`. Predice también los meses pasados.

**Qué sale:** una tabla con **muchas** columnas (`ds`, `trend`, `yhat_lower`, `yhat_upper`, `trend_lower`, `trend_upper`, `additive_terms`, `yearly`, `multiplicative_terms`…, `yhat`).

**Las únicas que nos importan:**

| Columna | Qué significa |
|---|---|
| `ds` | La fecha |
| `yhat` | **La predicción** ("y sombrero") |
| `yhat_lower` | El **límite de abajo** (mínimo probable) |
| `yhat_upper` | El **límite de arriba** (máximo probable) |

Las demás son **piezas internas**: `trend` es la tendencia y `yearly` el efecto del mes. Prophet las **suma**: `yhat = trend + yearly`.

**Comprobación con la primera fila (enero de 1992):** trend **1 608** + yearly **−551** = yhat **1 057**. La suma cuadra. (El valor real fue 1 414.)

**Qué decir:** "Prophet devuelve muchas columnas, pero nos importan tres: `yhat`, la predicción, y `yhat_lower` y `yhat_upper`, el rango. Las demás son las piezas con las que armó la predicción."

---

## Paso 11: Ver solo las columnas importantes

```python
forecast.columns
forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']].tail(12)
```

- `forecast.columns` muestra todos los nombres.
- El **doble corchete** `[[ ]]` elige varias columnas: el de afuera significa "selecciona" y el de adentro es la **lista**.
- `.tail(12)` muestra los **últimos 12 meses** pronosticados.

**Qué sale (redondeado):**

| ds | yhat_lower | yhat_upper | yhat |
|---|---|---|---|
| 2027-09-01 | 5 926 | 6 562 | 6 235 |
| 2027-10-01 | 5 986 | 6 609 | 6 301 |
| 2027-11-01 | 6 137 | 6 739 | 6 442 |
| **2027-12-01** | 7 290 | 7 886 | **7 585** ← el pico |
| 2028-01-01 | 5 557 | 6 178 | **5 866** ← el pozo |
| 2028-02-01 | 5 637 | 6 263 | 5 942 |
| 2028-03-01 | 5 953 | 6 558 | 6 263 |
| 2028-04-01 | 5 890 | 6 520 | 6 213 |
| 2028-05-01 | 6 262 | 6 867 | 6 560 |
| 2028-06-01 | 6 168 | 6 814 | 6 500 |
| 2028-07-01 | 6 350 | 6 963 | 6 644 |
| 2028-08-01 | 6 218 | 6 863 | 6 540 |

**Cómo leer una fila:** "Para diciembre de 2027, Prophet predice ventas de **7 585 millones de dólares**. Lo más probable es que el valor real quede **entre 7 290 y 7 886**."

**Qué decir:** "Esta es la tabla que le llevarías a tu jefe: para cada mes, un número y un rango. Diciembre salta a 7 585, muy por encima del resto del año, y enero es el más bajo."

---

## Paso 12: Graficar con Prophet

```python
%matplotlib inline
m.plot(forecast);
```

- `%matplotlib inline` hace que el gráfico aparezca **dentro** del notebook. El profesor dice: si no ves el gráfico, falta esta línea.
- `m.plot(forecast)` hace que Prophet dibuje solo.
- El `;` evita que el gráfico salga **dos veces**.

**Qué se ve:**

| Elemento | Qué es |
|---|---|
| **Puntos negros** | Las ventas **reales** |
| **Línea azul** | La **predicción** (`yhat`) |
| **Sombra celeste** | El **rango** |
| **Donde se acaban los puntos negros** | El **pronóstico** (sep 2026 → ago 2028) |

**Qué mirar:**

- Las ventas **suben con los años** (tendencia).
- Hay un **pico enorme cada diciembre** (estacionalidad).
- En **2020** hay un **salto**: las ventas suben de golpe. Lo vemos en el zoom.

**Qué decir:** "Los puntos negros son la realidad y la línea azul es lo que Prophet entendió. Donde se acaban los puntos negros, la línea sigue sola: ese es el pronóstico hasta 2028."

---

## Paso 13: Zoom al gráfico (2019–2029)

```python
import matplotlib.pyplot as plt

m.plot(forecast)
plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2029-01-01'));
```

`plt.xlim(inicio, fin)` pone **límites al eje horizontal**: es una **lupa**.

> **Diferencias con el video:** el profesor hace zoom de 2014 a 2021. Yo lo hago de **2019 a 2029** para ver la pandemia y el pronóstico nuevo. Además, él escribe las fechas como texto (`'2014-01-01'`), que en versiones nuevas puede fallar. Con `pd.Timestamp(...)` funciona siempre.

**Qué se ve (y qué contar):**

1. **La pandemia hizo subir las ventas de alcohol.** En **marzo de 2020** se vendieron **5 152**, contra 4 242 en febrero: **+21 % en un mes**. Con los bares cerrados, la gente compró para tomar en casa. El promedio mensual pasó de **4 669 en 2019** a **5 476 en 2020** (+17 %) y siguió subiendo hasta **5 946 en 2023**.
2. **Prophet se queda corto en los diciembres de la pandemia.** En diciembre de 2020 se vendieron **7 265** y el modelo esperaba **6 411**. En 2021: 7 652 contra 6 587. Los puntos negros quedan **por encima de la sombra**.
3. **Desde 2024 las ventas bajan:** promedio mensual de 5 810 en 2024 y 5 649 en 2025. Pero **el modelo sigue subiendo**, y en 2025–2026 muchos puntos negros quedan **por debajo** de la línea azul. Prophet todavía no "cree" que la tendencia cambió.
4. El pronóstico (2027–2028) repite el patrón: **pico en diciembre** y **pozo en enero y febrero**.

**Lo que dice el profesor sobre la sombra:** en su versión se veía más angosta en los picos. En la versión actual la sombra mide **más o menos lo mismo todo el año** (unos 620 de ancho).

**Limitación de `m.plot()`:** no acepta cosas como `figsize`. Para eso se grafica a mano (paso 13b).

**Qué decir:** "Con zoom se ve algo interesante que el video no tiene: en 2020, con la pandemia, la venta de alcohol saltó un 21 % en un solo mes, porque la gente tomaba en casa. Y desde 2024 está bajando, pero el modelo todavía no se dio cuenta: sigue proyectando una subida. Esto muestra un límite de Prophet: tarda en reaccionar a cambios recientes."

---

## Paso 13b: Graficar por tu cuenta (sin la función de Prophet)

```python
forecast.plot(x='ds', y='yhat', figsize=(8, 10));
```

- `forecast` es una **tabla normal de pandas**, así que se grafica sin `m.plot()`.
- `x='ds'`: fechas. `y='yhat'`: predicción.
- `figsize=(8, 10)`: tamaño (ancho, alto). **Esto no se puede con `m.plot()`**.

Sin el `;` aparece arriba un texto como `<Axes: xlabel='ds'>`: no es un error, es Python diciendo "aquí creé un gráfico".

| | `m.plot(forecast)` | `forecast.plot(...)` |
|---|---|---|
| Puntos negros (datos reales) | ✅ | ❌ |
| Sombra celeste (rango) | ✅ | ❌ |
| Cambiar tamaño, colores, etc. | ❌ Difícil | ✅ Fácil |

**Ejemplo simple:** `m.plot()` es una foto con **filtro automático**: bonita, pero casi no se edita. `forecast.plot()` es la foto **sin filtro**: más simple, pero la ajustas como quieras.

**Qué decir:** "Como `forecast` es una tabla normal, la podemos graficar nosotros. Perdemos los puntos y la sombra, pero ganamos control. Esto ya es más tema de matplotlib que de Prophet."

---

## Paso 14: Ver las piezas del modelo

```python
m.plot_components(forecast);
```

Separa la predicción en las **piezas que sumó**. Salen **2 gráficos**, porque Prophet apagó los patrones semanal y diario.

### Gráfico 1: Tendencia (`trend`)

**Qué se ve:** sube de unos **1 608** en 1992 a unos **6 187** en agosto 2026, y a **6 512** en agosto 2028. **Casi se cuadruplicó.**

**Cómo leerlo:**

- Es el **"camino de fondo"**: lo que queda si borramos los picos de cada año.
- Sube despacio en los 90, más rápido desde 1998, **todavía más rápido de 2015 a 2020**, y **un poco más lento desde 2020**. Esos **dobleces** son los **puntos de cambio** (clase 3).

**Qué decir:** "Si borramos los picos de cada año, las ventas casi se cuadruplicaron en 34 años. Fíjense que en 2020 la tendencia se dobla: el modelo empieza a notar que el crecimiento ya no es tan rápido."

### Gráfico 2: Estacionalidad anual (`yearly`)

Muestra cómo se comporta un **año típico**: **cuánto se suma o se resta a la tendencia** en cada época.

### ⚠️ Cuidado al leer este gráfico

La curva tiene muchas ondas, y **no todo es real**. Se ve una caída de casi **−1 450** a mediados de febrero y un pico de **+1 350** a mediados de noviembre. **Esos extremos no son reales.**

¿Por qué? Hay **un dato por mes, siempre el día 1**. Prophet dibuja una curva para todos los días del año, pero **solo el día 1 de cada mes tiene datos detrás**. Entre un día 1 y el siguiente, la curva se mueve sin control.

**Regla:** mira **solo el valor en el día 1 de cada mes**.

**Valores reales del efecto anual (día 1 de cada mes):**

| Mes | Efecto | Interpretación |
|---|---|---|
| Enero | −540 | Bajo (después de las fiestas) |
| Febrero | −521 | Bajo (y es el mes más corto) |
| Marzo | −250 | Algo bajo |
| Abril | −226 | Algo bajo |
| Mayo | +54 | Normal |
| Junio | +26 | Normal |
| Julio | +156 | Un poco alto (verano, 4 de julio) |
| Agosto | +17 | Normal |
| Septiembre | −128 | Algo bajo |
| Octubre | −75 | Normal |
| Noviembre | +52 | Normal (Acción de Gracias) |
| **Diciembre** | **+1 182** | **El gran pico: Navidad y Año Nuevo** |

**Qué se ve en simple:** **diciembre es, por lejos, el mes fuerte**; **enero y febrero son los más flojos**; el resto del año está cerca de 0.

### Cómo se juntan las dos piezas (el ejemplo clave)

```
Predicción = tendencia + estacionalidad anual
```

Con los **números reales** de 2027:

| Mes | Tendencia | + Efecto anual | = Predicción |
|---|---|---|---|
| **Diciembre 2027** | 6 403 | +1 182 | **7 585** (el pico) |
| **Enero 2027** | 6 255 | −540 | **5 715** (el pozo) |
| Junio 2027 | 6 322 | +26 | 6 348 (un mes "normal") |

**Qué decir:** "Esto conecta con la teoría: Prophet es una suma de piezas. El primer gráfico dice 'las ventas crecen cada año'. El segundo dice 'en diciembre se vende unos 1 200 millones más que en un mes normal, y en enero unos 540 menos'. Si sumamos las dos, sale la predicción. Ojo: en el gráfico anual solo valen los puntos del día 1, porque hay un dato por mes."

---

## Cierre

**Frase del profesor, adaptada:** "Prophet es muy poderoso y muy fácil: prepara, entrena, predice y grafica en pocas líneas. La desventaja es que es un poco una **caja negra**: no entendemos cada detalle por dentro como con ARIMA. El profesor dice que está 'a medio camino entre ARIMA y una red neuronal recurrente'. Y con los datos actualizados vimos además su límite: no puede prever una pandemia y tarda en notar cuando una tendencia cambia, como la baja de ventas desde 2024. En la próxima clase vamos a medir qué tan buenas son sus predicciones."

### Los pasos que hay que recordar

```python
df.columns = ['ds', 'y']                                   # 1. nombres obligatorios
df['ds'] = pd.to_datetime(df['ds'])                        # 2. fecha real
m = Prophet()                                              # 3. crear
m.fit(df)                                                  # 4. entrenar
future = m.make_future_dataframe(periods=24, freq='MS')    # 5. molde del futuro
forecast = m.predict(future)                               # 6. predecir
m.plot(forecast)                                           # 7. graficar
m.plot_components(forecast)                                # 8. ver las piezas
```

---

## Posibles preguntas del público

**"¿Por qué subieron las ventas en la pandemia?"**
Con los bares y restaurantes cerrados, la gente compró alcohol en tiendas para tomar en casa. Estos datos son solo de **tiendas** de cerveza, vino y licor, así que esas compras aparecen aquí.

**"¿Por qué el modelo sigue subiendo si las ventas bajan desde 2024?"**
Prophet cambia la tendencia con cuidado, y por defecto **no busca cambios en el último 20 % de los datos** (lo vemos en la clase 3). La baja reciente todavía es corta para que el modelo la tome como un cambio de tendencia.

**"¿Entonces el pronóstico hasta 2028 está bien?"**
Probablemente sale **alto**. Lo medí: si escondemos el último año (sep 2025 → ago 2026) y se lo pedimos al modelo, **se equivoca un 14 %**, prediciendo de más unos 760 millones por mes, porque las ventas están bajando y el modelo sigue proyectando subida. Ni siquiera el modo multiplicativo de la clase 4 lo arregla (13.7 %), porque el problema es la tendencia, no la estacionalidad. Por eso hace falta **evaluar** el modelo antes de confiar en él: es el tema de la clase 2.

**"¿Qué tan bien ajusta en general?"**
Bien para un modelo básico: en toda la historia se equivoca un **4.9 %** en promedio, y el **82 %** de los meses reales cae dentro de la sombra (promete 80 %). Donde más falla es en los diciembres (7.8 % de error contra 4.6 % del resto de los meses).

**"¿Por qué se queda corto en diciembre?"**
Porque el pico de diciembre **crece con los años** y el modelo aditivo suma **siempre lo mismo**. En la clase 4 vemos la solución: `seasonality_mode='multiplicative'`.

**"¿Qué diferencia hay con ARIMA?"**
Con ARIMA hay que revisar estacionariedad, elegir (p, d, q) y poner la fecha como índice. Prophet hace casi todo solo, pero es más **caja negra**.
