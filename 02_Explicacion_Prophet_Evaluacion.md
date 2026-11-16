# Clase 2 — Evaluación de Prophet: ¿qué tan bueno es el modelo?

Guía para exponer. El notebook es **`02_Prophet_Evaluacion_MilesTraveled.ipynb`**.

**Datos:** `Data/Miles_Traveled.csv`, **actualizados hasta agosto de 2026** (680 meses). Es la misma serie del video, descargada hoy de la Reserva Federal (FRED, serie `TRFVOLUSM227NFWA`).

**Cómo ejecutarlo:** ábrelo en VS Code y dale a **Run All**. Ya está todo instalado y el notebook viene ejecutado. **Todos los números de esta guía son los que salieron de verdad.**

---

## 0. La idea de esta clase

En la clase anterior entrenamos Prophet con **todos** los datos y pronosticamos el futuro. Quedó una pregunta sin responder: **¿el pronóstico es bueno?** No lo sabíamos, porque el futuro todavía no pasó.

La solución es **hacer como si estuviéramos en el pasado**: escondemos datos que sí conocemos, le pedimos al modelo que los prediga y comparamos. Lo hacemos de dos formas:

1. **La forma clásica:** esconder el último año, predecirlo y calcular el **RMSE**. Es lo mismo que hacíamos con ARIMA.
2. **La herramienta de Prophet:** la **validación cruzada** (*cross validation*): **varios exámenes** en distintos momentos de la historia.

**Qué decir para abrir:** "La clase pasada Prophet nos dio un pronóstico, pero un pronóstico sin evaluación es una apuesta. Hoy lo ponemos a prueba: primero con un examen final, el último año, y después con diez exámenes repartidos a lo largo de 50 años."

### Diferencias con el video

| | Video | Yo |
|---|---|---|
| Datos | 1970 → 2018 (588 meses) | 1970 → **agosto 2026** (680 meses) |
| Año de prueba | 2018 | **sep 2025 → ago 2026** |
| Exámenes de validación cruzada | 9 (108 filas) | **10 (120 filas)** |
| Librería | `fbprophet` | `prophet` |

> Lo más interesante de tener datos actualizados: **incluyen el COVID**, y lo vamos a ver aparecer en la validación cruzada.

---

## Paso 1: Importar y leer los datos

```python
import pandas as pd
from prophet import Prophet          # en el video: from fbprophet import Prophet
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('Data/Miles_Traveled.csv')
df.info()
```

**Qué sale:**

```
RangeIndex: 680 entries, 0 to 679
 #   Column            Non-Null Count  Dtype
 0   observation_date  680 non-null    str
 1   TRFVOLUSM227NFWA  680 non-null    int64
```

**Cómo leerlo:**

- **680 filas** = 680 meses, de enero 1970 a agosto 2026.
- La fecha es **`str`** (texto). En el video dice `object`; es lo mismo.
- `TRFVOLUSM227NFWA` es el **código de la Reserva Federal** para el **volumen de tráfico**: cuántas millas recorrieron en total los vehículos en EE. UU. ese mes, **en millones**. Es lo que vamos a predecir.

**Lo que dice el profesor:** con Prophet no hace falta convertir las fechas al leer el CSV, porque igual vamos a renombrar las columnas y convertir la fecha a mano.

**Qué decir:** "Ahora usamos otro dataset: millas recorridas por mes en EE. UU., desde 1970 hasta 2026. El nombre de la columna es un código feo de la Reserva Federal, pero lo vamos a renombrar."

---

## Paso 2: Preparar el formato que pide Prophet

```python
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df.head()
```

**Qué sale:**

|   | ds | y |
|---|---|---|
| 0 | 1970-01-01 | 80173 |
| 1 | 1970-02-01 | 77442 |
| 2 | 1970-03-01 | 90223 |

**Qué decir:** "Los dos pasos obligatorios de siempre: columnas `ds` y `y`, y la fecha convertida a fecha."

---

## Paso 3: Mirar los datos

```python
df.plot(x='ds', y='y');
```

**Qué se ve:**

- Una línea que sube de unos **80 000** en 1970 a unos **295 000** en 2026, con un **zigzag cada año** (más tráfico en verano, menos en invierno).
- Alrededor de **2008–2013** se aplana: la crisis financiera y la gasolina cara.
- En **abril de 2020** hay un **pozo enorme**: 167 617, casi **40 % menos** que en abril de 2019. Es el **COVID**, con las cuarentenas.

**Qué decir:** "Antes de modelar siempre miramos los datos. Hay una tendencia clara hacia arriba y un patrón que se repite cada año, que es justo lo que Prophet sabe capturar. Acuérdense del pozo de 2020, porque va a volver a aparecer."

---

## Paso 4: Separar entrenamiento y prueba

```python
len(df)                   # 680

train = df.iloc[:668]     # 680 - 12 = 668   (en el video: 588 - 12 = 576)
test  = df.iloc[668:]     # los últimos 12 meses (sep 2025 -> ago 2026)
```

- `iloc` corta la tabla **por posición de fila**.
- `[:668]` va desde el inicio **hasta la fila 667**. El 668 **no** entra. Es el **entrenamiento**.
- `[668:]` va desde la fila 668 **hasta el final**. Es la **prueba**.
- **680 − 12 = 668**: la prueba son los **últimos 12 meses**, de **septiembre 2025 a agosto 2026**.

**¿Por qué 12 meses?** Porque es **un ciclo completo**: un verano, un invierno, todo.

**Ejemplo simple:** es como estudiar con los exámenes de años anteriores (entrenamiento) y después rendir el examen de este año, que nunca viste (prueba). Si te va bien en el nuevo, aprendiste de verdad.

**Qué decir:** "Escondemos el último año. El modelo aprende sin verlo, y después comparamos lo que predice con lo que realmente pasó."

---

## Paso 5: Entrenar y predecir (las 4 líneas clave)

```python
m = Prophet()
m.fit(train)                                               # ¡solo con train!
future = m.make_future_dataframe(periods=12, freq='MS')    # 12 = tamaño de test
forecast = m.predict(future)
```

| Línea | Qué hace |
|---|---|
| `m = Prophet()` | Crea el modelo. |
| `m.fit(train)` | Lo entrena **solo con `train`**. Ya **no** con todo `df`. |
| `make_future_dataframe(periods=12, freq='MS')` | Crea las fechas a predecir: **12 meses**, el tamaño de `test`. Con datos diarios no haría falta `freq`. |
| `m.predict(future)` | Predice. |

**Lo que dice el profesor:** "estas cuatro líneas son, en el fondo, todo Prophet". Y un consejo: **cuando ya estés conforme con la evaluación, vuelves a entrenar con todos los datos** y ahí sí pronosticas el futuro real, como en la clase anterior.

**Qué decir:** "Crear, entrenar, preparar el futuro y predecir: cuatro líneas. Pedimos 12 periodos porque queremos comparar justo con los 12 meses que escondimos."

---

## Paso 6: Ver las predicciones

```python
forecast.tail()
```

**Qué sale:** las últimas filas, de **2026-04-01 a 2026-08-01**, con `trend`, `yhat_lower`, `yhat_upper`, etc. Al final de la tabla está **`yhat`**, la predicción.

**Qué decir:** "Las últimas fechas son de 2026, justo el año que escondimos. Esas filas son las que vamos a comparar."

---

## Paso 7: Graficar predicción vs. realidad

```python
ax = forecast.plot(x='ds', y='yhat', label='Predictions', legend=True, figsize=(12, 8))
test.plot(x='ds', y='y', label='True Test Data', legend=True, ax=ax,
          xlim=(pd.Timestamp('2025-09-01'), pd.Timestamp('2026-08-01')));
```

**Primera línea**, la predicción: grafica `yhat` y **guarda el gráfico en `ax`**. `label` es el nombre en la leyenda, `legend=True` la muestra y `figsize=(12, 8)` es el tamaño.

**Segunda línea**, la realidad: grafica los datos reales de `test`.

- **`ax=ax` es la clave:** "dibuja **en el mismo gráfico** de antes". Sin esto salen dos gráficos separados.
- `xlim` hace zoom solo al año de prueba (sep 2025 → ago 2026). El profesor hace lo mismo con 2018: mira las fechas de `test` y usa su inicio y su fin.

> En el video escribe las fechas como texto: `xlim=('2018-01-01', '2019-01-01')`. Con versiones nuevas puede fallar; con `pd.Timestamp(...)` funciona siempre.

**Qué se ve:** la línea azul (predicción) y la naranja (realidad) van **casi pegadas**. Bajan juntas hasta **febrero** y suben juntas hacia **julio y agosto**. La azul queda **un poco por debajo** en varios meses: el tráfico real creció un poco más de lo esperado.

**Qué decir:** "Ponemos predicción y realidad en el mismo gráfico. Van casi juntas: el modelo acertó la forma del año, el pozo de febrero y el pico del verano."

---

## Paso 8: Medir el error con números (RMSE)

```python
from statsmodels.tools.eval_measures import rmse

predictions = forecast.iloc[-12:]['yhat']
test['y']

rmse(predictions, test['y'])     # 5895.43
test.mean()                      # y: 277974.58
```

- Importa `rmse`, la **raíz del error cuadrático medio**, la misma que usamos con ARIMA.
- `forecast.iloc[-12:]['yhat']` toma **las últimas 12 predicciones**. `-12:` significa "las últimas 12 filas".
- `test['y']` son los 12 valores reales.
- Las dos tienen **los mismos números de fila (668 a 679)**, así que se comparan mes a mes.

> ⚠️ En el video el profesor primero escribe `rmse(predictions, test)` y lo corrige. Hay que pasar **solo la columna** `test['y']`.

**Comparación mes a mes:**

| Mes | Real | Predicho | Diferencia |
|---|---|---|---|
| sep 2025 | 279 611 | 272 999 | +6 612 |
| oct 2025 | 291 540 | 280 671 | +10 869 |
| nov 2025 | 263 138 | 265 967 | −2 829 |
| dic 2025 | 265 855 | 267 698 | −1 843 |
| ene 2026 | 252 805 | 255 625 | −2 820 |
| feb 2026 | 244 934 | 246 751 | −1 817 |
| mar 2026 | 283 246 | 274 626 | +8 620 |
| abr 2026 | 280 116 | 273 910 | +6 206 |
| may 2026 | 293 169 | 285 174 | +7 995 |
| jun 2026 | 289 387 | 285 095 | +4 292 |
| jul 2026 | 297 770 | 291 910 | +5 860 |
| ago 2026 | 294 124 | 292 372 | +1 752 |

**Resultados:**

| | Mis datos | Video |
|---|---|---|
| **RMSE** | **5 895** | 8 750 |
| Promedio de `test` | 277 975 | 268 740 |
| Error en % | **≈ 2.1 %** | ≈ 3.3 % |

> `test.mean()` también muestra un promedio de las **fechas** (ds = 2026-02-14), porque calcula el promedio de todas las columnas. Ignóralo: lo que importa es `y`.

**¿Qué es el RMSE, en simple?** Es **"el tamaño típico del error"**, en las mismas unidades que los datos. Para cada mes se resta real − predicho, se eleva al cuadrado (así los negativos no cancelan a los positivos), se promedia y se le saca la raíz.

**Ejemplo simple:** si alguien pesa 70 kg y le adivinas con un error de 1.5 kg, adivinaste muy bien. Aquí: 5 895 parece mucho, pero sobre 278 000 es un 2 %.

**Qué decir:** "El RMSE solo no dice nada. Hay que compararlo con el tamaño de los datos. Un error de 5 900 sobre valores de 278 000 es un 2 %: muy bueno, y confirma lo que vimos en el gráfico. Incluso mejor que en el video, que daba un 3 %."

---

## Paso 9: La validación cruzada (la parte del link)

Hasta ahora probamos el modelo **una sola vez**. Pero **un solo examen puede engañar**: quizás justo ese año fue fácil. Prophet trae una herramienta para **probarlo muchas veces, en distintas partes de la historia**: la **validación cruzada**.

### Lo que muestra el profesor en la documentación

En este momento del video, el profesor sale del notebook y abre la **página oficial de Prophet sobre diagnósticos**:

🔗 **https://facebook.github.io/prophet/docs/diagnostics.html**

Puedes abrir ese link en la exposición. Esto es lo que hay que explicar:

**1. El dibujo de la documentación.** Muestra una serie de tiempo (las visitas diarias a la página de Wikipedia del jugador de fútbol americano Peyton Manning) con:

- Una zona de datos al principio para entrenar: el **initial**.
- Una **línea vertical punteada**: el **cutoff** o corte. Hasta ahí el modelo puede "ver".
- Una zona **azul** después del corte: el **horizon**, lo que el modelo predice y luego se compara con la realidad.
- Datos al final que en ese intento **no se usan** (*unused data*, como dice el profesor).

Después, el corte **se mueve** y se repite: entrenar, cortar, predecir, comparar. En el dibujo se entrena con 5 años y se predice 1 año, justo lo que vamos a usar.

**2. Cómo lo llama la documentación:** ***simulated historical forecasts*** (pronósticos históricos simulados). Se **simula** que estamos en distintos momentos del pasado, se hace el pronóstico que habríamos hecho y se ve cómo nos habría ido.

**3. El paper de Prophet.** La documentación tiene un link al **artículo científico de Prophet**: *"Forecasting at Scale"*, de Sean J. Taylor y Benjamin Letham (Facebook, 2017). El profesor dice que si te interesa el detalle lo puedes leer: "son solo 25 páginas, no es tan terrible".

**4. Un dato de la documentación:** si no defines los parámetros, Prophet usa por defecto un *initial* de **3 veces el horizonte** y hace un corte cada **medio horizonte**.

### Los 3 parámetros

| Parámetro | Qué es | En esta clase |
|---|---|---|
| **initial** | Cuántos datos como mínimo para el **primer** entrenamiento | 5 años |
| **period** | **Cada cuánto** se mueve el corte | Cada 5 años |
| **horizon** | **Cuánto predice** después de cada corte | 1 año |

**Ejemplo simple:** en vez de dar un solo examen final, el alumno da **varios exámenes** a lo largo del año. Si le va bien en todos, sabe de verdad.

**Qué decir:** "Un solo examen puede ser suerte. La validación cruzada hace que el modelo rinda varios exámenes en distintos momentos del pasado. En cada uno solo ve los datos hasta el corte, predice el año siguiente y comparamos. La documentación lo explica con este dibujo, y si quieren el detalle matemático, está el paper original de Prophet."

---

## Paso 10: Importar las herramientas y definir los parámetros

```python
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
```

| Herramienta | Para qué sirve |
|---|---|
| `cross_validation` | Hace los "exámenes" |
| `performance_metrics` | Calcula los errores |
| `plot_cross_validation_metric` | Grafica los errores |

```python
# INITIAL
initial = 5 * 365
initial = str(initial) + ' days'
initial                       # '1825 days'

# PERIOD
period = 5 * 365
period = str(period) + ' days'

# HORIZON
horizon = 365
horizon = str(horizon) + ' days'
```

- `5 * 365` = **1825 días** = 5 años.
- `str(...)` convierte el número en **texto**.
- `+ ' days'` le pega " days". **El espacio es obligatorio:** `'1825 days'` funciona y `'1825days'` no. El profesor: "tiene que ser exactamente este formato, si no, no funciona".

### ¿Por qué este formato? (`pd.Timedelta`)

```python
pd.Timedelta(initial)         # Timedelta('1825 days 00:00:00')
```

El profesor muestra la documentación de **`pd.Timedelta`**. Un *Timedelta* es la forma en que pandas representa una **duración**, no una fecha. Prophet pide los parámetros como **texto que pandas entienda como duración**, en **días**. Agregué esa celda para **demostrarlo**: pandas lee `'1825 days'` sin problema.

**Qué decir:** "Otra manía de Prophet: no le puedes decir '5 años'. Quiere el texto exacto en días. Por eso multiplicamos por 365 y le pegamos ' days'."

---

## Paso 11: Ejecutar la validación cruzada

```python
df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)
```

**Qué hace, explicado bien:** `cross_validation` **no usa el modelo ya entrenado tal cual**. Toma su configuración y los datos con los que se entrenó (`train`), y **en cada corte entrena un modelo nuevo** con **solo los datos anteriores al corte**. Así el examen es honesto.

**Qué sale:** una **barra de progreso** `0/10 → 10/10` (**10 exámenes**) y 10 pares de mensajes `cmdstanpy`, uno por modelo. No son errores.

> En el video sale `Making 9 forecasts with cutoffs between 1976-12-11 and 2016-12-01` y varios `FutureWarning`, que el profesor dice que se ignoran ("cosas que cambió numpy"). Yo tengo **10** porque mis datos tienen 8 años más.

**¿Cómo elige los cortes?** **Cuenta hacia atrás desde el final:**

1. `train` termina en **agosto 2025**.
2. Al último corte le tiene que quedar **1 año** para predecir, así que es **agosto 2024**.
3. Desde ahí va hacia atrás **de 5 en 5 años**: 2019, 2014, 2009… hasta **1979**.
4. No hace un corte en 1974, porque antes no habría 5 años de datos (el *initial*).

**Qué decir:** "Prophet hace 10 exámenes solo. En cada uno entrena un modelo con los datos hasta el corte, predice el año siguiente y lo compara con lo que pasó."

---

## Paso 12: Ver la tabla de la validación cruzada

```python
df_cv.head()
```

|   | ds | yhat | yhat_lower | yhat_upper | y | cutoff |
|---|---|---|---|---|---|---|
| 0 | 1979-09-01 | 132 468 | 129 821 | 135 339 | 127 576 | 1979-08-13 |
| 1 | 1979-10-01 | 134 445 | 131 478 | 137 612 | 131 619 | 1979-08-13 |
| 2 | 1979-11-01 | 126 171 | 122 962 | 128 852 | 123 381 | 1979-08-13 |
| 3 | 1979-12-01 | 125 927 | 122 910 | 129 004 | 123 226 | 1979-08-13 |
| 4 | 1980-01-01 | 120 624 | 117 560 | 123 668 | 116 462 | 1979-08-13 |

| Columna | Qué significa |
|---|---|
| `ds` | La fecha que se predijo |
| `yhat` | Lo que predijo Prophet |
| `yhat_lower` / `yhat_upper` | El rango |
| `y` | **El valor real** |
| `cutoff` | El corte: hasta qué fecha "vio" el modelo |

**Cómo leer la primera fila:** "Prophet entrenó solo con datos hasta el 13 de agosto de 1979. Con eso predijo **132 468** para septiembre de 1979. En realidad hubo **127 576**."

**¿Por qué un día raro como el 13 de agosto?** Porque Prophet va hacia atrás exactamente **1825 días** cada vez, y 5 años no son exactamente 1825 días (hay años bisiestos). Por eso cada corte se corre un día o dos. No afecta en nada.

```python
len(df_cv)                    # 120
df_cv['cutoff'].unique()      # 1979, 1984, 1989, 1994, 1999, 2004, 2009, 2014, 2019, 2024
```

**10 cortes × 12 meses = 120 filas.** (En el video: 9 × 12 = 108.)

**Qué decir:** "Son 10 exámenes de 12 preguntas cada uno: 120 predicciones para comparar con la realidad."

---

## Paso 13 (extra mío): Dibujar los cortes sobre los datos

> No está en el video. Es **mi versión del dibujo de la documentación**, con nuestros datos.

```python
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train['ds'], train['y'], color='gray', lw=1, label='Datos de entrenamiento')
for i, c in enumerate(df_cv['cutoff'].unique()):
    c = pd.Timestamp(c)
    ax.axvline(c, color='red', ls='--', lw=1, label='Corte (cutoff)' if i == 0 else None)
    ax.axvspan(c, c + pd.Timedelta(horizon), color='tab:blue', alpha=0.25,
               label='Horizonte (1 año predicho)' if i == 0 else None)
ax.set_title('Validación cruzada: dónde corta y qué predice Prophet')
ax.legend();
```

**Qué hace (en simple):** dibuja los datos en gris. Para cada uno de los 10 cortes pone una **línea roja punteada** (`axvline`) y pinta de **azul** el año siguiente (`axvspan`), que es lo que el modelo tuvo que predecir.

**Qué se ve:** 10 líneas rojas cada 5 años, cada una seguida de una franja azul. **La franja de 2019–2020 cae justo sobre el pozo del COVID.**

**Qué decir:** "Este es el mismo dibujo de la documentación, con nuestros datos. Cada línea roja es un examen. Miren la franja de 2019: al modelo le tocó predecir, sin saberlo, el año del COVID."

---

## Paso 14: Las métricas de error

```python
performance_metrics(df_cv)
```

Toma las 120 predicciones y calcula **qué tan equivocado está el modelo según qué tan lejos predice**.

**Qué sale (primeras y últimas filas de 109):**

| horizon | rmse | mae | mape | mdape | smape | coverage |
|---|---|---|---|---|---|---|
| 51 days | 4 708 | 3 380 | 0.0165 | 0.0137 | 0.0163 | 0.75 |
| 52 days | 4 512 | 3 098 | 0.0140 | 0.0109 | 0.0139 | 0.83 |
| 53 days | 4 540 | 3 191 | 0.0143 | 0.0116 | 0.0142 | 0.83 |
| … | … | … | … | … | … | … |
| 364 days | 12 237 | 9 634 | 0.0402 | 0.0304 | 0.0394 | 0.33 |
| 365 days | 11 679 | 9 003 | 0.0378 | 0.0272 | 0.0369 | 0.42 |

(Quité `mse` de la tabla: es el error al cuadrado y sale enorme, por ejemplo 2.2e+07 = 22 000 000.)

| Columna | Nombre completo | Qué significa |
|---|---|---|
| `horizon` | Horizonte | Cuántos días después del corte se predice |
| `mse` | Error cuadrático medio | El error al cuadrado. Difícil de interpretar |
| `rmse` | Raíz del error cuadrático medio | El del paso 8, en millas |
| `mae` | Error absoluto medio | En promedio, cuánto se equivoca, hacia arriba o abajo |
| **`mape`** | **Error porcentual absoluto medio** | **El error en %. El más fácil** |
| `mdape` | Mediana del error porcentual | Como el MAPE, pero le afectan menos los casos extremos *(nueva, no está en el video)* |
| `smape` | MAPE simétrico | Variante del MAPE *(nueva)* |
| `coverage` | Cobertura | Qué proporción de valores reales cayó **dentro** del rango *(el profesor no la explica)* |

**Cómo leer el MAPE:** 0.0165 × 100 = **1.65 %**: a 51 días, el modelo se equivoca en promedio un 1.65 %.

**Cómo leer el coverage:** 0.75 = el **75 %** de los valores reales cayó dentro del rango. A 365 días baja a 42 %.

**¿Por qué empieza en 51 días?** Por la **ventana móvil**: Prophet **promedia grupos de predicciones vecinas**, de un 10 % de las filas cada uno (12 de 120). El primer grupo junta las 12 más cercanas al corte, y la última está a 51 días.

**¿Por qué los días saltan (51, 52, 53, 54, 56…)?** Porque los datos son mensuales: depende de cuántos días hay entre el corte y el día 1 de cada mes.

**Lo que dice el profesor:** la tabla en crudo **es difícil de leer**; mejor graficarla. (Después pone `# performance_metrics(df_cv)` solo para que no se muestre.)

**Qué decir:** "Esta tabla dice qué tan bien predice el modelo según qué tan lejos mira. La columna más fácil es `mape`, el error en porcentaje: a dos meses nos equivocamos un 1.5 %, y a un año, un 4 %. Pero en crudo es difícil de leer, así que la graficamos."

---

## Paso 15: Graficar el error RMSE

```python
plot_cross_validation_metric(df_cv, metric='rmse');
```

Dibuja el error según el horizonte. El `;` evita que salga **dos veces** (el profesor lo muestra).

| Elemento | Qué es |
|---|---|
| **Eje horizontal** | Cuántos días hacia adelante se predijo (0 a 365) |
| **Eje vertical** | Qué tan grande fue el error |
| **Puntos grises** | El error de **cada predicción** (120 puntos) |
| **Línea azul** | El **promedio móvil** del error |

**Qué se ve:**

- De **0 a unos 200 días**, la línea azul está **baja y estable**, entre unos **4 000 y 8 000**.
- Entre **240 y 300 días** hay un **salto fuerte**, hasta unos **36 000**.
- Hay **un punto gris solitario arriba, a unos 106 000**, a los ~242 días.
- Al final vuelve a bajar a unos 10 000–12 000.

### 🔎 ¿Qué es ese salto? ¡El COVID!

Ese punto es **abril de 2020**, en el examen que cortó el **3 de agosto de 2019**:

- Entre el 3 de agosto de 2019 y el 1 de abril de 2020 hay **242 días**: justo donde está el punto.
- El modelo predijo **273 725** y el valor real fue **167 617**: un error de **más de 106 000**, un **63 %**.

**El modelo no lo hizo mal:** en agosto de 2019 **nadie** podía saber que venía una pandemia.

**Diferencia con el video:** en el gráfico del profesor la línea sube y baja entre unos 4 000 y 8 500 **sin saltos grandes**, porque sus datos terminan en 2018. Él dice que en algunos horizontes el modelo va mejor que en otros "quizás por la estacionalidad o simplemente por el error". En mis datos, la razón del salto está clarísima.

**Qué decir:** "Cada punto gris es un error y la línea azul es el promedio. Hasta los 200 días el error es bajo y estable. Pero miren este salto cerca de los 240 días y este punto solitario arriba: es abril de 2020. En el examen de agosto de 2019, el modelo predijo un abril normal y llegó el COVID. Un solo evento extremo puede inflar mucho las métricas."

---

## Paso 16: Graficar el error en porcentaje (MAPE)

```python
plot_cross_validation_metric(df_cv, metric='mape');
```

**Qué se ve:**

- De 0 a unos 200 días, la línea azul está entre **0.015 y 0.04**: **entre 1.5 % y 4 % de error**.
- El salto del COVID entre 240 y 300 días sube hasta **0.10 (10 %)**.
- El punto de abril de 2020 está arriba, en **0.63 (63 %)**.

**Por qué es el mejor según el profesor:** el MAPE es **el más fácil de interpretar** y es el que usa la documentación. "Me equivoco un 2 %" lo entiende cualquiera.

### La versión de la documentación (con datos diarios)

En la documentación (el link del paso 9), el gráfico de MAPE usa datos **diarios**: tiene **miles de puntos grises** y una línea **suave** que sube de ~6 % a ~10 %. La documentación además explica:

- **`rolling_window`:** la línea azul es un promedio móvil del 10 % de las filas (`rolling_window=0.1`). Con un número más alto sale más suave.
- **`initial` tiene que ser suficientemente largo:** al menos **1 año** si hay estacionalidad anual y al menos **1 semana** si hay semanal.

**Lo que dice el profesor:** estos gráficos son **más útiles con datos diarios**, porque hay muchos más puntos. Con datos mensuales salen pocos.

**Qué decir:** "Este es el gráfico que le mostraría a un jefe. Hasta unos 7 meses de distancia nos equivocamos entre un 1.5 y un 4 %. El salto es otra vez el COVID."

---

## Paso 17 (extra mío): Resumen en una sola cifra

```python
print('MAPE promedio  :', round(df_cv.eval('abs(y - yhat) / y').mean() * 100, 2), '%')
print('Cobertura real :', round(((df_cv.y >= df_cv.yhat_lower) & (df_cv.y <= df_cv.yhat_upper)).mean() * 100, 1), '%')
```

```
MAPE promedio  : 3.58 %
Cobertura real : 50.0 %
```

**Cómo leerlo:**

- **MAPE de 3.58 %:** en los 10 exámenes, el modelo se equivoca en promedio un 3.6 %. **Sin el examen del COVID, baja a 2.56 %.**
- **Cobertura de 50 %:** Prophet dibuja un rango del **80 %** por defecto, así que **debería** acertar el 80 % de las veces. Acierta solo la mitad: **el rango es demasiado angosto**, o sea, el modelo es **más confiado de lo que debería**.

> La cobertura puede variar un poco (por ejemplo 49 % o 50 %) cada vez que se ejecuta, porque Prophet calcula el rango con simulaciones al azar.

**El error de cada examen:**

| Examen (corte) | MAPE | Comentario |
|---|---|---|
| 1979 | 5.1 % | Crisis del petróleo: el tráfico bajó en 1979–1980 |
| 1984 | 2.4 % | |
| 1989 | **1.0 %** | El mejor examen |
| 1994 | 1.7 % | |
| 1999 | 1.8 % | |
| 2004 | 1.4 % | |
| 2009 | 4.0 % | Crisis financiera |
| 2014 | 3.4 % | |
| **2019** | **12.8 %** | **COVID** |
| 2024 | 2.1 % | |

**Qué decir:** "En una cifra: en 10 exámenes a lo largo de 50 años, el modelo se equivoca un 3.6 %, y un 2.6 % sin el COVID. Los peores exámenes coinciden con crisis reales: el petróleo en 1979, la crisis financiera de 2009 y el COVID. En años normales, el error está entre 1 y 2 %."

---

## Conclusión

**Lo que dice el profesor:**

- **Para qué sirve:** con estas métricas ves **hasta qué distancia es seguro pronosticar**. Si el error se dispara a los 200 días, no deberías confiar más allá.
- **En este dataset** el error casi no cambia con el horizonte, porque hay una **tendencia y una estacionalidad muy fuertes**. Con datos más **ruidosos** sería mucho más difícil.

**Lo que agrego con los datos actualizados:** el salto a los 240 días **no** significa que el modelo empeore a esa distancia: significa que **justo ahí cayó el COVID** en uno de los exámenes. Antes de concluir "el modelo es malo a X días", hay que **mirar qué datos producen el error**.

**Qué decir para cerrar:** "Hoy evaluamos Prophet de dos formas. La clásica, escondiendo el último año, nos dio un error de unos 5 900 sobre valores de 278 000: un 2 %. La validación cruzada hizo 10 exámenes en 50 años y nos dio un 3.6 %, o un 2.6 % sin el COVID. Las dos dicen lo mismo: el modelo es bueno. Y aprendimos que ningún modelo predice una pandemia. En la próxima clase vamos a ver los **puntos de cambio de tendencia**."

---

## Posibles preguntas del público

**"¿Qué diferencia hay entre train/test y la validación cruzada?"**
Train/test es **un examen**; la validación cruzada son **muchos exámenes** en distintos momentos. Con uno solo puedes tener suerte; con muchos, el resultado es más confiable.

**"¿Por qué el profesor dice que es mejor con datos diarios?"**
Con datos diarios, cada examen de 1 año tiene 365 predicciones en vez de 12: hay miles de puntos y el gráfico se ve mucho más claro.

**"¿Cuál métrica uso?"**
Para explicarlo a otros, **MAPE**. Para comparar modelos sobre los mismos datos, **RMSE** sirve. Con eventos extremos como el COVID, la **MDAPE** (la mediana) es más justa.

**"¿Se puede hacer que Prophet ignore el COVID?"**
Sí: marcando esos meses como vacíos (`None`). Lo hacemos en las clases 3 y 4.

---

## Todo el código de esta clase

```python
import pandas as pd
from prophet import Prophet
%matplotlib inline

# 1. Datos
df = pd.read_csv('Data/Miles_Traveled.csv')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df.plot(x='ds', y='y')

# 2. Train / test
train = df.iloc[:668]
test  = df.iloc[668:]

# 3. Entrenar y predecir
m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=12, freq='MS')
forecast = m.predict(future)

# 4. Gráfico predicción vs. realidad
ax = forecast.plot(x='ds', y='yhat', label='Predictions', legend=True, figsize=(12, 8))
test.plot(x='ds', y='y', label='True Test Data', legend=True, ax=ax,
          xlim=(pd.Timestamp('2025-09-01'), pd.Timestamp('2026-08-01')))

# 5. RMSE
from statsmodels.tools.eval_measures import rmse
predictions = forecast.iloc[-12:]['yhat']
rmse(predictions, test['y'])           # 5895.43
test.mean()                            # y: 277974.58

# 6. Validación cruzada
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

initial = str(5 * 365) + ' days'
period  = str(5 * 365) + ' days'
horizon = str(365) + ' days'

df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)
df_cv.head()
len(df_cv)                             # 120
performance_metrics(df_cv)

plot_cross_validation_metric(df_cv, metric='rmse');
plot_cross_validation_metric(df_cv, metric='mape');
```
