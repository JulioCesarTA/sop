# Clase 4 — Prophet: estacionalidad aditiva vs. multiplicativa

Guía para exponer. El notebook es **`04_Prophet_Estacionalidad_VentasInternet.ipynb`**.

**Datos:** `Data/ventas_internet.csv`: **ventas por internet y por catálogo en EE. UU.** (Amazon, tiendas online, venta por correo), en **millones de dólares por mes**, de **enero 1992 a agosto 2026**. Fuente: Oficina del Censo de EE. UU., vía FRED (serie `MRTSSM4541USN`).

**Cómo ejecutarlo:** ábrelo en VS Code y dale a **Run All**. El notebook viene ejecutado. **Todos los números de esta guía son reales.**

> Para entender **cada gráfico en detalle**, mira también **`04b_Explicacion_Graficos_Estacionalidad.md`**.

---

## 0. La idea de esta clase: aditiva vs. multiplicativa

Hasta ahora Prophet usaba la estacionalidad **aditiva** (la que trae por defecto). Hoy vemos la **multiplicativa**.

### Aditiva: el efecto de la temporada es una **cantidad fija**

> "En diciembre **siempre** se venden **9 700 millones** más de lo normal."

Da igual si es 1992 (cuando se vendían unos 2 900 millones por mes) o 2025 (unos 106 000): siempre son **+9 700**.

### Multiplicativa: el efecto de la temporada es un **porcentaje**

> "En diciembre **siempre** se vende un **27 %** más de lo normal."

- Si normalmente se venden **3 000**, en diciembre se venden **3 810** (+810).
- Si normalmente se venden **110 000**, en diciembre se venden **139 700** (+29 700).

Como es un porcentaje, **el efecto crece junto con la tendencia**.

### Cómo reconocerla en un gráfico

- Si las **olas** (el zigzag de cada año) tienen **siempre el mismo tamaño**, es **aditiva**.
- Si las olas **se hacen más grandes** a medida que la serie crece, es **multiplicativa**.

Lo que dice el profesor: el crecimiento puede ser aditivo ("la tasa de cambio es lineal y estable") o multiplicativo ("la tasa de cambio va cambiando con el tiempo").

**Qué decir para abrir:** "En unos datos, la temporada suma siempre lo mismo. En otros suma un porcentaje, y como los datos crecen, las olas también crecen. Prophet por defecto asume lo primero. Hoy vemos qué pasa cuando eso no es cierto."

### ¿Por qué estos datos y no los del video?

| | Video | Yo |
|---|---|---|
| Datos | Pasajeros de avión **1949–1960** | Ventas por internet **1992–2026** |
| ¿Cuánto creció? | ×4 | **×37** (de 2 882 a 106 540 por mes) |
| ¿Se ve la diferencia? | Sí, clarísima | **Sí, clarísima** |

> El dataset del video es **histórico**: no existe una versión actualizada. Probé con pasajeros de avión actuales (2000–2026), pero solo crecieron un 45 % y el COVID los hizo caer un 96 %, así que aditivo y multiplicativo **daban casi igual**. Las ventas por internet crecieron **37 veces**, así que la diferencia se ve **igual de clara que en el video**. Además, en el COVID las ventas online **subieron** (no se desplomaron), así que no rompen el modelo.

---

## Paso 1: Importar y leer los datos

```python
import pandas as pd
from prophet import Prophet                        # en el video: from fbprophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('Data/ventas_internet.csv')
df.head()
```

**Qué sale:**

|   | observation_date | MRTSSM4541USN |
|---|---|---|
| 0 | 1992-01-01 | 2644 |
| 1 | 1992-02-01 | 2245 |
| 2 | 1992-03-01 | 2440 |

**Cómo leerlo:** `MRTSSM4541USN` es el código de la serie. Son las ventas del mes de las empresas que venden **sin tienda física**: por internet o por catálogo (por correo o teléfono). En enero de 1992 vendieron **2 644 millones de dólares**. En 1992 casi todo era venta por **catálogo** (internet casi no existía); hoy casi todo es **online**.

**Qué decir:** "Usamos las ventas por internet y por catálogo de EE. UU., desde 1992 hasta 2026. En 1992 se vendían unos 2 600 millones por mes, casi todo por catálogo. Hoy se venden más de 100 000 millones por mes."

---

## Paso 2: Preparar el formato y mirar los datos

```python
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df.plot(x='ds', y='y', figsize=(12, 5), legend=False, title='Ventas por internet y catálogo (millones de USD por mes)');
```

**Qué hace:** lo de siempre (renombra a `ds` y `y` y convierte la fecha) y además dibuja los datos.

**Qué se ve:**

- Una curva que sube **cada vez más rápido**: casi plana en los 90, más empinada desde 2010 y **muy empinada desde 2017**.
- **Cada diciembre hay una aguja** (compras de Navidad, *Black Friday*, *Cyber Monday*).
- **Las agujas se hacen cada vez más grandes**: en los 90 casi no se notan y en 2020–2025 son enormes. **Esta es la pista visual de la estacionalidad multiplicativa.**
- En **2020** las ventas **saltan** con el COVID: la gente compraba todo online.

**Qué decir:** "Miren las agujas de cada diciembre: al principio casi no se ven, y al final son gigantes. Las olas crecen junto con las ventas. Esa es la pista de que la estacionalidad es multiplicativa."

---

## Paso 3: ¿Las olas crecen con el tiempo?

```python
anual = df.groupby(df['ds'].dt.year)['y'].agg(['min', 'max', 'mean'])
anual = anual[df.groupby(df['ds'].dt.year).size() == 12]          # solo años completos
anual['ola'] = anual['max'] - anual['min']                         # tamaño de la ola (millones de USD)
anual['ola_%'] = (anual['ola'] / anual['mean'] * 100).round(1)     # tamaño de la ola (%)
anual.round(0)
```

Calcula, para cada año, el **mejor mes**, el **peor mes** y el **promedio**. La **ola** es mejor − peor, en millones y en %. (Explicado línea por línea en la guía `04b`.)

**Qué sale (algunos años):**

| Año | Promedio | Ola (millones) | Ola % |
|---|---|---|---|
| 1992 | 2 882 | **2 201** | 76 % |
| 2000 | 9 374 | 5 394 | 58 % |
| 2010 | 21 675 | 14 315 | 66 % |
| 2019 | 53 287 | 34 376 | 64 % |
| 2025 | 106 540 | **48 408** | 45 % |

**Cómo leerlo:**

- **En millones, la ola se multiplicó por 22** (de 2 201 a 48 408). **No es aditivo**: si lo fuera, la ola sería del mismo tamaño todos los años.
- **En %, la ola se mueve entre 40 % y 85 %**, sin crecer. Baja un poco en los últimos años (porque hoy se compra online todo el año, no solo en Navidad), pero **no explota** como la ola en millones.
- **Conclusión:** la ola crece **con** las ventas. Eso es estacionalidad **multiplicativa**.

**Qué decir:** "En 1992 la diferencia entre el mejor y el peor mes era de 2 200 millones. En 2025, de 48 000 millones: 22 veces más. Pero en porcentaje se mantuvo parecida. La temporada es un porcentaje de las ventas, y por eso crece con ellas."

---

## Paso 4: Primer modelo — aditivo (el de siempre)

```python
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(50, freq='MS')
forecast = m.predict(future)
fig = m.plot(forecast)
```

- Entrena con **todos los datos**. El profesor dice que **suponemos que ya hicimos la evaluación** con train/test.
- `make_future_dataframe(50, freq='MS')` predice **50 meses** (hasta **octubre 2030**). El `50` es lo mismo que `periods=50`.
- `fig = m.plot(forecast)` grafica.

**Lo que dice el profesor:** "estas son las líneas principales que siempre corremos. **Si entiendes estas líneas, prácticamente entiendes Prophet.**"

**Qué se ve (esto es lo importante):**

- **Al principio (años 90):** la línea azul hace **agujas enormes** cada diciembre, **muchísimo más altas** que los puntos negros. Y en los eneros y febreros **baja de cero**.
- **Al final (2020–2025):** las agujas azules se quedan **cortas**: los puntos negros de diciembre quedan **por encima**.

| Mes | Real ⚫ | Aditivo 🔵 | Problema |
|---|---|---|---|
| diciembre 1992 | 4 446 | **13 670** | ¡Predice **3 veces** lo real! |
| febrero 1993 | 2 690 | **−1 845** | ¡**Ventas negativas**! Imposible |
| diciembre 2024 | 131 582 | 116 056 | Se queda **15 500 corto** |
| diciembre 2025 | 139 317 | 124 046 | Se queda **15 300 corto** |

**¿Por qué?** El modelo aditivo suma **la misma cantidad** cada diciembre (≈ +9 700 millones) y resta lo mismo cada febrero (≈ −4 000). En 1992, cuando se vendían 2 800 por mes, sumar 9 700 es **absurdo**. Y restar 4 000 da **negativo**. En 2025, cuando se venden 113 000, sumar 9 700 se **queda corto**.

**Qué decir:** "Miren los años 90: la línea azul hace picos tres veces más altos que la realidad, y en febrero predice ventas negativas, que no existen. ¿Por qué? Porque el modelo aditivo suma y resta las mismas cantidades todos los años. En 1992 esas cantidades son gigantes; en 2025 son chicas."

---

## Paso 5: Las piezas del modelo aditivo

```python
fig = m.plot_components(forecast);
```

**Arriba, la tendencia:** sube de **2 043** en 1992 a **119 113** en agosto 2026 y **154 055** en 2030. Crece **cada vez más rápido**: +808 por año en los 90, +3 201 en 2012–2017 y **+8 386** desde 2019.

**Abajo, el efecto del mes, en millones de dólares** (solo valen los días 1; ver la guía `04b`):

| Mes | Efecto aditivo |
|---|---|
| Enero | −2 212 |
| **Febrero** | **−4 068** (el más bajo) |
| Marzo a octubre | entre −2 100 y −260 |
| Noviembre | +3 863 (*Black Friday*) |
| **Diciembre** | **+9 732** (Navidad, el más alto) |

**Detalle clave:** estos números son **iguales para todos los años**. Por eso falla en 1992 y en 2025.

**Lo que dice el profesor:** la estacionalidad del pronóstico es "**demasiado grande al inicio y demasiado pequeña al final**". No es un factor aditivo constante: **crece con la tendencia**.

**Qué decir:** "Abajo dice: cada diciembre se suman 9 700 millones, cada febrero se restan 4 000. Siempre igual. En 1992 la tendencia era de 2 800: sumarle 9 700 la multiplica por cuatro. En 2025 la tendencia era de 113 000: sumarle 9 700 es poquito."

---

## Paso 6: Los puntos de cambio

```python
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
```

Dibuja la **tendencia** (línea roja) y los **puntos de cambio** (líneas punteadas). El `a = ...` es la convención de la documentación.

**Qué se ve:** **14 líneas punteadas**, **todas aceleraciones** (la línea roja se pone cada vez más empinada). Se juntan sobre todo entre **2013 y 2018**: el **boom del comercio electrónico** (Amazon Prime, celulares, compras desde el teléfono).

**Lo que dice el profesor (con su dataset):** los cambios coinciden con la llegada de los **aviones a reacción** en los 50. Aquí coinciden con la **explosión de las compras online** en los 2010.

**Qué decir:** "Prophet encontró 14 momentos donde el crecimiento se aceleró, sobre todo entre 2013 y 2018. Es el boom de Amazon y de comprar desde el celular."

---

## Paso 7: Segundo modelo — multiplicativo (lo nuevo de esta clase)

```python
m = Prophet(seasonality_mode='multiplicative')
m.fit(df)
future = m.make_future_dataframe(50, freq='MS')
forecast = m.predict(future)
fig = m.plot(forecast)
```

**Qué cambia:** **solo la primera línea**. El profesor copia y pega todo y solo agrega `seasonality_mode='multiplicative'`.

**Qué se ve:**

- **Al principio:** las agujas azules son **chiquitas**, del tamaño de las reales. **No hay ventas negativas.**
- **Al final:** las agujas son **enormes** y llegan a los puntos negros de diciembre.
- **Las olas crecen con el tiempo**, igual que en los datos reales.
- En el pronóstico, el pico de diciembre 2029 llega a unos **189 000**.

| Mes | Real ⚫ | Aditivo | **Multiplicativo** |
|---|---|---|---|
| diciembre 1992 | 4 446 | 13 670 ❌ | **3 495** ✅ |
| febrero 1993 | 2 690 | −1 845 ❌ | **2 466** ✅ |
| diciembre 2019 | 77 951 | 72 941 | **79 800** ✅ |
| diciembre 2025 | 139 317 | 124 046 | **146 031** (algo alto) |

**Lo que dice el profesor:** "ahora los resultados funcionan **mucho mejor**, porque se ve claramente que la tendencia y la estacionalidad cambiaban con el tiempo".

> En los diciembres más recientes, el multiplicativo **se pasa un poco** (146 000 contra 139 300). Es porque la ola en % bajó en los últimos años (de ~65 % a ~45 %), así que el porcentaje promedio queda algo alto. Igual se equivoca **mucho menos** que el aditivo en el conjunto.

**Qué decir:** "Con una sola palabra cambiamos todo. Compárenlo con el gráfico anterior: ya no hay picos gigantes en los 90 ni ventas negativas, y las olas crecen igual que en la realidad."

---

## Paso 8: Las piezas del modelo multiplicativo

```python
fig = m.plot_components(forecast);
```

**Arriba:** la tendencia es **casi igual** a la del aditivo.

**Abajo:** el efecto del mes **ahora está en porcentaje**:

| Mes | Efecto multiplicativo |
|---|---|
| Enero | −7.0 % |
| **Febrero** | **−12.6 %** (el más bajo) |
| Marzo a octubre | entre −6.5 % y −0.3 % |
| Noviembre | +13.2 % |
| **Diciembre** | **+26.7 %** (el más alto) |

**La gran diferencia:**

| | Aditivo | Multiplicativo |
|---|---|---|
| Unidad | Millones de dólares | **Porcentaje** |
| Diciembre | "Suma **9 732 millones**" | "Suma un **26.7 %**" |
| Diciembre de 1992 (tendencia ≈ 2 700) | +9 732 | +≈ 720 |
| Diciembre de 2025 (tendencia ≈ 113 700) | +9 732 | +≈ 30 400 |

**Qué decir:** "Ahora la estacionalidad dice 'diciembre suma un 27 %'. Un 27 % de 2 700 son 720 millones, pero un 27 % de 113 000 son 30 000 millones. Por eso las olas crecen."

---

## Paso 9 (extra): ¿Cuál es mejor? Que decida el RMSE

> El profesor **recomienda** esto pero no lo hace en el video: "si no estás seguro, crea un modelo aditivo y uno multiplicativo, pruébalos con tus datos de prueba y quédate con el de menor RMSE".

```python
from statsmodels.tools.eval_measures import rmse

train = df.iloc[:-12]     # todo menos el último año
test  = df.iloc[-12:]     # el último año (sep 2025 -> ago 2026)

resultados = {}
for modo in ['additive', 'multiplicative']:
    mod = Prophet(seasonality_mode=modo)
    mod.fit(train)
    fc = mod.predict(mod.make_future_dataframe(periods=12, freq='MS'))
    pred = fc.iloc[-12:]['yhat']
    resultados[modo] = {
        'RMSE': round(rmse(pred, test['y']), 0),
        'Error %': round(rmse(pred, test['y']) / test['y'].mean() * 100, 2),
    }

pd.DataFrame(resultados).T
```

**Qué hace:** esconde el último año, entrena los dos modelos con el resto, predice ese año y compara. El `for` repite lo mismo con `'additive'` y con `'multiplicative'`.

**Qué sale:**

| Modo | RMSE | Error % |
|---|---|---|
| additive | 8 847 | 7.90 % |
| **multiplicative** | **6 508** | **5.81 %** |

**El multiplicativo se equivoca un 26 % menos.**

**Qué decir:** "El profesor dice: si dudan, prueben los dos y que decida el RMSE. Lo hice con el último año: el aditivo se equivoca un 7.9 % y el multiplicativo un 5.8 %. Gana el multiplicativo."

---

## Paso 10 (extra): Los dos modelos con zoom

```python
f_add = Prophet().fit(df).predict(df[['ds']])
f_mul = Prophet(seasonality_mode='multiplicative').fit(df).predict(df[['ds']])
# ... dos gráficos: 1992–1996 y 2021–2026
```

Dos gráficos lado a lado, con los datos reales (puntos negros), el aditivo (azul) y el multiplicativo (naranja).

- **1992–1996 (izquierda):** **el gráfico más claro de toda la clase.** La línea azul hace **agujas de 14 000** cuando lo real llega a 4 000–7 000, y **baja de cero** en los inviernos. La naranja va **pegada** a los puntos.
- **2021–2026 (derecha):** la azul se queda **corta** en los diciembres (llega a unos 116 000–124 000 cuando lo real es 131 000–139 000). La naranja **llega** a los picos (a veces se pasa un poco).

**Qué decir:** "Este gráfico lo resume todo. A la izquierda, los años 90: el aditivo inventa picos tres veces más grandes y ventas negativas. A la derecha, los años recientes: el aditivo se queda corto en Navidad. El multiplicativo anda bien en las dos épocas."

---

## Paso 11 (extra): ¿Cuánto se equivoca cada modelo en cada época?

```python
comp = df.copy()
comp['err_add_%'] = ((comp['y'] - f_add['yhat']).abs() / comp['y'] * 100)
comp['err_mul_%'] = ((comp['y'] - f_mul['yhat']).abs() / comp['y'] * 100)
comp['época'] = pd.cut(comp['ds'].dt.year, [1991, 1996, 2005, 2015, 2026],
                       labels=['1992-1996', '1997-2005', '2006-2015', '2016-2026'])
comp.groupby('época', observed=True)[['err_add_%', 'err_mul_%']].mean().round(1)
```

**Qué hace (en simple):**

- Calcula el **error en %** de cada mes, para los dos modelos: `|real − predicho| ÷ real × 100`.
- `pd.cut(...)` **agrupa los años en 4 épocas**.
- `groupby(...).mean()` saca el **error promedio** de cada época.

**Qué sale:**

| Época | Error aditivo | Error multiplicativo |
|---|---|---|
| **1992–1996** | **50.7 %** | **11.7 %** |
| 1997–2005 | 17.0 % | 5.9 % |
| 2006–2015 | 5.2 % | 4.5 % |
| 2016–2026 | 6.5 % | 5.4 % |

**Cómo leerlo:** el multiplicativo gana **en todas las épocas**, y la diferencia es **gigante al principio**: en 1992–1996 el aditivo se equivoca **la mitad** del valor real, y el multiplicativo solo un 12 %.

**Qué decir:** "En los primeros años el aditivo se equivoca un 50 %: está completamente perdido. El multiplicativo, un 12 %. Y gana en todas las épocas."

---

## Cierre: ¿cómo saber cuál usar?

**Lo que dice el profesor:**

- Si ves que la estacionalidad **crece o decrece con el tiempo**, prueba `seasonality_mode='multiplicative'`.
- Si **no estás seguro**: crea **dos modelos**, pruébalos con datos de prueba y **quédate con el de menor RMSE**.

**Qué decir para cerrar:** "Regla práctica: miren las olas. Si son siempre del mismo tamaño, aditivo. Si crecen con los datos, multiplicativo. En las ventas por internet, las olas de Navidad crecieron 22 veces, y el modelo aditivo se volvió loco: picos tres veces más grandes y ventas negativas en los 90. Con una sola palabra, `seasonality_mode='multiplicative'`, el error bajó de 7.9 % a 5.8 %. Y si dudan, prueben los dos y que decida el RMSE."

### Conexión con la clase 1

En la clase 1 (cerveza, vino y licor), Prophet **exageraba los diciembres de los 90** y se quedaba corto en los recientes. **Es el mismo problema** que vimos hoy, pero más chico. Probar `seasonality_mode='multiplicative'` ahí sería el siguiente paso lógico.

---

## Posibles preguntas del público

**"¿Por qué no usaste los datos del video?"**
Porque son de 1949–1960 y no tienen versión actualizada. Busqué un dataset actual donde la diferencia se viera igual de clara, y las ventas por internet son ideales: crecieron 37 veces.

**"¿Por qué el aditivo da ventas negativas?"**
Porque resta una cantidad fija (unos 4 000 millones en febrero) que es **más grande** que las ventas de 1992–1993 (unos 2 700). Un modelo aditivo no sabe que las ventas no pueden ser negativas.

**"¿Por qué el multiplicativo se pasa un poco en los diciembres recientes?"**
Porque la ola en % no es perfectamente constante: bajó de ~65 % a ~45 % en los últimos años (hoy se compra online todo el año, no solo en Navidad). El modelo usa un porcentaje promedio. Igual, en el conjunto se equivoca mucho menos.

**"¿Por qué Prophet no usa multiplicativo por defecto?"**
Porque el aditivo es más simple y funciona bien en muchos casos. Además, con datos que pueden ser cero o negativos, un porcentaje no tiene sentido.

---

## Todo el código de esta clase

```python
import pandas as pd
from prophet import Prophet                        # en el video: from fbprophet import Prophet
from prophet.plot import add_changepoints_to_plot  # en el video: from fbprophet.plot import ...
%matplotlib inline

# Datos
df = pd.read_csv('Data/ventas_internet.csv')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

# Modelo 1: aditivo (por defecto)
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(50, freq='MS')
forecast = m.predict(future)
fig = m.plot(forecast)
fig = m.plot_components(forecast)

# Puntos de cambio
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# Modelo 2: multiplicativo
m = Prophet(seasonality_mode='multiplicative')
m.fit(df)
future = m.make_future_dataframe(50, freq='MS')
forecast = m.predict(future)
fig = m.plot(forecast)
fig = m.plot_components(forecast)
```
