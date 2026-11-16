# Clase 3 — Prophet: dónde cambia la tendencia (changepoints)

Guía para exponer. El notebook es **`03_Prophet_Tendencia_Hospitality.ipynb`**.

**Datos:** `Data/HospitalityEmployees.csv`, **actualizados hasta septiembre de 2026**. Es la misma serie que usa el video, descargada hoy de la Reserva Federal (FRED, serie `CALEIHN`, fuente: Oficina de Estadísticas Laborales de EE. UU.).

**Cómo ejecutarlo:** ábrelo en VS Code y dale a **Run All**. Ya está todo instalado y el notebook viene ejecutado. **Todos los números de esta guía son los que salieron de verdad.**

---

## 0. La idea de esta clase

En la teoría dijimos que la tendencia de Prophet **no es una recta perfecta**: tiene **puntos de cambio** (*changepoints*), momentos en que la tendencia **acelera, frena o cambia de dirección**. Prophet los detecta solo. **Hoy los vemos en un gráfico.**

**¿Por qué importa?** Algunos cambios se ven a simple vista (como una crisis grande), pero **muchos son tan pequeños que el ojo no los detecta**. Prophet los encuentra y nos dice **en qué fecha** pasaron, para que podamos **investigar qué pasó**.

**Qué decir para abrir:** "Prophet no solo predice: también nos dice **cuándo cambió el rumbo** de los datos, incluso cuando el cambio es tan chico que no lo vemos."

### Diferencias con el video

| | Video | Yo |
|---|---|---|
| Datos | 1990 → 2018 (348 meses) | 1990 → **septiembre 2026** (441 meses) |
| Lo más importante | El cambio grande es la crisis de **2008** | Además de 2008, aparece el **COVID**, que **rompe el modelo**. Vemos cómo arreglarlo (paso 8). |
| Librería | `fbprophet` | `prophet` |

> **Qué mide exactamente:** el CSV se llama "Hospitality Employees", pero la serie es **empleados de ocio y hospitalidad en California** (hoteles, restaurantes, bares, entretenimiento), **no de todo EE. UU.** Está en **miles de personas**: 1064.9 = 1 064 900 empleados.

---

## Paso 1: Importar y preparar los datos

```python
import pandas as pd
from prophet import Prophet                        # en el video: from fbprophet import Prophet
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('Data/HospitalityEmployees.csv')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df.head()
```

**Qué hace:** lo de siempre. Lee el archivo, renombra a `ds` y `y` y convierte la fecha.

**Qué sale:**

|   | ds | y |
|---|---|---|
| 0 | 1990-01-01 | 1064.9 |
| 1 | 1990-02-01 | 1074.9 |
| 2 | 1990-03-01 | 1090.4 |

**Por qué este dataset:** según el profesor, tiene **un cambio de tendencia muy claro** y además **muchos cambios pequeños** que no se ven a simple vista.

**Qué decir:** "Usamos datos de empleo en hoteles, restaurantes y entretenimiento en California, en miles de personas, de 1990 a 2026. Los tres pasos de siempre: leer, renombrar y convertir la fecha."

---

## Paso 2: Mirar los datos

```python
df.plot(x='ds', y='y', figsize=(12, 10));
```

`figsize=(12, 10)` lo hace **grande** para explorarlo, como en el video.

**Qué se ve:**

- **De 1990 a 2008** la línea **sube**, de unos **1 065** a unos **1 600**.
- **Alrededor de 2008–2010** **baja** hasta unos **1 440**: la **recesión de 2008**.
- Después **vuelve a subir, y más rápido**, hasta unos **2 070** en 2019.
- **En abril de 2020 se desploma** a **1 071**: pasa de **2 026** en febrero a **1 071** en abril, **casi la mitad de los empleos (−47 %)** en dos meses. Es el **COVID**: hoteles, restaurantes y cines cerrados.
- Se recupera hasta **2022–2023** y queda **estable** alrededor de **2 000–2 080**. El máximo de toda la serie es **2 086 (junio 2026)**.

**Lo que dice el profesor (de su versión hasta 2018):**

- El **cambio grande** se ve a simple vista: **sube, baja en 2008 y vuelve a subir**.
- Pero hay **cambios pequeños que no se ven**. Por ejemplo, **entre 1994 y finales de los 90** la subida primero se pone **más suave** y luego **más rápida**.
- También es difícil ver **cuándo la tendencia está estable**.

**Qué decir:** "Con el ojo vemos dos cambios grandes: la crisis de 2008 y, sobre todo, el COVID, cuando se perdió casi la mitad de los empleos en dos meses. Pero los cambios chiquitos, cuando la subida se pone un poco más rápida o más lenta, son casi imposibles de ver. Para eso usamos Prophet."

---

## Paso 3: Entrenar y predecir

```python
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=12, freq='MS')
forecast = m.predict(future)
```

**Qué hace:** las 4 líneas de siempre.

- Entrena con **todos los datos**. Aquí **no** hay entrenamiento y prueba: el objetivo es solo **ver los puntos de cambio**.
- Predice **12 meses** hacia adelante (hasta septiembre 2027), con frecuencia mensual.

**Lo que dice el profesor:** si lo combinas con un *train/test split* también podrías evaluar el pronóstico, pero lo que importa en esta clase son **las tres líneas nuevas** que vienen ahora.

**Qué decir:** "Nada nuevo: las mismas cuatro líneas. Lo nuevo viene ahora."

---

## Paso 4: Dibujar los puntos de cambio (lo nuevo de esta clase)

```python
from prophet.plot import add_changepoints_to_plot   # en el video: from fbprophet.plot import ...

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
```

**Qué hace cada línea:**

| Línea | Qué hace |
|---|---|
| `from prophet.plot import add_changepoints_to_plot` | Trae la función **"agregar puntos de cambio al gráfico"**. |
| `fig = m.plot(forecast)` | Hace el **gráfico normal** de Prophet y lo guarda en `fig`. |
| `a = add_changepoints_to_plot(fig.gca(), m, forecast)` | Le dibuja **encima** la tendencia y los puntos de cambio. |

**Las 3 cosas que recibe `add_changepoints_to_plot`:**

1. **`fig.gca()`**: ***get current axis***, "dame los ejes del gráfico actual". Le dice **dónde** dibujar.
2. **`m`**: el modelo, que **sabe** dónde están los cambios.
3. **`forecast`**: las predicciones.

**¿Por qué `a = ...`?** Es **la convención de la documentación** de Prophet; evita que Jupyter imprima texto extra.

**Ejemplo simple:** primero **imprimes un mapa** (`m.plot`) y después le **clavas chinches rojas** en los lugares importantes (`add_changepoints_to_plot`). `fig.gca()` es decir "las chinches van **en este** mapa".

**Qué decir:** "Estas tres líneas son lo único nuevo de hoy: importar la función, hacer el gráfico normal y ponerle encima los puntos de cambio."

---

## Paso 5: Leer el gráfico (y el problema del COVID)

| Elemento | Qué es |
|---|---|
| **Puntos negros** | Los datos reales |
| **Línea azul y sombra** | La predicción, como siempre |
| **Línea roja continua** | **La tendencia** que calculó Prophet |
| **Líneas rojas punteadas** | **Los puntos de cambio** |

### ⚠️ Lo que pasa con los datos actualizados

El gráfico **no se parece al del video**, y aquí está la parte más interesante de la clase:

- La **línea roja es casi una recta** de 1990 a 2027. **No se nota la crisis de 2008**: la recta pasa por encima de los datos de 2009–2012.
- Solo hay **5 líneas punteadas**, todas pequeñas: 2009, 2011, 2015, 2016 y 2018.
- La **sombra celeste es enorme**: mide unos **300** de ancho, cuando los datos normales casi no se salen de la línea azul.
- La línea azul queda **por debajo** de los puntos negros de **2016 a 2019**: en promedio, el empleo real fue **144 mil más alto** que lo que dice el modelo.

**¿Por qué?** **Por el COVID.** La caída de casi 1 000 (miles de empleos) en 2020 es **gigante** comparada con todo lo demás. Prophet intenta explicarla con la tendencia y con el ruido:

- Como no puede poner un punto de cambio en 2020 (lo vemos en el paso 6), **aplana toda la tendencia** para "repartir" ese pozo.
- Como ve errores enormes en 2020, **cree que los datos son muy ruidosos** y agranda la sombra.

**Qué decir:** "Miren qué pasa con los datos de hoy: la línea roja es casi recta y no muestra ni la crisis de 2008. La sombra es enorme. ¿Por qué? Porque el COVID hizo caer el empleo a la mitad en dos meses, y ese pozo gigante confunde al modelo: aplana toda la tendencia y piensa que los datos son puro ruido. En un rato vemos cómo arreglarlo."

---

## Paso 6 (extra mío): ¿En qué fechas exactas cambió la tendencia?

```python
import numpy as np

cambios = pd.DataFrame({
    'fecha': m.changepoints.dt.date.values,
    'cambio_pendiente': m.params['delta'].mean(axis=0),
})
importantes = cambios[cambios['cambio_pendiente'].abs() >= 0.01].copy()   # el mismo umbral que usa el gráfico
importantes['tipo'] = np.where(importantes['cambio_pendiente'] > 0, 'acelera ↑', 'frena ↓')
importantes
```

**Qué hace (en simple):**

- `m.changepoints` son **todas las fechas candidatas** donde Prophet **permitió** que la tendencia cambiara.
- `m.params['delta']` dice **cuánto cambió la pendiente** en cada una.
- Nos quedamos con los cambios **importantes** (0.01 o más, hacia arriba o abajo), el mismo criterio del gráfico.
- `tipo` dice si **acelera** (positivo) o **frena** (negativo).

**Qué sale:**

| fecha | cambio_pendiente | tipo |
|---|---|---|
| 2009-12-01 | +0.070 | acelera ↑ |
| 2011-02-01 | +0.047 | acelera ↑ |
| 2015-10-01 | −0.044 | frena ↓ |
| 2016-12-01 | −0.117 | frena ↓ |
| 2018-02-01 | −0.050 | frena ↓ |

```python
print('Puntos de cambio candidatos :', len(m.changepoints))
print('Puntos de cambio importantes:', len(importantes))
print('Último candidato posible    :', m.changepoints.max().date())
```

```
Puntos de cambio candidatos : 25
Puntos de cambio importantes: 5
Último candidato posible    : 2019-04-01
```

**Cómo funciona por dentro:**

1. Prophet pone **25 puntos candidatos** repartidos en el tiempo (valor por defecto).
2. En cada uno, la tendencia **podría** cambiar de pendiente.
3. Al entrenar decide **cuánto cambia** en cada uno. En la mayoría, casi nada.
4. Con COVID, solo **5** cambios son importantes, y todos son **pequeños** (el mayor es −0.117).

**¿Por qué el último candidato es abril de 2019?** Prophet **solo pone candidatos en el primer 80 % de los datos** (`changepoint_range=0.8`). El 80 % de 441 meses llega a **abril de 2019**. Lo hace para **no confundir el final de la serie con un cambio**. Por eso **no puede** poner un punto de cambio en el COVID (2020), y "reparte" el pozo aplanando toda la tendencia.

**Qué decir:** "Prophet prueba 25 lugares posibles, pero como no busca cambios en el último 20 % de los datos, no puede marcar el COVID. Por eso solo encuentra 5 cambios chiquitos y la tendencia queda casi recta."

---

## Paso 7 (extra mío): Zoom a la crisis de 2008 y al COVID

```python
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
plt.xlim(pd.Timestamp('2005-01-01'), pd.Timestamp('2013-01-01'))
plt.ylim(1350, 1750);
```

**Zoom a 2005–2013:** la línea roja sigue **subiendo derecho** durante toda la crisis, mientras los puntos negros **bajan** de 2008 a 2010. El modelo **no captó** la crisis de 2008.

```python
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
plt.xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2028-01-01'));
```

**Zoom a 2018–2028:**

- Los puntos de **2018–2019** quedan **por encima** de la sombra.
- El **pozo del COVID** (2020–2021) cae **muy por debajo**, hasta unos 1 070.
- Desde **2022** los puntos vuelven a la zona de la línea azul.
- La sombra es **muy ancha** todo el tiempo.

**Qué decir:** "Con zoom se ve el problema: el modelo no captó ni la crisis de 2008 ni el nivel real de 2018–2019. Todo por culpa de un evento extremo que no puede explicar."

---

## Paso 8 (extra mío): Quitar el COVID (como recomienda la documentación)

> La **documentación de Prophet** tiene una sección sobre **datos atípicos** (*outliers*). Dice que la mejor forma de tratarlos es **quitarlos**: Prophet acepta datos vacíos sin problema, porque no los usa para aprender.

```python
df_sin = df.copy()
covid = (df_sin['ds'] >= '2020-03-01') & (df_sin['ds'] <= '2021-12-01')
df_sin.loc[covid, 'y'] = None          # marzo 2020 -> diciembre 2021 quedan vacíos

m_sin = Prophet()
m_sin.fit(df_sin)
forecast_sin = m_sin.predict(m_sin.make_future_dataframe(periods=12, freq='MS'))

fig = m_sin.plot(forecast_sin)
a = add_changepoints_to_plot(fig.gca(), m_sin, forecast_sin)
```

**Qué hace (en simple):**

- `df.copy()` hace una **copia** de la tabla, para no tocar la original.
- `covid = ...` marca las filas entre **marzo 2020 y diciembre 2021**, los meses del COVID y su recuperación.
- `df_sin.loc[covid, 'y'] = None` **borra los valores** de esas filas (quedan vacíos). Las fechas siguen ahí.
- Después entrenamos y graficamos **igual que antes**, pero con `df_sin`.

**Qué se ve ahora (¡se parece al video!):**

- **11 líneas punteadas**, y la línea roja **sigue de cerca** a los datos.
- **1990–1993:** tendencia **casi plana** (la recesión de principios de los 90). Sin líneas.
- **1993–2004:** varios **cambios pequeños**.
- **2007:** la tendencia **se da vuelta hacia abajo**: la **crisis de 2008**, ahora sí.
- **2010–2012:** **acelera** fuerte: la recuperación.
- **2016–2017:** **frena** y queda **casi plana**: el empleo deja de crecer, y esa tendencia plana sigue hasta el pronóstico.
- La **sombra** es **angosta**.

```python
cambios_sin = ...      # la misma tabla del paso 6, pero con m_sin
importantes_sin
```

| fecha | cambio_pendiente | tipo |
|---|---|---|
| 1993-05-01 | +0.311 | acelera ↑ |
| 1994-06-01 | +0.065 | acelera ↑ |
| 2001-03-01 | −0.012 | frena ↓ |
| 2003-05-01 | +0.037 | acelera ↑ |
| 2004-07-01 | +0.135 | acelera ↑ |
| **2007-11-01** | **−1.208** | **frena ↓ ← el cambio más grande: la crisis de 2008** |
| 2010-01-01 | +0.766 | acelera ↑ |
| 2011-03-01 | +0.932 | acelera ↑ |
| 2012-04-01 | +0.155 | acelera ↑ |
| 2016-10-01 | −0.266 | frena ↓ |
| 2017-11-01 | −0.961 | frena ↓ |

```python
print('Ancho promedio de la sombra CON COVID:', ...)    # 299.6
print('Ancho promedio de la sombra SIN COVID:', ...)    # 50.7
```

**La sombra pasó de 300 a 51: casi 6 veces más angosta.** El modelo ahora es mucho más preciso.

**Cómo leer la historia completa (sin COVID):**

1. **1993:** después de la recesión de principios de los 90, el empleo **empieza a crecer**.
2. **2001:** un **freno chiquito** (recesión de 2001 y el 11 de septiembre, que golpeó al turismo).
3. **2004:** **acelera** otra vez.
4. **Noviembre 2007:** **el gran cambio**. La tendencia pasa de subir a **bajar**: la **gran recesión**.
5. **2010–2012:** tres aceleraciones seguidas: la **recuperación**, cada vez más fuerte.
6. **2016–2017:** **dos frenos**. El empleo en hotelería de California **deja de crecer**, y se queda en unos 2 000. Esa tendencia estable es la que sigue en el pronóstico, igual que dice el profesor sobre los años recientes.

**Qué decir:** "La documentación de Prophet dice: los datos atípicos, mejor quitarlos. Borramos los meses del COVID y Prophet vuelve a ver todo: 11 cambios, con la crisis de 2008 como el más grande, y la sombra se achica casi 6 veces. Ahora el gráfico cuenta la misma historia que el profesor: estable a principios de los 90, muchos cambios chicos en los 90 y 2000, el gran cambio de 2008 y una tendencia estable en los últimos años."

---

## Cierre

**Qué decir:** "En resumen: Prophet no solo predice, también nos dice **cuándo cambió la tendencia**, incluso cuando el cambio es tan pequeño que no lo vemos. Pero con los datos actualizados aprendimos algo más: un evento extremo como el COVID puede **confundir** al modelo y esconder todos esos cambios. La solución que recomienda la documentación es **quitar los datos atípicos**, y ahí Prophet vuelve a encontrar la crisis de 2008 y los demás cambios. En la próxima clase vemos la **estacionalidad multiplicativa**."

---

## Posibles preguntas del público

**"¿Por qué no hay una línea punteada en el COVID?"**
Porque Prophet solo busca puntos de cambio en el primer 80 % de los datos (hasta abril de 2019). Además, el COVID fue una **caída temporal**, no un cambio permanente de la tendencia: el empleo volvió al nivel de antes.

**"¿No es trampa borrar datos?"**
No, si se explica. El COVID fue algo **excepcional** que no se va a repetir cada año. Si lo dejamos, el modelo aprende cosas falsas: que la tendencia es una recta y que los datos son muy ruidosos. La documentación de Prophet recomienda justamente esto para los datos atípicos.

**"¿Se puede hacer que encuentre más o menos cambios?"**
Sí, con `Prophet(changepoint_prior_scale=...)`. Por defecto es 0.05. Más alto = más flexible (más cambios); más bajo = más rígido.

**"¿Puedo decirle yo dónde están los cambios?"**
Sí: `Prophet(changepoints=['2007-11-01'])`.

---

## Todo el código de esta clase

```python
import pandas as pd
from prophet import Prophet                        # en el video: from fbprophet import Prophet
from prophet.plot import add_changepoints_to_plot  # en el video: from fbprophet.plot import ...
%matplotlib inline

# Datos
df = pd.read_csv('Data/HospitalityEmployees.csv')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df.plot(x='ds', y='y', figsize=(12, 10))

# Modelo
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=12, freq='MS')
forecast = m.predict(future)

# Puntos de cambio
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# Extra: sin COVID
df_sin = df.copy()
df_sin.loc[(df_sin['ds'] >= '2020-03-01') & (df_sin['ds'] <= '2021-12-01'), 'y'] = None
m_sin = Prophet()
m_sin.fit(df_sin)
forecast_sin = m_sin.predict(m_sin.make_future_dataframe(periods=12, freq='MS'))
fig = m_sin.plot(forecast_sin)
a = add_changepoints_to_plot(fig.gca(), m_sin, forecast_sin)
```
