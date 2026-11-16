# Todos los gráficos de la Clase 4, explicados desde cero

Esta guía explica **en detalle** cada gráfico del notebook `04_Prophet_Estacionalidad_VentasInternet.ipynb`, con el **paso 3** explicado línea por línea.

Todos los números son **reales**: salen de los datos y de las tablas `forecast` de cada modelo.

| Paso | Gráfico | Qué muestra |
|---|---|---|
| 2 | Los datos | Las ventas por internet mes a mes |
| 3 | 2 gráficos de "olas" | ¿Las temporadas crecen con los años? |
| 4 | `m.plot(forecast)` aditivo | Datos + predicción del modelo aditivo |
| 5 | `m.plot_components` aditivo | Las piezas del modelo aditivo |
| 6 | Puntos de cambio | Dónde se acelera la tendencia |
| 7 | `m.plot(forecast)` multiplicativo | Datos + predicción del multiplicativo |
| 8 | `m.plot_components` multiplicativo | Las piezas del multiplicativo |
| 10 | Los dos con zoom | Años 90 vs. años recientes |

---

## Antes de empezar: 5 ideas que necesitas

### 1. ¿Qué hay en los datos?

Cada fila es **un mes**. El número `y` son las **ventas de ese mes** de las empresas que venden **sin tienda física** (por internet o por catálogo) en EE. UU., **en millones de dólares**.

- `2025-12-01 → 139 317` significa: **en todo diciembre de 2025 se vendieron 139 317 millones de dólares** por internet y catálogo.
- La fecha dice "día 1" solo para marcar el mes.

### 2. Cuánto creció

| Año | Venta promedio por mes |
|---|---|
| 1992 | 2 882 |
| 2000 | 9 374 |
| 2010 | 21 675 |
| 2019 | 53 287 |
| 2025 | 106 540 |

**Se multiplicó por 37.** En 1992 casi todo era venta por **catálogo** (por correo o teléfono): internet casi no existía. Hoy casi todo es **online**.

### 3. ¿Qué aprende Prophet?

Dos piezas:

- **La tendencia:** cuánto se vende "de base", y cómo crece con los años.
- **La estacionalidad:** cuánto **más** o **menos** según el mes del año.

```
predicción = tendencia + efecto del mes            ← modelo ADITIVO
predicción = tendencia × (1 + efecto del mes)      ← modelo MULTIPLICATIVO
```

### 4. La diferencia en una frase

- **Aditivo:** "diciembre suma **9 732 millones**". Siempre la misma cantidad, en 1992 y en 2025.
- **Multiplicativo:** "diciembre suma un **26.7 %**". Si la tendencia crece, el efecto crece.

### 5. Un dato por mes, fechado el día 1

En los gráficos de piezas (`plot_components`), **solo valen los valores del día 1 de cada mes**. Las ondas entre un día 1 y el siguiente son un efecto del dibujo (como una **cuerda floja** sujeta en 12 puntos que hace panza entre ellos).

---

# PASO 2: los datos

```python
df.plot(x='ds', y='y', figsize=(12, 5), legend=False, title='Ventas por internet y catálogo (millones de USD por mes)')
```

| Eje | Qué muestra |
|---|---|
| Horizontal | La fecha (1992 a 2026) |
| Vertical | Ventas del mes, en millones de dólares |

## Qué se ve

1. **Una curva que se empina cada vez más.** En los 90 parece casi plana (porque 3 000 al lado de 140 000 se ve como 0). Desde 2010 sube más rápido, y desde 2017 muchísimo más.
2. **Una aguja cada diciembre.** Son las compras de Navidad, junto con el *Black Friday* y el *Cyber Monday* de fines de noviembre.
3. **Las agujas se hacen cada vez más grandes.** En los 90 casi no se notan; en 2020–2025 son enormes. **Esta es la pista visual de la estacionalidad multiplicativa.**
4. **2020: un salto.** Con el COVID y las tiendas cerradas, la gente compraba todo online. En abril de 2020 se vendieron 68 109, cuando en abril de 2019 se habían vendido 49 165 (+39 %).

**Qué decir:** "Las ventas por internet se multiplicaron por 37 desde 1992. Y miren las agujas de Navidad: al principio casi no se ven, y al final son gigantes. Las olas crecen junto con las ventas."

---

# PASO 3: ¿Las olas crecen con el tiempo?

Este paso **no es de Prophet**: es pandas puro. Sirve para **decidir con números** si conviene aditivo o multiplicativo **antes** de entrenar nada.

## La idea

Una **"ola"** es la diferencia entre el **mejor mes** y el **peor mes** de un año. Mide **qué tan grande es el efecto de las temporadas** ese año.

- Si las olas tienen **siempre el mismo tamaño en millones**, la estacionalidad es **aditiva**.
- Si las olas son **siempre el mismo porcentaje** de las ventas, la estacionalidad es **multiplicativa**.

## El código, línea por línea

```python
anual = df.groupby(df['ds'].dt.year)['y'].agg(['min', 'max', 'mean'])
```

- `df['ds'].dt.year` saca **el año** de cada fecha: `2025-12-01` → `2025`.
- `df.groupby(...)` **agrupa** las filas por año: los 12 meses de 1992 juntos, los 12 de 1993 juntos, etc.
- `['y']` toma solo la columna de ventas.
- `.agg(['min', 'max', 'mean'])` calcula **para cada año**: el mes **más bajo** (`min`), el **más alto** (`max`) y el **promedio** (`mean`).

Resultado: una tabla con **una fila por año**.

```python
anual = anual[df.groupby(df['ds'].dt.year).size() == 12]
```

- `.size()` cuenta **cuántos meses** tiene cada año.
- `== 12` se queda solo con años **completos**. 2026 tiene 8 meses (hasta agosto) y se descarta: le falta diciembre, así que su "ola" saldría mal.

```python
anual['ola'] = anual['max'] - anual['min']
```

- Columna nueva: **mejor mes − peor mes**. El tamaño de la ola **en millones de dólares**.

```python
anual['ola_%'] = (anual['ola'] / anual['mean'] * 100).round(1)
```

- Divide la ola por el promedio del año y multiplica por 100: el tamaño de la ola **en porcentaje**.
- `.round(1)` redondea a un decimal.

```python
anual.round(0)
```

- Muestra la tabla redondeada, sin decimales.

### Ejemplo calculado a mano: 1992 y 2025

**1992:** mejor mes 4 446 (diciembre), peor mes 2 245 (febrero), promedio 2 882.

- Ola = 4 446 − 2 245 = **2 201** millones.
- Ola % = 2 201 ÷ 2 882 × 100 = **76 %**.

**2025:** mejor mes 139 317 (diciembre), peor mes 90 909 (febrero), promedio 106 540.

- Ola = 139 317 − 90 909 = **48 408** millones.
- Ola % = 48 408 ÷ 106 540 × 100 = **45 %**.

**En palabras:** "En 1992, entre diciembre y febrero había 2 200 millones de diferencia. En 2025, 48 400 millones: **22 veces más**. Pero en porcentaje pasó de 76 % a 45 %: **bajó un poco, no explotó**."

## La tabla

| Año | Promedio | Ola (millones) | Ola % |
|---|---|---|---|
| 1992 | 2 882 | 2 201 | 76 % |
| 1995 | 4 314 | 3 433 | 80 % |
| 2000 | 9 374 | 5 394 | 58 % |
| 2005 | 14 456 | 8 871 | 61 % |
| 2010 | 21 675 | 14 315 | 66 % |
| 2015 | 34 290 | 24 275 | 71 % |
| 2018 | 47 573 | 21 911 | 46 % |
| 2019 | 53 287 | 34 376 | 64 % |
| 2020 | 71 142 | 49 026 | 69 % |
| 2022 | 88 535 | 34 031 | 38 % |
| 2025 | 106 540 | 48 408 | 45 % |

## Los 2 gráficos del paso 3

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
anual['ola'].plot(ax=axes[0], marker='o', title='...')
anual['ola_%'].plot(ax=axes[1], marker='o', color='tab:orange', title='...')
axes[1].set_ylim(0, 100)
```

- `plt.subplots(1, 2, ...)` crea **dos gráficos lado a lado**: `axes[0]` a la izquierda y `axes[1]` a la derecha.
- `marker='o'` pone un **punto por año**.
- `axes[1].set_ylim(0, 100)` hace que el eje del gráfico de la derecha vaya de **0 % a 100 %**. Así se ve en su escala real y no se exageran los altibajos.

### Gráfico izquierdo (azul): la ola en millones

| Eje | Qué muestra |
|---|---|
| Horizontal | El año (1992 a 2025) |
| Vertical | El tamaño de la ola, en millones de dólares |

**Qué se ve:**

1. **Una curva que sube y sube:** de unos **2 200** en 1992 a unos **48 400** en 2025. **La ola se multiplicó por 22.**
2. **Sube parecido a como suben las ventas** (gráfico del paso 2): despacio en los 90, rápido desde 2010, muy rápido desde 2017.
3. **Un bajón en 2018:** ese año diciembre (61 884) vendió **menos** que diciembre de 2017 (63 652), algo raro en los datos.
4. **El pico de 2020** (49 026): el COVID hizo que todo el año 2020 fuera muy alto en ventas online.

**Conclusión:** la ola **no** tiene siempre el mismo tamaño. **No es aditivo.**

### Gráfico derecho (naranja): la ola en porcentaje

| Eje | Qué muestra |
|---|---|
| Horizontal | El año |
| Vertical | El tamaño de la ola, en % del promedio del año (de 0 a 100) |

**Qué se ve:**

1. **No hay una tendencia a crecer.** La línea se mueve entre **40 % y 85 %**, sube y baja, pero **no explota** como la azul.
2. **Una leve bajada en los últimos años** (de ~65 % a ~40–45 %). Tiene sentido: hoy se compra online **todo el año**, no solo para Navidad, así que diciembre "destaca" un poco menos.

**Conclusión:** la ola en % se mantiene en un rango parecido. **Se acerca más a multiplicativo.**

## Conclusión del paso 3

| | Ola en millones | Ola en % |
|---|---|---|
| ¿Cambia con los años? | **Se multiplica por 22** | Se mueve entre 40 % y 85 %, sin crecer |
| ¿Qué indica? | No es aditivo | Es (más o menos) multiplicativo |

**No es multiplicativo perfecto** (el % bajó un poco), pero está **mucho más cerca** de multiplicativo que de aditivo. Lo confirmamos con el RMSE en el paso 9.

**Qué decir:** "Antes de entrenar, medimos. A la izquierda, el tamaño de la ola de cada año en millones: se multiplicó por 22, así que no es aditivo. A la derecha, el mismo tamaño en porcentaje: se mueve entre 40 % y 85 %, sin crecer. La temporada es un porcentaje de las ventas: eso es multiplicativo."

---

# PASO 4: el modelo aditivo, `m.plot(forecast)`

```python
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(50, freq='MS')
forecast = m.predict(future)
fig = m.plot(forecast)
```

Entrena con todos los datos y predice **50 meses** (hasta **octubre 2030**).

## Ejes y elementos

| Elemento | Qué es |
|---|---|
| **Eje horizontal** | La fecha, de 1992 a 2031 |
| **Eje vertical** | Ventas del mes, en millones de dólares |
| ⚫ **Puntos negros** | Lo que **de verdad** se vendió (416 meses) |
| 🔵 **Línea azul** | Lo que el modelo **predice** (`yhat`) |
| 🟦 **Sombra celeste** | El rango probable (`yhat_lower` a `yhat_upper`) |

## Lo que salta a la vista

### 1. En los 90, la línea azul hace agujas ENORMES

Mira 1992–2000: los puntos negros están casi pegados al piso (2 000–10 000), pero la línea azul hace **agujas hasta 14 000–20 000** cada diciembre.

| Diciembre | Real ⚫ | Aditivo 🔵 | El modelo predice… |
|---|---|---|---|
| 1992 | 4 446 | **13 670** | **3 veces** lo real |
| 1993 | 5 446 | **14 068** | **2.6 veces** lo real |
| 2000 | 13 382 | **20 088** | 1.5 veces lo real |

### 2. En los 90, la línea azul BAJA DE CERO

Mira los eneros y febreros de 1992–1993: la línea azul queda **por debajo del 0**.

| Mes | Real ⚫ | Aditivo 🔵 |
|---|---|---|
| enero 1992 | 2 644 | **−102** |
| febrero 1992 | 2 245 | **−1 610** |
| febrero 1993 | 2 690 | **−1 845** |

**¡Ventas negativas!** No existen. Es un error claro del modelo.

### 3. En los años recientes, las agujas azules se quedan CORTAS

Mira 2020–2025: los puntos negros de diciembre quedan **por encima** de la línea azul y de la sombra.

| Diciembre | Real ⚫ | Aditivo 🔵 | Diferencia |
|---|---|---|---|
| 2020 | 99 366 | 82 514 | **16 852 corto** |
| 2024 | 131 582 | 116 056 | **15 526 corto** |
| 2025 | 139 317 | 124 046 | **15 271 corto** |

### ¿Por qué pasan las 3 cosas?

Porque el modelo aditivo **suma y resta las mismas cantidades todos los años**:

- Cada **diciembre suma ≈ +9 700**. En 1992 la tendencia era ≈ 2 800: sumarle 9 700 da ≈ 12 500, **4 veces** lo normal. En 2025 la tendencia era ≈ 113 500: sumarle 9 700 es **apenas un 9 %**, y lo real fue un 23 % más.
- Cada **febrero resta ≈ 4 000**. En 1992–1993 la tendencia era ≈ 2 100–2 900: **restarle 4 000 da negativo**.

**Ejemplo simple:** es como dar a todos los empleados un **bono fijo de 1 000 dólares en Navidad**. Para quien gana 500 al mes es enorme (3 veces su sueldo); para quien gana 50 000 es nada. Un **bono del 10 %** sería justo para los dos.

### 4. La sombra tiene siempre el mismo ancho

La sombra mide unos **9 900** de ancho **en todo el gráfico**. En 1992 eso es **más de 3 veces** lo que se vendía en un mes; en 2025, menos del 10 %. Es otra señal de que "lo mismo para todos los años" no funciona con datos que crecen tanto.

### 5. El pronóstico (2026 → 2030)

A la derecha, donde no hay puntos negros, la línea azul sigue subiendo con sus agujas, hasta unos **157 600** en diciembre de 2029. Las agujas del pronóstico tienen **el mismo tamaño** que las de 1992.

## Qué decir

"Este es el modelo aditivo. Miren los años 90: los puntos negros están casi en el piso, pero la línea azul hace agujas enormes cada diciembre, tres veces más altas que la realidad, y en febrero baja de cero: ventas negativas, que no existen. Y en los años recientes pasa lo contrario: las agujas se quedan cortas. Todo por lo mismo: el modelo suma y resta las mismas cantidades todos los años."

---

# PASO 5: las piezas del modelo aditivo, `m.plot_components(forecast)`

```
línea azul = tendencia (arriba) + efecto del mes (abajo)
```

## Gráfico de arriba: la tendencia

| Eje | Qué muestra |
|---|---|
| Horizontal | La fecha, de 1992 a 2031 |
| Vertical | Ventas "de base", en millones: lo que se vendería un mes **sin** temporadas |

**Qué se ve:** una curva que empieza casi plana y se **empina cada vez más**, hasta quedar como una **recta muy inclinada** desde 2019.

| Tramo | Tendencia | Crecimiento por año |
|---|---|---|
| 1992 → 2002 | de 2 043 a 10 125 | **+808** |
| 2002 → 2012 | de 10 125 a 25 309 | **+1 519** |
| 2012 → 2017 | de 25 309 a 41 323 | **+3 201** |
| 2017 → 2019 | de 41 323 a 55 541 | **+7 114** |
| 2019 → 2030 | de 55 541 a 154 055 | **+8 386** (recta) |

**¿Por qué es recta desde 2019?** Porque Prophet **solo busca cambios de tendencia en el primer 80 % de los datos**, que aquí llega hasta **agosto de 2019**. Después, la tendencia sigue con la última inclinación.

**Qué decir (arriba):** "Las ventas 'de base' pasaron de 2 000 a 119 000 millones por mes. Y cada vez crecen más rápido: en los 90 sumaban 800 millones por año; desde 2019, más de 8 000 por año."

## Gráfico de abajo: el efecto del mes (aditivo)

| Eje | Qué muestra |
|---|---|
| Horizontal | **Los días de UN año cualquiera**, del 1 de enero al 31 de diciembre. **No son años.** |
| Vertical | **Cuántos millones se suman o restan** a la tendencia ese día del año |

- **0** = un mes "normal".
- **Arriba de 0** = se vende **más** que la tendencia.
- **Abajo de 0** = se vende **menos**.

Las marcas del eje están **cada 2 meses**. **Diciembre 1** está en la mitad entre "November 1" y el último "January 1"; **febrero 1**, en la mitad entre "January 1" y "March 1".

### ⚠️ Las ondas que NO son reales

La curva hace un **pozo de −16 110** (≈ **18 de marzo**) y un **pico de +20 792** (≈ **12 de diciembre**). **Ninguno de los dos es real.** La curva solo está "sujeta" en los **12 días 1** del año; entre medio se mueve libremente.

**¿Por qué el pico cae el 12 de diciembre y no el 1?** Porque la curva tiene que subir de **+3 863** (1 de noviembre) a **+9 732** (1 de diciembre) y después bajar a **−2 212** (1 de enero). Una curva hecha de ondas suaves **no puede hacer una punta filosa** en un día exacto: hace una "loma" redonda, y la loma queda un poco después del 1 de diciembre. **El valor real de diciembre es el del día 1: +9 732.**

### Los valores reales (día 1 de cada mes)

| Mes | Efecto | En palabras |
|---|---|---|
| Enero | −2 212 | Después de las fiestas se compra menos |
| **Febrero** | **−4 068** | **El más bajo** (además es el mes más corto) |
| Marzo | −574 | |
| Abril | −1 870 | |
| Mayo | −702 | |
| Junio | −1 711 | |
| Julio | −1 506 | |
| Agosto | −960 | |
| Septiembre | −2 094 | |
| Octubre | −260 | |
| Noviembre | **+3 863** | *Black Friday* y *Cyber Monday* |
| **Diciembre** | **+9 732** | **Navidad: el más alto** |

> **Por qué casi todos los meses son negativos:** porque noviembre y diciembre suman mucho (+13 600 entre los dos). Los otros 10 meses tienen que compensar, quedando un poco por debajo, para que el año "promedie" 0.

**Detalle clave del aditivo:** estos números son **los mismos para todos los años**. Diciembre suma **9 732** en 1992 y en 2025.

## Qué decir

"Arriba, la tendencia: crece cada vez más rápido. Abajo, el efecto de cada mes, en millones: diciembre suma 9 700, noviembre 3 900 y febrero resta 4 000. Son los mismos números para 1992 y para 2025, y ese es el problema. Ojo: el pico del 12 de diciembre y el pozo de marzo son efectos del dibujo; solo valen los días 1."

---

# PASO 6: los puntos de cambio

```python
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
```

Es el **gráfico del paso 4** con dos cosas **encima**:

| Elemento | Qué es |
|---|---|
| 🔴 **Línea roja continua** | **La tendencia** (la del gráfico de arriba del paso 5) |
| 🔴 **Líneas rojas punteadas** | **Los puntos de cambio**: dónde la tendencia cambia de inclinación |

## Qué se ve

**14 líneas punteadas, TODAS son aceleraciones**: en cada una, la línea roja se pone un poco **más empinada**.

| Época | Líneas | Qué pasaba |
|---|---|---|
| 1999–2005 | 5 | Nacen las tiendas online (Amazon, eBay). Las aceleraciones son pequeñas. |
| 2009–2013 | 4 | Recuperación de la crisis de 2008; empiezan los *smartphones*. |
| **2014–2018** | **5** | **Las aceleraciones más fuertes**: el boom de comprar desde el celular, Amazon Prime, envíos en 1–2 días. |

La última línea es de **julio 2018**. Después, la línea roja sigue **recta**.

> **Comparación con el video:** en los pasajeros de 1949–1960, el profesor ve **5 líneas en los años 50**, por la llegada de los **aviones a reacción**. Aquí también hay una revolución tecnológica: **el comercio electrónico**.

## Qué decir

"Las líneas punteadas son los momentos en que el crecimiento se aceleró. Hay 14, y todas son aceleraciones: el comercio online no paró de crecer cada vez más rápido. Las más fuertes están entre 2014 y 2018, el boom de comprar desde el celular."

---

# PASO 7: el modelo multiplicativo, `m.plot(forecast)`

```python
m = Prophet(seasonality_mode='multiplicative')
```

Cambia **una sola palabra**. Ahora el efecto del mes es un **porcentaje**.

## Qué se ve (compáralo con el paso 4)

### En los 90: las agujas son chiquitas, como las reales

| Mes | Real ⚫ | Aditivo (paso 4) | **Multiplicativo** 🔵 |
|---|---|---|---|
| enero 1992 | 2 644 | −102 ❌ | **1 887** |
| febrero 1992 | 2 245 | −1 610 ❌ | **1 830** |
| diciembre 1992 | 4 446 | 13 670 ❌ | **3 495** |
| febrero 1993 | 2 690 | −1 845 ❌ | **2 466** |
| diciembre 1993 | 5 446 | 14 068 ❌ | **4 417** |
| diciembre 2000 | 13 382 | 20 088 | **11 648** |

- **Ya no hay ventas negativas.** La predicción más baja es **1 830** (febrero 1992).
- Las agujas de los 90 son **del tamaño correcto**.

### En los años recientes: las agujas son enormes y llegan a los puntos

| Diciembre | Real ⚫ | Aditivo | **Multiplicativo** 🔵 |
|---|---|---|---|
| 2019 | 77 951 | 72 941 | **79 800** ✅ |
| 2020 | 99 366 | 82 514 | **92 383** (más cerca) |
| 2024 | 131 582 | 116 056 | **136 084** (se pasa un poco) |
| 2025 | 139 317 | 124 046 | **146 031** (se pasa un poco) |

**Las olas crecen con el tiempo**, igual que en los datos reales.

> En 2024–2025 el multiplicativo **se pasa un poco**. ¿Por qué? Porque la ola en % **bajó** en los últimos años (paso 3: de ~65 % a ~45 %), y el modelo usa un porcentaje **promedio** de toda la historia. Se pasa unos 5 000–7 000, mientras que el aditivo se quedaba corto unos 15 000.

### El pronóstico

El pico de **diciembre 2029** llega a unos **189 400** (contra 157 600 del aditivo). En el multiplicativo, las agujas del futuro son **mucho más grandes**, porque el 26.7 % de diciembre se aplica sobre una tendencia que sigue creciendo.

### La sombra

Mide unos **6 650** de ancho (más angosta que los 9 900 del aditivo), porque el modelo se equivoca menos. Igual tiene el mismo ancho en todo el gráfico.

## Qué decir

"Con una sola palabra, todo cambia. Compárenlo con el aditivo: en los 90 ya no hay agujas gigantes ni ventas negativas; las olas son chiquitas, como las reales. Y en los años recientes las agujas son grandes y llegan a los puntos de Navidad. Las olas crecen con las ventas, igual que en la realidad."

---

# PASO 8: las piezas del modelo multiplicativo

## Gráfico de arriba: la tendencia

**Casi idéntica a la del aditivo** (de 2 027 en 1992 a 119 279 en agosto 2026). Con 12 puntos de cambio, todos aceleraciones. **La tendencia no cambia mucho entre los dos modelos: lo que cambia es la estacionalidad.**

## Gráfico de abajo: el efecto del mes, AHORA EN %

**La gran diferencia:** el eje vertical ya no dice "millones", dice **%** (de −80 % a +60 %).

| Mes | Efecto | En palabras |
|---|---|---|
| Enero | −7.0 % | Se vende un 7 % menos que la tendencia |
| **Febrero** | **−12.6 %** | **El más bajo** |
| Marzo | −0.7 % | |
| Abril | −4.4 % | |
| Mayo | −0.5 % | |
| Junio | −3.5 % | |
| Julio | −2.8 % | |
| Agosto | −2.0 % | |
| Septiembre | −6.5 % | |
| Octubre | −0.3 % | |
| Noviembre | **+13.2 %** | *Black Friday* |
| **Diciembre** | **+26.7 %** | **Navidad: el más alto** |

> Las mismas ondas falsas: el pozo de **−55 %** (≈ 16 de marzo) y el pico de **+45 %** (≈ 10 de diciembre) **no son reales**. Solo valen los días 1.

**La forma es igual a la del aditivo** (febrero abajo, noviembre y diciembre arriba). Lo que cambia es la **unidad**, y eso cambia **cómo crece la ola**:

| | Tendencia | Diciembre aditivo | Diciembre multiplicativo |
|---|---|---|---|
| 1992 | ≈ 2 700 | + 9 732 (¡más que la tendencia!) | + 26.7 % = **+ 720** |
| 2010 | ≈ 23 000 | + 9 732 | + 26.7 % = **+ 6 100** |
| 2025 | ≈ 113 700 | + 9 732 | + 26.7 % = **+ 30 400** |

**En el aditivo, el "extra de Navidad" es siempre 9 732. En el multiplicativo, va de 720 a 30 400: crece con las ventas.** Eso es lo que muestra el paso 3.

## Qué decir

"Las piezas del multiplicativo se ven casi iguales, pero fíjense en el eje de abajo: ya no dice millones, dice porcentaje. Diciembre suma un 27 %. Y un 27 % de 2 700 son 720 millones, pero un 27 % de 113 000 son 30 000 millones. Así crece la ola."

---

# PASO 9 (tabla): ¿cuál es mejor?

Escondimos el último año (sep 2025 → ago 2026), entrenamos los dos modelos con el resto y comparamos:

| Modo | Error (RMSE) | Error % |
|---|---|---|
| Aditivo | 8 847 | 7.90 % |
| **Multiplicativo** | **6 508** | **5.81 %** |

**El multiplicativo se equivoca un 26 % menos.**

---

# PASO 10: los dos modelos con zoom

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# izquierda: 1992–1996    derecha: 2021–2026
```

Dos gráficos lado a lado, **sin pronóstico ni sombra**, para comparar los modelos con la realidad:

| Elemento | Qué es |
|---|---|
| ⚫ Puntos negros | Lo real |
| 🔵 Línea azul | **Aditivo** |
| 🟠 Línea naranja | **Multiplicativo** |

Cada gráfico tiene **su propia escala vertical**: el izquierdo va de −2 500 a 17 500 y el derecho de 60 000 a 145 000. Por eso los dos se ven "del mismo tamaño", aunque las ventas de 2025 son 40 veces las de 1992.

## Izquierda: 1992–1996 (el gráfico más claro de toda la clase)

- Los **puntos negros** suben despacio, de unos 2 500 a unos 7 700, con una **subidita** en cada diciembre.
- La **línea azul (aditivo)** está **loca**:
  - Hace **agujas de ~14 000–17 000** cada diciembre, cuando lo real es de 4 000–7 700.
  - **Baja de cero** en los inviernos de 1992 y 1993 (hasta −1 845).
- La **línea naranja (multiplicativo)** va **pegada a los puntos negros**, con olas chiquitas del tamaño correcto.

## Derecha: 2021–2026

- Los **puntos negros** van de unos 70 000 a 140 000, con **agujas grandes** cada diciembre.
- La **línea azul (aditivo)** se queda **corta** en cada Navidad: llega a unos 116 000–124 000 cuando lo real es 131 000–139 000.
- La **línea naranja (multiplicativo)** **llega a los picos** (y en 2024–2025 se pasa un poco).
- En los meses "normales", las dos líneas andan parecido.

## Qué decir

"Este gráfico lo resume todo. A la izquierda, los años 90: la línea azul, el aditivo, inventa picos tres veces más grandes que la realidad y ventas negativas; la naranja, el multiplicativo, va pegada a los datos. A la derecha, los años recientes: el aditivo se queda corto en Navidad y el multiplicativo llega. El aditivo falla en las dos puntas; el multiplicativo anda bien en las dos."

---

# PASO 11 (tabla): el error de cada modelo por época

| Época | Error aditivo | Error multiplicativo |
|---|---|---|
| **1992–1996** | **50.7 %** | **11.7 %** |
| 1997–2005 | 17.0 % | 5.9 % |
| 2006–2015 | 5.2 % | 4.5 % |
| 2016–2026 | 6.5 % | 5.4 % |

- **Donde más se nota es al principio:** el aditivo se equivoca **la mitad** del valor real; el multiplicativo, un 12 %.
- En **2006–2015** los dos andan parecido: es la época "del medio", donde el efecto fijo del aditivo (9 732) se parece bastante al 27 % del multiplicativo.
- **El multiplicativo gana en todas las épocas.**

---

## Resumen en 7 frases

1. **Paso 2:** las ventas por internet se multiplicaron por 37, y las agujas de Navidad crecen con ellas.
2. **Paso 3:** la ola de cada año en millones se multiplicó por 22, pero en % se mantuvo entre 40 % y 85 %. Eso es **multiplicativo**.
3. **Paso 4:** el aditivo hace agujas 3 veces más grandes que lo real en los 90, predice **ventas negativas** en febrero de 1992–1993, y se queda corto 15 000 en los diciembres recientes.
4. **Paso 5:** el aditivo suma **+9 732** cada diciembre y resta **−4 068** cada febrero, **iguales todos los años**. Solo valen los días 1 del gráfico.
5. **Paso 6:** 14 puntos de cambio, todos aceleraciones; las más fuertes en 2014–2018 (el boom de comprar desde el celular).
6. **Pasos 7 y 8:** el multiplicativo suma **+26.7 %** cada diciembre: 720 millones en 1992, 30 400 en 2025. Las olas crecen y siguen a los datos.
7. **Pasos 9 a 11:** el multiplicativo gana en todo: **5.8 % de error contra 7.9 %** en el último año, y **12 % contra 51 %** en los primeros años.
