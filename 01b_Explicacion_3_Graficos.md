# Los 3 gráficos de la Clase 1, explicados desde cero

Esta guía explica **en detalle** los tres gráficos del notebook `01_Prophet_Basico_BeerWineLiquor.ipynb`:

1. **Paso 12:** `m.plot(forecast)`, el gráfico completo.
2. **Paso 13:** el mismo gráfico con **zoom** de 2019 a 2029.
3. **Paso 14:** `m.plot_components(forecast)`, las **piezas** del modelo.

Todos los números son **reales**: salen de la tabla `forecast` y de los datos.

---

## Antes de empezar: 4 ideas que necesitas

### 1. ¿Qué hay en los datos?

Cada fila es **un mes**. El número `y` son las **ventas de ese mes completo** en las tiendas de cerveza, vino y licor de EE. UU., **en millones de dólares**.

- `2022-12-01 → 7 848` significa: **en todo diciembre de 2022 se vendieron 7 848 millones de dólares**.
- La fecha dice "día 1" solo para marcar el mes. No es lo vendido el día 1.

### 2. ¿Qué hizo Prophet?

Prophet miró los 416 meses reales (enero 1992 → agosto 2026) y aprendió **dos cosas**:

- **La tendencia:** cuánto venden "de base", y cómo eso crece con los años.
- **La estacionalidad:** cuánto **más** o **menos** se vende según el mes del año (diciembre mucho más, enero menos, etc.).

Con eso arma una **predicción para cada mes**, sumando:

```
predicción (yhat) = tendencia (trend) + efecto del mes (yearly)
```

### 3. ¿Qué es la tabla `forecast`?

Es la tabla con las predicciones. Tiene **una fila por mes**: los 416 meses que ya pasaron **y** los 24 meses del futuro (septiembre 2026 → agosto 2028). Para cada mes tiene:

| Columna | Qué es |
|---|---|
| `yhat` | La predicción |
| `yhat_lower` | El mínimo probable |
| `yhat_upper` | El máximo probable |
| `trend` | La tendencia en ese mes |
| `yearly` | El efecto del mes del año |

**Los 3 gráficos son solo dibujos de esta tabla.**

### 4. ¿Por qué Prophet predice también el pasado?

Para que **compares**: si en los meses pasados la predicción queda cerca de lo real, el modelo aprendió bien. Si queda lejos, algo no entendió.

---

# GRÁFICO 1: `m.plot(forecast)`, el gráfico completo

## ¿Qué hay en los ejes?

| Eje | Qué muestra | Rango |
|---|---|---|
| **Horizontal (`ds`)** | La **fecha** | De 1992 a 2029 |
| **Vertical (`y`)** | Las **ventas del mes**, en millones de dólares | De unos 1 000 a 8 000 |

Cada **posición horizontal** es un mes. Cuanto **más arriba**, **más se vendió** ese mes.

## Los 3 elementos del gráfico

### ⚫ Los puntos negros = la realidad

- Hay **un punto por cada mes real**: 416 puntos, de enero 1992 a agosto 2026.
- Su altura es **lo que de verdad se vendió**.
- **No hay puntos después de agosto 2026**, porque ese futuro todavía no pasó.

### 🔵 La línea azul oscura = la predicción de Prophet (`yhat`)

- Es **lo que el modelo cree** que se vendió (o se venderá) cada mes.
- Va de 1992 hasta **agosto 2028**: cubre el pasado **y** el futuro.
- **No pasa exactamente por los puntos.** No es una línea que "une los puntos": es el **patrón** que Prophet aprendió. Por eso a veces queda arriba y a veces abajo de los puntos.

### 🟦 La sombra celeste = el rango probable

- Va desde `yhat_lower` (borde de abajo) hasta `yhat_upper` (borde de arriba).
- Prophet dice: **"estoy 80 % seguro de que el valor real cae dentro de esta sombra"**.
- Mide unos **620** de ancho (en millones de dólares), **casi igual en todo el gráfico**.
- **¿Funciona?** En los 416 meses reales, el **77.5 %** de los puntos cayó dentro de la sombra. Prometía 80 %, así que está **bastante bien calibrada**.

## Cómo leer un mes cualquiera (con ejemplo)

Imagina una **línea vertical** en un mes. Esa línea toca tres cosas: el punto negro, la línea azul y la sombra. Comparas.

**Ejemplo: diciembre de 2022**

| Elemento | Valor |
|---|---|
| ⚫ Punto negro (real) | **7 848** |
| 🔵 Línea azul (predicción) | **6 762** |
| 🟦 Sombra (rango) | de **6 427** a **7 100** |

**Lectura:** "En diciembre de 2022 Prophet esperaba vender unos 6 762 millones, y como mucho 7 100. Se vendieron **7 848**: el punto quedó **por encima de la sombra**. Ese mes **el modelo se quedó corto** por unos 1 100 millones."

**Otro ejemplo: diciembre de 2019**

| Elemento | Valor |
|---|---|
| ⚫ Real | **6 241** |
| 🔵 Predicción | **6 287** |
| 🟦 Rango | 5 957 a 6 598 |

**Lectura:** "El punto cae **justo sobre la línea**, dentro de la sombra. **Acertó casi exacto.**"

## Qué patrones se ven (de lo más general a lo más fino)

### Patrón 1: todo sube → la **tendencia**

De izquierda a derecha, todo va **subiendo**: de unos **1 400–1 800** por mes en 1992 a unos **5 500–6 000** en 2026. Las ventas **casi se cuadruplicaron** en 34 años (por inflación, por crecimiento de la población y porque se compra más).

### Patrón 2: una "aguja" cada año → **diciembre**

Cada año hay **un pico muy alto y angosto**. Es **diciembre**: Navidad y Año Nuevo, cuando se compra mucho más alcohol. Justo después hay una caída: **enero y febrero**, los meses más flojos.

> Por eso la línea azul parece un **serrucho que sube**: sube con los años (tendencia) y hace un diente por año (estacionalidad).

### Patrón 3: al final, la línea azul sigue sola → el **pronóstico**

A la derecha de **agosto 2026** ya no hay puntos negros, pero la línea azul **sigue** hasta **agosto 2028**, con sus picos en diciembre. Eso es **lo que Prophet pronostica**.

## Las 3 cosas "raras" del gráfico (y por qué pasan)

### Rara 1: en los años 90, las agujas azules son **más altas** que los puntos negros

Mira 1992–1999: la línea azul sube en diciembre **más alto** que el punto negro.

| Diciembre | Real ⚫ | Predicción 🔵 | Diferencia |
|---|---|---|---|
| 1992 | 2 354 | **2 805** | el modelo se pasó por **+451** |
| 1994 | 2 416 | **2 948** | se pasó por **+532** |
| 1996 | 2 395 | **3 042** | se pasó por **+647** |

**¿Por qué?** Prophet usa por defecto el modelo **aditivo**: suma **la misma cantidad** cada diciembre, unos **+1 180** millones, sea 1992 o 2025. Pero en 1992 el "salto de diciembre" real era mucho más chico (las ventas eran 4 veces menores). Entonces **en los 90 exagera** el pico. Esto se arregla en la clase 4 con el modelo **multiplicativo**, que suma un **porcentaje** en vez de una cantidad fija.

### Rara 2: de 2020 a 2023, los diciembres reales quedan **muy por encima**

| Diciembre | Real ⚫ | Predicción 🔵 | Máximo de la sombra |
|---|---|---|---|
| 2020 | **7 265** | 6 411 | 6 726 |
| 2021 | **7 652** | 6 587 | 6 887 |
| 2022 | **7 848** | 6 762 | 7 100 |
| 2023 | **7 819** | 6 936 | 7 259 |

Los cuatro puntos están **fuera de la sombra, por arriba**. Es el efecto de la **pandemia**: con bares y restaurantes cerrados, la gente compró alcohol en tiendas para tomar en casa, y los diciembres se dispararon. Prophet no podía saber que venía una pandemia.

### Rara 3: en 2025–2026, los puntos quedan **por debajo** de la línea

Desde 2024 las ventas reales **empezaron a bajar**, pero la línea azul **sigue subiendo**. Lo vemos con detalle en el gráfico 2.

## Qué decir de este gráfico

"Los puntos negros son lo que se vendió de verdad, la línea azul es lo que Prophet entendió, y la sombra es su margen de error. Se ven dos cosas: las ventas suben con los años, y cada diciembre hay un pico enorme por las fiestas. A la derecha, donde ya no hay puntos, la línea sigue sola: ese es el pronóstico hasta 2028. En general acierta bien (casi 8 de cada 10 meses caen dentro de la sombra), pero exagera los diciembres de los 90 y se queda corto en los de la pandemia."

---

# GRÁFICO 2: el zoom de 2019 a 2029

## ¿Qué cambió respecto al gráfico 1?

**Nada del modelo.** Es **el mismo gráfico**, pero con una **lupa**:

```python
plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2029-01-01'))
```

`xlim` = **límites del eje X**. Le dice al gráfico: "mostrá solo de 2019 a 2029".

> **¿Por qué el eje vertical sigue empezando en 1 000?** Porque solo hicimos zoom en el eje **horizontal**. El vertical quedó igual, por eso hay mucho espacio vacío abajo. Si quisieras hacer zoom también en el vertical, se agrega `plt.ylim(4000, 8500)`.

Con la lupa, ahora **sí se ven los meses uno por uno**: cada punto negro es un mes, y entre dos líneas verticales de la cuadrícula hay **12 puntos** (un año).

## Cómo leer un año (por ejemplo, 2024)

Dentro de cada año, la línea azul hace siempre **la misma forma**:

1. **Enero:** cae fuerte (el pozo después de las fiestas).
2. **Febrero a noviembre:** sube despacio, con pequeños altibajos.
3. **Diciembre:** salta hacia arriba (la aguja).
4. Vuelve a caer en enero del año siguiente.

Esa forma que se repite es la **estacionalidad**, y viene del gráfico 3 (abajo).

## La historia año por año

### 2019: el modelo acierta

Los puntos negros están **cerca de la línea azul**, casi todos dentro de la sombra. Diciembre 2019: real **6 241**, predicción **6 287**.

### 2020–2023: la pandemia, los puntos quedan **arriba**

- En **marzo de 2020** las ventas saltan de **4 242** (febrero) a **5 152**: **+21 % en un mes**. Con los bares cerrados, la gente compraba en tiendas.
- Durante 2020, 2021, 2022 y 2023, **muchos puntos negros quedan por encima** de la línea azul.
- Los **diciembres** quedan **muy arriba de la sombra**: 7 265, 7 652, 7 848 y 7 819.

**Qué significa:** se vendió **más** de lo que el modelo esperaba.

### 2024: todo vuelve a la normalidad

Los puntos vuelven a estar **cerca de la línea**:

| Mes 2024 | Real | Predicción |
|---|---|---|
| marzo | 5 617 | 5 614 |
| julio | 6 017 | 5 994 |
| agosto | 5 952 | 5 891 |
| diciembre | 7 493 | 7 060 |

### 2025–2026: los puntos quedan **abajo**

| Mes | Real ⚫ | Predicción 🔵 | Mínimo de la sombra |
|---|---|---|---|
| ene 2025 | 4 824 | 5 416 | 5 100 |
| feb 2025 | 4 807 | 5 361 | 5 049 |
| sep 2025 | 5 273 | 5 879 | 5 570 |
| ene 2026 | 4 979 | 5 565 | 5 251 |
| **feb 2026** | **4 778** | **5 555** | **5 252** |
| ago 2026 | 5 637 | 6 208 | 5 896 |

Muchos puntos quedan **por debajo de la sombra**. Las ventas **bajaron** (el promedio mensual cayó de **5 946 en 2023** a **5 649 en 2025**), pero **la línea azul sigue subiendo**.

**¿Por qué el modelo no se da cuenta?** Por una regla de Prophet: **solo busca cambios de tendencia en el primer 80 % de los datos**. En estos datos, el último lugar donde puede cambiar la tendencia es **agosto de 2019**. Todo lo que pasó después (la subida de la pandemia y la bajada reciente) **no puede cambiar la tendencia**: el modelo sigue con la misma pendiente que tenía en 2019. Lo hace a propósito, para no "sobreactuar" con los últimos datos, pero por eso **tarda en reaccionar**.

### Septiembre 2026 → agosto 2028: el pronóstico

Ya no hay puntos negros. La línea azul **repite la forma** de cada año, cada vez un poco más arriba:

| Mes | Predicción | Rango |
|---|---|---|
| diciembre 2026 | **7 411** | 7 111 a 7 724 |
| enero 2027 | 5 715 | — |
| diciembre 2027 | **7 585** | 7 247 a 7 916 |
| enero 2028 | 5 866 | 5 557 a 6 178 |
| agosto 2028 | 6 540 | 6 218 a 6 863 |

> Ojo: como el modelo no captó la bajada de 2024–2026, **este pronóstico probablemente es optimista**: es posible que las ventas reales queden por debajo.

## Qué decir de este gráfico

"Con la lupa se ven los meses uno por uno. Cada año tiene la misma forma: pozo en enero, subida suave y aguja en diciembre. Y se ve la historia reciente: en 2019 el modelo acierta; con la pandemia la gente compra mucho más alcohol para tomar en casa y los puntos quedan arriba; en 2024 todo vuelve a la normalidad; y en 2025–2026 las ventas bajan, pero el modelo sigue subiendo porque no busca cambios de tendencia en los últimos años. Por eso el pronóstico hasta 2028 probablemente es optimista."

---

# GRÁFICO 3: `m.plot_components(forecast)`, las piezas

Este gráfico **desarma** la línea azul en sus piezas. Recuerda la fórmula:

```
línea azul (yhat) = tendencia (trend) + efecto del mes (yearly)
```

- El **gráfico de arriba** es la **tendencia**.
- El **gráfico de abajo** es el **efecto del mes**.

Si sumas los dos, obtienes la línea azul de los gráficos 1 y 2.

> **¿Por qué solo 2 piezas?** Prophet también puede tener patrones **semanales** y **diarios**, pero los **apagó** porque hay un solo dato por mes: no hay forma de ver qué pasa dentro de una semana o de un día.

---

## Gráfico 3 (arriba): la **tendencia**

### Ejes

| Eje | Qué muestra |
|---|---|
| Horizontal (`ds`) | La fecha, de 1992 a 2028 |
| Vertical (`trend`) | Las ventas **"de base"**, en millones de dólares |

### ¿Qué es la tendencia?

Es **cómo serían las ventas si no existieran las temporadas**: si todos los meses fueran "normales", sin diciembres altos ni eneros bajos. Es el **camino de fondo**.

Por eso es una línea **suave**, sin serrucho: le sacamos el zigzag de cada año.

### Qué valores tiene

| Fecha | Tendencia |
|---|---|
| enero 1992 | 1 608 |
| enero 2000 | 2 158 |
| enero 2010 | 3 264 |
| enero 2015 | 3 918 |
| enero 2020 | 5 119 |
| agosto 2026 (último dato) | 6 187 |
| agosto 2028 (fin del pronóstico) | 6 512 |

### La línea no es recta: cambia de inclinación

La **inclinación** (qué tan empinada está) dice **qué tan rápido crecen las ventas**. Calculando cuánto sube por año en cada tramo:

| Tramo | Crecimiento por año | Cómo se ve en el gráfico |
|---|---|---|
| 1992 → 2000 | **+69** por año | Casi plana, sube despacio |
| 2000 → 2010 | **+111** por año | Un poco más empinada |
| 2010 → 2015 | **+131** por año | Más empinada |
| **2015 → 2020** | **+240** por año | **La parte más empinada** |
| 2020 → 2028 | **+162** por año | **Se dobla** y queda menos empinada |

**Los "dobleces"** donde cambia la inclinación son los **puntos de cambio** (*changepoints*), el tema de la clase 3. Prophet encontró varios: aceleraciones en 1996–1998, 2003–2004 y **2013–2016**, y un **freno fuerte en agosto de 2019**. Ese freno es el doblez que ves cerca de 2020.

**Lo que se ve a la derecha (2026–2028):** la línea sigue **recta**, con la misma inclinación de +162 por año. Prophet supone que el crecimiento de los últimos años va a continuar.

### Qué decir de este gráfico

"Arriba está la tendencia: las ventas 'de base', sin las temporadas. Subió de 1 600 a 6 200 millones por mes en 34 años. No es una recta: crecía despacio en los 90, cada vez más rápido hasta 2020, y ahí se dobla y crece un poco menos. Hacia el futuro, Prophet sigue esa misma línea."

---

## Gráfico 3 (abajo): el **efecto del mes** (estacionalidad anual)

### Ejes (esto es lo que más confunde)

| Eje | Qué muestra |
|---|---|
| **Horizontal (`Day of year`)** | **Los días de UN año cualquiera**, del 1 de enero al 31 de diciembre. **No son años**: es "un año típico". |
| **Vertical (`yearly`)** | **Cuánto se suma o se resta a la tendencia** ese día del año, en millones de dólares |

**Cómo leer el eje vertical:**

- **0** = un mes "normal": no se suma ni se resta nada.
- **Arriba de 0** = ese mes se vende **más** que la tendencia.
- **Abajo de 0** = ese mes se vende **menos** que la tendencia.

**Cómo leer el eje horizontal:** las marcas están cada **2 meses** (January 1, March 1, May 1, July 1, September 1, November 1, January 1). Entonces:

- **Febrero 1** está **en la mitad** entre "January 1" y "March 1".
- **Diciembre 1** está **en la mitad** entre "November 1" y el último "January 1".

### ⚠️ El problema de este gráfico: la curva tiene partes FALSAS

Mira la curva: tiene **muchas ondas**. Hay una caída enorme hasta **−1 471** (alrededor del **13 de febrero**) y un pico de **+1 363** (alrededor del **24 de noviembre**).

**Esos dos extremos NO son reales.** ¿Por qué?

1. Los datos tienen **un solo número por mes**, siempre fechado el **día 1**. En un año hay **solo 12 puntos con datos**: el 1 de enero, el 1 de febrero, …, el 1 de diciembre.
2. Pero Prophet dibuja la curva **para los 365 días del año**. Para hacerlo, arma la curva **sumando muchas ondas suaves** (técnicamente, 10 pares de ondas seno y coseno).
3. La curva **tiene que pasar cerca de los 12 puntos**, pero **entre un punto y el siguiente nadie la controla**: no hay ningún dato del 13 de febrero ni del 24 de noviembre.
4. Entonces la curva **sube y baja libremente** entre los puntos. Esas ondas son **un efecto del dibujo**, no de las ventas.

**Ejemplo simple:** es como unir 12 puntos con una cuerda floja. En cada punto la cuerda está bien sujeta, pero **entre los puntos puede hacer panza** para arriba o para abajo. Lo que importa es dónde está sujeta, no las panzas.

**Regla de oro:** **mira solo el valor de la curva en el día 1 de cada mes.** Lo demás, ignóralo.

### Los valores reales (día 1 de cada mes)

| Mes | Efecto | En palabras |
|---|---|---|
| Enero | **−540** | Se vende 540 millones **menos** que la tendencia (resaca de las fiestas) |
| Febrero | **−521** | Menos (y es el mes más corto: tiene menos días para vender) |
| Marzo | −250 | Algo menos |
| Abril | −226 | Algo menos |
| Mayo | +54 | Normal |
| Junio | +26 | Normal |
| Julio | +156 | Un poco más (verano, 4 de julio) |
| Agosto | +17 | Normal |
| Septiembre | −128 | Algo menos |
| Octubre | −75 | Normal |
| Noviembre | +52 | Normal (Acción de Gracias) |
| **Diciembre** | **+1 182** | **Muchísimo más: Navidad y Año Nuevo** |

**Cómo encontrar estos valores en el gráfico:**

- **Diciembre:** busca la mitad entre "November 1" y el último "January 1". La curva está **bajando**, justo después del gran pico, en unos **+1 180**.
- **Febrero:** busca la mitad entre "January 1" y "March 1". La curva está empezando a caer hacia el pozo falso, en unos **−520**.

### Qué nos dice en simple

1. **Diciembre es, por lejos, el mes más fuerte:** se venden unos **1 200 millones extra**. Es más del doble de lo que se pierde en cualquier otro mes.
2. **Enero y febrero son los más flojos:** unos **530 millones menos** cada uno.
3. **El resto del año está cerca de 0:** de marzo a noviembre las ventas son "normales", con un verano apenas más alto.

> **Importante:** como el modelo es **aditivo**, estos números son **iguales para todos los años**. Prophet suma +1 182 a cada diciembre, sea 1992 o 2027. Por eso exageraba los diciembres de los 90 (gráfico 1, "rara 1").

### Qué decir de este gráfico

"Abajo está el efecto de cada mes. El eje horizontal no son años, es un año cualquiera, de enero a diciembre. El 0 es un mes normal. Hay que tener cuidado: como hay un dato por mes, solo valen los puntos del día 1; las ondas del medio son un efecto del dibujo. Leído bien, dice algo muy lógico: diciembre suma unos 1 200 millones por las fiestas, enero y febrero restan unos 500, y el resto del año es casi normal."

---

## Cómo se juntan las piezas (la prueba final)

Ahora puedes **reconstruir la línea azul a mano**, sumando arriba + abajo:

| Mes | Tendencia (arriba) | + Efecto del mes (abajo) | = Línea azul (gráfico 1 y 2) |
|---|---|---|---|
| **Diciembre 2027** | 6 403 | **+1 182** | **7 585** ← la aguja de 2027 |
| **Enero 2027** | 6 255 | **−540** | **5 715** ← el pozo de 2027 |
| Junio 2027 | 6 322 | +26 | 6 348 ← un mes normal |

Si miras el gráfico 2, en diciembre de 2027 la línea azul llega justo a unos **7 585**, y en enero de 2027 baja a unos **5 715**. **Cuadra perfecto.**

**Esto es lo más importante de todo:** la línea azul **no es magia**. Es **la tendencia** (una línea suave que sube) **más** un **patrón que se repite cada año** (diciembre arriba, enero abajo). Por eso parece un **serrucho que sube**.

---

## Resumen en 5 frases

1. **Gráfico 1:** puntos negros = lo que se vendió, línea azul = lo que Prophet cree, sombra = su margen de error (acierta casi 8 de cada 10 meses).
2. **Gráfico 2:** es el mismo con lupa. Se ve que en la pandemia se vendió más de lo esperado, y que desde 2025 se vende menos, pero el modelo no se dio cuenta.
3. **Gráfico 3 arriba (tendencia):** las ventas "de base" subieron de 1 600 a 6 200 millones por mes; crecían cada vez más rápido hasta 2020 y ahí se frenaron un poco.
4. **Gráfico 3 abajo (efecto del mes):** diciembre +1 182, enero −540, febrero −521, el resto cerca de 0. Solo valen los puntos del día 1.
5. **Línea azul = tendencia + efecto del mes.** Ejemplo: diciembre 2027 = 6 403 + 1 182 = 7 585.
