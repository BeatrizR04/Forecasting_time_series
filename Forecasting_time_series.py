#!/usr/bin/env python
# coding: utf-8

# In[233]:


# Importamos librerias
import pandas as pd
import numpy as np
from datetime import datetime

# from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split

# import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import statsmodels.api as sm
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")


# ### Proceso
# - Disponemos de unos datasets donde vemos las ventas de aceite en Ecuador
# - El usuario elige un cluster (tiendas similares) para hacer los análisis
# - Análisis principal: Se van a predecir las ventas de aceite a nivel global. Es decir, ventas totales en un día en todas las tiendas
#     - Para ello necesitamos agrupar la serie por fecha y añadirle variables que nos vayan a influir en el cálculo
#     - Se realizan diversos cálculos para ver cómo es la serie y analizarla
# - Funciones de modelado: Se van a crear funciones que cargen los modelos, predigan el resultado y visualicen gráficas de pronósticos
#     - Necesitamos funciones para cada modelo
#     - Se 'repite' algún modelo con parámetros distintos
#     - Se ejecuta el resultado
# - Tabla de resultados: Se va a generar una tabla con métricas que nos indicará los resultados de las predicciones hechas

# ### Lectura de los datos

# In[3]:


# Leemos los archivos
df_oil = pd.read_csv('oil.csv')
df_holidays = pd.read_csv('holidays_events.csv')
df_stores = pd.read_csv('stores.csv')
df_transactions = pd.read_csv('transactions.csv')
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Consolidamos los datos para tener toda la información en una solo df
df = pd.merge(df_train, df_oil, how = 'left', on='date')
df = pd.merge(df, df_holidays, how = 'left',on = 'date')
df = pd.merge(df, df_transactions, how ='left', on =['date','store_nbr'])
df = pd.merge(df, df_stores, how = 'left', on = 'store_nbr')
df.rename(columns={'type_x':'holiday_type', 'type_y':'store_type'}, inplace = True)


# In[4]:


df.head()


# In[5]:


# Ponemos fecha en formato datetime y la ponemos en el índice
df.date = [datetime.strptime(x, '%Y-%m-%d') for x in df.date]


# In[6]:


# Analizamos los valores nulos
df.isnull().sum()


# In[7]:


# Nos deshacemos de los valores nulos porque esas variables no nos interesan
df['dcoilwtico'] = df['dcoilwtico'].fillna(method='bfill')
df[['holiday_type']] = df[['holiday_type']].fillna(0)
df['locale'] = df['locale'].fillna(0)
df['locale_name'] = df['locale_name'].fillna(0)
df['description'] = df['description'].fillna(0)
df['transferred'] = df['transferred'].fillna(0)
df['transactions'] = df['transactions'].fillna(0)


# In[8]:


# Comprobamos que ya no tenemos nulos
df.isnull().sum()


# In[9]:


df.shape


# ### Elección del cluster a analizar

# In[10]:


select_cluster = input('¿Qué cluster quieres analizar?\n Las opciones son: 1,2,..,17\n El cluster a analizar es: ')


# In[11]:


df = df[df.cluster == int(select_cluster)]


# ## Análisis principal
# > **VENTA DE ACEITE POR DÍA**

# In[236]:


# Agrupamos por fecha y total de ventas
df1 = df.groupby('date', as_index = False).sales.sum()


# In[237]:


# Visualizamos la serie a través del tiempo
df1.set_index('date').sales.plot(figsize=(15,5), title=f'VENTAS TOTALES DE ACEITE EN EL TIEMPO')


# In[238]:


# Podemos visualizar la serie por años
fig, ax = plt.subplots(5,1, figsize=(12,8))
ax[0].plot(df1.set_index('date').sales[df1.set_index('date').index < '2014-01-01'], label='2013', color='purple')
ax[0].legend(loc='upper left')
ax[1].plot(df1.set_index('date').sales[(df1.set_index('date').index >= '2014-01-01') &                                      (df1.set_index('date').index < '2015-01-01')], label='2014', color='blue')
ax[1].legend(loc='upper left')
ax[2].plot(df1.set_index('date').sales[(df1.set_index('date').index >= '2015-01-01') &                                      (df1.set_index('date').index < '2016-01-01')], label='2015', color='yellow')
ax[2].legend(loc='upper left')
ax[3].plot(df1.set_index('date').sales[(df1.set_index('date').index >= '2016-01-01') &                                      (df1.set_index('date').index < '2017-01-01')], label='2016', color='black')
ax[3].legend(loc='upper left')
ax[4].plot(df1.set_index('date').sales[df1.set_index('date').index >= '2017-01-01'], label='2017', color='green')
ax[4].legend(loc='upper left')
plt.tight_layout()
plt.show()


# #### Descomposición de la serie

# In[239]:


# Tendencia
trend = sm.tsa.seasonal_decompose(df1.set_index('date')['sales'], model='additive', period=365).trend

# Estacionalidad
seasonal = sm.tsa.seasonal_decompose(df1.set_index('date')['sales'], model='additive', period=365).seasonal

# Residuos
resid = sm.tsa.seasonal_decompose(df1.set_index('date')['sales'], model='additive', period=365).resid

# Graficamos
fig, ax = plt.subplots(4,1, figsize=(12,10))
ax[0].plot(df1.set_index('date')['sales'], label='Ventas en el tiempo')
ax[0].legend(loc='upper left')
ax[1].plot(trend, label='Tendencia')
ax[1].legend(loc='upper left')
ax[2].plot(seasonal, label='Estacionalidad')
ax[2].legend(loc='upper left')
ax[3].plot(resid, label='Residuo')
ax[3].legend(loc='upper left')
plt.tight_layout()
plt.show()

# Se podría aplicar también la función seasonal_decompose y se obtiene todo a la vez. Es lo mismo


# #### Autocorrelación y estacionareidad

# In[240]:


# ACF y PACF (se pueden utilizar para sleccionar los parámetros de los modelos ARIMA)
plot_acf(df1.sales)
plot_pacf(df1.sales)


# In[241]:


# Test de Dickey Fuller (Para comprobar la estacionariedad)
from statsmodels.tsa.stattools import adfuller
adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(df1.sales)
print('ADF test statistic:', adf)
print('ADF p-values:', pval)
print('ADF number of lags used:', usedlag)
print('ADF number of observations:', nobs)
print('ADF critical values:', crit_vals)
print('ADF best information criterion:', icbest)

# Si el p-valor es menor que 0.05, es estacionaria


# In[242]:


# Tenemos que diferenciarla para que sea estacionaria
df1['diff'] = df1.sales.diff()
df1 = df1.dropna()
df1.head()


# In[243]:


# Volvemos a aplicar la prueba de Dickey-Fuller
adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(df1.sales)
print('ADF test statistic:', adf)
print('ADF p-values:', pval)
print('ADF number of lags used:', usedlag)
print('ADF number of observations:', nobs)
print('ADF critical values:', crit_vals)
print('ADF best information criterion:', icbest)


# In[244]:


# Creamos columnas de mes y día de la semana
df1['month'] = df1['date'].apply(lambda x: x.strftime('%B'))
df1['week_day'] = df1['date'].dt.day_name()

df1


# In[245]:


# Visualizamos la distribucción de las ventas por mes y día de la semana
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x=df1.month, y=df1.sales, data=df1, ax=axes[0])
sns.boxplot(x=df1.week_day, y=df1.sales, data=df1, ax=axes[1])


axes[0].set_title('Distribucción por mes', fontsize=18); 
axes[1].set_title('Distribucción por día de la semana', fontsize=18)
plt.show()


# Podemos comprobar con esta gráfica más claramente que tenemos outliers a mediados y finales de año (no sé qué conclusión poner para los días de la semana) (la cajita es Q1-Q3 y la raya la mediana. Los bigotes son los datos max/min si considerar los atópicos)

# In[246]:


# Estos outliers van a 'molestar' a la hora de modelar, por lo que vamos a cambiar sus valores por las medias de esos meses

# Media de cada mes
monthly_means = df1.groupby('month')['sales'].mean()

# Obtener el bigote superior para cada mes
boxplot_stats = df1.groupby('month')['sales'].describe()['75%']
upper_whisker = boxplot_stats + 1.5 * (boxplot_stats - df1.groupby('month')['sales'].describe()['25%'])

# Crear una máscara que seleccione los valores atípicos
mask = df1.sales > upper_whisker[df1.month.values].values

# Reemplazar los valores atípicos con la media del mes
df1.sales[mask] = df1.groupby('month')['sales'].transform('mean')

# Hemos buscado el valor del bigote superior para coger todos los outliers 
# Límite inferior del bigote = Q1 - 1.5 * RI
# Límite superior del bigote = Q3 + 1.5 * RI


# In[247]:


# Visualizamos la distribucción de las ventas por mes y día de la semana después de quitar los valores atípicos
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x=df1.month, y=df1.sales, data=df1, ax=axes[0])
sns.boxplot(x=df1.week_day, y=df1.sales, data=df1, ax=axes[1])


axes[0].set_title('Distribucción por mes', fontsize=18); 
axes[1].set_title('Distribucción por día de la semana', fontsize=18)
plt.show()


# In[248]:


# Vamos a añadir como variables ventanas temporales (medias móviles) para usarlas a la hora de predecir
df1 = df1.reset_index(drop=True)
df1['Wind_1day'] = df1.sales.rolling(2).mean().reset_index(0,drop=True)
df1['Wind_7day'] = df1.sales.rolling(8).mean().reset_index(0,drop=True)
df1['Wind_14day'] = df1.sales.rolling(15).mean().reset_index(0,drop=True)
df1['Wind_21day'] = df1.sales.rolling(22).mean().reset_index(0,drop=True)
df1['Wind_month'] = df1.sales.rolling(31).mean().reset_index(0,drop=True)
# Las siguientes son medias móviles del mismo día que estás mirando pero de las semanas anteriores
# Última semana
df1['last_week_avg'] = df1.groupby('week_day')['sales'].rolling(window=7).mean().reset_index(0, drop=True)
# Últimas dos semanas
df1['two_week_avg'] = df1.groupby('week_day')['sales'].rolling(window=14).mean().reset_index(0, drop=True)
# Últimas tres semanas
df1['three_week_avg'] = df1.groupby('week_day')['sales'].rolling(window=21).mean().reset_index(0, drop=True)


# In[249]:


# Eliminamos las primeras filas para quitarnos los nulos y no tener problemas con los modelos. 
# Al ser pocos datos no van a afectarnos mucho
df1.dropna(inplace=True)
df1.head()


# In[250]:


# Antes de entrenar los modelos, vamos a hacer dummies en las columnas categóricas ya que la mayoría de los modelos
# solo admiten numéricas
df1_dummies = pd.concat([df1, pd.get_dummies(df1[['month','week_day']])], axis=1)
df1_dummies.drop(columns=['month','week_day'], inplace = True)
df1_dummies.head()


# In[251]:


df1_dummies.columns


# In[252]:


# Vamos a separar los datos de entrenamiento y test de tal manera que lo que vamos a predecir sea el último año y creamos
# una columna para facilitarnos la separación
df1_dummies['set'] = ['train' if x <= datetime(2017,1,1) else 'test' for x in df1_dummies.date]

df1_dummies.set_index('date', drop=True, inplace=True)
df1_dummies.index.name = None

X_train1_dummies = df1_dummies[df1_dummies.set == 'train'].drop(columns=['sales','set'])
X_test1_dummies =df1_dummies[df1_dummies.set == 'test'].drop(columns=['sales','set'])
y_train1_dummies = df1_dummies[df1_dummies.set == 'train'].sales
y_test1_dummies = df1_dummies[df1_dummies.set == 'test'].sales


# In[260]:


# Vamos a realizar la misma separación pero para los datos sin dummies para los modelos que puedan usarlos
df1['set'] = ['train' if x <= datetime(2017,1,1) else 'test' for x in df1.date]

df1.set_index('date', drop=True, inplace=True)
df1.index.name = None


df1['month'] = df1['month'].astype('category')
df1['week_day'] = df1['week_day'].astype('category')

X_train1 = df1[df1.set == 'train'].drop(columns=['sales','set','diff'])
X_test1 =df1[df1.set == 'test'].drop(columns=['sales','set','diff'])
y_train1 = df1[df1.set == 'train'].sales
y_test1 = df1[df1.set == 'test'].sales


# ### Material utilizado en los modelos

# In[254]:


# Vamos a generar un df con los resultados de los distintos modelos para compararlos
# Lo generamos antes de elegir el cluster por si elegimos distintos en la misma ejecucción que se guarden todos
results = pd.DataFrame(columns = ['Model', 'Cluster', 'MAE', 'MAPE', 'MSE', 'RMSE', 'R2 Score'])


# ## MODELAJE

# ### Modelos

# In[272]:


def linear_regression(X_train, y_train, X_test, y_test):
    # Entrenamos el modelo
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
            
    # Graficar los datos de prueba y el pronóstico
    plt.figure(figsize=(15, 5))
    plt.plot(X_test.index, y_test, label='Datos de prueba')
    plt.plot(X_test.index, y_pred_lr, label='Pronóstico Regresión lineal')
    plt.title('Pronóstico Regresión lineal')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.show()
    
    return y_pred_lr

def linear_regression_normalize(X_train, y_train, X_test, y_test):
    # Entrenamos el modelo y normalizamos las variables de entrada
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
            
    # Graficar los datos de prueba y el pronóstico
    plt.figure(figsize=(15, 5))
    plt.plot(X_test.index, y_test, label='Datos de prueba')
    plt.plot(X_test.index, y_pred_lr, label='Pronóstico Regresión lineal')
    plt.title('Pronóstico Regresión lineal con variables de entrada normalizadas')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.show()
    
    return y_pred_lr

def random_forest(X_train, y_train, X_test, y_test):
    # Entrenamos el modelo
    rf = RandomForestRegressor(random_state=0)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
            
    # Graficar los datos de prueba y el pronóstico
    plt.figure(figsize=(15, 5))
    plt.plot(X_test.index, y_test, label='Datos de prueba')
    plt.plot(X_test.index, y_pred_rf, label='Pronóstico Random Forest')
    plt.title('Pronóstico Random Forest')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.show()
    
    return y_pred_rf

def random_forest_ntrees(X_train, y_train, X_test, y_test):
    # Entrenamos el modelo y seleccionamos el número de árboles en el bosque y la profundidad max de cada árbol
    # Cuántos más árboles más se ajustaría el modelo
    rf = RandomForestRegressor(random_state=0, n_estimators = 20, max_depth = 5)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
        
    # Graficar los datos de prueba y el pronóstico
    plt.figure(figsize=(15, 5))
    plt.plot(X_test.index, y_test, label='Datos de prueba')
    plt.plot(X_test.index, y_pred_rf, label='Pronóstico Random Forest')
    plt.title('Pronóstico Random Forest con elección del número de árboles y su profundidad')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.show()
    
    return y_pred_rf

def xgboost(X_train, y_train, X_test, y_test):
    # Entrenamos el modelo
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
        
    # Graficar los datos de prueba y el pronóstico
    plt.figure(figsize=(15, 5))
    plt.plot(X_test.index, y_test, label='Datos de prueba')
    plt.plot(X_test.index, y_pred_xgb, label='Pronóstico XGB')
    plt.title('Pronóstico XGBoost')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.show()
    
    return y_pred_xgb


def xgboost_cat(X_train, y_train, X_test, y_test):
    # Codificar variables categóricas usando one-hot encoding
    encoder = OneHotEncoder()
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    
    # Entrenar el modelo
    xgb = XGBRegressor()
    xgb.fit(X_train_encoded, y_train)
    
    # Realizar predicciones
    y_pred_xgb = xgb.predict(X_test_encoded)
    
    # Graficar los datos de prueba y el pronóstico
    plt.figure(figsize=(15, 5))
    plt.plot(X_test.index, y_test, label='Datos de prueba')
    plt.plot(X_test.index, y_pred_xgb, label='Pronóstico XGB')
    plt.title('Pronóstico XGBoost con variables categóricas')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.show()
    
    return y_pred_xgb


def sarima(X_train, y_train, X_test, y_test, order, seasonal_order):
    # Entrenamos y ajustamos el modelo
    # order son las variables (p,d,q) y seasonal_order las variables (P,D,Q,S) del modelo SARIMAX
    sarima = SARIMAX(endog=y_train, order=order, seasonal_order=seasonal_order)
    sarima_fit = sarima.fit()
    y_pred_sarima = sarima_fit.forecast(steps=len(X_test))
    
    # Graficar los datos de prueba y el pronóstico
    plt.figure(figsize=(15, 5))
    plt.plot(X_test.index, y_test, label='Datos de prueba')
    plt.plot(X_test.index, y_pred_sarima, label='Pronóstico SARIMA')
    plt.title('Pronóstico SARIMA')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.show()

    return y_pred_sarima


# In[273]:


dict_models = {'Regresión lineal': linear_regression, 'Regresión lineal normalizada': linear_regression_normalize, 
               'Random Forest': random_forest, 'Random Forest con nº de árboles' : random_forest_ntrees,
               'XGBoost': xgboost, 'XGBoost categorico' : xgboost_cat, 'SARIMA' : sarima}


# ### Predicción

# Para selecionar directamente el modelo y que se ejecute con ese, ejecutamos la siguiente celda

# In[257]:


# select_model = input('¿Qué modelo deseas probar? \n Opciones: \n a: Regresión lineal \n b: Regresión lineal normalizada \n')


# In[274]:


# if select_model == 'a':
#     model_name = 'Regresión lineal'
# if select_model == 'b':
#     model_name = 'Regresión lineal normalizada'
   

def forecast(df, model_name, X_train, y_train, X_test, y_test, results, order=None, seasonal_order=None):
    model = dict_models[model_name]
    if model_name == 'SARIMA':
        y_pred = model(X_train, y_train, X_test, y_test, order, seasonal_order)
    else:
        y_pred = model(X_train, y_train, X_test, y_test)
    
    #Calculamos algunas de sus métricas
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    
    # Creamos diccionario con métricas del modelo
    model_metrics = {
        'Model': model_name,
        'Cluster' : select_cluster,
        'MAE': mae,
        'MAPE': mape,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2,
        'Order' : order,
        'Seasonal order' : seasonal_order
    }
    
    # Agregamos una fila a la tabla de resultados
    results = results.append(model_metrics, ignore_index = True)
    
    return results
    
# Y si queremos cambiar parámetros en los modelos? Otra función distinta?


# In[277]:


# En esta linea probamos los distintos modelos y se van guardando
# Importante tener en cuanta que si elegimos el XGB con categóricas hay que meter los datos sin dummies
results = forecast(df1, 'XGBoost', X_train1_dummies, y_train1_dummies, X_test1_dummies, y_test1_dummies,                   results, order = None, seasonal_order = None)
results


# In[ ]:





# In[ ]:




