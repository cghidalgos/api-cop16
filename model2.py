import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Cargar datos desde la URL
url = 'https://raw.githubusercontent.com/Itsmynool/datasets/main/frijol.csv'
data = pd.read_csv(url)

data = data.drop(columns=['Tipo_Planta'])
# Agregar variable CO2
data['CO2'] = np.random.rand(len(data))

# Separar las características (features) y la variable objetivo (target)
X = data.drop(columns=['Porcentaje_Supervivencia'])
y = data['Porcentaje_Supervivencia']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error Cuadrático Medio (MSE): {mse}')
print(f'Coeficiente de Determinación (R^2): {r2}')

# Guardar el modelo entrenado
with open('modelo_entrenado.pkl', 'wb') as f:
    pickle.dump(model, f)
