import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("rides.csv") #base de dados kaggle
print("Colunas do dataset:", df.columns)
df = df[['Drivers Active Per Hour', 'Riders Active Per Hour', 'Rides Completed']]
df = df.dropna(subset=['Rides Completed'])

X = df[['Drivers Active Per Hour', 'Riders Active Per Hour']]
y = df['Rides Completed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nAvaliação do Modelo:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['Rides Completed'], bins=30, kde=True, color='blue', ax=axes[0])
axes[0].set_title("Distribuição de Rotas Completas")
axes[0].set_xlabel("Número de Rotas Completas")
axes[0].set_ylabel("Frequência")

axes[1].scatter(y_test, y_pred, alpha=0.7, color='red')
axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='dashed', color='black')  
axes[1].set_xlabel("Valores Reais")
axes[1].set_ylabel("Valores Previstos")
axes[1].set_title("Comparação: Rotas Completas (Reais vs. Previstos)")
residuos = y_test - y_pred
sns.histplot(residuos, bins=30, kde=True, color='purple', ax=axes[2])

axes[2].set_title("Distribuição dos Erros de Previsão (Resíduos)")
axes[2].set_xlabel("Erro")
axes[2].set_ylabel("Frequência")
plt.tight_layout()
plt.show()