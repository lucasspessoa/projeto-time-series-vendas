import pandas as pd
df = pd.read_excel('time-series-data.xlsx')
df.set_index('date', inplace=True)
# print(df.head())

# Verificação gráfica da presença de tendência, sazonalidade e outliers
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(df, color="blue")
plt.tight_layout
# plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
# Plotagem dos resultados
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
decomposicao = seasonal_decompose(df, period=12)
# Componente de tendência
decomposicao.trend.plot(ax=axes[0], title='Componente de Tendência')
# Componente sazonal
decomposicao.seasonal.plot(ax=axes[1], title='Componente Sazonal')
# Componente residual
decomposicao.resid.plot(ax=axes[2], title='Componente Residual')
plt.tight_layout()
# plt.show()

# Decompondo o Ruído da Série
df['Ruído'] = decomposicao.resid
# Visualizando Série e Ruído graficamente
df[['ProductP3', 'Ruído']].plot(title='Série e Vendas')
# plt.show()

# Verificação de estacionariedade
from statsmodels.tsa.stattools import adfuller
resultado = adfuller(df['Ruído'].dropna())
print(f"Estatística do Teste ADF: {resultado[0]}")
print(f"P-valor: {resultado[1]}")
# Teste de Hipótese a 5% de Significância
if resultado[1] <= 0.05:
    print("A série é estacionária")
else:
    print("A série é não estacionária")

# Remoção da Sazonalidade: uso de Diferenciação Sazonal
df['Diferenciada_Sazonal'] = df['Ruído'] - df['Ruído'].shift(12)
resultado_sazonal = adfuller(df['Diferenciada_Sazonal'].dropna())
print(f"P-valor após diferenciação sazonal: {resultado_sazonal[1]}")

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Plotando gráfico ACF
plt.subplot(1, 2, 1)
plot_acf(df['Diferenciada_Sazonal'].dropna(), ax=plt.gca(), lags=36, title="ACF")
# Plotando gráfico PACF
plt.subplot(1, 2, 2)
plot_pacf(df['Diferenciada_Sazonal'].dropna(), ax=plt.gca(), lags=36, title="PACF")
# plt.show()

# Ajustando o modelo e validando
from statsmodels.tsa.statespace.sarimax import SARIMAX
