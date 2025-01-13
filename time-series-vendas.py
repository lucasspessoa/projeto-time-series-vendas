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
plt.show()