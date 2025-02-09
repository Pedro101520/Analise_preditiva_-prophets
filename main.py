import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error
from prophet.plot import plot_plotly

df = pd.read_csv("https://raw.githubusercontent.com/alura-cursos/data_science_projeto/main/Dados/bicicletas.csv")

# Tratando valores nulos, neste caso, os campos que contem valores nulos vão ser iguais a media dos valores que eles estievrem entre
df['temperatura'] = df['temperatura'].interpolate(method='linear')
df['sensacao_termica'] = df['sensacao_termica'].interpolate(method='linear')

# Forma de remover valores duplicados do DF, foi criado uma nova variavel para não sobrescrever os dados que já estavam gravados
df_limpo = df.drop_duplicates()

# Gráficos usados para análisar as informações do DF
# sns.displot(df_limpo, x='temperatura', bins=20)
# plt.show()

# Gráfico para analisar a correlação de variaveis do DF
# plt.figure(figsize=(8,4))
# sns.heatmap(df_limpo.corr(numeric_only=True), annot=True, cmap='Blues')
# plt.show()

# Transformando os dados da coluna date para o tipo datetime
df_data = df_limpo.copy()
df_data['data_hora'] = pd.to_datetime(df_data['data_hora'])

df_data['mes'] = df_data['data_hora'].dt.month
df_data['horario'] = df_data['data_hora'].dt.hour

df_data['data_hora'] = df_data['data_hora'].dt.date

df_data = df_data.rename(columns={'data_hora' : 'data'})

# Após o acesso na data, o tipo da variavel voltou a ser objeto, ai abaixo vou transformar novamente em datetime
df_data['data'] = pd.to_datetime(df_data['data'])


# print(df_data.info())
# print(df_data.head(10))

# Para fazer a prvisão de valores, a biblioteca prophet precisa de um DF com duas colunas
df_prophet = df_data[['data', 'contagem']].rename(columns={'data' : 'ds', 'contagem' : 'y'})
df_prophet = df_prophet.groupby('ds')['y'].sum().reset_index()

# np.random.seed(4587)
# modelo = Prophet()

# modelo.fit(df_prophet)
# futuro = modelo.make_future_dataframe(periods=90, freq='D')
# previsao = modelo.predict(futuro)


# fig1 = modelo.plot(previsao)
# plt.show()

# Exibindo os dados previstos
# previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# print(previsao)

# fig2 = modelo.plot_components(previsao)
# plt.show() 

# print(df_prophet.head())




# -------------------------------------------------------------------------------------------------------------------------------------
# Com outliers
# Deixar 80% de dados para treinamento
# df_treino = pd.DataFrame()
# df_treino['ds'] = df_prophet['ds'][:584]
# df_treino['y'] = df_prophet['y'][:584]

# # Deixar 20% para teste
# df_teste = pd.DataFrame()
# df_teste['ds'] = df_prophet['ds'][584:]
# df_teste['y'] = df_prophet['y'][584:]

# np.random.seed(4587)
# modelo = Prophet(yearly_seasonality=True)
# modelo.fit(df_treino)
# futuro = modelo.make_future_dataframe(periods=150, freq='D')
# previsao = modelo.predict(futuro)

# fig1 = modelo.plot(previsao)
# plt.plot(df_teste['ds'], df_teste['y'], '.r')
# plt.show()

# Comparar os valores reais com os previstos
# df_previsao = previsao[['ds', 'yhat']]
# df_comparacao = pd.merge(df_previsao, df_teste, on='ds')
# print(df_comparacao)

# Abaixo, é como calcular o MSE (Que é a quantidade de erros que o modelo esta errando)
# mse = mean_squared_error(df_comparacao['y'], df_comparacao['yhat'])
# rmse = np.sqrt(mse)

# print(f'MSE:{mse}, RMSE:{rmse}')
# -------------------------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------
# Removendo outliers, em casos onde o objetivo é encontrar anomalias, não é interessante remover os outliers
np.random.seed(4587)

modelo_info = Prophet()
modelo_info.fit(df_prophet)
futuro_info = modelo_info.make_future_dataframe(periods=0)
previsao_info = modelo_info.predict(futuro_info)

sem_outliers = df_prophet[(df_prophet['y'] > previsao_info['yhat_lower']) & (df_prophet['y'] < previsao_info['yhat_upper'])]
# print(sem_outliers)

df_treino2 = pd.DataFrame()

df_treino2['ds'] = sem_outliers['ds'][:505]
df_treino2['y'] = sem_outliers['y'][:505]

df_teste2 = pd.DataFrame()

df_teste2['ds'] = sem_outliers['ds'][505:]
df_teste2['y'] = sem_outliers['y'][505:]

# Modelo treinado sem os outliers
np.random.seed(4587)
modelo_sem_outliers = Prophet(yearly_seasonality=True)
modelo_sem_outliers.fit(df_treino2)
futuro = modelo_sem_outliers.make_future_dataframe(periods=150, freq='D')
previsao = modelo_sem_outliers.predict(futuro)

fig1 = modelo_sem_outliers.plot(previsao)
plt.plot(df_teste2['ds'], df_teste2['y'], '.r')
# plt.show()

df_previsao = previsao[['ds', 'yhat']]
df_comparacao = pd.merge(df_previsao, df_teste2, on='ds')
print(df_comparacao)

mse = mean_squared_error(df_comparacao['y'], df_comparacao['yhat'])
rmse = np.sqrt(mse)

print(f'MSE: {mse}, RMSE: {rmse}')
# -------------------------------------------------------------------------------------------------------------

# Criando gráfico interativo
fig = plot_plotly(modelo_sem_outliers, previsao)
fig.show()