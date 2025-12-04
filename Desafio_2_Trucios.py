import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import calendar
from matplotlib.colors import LinearSegmentedColormap

dados = pd.read_csv("C:\\Users\\Felip\\Downloads\\flights.csv")

# Ver todas as colunas disponíveis
print("Colunas disponíveis:")
print(dados.columns.tolist())

# Ver primeiras linhas para visualizar
print("\nPrimeiras 5 linhas:")
print(dados.head())

#1) Quais são as estatísticas suficientes para a determinação do percentual de vôos atrasados na chegada (ARRIVAL_DELAY > 10)?
Arrival_Delay = dados["ARRIVAL_DELAY"]
atrasos = [c for c in Arrival_Delay if c > 10]
print(f"Total de voos atrasados: {len(atrasos)}")
print(f"Total de voos: {len(Arrival_Delay)}")
Percentual_Voos_Atrasados = len(atrasos)/len(Arrival_Delay)
print(f"Percentual de voos atrasados: {Percentual_Voos_Atrasados:.4f}")

#2) Crie uma função chamada getStats
def getStats(dados):
    # a - Filtrar apenas as companhias AA, DL, UA e US
    companhias_especificas = dados[dados["AIRLINE"].isin(["AA", "DL", "UA", "US"])]
    
    # b - Remover observações com valores faltantes nos campos de interesse
    campos_interesse = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 
                       'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']
    
    companhias_especificas = companhias_especificas.dropna(subset=campos_interesse)
    
    # c - Agrupar os dados por dia, mês e companhia aérea
    companhias_agrupadas = companhias_especificas.groupby(['DAY', 'MONTH', 'AIRLINE'])
    
    # d - Calcular estatísticas para cada grupo
    stats = companhias_agrupadas.agg(
        total_voos=('ARRIVAL_DELAY', 'count'),
        voos_atrasados=('ARRIVAL_DELAY', lambda x: (x > 10).sum()),
        atraso_medio=('ARRIVAL_DELAY', 'mean'),
        atraso_mediano=('ARRIVAL_DELAY', 'median'),
        desvio_padrao_atraso=('ARRIVAL_DELAY', 'std')
    ).reset_index()
    
    # Calcular proporção de voos atrasados
    stats['proporcao_atrasados'] = stats['voos_atrasados'] / stats['total_voos']
    
    return stats

#3) Processamento em chunks
def process_chunk(chunk):
    return getStats(chunk)

colunas_interesse = {
    'YEAR': 'int64',
    'MONTH': 'int64', 
    'DAY': 'int64',
    'DAY_OF_WEEK': 'int64',
    'AIRLINE': 'str',
    'FLIGHT_NUMBER': 'int64',
    'TAIL_NUMBER': 'str',
    'ARRIVAL_TIME': 'str',
    'ARRIVAL_DELAY': 'float64'
}

use_cols = list(colunas_interesse.keys())
chunk_size = 100000
results = []

# Processar dados já carregados (sem chunking pois já está em memória)
print("Processando dados...")
chunk_result = process_chunk(dados)
results.append(chunk_result)

if results:
    final_result = pd.concat(results, ignore_index=True)
    
    final_stats = final_result.groupby(['DAY', 'MONTH', 'YEAR', 'AIRLINE']).agg({
        'total_voos': 'sum',
        'voos_atrasados': 'sum',
        'atraso_medio': 'mean',
        'atraso_mediano': 'median',
        'desvio_padrao_atraso': 'mean',
        'proporcao_atrasados': 'mean'
    }).reset_index()

#4) Função computeStats
def computeStats(stats_df):
    stats_df['Data'] = pd.to_datetime(
        stats_df['YEAR'].astype(str) + '-' + 
        stats_df['MONTH'].astype(str) + '-' + 
        stats_df['DAY'].astype(str),
        format='%Y-%m-%d',
        errors='coerce'
    )

    stats_df['Perc'] = stats_df['voos_atrasados'] / stats_df['total_voos']
    stats_df['Perc'] = stats_df['Perc'].clip(0, 1)

    result_df = stats_df[['AIRLINE', 'Data', 'Perc']].copy()
    result_df.columns = ['Cia', 'Data', 'Perc']
    
    result_df = result_df.dropna(subset=['Data'])
    result_df = result_df.sort_values(['Data', 'Cia']).reset_index(drop=True)
    
    return result_df

# Aplicar computeStats aos dados processados
resultado_final = computeStats(final_stats)
print("\nResultado final da computeStats:")
print(resultado_final.head())
print(f"\nTotal de registros: {len(resultado_final)}")

#5) Mapas de calor em formato de calendário

# b. Definir paleta de cores
def create_palette():
    """Cria a paleta de cores gradiente do azul (#4575b4) ao vermelho (#d73027)"""
    colors = ["#4575b4", "#d73027"]
    return LinearSegmentedColormap.from_list("custom_palette", colors)

pal = create_palette()

# c. Função baseCalendario
def baseCalendario(stats, cia):
    """
    Cria base de calendário para uma companhia aérea específica
    """
    # i. Filtrar dados para a companhia específica
    cia_data = stats[stats['Cia'] == cia].copy()
    
    if cia_data.empty:
        print(f"Nenhum dado encontrado para a companhia {cia}")
        return None
    
    # Adicionar colunas para ano, mês e dia
    cia_data['Ano'] = cia_data['Data'].dt.year
    cia_data['Mes'] = cia_data['Data'].dt.month
    cia_data['Dia'] = cia_data['Data'].dt.day
    cia_data['DiaSemana'] = cia_data['Data'].dt.weekday
    
    return cia_data

# d. Executar baseCalendario para cada companhia
print("\nCriando bases de calendário...")
cAA = baseCalendario(resultado_final, 'AA')
cDL = baseCalendario(resultado_final, 'DL') 
cUA = baseCalendario(resultado_final, 'UA')
cUS = baseCalendario(resultado_final, 'US')

# e. Função para plotar mapa de calor de calendário
def plot_calendar_heatmap(calendar_data, cia, pal):
    """
    Plota mapa de calor em formato de calendário para uma companhia aérea
    """
    if calendar_data is None or calendar_data.empty:
        print(f"Dados insuficientes para {cia}")
        return
    
    # Criar pivot table para o heatmap
    pivot_data = calendar_data.pivot_table(
        values='Perc',
        index='Dia',
        columns='Mes',
        aggfunc='mean'
    )
    
    # Plotar o heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(pivot_data.T, cmap=pal, aspect='auto', vmin=0, vmax=1)
    
    # Configurações do gráfico
    plt.colorbar(im, label='Percentual de Atraso')
    plt.title(f'Mapa de Calor - Companhia {cia}\nPercentual de Voos Atrasados por Dia e Mês', fontsize=14)
    plt.xlabel('Dia do Mês')
    plt.ylabel('Mês')
    
    # Configurar eixos
    plt.xticks(range(0, 31, 5), range(1, 32, 5))
    plt.yticks(range(12), [calendar.month_abbr[i+1] for i in range(12)])
    
    # Adicionar valores nos cells
    for i in range(pivot_data.shape[1]):  # Meses
        for j in range(pivot_data.shape[0]):  # Dias
            if not np.isnan(pivot_data.iloc[j, i]):
                plt.text(j, i, f'{pivot_data.iloc[j, i]:.2f}', 
                        ha='center', va='center', fontsize=8,
                        color='white' if pivot_data.iloc[j, i] > 0.5 else 'black')
    
    plt.tight_layout()
    plt.show()
