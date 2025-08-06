import pandas as pd

# Caminho para o arquivo CSV (usar \\ ou / ou string de caminho)
caminhoDados = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\ColetaContinua\\2025\\AL\\104\\104_AL_km89_2025.csv"
df = pd.read_csv(caminhoDados, sep=';', skiprows=2)

# Exibir os tipos de dados das colunas
#print('Tipo da Coluna  "Hora": '+ str(df['Hora'].dtype)+'\n')
#print(df.columns)

colunas_veiculos = [
    '(A) Ônibus/Caminhão de 2 eixos',
    '(B) Ônibus/Caminhão de 3 eixos',
    '(C) Caminhão de 4 eixos',
    '(D) Caminhão de 5 eixos',
    '(E) Caminhão de 6 eixos',
    '(F) Caminhão de 7 eixos',
    '(G) Caminhão de 8 eixos',
    '(H) Caminhão de 9 eixos',
    '(I) Passeio',
    '(J) Motocicleta',
    '(L) Indefinido'
]

# Soma todas essas colunas linha a linha
df['total_veiculos'] = df[colunas_veiculos].sum(axis=1)

# Encontra o índice da linha com o maior total
indice_pico = df['total_veiculos'].idxmax()

# Acha a hora correspondente
hora_pico = df.loc[indice_pico, 'Hora']
total_naquela_hora = df.loc[indice_pico, 'total_veiculos']


#print(df['total_veiculos'])
#print(f'Hora de pico: {hora_pico} (Total de veículos: {total_naquela_hora})')
#print('index do pico: ' + str(indice_pico))

print(df.iloc[indice_pico])
print('\n')
print(df.iloc[indice_pico+1])

# duarouter -n ufalNetwork.net.xml -t ufalTrips.trips.xml --additional ufalTyp.typ.xml -o ufalRoutes.rou.xml
# sumo-gui -c .\ufalConfig.sumocfg   