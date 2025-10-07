# -*- coding: utf-8 -*-
"""
Implementação completa de um Deep Q-Network (DQN) para Controle de Semáforos no SUMO.
Baseado no artigo "Real Time Traffic Light Timing Optimisation Using Reinforcement Learning"
(Andrew et al., 2025) - Implementação fiel ao método descrito.
"""

import os
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt

# ONFIGURAÇÃO DO AMBIENTE SUMO ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Por favor, declare a variável de ambiente 'SUMO_HOME'")

import traci

#CONSTANTES E HIPERPARÂMETROS ---

# Configuração da Simulação
SUMO_BINARY = "sumo"  # Use "sumo-gui" para visualização
SUMO_CFG_PATH = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalConfig.sumocfg"

# IDs dos semáforos a serem controlados
JUNCTION_IDS = ['6022602047', '2621508821', '795820931']

# Parâmetros conforme o artigo
EPISODES = 50  # 50 episódios conforme artigo
SIMULATION_STEPS = 1000  # 1000 steps por episódio (não está conforme o artigo, mas sim conforme os dados do DNIT)

# Hiperparâmetros do DQN (baseado no artigo)
MEMORY_SIZE = 2000
BATCH_SIZE = 32
GAMMA = 0.95  # Fator de desconto
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Parâmetros de Controle do Semáforo
DECISION_INTERVAL = 10  # Agente toma decisão a cada 10 segundos
YELLOW_TIME = 4         # Duração da fase amarela
MIN_GREEN_TIME = 10     # Tempo mínimo de verde
MAX_GREEN_TIME = 60     # Tempo máximo de verde

# Dicionário para armazenar informações dos cruzamentos
junction_info = {}

# ==============================================================================
# CLASSE DO AGENTE DEEP Q-NETWORK
# ==============================================================================
class DQNAgent:
    """
    Agente DQN com arquitetura de 3 camadas conforme descrito no artigo.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = self._build_model()

    def _build_model(self):
        """
        Rede neural: 3 camadas densas com ativação ReLU.
        Entrada: vetor de estado (veículos + tempo de espera por faixa)
        Saída: Q-valores para cada ação possível
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Armazena transição (s, a, r, s', done) na memória de replay.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Seleciona ação usando estratégia epsilon-greedy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """
        Treina a rede com minibatch da memória.
        Usa Q-learning: Q(s,a) = r + γ * max Q(s',a')
        Equação de Bellman
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([t[0][0] for t in minibatch])
        next_states = np.array([t[3][0] for t in minibatch])
        
        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                target = reward + GAMMA * np.amax(q_next[i])
            else:
                target = reward
            q_current[i][action] = target

        self.model.fit(states, q_current, epochs=1, verbose=0)

    def adapt_epsilon(self):
        """
        Decaimento epsilon para reduzir exploração ao longo do tempo.
        """
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

# ==============================================================================
# FUNÇÕES DE INTERAÇÃO COM O SUMO
# ==============================================================================

def initialize_junction_info():
    """
    Inicializa informações dos cruzamentos: faixas, fases verdes, etc.
    """
    print("Inicializando informações dos cruzamentos...")
    for j_id in JUNCTION_IDS:
        incoming_lanes = sorted(list(set(traci.trafficlight.getControlledLanes(j_id))))
        
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(j_id)[0]
        green_phases_indices = [i for i, phase in enumerate(logic.phases) 
                               if 'g' in phase.state.lower() and 'y' not in phase.state.lower()]
        
        # Mapeia cada fase verde para sua fase amarela correspondente
        yellow_phase_map = {}
        for i, green_idx in enumerate(green_phases_indices):
            # Assume que amarelo vem logo após o verde
            yellow_idx = green_idx + 1 if green_idx + 1 < len(logic.phases) else green_idx
            yellow_phase_map[green_idx] = yellow_idx

        # Espaço de ação: manter fase atual + mudar para cada fase verde disponível
        action_size = len(green_phases_indices) + 1  # +1 para "manter"
        
        # Espaço de estado: 2 features por faixa (veículos + tempo de espera)
        state_size = len(incoming_lanes) * 2

        junction_info[j_id] = {
            'incoming_lanes': incoming_lanes,
            'state_size': state_size,
            'action_size': action_size,
            'green_phases': green_phases_indices,
            'yellow_phase_map': yellow_phase_map,
            'current_green_phase_idx': green_phases_indices[0],
            'current_green_duration': 0,
            'last_waiting_time': 0
        }
        
        print(f"  - Cruzamento '{j_id}':")
        print(f"    -> {len(incoming_lanes)} faixas de entrada")
        print(f"    -> {len(green_phases_indices)} fases verdes: {green_phases_indices}")
        print(f"    -> Estado: {state_size} features | Ações: {action_size}")


def get_state(junction_id):
    """
    Retorna o estado do cruzamento:
    - Número de veículos em cada faixa
    - Tempo de espera acumulado em cada faixa
    """
    lanes = junction_info[junction_id]['incoming_lanes']
    state = []
    
    for lane in lanes:
        # Feature 1: Contagem de veículos
        vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
        state.append(vehicle_count)
        
        # Feature 2: Tempo de espera acumulado
        waiting_time = traci.lane.getWaitingTime(lane)
        state.append(waiting_time)
    
    return np.array(state).reshape(1, junction_info[junction_id]['state_size'])


def get_reward(junction_id):
    """
    Recompensa = negativo do tempo de espera total.
    Calcula a diferença para evitar acumulação.
    """
    lanes = junction_info[junction_id]['incoming_lanes']
    current_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
    
    # Recompensa baseada na mudança no tempo de espera
    reward = -(current_waiting_time - junction_info[junction_id]['last_waiting_time'])
    junction_info[junction_id]['last_waiting_time'] = current_waiting_time
    
    return reward


def apply_action(junction_id, action):
    """
    Aplica a ação escolhida pelo agente:
    - Ação 0: Manter fase atual (se possível)
    - Ação 1-N: Mudar para fase verde específica
    """
    info = junction_info[junction_id]
    green_phases = info['green_phases']
    current_phase = info['current_green_phase_idx']
    
    if action == 0:  # Manter fase atual
        # Estende a fase atual se não exceder o tempo máximo
        if info['current_green_duration'] < MAX_GREEN_TIME:
            traci.trafficlight.setPhaseDuration(junction_id, DECISION_INTERVAL)
            info['current_green_duration'] += DECISION_INTERVAL
            return 0  # Retorna 0 steps adicionais (já contados no intervalo)
        else:
            # Se excedeu tempo máximo, força mudança para próxima fase
            action = 1
    
    if action > 0 and len(green_phases) > 1:  # Mudar para nova fase
        # Determina a fase destino
        target_phase_idx = (action - 1) % len(green_phases)
        target_phase = green_phases[target_phase_idx]
        
        if target_phase != current_phase:
            # Transição: verde -> amarelo -> nova fase verde
            yellow_phase = info['yellow_phase_map'].get(current_phase, current_phase)
            
            # Fase amarela
            traci.trafficlight.setPhase(junction_id, yellow_phase)
            for _ in range(YELLOW_TIME):
                traci.simulationStep()
            
            # Nova fase verde
            traci.trafficlight.setPhase(junction_id, target_phase)
            traci.trafficlight.setPhaseDuration(junction_id, MIN_GREEN_TIME)
            
            info['current_green_phase_idx'] = target_phase
            info['current_green_duration'] = 0
            
            return YELLOW_TIME  # Retorna steps gastos na transição
    
    return 0


def run_fixed_time_baseline():
    """
    Executa simulação com controle fixed-time (baseline para comparação).
    """
    print("\n=== EXECUTANDO BASELINE FIXED-TIME ===")
    traci.start([SUMO_BINARY, "-c", SUMO_CFG_PATH, "--no-warnings", "true"])
    
    total_waiting_time = 0
    
    for step in range(SIMULATION_STEPS):
        traci.simulationStep()
        
        # Acumula tempo de espera de todos os cruzamentos
        for j_id in JUNCTION_IDS:
            lanes = junction_info[j_id]['incoming_lanes']
            total_waiting_time += sum(traci.lane.getWaitingTime(lane) for lane in lanes)
    
    traci.close()
    
    avg_waiting_time = total_waiting_time / SIMULATION_STEPS
    print(f"Tempo de espera médio (Fixed-Time): {avg_waiting_time:.2f} s")
    
    return avg_waiting_time


# ==============================================================================
# LOOP PRINCIPAL DE TREINAMENTO
# ==============================================================================
if __name__ == "__main__":
    agents = {}
    episode_rewards = []
    episode_waiting_times = []

    # Fase 1: Treinamento com Reinforcement Learning
    print("=== INICIANDO TREINAMENTO RL ===\n")
    
    for episode in range(EPISODES):
        traci.start([SUMO_BINARY, "-c", SUMO_CFG_PATH, "--no-warnings", "true"])
        
        # Inicializa na primeira execução
        if episode == 0:
            initialize_junction_info()
            for j_id in JUNCTION_IDS:
                agents[j_id] = DQNAgent(
                    junction_info[j_id]['state_size'],
                    junction_info[j_id]['action_size']
                )
        
        # Reset estado inicial
        for j_id in JUNCTION_IDS:
            junction_info[j_id]['last_waiting_time'] = 0
            junction_info[j_id]['current_green_duration'] = 0
        
        current_states = {j_id: get_state(j_id) for j_id in JUNCTION_IDS}
        
        step = 0
        episode_total_reward = 0
        episode_total_waiting = 0
        
        while step < SIMULATION_STEPS:
            # Toma decisão em intervalos definidos
            if step % DECISION_INTERVAL == 0:
                
                for j_id in JUNCTION_IDS:
                    # Escolhe ação
                    action = agents[j_id].act(current_states[j_id])
                    
                    # Aplica ação
                    extra_steps = apply_action(j_id, action)
                    
                    # Avança simulação pelo intervalo (descontando transições)
                    steps_to_run = DECISION_INTERVAL - extra_steps
                    for _ in range(max(0, steps_to_run)):
                        if step < SIMULATION_STEPS:
                            traci.simulationStep()
                            step += 1
                    
                    # Obtém recompensa e próximo estado
                    reward = get_reward(j_id)
                    next_state = get_state(j_id)
                    done = step >= SIMULATION_STEPS
                    
                    # Armazena experiência
                    agents[j_id].remember(current_states[j_id], action, reward, next_state, done)
                    
                    # Treina
                    agents[j_id].replay(BATCH_SIZE)
                    
                    # Atualiza estado
                    current_states[j_id] = next_state
                    episode_total_reward += reward
                    
            else:
                traci.simulationStep()
                step += 1
            
            # Acumula tempo de espera
            for j_id in JUNCTION_IDS:
                lanes = junction_info[j_id]['incoming_lanes']
                episode_total_waiting += sum(traci.lane.getWaitingTime(lane) for lane in lanes)
        
        traci.close()
        
        # Decaimento epsilon
        for agent in agents.values():
            agent.adapt_epsilon()
        
        avg_waiting = episode_total_waiting / SIMULATION_STEPS
        episode_rewards.append(episode_total_reward)
        episode_waiting_times.append(avg_waiting)
        
        print(f"Episódio {episode+1}/{EPISODES} | "
              f"Reward: {episode_total_reward:.2f} | "
              f"Tempo Espera Médio: {avg_waiting:.2f} s | "
              f"Epsilon: {agents[JUNCTION_IDS[0]].epsilon:.4f}")
    
    print("\n=== TREINAMENTO CONCLUÍDO ===\n")
    
    # Salva modelos treinados
    for j_id in JUNCTION_IDS:
        agents[j_id].save(f"model_{j_id}.weights.h5")
    
    # Fase 2: Comparação com baseline fixed-time
    baseline_waiting = run_fixed_time_baseline()
    rl_final_waiting = episode_waiting_times[-1]
    
    improvement = ((baseline_waiting - rl_final_waiting) / baseline_waiting) * 100
    
    print(f"\n=== RESULTADOS FINAIS ===")
    print(f"Tempo de espera (Fixed-Time): {baseline_waiting:.2f} s")
    print(f"Tempo de espera (RL): {rl_final_waiting:.2f} s")
    print(f"Melhoria: {improvement:.2f}%")
    
    # Plotagem dos resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico 1: Tempo de espera por episódio (Figura 1 do artigo)
    ax1.plot(range(1, EPISODES+1), episode_waiting_times, 'b-', linewidth=2)
    ax1.axhline(y=baseline_waiting, color='r', linestyle='--', 
                label='Baseline Fixed-Time', linewidth=2)
    ax1.set_xlabel('Episódio', fontsize=12)
    ax1.set_ylabel('Tempo de Espera Médio (s)', fontsize=12)
    ax1.set_title('Desempenho do Agente RL vs Fixed-Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráfico 2: Comparação final (Figura 2 do artigo)
    systems = ['Fixed-Time', 'RL Model']
    waiting_times = [baseline_waiting, rl_final_waiting]
    colors = ['#ff6b6b', '#4ecdc4']
    
    bars = ax2.bar(systems, waiting_times, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Tempo de Espera Médio (s)', fontsize=12)
    ax2.set_title(f'Comparação de Desempenho\n(Melhoria: {improvement:.1f}%)', fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Adiciona valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("resultados_completos.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nGráficos salvos em 'resultados_completos.png'")