"""
Multi-agent DQN + Global Reward + Full Traffic Metrics
SUMO + TraCI evaluation-ready script - VERSÃO CORRIGIDA

    Coleta métricas completas da simulação com agentes DQN apenas na ultima step
    (Bom para comparar cenário com e sem DQN, mas não mostra evolução durante o treinamento)
"""

import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci
import time

# ---------------- CONFIG ---------------- #

SUMO_BINARY = "sumo"  # ou "sumo-gui"
NET_FILE = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalNetwork.net.xml"
ROUTE_FILE = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalRoutes.rou.xml"

TL_IDS = ["J15", "J20", "J7", "J8"]

STEP_PER_ACTION = 5
MAX_EPISODES = 50
MAX_STEPS = 3600

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 20000
INITIAL_EPS = 1.0
FINAL_EPS = 0.05
EPS_DECAY = 0.02

MIN_GREEN = 10
STATE_SIZE = 10

# --------------- METRICS STORAGE ---------------- #

all_episode_metrics = []


# ---------------- REPLAY BUFFER ---------------- #

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)


# ---------------- DQN NETWORK ---------------- #

class DQNNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, tl_id, state_dim, action_dim, device="cpu"):
        self.id = tl_id
        self.device = device
        self.policy = DQNNet(state_dim, action_dim).to(device)
        self.target = DQNNet(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.replay = ReplayBuffer(REPLAY_CAPACITY)
        self.action_dim = action_dim
        self.eps = INITIAL_EPS
        self.update_count = 0

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randrange(self.action_dim)

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q = self.policy(s)

        return int(q.argmax().cpu())

    def push(self, *args):
        self.replay.push(*args)

    def update(self):
        if len(self.replay) < BATCH_SIZE:
            return

        s, a, r, ns, d = self.replay.sample(BATCH_SIZE)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.policy(s).gather(1, a)

        with torch.no_grad():
            q_next = self.target(ns).max(1)[0].unsqueeze(1)
            q_target = r + GAMMA * q_next * (1 - d)

        loss = nn.functional.mse_loss(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        if self.update_count % 200 == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def decay_epsilon(self):
        """Decai epsilon de forma controlada"""
        self.eps = max(FINAL_EPS, self.eps - EPS_DECAY)


# -------------- STATE ---------------- #

def get_local_state(tl_id):
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    values = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
    arr = np.array(values[:STATE_SIZE], dtype=np.float32)
    return np.pad(arr, (0, STATE_SIZE - arr.shape[0]), "constant")


# -------------- GLOBAL REWARD ---------------- #

def compute_global_reward():
    total = sum(traci.lane.getWaitingTime(l) for l in traci.lane.getIDList())
    return -(total/100)  # menor espera = melhor


# -------------- ACTION APPLICATION ---------------- #

def apply_action_to_tl(tl_id, action, min_green=MIN_GREEN):
    current = traci.trafficlight.getPhase(tl_id)

    try:
        elapsed = traci.trafficlight.getPhaseTime(tl_id)
    except:
        elapsed = 0

    if elapsed < 0:
        elapsed = 0

    if action != current and elapsed < min_green:
        action = current

    traci.trafficlight.setPhase(tl_id, action)


# -------------- METRICS COLLECTION CLASS ---------------- #

class EpisodeMetrics:
    """Classe para gerenciar métricas de um episódio"""
    
    def __init__(self, tl_ids):
        self.depart_times = {}
        self.travel_times = []
        self.waiting_times = []
        self.queue_lengths = []
        self.vehicle_stops = []  # stops acumulados por veículo
        self.co2_emissions = []
        self.fuel_consumptions = []
        self.teleports = []
        self.throughput = 0
        self.phase_changes = {tl: 0 for tl in tl_ids}
        self.last_phase = {}
        self.vehicle_stop_count = {}  # rastreia stops por veículo
        
    def collect_step_metrics(self, tl_ids):
        """Coleta métricas de um step"""
        
        # Travel time tracking
        for vid in traci.simulation.getDepartedIDList():
            self.depart_times[vid] = traci.simulation.getTime()
            self.vehicle_stop_count[vid] = 0  # inicializa contador
        
        for vid in traci.simulation.getArrivedIDList():
            self.throughput += 1
            if vid in self.depart_times:
                self.travel_times.append(
                    traci.simulation.getTime() - self.depart_times[vid]
                )
                del self.depart_times[vid]
            if vid in self.vehicle_stop_count:
                del self.vehicle_stop_count[vid]
        
        # Aggregate network metrics
        self.waiting_times.append(
            sum(traci.lane.getWaitingTime(l) for l in traci.lane.getIDList())
        )
        self.queue_lengths.append(
            sum(traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList())
        )
        
        # Conta stops por veículo (quando velocidade cai abaixo de threshold)
        for vid in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(vid)
            if vid not in self.vehicle_stop_count:
                self.vehicle_stop_count[vid] = 0
            # Se parou (velocidade < 0.1 m/s)
            if speed < 0.1:
                self.vehicle_stop_count[vid] += 1
        
        total_stops = sum(self.vehicle_stop_count.values())
        self.vehicle_stops.append(total_stops)
        
        self.co2_emissions.append(
            sum(traci.vehicle.getCO2Emission(v) for v in traci.vehicle.getIDList())
        )
        self.fuel_consumptions.append(
            sum(traci.vehicle.getFuelConsumption(v) for v in traci.vehicle.getIDList())
        )
        self.teleports.append(traci.simulation.getStartingTeleportNumber())
        
        # Phase switches
        for tl in tl_ids:
            phase = traci.trafficlight.getPhase(tl)
            if tl in self.last_phase and phase != self.last_phase[tl]:
                self.phase_changes[tl] += 1
            self.last_phase[tl] = phase
    
    def get_summary(self):
        """Retorna resumo das métricas do episódio"""
        return {
            "mean_travel_time": np.mean(self.travel_times) if self.travel_times else 0,
            "mean_waiting_time": np.mean(self.waiting_times) if self.waiting_times else 0,
            "mean_queue_length": np.mean(self.queue_lengths) if self.queue_lengths else 0,
            "mean_stops": np.mean(self.vehicle_stops) if self.vehicle_stops else 0,
            "mean_co2": np.mean(self.co2_emissions) if self.co2_emissions else 0,
            "mean_fuel": np.mean(self.fuel_consumptions) if self.fuel_consumptions else 0,
            "total_throughput": self.throughput,
            "total_teleports": sum(self.teleports),
            "phase_changes": self.phase_changes.copy()
        }


# -------------- MAIN TRAINING LOOP ---------------- #

def main():
    
    # Cria agentes UMA VEZ (fora do loop de episódios)
    print("Inicializando agentes...")
    traci.start([SUMO_BINARY, "-n", NET_FILE, "-r", ROUTE_FILE])
    
    agents = {}
    for tl in TL_IDS:
        num_phases = len(
            traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0].phases
        )
        agents[tl] = Agent(tl, STATE_SIZE, num_phases)
        print(f"  {tl}: {num_phases} phases")
    
    traci.close()
    
    # Loop de episódios
    for ep in range(1, MAX_EPISODES + 1):
        
        print(f"\n===== EPISODE {ep}/{MAX_EPISODES} =====")
        print(f"Epsilon: {agents[TL_IDS[0]].eps:.4f}")
        
        # Inicia simulação
        traci.start([SUMO_BINARY, "-n", NET_FILE, "-r", ROUTE_FILE])
        
        # Cria novo objeto de métricas para este episódio
        metrics = EpisodeMetrics(TL_IDS)
        
        total_reward = 0
        
        # Loop de steps
        for step in range(0, MAX_STEPS, STEP_PER_ACTION):
            
            # Observa estados
            states = {tl: get_local_state(tl) for tl in TL_IDS}
            
            # Seleciona ações
            actions = {tl: agents[tl].select_action(states[tl]) for tl in TL_IDS}
            
            # Aplica ações
            for tl in TL_IDS:
                apply_action_to_tl(tl, actions[tl])
            
            # Executa steps da simulação
            for _ in range(STEP_PER_ACTION):
                traci.simulationStep()
                metrics.collect_step_metrics(TL_IDS)
            
            # Calcula recompensa
            reward = compute_global_reward()
            total_reward += reward
            
            # Observa próximos estados
            next_states = {tl: get_local_state(tl) for tl in TL_IDS}
            
            # Determina se é o último step
            done = (step + STEP_PER_ACTION >= MAX_STEPS)
            
            # Armazena transições e treina
            for tl in TL_IDS:
                agents[tl].push(
                    states[tl], 
                    actions[tl], 
                    reward, 
                    next_states[tl], 
                    done
                )
                agents[tl].update()
        
        traci.close()
        
        # Decai epsilon APÓS o episódio
        for tl in TL_IDS:
            agents[tl].decay_epsilon()
        
        # Obtém resumo das métricas
        summary = metrics.get_summary()
        
        # Imprime resultados
        print(f"Reward: {total_reward:.2f}")
        print(f"Travel Time: {summary['mean_travel_time']:.2f} s")
        print(f"Waiting Time: {summary['mean_waiting_time']:.2f} s")
        print(f"Queue Length: {summary['mean_queue_length']:.2f}")
        print(f"Stops/vehicle: {summary['mean_stops']:.2f}")
        print(f"CO₂: {summary['mean_co2']:.2f} g/s")
        print(f"Fuel: {summary['mean_fuel']:.2f} ml/s")
        print(f"Throughput: {summary['total_throughput']}")
        print(f"Teleports: {summary['total_teleports']}")
        print("Phase switches:", summary['phase_changes'])
        
        # Armazena para análise posterior
        all_episode_metrics.append({
            "episode": ep,
            "reward": total_reward,
            **summary
        })
    
    print("\nTREINAMENTO CONCLUÍDO")
    
    # Salva modelos
    print("\nSalvando modelos...")
    for tl in TL_IDS:
        torch.save(
            agents[tl].policy.state_dict(), 
            f"model_{tl}_ep{MAX_EPISODES}.pth"
        )
    print("Modelos salvos!")


if __name__ == "__main__":
    main()