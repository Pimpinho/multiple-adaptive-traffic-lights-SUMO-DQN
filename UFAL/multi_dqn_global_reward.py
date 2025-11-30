"""
Multi-agent DQN with GLOBAL shared reward for SUMO/TraCI
- 1 agente por semáforo
- cada agente controla apenas seu TL
- recompensas compartilhadas (cooperação implícita)
- troca de fase só permitida após tempo mínimo (min_green)
"""

import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# SUMO/TraCI
import traci

# ---------------- CONFIG ---------------- #

SUMO_BINARY = "sumo"   # use "sumo-gui" para visualização
NET_FILE = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalNetwork.net.xml"
ROUTE_FILE = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalRoutes.rou.xml"  # substitua pelo seu .rou.xml

TL_IDS = ["J15", "J20", "J7", "J8"]  # IDs que aparecem no seu XML

STEP_PER_ACTION = 5
MAX_EPISODES = 200
MAX_STEPS = 3600

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 20000
INITIAL_EPS = 1.0
FINAL_EPS = 0.05
EPS_DECAY = 1e-4

MIN_GREEN = 10  # impede troca rápida de fase

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

# ---------------- DQN ---------------- #

class DQNNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_out)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, id, state_dim, action_dim, device="cpu"):
        self.id = id
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
        return int(q.argmax().cpu().numpy())

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

        if self.update_count % 100 == 0:
            self.target.load_state_dict(self.policy.state_dict())

        self.eps = max(FINAL_EPS, self.eps - EPS_DECAY)

# ---------------- SUMO HELPERS ---------------- #

def start_sumo(gui=False):
    cmd = [SUMO_BINARY, "-n", NET_FILE]
    if os.path.exists(ROUTE_FILE):
        cmd += ["-r", ROUTE_FILE]
    traci.start(cmd)

def get_local_state(tl_id):
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    values = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
    max_len = 10
    arr = np.array(values[:max_len], dtype=np.float32)
    return np.pad(arr, (0, max_len - arr.shape[0]), "constant")

def compute_global_reward():
    total = 0
    for l in traci.lane.getIDList():
        total += traci.lane.getWaitingTime(l)
    return -total  # minimizar espera → reward negativo

# ✅ alteração pedida — evita trocas frenéticas
    
def apply_action_to_tl(tl_id, action_idx, min_green=10):
    current = traci.trafficlight.getPhase(tl_id)
    # tempo decorrido desde o início da fase atual
    try:
        elapsed = traci.trafficlight.getPhaseTime(tl_id)
    except:
        elapsed = 0  # fallback seguro
    # ---- BUG CRÍTICO CORRIGIDO ----
    # getPhaseTime pode retornar < 0 se a fase acabou de ser definida,
    # então garantimos mínimo 0 para evitar comportamento incorreto
    if elapsed < 0:
        elapsed = 0
    # impede troca prematura
    if action_idx != current and elapsed < min_green:
        action_idx = current
    traci.trafficlight.setPhase(tl_id, int(action_idx))

# ---------------- MAIN TRAINING LOOP ---------------- #

def main():
    start_sumo(gui=False)

    # cria agentes com actions = nº de fases real
    agents = {}
    for tl in TL_IDS:
        phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0].phases
        agents[tl] = Agent(tl, state_dim=10, action_dim=len(phases))

    for ep in range(MAX_EPISODES):
        print(f"\n===== EPISODE {ep+1}/{MAX_EPISODES} =====")
        traci.load(["-n", NET_FILE, "-r", ROUTE_FILE])

        total_reward = 0

        for step in range(0, MAX_STEPS, STEP_PER_ACTION):

            states = {tl: get_local_state(tl) for tl in TL_IDS}
            actions = {tl: agents[tl].select_action(states[tl]) for tl in TL_IDS}

            # aplica ações com min_green
            for tl in TL_IDS:
                apply_action_to_tl(tl, actions[tl])

            for _ in range(STEP_PER_ACTION):
                traci.simulationStep()

            reward = compute_global_reward()
            total_reward += reward

            next_states = {tl: get_local_state(tl) for tl in TL_IDS}

            for tl in TL_IDS:
                agents[tl].push(states[tl], actions[tl], reward, next_states[tl], False)

            for tl in TL_IDS:
                agents[tl].update()

            if step % 300 == 0:
                print(f"step={step}, reward={reward:.1f}")

        print(f"EP {ep+1} total reward = {total_reward:.1f}")

    traci.close()
    print("\n✅ Treinamento finalizado!\n")

if __name__ == "__main__":
    main()
