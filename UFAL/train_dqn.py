# train_dqn_single_agent.py
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from sumo_env_norm import SUMOEnv


# ==========================
# 1) Hyperparâmetros
# ==========================
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
TARGET_UPDATE_STEPS = 1000

NUM_EPISODES = 20     # ajuste depois
MAX_STEPS_PER_EPISODE = 720  # 3600s / 5s (control_interval=5)

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 20_000     # steps globais


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# 2) Replay Buffer
# ==========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # tudo como numpy/float simples aqui
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ==========================
# 3) Rede Q (MLP simples)
# ==========================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==========================
# 4) Agente DQN
# ==========================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim

        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

        self.global_step = 0

    def select_action(self, state, epsilon):
        """
        state: numpy array (state_dim,)
        epsilon: float
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def maybe_update_target(self):
        if self.global_step % TARGET_UPDATE_STEPS == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================
# 5) Função auxiliar: epsilon por step global
# ==========================
def get_epsilon(global_step):
    if global_step >= EPS_DECAY_STEPS:
        return EPS_END
    frac = global_step / EPS_DECAY_STEPS
    return EPS_START + frac * (EPS_END - EPS_START)


# ==========================
# 6) Loop de treinamento
# ==========================
def train():
    # Cria ambiente SUMO
    env = SUMOEnv(
        sumo_binary="sumo",
        # se quiser ver a simulação, troque por "sumo-gui"
        sumo_cfg=r"C:\Users\abraao\Documents\GitHub\adaptative-traffic-lights\UFAL\sumo\ufalConfig.sumocfg",
        tl_ids=("tl1", "tl2", "tl3"),
        lanes_by_tl={
            "tl1": [
                "inter1Origem_0",
                "E5_0", "E5_1", "E5_2",
                "dOrigem_0", "dOrigem_1", "dOrigem_2"
            ],
            "tl2": [
                "E1_0", "E1_1", "E1_2",
                "-E5_0", "-E5_1", "-E5_2"
            ],
            "tl3": [
                "saidaufal_0", "saidaufal_1",
                "E4_0", "E4_1", "E4_2", "E4_3",
                "-E1_0", "-E1_1", "-E1_2"
            ]
        },
        step_length=1.0,
        control_interval=5
    )

    # Estado inicial só para descobrir dimensão
    state = env.reset()
    state_dim = len(state)
    action_dim = 4  # 0=tl1, 1=tl2, 2=tl3, 3=no OP

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    agent = DQNAgent(state_dim, action_dim)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0.0
        episode_losses = []

        for step_idx in range(MAX_STEPS_PER_EPISODE):
            epsilon = get_epsilon(agent.global_step)
            action = agent.select_action(state, epsilon)

            next_state, reward, done, info = env.step(action, episode=episode, step_idx=step_idx)

            # IMPORTANTE: reward aqui deve ser deltaWaitingTime no SUMOEnv
            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            agent.global_step += 1
            agent.maybe_update_target()

            total_reward += reward
            state = next_state

            # critério de parada opcional
            sim_time = info.get("sim_time", 0.0)
            if sim_time >= 3600.0:
                break
            if done:
                break

        # fim do episódio
        stats = env.get_episode_stats()

        mean_loss = np.mean(episode_losses) if episode_losses else 0.0
        print(
            f"\n--------------------------------------------------------------------------------\n"
            f"[Episode {episode:03d}] \n"
            f"total_reward={total_reward:.2f} \n"
            f"mean_loss={mean_loss:.4f} epsilon={epsilon:.3f} \n"
            f"finished={stats['finished_vehicles']} \n"
            f"mean_travel={stats['mean_travel_time']:.2f}s \n"
            f"mean_halted={stats['mean_halted_per_step']:.2f}\n"
            f"--------------------------------------------------------------------------------\n"
        )


    # Salva pesos
    torch.save(agent.q_net.state_dict(), "dqn_single_agent.pth")
    print("Treinamento finalizado. Modelo salvo em dqn_single_agent.pth")
    env.close()


if __name__ == "__main__":
    train()
