"""
Coleta HALTED VEHICLES (veículos parados) rodando o modelo DQN treinado
- halted por step
- halted médio
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import traci

from sumo_env_norm import SUMOEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# mesma arquitetura usada no treino
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def select_action_greedy(model, state_np):
    state_t = torch.from_numpy(state_np).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_values = model(state_t)
    return int(q_values.argmax(dim=1).item())


def run_dqn_halted():

    env = SUMOEnv(
        sumo_binary="sumo",
        sumo_cfg="C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\sumo\\ufalConfig.sumocfg",
        tl_ids=("tl1", "tl2", "tl3"),
        control_interval=5
    )

    state = env.reset()
    state_dim = len(state)
    action_dim = 4

    model = DQN(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load("dqn_single_agent.pth", map_location=DEVICE))
    model.eval()

    halted_values = []
    step = 0

    while True:
        action = select_action_greedy(model, state)
        next_state, reward, done, info = env.step(action, episode=0, step_idx=step)

        # coleta halted
        halted = sum(traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList())
        halted_values.append(halted)

        state = next_state
        step += 1

        if info["sim_time"] >= 3600:
            break

    env.close()

    halted_mean = np.mean(halted_values)

    print("\n=== HALTED (DQN) ===")
    print(f"Média de halted: {halted_mean:.2f}")
    print(f"Máximo: {max(halted_values)}")
    print(f"Mínimo: {min(halted_values)}")

    # gráfico por step
    plt.figure(figsize=(10, 5))
    plt.plot(halted_values, color="green", label="Halted por step (DQN)")
    plt.title("Veículos Parados (Halted) — DQN")
    plt.xlabel("Step")
    plt.ylabel("Halted")
    plt.grid(True)
    plt.legend()
    plt.savefig("halted_dqn_steps.png", dpi=300)
    plt.show()

    return halted_values, halted_mean


if __name__ == "__main__":
    run_dqn_halted()
