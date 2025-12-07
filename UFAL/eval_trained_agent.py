# eval_trained_agent.py
import time
import numpy as np
import torch
import torch.nn as nn

from sumo_env_norm import SUMOEnv  # ambiente normalizado com base no SUMO usado
# from sumo_env import SUMO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === DQN: use a MESMA arquitetura que você usou no treino ===
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def select_action_greedy(model, state_np):
    """Escolhe a ação com maior Q (sem exploração)."""
    state_t = torch.from_numpy(state_np).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_values = model(state_t)
    action = int(q_values.argmax(dim=1).item())
    return action


if __name__ == "__main__":
    # 1) Cria o ambiente usando o SUMO-GUI
    env = SUMOEnv(
        sumo_binary="sumo-gui",  # <<< IMPORTANTE para ver a simulação
        sumo_cfg=r"C:\Users\USUARIO(A)\Documents\GitHub\adaptative-traffic-lights\UFAL\sumo\ufalConfig.sumocfg",
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

    # 2) Reset ambiente e descobre o tamanho do estado
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = 4  # ações: 0=tl1, 1=tl2, 2=tl3

    print(f"State dim = {state_dim}, action dim = {action_dim}")

    # 3) Cria a rede e carrega o modelo treinado
    model = DQN(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load("dqn_single_agent.pth", map_location=DEVICE))
    model.eval()

    # 4) Roda um episódio completo (sim_time até 3600)
    done = False
    total_reward = 0.0
    step_idx = 0

    print("\n=== Rodando episódio com política treinada (modo avaliação) ===")
    while True:
        # ação puramente greedy
        action = select_action_greedy(model, state)

        next_state, reward, done, info = env.step(
            action, episode=0, step_idx=step_idx
        )

        total_reward += reward
        state = next_state
        step_idx += 1

        sim_time = info["sim_time"]
        phases = info["phases"]

        print(
            f"Step {step_idx:03d} | sim_time={sim_time:.1f} | "
            f"action={action} | reward={reward:.2f} | phases={phases}"
        )

        # critério de parada: 3600 segundos de simulação
        if sim_time >= 3600:
            break

        # OPCIONAL: desacelerar um pouco para você conseguir ver melhor no GUI
        # time.sleep(0.1)

    print("\n==================== Episódio finalizado ====================")
    print(f"Total reward (episódio) = {total_reward:.2f}")
    env.close()
