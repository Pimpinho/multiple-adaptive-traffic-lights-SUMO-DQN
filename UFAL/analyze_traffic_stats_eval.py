# analyze_traffic_stats_eval.py

import csv
import time

import numpy as np
import torch
import torch.nn as nn
import traci

from sumo_env_norm import SUMOEnv  # mesmo ambiente usado no treino

# ===== CONFIGURAÇÕES =====

SUMO_BINARY = "sumo-gui"  # ou "sumo" se quiser sem interface
SUMO_CFG = r"C:\Users\USUARIO(A)\Documents\GitHub\adaptative-traffic-lights\UFAL\sumo\ufalConfig.sumocfg"
MODEL_PATH = "dqn_single_agent.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Lanes monitoradas ----
lanes_by_tl = {
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
}

ALL_LANES = sorted({lane for lanes in lanes_by_tl.values() for lane in lanes})


# ===== DQN (MESMA ARQUITETURA DO TREINO) =====

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


def save_csv(max_count, max_wait, filename="lane_stats_eval.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lane", "MaxVehicleCount", "MaxWaitingTime"])
        for lane in ALL_LANES:
            writer.writerow([lane, max_count[lane], max_wait[lane]])
    print(f"Saved {filename}")


if __name__ == "__main__":
    # 1) Cria o ambiente SUMO **igual ao do treino**
    env = SUMOEnv(
        sumo_binary=SUMO_BINARY,
        sumo_cfg=SUMO_CFG,
        tl_ids=("tl1", "tl2", "tl3"),
        lanes_by_tl=lanes_by_tl,
        step_length=1.0,
        control_interval=5
    )

    # 2) Reset: descobre state_dim (deve bater com o do treino → 47)
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = 3  # 0=tl1, 1=tl2, 2=tl3

    print(f"State dim = {state_dim}, action dim = {action_dim}")

    # 3) Cria o DQN e carrega o modelo treinado
    model = DQN(state_dim, action_dim).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Se você salvou direto o state_dict:
    # torch.save(model.state_dict(), MODEL_PATH)
    if isinstance(checkpoint, dict) and all(
        k in checkpoint for k in ["net.0.weight", "net.0.bias"]
    ):
        # Parece ser diretamente o state_dict
        model.load_state_dict(checkpoint)
    else:
        # Caso tenha salvo como {"q_network": ..., ...}
        model.load_state_dict(checkpoint["q_network"])

    model.eval()

    # 4) Estruturas para análise das lanes
    max_count = {lane: 0 for lane in ALL_LANES}
    max_wait = {lane: 0.0 for lane in ALL_LANES}

    total_reward = 0.0
    step_idx = 0

    print("\n=== Rodando episódio com política treinada (modo avaliação + análise de tráfego) ===")

    while True:
        # Ação puramente greedy do DQN treinado
        action = select_action_greedy(model, state)

        next_state, reward, done, info = env.step(
            action, episode=0, step_idx=step_idx
        )

        total_reward += reward
        sim_time = info["sim_time"]
        phases = info.get("phases", None)

        # ---- Coleta dos dados de tráfego por lane (usando TraCI) ----
        for lane in ALL_LANES:
            try:
                c = traci.lane.getLastStepVehicleNumber(lane)
                w = traci.lane.getWaitingTime(lane)
            except Exception:
                continue

            if c > max_count[lane]:
                max_count[lane] = c
            if w > max_wait[lane]:
                max_wait[lane] = w

        print(
            f"Step {step_idx:03d} | sim_time={sim_time:.1f} | "
            f"action={action} | reward={reward:.2f} | phases={phases}"
        )

        state = next_state
        step_idx += 1

        # critério de parada: 3600 segundos de simulação ou done
        if sim_time >= 3600 or done:
            break

        # Se quiser deixar mais lento pra visualizar no GUI:
        # time.sleep(0.1)

    print("\n=== Episódio finalizado ===")
    print(f"Total reward (episódio) = {total_reward:.2f}")

    # Fecha ambiente (desconecta do TraCI)
    env.close()

    # 5) Mostra resultados e salva CSV
    print("\n====== RESULTADOS (máx por lane) ======")
    for lane in ALL_LANES:
        print(f"{lane:15}  maxCount={max_count[lane]:4}   maxWait={max_wait[lane]:6.2f}")

    save_csv(max_count, max_wait)

    print("\n🎉 ANÁLISE COMPLETA — veja 'lane_stats_eval.csv' para a tabela detalhada.\n")
