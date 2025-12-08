"""
Coleta HALTED VEHICLES (veículos parados) do SUMO SEM DQN
- halted por step
- halted médio
Executa diretamente com o arquivo .sumocfg
"""

import traci
import numpy as np
import matplotlib.pyplot as plt

SUMO_BINARY = "sumo"
SUMO_CFG = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\sumo\\ufalConfig.sumocfg"

MAX_STEPS = 3600


def run_baseline_halted():
    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CFG,           
        "--no-step-log", "true"
    ])

    halted_values = []

    for _ in range(MAX_STEPS):
        traci.simulationStep()

        halted = sum(traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList())
        halted_values.append(halted)

    traci.close()

    halted_mean = np.mean(halted_values)

    print("\n=== HALTED (BASELINE) ===")
    print(f"Média de halted: {halted_mean:.2f}")
    print(f"Máximo: {max(halted_values)}")
    print(f"Mínimo: {min(halted_values)}")

    # gráfico por step
    plt.figure(figsize=(10, 5))
    plt.plot(halted_values, label="Halted por step")
    plt.title("Veículos Parados (Halted) — Baseline")
    plt.xlabel("Step")
    plt.ylabel("Halted")
    plt.grid(True)
    plt.legend()
    plt.savefig("halted_baseline_steps.png", dpi=300)
    plt.show()

    return halted_values, halted_mean


if __name__ == "__main__":
    run_baseline_halted()
