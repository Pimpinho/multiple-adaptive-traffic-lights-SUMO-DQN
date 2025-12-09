import traci
import numpy as np
from subprocess import Popen
import time
import csv

# Normalizar os dados de extração de acordo com o ambiente SUMO usado.
# Captura o maximo número de veículos e o máximo tempo de espera em cada faixa monitorada.


SUMO_BINARY = "sumo"  # ou "sumo-gui"
SUMO_CFG = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\sumo\\ufalConfig.sumocfg"

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

ALL_LANES = []
for lanes in lanes_by_tl.values():
    ALL_LANES.extend(lanes)
ALL_LANES = list(set(ALL_LANES))


def start_sumo():
    cmd = [
        SUMO_BINARY,
        "-c", SUMO_CFG,
        "--remote-port", "8813",
        "--step-length", "1.0"
    ]
    print("Starting SUMO:", cmd)
    Popen(cmd)
    time.sleep(0.4)
    traci.init(8813)


def analyze():
    max_count = {lane: 0 for lane in ALL_LANES}
    max_wait = {lane: 0.0 for lane in ALL_LANES}

    sim_time = 0

    while True:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # coleta
        for lane in ALL_LANES:
            try:
                c = traci.lane.getLastStepVehicleNumber(lane)
                w = traci.lane.getWaitingTime(lane)
            except:
                continue

            if c > max_count[lane]:
                max_count[lane] = c

            if w > max_wait[lane]:
                max_wait[lane] = w

        # fim da simulação?
        if traci.simulation.getMinExpectedNumber() == 0:
            break

    traci.close()

    return max_count, max_wait


def save_csv(max_count, max_wait):
    with open("lane_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lane", "MaxVehicleCount", "MaxWaitingTime"])
        for lane in ALL_LANES:
            writer.writerow([lane, max_count[lane], max_wait[lane]])
    print("Saved lane_stats.csv")


if __name__ == "__main__":
    start_sumo()
    max_count, max_wait = analyze()

    print("\n====== RESULTADOS ======")
    print(f"(Simulação completa analisada)\n")

    for lane in ALL_LANES:
        print(f"{lane:15}  maxCount={max_count[lane]:4}   maxWait={max_wait[lane]:6.2f}")

    save_csv(max_count, max_wait)

    print("\n veja 'lane_stats.csv' para tabela detalhada.\n")
