"""
Baseline SUMO Traffic Simulation (Sem DQN)
Versão final – SEM métricas de teleporte
"""

import traci
import numpy as np


# ============================================================
# CONFIGURAÇÃO
# ============================================================

SUMO_BINARY = "sumo"
SUMO_CONFIG = r"C:\Users\abraao\Documents\GitHub\adaptative-traffic-lights\UFAL\sumo\ufalConfig.sumocfg"

MAX_STEPS = 3600
TL_IDS = ["tl1", "tl2", "tl3"]


# ============================================================
# CLASSE DE MÉTRICAS (SEM TELEPORTS)
# ============================================================

class EpisodeMetrics:
    def __init__(self, tl_ids):
        self.tl_ids = tl_ids

        # Travel time
        self.depart_times = {}
        self.travel_times = []

        # Stops por evento
        self.prev_speed = {}
        self.stop_events = {}
        self.mean_stops_series = []

        # Métricas de rede
        self.waiting_times = []
        self.queue_lengths = []
        self.co2_emissions = []
        self.fuel_consumptions = []

        # Throughput
        self.throughput = 0

        # Phase changes
        self.last_phase = {tl: None for tl in tl_ids}
        self.phase_changes = {tl: 0 for tl in tl_ids}


    def collect_step_metrics(self):
        sim_time = traci.simulation.getTime()

        # ---------------------------
        # Departures
        t = sim_time
        for vid in traci.simulation.getDepartedIDList():
            self.depart_times[vid] = t
            self.prev_speed[vid] = traci.vehicle.getSpeed(vid)
            self.stop_events[vid] = 0

        # ---------------------------
        # Arrivals
        for vid in traci.simulation.getArrivedIDList():
            self.throughput += 1

            if vid in self.depart_times:
                self.travel_times.append(t - self.depart_times[vid])

            # cleanup
            self.depart_times.pop(vid, None)
            self.prev_speed.pop(vid, None)
            self.stop_events.pop(vid, None)

        # ---------------------------
        # Waiting time & Queue length
        lanes = traci.lane.getIDList()
        self.waiting_times.append(sum(traci.lane.getWaitingTime(l) for l in lanes))
        self.queue_lengths.append(sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes))

        # ---------------------------
        # Stops (evento)
        for vid in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(vid)

            if speed < 0.1 and self.prev_speed.get(vid, 1.0) >= 0.1:
                self.stop_events[vid] = self.stop_events.get(vid, 0) + 1

            self.prev_speed[vid] = speed

        if len(self.stop_events) > 0:
            self.mean_stops_series.append(
                sum(self.stop_events.values()) / len(self.stop_events)
            )

        # ---------------------------
        # Emissões
        vehicles = traci.vehicle.getIDList()
        self.co2_emissions.append(sum(traci.vehicle.getCO2Emission(v) for v in vehicles))
        self.fuel_consumptions.append(sum(traci.vehicle.getFuelConsumption(v) for v in vehicles))

        # ---------------------------
        # Phase changes
        for tl in self.tl_ids:
            phase = traci.trafficlight.getPhase(tl)
            if self.last_phase[tl] is not None and phase != self.last_phase[tl]:
                self.phase_changes[tl] += 1
            self.last_phase[tl] = phase


    def summary(self):
        return {
            "mean_travel_time": np.mean(self.travel_times) if self.travel_times else 0,
            "mean_waiting_time": np.mean(self.waiting_times),
            "mean_queue_length": np.mean(self.queue_lengths),
            "mean_stops": np.mean(self.mean_stops_series),
            "mean_co2": np.mean(self.co2_emissions),
            "mean_fuel": np.mean(self.fuel_consumptions),
            "total_throughput": self.throughput,
            "phase_changes": self.phase_changes.copy(),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CONFIG,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000"
    ])

    metrics = EpisodeMetrics(TL_IDS)

    for _ in range(MAX_STEPS):
        traci.simulationStep()
        metrics.collect_step_metrics()

    traci.close()

    summary = metrics.summary()

    print("\n #RESULTADOS BASELINE")
    print(f"# Throughput:        {summary['total_throughput']}")
    print(f"# Travel Time:       {summary['mean_travel_time']:.2f}s")
    print(f"# Waiting Time:      {summary['mean_waiting_time']:.2f}")
    print(f"# Queue Length:      {summary['mean_queue_length']:.2f}")
    print(f"# Stops/vehicle:     {summary['mean_stops']:.2f}")
    print(f"# CO₂ Emission:      {summary['mean_co2']:.2f}")
    print(f"# Fuel Consumption:  {summary['mean_fuel']:.2f}")
    print(f"# Phase switches:    {summary['phase_changes']}")

if __name__ == "__main__":
    main()
