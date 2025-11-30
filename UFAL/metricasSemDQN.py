"""
Baseline SUMO Traffic Simulation (Sem DQN)
- Roda apenas 1 vez
- Usa controle semafórico original do arquivo .net.xml
- Coleta mesmas métricas da versão com DQN
"""

import traci
import numpy as np

# ---------- CONFIG ---------- #

SUMO_BINARY = "sumo"  # ou "sumo-gui"
NET_FILE = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalNetwork.net.xml"
ROUTE_FILE = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalRoutes.rou.xml"

MAX_STEPS = 3600
TL_IDS = ["J15", "J20", "J7", "J8"]  # apenas para medir trocas de fase


# ---------- METRICS CLASS ---------- #

class EpisodeMetrics:
    def __init__(self, tl_ids):
        self.depart_times = {}
        self.travel_times = []
        self.waiting_times = []
        self.queue_lengths = []
        self.vehicle_stops = []
        self.co2_emissions = []
        self.fuel_consumptions = []
        self.teleports = []
        self.throughput = 0
        self.phase_changes = {tl: 0 for tl in tl_ids}
        self.last_phase = {}
        self.vehicle_stop_count = {}

    def collect_step_metrics(self, tl_ids):

        # Track vehicle entry & exit for travel time
        for vid in traci.simulation.getDepartedIDList():
            self.depart_times[vid] = traci.simulation.getTime()
            self.vehicle_stop_count[vid] = 0
        
        for vid in traci.simulation.getArrivedIDList():
            self.throughput += 1
            if vid in self.depart_times:
                self.travel_times.append(
                    traci.simulation.getTime() - self.depart_times[vid]
                )
                del self.depart_times[vid]
            if vid in self.vehicle_stop_count:
                del self.vehicle_stop_count[vid]

        # Whole-network metrics
        self.waiting_times.append(
            sum(traci.lane.getWaitingTime(l) for l in traci.lane.getIDList())
        )
        self.queue_lengths.append(
            sum(traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList())
        )

        # Stops, CO2, Fuel
        for vid in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(vid) < 0.1:
                self.vehicle_stop_count[vid] = self.vehicle_stop_count.get(vid, 0) + 1
        
        self.vehicle_stops.append(sum(self.vehicle_stop_count.values()))
        self.co2_emissions.append(
            sum(traci.vehicle.getCO2Emission(v) for v in traci.vehicle.getIDList())
        )
        self.fuel_consumptions.append(
            sum(traci.vehicle.getFuelConsumption(v) for v in traci.vehicle.getIDList())
        )

        # Teleports
        self.teleports.append(traci.simulation.getStartingTeleportNumber())

        # Phase changes per TL
        for tl in tl_ids:
            phase = traci.trafficlight.getPhase(tl)
            if tl in self.last_phase and phase != self.last_phase[tl]:
                self.phase_changes[tl] += 1
            self.last_phase[tl] = phase

    def get_summary(self):
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


# ---------- MAIN ---------- #

def main():

        print("🚦 Executando baseline sem DQN...")

        traci.start([
            SUMO_BINARY,
            "-n", NET_FILE,
            "-r", ROUTE_FILE,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000"
        ])

        metrics = EpisodeMetrics(TL_IDS)

        for _ in range(MAX_STEPS):
            traci.simulationStep()
            metrics.collect_step_metrics(TL_IDS)

        traci.close()

        summary = metrics.get_summary()

        print("\n📊 RESULTADOS BASELINE (SEM DQN)")
        print(f"  Throughput: {summary['total_throughput']}")
        print(f"  Travel Time: {summary['mean_travel_time']:.2f}s")
        print(f"  Waiting Time: {summary['mean_waiting_time']:.2f}s")
        print(f"  Queue Length: {summary['mean_queue_length']:.2f}")
        print(f"  CO2: {summary['mean_co2']:.2f}")
        print(f"  Fuel: {summary['mean_fuel']:.2f}")
        print(f"  Stops/vehicle: {summary['mean_stops']:.2f}")
        print(f"  Teleports: {summary['total_teleports']}")
        print("  Phase switches:", summary["phase_changes"])

        print("\n✅ Baseline concluído — pronto para comparar com DQN!")


if __name__ == "__main__":
    main()