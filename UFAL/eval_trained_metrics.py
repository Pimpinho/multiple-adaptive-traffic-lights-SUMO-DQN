"""
Avaliação do modelo DQN treinado
Versão final — SEM métricas de teleporte
"""

import time
import numpy as np
import torch
import torch.nn as nn
import traci

from sumo_env_norm import SUMOEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DQN
# ============================================================
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
    state_t = torch.from_numpy(state_np).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_values = model(state_t)
    return int(q_values.argmax(dim=1).item())


# ============================================================
# MÉTRICAS — MESMO FORMATO DO BASELINE, SEM TELEPORTS
# ============================================================

class EpisodeMetrics:
    def __init__(self, tl_ids):
        self.tl_ids = tl_ids

        self.depart_times = {}
        self.travel_times = []

        self.prev_speed = {}
        self.stop_events = {}
        self.mean_stops_series = []

        self.waiting_times = []
        self.queue_lengths = []
        self.co2_emissions = []
        self.fuel_consumptions = []

        self.throughput = 0

        self.last_phase = {tl: None for tl in tl_ids}
        self.phase_changes = {tl: 0 for tl in tl_ids}

    def collect_step(self):
        sim_time = traci.simulation.getTime()

        # Departures
        for vid in traci.simulation.getDepartedIDList():
            self.depart_times[vid] = sim_time
            self.prev_speed[vid] = traci.vehicle.getSpeed(vid)
            self.stop_events[vid] = 0

        # Arrivals
        for vid in traci.simulation.getArrivedIDList():
            self.throughput += 1

            if vid in self.depart_times:
                self.travel_times.append(sim_time - self.depart_times[vid])

            self.depart_times.pop(vid, None)
            self.prev_speed.pop(vid, None)
            self.stop_events.pop(vid, None)

        # Waiting & Queue
        lanes = traci.lane.getIDList()
        self.waiting_times.append(sum(traci.lane.getWaitingTime(l) for l in lanes))
        self.queue_lengths.append(sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes))

        # Stops por evento
        for vid in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(vid)
            if speed < 0.1 and self.prev_speed.get(vid, 1.0) >= 0.1:
                self.stop_events[vid] = self.stop_events.get(vid, 0) + 1
            self.prev_speed[vid] = speed

        if len(self.stop_events) > 0:
            self.mean_stops_series.append(
                sum(self.stop_events.values()) / len(self.stop_events)
            )

        # Emissões
        vehicles = traci.vehicle.getIDList()
        self.co2_emissions.append(sum(traci.vehicle.getCO2Emission(v) for v in vehicles))
        self.fuel_consumptions.append(sum(traci.vehicle.getFuelConsumption(v) for v in vehicles))

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
# AVALIAÇÃO DO MODELO DQN
# ============================================================

if __name__ == "__main__":

    env = SUMOEnv(
        sumo_binary="sumo",
        sumo_cfg="C:\\Users\\abraao\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\sumo\\ufalConfig.sumocfg",
        tl_ids=("tl1", "tl2", "tl3"),
        step_length=1.0,
        control_interval=5
    )

    state = env.reset()
    state_dim = len(state)
    action_dim = 4

    model = DQN(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load("dqn_single_agent.pth", map_location=DEVICE))
    model.eval()

    metrics = EpisodeMetrics(["tl1", "tl2", "tl3"])

    total_reward = 0
    step_idx = 0

    while True:
        action = select_action_greedy(model, state)
        next_state, reward, done, info = env.step(action, episode=0, step_idx=step_idx)

        metrics.collect_step()

        total_reward += reward
        state = next_state
        step_idx += 1

        if info["sim_time"] >= 3600:
            break

    env.close()

    summary = metrics.summary()

    print("\n============ RESULTADOS DO DQN ============")
    print(f"# Total reward:       {total_reward:.2f}")
    print(f"# Throughput:         {summary['total_throughput']}")
    print(f"# Travel Time:        {summary['mean_travel_time']:.2f}s")
    print(f"# Waiting Time:       {summary['mean_waiting_time']:.2f}")
    print(f"# Queue Length:       {summary['mean_queue_length']:.2f}")
    print(f"# Stops/vehicle:      {summary['mean_stops']:.2f}")
    print(f"# CO₂ Emission:       {summary['mean_co2']:.2f}")
    print(f"# Fuel Consumption:   {summary['mean_fuel']:.2f}")
    print(f"# Phase switches:     {summary['phase_changes']}")
    print("\n")
