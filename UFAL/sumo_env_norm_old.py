import os
import csv
import time
import numpy as np
import traci
import traci.constants as tc
from subprocess import Popen

# Máximos extraídos do analyze_traffic_stats
LANE_MAX_VEHICLES = {
    "dOrigem_1": 7,
    "E4_1": 15,
    "E5_0": 20,
    "E1_1": 68,
    "dOrigem_0": 8,
    "E5_1": 21,
    "E5_2": 25,
    "-E5_0": 15,
    "E1_2": 63,
    "-E5_2": 17,
    "saidaufal_0": 5,
    "dOrigem_2": 8,
    "E4_3": 1,   # era 0 -> coloquei 1 pra evitar divisão por zero
    "-E1_0": 26,
    "saidaufal_1": 4,
    "E1_0": 75,
    "inter1Origem_0": 6,
    "-E5_1": 12,
    "E4_0": 16,
    "E4_2": 15,
    "-E1_2": 17,
    "-E1_1": 23,
}

LANE_MAX_WAITING = {
    "dOrigem_1": 212.0,
    "E4_1": 474.0,
    "E5_0": 764.0,
    "E1_1": 1256.0,
    "dOrigem_0": 217.0,
    "E5_1": 816.0,
    "E5_2": 987.0,
    "-E5_0": 426.0,
    "E1_2": 1302.0,
    "-E5_2": 441.0,
    "saidaufal_0": 199.0,
    "dOrigem_2": 225.0,
    "E4_3": 1.0,   # era 0 -> coloquei 1 pra evitar divisão por zero
    "-E1_0": 192.0,
    "saidaufal_1": 140.0,
    "E1_0": 1029.0,
    "inter1Origem_0": 231.0,
    "-E5_1": 421.0,
    "E4_0": 471.0,
    "E4_2": 406.0,
    "-E1_2": 156.0,
    "-E1_1": 158.0,
}

class SUMOEnv:
    def __init__(self,
                 sumo_binary="sumo",         # "sumo" or "sumo-gui"
                 sumo_cfg="C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalConfig.sumocfg",
                 tl_ids=("tl1","tl2","tl3"),
                 lanes_by_tl=None,
                 step_length=1.0,
                 control_interval=5):
        """
        lanes_by_tl: dict mapping tl_id -> list of lane_ids (strings)
        """
        self.SUMO_BINARY = sumo_binary
        self.sumo_cfg = sumo_cfg
        self.tl_ids = list(tl_ids)
        assert len(self.tl_ids) == 3
        self.lanes_by_tl = lanes_by_tl or {
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
        self.step_length = step_length
        self.control_interval = control_interval
        assert control_interval % step_length == 0
        self.control_steps = int(control_interval / step_length)
        self.sumo_proc = None
        self.connected = False
        self.csvfile = None
        self.csvwriter = None

        # para reward com delta de waiting time
        self.last_total_wait = 0.0

    def start_sumo(self, gui=False):
        sumo_bin = self.SUMO_BINARY

        cmd = [
            sumo_bin,
            "-c", self.sumo_cfg,
            "--step-length", str(self.step_length),
            "--remote-port", "8813"     # ESSENCIAL para TraCI
        ]

        print("Starting SUMO with command:", cmd)

        # Inicia SUMO
        self.sumo_proc = Popen(cmd)

        # Aguarda SUMO abrir a porta
        time.sleep(0.5)

        # Conecta ao TraCI
        traci.init(port=8813)
        self.connected = True

    def close(self):
        try:
            if self.connected:
                traci.close()
        except Exception:
            pass
        if self.sumo_proc:
            try:
                self.sumo_proc.terminate()
            except Exception:
                pass
        self.connected = False

    def reset(self):
        # Start sumo and run a tiny warmup
        if self.connected:
            try:
                traci.load(["-c", self.sumo_cfg, "--step-length", str(self.step_length)])
            except Exception:
                # fallback: restart process (safer)
                self.close()
                self.start_sumo(gui=False)
        else:
            self.start_sumo(gui=False)

        # warmup steps to let vehicles appear
        for _ in range(5):
            traci.simulationStep()

        # inicializa o acumulador de espera para o reward delta
        self.last_total_wait = self._compute_total_wait()

        # prepare csv logging
        self._open_csv()
        return self._get_state()

    def step(self, action, episode=0, step_idx=0):
        """
        action: 0,1,2 -> advance phase for tl1/tl2/tl3 respectively
        Returns: next_state, reward, done, info
        """
        # apply action: advance phase of selected TL
        if action == 1:
            self._advance_tl(self.tl_ids[0])  # tl1
        elif action == 2:
            self._advance_tl(self.tl_ids[1])  # tl2
        elif action == 3:
            self._advance_tl(self.tl_ids[2])  # tl3
        elif action == 0:
            pass  # NO-OP
        else:
            raise ValueError("Invalid action")

        # simula control_interval segundos
        for _ in range(self.control_steps):
            traci.simulationStep()

        # próximo estado
        next_state = self._get_state()

        # total de waiting time atual (em segundos)
        total_wait = self._compute_total_wait()

        # reward com delta de waiting time:
        # se o espera total diminuiu, reward > 0
        # se aumentou, reward < 0
        reward = self.last_total_wait - total_wait
        self.last_total_wait = total_wait

        done = False  # controle de término fica fora (episódio = duração da simulação)

        # log
        sim_time = traci.simulation.getTime()
        phases = [traci.trafficlight.getPhase(t) for t in self.tl_ids]
        self._log_csv(episode, step_idx, sim_time, total_wait, action, phases)
        return next_state, reward, done, {"sim_time": sim_time, "phases": phases}

    def _compute_total_wait(self):
        """Soma o waiting time de todas as lanes monitoradas."""
        total_wait = 0.0
        for lanes in self.lanes_by_tl.values():
            for lane_id in lanes:
                try:
                    total_wait += traci.lane.getWaitingTime(lane_id)
                except Exception:
                    total_wait += 0.0
        return total_wait

    def _advance_tl(self, tl_id):
        cur = traci.trafficlight.getPhase(tl_id)
        new = (cur + 1) % 2
        traci.trafficlight.setPhase(tl_id, new)

    def _get_state(self):
        vec = []
        # para cada TL, acrescenta (para cada lane: count_norm, wait_norm) e depois a fase
        for tl in self.tl_ids:
            lanes = self.lanes_by_tl.get(tl, [])
            for lane in lanes:
                try:
                    cnt = traci.lane.getLastStepVehicleNumber(lane)
                    wait = traci.lane.getWaitingTime(lane)
                except Exception:
                    cnt = 0.0
                    wait = 0.0

                # pega máximos por lane; se não achar, usa defaults conservadores
                max_cnt = LANE_MAX_VEHICLES.get(lane, 80)      # fallback: 80 veículos
                max_wait = LANE_MAX_WAITING.get(lane, 1400.0)  # fallback: ~23 min

                # evita divisão por zero
                if max_cnt <= 0:
                    max_cnt = 1.0
                if max_wait <= 0:
                    max_wait = 1.0

                # normalização [0, 1], com clipping
                cnt_norm = min(cnt / max_cnt, 1.0)
                wait_norm = min(wait / max_wait, 1.0)

                vec.append(float(cnt_norm))
                vec.append(float(wait_norm))

            # add phase normalizada (0 ou 1, já está em faixa boa)
            try:
                phase = traci.trafficlight.getPhase(tl)
            except Exception:
                phase = 0
            # como cada tl tem 2 fases (0 ou 1), isso já é "normalizado"
            vec.append(float(phase))

        return np.array(vec, dtype=np.float32)

    def _open_csv(self):
        if self.csvfile is None:
            fname = "sumo_env_log.csv"
            self.csvfile = open(fname, "w", newline="")
            self.csvwriter = csv.writer(self.csvfile)
            self.csvwriter.writerow([
                "episode", "step", "sim_time",
                "total_waiting_time", "action",
                "phase_tl1", "phase_tl2", "phase_tl3"
            ])
            self.csvfile.flush()

    def _log_csv(self, episode, step, sim_time, total_wait, action, phases):
        if self.csvwriter:
            row = [episode, step, sim_time, total_wait, action] + phases
            self.csvwriter.writerow(row)
            self.csvfile.flush()
