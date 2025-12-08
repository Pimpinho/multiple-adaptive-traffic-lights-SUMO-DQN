# test_state_extraction.py
import time
import numpy as np
from sumo_env_norm import SUMOEnv
import random

if __name__ == "__main__":
    # Ajuste aqui:
    SUMO_BINARY = "sumo"       
    SUMO_CFG = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\UFAL\\ufalConfig.sumocfg"  # caminho para seu arquivo .sumocfg
    
    env = SUMOEnv(sumo_binary=SUMO_BINARY,
                  sumo_cfg=SUMO_CFG,
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
                    },
                  step_length=1.0,
                  control_interval=5)

    # reset environment (inicia SUMO)
    state = env.reset()
    print("Initial state vector (len={}):".format(len(state)))
    print(state)

    # run 20 agent steps with random actions and print key metrics

    for i in range(20):
        action = random.choice([0,1,2])
        next_state, reward, done, info = env.step(action, episode=0, step_idx=i)
        print(f"Step {i}: action={action}, sim_time={info['sim_time']:.1f}, reward={reward:.2f}, phases={info['phases']}")
        # optionally inspect a slice of state
        print(" next_state (first 12 values):", next_state[:12])
    env.close()
    print("Test finished.")
