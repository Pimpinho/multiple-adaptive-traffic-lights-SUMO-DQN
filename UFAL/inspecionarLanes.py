# inspect_sumo.py

# Inspeciona e imprime informações sobre os semáforos e lanes controladas em uma simulação SUMO.
# Útil para entender a topologia do tráfego e verificar se as lanes estão corretas.
# Pega exatamente qual lanes (fala a lane exata e não a via principal) são controladas por cada semáforo.

import os
import traci
import sys

SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo")
SUMOCFG = r"C:\Users\USUARIO(A)\Documents\GitHub\adaptative-traffic-lights\UFAL\sumo\ufalConfig.sumocfg"

traci.start([SUMO_BINARY, "-c", SUMOCFG])
print("Traffic Lights (ids):")
tls = traci.trafficlight.getIDList()
print(tls)

print("\nFor each TL, controlled lanes and controlled links:")
for tl in tls:
    try:
        cl = traci.trafficlight.getControlledLanes(tl)
        links = traci.trafficlight.getControlledLinks(tl)
        print(f"\n{tl}:")
        print("  Controlled lanes:", cl)
        print("  Controlled links:", links)
    except Exception as e:
        print(f"  Error getting data for {tl}: {e}")

print("\nAll lane IDs (first 200 shown):")
all_lanes = traci.lane.getIDList()
for i, lid in enumerate(all_lanes[:200]):
    print(f"  {lid}")
print(f"... total lanes: {len(all_lanes)}")
traci.close()
