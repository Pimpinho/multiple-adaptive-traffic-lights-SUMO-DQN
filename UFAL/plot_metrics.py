import matplotlib.pyplot as plt
import numpy as np


# ======================================================
# FUNÇÃO PRINCIPAL PARA GERAR TODOS OS GRÁFICOS
# ======================================================
def plot_all_metrics(baseline, dqn, save=False):

    # --------------------------------------------------
    # 1) Métricas escalares simples (barras lado a lado)
    # --------------------------------------------------
    scalar_metrics = [
        ("mean_travel_time", "Tempo Médio de Viagem (s)"),
        ("mean_waiting_time", "Tempo Médio de Espera (s)"),
        ("mean_queue_length", "Fila Média na Rede (veículos)"),
        ("mean_stops", "Stops por Veículo"),
        ("mean_co2", "Emissão Média de CO₂ (mg/s)"),
        ("mean_fuel", "Consumo Médio de Combustível (ml/s)"),
        ("total_throughput", "Throughput Total (veículos)")
    ]

    for key, title in scalar_metrics:
        baseline_value = baseline[key]
        dqn_value = dqn[key]

        plt.figure(figsize=(7, 5))
        plt.bar(["Baseline", "DQN"], [baseline_value, dqn_value], color=["gray", "green"])
        plt.title(f"{title} — Comparação Baseline × DQN", fontsize=13)
        plt.ylabel(title)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        if save:
            plt.savefig(f"plot_{key}.png", dpi=300)
        plt.show()


    # --------------------------------------------------
    # 2) Fase por semáforo
    # --------------------------------------------------
    tl_ids = list(baseline["phase_changes"].keys())

    baseline_phases = [baseline["phase_changes"][tl] for tl in tl_ids]
    dqn_phases = [dqn["phase_changes"][tl] for tl in tl_ids]

    x = np.arange(len(tl_ids))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, baseline_phases, width, label="Baseline", color="gray")
    plt.bar(x + width/2, dqn_phases, width, label="DQN", color="green")

    plt.xticks(x, tl_ids)
    plt.ylabel("Trocas de Fase")
    plt.title("Trocas de Fase por Semáforo — Baseline × DQN")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    if save:
        plt.savefig("plot_phase_changes.png", dpi=300)
    plt.show()

    print("\n🎉 Todos os gráficos foram gerados com sucesso!\n")


# ======================================================
# EXEMPLO DE USO
# ======================================================
if __name__ == "__main__":

    # Exemplo — substitua pelos seus resultados reais
    baseline = {
        "mean_travel_time": 12.5,
        "mean_waiting_time": 201.2,
        "mean_queue_length": 33.1,
        "mean_stops": 2.1,
        "mean_co2": 350.0,
        "mean_fuel": 1.42,
        "total_throughput": 3288,
        "phase_changes": {"tl1": 22, "tl2": 30, "tl3": 28}
    }

    dqn = {
        "mean_travel_time": 9.8,
        "mean_waiting_time": 150.5,
        "mean_queue_length": 20.1,
        "mean_stops": 1.2,
        "mean_co2": 280.0,
        "mean_fuel": 1.20,
        "total_throughput": 3293,
        "phase_changes": {"tl1": 17, "tl2": 21, "tl3": 19}
    }

    plot_all_metrics(baseline, dqn, save=True)
