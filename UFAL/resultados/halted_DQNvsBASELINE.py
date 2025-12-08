import matplotlib.pyplot as plt

# ==========================================
# INSIRA AQUI OS VALORES MEDIDOS
# ==========================================
halted_baseline_mean = 93.47
halted_dqn_mean = 21.71
# ==========================================

labels = ['Baseline (Sem DQN)', 'Com DQN']
values = [halted_baseline_mean, halted_dqn_mean]

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, values, color=['blue', 'red'])

# mostrar valores no topo das barras
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.5,
        f'{height:.2f}',
        ha='center',
        va='bottom',
        fontsize=12
    )

plt.title("Comparação da Média de Veículos Parados (Halted)\nBaseline vs. DQN")
plt.ylabel("Halted Médio")
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("halted_comparison_bar.png", dpi=300)
plt.show()
