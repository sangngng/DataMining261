import pandas as pd
import numpy as np

train_df = pd.read_csv('source-code/data/train.csv')
title_lower = train_df['title'].str.lower().fillna('')

topics = {
    "ASP / Answer Set": "answer set|asp |clingo|gringo|dlv|hex-program|xasp",
    "NNV / Neural Network Verif": "neural network|nnv|imagestar|deep neural|verification of neural",
    "Epistemic / Knowledge": "epistemic|reasoning about knowledge|multi-agent epistemic",
    "Argumentation": "argumentation|bipolar|abstract argumentation|aspic",
    "Description Logics / Ontol": "description logic|ontology|ontologies|el |dl-lite",
    "Logic Programming": "logic program|logic programming|stable model|disjunctive logic",
    "Planning": "planning|pddl|temporal planning",
    "Causality / Abduction": "causality|causal|abduction|abductive",
    "SMT / SAT Solving": "smt solver|sat solver|satisfiability|cdcl",
    "Hardware / Model Checking": "hardware|verilog|vhdl|fpga|cache coherence|circuit",
    "Petri Nets": "petri net",
    "Quantum Computation": "quantum",
    "Smart Contracts": "smart contract",
}

print("="*85)
print("=== TOPIC PERFORMANCE IN TRAINING SET ===")
print("="*85)
topic_stats = []
for name, pattern in topics.items():
    mask = title_lower.str.contains(pattern)
    sub = train_df[mask]
    n = len(sub)
    if n > 0:
        mean_r = sub['Label'].mean()
        r5 = (sub['Label'] == 5).mean()
        r4 = (sub['Label'] == 4).mean()
        pos = (sub['Label'] >= 2).mean()
        topic_stats.append({
            "Topic": name, "N": n, "Mean": mean_r,
            "5-star%": r5 * 100, "4-star%": r4 * 100, "Pos%": pos * 100
        })

df_res = pd.DataFrame(topic_stats).sort_values("Mean", ascending=False)
for _, r in df_res.iterrows():
    print(f"{r['Topic']:28s} | N={r['N']:3d} | Mean={r['Mean']:.2f} | 5*={r['5-star%']:4.1f}% | 4*={r['4-star%']:4.1f}% | Pos={r['Pos%']:5.1f}%")
