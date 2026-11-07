from pathlib import Path

tex_file = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\report\chapters\06_evaluation.tex")
with open(tex_file, "r", encoding="utf-8") as f:
    text = f.read()

target = r"""    \textbf{V42 (Apex Prime)} & Frozen Top-35 + Middle-Tier Polish (\#1372 $\to$ 40, \#877 $\to$ 50, \#899 $\to$ 81) & $\mathbf{0.62810}$ & $\mathbf{0.55080}$ & $\mathbf{0.81500}$ & $374$ & \textbf{Target Top 5 / 4} \\
    \bottomrule"""

replacement = r"""    \textbf{V42 (Apex Prime)} & Frozen Top-35 + Middle-Tier Polish (\#1372 $\to$ 40, \#877 $\to$ 50, \#899 $\to$ 81) & $0.62530$ & $0.54710$ & $0.81500$ & $374$ & $0.66823$ (KG Displacement) \\
    \rowcolor{PrismLabel}
    \textbf{V43 (Multi-Apex Consensus)} & 4-Peak Weighted Rank Ensemble (40\% V41 + 30\% V37 + 20\% V36 + 10\% V34) & $\mathbf{0.62860}$ & $\mathbf{0.55150}$ & $\mathbf{0.81500}$ & $374$ & \textbf{Consensus Zenith} \\
    \bottomrule"""

text_normalized = text.replace("\r\n", "\n")
target_normalized = target.replace("\r\n", "\n")
replacement_normalized = replacement.replace("\r\n", "\n")

if target_normalized in text_normalized:
    updated = text_normalized.replace(target_normalized, replacement_normalized)
    with open(tex_file, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(updated)
    print("SUCCESS: Updated Table 6.1 in 06_evaluation.tex for V43")
else:
    print("ERROR: Target text not found!")
