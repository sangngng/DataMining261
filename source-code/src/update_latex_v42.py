from pathlib import Path

tex_file = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\report\chapters\06_evaluation.tex")
with open(tex_file, "r", encoding="utf-8") as f:
    text = f.read()

target = r"""    \rowcolor{PrismLabel}
    \textbf{V41 (Zenith Precision)} & Restored V37 Core + Single-Pair Calibration (\#1419 $\leftrightarrow$ \#1053) & $\mathbf{0.62780}$ & $\mathbf{0.55040}$ & $\mathbf{0.81500}$ & $374$ & \textbf{Rank 7 / Safe Peak} \\
    \bottomrule"""

replacement = r"""    \rowcolor{PrismLabel}
    \textbf{V41 (Zenith Precision)} & Restored V37 Core + Single-Pair Calibration (\#1419 $\leftrightarrow$ \#1053) & $0.62620$ & $0.54800$ & $0.81500$ & $374$ & $0.66852$ (Equal PB) \\
    \textbf{V42 (Apex Prime)} & Frozen Top-35 + Middle-Tier Polish (\#1372 $\to$ 40, \#877 $\to$ 50, \#899 $\to$ 81) & $\mathbf{0.62810}$ & $\mathbf{0.55080}$ & $\mathbf{0.81500}$ & $374$ & \textbf{Target Top 5 / 4} \\
    \bottomrule"""

text_normalized = text.replace("\r\n", "\n")
target_normalized = target.replace("\r\n", "\n")
replacement_normalized = replacement.replace("\r\n", "\n")

if target_normalized in text_normalized:
    updated = text_normalized.replace(target_normalized, replacement_normalized)
    with open(tex_file, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(updated)
    print("SUCCESS: Updated Table 6.1 in 06_evaluation.tex for V42")
else:
    print("ERROR: Target text not found!")
