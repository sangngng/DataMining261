from pathlib import Path

tex_file = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\report\chapters\06_evaluation.tex")
with open(tex_file, "r", encoding="utf-8") as f:
    text = f.read()

target = r"""    \textbf{V40 (Breakthrough Apex)} & Frozen Top-24 Core + Clinical Drag Purge (\#1006, \#1003) + ASP/KR Gems (\#1372, \#877) & $\mathbf{0.62850}$ & $\mathbf{0.55120}$ & $\mathbf{0.81500}$ & $374$ & \textbf{Target Top 5 / 4} \\
    \bottomrule"""

replacement = r"""    \textbf{V40 (Breakthrough Apex)} & Frozen Top-24 Core + Over-Pruning of KR Medical Workshop (\#1006, \#1003) & $0.62410$ & $0.54520$ & $0.81500$ & $374$ & $0.66802$ (Over-Pruning Dip) \\
    \rowcolor{PrismLabel}
    \textbf{V41 (Zenith Precision)} & Restored V37 Core + Single-Pair Calibration (\#1419 $\leftrightarrow$ \#1053) & $\mathbf{0.62780}$ & $\mathbf{0.55040}$ & $\mathbf{0.81500}$ & $374$ & \textbf{Rank 7 / Safe Peak} \\
    \bottomrule"""

text_normalized = text.replace("\r\n", "\n")
target_normalized = target.replace("\r\n", "\n")
replacement_normalized = replacement.replace("\r\n", "\n")

if target_normalized in text_normalized:
    updated = text_normalized.replace(target_normalized, replacement_normalized)
    with open(tex_file, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(updated)
    print("SUCCESS: Updated Table 6.1 in 06_evaluation.tex for V40 and V41")
else:
    print("ERROR: Target text not found!")
