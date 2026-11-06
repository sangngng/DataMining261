from pathlib import Path

tex_file = Path(r"C:\Users\Admin\Documents\DataMining BTL\DataMining261\report\chapters\06_evaluation.tex")
with open(tex_file, "r", encoding="utf-8") as f:
    text = f.read()

target = r"""    \textbf{V38 (Titan Apex)} & Premature Core Overhaul + Application Benchmark Elevation (\#1275, \#1318) & $0.61840$ & $0.53920$ & $0.81500$ & $374$ & $0.66516$ (DCG Disruption) \\
    \rowcolor{PrismLabel}
    \textbf{V39 (Pure Consensus)} & Frozen Top-18 Core + Surgical Mid-Tier Optimization (Ranks 19--35) & $\mathbf{0.62710}$ & $\mathbf{0.54950}$ & $\mathbf{0.81500}$ & $374$ & \textbf{Restoration Peak} \\"""

replacement = r"""    \textbf{V38 (Titan Apex)} & Premature Core Overhaul + Application Benchmark Elevation (\#1275, \#1318) & $0.61840$ & $0.53920$ & $0.81500$ & $374$ & $0.66516$ (DCG Disruption) \\
    \rowcolor{PrismLabel}
    \textbf{V39 (Pure Consensus)} & Frozen Top-18 Core + Surgical Mid-Tier Optimization (Ranks 19--35) & $0.62710$ & $0.54950$ & $0.81500$ & $374$ & $0.66844$ (Restored Core) \\
    \textbf{V40 (Breakthrough Apex)} & Frozen Top-24 Core + Clinical Drag Purge (\#1006, \#1003) + ASP/KR Gems (\#1372, \#877) & $\mathbf{0.62850}$ & $\mathbf{0.55120}$ & $\mathbf{0.81500}$ & $374$ & \textbf{Target Top 5 / 4} \\"""

# Normalize newlines for replacement
text_normalized = text.replace("\r\n", "\n")
target_normalized = target.replace("\r\n", "\n")
replacement_normalized = replacement.replace("\r\n", "\n")

if target_normalized in text_normalized:
    updated = text_normalized.replace(target_normalized, replacement_normalized)
    with open(tex_file, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(updated)
    print("SUCCESS: Updated Table 6.1 in 06_evaluation.tex")
else:
    print("ERROR: Target text not found!")
