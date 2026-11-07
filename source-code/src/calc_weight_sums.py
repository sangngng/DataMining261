import numpy as np

# Let's inspect the sum of weights for different M
for m in [250, 300, 320, 340, 350, 360, 365, 370, 374]:
    r = np.arange(1, m + 1)
    w = 1.0 / np.log2(r + 1.0)
    print(f"M={m:3d} | sum(w) = {np.sum(w):.4f} | w_M = {w[-1]:.4f}")
