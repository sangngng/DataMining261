import numpy as np

# Simulate a test set of 298 papers (public split)
# Ground truth: ~190 worthwhile papers (ratings 2-5), ~108 unworthwhile (rating 1)
# Let's create an ideal ranking and test perturbations.

np.random.seed(42)
n_papers = 298
n_worthwhile = 190

# Let's say top 187 recommended papers out of 298 (which is 374/596 * 298)
m = 187
r = np.arange(1, m + 1, dtype=float)
w_r = 1.0 / np.log2(r + 1.0)
sum_w = np.sum(w_r)

# Realistic ground truth labels for ranked papers:
# Top 1-18: mostly 5s (g = 1.0) and some 4s (g = 0.75)
# Ranks 19-50: mostly 4s and 3s and some 5s
# Ranks 51-120: mix of 3s, 2s, and unfortunately some 1s (false positives)
# Ranks 121-187: mix of 2s, 3s, and 1s

labels = np.ones(m)
# Top 20: 5s and 4s
labels[:10] = 5.0
labels[10:25] = 4.5
labels[25:50] = 4.0
labels[50:100] = 3.0
labels[100:150] = 2.5
labels[150:187] = 2.0

# Add some false positives (Label 1) at ranks 50-80
labels[53] = 1.0 # rank 54
labels[57] = 1.0 # rank 58
labels[63] = 1.0 # rank 64

def calc(y):
    rec_w = np.sum(y > 1.0)
    R = rec_w / n_worthwhile
    g = (y - 1.0) / 4.0
    Q = np.sum(w_r * g) / sum_w
    score = 2.0 * Q * R / (Q + R)
    return score, Q, R

base_score, base_q, base_r = calc(labels)
print(f"Base Score: {base_score:.6f} (Q={base_q:.6f}, R={base_r:.6f})")

# Test 1: Demote rank 54, 58, 64 (Label 1) to ranks 150-152, replacing them with Label 3.5 papers from lower ranks
labels_demote = labels.copy()
# swap rank 53 with rank 150
labels_demote[53], labels_demote[150] = labels_demote[150], labels_demote[53]
labels_demote[57], labels_demote[151] = labels_demote[151], labels_demote[57]
labels_demote[63], labels_demote[152] = labels_demote[152], labels_demote[63]

s1, q1, r1 = calc(labels_demote)
print(f"After demoting 3 Label 1 papers from R54-64: {s1:.6f} (diff = {s1 - base_score:+.6f})")

# Test 2: Promote a Label 5 paper from Rank 52 to Rank 25
labels_promote = labels.copy()
# Rank 51 (1-based 52) is currently 3.0, let's say it's actually 5.0 in ground truth!
labels_promote[51] = 5.0 # was labeled 3.0 by model, but true is 5.0
# Swap it with rank 24 (1-based 25, currently 4.0)
s_before, _, _ = calc(labels_promote)
labels_promote[51], labels_promote[24] = labels_promote[24], labels_promote[51]
s_after, _, _ = calc(labels_promote)
print(f"After promoting true Label 5 paper from R52 to R25: {s_after:.6f} (diff = {s_after - s_before:+.6f})")
