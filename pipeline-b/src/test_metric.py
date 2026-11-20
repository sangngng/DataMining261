import numpy as np
from metric import score
assert abs(2*.8*.6/1.4 - 0.685714) < 1e-6
# empty list -> 0
assert score([1,2],[0,0],[5,5]) == 0
# no worthwhile paper -> R=0 -> 0
assert score([1,2],[1,2],[1,1]) == 0
# tie-break by id as text: "1000" before "999"
s,Q,R = score([999,1000],[5,5],[1,5],True); assert abs(Q - 1/(1+1/np.log2(3))) < 1e-9, Q
# priority 0 excluded but still in recall denominator
s,Q,R = score([1,2],[1,0],[5,3],True); assert Q == 1 and R == 0.5
# perfect single paper rating 5 -> Q=1
print("metric tests passed")
