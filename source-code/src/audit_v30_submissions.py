import pandas as pd
import numpy as np

test = pd.read_csv('source-code/data/test.csv')

def audit_submission(path, name):
    df = pd.read_csv(path)
    print(f"=== AUDITING {name} ({path}) ===")
    assert len(df) == 596, f"Wrong length: {len(df)}"
    assert list(df.columns) == ['id', 'priority'], f"Wrong columns: {df.columns}"
    assert (df['id'] == test['id']).all(), "IDs do not match test.csv exactly!"
    assert not df['priority'].isna().any(), "Contains NaNs!"
    assert not np.isinf(df['priority']).any(), "Contains Infs!"
    
    active = df[df['priority'] > 0]
    zeros = df[df['priority'] == 0]
    n_act = len(active)
    n_zero = len(zeros)
    n_uniq = len(active['priority'].unique())
    
    print(f"Total rows: {len(df)}/596")
    print(f"Active count: {n_act} (Expected: 374)")
    print(f"Zero count: {n_zero} (Expected: 222)")
    print(f"Unique active priorities: {n_uniq} (Expected: 374, ZERO TIES)")
    print(f"Max priority: {df['priority'].max():.6f}")
    print(f"Min active priority: {active['priority'].min():.6f}")
    assert n_act == 374
    assert n_zero == 222
    assert n_uniq == 374
    print(f"[PASS] 100% Compliant and Ready for Kaggle Evaluation!\n")

audit_submission('source-code/submissions/submission_v30a_pinnacle_apex_374.csv', 'V30A Pinnacle Apex')
audit_submission('source-code/submissions/submission_v30b_consolidated_apex_374.csv', 'V30B Consolidated Dual-Apex')
