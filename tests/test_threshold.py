import joblib 

def test_threshold_is_saved_and_reasonable():
    # Assume find_threshold.py has already been run and saved this file 
    threshold = joblib.load("models/threshold.joblib")

    # The threshold must be a valid probability cutoff 
    assert 0.0 < threshold < 1.0

    # Given our cost ratio favors catching defaulters, we expect the threshold 
    # to sit below 0.5 (more cautions than a naive 50/50 cut) but not collape 
    # to something extreme like 0.05, which would mean declining almost everyone 
    # (the failure mode we hit and fixed earlier when the search wasn't constrained).
    assert 0.15 <= threshold <= 0.5