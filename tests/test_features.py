import numpy as np
from src.data.load_data import load_raw
from src.features.build_features import build_preprocessor
from scipy.sparse import issparse 

def test_preprocessor_output_is_finite_and_2d():
    df = load_raw()
    X = df.drop(columns=["target"])

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Output should be a 2D array (rows x features) - anything else
    # means something in the pipeline returned an unexpected shape.
    assert X_transformed.ndim == 2

    # Row count must match the input - no rows should be silently
    # dropped or duplicated during transformation.
    assert X_transformed.shape[0] == len(X)

    # Handle both sparse and dense output, since scikit learn versions differ
    # in whether OneHotEncoder return a sparse matrix or a plain array.
    danse = X_transformed.toarray() if issparse(X_transformed) else X_transformed
    assert np.isfinite(danse).all()
    