from pathlib import Path
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler


def scale_features(
    X_train,
    X_test,
    output_folder,
):
    """
    Scale training and testing features using StandardScaler.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    output_folder = Path(output_folder)

    joblib.dump(
        scaler,
        output_folder / "standard_scaler.pkl"
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )