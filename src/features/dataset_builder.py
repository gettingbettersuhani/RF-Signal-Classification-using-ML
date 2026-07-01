"""
dataset_builder.py

Production-ready dataset builder for the RadioML project.
"""

from pathlib import Path
import logging
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import src.features.extract_features as feat


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def build_feature_dataset(dataset):
    """
    Convert the RadioML dictionary into a feature dataframe.
    """

    rows = []
    total_groups = len(dataset)

    for idx, ((modulation, snr), signals) in enumerate(dataset.items(), start=1):

        if idx == 1 or idx % 10 == 0:
            logging.info(
                "Processing group %d/%d (%s, SNR=%s)",
                idx,
                total_groups,
                modulation,
                snr,
            )

        for signal in signals:

            features = feat.extract_features(signal)

            features["Modulation"] = modulation
            features["SNR"] = snr

            rows.append(features)

    dataframe = pd.DataFrame(rows)

    logging.info("Feature extraction completed.")

    return dataframe


def validate_dataset(df):
    """
    Validate dataframe before training.
    """

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    numeric = df.select_dtypes(include="number")

    report["infinite_values"] = int(
        numeric.isin([float("inf"), float("-inf")]).sum().sum()
    )

    return report


def encode_labels(df):
    """
    Encode modulation labels.
    """

    encoder = LabelEncoder()

    df["Label"] = encoder.fit_transform(df["Modulation"])

    logging.info("Labels encoded successfully.")

    return df, encoder


def split_dataset(df, test_size=0.2, random_state=42):
    """
    Stratified train-test split.
    """

    X = df.drop(columns=["Modulation", "Label"])
    y = df["Label"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def save_outputs(
    output_folder,
    dataframe,
    X_train,
    X_test,
    y_train,
    y_test,
    encoder,
):
    """
    Save processed datasets.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_folder / "feature_dataset.csv", index=False)

    try:
        dataframe.to_parquet(
            output_folder / "feature_dataset.parquet",
            index=False,
        )
    except Exception as exc:
        logging.warning("Parquet not saved: %s", exc)

    X_train.to_csv(output_folder / "X_train.csv", index=False)
    X_test.to_csv(output_folder / "X_test.csv", index=False)

    y_train.to_frame("Label").to_csv(
        output_folder / "y_train.csv",
        index=False,
    )

    y_test.to_frame("Label").to_csv(
        output_folder / "y_test.csv",
        index=False,
    )

    joblib.dump(
        encoder,
        output_folder / "label_encoder.pkl",
    )

    report = validate_dataset(dataframe)

    with open(
        output_folder / "dataset_statistics.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write("=" * 60 + "\n")
        file.write("DATASET SUMMARY\n")
        file.write("=" * 60 + "\n\n")

        for key, value in report.items():
            file.write(f"{key:20}: {value}\n")

        file.write("\n")
        file.write("Class Distribution\n")
        file.write("=" * 60 + "\n")

        counts = dataframe["Modulation"].value_counts()

        for label, count in counts.items():
            file.write(f"{label:15} : {count}\n")

    logging.info("Outputs saved successfully.")


def build_complete_pipeline(dataset, output_folder):
    """
    Complete production pipeline.
    """

    logging.info("Building feature dataset...")

    dataframe = build_feature_dataset(dataset)

    dataframe, encoder = encode_labels(dataframe)

    X_train, X_test, y_train, y_test = split_dataset(dataframe)

    save_outputs(
        output_folder,
        dataframe,
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
    )

    logging.info("Pipeline completed successfully.")

    return dataframe, X_train, X_test, y_train, y_test, encoder
