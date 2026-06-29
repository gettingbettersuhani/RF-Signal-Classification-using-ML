from pathlib import Path
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def load_dataset(dataset_path):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )

    logging.info("Loading dataset...")

    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file, encoding="latin1")

    logging.info("Dataset loaded successfully.")

    return dataset


def get_modulations(dataset):
    """
    Return sorted modulation list.
    """

    return sorted(
        list(
            set(key[0] for key in dataset.keys())
        )
    )


def get_snrs(dataset):
    """
    Return sorted SNR values.
    """

    return sorted(
        list(
            set(key[1] for key in dataset.keys())
        )
    )


def get_total_samples(dataset):

    total = 0

    for value in dataset.values():

        total += value.shape[0]

    return total

def dataset_summary(dataset):
    """
    Generate a summary dictionary for the dataset.
    """

    modulations = get_modulations(dataset)
    snrs = get_snrs(dataset)
    total_samples = get_total_samples(dataset)

    first_key = list(dataset.keys())[0]

    sample_shape = dataset[first_key].shape

    summary = {
        "Total Samples": total_samples,
        "Total Modulations": len(modulations),
        "Total SNR Levels": len(snrs),
        "Samples Per Combination": sample_shape[0],
        "IQ Shape": sample_shape
    }

    return summary