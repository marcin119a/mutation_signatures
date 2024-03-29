import numpy as np


def FrobeniusNorm(M, P, E):
    return np.sqrt(np.sum((M - np.dot(P, E))**2))


def is_wholenumber(x, tol=1e-15):
    return np.abs(x - np.round(x)) < tol


def calculate_BIC(M, exposures, errors):
    n = M.shape[1]  # Number of patients
    k = exposures.shape[0]  # Number of signatures
    RSS = np.sum(errors**2)

    log_likelihood = -n/2 * np.log(RSS)
    BIC = k * np.log(n) - 2 * log_likelihood

    return BIC


def load_and_process_data(patient_index, mutational_profiles, predf_mutational_signatures):
        # Load and process the first file
        profiles = np.genfromtxt(mutational_profiles, delimiter=',', skip_header=1)
        if profiles.size == 0:
            raise ValueError(f"Empty data in {mutational_profiles}")
        profiles = np.delete(profiles, 0, axis=1)

        profile = profiles
        if patient_index is not None:
            profile = profiles[:, patient_index]

        # Load and process the second file
        signatures = np.genfromtxt(predf_mutational_signatures, delimiter=',', skip_header=1)
        if signatures.size == 0:
            raise ValueError(f"Empty data in {predf_mutational_signatures}")
        signatures = np.delete(signatures, 0, axis=1)

        # Return processed data
        return profile, signatures

def calculate_sensitivity_specificity(predicted, actual, total_values):
    predicted_set = set(predicted)
    actual_set = set(actual)
    all_values_set = set(range(total_values))

    true_positives = len(predicted_set.intersection(actual_set))
    true_negatives = len(all_values_set.difference(actual_set).difference(predicted_set))

    total_positives = len(actual_set)
    total_negatives = total_values - total_positives

    sensitivity = true_positives / total_positives if total_positives else 0
    specificity = true_negatives / total_negatives if total_negatives else 0

    return sensitivity, specificity