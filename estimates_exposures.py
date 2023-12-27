import numpy as np
from decompose import decomposeQP
from utils import FrobeniusNorm, is_wholenumber


def findSigExposures(M, P, decomposition_method=decomposeQP):
    """
     Find signature exposures for tumor profiles using specified decomposition method.

     This function allows obtaining the optimal solution by specifying quadratic programming to solve the optimization problem.

     Parameters:
         M (numpy.ndarray): Observed tumor profile matrix for all patients/samples.
             It should have a shape of (96, G), where G is the number of patients.
             Each column can represent mutation counts or mutation probabilities, and
             each column will be normalized to sum up to 1.
         P (numpy.ndarray): Signature profile matrix with a shape of (96, N),
             where N is the number of signatures (e.g., COSMIC: N=30).
         decomposition_method (function, optional): The method selected to get the
             optimal solution. It should be a function. Default is 'decomposeQP'.

     Returns:
         tuple: A tuple containing two numpy arrays.
             - exposures (numpy.ndarray): Matrix of signature exposures per sample/patient (column).
             - errors (numpy.ndarray): Estimation error for each sample/patient (Frobenius norm).

     Raises:
         ValueError: If 'M' and 'P' do not have the same number of rows (mutations types),
             or if 'P' has less than 2 columns, or if 'decomposition_method' is not a function.

     Examples:
         E1 = findSigExposures(tumorBRCA, signaturesCOSMIC, decomposeQP)
         sigsBRCA = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
         E2 = findSigExposures(tumorBRCA, signaturesCOSMIC[:, sigsBRCA], decomposeQP)
         E3 = findSigExposures(np.round(tumorBRCA * 10000), signaturesCOSMIC, decomposeQP)
     """
    # Process and check function parameters
    # M, P
    if M.shape[0] != P.shape[0]:
        raise ValueError("Matrices 'M' and 'P' must have the same number of rows (mutations types).")

    if P.shape[1] == 1:
        raise ValueError("Matrices 'P' must have at least 2 columns (signatures).")

    # decomposition.method
    if not callable(decomposition_method):
        raise ValueError("Parameter 'decomposition_method' must be a function.")

    # Normalize M by column (just in case it is not normalized)
    M = M / M.sum(axis=0)

    # Find solutions
    # Matrix of signature exposures per sample/patient (column)
    exposures = np.apply_along_axis(decomposition_method, 0, M, P)

    # Compute estimation error for each sample/patient (Frobenius norm)
    errors = np.vectorize(lambda i: FrobeniusNorm(M[:, i], P, exposures[:, i]))(range(M.shape[1]))

    return exposures, errors



def bootstrapSigExposures(m, P, R, mutation_count=None, decomposition_method=decomposeQP):
    """
    Obtain the bootstrap distribution of signature exposures for a tumor sample.

    This function allows obtaining the bootstrap distribution of the signature exposures for
    a specific tumor sample using a specified decomposition method.

    Parameters:
        m (numpy.ndarray): Observed tumor profile vector for a patient/sample.
            It should have a shape of (96, 1) and can represent mutation counts or mutation probabilities.
        P (numpy.ndarray): Signature profile matrix with a shape of (96, N),
            where N is the number of signatures (e.g., COSMIC: N=30).
        R (int): The number of bootstrap replicates.
        mutation_count (int, optional): If 'm' is a vector of counts, then 'mutation_count' equals
            the summation of all the counts. If 'm' is probabilities, 'mutation_count' must be specified.
        decomposition_method (function, optional): The method selected to get the optimal solution.
            It should be a function. Default is 'decomposeQP'.

    Returns:
        tuple: A tuple containing two numpy arrays.
            - exposures (numpy.ndarray): Matrix of signature exposures for each bootstrap replicate (column).
            - errors (numpy.ndarray): Estimation error for each bootstrap replicate (Frobenius norm).

    Raises:
        ValueError: If the length of vector 'm' and the number of rows of matrix 'P' do not match,
            if 'P' has less than 2 columns, if 'mutation_count' is not specified and 'm' does not contain counts.

    Examples:
        bootstrapSigExposures(tumorBRCA[:, 1], signaturesCOSMIC, 100, 2000, decomposeQP)
        sigsBRCA = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
        bootstrapSigExposures(tumorBRCA[:, 1], signaturesCOSMIC[:, sigsBRCA], 10, 1000, decomposeQP)
    """

    if len(m) != P.shape[0]:
        raise ValueError("Length of vector 'm' and number of rows of matrix 'P' must be the same.")
    if m.shape[0] != P.shape[0]:
        raise ValueError("Elements of vector 'm' and rows of matrix 'P' must have the same names (mutations types).")
    #if P.shape[1] == 1:
    #    raise ValueError("Matrices 'P' must have at least 2 columns (signatures).")

    # If 'mutation_count' is not specified, 'm' has to contain counts
    if mutation_count is None:
        if all(is_wholenumber(val) for val in m):
            mutation_count = int(m.sum())
        else:
            raise ValueError("Please specify the parameter 'mutation_count' in the function call or provide mutation counts in parameter 'm'.")

    # Normalize m to be a vector of probabilities.
    m = m / np.sum(m)

    # Find optimal solutions using provided decomposition method for each bootstrap replicate
    # Matrix of signature exposures per replicate (column)
    K = len(m)  # number of mutation types

    def bootstrap_sample(m, mutation_count, K):
        mutations_sampled = np.random.choice(K, size=mutation_count, p=m)
        return np.bincount(mutations_sampled, minlength=K) / mutation_count

    exposures = np.column_stack([
        decomposition_method(bootstrap_sample(m, mutation_count, K), P) for _ in range(R)
    ])
    exposures = exposures / np.sum(exposures, axis=0)  # Normalize exposures

    # Compute estimation error for each replicate/trial (Frobenius norm)
    # G x R
    errors = np.vectorize(lambda i: FrobeniusNorm(m, P, exposures[:, i]))(range(exposures.shape[1]))

    return exposures, errors


def crossValidationSigExposures(m, P, fold_size, shuffle=True, decomposition_method=decomposeQP):
    """
    Perform cross-validation to estimate signature exposures for a tumor sample.

    This function performs cross-validation to estimate signature exposures for a specific tumor sample
    using a specified decomposition method.

    Parameters:
        m (numpy.ndarray): Observed tumor profile vector for a patient/sample.
            It should have a shape of (n, 1), where n is the number of mutations.
        P (numpy.ndarray): Signature profile matrix with a shape of (n, N),
            where N is the number of signatures.
        fold_size (int): The number of cross-validation size.
        shuffle (bool): Change the order of mutations
        decomposition_method (function, optional): The method selected to get the optimal solution.
            It should be a function. Default is 'decomposeQP'.

    Returns:
        tuple: A tuple containing two numpy arrays.
            - fold_exposures (numpy.ndarray): Matrix of signature exposures for each cross-validation fold (column).
            - errors (numpy.ndarray): Estimation error for each fold (Frobenius norm).

    Raises:
        ValueError: If the length of vector 'm' and the number of rows of matrix 'P' do not match,
            if 'P' has less than 2 columns.

    Examples:
        num_folds = 5
        sigsBRCA = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
        fold_exposures, errors = crossValidationSigExposures(tumorBRCA[:, 1], signaturesCOSMIC[:, sigsBRCA], num_folds=5, decomposeQP)
    """
    # Process and check function parameters
    P = np.array(P)

    if len(m) != P.shape[0]:
        raise ValueError("Length of vector 'm' and number of rows of matrix 'P' must be the same.")

    if P.shape[1] == 1:
        raise ValueError("Matrices 'P' must have at least 2 columns (signatures).")

    m = m / np.sum(m)

    if shuffle:
        permutation_indices = np.random.permutation(len(m))
        m = m[permutation_indices]
        P = P[permutation_indices,:]

    folds = [m[i:i + fold_size] for i in range(0, (len(m) - (len(m) % fold_size)), fold_size)]

    # Handle the remaining elements that do not fit in full folds
    if len(m) % fold_size != 0:
        last_fold = m[-(len(m) % fold_size):]
        folds.append(last_fold)

    def calculate_fold_exposures(i, num_folds):
        fold = np.concatenate([folds[j] if j != i else [0] * len(folds[i] + 1) for j in range(num_folds)])
        normalized_fold = fold / fold.sum()
        return normalized_fold

    # Perform cross-validation for each replicate
    fold_exposures = np.column_stack([
        decomposition_method(calculate_fold_exposures(i, len(folds)), P)
        for i in range(len(folds))
    ])
    fold_exposures = fold_exposures / np.sum(fold_exposures, axis=0)

    # Compute estimation error for each replicate/trial (Frobenius norm)
    errors = np.vectorize(lambda i: FrobeniusNorm(m, P, fold_exposures[:, i]))(range(fold_exposures.shape[1]))

    return fold_exposures, errors


