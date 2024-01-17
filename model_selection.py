import numpy as np
from estimates_exposures import bootstrapSigExposures, crossValidationSigExposures, findSigExposures
from utils import calculate_BIC
from decompose import decomposeQP

def runBootstrapOnMatrix(
    m, P, R, mutation_count, threshold=0.01, decomposition_method=decomposeQP
):
    def process_column(column):
        exposures, errors = bootstrapSigExposures(
            column, P, R, mutation_count, decomposition_method
        )
        return exposures > threshold

    all_exposures = np.apply_along_axis(process_column, 0, m)

    return 1 - all_exposures.sum(axis=1) / all_exposures.shape[1]



def runCrossvaldiationOnMatrix(
    m, P, fold_size=4, threshold=0.01, decomposition_method=decomposeQP
):
    def process_column(column):
        exposures, errors = crossValidationSigExposures(
            column,
            P,
            fold_size,
            True,
            decomposition_method,
        )
        return exposures > threshold

    all_exposures = np.apply_along_axis(process_column, 0, m)

    return 1 - all_exposures.sum(axis=1) / all_exposures.shape[1]



def backward_elimination(
    m, P, R, threshold, mutation_count, significance_level, decomposition_method=decomposeQP
):
    best_columns = np.arange(P.shape[1])
    P_temp = P
    model_error, _ = findSigExposures(
        m.reshape(-1, 1), P_temp, decomposition_method
    )
    while True:
        changed = False
        p_values = runBootstrapOnMatrix(
            m,
            P_temp,
            R,
            mutation_count=mutation_count,
            threshold=threshold,
            decomposition_method=decomposition_method,
        )

        max_p_value = p_values.max()
        if max_p_value > significance_level:
            indices_with_max = np.where(p_values == max_p_value)[0]
            max_p_var = np.random.choice(indices_with_max)
            best_columns = np.delete(best_columns, max_p_var)
            P_temp = P[:, best_columns]

            changed = True

        if not changed:
            break

    return (
        best_columns,
        bootstrapSigExposures(
            m,
            P_temp,
            mutation_count=mutation_count,
            R=R,
            decomposition_method=decomposition_method,
        ),
        findSigExposures(
            m.reshape(-1, 1), P_temp, decomposition_method=decomposition_method
        ),
    )



def forward_elimination(
    m, P, R, threshold, mutation_count, significance_level, decomposition_method=decomposeQP
):
    best_columns = [1]
    P_temp = np.zeros((P.shape[0], 0))
    while True:
        changed = False
        best_p_value = 1
        best_p_var = -1

        for i in range(P.shape[1] - 1, 0, -1):
            if i in best_columns:
                continue

            # Try adding the ith column to the model
            P_temp_with_i = np.hstack([P_temp, P[:, [i]]])
            p_values = runBootstrapOnMatrix(
                m, P_temp_with_i, R, mutation_count=mutation_count, threshold=threshold
            )

            # Check if the new variable is significant and better than the current best
            if p_values[-1] < best_p_value and p_values[-1] < significance_level:
                best_p_value = p_values[-1]
                best_p_var = i

        # If a significant variable was found, add it to the model
        if best_p_var != -1:
            best_columns.append(best_p_var)
            P_temp = np.hstack([P_temp, P[:, [best_p_var]]])
            changed = True

        # Stop if no new significant variable was found
        if not changed:
            break

    return (
        best_columns,
        bootstrapSigExposures(
            m,
            P_temp,
            mutation_count=mutation_count,
            R=R,
            decomposition_method=decomposition_method,
        ),
        findSigExposures(
            m.reshape(m.shape[0], 1), P_temp, decomposition_method=decomposition_method
        ),
    )


def save_to_dataframe(best_columns, findSigExposures, cancer_type, patient):
    """
    Saves the best_columns and findSigExposures to a pandas DataFrame.
    """
    # Create a DataFrame with findSigExposures as a column
    df = pd.DataFrame(findSigExposures, columns=['findSigExposures'])
    # Add the best_columns as another column, ensuring the length matches
    # If best_columns is shorter, pad with None or a default value
    df['best_columns'] = pd.Series(best_columns).reindex(df.index)
    df['Cancer Types'] = cancer_type
    df['Sample Names'] = patient


    return df

#to test
import pandas as pd
if __name__ == '__main__':
    tumorBRCA = np.genfromtxt('output/M.csv', delimiter='\t', skip_header=1)
    tumorBRCA = np.delete(tumorBRCA, 0, axis=1)

    signaturesCOSMIC = np.genfromtxt('output/WGS_signatures__sigProfiler_SBS_signatures_2019_05_22.modified.csv', delimiter='\t', skip_header=1)
    signaturesCOSMIC = np.delete(signaturesCOSMIC, 0, axis=1)
    df = pd.read_csv('output/WGS-decomposition__PCAWG_sigProfiler_SBS_signatures_in_samples.csv')
    ground_truth = df.drop(columns=['Cancer Types', 'Sample Names', 'Accuracy'])
    ground_truth.columns = [x for x in range(0, 65)]

    ground_truth_df, experiment_df = pd.DataFrame(), pd.DataFrame()

    for i in range(tumorBRCA.shape[1]):
        first_col = tumorBRCA[:, i]
        patient = ground_truth.iloc[i]
        patient = patient / patient.sum()

        non_zero_condition = (patient != 0)
        indexes = non_zero_condition[non_zero_condition].index.tolist()
        try:
            best_columns, b, estimation_exposures = backward_elimination(first_col, signaturesCOSMIC, threshold=0.01, mutation_count=None, R=20, significance_level=0.01)
        except:
            continue
        r = save_to_dataframe(indexes, patient[indexes].to_numpy(), df.iloc[i]['Sample Names'], df.iloc[i]['Cancer Types'])
        experiment_df = pd.concat([r, experiment_df], ignore_index=True)

        r = save_to_dataframe(best_columns, estimation_exposures[0], df.iloc[i]['Sample Names'], df.iloc[i]['Cancer Types'])
        result_df = pd.concat([r, ground_truth_df], ignore_index=True)


    experiment_df.to_csv('output/experiment.csv')
    ground_truth_df.to_csv('output/ground_truth.csv')





