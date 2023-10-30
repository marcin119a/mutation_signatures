from decompose import decomposeQP
import pandas as pd

if __name__ == '__main__':
    tumorBRCA = pd.read_csv('data/tumorBRCA.csv', index_col=0)
    signaturesCOSMIC = pd.read_csv('data/signaturesCOSMIC.csv', index_col=0)
    first_col = tumorBRCA.iloc[:, 0]
    res = decomposeQP(first_col, signaturesCOSMIC.values)
    print(res)