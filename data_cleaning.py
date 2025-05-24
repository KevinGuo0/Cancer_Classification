import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def clean_data(file_path='project_data.csv'):
    data = pd.read_csv(file_path)

    # Useless features
    drop_cols = [
        'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE', 'SEQNO',
        'QSTVER', 'QSTLANG', 'HEIGHT3', 'WEIGHT2', 'HTIN4'
    ]
    data = data.drop(columns=drop_cols, errors='ignore')

    # 替换特殊编码
    data['GENHLTH'] = data['GENHLTH'].replace({7: 4})
    data['PHYSHLTH'] = data['PHYSHLTH'].replace({88: 0, 77: np.nan, 99: np.nan})
    data['MENTHLTH'] = data['MENTHLTH'].replace({88: 0, 77: np.nan, 99: np.nan})
    data['CHECKUP1'] = data['CHECKUP1'].replace({8: 0})
    data['CHOLCHK2'] = data['CHOLCHK2'].replace({8: 7})
    data['CPDEMO1B'] = data['CPDEMO1B'].replace({8: 0})
    data['CHILDREN'] = data['CHILDREN'].replace({88: 0, 99: np.nan})
    data['INCOME2'] = data['INCOME2'].replace({77: np.nan, 99: np.nan})
    data['ALCDAY5'] = data['ALCDAY5'].replace({888: 100, 777: np.nan, 999: np.nan})

    # ALCDAY5 change frequency
    data['ALCDAY5'] = data['ALCDAY5'].apply(lambda x:
        (x - 100) * 4 if 100 <= x < 200 else
        x - 200 if 200 <= x < 300 else x
    )

    # STRENGTH change frequency
    data['STRENGTH'] = data['STRENGTH'].replace({200: np.nan, 777: np.nan, 999: np.nan, 888: 100})
    data['STRENGTH'] = data['STRENGTH'].apply(lambda x:
        (x - 100) * 4 if 100 <= x <= 199 else
        x - 200 if 201 <= x <= 299 else x
    )

    # Change frequency
    freq_cols = ['FRUIT2', 'FRUITJU2', 'FVGREEN1', 'FRENCHF1', 'POTATOE1', 'VEGETAB2']
    for col in freq_cols:
        data[col] = data[col].replace({300: 100, 555: 100, 777: np.nan, 999: np.nan})
        data[col] = data[col].apply(lambda x:
            (x - 100) * 4 if 100 <= x < 200 else
            x - 200 if 201 <= x < 300 else
            (x - 300) // 12 if 301 <= x < 400 else x
        )

    # Sub 7 / 9 to NaN
    col_7_9 = ['CPDEMO1B', 'CHOLCHK2', 'CHECKUP1', 'CHCCOPD2', 'ADDEPEV3', 'CHCKDNY2',
               'DIABETE4', 'HAVARTH4', 'ASTHMA3', 'CVDSTRK3', 'CVDCRHD4', 'CVDINFR4',
               'TOLDHI2', 'BPHIGH4', 'MEDCOST', 'PERSDOC2', 'HLTHPLN1', 'RENTHOM1',
               'VETERAN3', 'DEAF', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON',
               'SMOKE100', 'USENOW3', 'EXERANY2', 'FLUSHOT7', 'TETANUS1', 'PNEUVAC4',
               'HIVTST7', 'HIVRISK5', 'DRNKANY5']
    for col in col_7_9:
        data[col] = data[col].replace({7: np.nan, 9: np.nan})

    data['GENHLTH'] = data['GENHLTH'].replace({9: np.nan})
    data['EDUCA'] = data['EDUCA'].replace({9: np.nan})
    data['MARITAL'] = data['MARITAL'].replace({9: np.nan})
    data['EMPLOY1'] = data['EMPLOY1'].replace({9: np.nan})

    # Category variables
    factor_cols = col_7_9 + ['GENHLTH', 'EDUCA', 'MARITAL', 'EMPLOY1']
    for col in factor_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')

    # To integer
    int_cols = ['PHYSHLTH', 'MENTHLTH', 'CHILDREN', 'HTM4', 'WTKG3', 'ALCDAY5']
    for col in int_cols:
        if col in data.columns:
            data[col] = data[col].astype('Int64')

    # Fill NA
    num_cols = data.select_dtypes(include='number').columns
    imputer = SimpleImputer(strategy='median')
    data[num_cols] = imputer.fit_transform(data[num_cols])

    cat_cols = data.select_dtypes(include='category').columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])

    # Data balance check
    class_counts = data['Class'].value_counts()
    print("Class distribution:")
    print(class_counts)

    return data
