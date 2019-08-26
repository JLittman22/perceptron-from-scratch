import pandas as pd
import numpy as np

def run_algorithm(df, learning_rate):
    df_copy = df.copy()

    df_copy.insert(0, 'intercept', 1)

    df_copy = transform_data(df_copy)

    weights = compute_weights(df_copy, learning_rate)

    print("\nClassifier Weights: {}".format(weights))
    bias_term = weights[0]

    normalized_weights = []

    for weight in weights[1:]:
        normalized_weights.append(weight / - bias_term)

    print("\nNormalized Weights: {}".format(normalized_weights))


def transform_data(df):
    for index, row in df.iterrows():
        if row['Y'] == 1:
            continue
        df.loc[index] = -df.loc[index]

    return df


def normalize_data(data):
    for column in data.iloc[:, :data.shape[1] - 1].columns:
        min = np.amin(data[column])
        max = np.amax(data[column])
        for i in range(len(data[column])):
            value = (data[column][i] - min) / float(max)
            data.at[i, column] = value

    return data


def compute_weights(df, learning_rate):
    np.random.seed(99)

    X = df.iloc[:, :df.shape[1] - 1]
    Y = df.iloc[:, -1]

    weights = []

    for w in range(X.shape[1]):
        weights.append(np.random.normal())

    weights = np.asarray(weights)
    iter = 0
    while True:
        iter+= 1
        error_matrix = X[X.dot(weights) < 0]

        errors = [np.sum(error_matrix.iloc[:,0]), np.sum(error_matrix.iloc[:,1]), np.sum(error_matrix.iloc[:,2]), np.sum(error_matrix.iloc[:,3]), np.sum(error_matrix.iloc[:,4])]

        errors = np.asarray(errors)

        print("Iteration: " + str(iter) + ', total mistakes: ' + str(error_matrix.shape[0]))

        weights = weights + learning_rate * errors.T

        if error_matrix.shape[0] == 0:
            break

    return weights


df = pd.read_csv('perceptronData.txt', header=None, sep='\t')
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y']

learning_rate = .01

run_algorithm(df, learning_rate)