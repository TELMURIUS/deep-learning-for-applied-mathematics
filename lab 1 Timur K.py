import numpy as np


def gauss_inverse(A):
    n = len(A)
    identity = np.zeros((n, n))
    for i in range(n):
        identity[i][i] = 1

    augmented = np.zeros((n, 2 * n))
    for i in range(n):
        for j in range(n):
            augmented[i][j] = A[i][j]
            augmented[i][j + n] = identity[i][j]

    for i in range(n):
        d = augmented[i][i]
        for j in range(2 * n):
            augmented[i][j] /= d

        for j in range(n):
            if i != j:
                factor = augmented[j][i]
                for k in range(2 * n):
                    augmented[j][k] -= augmented[i][k] * factor

    inverse_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inverse_matrix[i][j] = augmented[i][j + n]

    return inverse_matrix


def tikhonov_regularization(A, b, lam):
    n = len(A[0])
    regularized_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            regularized_matrix[i][j] = A[i][j] * A[i][j]
            if i == j:
                regularized_matrix[i][j] += lam

    inverse_regularized = gauss_inverse(regularized_matrix)

    A_transposed_onB = np.zeros(n)
    for i in range(n):
        for j in range(len(b)):
            A_transposed_onB[i] += A[j][i] * b[j]

    x = np.zeros(n)
    for i in range(n):
        for j in range(n):
            x[i] += inverse_regularized[i][j] * A_transposed_onB[j]

    return x



A = [[1, 2], [2, 4.001]]
b = [3, 6.001]

sol_1 = gauss_inverse(A)
x_direct = [0, 0]
for i in range(len(A)):
    for j in range(len(b)):
        x_direct[i] += sol_1[i][j] * b[j]
print(f"решение без регуляризации:{x_direct}")

lambda_values = [0, 1, 10, 100]
for i in lambda_values:
    sol_2 = tikhonov_regularization(A, b, i)
    print(f"решение с регуляризацией (lambda={i}):{sol_2}")

#при увеличений значений лямбда решение становится более точным. однако результат с лямдбой равой 100
#мы можем заметить что в задачах регляризаций нужно подбирать значение лямда для более точного ответа