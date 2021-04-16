import numpy as np

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciąg wejściowy Nx1
    :param y: ciąg wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
    uzyskanymi z wielowamiu o parametrach w dla wejść x
    """
    predictedY = polynomial(x, w)
    size = np.shape(x)[0]
    error = 0.0
    index = 0
    while index < size:
        error += (y[index] - predictedY[index])**2
        index += 1
    error /= size
    arr = np.array(error)[0]
    return arr


def design_matrix(x_train, M):
    """
    :param x_train: ciąg treningowy Nx1
    :param M: stopień wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """
    sizeX = np.shape(x_train)[0]
    designMatrix = np.zeros((sizeX, M+1))
    rowIndex = 0
    while rowIndex < sizeX:
        cellIndex = 0
        while cellIndex < M+1:
            designMatrix[rowIndex][cellIndex] = x_train[rowIndex]**cellIndex
            cellIndex += 1
        rowIndex += 1
    return designMatrix


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego 
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """
    designedMatrix = design_matrix(x_train, M)
    transposed = np.transpose(designedMatrix)
    multiplied = np.matmul(transposed, designedMatrix)
    inverse = np.linalg.inv(multiplied)
    w = np.matmul(np.matmul(inverse, transposed), y_train)
    return (w, mean_squared_error(x_train, y_train, w))


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd 
    średniokwadratowy dopasowania
    """
    designedMatrix = design_matrix(x_train, M)
    transposed = np.transpose(designedMatrix)
    multiplied = np.matmul(transposed, designedMatrix)
    matrixSize = np.shape(multiplied)[0]
    identity = np.eye(matrixSize) * regularization_lambda
    summed = np.add(multiplied, identity)
    inverse = np.linalg.inv(summed)
    w = np.matmul(np.matmul(inverse, transposed), y_train)
    return (w, mean_squared_error(x_train, y_train, w))
    pass


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na 
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na 
    ciągach treningowym i walidacyjnym
    """
    minimumValidatedError = float("inf")
    minimumTrainedError = None
    minimum_w = None
    for M in M_values:
        (trained_w, trainedError) = least_squares(x_train, y_train, M)
        currError = mean_squared_error(x_val, y_val, trained_w)
        if currError < minimumValidatedError:
            minimumValidatedError = currError
            minimumTrainedError = trainedError
            minimum_w = trained_w
    return (minimum_w, minimumTrainedError, minimumValidatedError)


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    minimumValidatedError = float("inf")
    minimumTrainedError = None
    minimum_w = None
    bestLambda = None
    for lambdaL2 in lambda_values:
        (trained_w, trainedError) = regularized_least_squares(
            x_train, y_train, M, lambdaL2)
        currError = mean_squared_error(x_val, y_val, trained_w)
        if currError < minimumValidatedError:
            minimumValidatedError = currError
            minimumTrainedError = trainedError
            minimum_w = trained_w
            bestLambda = lambdaL2
    return (minimum_w, minimumTrainedError, minimumValidatedError, bestLambda)
    pass
