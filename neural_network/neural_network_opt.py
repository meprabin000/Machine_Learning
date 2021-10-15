# Name : Prabin Lamichhane
# ID: 1001733599

import numpy as np
import sys

def one_hot_encode(y, labels):
    return np.eye(len(labels))[np.where(labels == y)][0]

def load_data( path ):
    def normalize( x ):
        return x / np.max(x)
    data = np.loadtxt(path, dtype=str)
    x = np.asarray(data[:,0:-1], dtype = np.float64)
    y = data[:,-1]
    return normalize(x), y

def initialize_weights(input_units, output_units, layers, units_per_layer):
    weights = []
    if layers > 2:
        for l in range(layers - 1):
            if l == 0:
                weights.append((np.random.rand(input_units+1,units_per_layer)-0.50)*0.05)
            elif l == layers - 2:
                weights.append((np.random.rand(units_per_layer+1,output_units)-0.50)*0.05)
            else:
                weights.append((np.random.rand(units_per_layer+1,units_per_layer)-0.50)*0.05)
    else:
        weights.append((np.random.rand(input_units+1,output_units)-0.50)*0.05)
    return weights

def initialize_ones(input_units, output_units, layers, units_per_layer):
    var = []
    for l in range(layers):
        if l == 0:
            var.append(np.ones((input_units + 1,1)))
        elif l == layers - 1:
            var.append(np.ones((output_units + 1,1)))
        else:
            var.append(np.ones((units_per_layer + 1,1)))
    return var

def initialize_num_units_per_layer (input_units, output_units, layers, units_per_layer):
    J = []
    J.append(input_units)
    for l in range(1, layers - 1):
        J.append(units_per_layer)
    J.append(output_units)
    return J

def sigmoid_activation( a ):
    return 1/(1 + np.exp(-a))

def learning_rate(r):
    return 0.98**r

def activate( activation_fn, a_s, z_s, l ):
    for i,a in enumerate(a_s[l]):
        z_s[l][i+1][0] = activation_fn(a[0])
    return z_s

def Err( a_s, z_s, t_s, N, layers, K, D, x_train, weights ):
    err = 0
    a_s = a_s.copy()
    z_s = z_s.copy()
    for train_pt_i in range(N):
        for d in range(1,D+1):
            z_s[0][d] = x_train[train_pt_i][d-1]
        for l in range(1, layers):
            a_s[l] = weights[l-1].T @ z_s[l-1]
            z_s = activate(sigmoid_activation, a_s, z_s, l)
        err += sum([(t_s[train_pt_i][c] - z_s[layers-1][c+1][0])**2 for c in range(K)])
    return err

def predict(weights, x_train, layers):
    result = []
    N, D = x_train.shape
    result.append(np.insert(x_train, 0, np.ones((N,)), axis = 1))
    for l in range(layers-1):
        result.append(sigmoid_activation(result[l]@weights[l]))
        if l != layers - 2:
            result[l+1] = np.insert(result[l+1], 0, np.ones((N,)), axis = 1)
    return np.amax(result[layers-1], axis = 1), np.argmax(result[layers - 1], axis = 1)



def calc_accuracy(prediction, y_train):
    # filt = lambda x: x == 0.0
    return np.sum(prediction == y_train)/y_train.shape[0]

def print_test(predict_acc, prediction, y_test):
    for i in range(predict_acc.shape[0]):
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (i+1, prediction[i], y_test[i], predict_acc[i]))

def neural_network( training_file, test_file, layers, units_per_layer, rounds ):
    x_train, y_train = load_data(training_file)
    x_test, y_test = load_data(test_file)
    y_labels = np.unique(y_train)

    N, D = x_train.shape
    #initialization
    z_s = initialize_ones(D, len(y_labels), layers, units_per_layer) # doesn't matter what we initilize to
    a_s = initialize_ones(D, len(y_labels), layers, units_per_layer) # as these values will be updated with the first run
    gradient_loss = initialize_ones(D, len(y_labels), layers, units_per_layer)
    t_s = [one_hot_encode(y, y_labels) for y in y_train]
    weights = initialize_weights(D, len(y_labels), layers, units_per_layer)
    J = initialize_num_units_per_layer(D, len(y_labels), layers, units_per_layer)
    last_error = error = 0
    stop_condition = False
    round_i = 0

    # backpropagation summary
    while not stop_condition:
        # compute error when weights are random
        last_error = 0.5 * Err(a_s, z_s, t_s, N, layers, len(y_labels), D, x_train, weights )
        for train_pt_i in range(N):
            for d in range(1,D+1):
                z_s[0][d] = x_train[train_pt_i][d-1]
            for l in range(1, layers):
                a_s[l] = weights[l-1].T @ z_s[l-1]
                z_s = activate(sigmoid_activation, a_s, z_s, l) #updates z_s[l]
            for l in reversed(range(1, layers)):
                    for p_ind in range(J[l]): #p_ind is the perceptron index
                        if l == layers-1:
                            gradient_loss[l][p_ind+1] = (z_s[l][p_ind+1] - t_s[train_pt_i][p_ind] ) * z_s[l][p_ind+1] * (1 - z_s[l][p_ind+1])
                        else:
                            gradient_loss[l][p_ind+1] = sum([gradient_loss[l+1][k+1][0] * weights[l][p_ind+1][k] for k in range(J[l+1])]) * z_s[l][p_ind+1] * (1 - z_s[l][p_ind+1])
            for l in range(1, layers):
                for i in range(J[l]):
                    weights[l-1][0][i] = weights[l-1][0][i] - learning_rate(round_i) * gradient_loss[l][i+1]
                    for j in range(J[l-1]):
                        weights[l-1][j+1][i] = weights[l-1][j+1][i] - learning_rate(round_i) * gradient_loss[l][i+1] * z_s[l - 1][j+1]
        error = 0.5 * Err(a_s, z_s, t_s, N, layers, len(y_labels), D, x_train, weights )
        # print("last_error ", last_error)
        # print("error ", error)
        if (abs(last_error - error) <= 0.0001 ):
            stop_condition = True
        round_i += 1

    prediction_acc, prediction = predict(weights, x_test, layers)
    labeled_prediction = [y_labels[p] for p in prediction]
    accuracy = calc_accuracy(labeled_prediction, y_test)
    print_test(prediction_acc, labeled_prediction, y_test )
    print("classification accuracy=%6.4f\n" % (accuracy))

    return accuracy

def main():
    if len(sys.argv) == 6:
        accuracy = neural_network(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        print("Not enough arguments")

if __name__ == "__main__":
    main()
