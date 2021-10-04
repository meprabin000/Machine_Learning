# Prabin Lamichhane
# 1001733599

import numpy as np
import sys

def load_data( path ):
    data = np.loadtxt(path)
    x = data[:,0:-1]
    y = data[:,-1]
    return x, y

def compute_basis_fn( data, degree ): # each row in 'data' is a training point. shape = N X D ( N data points, D dimensions)
    updated_data = []
    for data_point in data:
        updated_datapoint = np.array([])
        for feature in data_point:
            updated_datapoint = np.append(updated_datapoint, [feature**power for power in range(1, degree+1)] )
        updated_data.append(updated_datapoint)
    updated_data = np.array(updated_data)
    return np.hstack((np.ones((updated_data.shape[0],1)), updated_data))

def test( wt, x_test, degree ):
    phi_x_test = compute_basis_fn( x_test, degree )
    return phi_x_test@wt

def print_training( wts ):
    for i, wt in enumerate( wts ):
        fmt = "w%d=%.4f" % (i, wt)
        print(fmt)

def print_test( actual, prediction, sqr_err ):
    for i in range( prediction.shape[0] ):
        fmt = 'ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f' % (i+1, prediction[i], actual[i], sqr_err[i])
        print(fmt)

def linear_regression( train_filename, test_filename, degree, regularizer ):
    x_train, y_train = load_data( train_filename )
    x_test, y_test = load_data( test_filename )
    phi_x = compute_basis_fn( x_train, degree )
    dimen = phi_x.shape[1]
    wt = np.linalg.pinv(regularizer * np.identity(dimen)+ phi_x.T@phi_x)@(phi_x.T@y_train)
    test_result = test( wt, x_test, degree )
    sqr_err = (test_result - y_test) ** 2
    print_training( wt )
    print_test( y_test, test_result, sqr_err )

def main():
    if len(sys.argv) == 5:
        linear_regression(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        print("Please provide 2 file paths through command line: training_filepath, and testing_filepath")

if __name__ == "__main__":
    main()
