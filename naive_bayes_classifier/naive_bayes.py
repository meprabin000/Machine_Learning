 # Name: Prabin Lamichhane
# ID: 1001733599

import numpy as np
import sys

def load_data( path ):
    data = np.loadtxt(path)
    x = data[:,0:-1]
    y = data[:,-1]
    return x, y

def class_and_prob( class_data ):
    n = class_data.shape[0]
    unique = np.unique(class_data, return_counts=True)
    return unique[0], unique[1] / n

def calc_mean_std( x_data, y_data ):
    n, d = x_data.shape    
    labels = np.unique(y_data)
    mean, std = np.zeros((d,labels.shape[0])), np.zeros((d,labels.shape[0]))
    for dim in range(d):
        for i, label in enumerate(labels):
            x = x_data[y_data == label]
            mean[:, i], std[:, i] = np.mean(x, axis = 0), np.std(x, axis = 0)
    std[std < 0.01] = 0.01
    return mean, std

def gaussian(x, mean, std):
    i = -( x - mean)**2 / (2 * std**2)
    expo = np.exp(i) if np.exp(i) != 0 else 0.000000001
    return 1/(std*np.sqrt(2*np.pi)) * expo

def classification_accuracy(guess, actual, labels):
    accuracy = 0.0
    for i in range(guess.shape[0]):
        max_prob = np.max(guess[i])
        guessed_ind = np.argwhere(guess[i] == max_prob)
        guessed_labels = np.array([labels[j] for j in guessed_ind])
        curr_accuracy = 0
        if guessed_labels.shape[0] == 1:
            curr_accuracy = 1 if guessed_labels == actual[i] else 0
        elif np.sum(np.isin(guessed_labels,guess[i])) == 1:
            curr_accuracy = 1 / guessed_labels.shape[0]  
        accuracy += curr_accuracy
        format_out = 'ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f' % (i+1, np.random.choice(guessed_labels.flatten()), max_prob, actual[i], curr_accuracy)
        print(format_out)
    accuracy = accuracy / guess.shape[0]
    print("classification accuracy=%6.4f" % (accuracy))
    return accuracy
        

def print_training(mean, std, labels):
    mean = mean.T
    std = std.T
    label_n, attr_n = mean.shape
    for label_ind in range(label_n):
        for attr_ind in range(attr_n):
            format_out = 'Class %d, attribute %d, mean %.2f, std %.2f' %(labels[label_ind], attr_ind+1, mean[label_ind][attr_ind], std[label_ind][attr_ind])
            print(format_out)
            
def naive_bayes( training_filepath, testing_filepath):
    #training
    x_train, y_train = load_data(training_filepath)
    x_test, y_test = load_data(testing_filepath)
    n_train, d = x_train.shape
    labels, label_prob = class_and_prob(y_train)
    mean, std = calc_mean_std(x_train, y_train)
    
    # testing
    n_test = x_test.shape[0]
    P_XgC = np.ones((n_test, labels.shape[0]))
    P_CgX = np.ones((n_test, labels.shape[0]))
    for point in range(n_test):
        x = x_test[point]
        for i, label in enumerate(labels):        
            for j in range(d):
                P_XgC[point][i] *= gaussian(x[j], mean[j][i], std[j][i])
    
    P_X = np.dot(P_XgC,label_prob)
    for point in range(n_test):
        for c_ind in range(labels.shape[0]):
            # Bayes rule
            P_CgX[point][c_ind] = P_XgC[point][c_ind]*label_prob[c_ind] / P_X[point]
     
    print_training(mean, std, labels)
    classification_accuracy(P_CgX, y_test, labels)

def main():
    if len(sys.argv) == 3:
        naive_bayes(sys.argv[1], sys.argv[2]) 
    else:
        print("Please provide 2 file paths through command line: training_filepath, and testing_filepath")

if __name__ == "__main__":
    main()