import numpy as np

def naive_bayes(training_file, test_file):
    training_file = "yeast_training.txt"
    test_file = "yeast_test.txt"
    training_data = np.loadtxt("yeast_training.txt")
    
    # To make training data dynamic 
    size_attr = len(training_data[0]) - 1

    labels = [] # Hosts the values of labels 
    for i in range(len(training_data)):
        if training_data[i][-1] in dict_defined:
            dict_defined[training_data[i][-1]].append(training_data[i][0:size_attr].flatten())
        else:
            labels.append(training_data[i][-1])
            dict_defined[training_data[i][-1]] = [training_data[i][0:size_attr].flatten()]
    
    labels.sort()
    
    # Making a dynamic array for labels storage and their respective statistics 
    class_attribute_description = [] 
    
    for i in range(len(labels)):
        complete_column = []
        new_array = []
        for k in range(size_attr):
            for j in range(len(dict_defined[labels[i]])):
                complete_column.append(dict_defined[labels[i]][j][k])
            # TODO : Check if the standard deviation matches ;; see question requirements 

            standard_deviation = np.std(complete_column,ddof=1)

            if standard_deviation < 0.01 : 
                print(standard_deviation)
                standard_deviation = 0.01 # Given on question 

            mean_calculated = np.mean(complete_column)
            new_array.append([mean_calculated,standard_deviation]) # This is the mean and std calculated with the help of numpy of the entire attribute array 
            complete_column = [] # This is to revert back the previous changes
        class_attribute_description.append(new_array)

    copy_arr = {}
    for i in range(len(training_data)):
        if training_data[i][-1] in copy_arr:
            copy_arr[training_data[i][-1]].append(training_data[i][0:size_attr].flatten())
        else:
            copy_arr[training_data[i][-1]] = [training_data[i][0:size_attr].flatten()]

    
    for i in range(len(labels)):
        for j in range(len(dict_defined[labels[i]])):
            for k in range(len(dict_defined[labels[i]][j])):
                print("The index of label is "+str(i)+" the index inside label's value is "+str(j)+" the value of k " + str(k) + " loops inside")
                print("The value  of each element is "+str(copy_arr[labels[i]][j][k])+" the mean is "+str(class_attribute_description[i][k][0])+" the value of std is " + str(class_attribute_description[i][k][1]))
                exp_val = np.exp(- ((np.power((copy_arr[labels[i]][j][k]-class_attribute_description[i][k][0]),2)))*(1/((2*np.power(class_attribute_description[i][k][1],2)))))
                copy_arr[labels[i]][j][k] = ((1/((class_attribute_description[i][k][1])*(np.sqrt(2*np.pi))))* ())
    
    # Now calculating individual probability density 
    individual_prob = [0] * len(labels)
    for i in range(len(individual_prob)):
        total = 0
        for j in range(len(dict_defined[labels[i]])):
            row_sum = 1 
            for k in range(len(dict_defined[labels[i]][j])):
                row_sum *= copy_arr[labels[i]][j][k]
            total += row_sum
        individual_prob[i] = total

    # This individual_prob contains probability density value for each function 

    # Calculating p(x)
    probability_individual = [0] * len(labels)
    total_elements = 0

    for i in range(len(labels)):
        total_elements += len(dict_defined[labels[i]])

    for i in range(len(probability_individual)):
        probability_individual[i] = (len(dict_defined[labels[i]])/total_elements)

    for i in range(len(labels)):
        individual_prob[i] = individual_prob[i] * probability_individual[i]


    # So far we have :
    # probability of Ck, probability of P(x|ck) - can be calculated using the formula above , p(x)
    
    # Training part is complete 
    # Printing the data values 
    # for i in range(len(labels)):
    #     for j in range(size_attr):
    #         print("Class "+ str(labels[i].astype(int)) + ", attribute " + str(j+1)+", mean = "+str(np.around(class_attribute_description[i][j][0],2))+",std = "+str(np.around(class_attribute_description[i][j][1],2)))

    # Opening the yeast_test.txt 
    test_data = np.loadtxt(test_file)
    accuracy = 0
    for i in range(len(test_data)):
        associated_array = test_data[i]
        real_class = associated_array[-1].astype(int)
        test_values = associated_array[:-1]
        total_gausians = 0
        for j in range(len(test_values)):
            highest = 0
            index = 0
            for k in range(len(labels)):
                value_of_gausian = ((1/((class_attribute_description[k][j][1])*(np.sqrt(2*np.pi))))* (np.exp(
                                                - ((np.power((test_values[j]-class_attribute_description[k][j][0]),2)))*(1/((2*np.power(class_attribute_description[k][j][1],2))))
                                            )))
                if highest == 0:
                    highest = value_of_gausian
                    index = j
                else:
                    if value_of_gausian > highest:
                        highest = value_of_gausian
                        indexa = j
            
            total_gausians += highest
        # Maximizing by each class 
        highest = 0
        index = 0
        for n in range(len(labels)):
            new_val = total_gausians * probability_individual[n] / individual_prob[n]
            if highest == 0:
                    highest = new_val
                    index = n
            else:
                if new_val > highest:
                    highest = new_val
                    index = n
                # Do nothing
        accuracy_new = 0.00
        if labels[index] == real_class:
            accuracy_new = 1.00
            accuracy += 1 
        # print("ID= "+str(i+1)+", predicted= "+ str(labels[index].astype(int)) + ",probability = "+ str(np.around(highest,4))+", true="+str(real_class)+", accuracy="+str(accuracy_new))
    # print("Classification accuracy="+str(np.around(accuracy/len(test_data),4)))

naive_bayes("yeast_training.txt","yeast_test.txt")