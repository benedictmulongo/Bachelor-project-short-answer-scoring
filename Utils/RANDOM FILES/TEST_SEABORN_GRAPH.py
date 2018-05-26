import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

def plot_all(accuracy,cross_val, First_accuracy,Cross_accuracy  ):
    
    # data to plot
    n_groups = 5
    
    
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8
    
    rects1 = plt.bar(index, accuracy, bar_width,alpha=opacity,color='b',label='Test')
    
    rects2 = plt.bar(index + bar_width, cross_val, bar_width,alpha=opacity,color='r',label='Cross validation')
    
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('0.1 L 0.6 U 0.2 Test 0.1 Val')
    plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # plt.figure()
    
    n_groups = 2
    
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8
    
    rects1 = plt.bar(index, First_accuracy, bar_width,alpha=opacity,color='b',label='Test')
    
    rects2 = plt.bar(index + bar_width, Cross_accuracy, bar_width,alpha=opacity,color='r',label='Cross validation')
    
    # plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('0.1 L 0.6 U vs 0.7 train data')
    plt.xticks(index + bar_width, ('0.1 train data', '0.7 train data'))
    plt.legend()
    plt.show()


plot_all(accuracy,cross_val, First_accuracy,Cross_accuracy  )

    # 0.2 L 0.2 U 0.5 Test 0.1 Val
    # partition1 = [0.2, 0.25, 0.16]
# First accuracy =  [0.5668, 0.5577 ] 
# Cross accuracy =  [0.5117,0.5702 ]

# accuracy = [0.5161, 0.5244 ,0.5089,0.5006 , 0.5018 ]
# cross_val = [0.6605 ,0.7054 ,0.7046, 0.7015 , 0.7015 ]


    # 0.1 L 0.2 U 0.6 Test 0.1 Val
    # partition2 = [0.1, 0.22, 0.143]
# First_accuracy =  [0.5154,0.5512
# Cross_accuracy =  [0.5129, 0.543]

# accuracy = [0.5154, 0.5144 ,0.4975,0.5015, 0.4965 ]
# cross_val = [0.6778 ,0.7324 ,0.7338, 0.7294, 0.7189 ]



    # 0.1 L 0.6 U 0.2 Test 0.1 Val
    # partition3 = [0.1, 0.22, 0.143]

# First_accuracy =  [0.5482,0.5813]
# Cross_accuracy =  [0.5129, 0.5672]

# accuracy = [0.5481, 0.5241 ,0.5241,0.5090 , 0.5241 ]
# cross_val = [0.7885 ,0.8402 ,0.8322 , 0.8191 , 0.7886 ]


    # 0.3 L 0.4 U 0.2 Test 0.1 Val
#     # part = [0.3, 0.57, 0.33]
# First_accuracy =  [0.5668,0.5816 ]
# Cross_accuracy = [ 0.5117, 0.55774]
# # All_accuracy =  [0.5816]
# # All_Cross_accuracy = [0.55774] 
# 
# accuracy = [0.5668, 0.5549 ,0.5697,0.5757 , 0.5727  ]
# cross_val = [0.6518 ,0.6911 ,0.6925 , 0.6756 , 0.6745 ]