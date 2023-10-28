import math 
import numpy as np
import matplotlib.pyplot as plt

#Calculates the Euclidean Distance
def distance(x1, y1, x2, y2):
    return math.sqrt(((x2 - x1 )**2) + ((y2-y1)**2))

#convert distinct to transmission in J
def to_transmission(distance):
    elec=100*pow(10,-9)
    amp=100*pow(10,-12)
    k=3200
    return (2*elec*k) + (amp*k*pow(distance,2))


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)