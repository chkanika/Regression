from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


xs= np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys= np.array([5, 6, 7, 7, 9], dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys): 
    m =( ((mean(xs) * mean(ys)) - mean(xs*ys))  / 
       ((mean(xs)**2) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coeficient_of_determination(ys_orig, ys_line):
    y_mean_line= [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean  = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

m,b = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs] # for x in xs: #     regression_line.append((m*x)+b)

predict_x = 8 
predict_y = (m*predict_x)+ b 

r_squared = coeficient_of_determination(ys, regression_line)
print(r_squared )

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, color= "red")
plt.plot(xs, regression_line)
plt.show()






