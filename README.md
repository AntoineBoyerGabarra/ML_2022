_ML_2022_

## Machine Learning Course projects
Class Project 1

## _Introduction_
The project aims at classifying decay signature of collision events to be able to predict if they are the result of a Higgs Boson signal or only due to background noise. _à changer_. The program is coded in python and run using Jupyter.
To make predictions, mutliples methods are implemented.
Before training our data, the set was visualized, analyzed and cleaned. 

## _README.md_
This file explains the structure of the repository.

## _implementations.py/ipynb_
contains the functions that had to be implemented for the training.

The functions are the following :

>[mean_squared_error_gd] : uses the gradient descent method in order to minimize the mean square error. 

>[mean_squared_error_sgd] : uses the stochastic gradient descent method in order to minimize the mean square error. It allows to decrease the complexity of the function using a subset. 

>[least_squares] : uses the least square method

>[ridge_regression] : uses mean squared gradient descent with a regularizer to avoid overfit.

>[logistic_regression] : uses the gradient descent method in order to minimize the logistic function.

>[reg_logistic_regression] : uses logistic gradient descent with a regularizer to avoid overfit.

>[logistic_regression_newton_method] : logisitc function minimized by quadratic approximation.


## _run.py/ipynb_
regroups the process of data visualization, cleaning and the training of the methods.

implements and runs the functions of _implementations.py/ipynb_  



