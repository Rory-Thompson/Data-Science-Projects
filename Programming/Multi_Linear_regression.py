"""
In this assignment we create a Python module
to perform some basic data science tasks. While the
instructions contain some mathematics, the main focus is on 
implementing the corresponding algorithms and finding 
a good decomposition into subproblems and functions 
that solve these subproblems. 

To help you to visually check and understand your
implementation, a module for plotting data and linear
prediction functions is provided.

The main idea of linear regression is to use data to
infer a prediction function that 'explains' a target variable 
of interest through linear effects of one 
or more explanatory variables. 

Part I - Univariate Regression

Task A: Optimal Slope

-> example: price of an apartment

Let's start out simple by writing a function that finds
an "optimal" slope (a) of a linear prediction function 
y = ax, i.e., a line through the origin. A central concept
to solve this problem is the residual vector defined as

(y[1]-a*x[1], ..., y[m]-a*x[m]),

i.e., the m-component vector that contains for each data point
the difference of the target variable and the corresponding
predicted value.

With some math (that is outside the scope of this unit) we can show
that for the slope that minimises the sum of squared the residual

x[1]*(y[1]-a*x[1]) + ... + x[m]*(y[m]-a*x[m]) = 0

Equivalently, this means that

a = (x[1]*y[1]+ ... + x[m]*y[m])/(x[1]*x[1]+ ... + x[m]*x[m])

Write a function slope(x, y) that, given as input
two lists of numbers (x and y) of equal length, computes
as output the lest squares slope (a).

Task B: Optimal Slope and Intercept

To get a better fit, we have to consider the intercept b as well, 
i.e., consider the model f(x) = ax +b. 
To find the slope of that new linear model, we \centre the explanatory variable 
by subtracting the mean from each data point. 
The correct slope of the linear regression f(x)=ax + b is the same 
slope as the linear model without intercept, f(x)=ax, calculated on the 
centred explanatory variables instead of the original ones. 
If we have calculated the correct slope a, we can calculate the intercept as
b = mean(y) - a*mean(x).

Write a function line(x,y) that, given as input
two lists of numbers (x and y) of equal length, computes
as output the lest squares slope a and intercept b and
returns them as a tuple a,b.


Task C: Choosing the Best Single Predictor

We are now able to determine a regression model that represents 
the linear relationship between a target variable and a single explanatory variable.
However, in usual settings like the one given in the introduction, 
we observe not one but many explanatory variables (e.g., in the example `GDP', `Schooling', etc.). 
As an abstract description of such a setting we consider n variables 
such that for each j with 0 < j < n we have measured m observations 

$x[1][j], ... , x[m][j]$. 

These conceptually correspond to the columns of a given data table. 
The individual rows of our data table then become n-dimensional 
data points represented not a single number but a vector.

A general, i.e., multi-dimensional, linear predictor is then given by an n-dimensional 
weight vector a and an intercept b that together describe the target variable as

y = dot(a, x) + b 

i.e., we generalise y = ax + b by turning the slope a into an n-component linear weight vector
and replace simple multiplication by the dot product (the intercept b is still a single number).
Part 2 of the assignment will be about finding such general linear predictors. 
In this task, however, we will start out simply by finding the best univariate predictor 
and then represent it using a multivariate weight-vector $a$. %smooth out with the text that follows.

Thus, we need to answer two questions: (i) how do we find the best univariate predictor, 
and (ii) how to we represent it as a multivariate weight-vector. 

Let us start with finding the best univariate predictor. For that, we test all possible
predictors and use the one with the lowest sum of squared residuals.
Assume we have found the slope a^j and intercept b^j of the best univariate predictor---and assume it 
uses the explanatory variable x^j---then we want to represent this as a multivariate 
slope a and intercept b. That is, we need to find a multivariate slop a such that dot(a, x) + b 
is equivalent to a^jx^j + b^j. Hint: The intercept remains the same, i.e., $b = b^j$.

Task D: Regression Analysis

You have now developed the tools to carry out a regression analysis. 
In this task, you will perform a regression analysis on the life-expectancy 
dataset an excerpt of which was used as an example in the overview. 
The dataset provided in the file /data/life_expectancy.csv.


Part 2 - Multivariate Regression

In part 1 we have developed a method to find a univariate linear regression model 
(i.e., one that models the relationship between a single explanatory variable and the target variable), 
as well as a method that picks the best univariate regression model when multiple 
explanatory variables are available. In this part, we develop a multivariate regression method 
that models the joint linear relationship between all explanatory variables and the target variable. 


Task A: Greedy Residual Fitting

We start using a greedy approach to multivariate regression. Assume a dataset with m data points 
x[1], ... , x[m] 
where each data point x[i] has n explanatory variables x[i][1], ... , x[i][m], 
and corresponding target variables y[1], ... ,y[m]. The goal is to find the slopes for 
all explanatory variables that help predicting the target variable. The strategy we 
use greedily picks the best predictor and adds its slope to the list of used predictors. 
When all slopes are computed, it finds the best intercept. 
For that, recall that a greedy algorithm iteratively extends a partial solution by a 
small augmentation that optimises some selection criterion. In our setting, those augmentation 
options are the inclusion of a currently unused explanatory variable (i.e., one that currently 
still has a zero coefficient). As selection criterion, it makes sense to look at how much a 
previously unused explanatory variable can improve the data fit of the current predictor. 
For that, it should be useful to look at the current residual vector r,
because it specifies the part of the target variable that is still not well explained. 
Note that a the slope of a predictor that predicts this residual well is a good option for 
augmenting the current solution. Also, recall that an augmentation is used only if it 
improves the selection criterion. In this case, a reasonable selection criterion is 
again the sum of squared residuals.

What is left to do is compute the intercept for the multivariate predictor. 
This can be done  as


b = ((y[1]-dot(a, x[1])) + ... + (y[m]-dot(a, x[m]))) / m

The resulting multivariate predictor can then be written as 

y = dot(a,x) + b .



Task B: Optimal Least Squares Regression

Recall that the central idea for finding the slope of the optimal univariate regression line (with intercept) 
that the residual vector has to be orthogonal to the values of the centred explanatory variable. 
For multivariate regression we have many variables, and it is not surprising that for an optimal 
linear predictor dot(a, x) + b, it holds that the residual vector is orthogonal to each of the 
centred explanatory variables (otherwise we could change the predictor vector a bit to increase the fit). 
That is, instead of a single linear equation, we now end up with n equations, one for each data column.
For the weight vector a that satisfies these equations for all i=1, ... ,n, you can again simply find the 
matching intercept b as the mean residual when using just the weights a for fitting:

b = ((y[1] - dot(a, x[1])) + ... + (y[m] - dot(a, x[m])))/m .

In summary, we know that we can simply transform the problem of finding the least squares predictor to solving a system of linear equation, which we can solve by Gaussian Elimination as covered in the lecture. An illustration of such a least squares predictor is given in Figure~\ref{fig:ex3dPlotWithGreedyAndLSR}.
"""

from math import inf, sqrt
from copy import deepcopy




def slope(x, y):
    """
    Computes the slope of the least squares regression line
    (without intercept) for explaining y through x.

    For example:
    >>> slope([0, 1, 2], [0, 2, 4])
    2.0
    >>> slope([0, 2, 4], [0, 1, 2])
    0.5
    >>> slope([0, 1, 2], [1, 1, 2])
    1.0
    >>> slope([0, 1, 2], [1, 1.2, 2])
    1.04
    """
    z=0
    k=0
    res=0
    result=0
    for i in range(len(x)):##sumsx*y
        res+=x[k]*y[k]
        k+=1
    for i in range(len(x)):
        result+=x[z]*x[z]#sums x^2
        z+=1
    final_results=res/result#devides the result
    return final_results
"""
This is a relatively simple, task, first a for loop is run that calculates
the sum of the vector x*y
then another for loop that finds the sum of x*x, then the final result is the
res/ result or vector x*y/x*x
"""




def line(x, y):
    """
    Computes the least squares regression line (slope and intercept)
    for explaining y through x.

    For example:
    >>> a, b = line([0, 1, 2], [1, 1, 2])
    >>> round(a,1)
    0.5
    >>> round(b,2)
    0.83
    """
    def mean_x(x):
        k=0
        res=0
        for i in range(len(x)):
            res+=x[k]
            k+=1
        result = res/(len(x))
        return result
    mean = mean_x(x)
    def x_bar(x):##returns the vector of x_bars
        lst=[]
        z=0
        for i in range(len(x)):
            lst.append(x[z]-mean)
            z+=1
        return lst
    x_bars=x_bar(x)
    def a(x_bars, y):##returns the value a
        res=0
        k=0
        res2=0
        for i in range(len(x_bars)):
            res+=x_bars[k]*x_bars[k]##summing the xbar^2
            res2+=x_bars[k]*y[k]##summing xbar * y
            k+=1
        final_result = res2/res #deviding the above results
        return final_result
    a=a(x_bars, y)
    def value_b(x,y):#using previous results finds b
        res=0
        k=0
        for i in range(len(x)):
            res+=(y[k]-(a*x[k]))
            k+=1
        final_result = res/(len(y))
        return final_result
    b=value_b(x,y)
    return a, b
"""
this code was a little more complex, needing to find the final value a and b.
First a mean function was created to find the mean of x, using a simple for loop
adding all the values then deviding by the number of times the loop was run.
This value of mean of x is assigned to "mean", using the function "line" input x.
Next the x_bar function is assigned, making a list that will be a vector of each
value x - the mean assigned previously.
Next the function a will find the value for a, using two result variables summing
the squares of x bar and x bar*y, then deviding these two values to get a.
finally the function value_b finds the value b, using the assigned value a, multiplying
it by the vector x, and minusing it from the vector y, then summing these results
then deviding the sum by the number of y values in the function, hence this should be the
correct b value and a value, in all of these cases, a for loop seemed natural, as
it is known the number of times the loop will run based on the input.
Commonly using the append function keeps everything in order, as it will append
to a new list in the same order as the results on the list the info is recieved from.
"""


def best_single_predictor(data, y, current_weights=[0,0,0,0]):
    """
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> weights, b = best_single_predictor(data, y)
    >>> weights[0]
    0.0
    >>> round(weights[1],2)
    -0.29
    >>> round(b,2)
    2.14
    """
    def inverse(data): ##the inverse of the x data is easier to manipulate for me in this task
        lst2=[]#i found it easier to iterate through, when the columns and rows are swapped
        k=0#this may not be neccassary but for simplifying the problem in my head i found it easier
        z=0
        for i in range(len(data[0])):
            lst1=[]
            for j in range(len(data)):
                lst1.append(data[k][z])
                k+=1
            lst2.append(lst1)
            k=0
            z+=1
        return lst2
    inverse_x=inverse(data)
    def list_tuple(lst1):
        lst=[]
        k=0
        for i in range(len(data[0])):##the length is based on number number of x values
            lst.append(line(lst1[k], y))#reuse previous function 
            k+=1
        return lst
    tuple_list= list_tuple(inverse_x)
    ## this should return the values for a and b for each column in a list of tuples
    def thing(tuple_list, y, x): ## this is a list containing the vector r for each value x
        lst=[]
        z=0
        k=0
        lst1=[]
        for i in range(len(tuple_list)):
            lst=[]
            for i in range(len(y)):
                lst.append(y[k]-tuple_list[z][0]*x[z][k]-tuple_list[z][1])
                k+=1
            z+=1
            k=0
            lst1.append(lst)
        return lst1
    c=thing(tuple_list, y, inverse_x)## the list of lists containing vector r,maintaining relative order ie(x(0)=c[0]
    def r_value(c): ##finds the vector containing r^2, then summing it and appending it to a new list
        lst=[]
        k=0
        lst1=[]
        for i in range(len(c)):
            lst=[]
            for j in range(len(c[i])):
                lst.append(c[i][j]*c[i][j])
            lst1.append(lst)
        lst2=[]
        for i in range(len(lst1)):
            res=0
            for j in range(len(lst1[i])):##it is key in this section to maintain order hence append is always used,
                res+=lst1[i][j]##otherwise the index below wont match the true best model even if calcs are correct
            lst2.append(res)
        return lst2 
    r=r_value(c)## the list of the sums of r^2 for the different x values
    best_model=r.index(min(r))##returns the index of the best model
    lst5=[]
    for i in range(len(inverse_x)):
        lst5.append(0.0)
    lst5[best_model]=tuple_list[best_model][0]
    final_b=tuple_list[best_model][1] ## here we access the best model by finding the a and b
    return lst5, final_b

'''
One of the key problems that could be run into in this code, is variables being reused
and potentially having the wrong values, leading to errors, it is for this reason
i have chosen to include a large amount of nested functions, so variables i use in the
function will be looked for first in the function parameters, leading to a smaller
chance of bugs. While i can use a while loop for any of the situations i have used a for loop
as i am more confortable with the use of a for loop when the number of iterations is known
which appears to be always the case in this problem.
Another decision i made was to make an inverse function, it may not be neccassary
but I decided to do it early on, as i believe it is easier to work with when doing
later functions as i can just iterate over it as usual, with a double for loop, with
the inside representing the columns and the outside represnting the rows.
The last potential problem I could run into is the order of answers could be lost
as there are many lists that are created from previous results(for instance the
final index, labeled best model, must match the original tuple of values a and b),
and to make sure, they remain in realative order, when adding to a list, append was
used almost all the time to ensure the lists remain in relative order.
It was also important for this function to reuse the line function, to break this question down
and make it alot easier (problem decomposition)

'''
    
    
    
def best_single_predictor2(data, y, current_weights=[0,0,0,0]):
    """
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> weights, b = best_single_predictor(data, y)
    >>> weights[0]
    0.0
    >>> round(weights[1],2)
    -0.29
    >>> round(b,2)
    2.14
    """
    def inverse(data): ##the inverse of the x data is easier to manipulate for me in this task
        lst2=[]#i found it easier to iterate through, when the columns and rows are swapped
        k=0#this may not be neccassary but for simplifying the problem in my head i found it easier
        z=0
        for i in range(len(data[0])):
            lst1=[]
            for j in range(len(data)):
                lst1.append(data[k][z])
                k+=1
            lst2.append(lst1)
            k=0
            z+=1
        return lst2
    inverse_x=inverse(data)
    def list_tuple(lst1):
        lst=[]
        k=0
        for i in range(len(data[0])):##the length is based on number number of x values
            lst.append(line(lst1[k], y))#reuse previous function 
            k+=1
        return lst
    tuple_list= list_tuple(inverse_x)
    ## this should return the values for a and b for each column in a list of tuples
    def thing(tuple_list, y, x): ## this is a list containing the vector r for each value x
        lst=[]
        z=0
        k=0
        lst1=[]
        for i in range(len(tuple_list)):
            lst=[]
            for i in range(len(y)):
                lst.append(y[k]-tuple_list[z][0]*x[z][k]-tuple_list[z][1])
                k+=1
            z+=1
            k=0
            lst1.append(lst)
        return lst1
    c=thing(tuple_list, y, inverse_x)## the list of lists containing vector r,maintaining relative order ie(x(0)=c[0]
    def r_value(c): ##finds the vector containing r^2, then summing it and appending it to a new list
        lst=[]
        k=0
        lst1=[]
        for i in range(len(c)):
            lst=[]
            for j in range(len(c[i])):
                lst.append(c[i][j]*c[i][j])
            lst1.append(lst)
        lst2=[]
        for i in range(len(lst1)):
            res=0
            for j in range(len(lst1[i])):##it is key in this section to maintain order hence append is always used,
                res+=lst1[i][j]##otherwise the index below wont match the true best model even if calcs are correct
            lst2.append(res)
        return lst2
    def search(lst, value):##a function to find if a value is in a list
        ##note this is added in
        res=False
        for i in range(len(lst)):
            if lst[i]==value:
                res=True
        return res##returns a True or False
    def find_non_0_min(lst):
        ##note this is added in
        res=100##the r value cant be above 1 anyway
        res2=100##note to check if there is no none 0, this was added in, for a check further on
        for i in range(len(lst)):
            if lst[i]<res2 and lst[i] !=0:
                res=i
                res2=lst[i]
        return res
    
            
            
    def new_best_model_for_greedy(r, current_weights):
        ##note this is added in
        lst=[]
        lst1=len(r)*[0]
        lst2=deepcopy(lst1)
        for i in range(len(current_weights)):
            if current_weights[i]==0:
                lst.append(i)##returns the index of weights not used
        for i in range(len(r)):
            if search(lst, i)== True:##checks to see if the index is a non used weight
                lst2[i]=r[i]
        best_model=find_non_0_min(lst2)
            
         
            
        
        return best_model
    
       
    r=r_value(c)## the list of the sums of r^2 for the different x values
    best_model=r.index(min(r))##returns the index of the best model
    lst5=[]
    if new_best_model_for_greedy(r, current_weights)==100:##in the event that there is no better models, the function find non 0 min wont work
        best_model=current_weights##so it returns the original weights for that iteration
        return best_model, 5##and then returns the function, so it is properly returned, without changing the weights
        
        
    best_model=new_best_model_for_greedy(r, current_weights)
    for i in range(len(inverse_x)):
        lst5.append(0.0)
    lst5[best_model]=tuple_list[best_model][0]
    final_b=tuple_list[best_model][1] ## here we access the best model by finding the a and b
    return lst5, final_b
'''
The adapting of this function: (note: added in functions have a note under there definition)
this functions adaption was integral to the greedy predictor. Some nested functions were added, the search, new_best_model
and find non 0 min. The main problem for this function was using a list, that wont allow for it to use a weight already used.
Hence a weight paraemter is created, that includes the current weight. The new best model for greedy then, finds and creates a list
of all possible weights that are not used, then uses the search function to make sure the residuals list only contains non used variables.
Next the the find non 0 min is used. the list it works with, will have a 0, where a weight already used is, hence it must be non 0.
One key problem, is if there is no more weights not used that works, it must, be a given value, in this instance it is 100, note the code
might get bugs is there is over 100 variables, this number can be changed though based on variable size. if the result of find non 0 is 100,
a return is done to return the original weights, stopping the rest of the function where the weights is changed. It is very important
that this return statement is before the new best_model is calculated , or the function will continue to look for index 100 of the weights
which clearly wont be defined. Finally if there is a weight the value of best model will be a legit number as the find non 0 code should have
worked, the function returns only this new weight, it is added to the original weights in the greedy predictor code. The changing of this code
while hard to think through initially, becomes very hard when you need to track indexs, and its associated sum of square residuals, while
only looking at unused variables, it is for this reason the search and find non 0 min become very important in deconstructing this problem
into an easier to think through solution.

'''
                
                

def greedy_predictor(data, y):
    """
    This implements a greedy correlation pursuit algorithm.

    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> weights, intercept = greedy_predictor(data, y)
    >>> round(weights[0],2)
    0.21
    >>> round(weights[1],2)
    -0.29
    >>> round(intercept, 2)
    1.64
    
    >>> data = [[0, 0],
    ...         [1, 0],
    ...         [0, -1]]
    >>> y = [1, 0, 0]
    >>> weights, intercept = greedy_predictor(data, y)
    >>> round(weights[0],2)
    -0.5
    >>> round(weights[1],2)
    0.75
    >>> round(intercept, 2)
    0.75
    """
    weights, b_first = best_single_predictor(data, y)##can use the original predictor as only one original variable is required
    copy_data= deepcopy(data)
    copy_weights= deepcopy(weights)
    def get_new_single(data, y, weights):##finds the new best predictor based on the new residuals
        copy_weights_temp= deepcopy(weights)
        weights_temp, y_temp=best_single_predictor2(data, y, copy_weights)##uses the new nest_single_predictor
        for i in range(len(weights_temp)):
            if weights_temp[i] !=0:
                res=i
        copy_weights_temp[res]=weights_temp[res]
        return copy_weights_temp
    def residual_vector(data, y, weights, b):##input, a list of lists of data frame, y values, weights and intercept 
        lst2=[]
        for i in range(len(data)):##i
            lst=[]
            for j in range(len(data[i])):## iterates the find the value x*a of all weights for a row
                lst.append(data[i][j]*weights[j])
            a=sum(lst)##sums the temporary list
            lst2.append(a)##appends the sum for each row
        lst3=[]
        for i in range(len(lst2)):##uses the equation of residuals and applys it to all rows
            lst3.append(y[i]-lst2[i]-b)
        return lst3##returns the list of resdual for the respective rows of the data frame
    def get_b(data, weights, y):#gets the value b
        lst=[]
        lst1=[]
        for i in range(len(data)):#iterates through x values i
            lst=[]
            for j in range(len(data[i])):##iterates through all variables
                lst.append(weights[j]*data[i][j])# a temp list containing the multiplied weights and x values for row 1
            lst1.append(sum(lst))#sums the temporary list and appends it to a new list
        lst2=[]
        for i in range(len(y)):##this then finds y - the matching equation to find the residual
            lst2.append(y[i]-lst1[i])
        b=sum(lst2)/len(y)
        return b#returning b
    def get_sum_of_square(lst):
        res=0
        for i in range(len(lst)):
            res+=lst[i]*lst[i]
        return res
    
            
    b=b_first       
    for i in range(len(weights)-1):
        copy_weights_before= deepcopy(weights)##creates a copy of the current true weights
        b_before=get_b(data, copy_weights, y)#gets b using the get_b function
        new_y=residual_vector(data, y, copy_weights_before,b)#using the residual it finds the new explanatory variable
        sum_square_error_before = get_sum_of_square(y)##finds the sum of square error before, to compare with after (uses real y)
        new_weights=get_new_single(data, new_y, copy_weights_before)##finds the new weights using the get_newe_single that uses the new get best predictor
        b_new_temp=get_b(data, new_weights, new_y)##finds b for these new weights
        b_real_temp=get_b(data, new_weights, y)##finds the actual b value, for when the real y value is used
        new_residual = residual_vector(data, y, new_weights, b_real_temp)##calculates a temp residual to check
        temp_sum_square= get_sum_of_square(new_residual)##finds the sump of square of the temp residual
        if temp_sum_square < sum_square_error_before:##checks to see if the model is improved, then makes the actual weights the temporary weights
            weights = new_weights
        else:
            weights=copy_weights_before##if it didnt change it, the weights remain as the initial weight
        
    final_b=get_b(data, weights, y)##finally calculates b
    return weights, final_b##returns the weights and intercept

"""
challenges and my approach the problem:
Over all this problem was difficult to think through, with many different variables being updated and changed and calculated,
but problem decomposition and the different functions created help immensely in breaking it down.
the residual vector function, is hard to think through as it requires to work through all of the weights, and to make it not hard coded
to a paticular number of weights. To approach this problem, a nested for loop was implemented that finds the value of each
weight * the paticular x value, for all of the rows in the data set. the inside for loop clearly will work through all of
the columns, then the for exterior for loop iterates through all of the rows. The result is a list of length n, n
being the number of y values/xvalues, as the temporary list will be summed. The function then uses the definition of residual vector to get, y-weights*x-b,
then finally the function returns this list. The next function that is required is the function that finds b, first iterating through all of
the rows, then inside this all of the columns, then generates a list that is the multiple of the weights, * x values, it works very similar to
the residual vector function, except summing to find the average. the final function to consider is something that gets the sum of squares for a list.
This is required as, the function needs to check if the new weight improves the accuracy of the function. This is a simple task, only needing to
iterate through all of the elements in the list of residuals, then squaring it, then summing the final list. Finally the last function
I considered creating was a remake of the original best_single_predictor. The reason for this is so it can output a new weight that is
not already being used. To do this, (the function best_single_predictor2) was created, but it checks to make sure if the output is an
unused weight, and if there are no more weights it must just release the original weights. This function and its implementation will be
in a paragraph below best_single_predictor2. In the implementation of greedy predictor, after the creation of these functions  the task appears
to be an easier task. In the final implementation, first the original best single predictor is used to find the best predictor. next a for loop
is run, that iterates through the possible nwe weights, so weights -1, as one has already been selected. in the loop, b is calculated using get b,
and new_y (the residuals of the new model) is produced, then the sum of r^2 is calculated. next the new weights is calculated, it uses the
get_new_single function, that, then uses the best_single_predictor2, then finds the new weight, by finding when it is non 0, and adding the new
weight to the weights, to give new_weights. Next, some temp values are calculated, they are temporary, because the new model may not work
better. using some previously discussed functions, the new residual, b is calculated then the sum of square residuals is calculated, hence
the new sum square resduals is checked, if it is better, then the proper weights is updated to = the temporary, if it isnt better, then the
weights remains the initial copy. Finally once the loop is over, b is calculated and the real weights is returned.

Computational complexity:
function get_residuals and get_b both are similar, both dependent on the number of columns and rows, hence a O(n^2) as it is only a nested for loop,
and there are no more functions inside the second loops that require more computation based on n, so O(n^2) for these two functions. 
this function doesnt have the best time complexity, consider, within the for loop, many other functions are calculated, with most of them having
n^2, Due to them having two for loops, such as residuals, and so on that need to iterate through all of the values of the data set. The function
also runs best_single_predictor2, which most likely has a n^2, as there is nothing worse than a two nested loop in it. hence the total loop, must
iterate n*((N^2+N^2+N^2)), as inside the for loop, there are many functions that all have N^2 that need to run each time. this would become
O(n^3), as there can be a constant that isnt included in big oh notation. For this reason it is most likely O(n^3), not the best even though greedy
is generally easier than optimal, the optimal seems to have similar run time, maybe even faster, and with most likely better accuracy.
The implementation is relativly rough around the edges, but seems to work well, even though it is most likely big-oh(n^3), there are alot
more functions running for each loop in the main loop, hence making alot more time is required compared to the other functions in this regression file.

"""




def equation(i, data, y):
    """
    Finds the row representation of the i-th least squares condition,
    i.e., the equation representing the orthogonality of
    the residual on data column i:

    (x_i)'(y-Xb) = 0
    (x_i)'Xb = (x_i)'y

    x_(1,i)*[x_11, x_12,..., x_1n] + ... + x_(m,i)*[x_m1, x_m2,..., x_mn] = <x_i , y>

    For example:
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> coeffs, rhs = equation(0, data, y)
    >>> round(coeffs[0],2)
    4.67
    >>> round(coeffs[1],2)
    2.33
    >>> round(rhs,2)
    0.33
    >>> coeffs, rhs = equation(1, data, y)
    >>> round(coeffs[0],2)
    2.33
    >>> round(coeffs[1],2)
    4.67
    >>> round(rhs,2)
    -1.33
    """
    def get_xbar(data):##input: a list of x values
        lst=[]
        total=sum(data)
        mean = total/len(data)
        for i in range(len(data)):
            lst.append(data[i]-mean)
        return lst#output: a new list of xbar, each value - the mean of the original list
    def get_inverse(data):##finds the inverse of the x data
        lst=[]##input: a data set, list of lists in m*n
        for i in range(len(data[0])):
            lst1=[]
            for j in range(len(data)):
                lst1.append(data[j][i])
            lst.append(lst1)
        return lst#output a list of lists in n*m form
    inverse_data=get_inverse(data)
    def xbar_for_all_x(x_data):#input: the x data set in inverse form
        lst=[]
        for i in range(len(x_data)):
            lst.append(get_xbar(x_data[i]))
        return lst#output: a list of lists where the xbar is calculculated for each row (inverse column)
    x_bars=xbar_for_all_x(inverse_data)
    def dot_product(x1,x2):##input: two lists of the same length
        res=0
        for i in range(len(x1)):
            res+=x1[i]*x2[i]
        return res##output, the sum of the multiple of matching indexes of the lists, IE the dot product
    def left_coefficients(x_data,i):##input: the xbars, a list of lists
        lst=[]
        main_vec=x_data[i]
        for j in range(len(x_data)):
            lst.append(dot_product(main_vec,x_data[j]))
        return lst#output, each x bars dot product with a given xbar
    left_side = left_coefficients(x_bars, i)
    right_side = dot_product(x_bars[i],y)
    return left_side,right_side#the left and right coefficients of the required function
'''
challenges and my approach to the equation problem:
first, a function was produced, that inputs a list of numbers, then calculates the mean and minuses each value by the mean,
this will give xbar for a paticular column. The next function defined, iterating through the columns, then the rows
within this, appending each value, to get the inverse of the x data. The reason for this function, is due to the  xbar
function i created, inputs a list e.g: [1,2,3]. To find the xbar for all x columns, the getxbar is used, but requires the
data in a list form, not able to iterate through columns hence the inverse is calculated. The next function, when used with xbar
for all x, finds the xbars for all of the x expanatory variables, in order, by iterating through the inverse of data, and finding
x bar for all the lists inside the input and appending it to a new list. getting the xbar, can be hard to think through, but with function
decomposition, and using the inverse, the code looks alot nicer, and is much easier to think through instead of iterating
through and calculating xbar through columns of a list of lists. Next a dot product function was written, iterating through
the index, and multiplying the matching indexs, then finally summing them. This is another example of problem decomposition,
allowing me to use this function within the left coefficients function, that finds the dot product between different possibilities
of two xbars, ie the dot product of xbars, using a for loop, it will find an xbar of index i, and only xbar[i] . (all the  other
xbars), where xbar[i] is defined as main_vec, it then will return the list of all these dot products in the same order. It is
key within this function xbars maintain relative order with the original data, all of the for loops iterate through, 0-n, never
n-0, hence making it in the same order. Finally the right side of the equation quoted in the intstructions is calculated with a simple
dot product, to give the coefficients for a paticular vector of index i.

Computational complexity:
the first function to consider is the get_inverse, as it iterates through columns then the rows within this column, it will iterate
N*m times where N and m are the lengths of the columns and rows of the data respectively. this suggests it has O(n^2).
the next function is xbar_for all x, finding the xbars for all the columns (or rows of the inverse), it will also iterate N*m times as
it must go through each value and change it to x bar, with two for loops, hence O(n^2). The dot product function is just a single loop
iterating n times, n being the length of the xbars. so only O(n). The left coefficients iterates the same amount, N*dot product
hence N*n wich is O(n^2), suggesting that the total time complexity of this  function should be near O(n^2), a nice polynomial time complexity.
'''

def solve_by_back_sub(u,b):##input matrix in upper triangular and column matrix
    '''
    This function, once x matrix is in upper triangular form from the original matrix, it back solves than appends
    the coefficients in their correct spots. one important challenge is to append the coefficients in the correct spot,
    to do this the function creates a list of the length of the coefficients, and solves backwards, iterating from
    n-1 to 0 inclusive, this works as it will first solve the last coefficient, (last row), then make it == to
    the last spot of the new list created, a simple and effective way to counter this problem. it will solve algebraicly
    as when a matrix multiplication is expanded out, a4*x4 = y4, so a4=y4/x4 will give the solution, it will then find
    all of the coefficients values and append it correctly in order.

    '''
    n=len(b)
    x=n*[0]##produces a list of length n containing 0's
    for i in range(n-1, -1, -1):     #iterates through backwards from n-1, to 0 inclusive
        s=0
        for j in range(n-1, -1, -1):
            s+=u[i][j]*x[j]##finds the sum for a given row
        x[i] = (b[i]-s)/u[i][i]##finds the solution
    return x##output: a list of coefficients of length n
'''
cite:
lecture 18, week 9, Michael Kamp, 31st minute fit1045
'''
def pivot_index(a, j):##input: a list and index
    '''
    this function finds the pivot index, the index of a non 0 cell in the same row j so cancelation can occur for the triangular function.
    this function iterates from j, so that if the value above the given cell is 0, then the swap won't actually make any changes
    in the triangular function as k and j will be the same, but if it is a 0 above, the loop will run until a proper row is found
    and make the swap accordingly. Originally  the loop ran from 0, len(a), but in the last loop in triangular, it would make a swap
    that wasnt neccassary, ruining the upper triangular form, and making the back substitution immpossible.

    '''
    res=0
    for i in range(j, len(a)):
        if a[i][j] != 0:
            res=i
            break
    return res##the index of pivot
'''
cite:
idea from lecture 18, fit1045, Michael Kamp
my implementation
'''

def triangular(a, b):##input, a square matrix, and a column matrix of same length
    '''
    this function, is used to get the matrice, into upper triangular form, to do this it first iterates the columns from 0 , len(u)
    it then ensures for the chosen cell, there is not a 0 above it otherwise there would be a devision error, by swapping the rows
    if this occurs. It then will iterate through the rows (i), finds the value q each time, to produce the required 0 cancelation, for upper
    triangular form. the next loop using list comprehension, will change the given row, so that all columns in the paticular row recieve the same
    multiplication from the pivot row, hence following gaussian elimination, this function will run until it is in upper triangular form,
    when it is, it will run one more time but then stop without making further changes as there is no more need for changes.
    a difficult program to run, but thanks to the lectures, it was much easier. iterating through three different loops is always tricky
    to keep up to date with i, j, l and understanding what needs to be done each time.

    '''
    u, c = deepcopy(a), deepcopy(b)#copys them so the inputs are not changed
    for j in range(len(u)):#iterates through the columns
        k=pivot_index(u, j)#ensures the value above the given cell, is not a 0
        u[j], u[k] = u[k], u[j]
        c[j], c[k] = c[k], c[j]
        for i in range(j+1, len(a)):##iterates through the rows
            q=u[i][j]/u[j][j]#this is the value, that will generate the cancelation to created a 0 for each item in the matrix that requires it 
            u[i]=[u[i][l]-q*u[j][l] for l in range(len(u))]##this for loop ensures all values in the row of u, are changed based on the value q
            c[i] = c[i] - q*c[j]#this changes the value of b (y) for the given row
    return u, c##output, a matrix in upper triangular and the corresponding column matrix
'''
cite:
Michael Kamp lecture 18, week 9, 45th minute of the lecture fit1045
'''
def solve_gauss_elim(a, b):##input:two matrices of same row number, one square and the other a column
    u, c = triangular(a,b)
    return solve_by_back_sub(u, c)##output:the coefficients
'''

cite:
Michael Kamp lecture 18, week 9, 47th minute of the lecture fit1045
'''

def least_squares_predictor(data, y):
    """
    Finding the least squares solution by:
    - centering the variables (still missing)
    - setting up a system of linear equations
    - solving with elimination from the lecture

    For example:
    >>> data = [[0, 0],
    ...         [1, 0],
    ...         [0, -1]]
    >>> y = [1, 0, 0]
    >>> weights, intercept = least_squares_predictor(data, y)
    >>> round(weights[0],2)
    -1.0
    >>> round(weights[1],2)
    1.0
    >>> round(intercept, 2)
    1.0
    
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> weights, intercept = least_squares_predictor(data, y)
    >>> round(weights[0],2)
    0.29
    >>> round(weights[1],2)
    -0.43
    >>> round(intercept, 2)
    1.71
    """
    matrix_x=[]
    matrix_y=[]
    for i in range(len(data[0])):##gets the two matrices required gaussian elimination
        x,yy=equation(i, data, y)##using the equation function it will iterate through all equation possibilites
        matrix_x.append(x)#then append the the final x and y matrices
        matrix_y.append(yy)
    a=solve_gauss_elim(matrix_x, matrix_y)# uses the solve_gauss_elim function defined outside of the file
    def get_b(data, weights, y):#gets the value b
        lst=[]
        lst1=[]
        for i in range(len(data)):#iterates through x values i
            lst=[]
            for j in range(len(data[i])):##iterates through all variables
                lst.append(weights[j]*data[i][j])# a temp list containing the multiplied weights and x values for row 1
            lst1.append(sum(lst))#sums the temporary list and appends it to a new list
        lst2=[]
        for i in range(len(y)):##this then finds y - the matching equation to find the residual
            lst2.append(y[i]-lst1[i])
        b=sum(lst2)/len(y)
        return b#returning b
    b= get_b(data, a, y)
    return a, b
##note this function appears to be wrong to at arond the 16th decimal place for the extra examples, and i dont know why, otherwise
##the answers are correct up to around the 15th/16th depending on the problem
'''
How this function works:
this function first iterates through all possible equations, generating the coefficients that will create the augmented matrix
from there, gaussian elimination will be used (the problems and strategy to guassian will be in the functions triangular, pivot, and solve_by_back_sub)
and then the weights will be produced. The next challenge is to produce the residuals for each values of x1, x2 etc. this is hard,
as it needs to iterate a different amount of times depending on the number of x variables, and x values per column. To approach this,
I first, iterated through the different x explanatory variables, then within that, iterated through the individual x values in each column,
for the length of these columns. This then produces a temporary, that is summed as a1(x1)+a2(x2)....an(xn), clealy is just a summation,
this is then appended to a new list. Once the function has iterated through all of the variables, there will be a new list of length n,
this is the summation of x*weights, for each row, so to get the residuals, y[i]-the matching lst2[i] should work, as the list should have
maintained the same order, as it starts at index 0. Once this is completed we have calculated b, and can return the function.

computational complexity:
the first loop will run n times, with that being of the function equations time complexity each time, then gaussian elimination occurs.
the function triangular is where most of the computation happends, a for loop is run, then inside this for loop, the function pivot is run,
this will run, differently depending on the inputs, so best case is only one run, worst case is n-j, as it will run from j, n. a second
for loop is run, this time calculating q, but also, running another for loop, this for loop will iterate through all the values in the given row
of the x matrix, and y matrix, suggesting that all in all the triangular function will have a O(n^3), as there will be three for loops of similar
length that will need to be run. the solve_by_back_sub will run at O(n^2), as the first for loop will iterate backwards, but still need to run,
through all values n, then the second forloop inside that will run the same number of times, based on the length of u, so in total, O(n^2).
finally this implies the function solve_gauss_elim will run n^2, + n^3, making the gaussian elimination run at n^3.
next problem was finding b, first iterating through all rows, then all columns within this loop, it suggests row num * col num is the complexity,
which could probably be written as O(n^2). The sum function is also used which should be O(n), but this is used within the first loop not the second
so overall it was O(n^2) to find b (intercept). That implies that overall complexity for this function that the time complexity was O(n^3) (without the equation
function time complexity included), even with it included, the equation function has a n^2 according to my analysis

'''


def get_x_y(data):##before I use the other function it would be preferable to get the data first'
    """
    >>> x, y = get_x_y("life_expectancy.csv")
    >>> round(x[0][0],1)
    29.9
    >>> round(x[-1][-1],1)
    11.6
    >>> len(x)
    20
    >>> len(x[0])
    15
    >>> round(y[0], 1)
    61.1
    >>> len(y)
    20
    """
    ##above are just some basic tests to ensure that are not fool proof,
    #but should give a general idea if the data is correct

    
    file1=open(data, 'r')
    lst1=[]
    contents = file1.read()
    x=contents.strip()
    z=x.split('\n')
    f=0
    q=0
    for index in range(len(z)):
        lst1.append(z[f].split(','))
        f+=1
    for i in range(1, len(lst1)):
        for j in range(1, len(lst1[q])):
            lst1[i][j]=float(lst1[i][j])
        q+=1
    def get_x(list1):
        lst=[]
        lst1=[]
        lst2=[]
        lst3=[]
        for i in range(1, len(list1)):
            lst2=[]
            for j in range(1, len(list1[1])-1):
                lst2.append(list1[i][j])
            lst3.append(lst2)
        return lst3
    def get_y(lst):
        lst2=[]
        for i in range(1, len(lst)):
            lst2.append(lst[i][-1])
        return lst2
    x=get_x(lst1)
    y=get_y(lst1)
    return x, y
"""
The code above i decided to make seperate from analysis, as it more involves preparing
the CSV into a format readable by the best single predictor model.
it is relatively generic, using .read(), .strip, then .split to remove the /n at each
end of line, it is then appended in a list using a for loop, after every comma using .split.
The order of the code also simplifies the problem alot, is becomes alot harder to strip and
split correctly if the appending is done first. One problem i ran into was orginally instead
of using float() i was using int(), to convert the string values to integers, it was unable to
read the power numbers, so it was changed to float to ensure it is all correct. To get the x
values, a for loop runs skips index 0 of the current csv read list (as the first columns and rows
only contain strings, not values, and appends each value to maintain order then y simply runs
through the last column of the data and appends it to a fresh list.

"""
            
          
def regression_analysis(file):        
    """
    The regression analysis can be performed in this function or in any other form you see
    fit. The results of the analysis can be provided in this documentation. If you choose
    to perform the analysis within this funciton, the function could be implemented 
    in the following way.
    
    The function reads a data provided in "life_expectancy.csv" and finds the 
    best single predictor on this dataset.
    It than computes the predicted life expectancy of Rwanda using this predictor, 
    and the life expectancy of Liberia, if Liberia would improve its schooling 
    to the level of Austria.
    The function returns these two predicted life expectancies.
    
    For example:
    >>> predRwanda, predLiberia, predLiberia_after, rawanda_diff = regression_analysis("life_expectancy.csv")
    >>> round(predRwanda)
    65
    >>> round(predLiberia_after)
    79
    >>> round(predLiberia)
    63
    >>> round(rawanda_diff,2)
    -0.19
    
    
    """
    x, y = get_x_y(file)
    weights, b = best_single_predictor(x, y)#reuse previous function
    def not_0(weights):##to find the weights that is correct
        x=0
        for i in range(len(weights)):
            if weights[i] != 0.0:
                x=i
        return x
    index_a=not_0(weights)
    a_value = weights[index_a] ##returns a
    predicted_y=[]
    for j in range(len(y)):
        predicted_y.append(x[j][index_a]*a_value+b)##simple y predicted column
    lst4=[]
    for index in range(len(y)):#returns the 
        lst4.append(y[index]-predicted_y[index])
    diff_ped_act=lst4
    
        
    austria= predicted_y[13]
    rwanda=predicted_y[2]
    liberia=predicted_y[0]
    rwanda_diff= diff_ped_act[2]
    return rwanda, liberia, austria, rwanda_diff

'''
here we can see the regression model in action and compare the results,
the model predicts for Rwanda a life expectancy close to 65, the difference
between the actual and predicted being -0.19, a relatively good prediction
of life expectancy.
The model predicts that the life expectancy of liberia, would rise all the
way from 63 to 79, a huge increase relatively speaking, being nearly 16 years.
overall the model seems to peform well when looking at the diff_ped_act, showing
a decent level of proediction.
    
'''    
##Regression analysis code:
'''
With the main choice here being for loops, due to knowing the, amount of times it will
run in all of the cases, it seemed applicable. Furthermore the append function
was used often as it adds to the list as the index increases, making it maintain
the initial order as well (so the difference can be found easily)
Also a simple for loop with an if statement was employed to find the index
in which the correct weight was used, this makes the code more reusable,
instead of hardcoding the location of the true a value.
this index was then assigned to a variable and reused.
finally to find the desired results, I coulddnt think of a way to make it less
hardcoded, but instead i just returned a paticular index for predicted y
then returned it. I also used the previous function to allow for neat and
readable code.

'''

            
        
 
    
 
if __name__=='__main__':
    import doctest
    doctest.testmod()
