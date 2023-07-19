import numpy as np 
from matplotlib import pyplot as ptl
import sklearn.linear_model
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import random

df = pd.read_csv(r'C:\Users\ayoya\OneDrive\Documents\ML project\Salary.csv')
x = df.YearsExperience
y = df.Salary 
print(df)

w , b = np.polyfit(x, y, 1) #will applay the gridient decent and cost function to determine the best value for w and b 
print ("The value of parameter : " , w , b  )#will print the value of w , b 
model = w , b   # model is array 
print (type(model))

ptl.plot(x , w*x+b) # the value of x and y  // and wiil plot the point for the line 

predict = np.poly1d(model) #to predict the value when you enter the number of years experince 


x_test= random.randrange(0, 14) 
print( "The Salary for" ,x_test , "years of experience is : " , f"{predict( x_test ):.2f}" " $"  )

print ( "the Model accuracy :" ,  f"{r2_score(y, predict(x))*100 :.0f}" " %" ) #calculate the accuracy

ptl.scatter(df.YearsExperience, df.Salary , c='r' , marker= 'x'  )
ptl.title("YearsExperince vs. Salary")
ptl.ylabel('Salary')
ptl.xlabel('YearsExperince')
ptl.show()

