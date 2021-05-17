```python
# check scikit-learn version
import sklearn
print(sklearn.__version__)
```

    0.24.1



```python
import pandas as pd
```


```python
data_set= pd.read_csv('Zoey.csv')  
```


```python
data_new=data_set[["Age","Timing"]]
```


```python
data_final=data_new[0:50]
```


```python
print(data_final)
```

         Age  Timing
    0    7.0   51.24
    1    7.0   52.86
    2    7.0   53.24
    3    8.0   37.79
    4    8.0   40.84
    5    8.0   40.95
    6    8.0   41.20
    7    9.0   34.94
    8    9.0   34.99
    9    9.0   35.68
    10   9.0   35.99
    11   9.0   36.40
    12   9.0   38.07
    13   9.0   38.23
    14   9.0   39.17
    15  10.0   31.08
    16  10.0   31.17
    17  10.0   31.41
    18  10.0   31.47
    19  10.0   31.85
    20  10.0   31.91
    21  10.0   31.96
    22  10.0   32.56
    23  10.0   33.53
    24  10.0   34.60
    25  11.0   29.14
    26  11.0   29.58
    27  11.0   29.67
    28  11.0   30.35
    29  12.0   28.13
    30  12.0   28.17
    31  12.0   28.50
    32  12.0   28.77
    33  12.0   29.06
    34  12.0   29.15
    35  13.0   26.38
    36  13.0   26.71
    37  13.0   26.76
    38  13.0   27.02
    39  13.0   27.42
    40  14.0   26.42
    41  14.0   26.55
    42  15.0   25.07
    43  15.0   25.16
    44  15.0   26.44
    45  16.0   26.01
    46  17.0   25.73
    47  24.0   24.40
    48  24.0   24.47
    49  24.0   25.56



```python
import numpy as np
import matplotlib.pyplot as plt

X = data_final.iloc[:, :-1].values #get a copy of dataset exclude last column
Y = data_final.iloc[:, 1].values #get array of dataset in column 1st
```


```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
```


```python
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```




    LinearRegression()




```python
# Visualizing the Training set results
viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('Timing VS Age (Training set)')
viz_train.xlabel('Age')
viz_train.ylabel('Timing')
viz_train.show()

# Visualizing the Test set results
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('Timing VS Age (Test set)')
viz_test.xlabel('Age')
viz_test.ylabel('Timing')
viz_test.show()
```


    
![png](output_9_0.png)
    



    
![png](output_9_1.png)
    



```python
lin_reg_2.predict(np.array([6.5]).reshape(1, 1))
```


```python
# Predicting the result of someone at the age of 24, 18, and 16 Years 
y_pred = regressor.predict([[24],[18],[16]])
```


```python
y_pred
```




    array([24., 18., 16.])


