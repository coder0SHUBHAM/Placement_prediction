import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_excel('/content/student_dataset.xlsx')
print(df.head())
print(df.notnull().sum())

X=df[['CGPA','IQ']]
y=df['Placement']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

rl=LogisticRegression()
rl.fit(X_train,y_train)

cgpa= float(input("enter cgpa"))
iq=float(input("enter iq"))
y_pred=rl.predict([[cgpa,iq]])

print("prediction : 1=placed\n",y_pred)

y_pred_test = rl.predict(X_test)

X_test_np = X_test.to_numpy()

plt.scatter(
    X_test_np[:, 0],      # CGPA
    X_test_np[:, 1],      # IQ
    c=y_test, 
    cmap='bwr',
    marker='o',
    edgecolors='k',
    label='Actual'
)

plt.scatter(
    X_test_np[:, 0], 
    X_test_np[:, 1],
    c=y_pred_test,
    cmap='bwr',
    marker='x',     
    s=100,
    label='Predicted'
)

plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.title("Actual vs Predicted Placement (Scatter Plot)")
plt.legend()
plt.show()