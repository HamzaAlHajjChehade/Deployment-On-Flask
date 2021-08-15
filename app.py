#Importing the Libraries:
import numpy as np
import pandas as pd
import pickle


#Importing the Dataset:
df=pd.read_csv('C:/Users/mizoh/Desktop/Data Glacier/50_Startups.csv')
X=df.iloc[:,:3].values
y=df.iloc[:,-1].values

'''#Encoding the categorical variables:
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])'''

#Splitting the dataset into Training set and Test set:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

#Linear Regression Model:
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

#Serialization(Convert python model into a file):
with open('model.pkl','wb') as model:
    pickle.dump(regressor,model)


from flask import Flask,render_template,request

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Profit is {}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)