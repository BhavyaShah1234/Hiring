import pandas as pd
import pickle as pkl
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r'C:\Users\user\Desktop\Machine Learning\Deployment\Hiring.csv')
df = pd.DataFrame(dataset)

df['experience'] = df['experience'].fillna(value='zero')
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(value=df['test_score(out of 10)'].mean())

X = df.drop(columns='salary($)')
Y = df['salary($)']

word_dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12}
for i in range(X.shape[0]):
    X.iloc[i, 0] = word_dict[X.iloc[i, 0]]

X['experience'] = X['experience'].astype('int32')

lin = LinearRegression()
lin.fit(X, Y)

pkl.dump(lin, open('Hiring.pkl', 'wb'))

model = pkl.load(open('Hiring.pkl', 'rb'))
pred = model.predict([[2, 9, 6]])
