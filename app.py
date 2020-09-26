import flask
import pickle
import numpy as np

app = flask.Flask(__name__, template_folder=r'C:\Users\user\Desktop\Machine Learning\Deployment\template')
model = pickle.load(open(r'C:\Users\user\Desktop\Machine Learning\Deployment\Hiring.pkl', 'rb'))

@app.route('/')
def home():
    return flask.render_template('Hiring.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    output = model.predict(final_features)
    return flask.render_template('Hiring.html', prediction_text=f'The salary should be ${output[0]}')

if __name__ == '__main__':
    app.run(debug=True)
