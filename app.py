import pandas as pd 
from flask import Flask, jsonify, request
import pickle 

# Load Model: 
model = pickle.load(open("lin_model.pkl", "rb"))

# Define App: 
app = Flask(__name__)

# Routes: 
@app.route("/", methods=['POST'])

def predict(): 
    data = request.get_json(force=True)
    data.update((x, [y]) for x, y in data.items())
    df = pd.DataFrame.from_dict(data)

    result = model.predict(df)

    output = {'results': int(result[0])}

    return jsonify(results=output)

if __name__ == '__main__': 
    app.run(port=5000, debug=True)