import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random
from flask import Flask, request, jsonify

# Parsing URLs
def parsing(URL):
    tokens = []
    # split by "/"
    bySlash = str(URL).split("/")
    for i in bySlash:
        # split by "-"
        byDash = str(i).split("-")
        byDot = []
        for j in range(0, len(byDash)):
            # split by "."
            temp = str(byDash[j]).split(".")
            byDot = byDot + temp
        tokens = tokens + byDash + byDot
    tokens = list(set(tokens))
    # remove unnecessary part
    if ".com" in tokens:
        # remove .com
        tokens.remove(".com")
    if " " in tokens:
        # remove space
        tokens.remove(" ")
    if "" in tokens:
        # remove none
        tokens.remove("")
    if "www" in tokens:
        # remove www
        tokens.remove("www")
    if "https:" in tokens:
        # remove https:
        tokens.remove("https:")
    return tokens


# Read data

# Creating Data Frame
data_training = pd.DataFrame(pd.read_csv("data_index.csv", ",", error_bad_lines=False))
print("Data Size: ", len(data_training))
# Converting it into an array
data_training = np.array(data_training)
random.shuffle(data_training)
# error_bad_lines : bool, default True
# Lines with too many fields (e.g. a csv line with too many commas) will by default cause an exception to be raised,
# and no DataFrame will be returned. If False, then these “bad lines” will dropped from the DataFrame that is returned.

# Label array (0 / 1) 0 = bad, 1 = good
Y = [a[2] for a in data_training]
# URL array (FULL URL)
arrayX = [a[1] for a in data_training]
# Parsing URL
cusTokenizer = TfidfVectorizer(tokenizer=parsing)
# URL array (parsed URL)
X = cusTokenizer.fit_transform(arrayX)

# Training data

# Logistic Regression
# Split into training and testing set 70/30 ratio
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# using logistic regression
lgs = LogisticRegression()
lgs.fit(x_train, y_train)
# Print the score
acc = lgs.score(x_test, y_test)
print("Logistic Regression Accurate:", acc)

# output model
'''
with open("LGSmodel.pickle", "wb") as f:
    pickle.dump(lgs, f)
print("Output Model Successfully")

# load model

lgsModel = pickle.load(open('LGSmodel.pickle', 'rb'))
print("Model load successfully!")
print("Coefficient: ", lgsModel.coef_)
print("Intercept: ", lgsModel.intercept_)
'''
# test

# Website
Website = Flask(__name__, static_folder='site/static')
@Website.route('/')
def index():
    return Website.send_static_file('html/index.html')


@Website.route('/classify', methods=['POST'])
def classify():
    text = request.form.get('text', None)
    assert text is not None
    print("text: ", text)
    # Parsing URL
    url_pred = cusTokenizer.transform([text])
    # print(lgs.predict(url_pred))
    # acc = lgsModel.score(x_test, y_test)
    # print("LGSmodel Accurate:", acc)
    # prediction = lgsModel.predict(x_test)
    prediction=lgs.predict(url_pred)

    prob_neg, prob_pos = lgs.predict_proba(url_pred)[0]
    s = 'Positive' if prob_pos >= prob_neg else 'Negative'
    p = prob_pos if prob_pos >= prob_neg else prob_neg
    return jsonify({
        'sentiment': s,
        'probability': p
    })

Website.run()


