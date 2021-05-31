from flask import request, jsonify,render_template
import json
import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    covid = {'result':"positive",'value':"36"}
    return render_template('index.html',covid = json.dumps(covid))

app.run()