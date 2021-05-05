from flask import Flask,render_template,request
import crf_model_test
import knnk
import jinja2
env = jinja2.Environment()
env.globals.update(zip=zip)

app = Flask(__name__)

@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/',methods=['POST'])
def span_detection():
	if request.method=='POST':
		data=request.form['data']

		if request.form['action']=='crf':
			X=crf_model_test.prediction(data)
			print(X)
			return render_template("index.html",X=X)

		elif request.form['action']=='naive-bayes':
			X=knnk.naive_bayes_prediction(data)
			print(X)
			return render_template("index.html",X=X)

		elif request.form['action']=='random-forest':
			X=knnk.random_forest_prediction(data)
			print(X)
			return render_template("index.html",X=X)

		elif request.form['action']=='ann':
			X=knnk.ann_prediction(data)
			print(X)
			return render_template("index.html",X=X)




if __name__ == '__main__':
    app.run(debug=False, threaded=False, host='0.0.0.0')
