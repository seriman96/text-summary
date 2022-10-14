from flask import Flask, render_template, request, redirect, url_for, session
from Text_Summary import text_sum_alg

app = Flask(__name__)
app.secret_key = "#$%#$%^%^BFGBFGBSFGNSGJTNADFHH@#%$%#T#FFWF$^F@$F#$FW"

@app.route("/")
def index():
	return render_template("index.html")


@app.route("/search", methods=["POST", "GET"])
def searchr():
	if request.method == "POST":
		txt = '{}'.format(request.form["text"])
		results = text_sum_alg(txt)

		session["results"] = results
		session["query"] = txt
		return redirect(url_for("searchr"))
	return render_template("index.html", results=session["results"], query=session["query"])


if __name__ == '__main__':
	app.run(debug=True)