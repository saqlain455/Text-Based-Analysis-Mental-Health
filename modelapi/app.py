from flask import Flask,jsonify,request
import pickle
app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def tryit():
    args=request.args.get('text')
    print(args)
    return "flask api for mental health"


@app.route("/predict",methods=['GET','POST'])
def predict_model():
    filename='filem.sav'
    loaded_model=pickle.load(open(filename,'rb'))
    print("arguments")
    ex1=request.args.get('text')
    # ex1=request.form['text']
    # ex1 = "I do not sleep"
    print(ex1)
    print(loaded_model.predict([ex1]))
    print(loaded_model.predict_proba([ex1]))
    array=loaded_model.predict_proba([ex1])
    d = dict(enumerate(array.flatten(), 1))
    print(d)
    print(loaded_model.classes_)

    # return str(loaded_model.predict([ex1]))
    # return str(loaded_model.predict_proba([ex1]))
    # return d
    # return str(d[5])

    return jsonify({"anger":str(d[1]),"boredom":str(d[2]),"fear":str(d[3]),"hate":str(d[4]),"insomnia":str(d[5]),"sadness":str(d[6])})

if __name__ == "__main__":
    app.run()