from flask import Flask,request,render_template
import DistanceEstimation
import sys,os,signal
import threading


app=Flask(__name__)

t1 = threading.Thread(target=app.run)
t2 = threading.Thread(target=DistanceEstimation.SWAB)

@app.route("/")
def home(): 
    
    return render_template("Main.html")

@app.route("/exit")
def exit():
    os._exit(0)


if __name__=="__main__":
    t2.start()
    t1.start()