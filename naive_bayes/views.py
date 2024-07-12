from django.shortcuts import render
from django.http import request
import pickle
import nltk 
from nltk.corpus import stopwords
nltk_stopwords=set(stopwords.words('english'))

def sms_predictor(request):
    context = {}
    if request.method=="POST":
        data= request.POST
        sms_data = data.get('sms_data')
        msg = []
        sms_data=sms_data.split()
        filtered=[j for j in sms_data if j not in nltk_stopwords]
        filtered =' '.join(filtered)
        msg.append(filtered)
        model = pickle.load(open('naive_bayes/naive_dump.sav','rb'))
        prediction = model.predict(msg)
        context = {'prediction': prediction}
    return render(request,"sms_predictor.html",context)

def welcome_page(request):
    return render(request,"index.html")