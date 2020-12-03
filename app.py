from bs4 import BeautifulSoup
import requests
import re
from collections import OrderedDict
import numpy as np
import math
import json
from numpy import inf
from flask import Flask, request, render_template  

app = Flask(__name__)

@app.route('/', methods =["GET","POST"]) 
def gfg(): 
    if request.method == "POST": 
       # getting input with name = fname in HTML form 
       queryEntered = request.form.get("queryE") 
       mainList = getList()
       posting = getPosting(mainList)
       similarity = getSimilarity(mainList, posting, queryEntered)
       #return "Your name is "+str(similarity)
       #return "<h1>Test</h1>"
       dlist = '<ul>'
       inlist = ''
       for i in similarity:
           inlist += "<li><a href=" + mainList[int(i)]["url"]  + ">" + mainList[int(i)]["title"] + "</a></li>"
       
       dlist += inlist + "<br /><a href=.>New Search</a></ul>"
       
       return dlist
    return render_template("index.html")
def getText(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    text = soup.find_all(text=True)
    output = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        # there may be more elements you don't want, such as "style", etc.
    ]
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output
def getList():
    #main url
    url = 'https://lite.cnn.com/en'
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')


    articles = soup.find_all('a')[2:-2]
    #cacm will contain title and its url and content
    cacm = {}
    for i in range(len(articles)):
        cacm[i] = {
            'title': articles[i].get_text(),
            'url': url[:-3] + articles[i].get('href'),
            'content': getText(url[:-3] + articles[i].get('href'))
        }
    return cacm

def getPosting(mainList):
    posting = {}
    for doc in mainList:
        docID = doc
        splitArrT = re.findall('[$\w][A-Za-z0-9.]+[$\w]',
                            mainList[docID]['content'])
        #implement stemming if you want
        for i in range(len(splitArrT)):
            word = splitArrT[i]
            tWords = word.lower()
            #implement stop words here
            if(tWords not in posting):
                posting[tWords] = {docID: {"tFreq": 1, "position": [i+1]}}
            else:
                if(doc in posting[tWords]):
                    #More occurance of term (already in posting) in the current document
                    posting[tWords][docID]["tFreq"] = posting[tWords][docID]["tFreq"]+1
                    posting[tWords][docID]["position"].append(i+1)
                else:
                    #Occurance of term (already in posting) in another document
                    posting[tWords][docID] = {"tFreq": 1, "position": [i+1]}


    #ordering posting file
    posting = dict(OrderedDict(sorted(posting.items())))

    return posting

def getSimilarity(mainList, posting, queryTT):
    queryT = queryTT


    queryT = re.findall('[$\w][A-Za-z0-9.]+[$\w]', queryT)
    queryT = [item.lower() for item in queryT]
    query = []
    #stopwords
    # if(stopWords):
    #     for word in queryT:
    #         #print(word)
    #         if(word not in stopWordFile):
    #             query.append(word)
    # else:
    query = queryT
    #stemmer implement here
    #All words that are not preset in posting, added in there temporarily
    addedWords = []  # Remove these from posting if doing another query in same program run
    for word in query:
        if(word.lower() not in posting):
            posting[word.lower()] = {}
            addedWords.append(word.lower())

    #lists to grab index for a term
    pK = list(posting.keys())

    #Zero arrays thats used for eacbh doc and query
    zarray = np.zeros(len(pK))

    #Docs matrix (bag of words). Each row represents a document
    docs = []
    for i in range(len(mainList)):
        docs.append(np.copy(zarray))
    docs = np.array(docs)

    #Docs Processing
    for term in posting:
        indOfTerm = pK.index(term)
        for d in posting[term]:
            indOfDoc = int(d)
            docs[indOfDoc][indOfTerm] = posting[term][d]['tFreq']
    #Query Processing
    q1 = np.copy(zarray)
    for word in query:
        if(word.lower() in posting):
            ind = pK.index(word.lower())
            #print(ind)
            q1[ind] += 1

    with np.errstate(divide='ignore'):
        docs = 1+np.log10(docs)
        q1 = 1+np.log10(q1)
    q1[q1 == -inf] = 0
    docs[docs == -inf] = 0

    #IDF
    idf = np.copy(zarray)
    N = len(mainList)  # No. of document
    for word in posting:
        ind = pK.index(word.lower())
        if(len(posting[word]) > 0):
            idfVal = math.log10(N/len(posting[word]))
        else:
            idfVal = 0
        idf[ind] = idfVal

    #lambda function multiply IDF to a row


    def f(x): return np.multiply(x, idf)


    docsW = f(docs)

    q1W = np.multiply(q1, idf)

    #Lambda function to square


    def sq(x): return x**2


    #Processing similarity
    similarity = {}

    for d in range(len(docsW)):
        numerator = np.sum(np.multiply(docsW[d], q1W))
        #denominator calculations
        dTotalSqrt = math.sqrt(np.sum(sq(docsW[d])))
        qTotalSqrt = math.sqrt(np.sum(sq(q1W)))
        denominator = dTotalSqrt * qTotalSqrt
        if denominator == 0 :
            similarity[str(d)] = 0
        else :
            similarity[str(d)] = numerator/denominator
    
        #similarity[str(d)] = numerator/denominator

    similarity = dict(filter(lambda elem: elem[1] > 0, similarity.items()))
    similarity = {k: v for k, v in sorted(
        similarity.items(), key=lambda item: item[1], reverse=True)}

    for wd in addedWords:
        del posting[wd]
    #need to add page rank score to the sim score
    return similarity

#EXAMPLE RUNNING OF CODE
#mainList = getList()
#posting = getPosting(mainList)
#get similarity takes input right now but can be replaced with input online
#similarity = getSimilarity(mainList, posting)
if __name__ == '__main__':
	app.run(debug=True)




    
