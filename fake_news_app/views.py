from django.shortcuts import render
import pickle as pk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

# Load model and vectorizer
model = pk.load(open('model.pkl', 'rb'))
vectorizer = pk.load(open('scaler.pkl', 'rb'))

port_stem = PorterStemmer()

def stemming(content):
    """Preprocess and stem the input content."""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        port_stem.stem(word)
        for word in stemmed_content
        if not word in stopwords.words('english')
    ]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def index(request):
    prediction = None
    if request.method == 'POST':
        news_content = request.POST.get('news')
        
        if news_content:            
            processed_content = stemming(news_content)
            vectorized_content = vectorizer.transform([processed_content])
            prediction = model.predict(vectorized_content)[0]
            prediction = "Real" if prediction == 1 else "Fake"
        else:
            prediction = "Please enter some news."

    return render(request, 'index.html', {'prediction': prediction})
