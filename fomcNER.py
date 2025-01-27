import pandas as pd
import matplotlib.pyplot as plt
import warnings
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import re
import spacy
from sklearn.metrics import classification_report,accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding,SpatialDropout1D,Bidirectional,Dropout
import demoji
from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('omw-1.4')
warnings.filterwarnings("ignore")


df = pd.read_csv("C:/nlp/Fed_Scrape-2015-2023.csv",delimiter=',',nrows=20000)


df.drop('Unnamed: 0',inplace=True,axis=1)
df.head(10)

print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)

def clean_text(text):

    text = str(text).lower()
    text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9_]+)', '',text)

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'#(\w+)','',text)

    text = demoji.replace(text,'')

    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'[^\w\s]', '',text)

    text = re.sub(r'\@w+|\#','',text)

    text = re.sub(r'ï½','',text)




    return text





def remove_stopwords(text):
    sw = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.lower() not in sw]
    return " ".join(cleaned_tokens)




def lemmatizer(text):
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemma_tokens = [lemma.lemmatize(token) for token in tokens]
    return " ".join(lemma_tokens)




df['Text'] = df['Text'].apply(lambda x: clean_text(x))
df['Text'] = df['Text'].apply(lambda x: remove_stopwords(x))
df['Text'] = df['Text'].apply(lambda x: lemmatizer(x))
df['Text'].head(10)

sample_txt = " ".join(i for i in df['Text'])

wc = WordCloud(colormap="Set2",collocations=False).generate(sample_txt)
plt.axis("off")
plt.imshow(wc,interpolation='bilinear')
plt.show()

blob = TextBlob(sample_txt)
most_common_words = FreqDist(blob.words).most_common(50)
print(f'top 50 most common words: {most_common_words}')

nlp = spacy.load("en_core_web_sm")


doc = nlp(sample_txt[:2000])


for ent in doc.ents:
    print(ent.text, "|",spacy.explain(ent.label_))

cv = CountVectorizer()


X = df['Text']
X = cv.fit_transform(X).toarray()
y = df['Type']





X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)




pa = PassiveAggressiveClassifier()
gnb = GaussianNB()
lr = LogisticRegression()
GBC= GradientBoostingClassifier()

def evaluate_sklearn_models(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    clf_rpt = classification_report(y_test,pred)
    acc = accuracy_score(y_test, pred)
    print(f'{model.__class__.__name__}, --Classification Report-- {clf_rpt}; --Accuracy-- {acc*100:.2f}')
    return pred


gnb_pred = evaluate_sklearn_models(X_train, X_test, y_train, y_test, gnb)
lr_pred = evaluate_sklearn_models(X_train, X_test, y_train, y_test, lr)
pa_pred = evaluate_sklearn_models(X_train, X_test, y_train, y_test, pa)
GBC_pred = evaluate_sklearn_models(X_train,X_test,y_train,y_test,GBC)

def plot_confusion_matrix(y_true,y_pred,model):
    confmat = confusion_matrix(y_test, y_pred)
    sns.heatmap(confmat,fmt='d',annot=True,cmap='coolwarm')
    plt.title(f"Confusion Matrix for {model.__class__.__name__}")
    plt.show()

plt.figure(figsize=(10,6))
plot_confusion_matrix(y_test, lr_pred, lr)


plt.figure(figsize=(10,6))
plot_confusion_matrix(y_test, gnb_pred, gnb)


plt.figure(figsize=(10,6))
plot_confusion_matrix(y_test, pa_pred, pa)


plt.figure(figsize=(10,6))
plot_confusion_matrix(y_test,GBC_pred,GBC)

X = df['Text']
y = df['Type']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print(f'length of word index: {len(word_index)}')


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


max_length = 0
for sequence in X_train:
    sequence_length = len(sequence)
    if sequence_length > max_length:
        max_length = sequence_length

print("Max Length of Sequences: ",max_length)

X_train = pad_sequences(X_train,padding="post")
X_test = pad_sequences(X_test,padding="post")


RNN = Sequential()
RNN.add(Embedding(len(word_index)+1,output_dim=300,input_length=max_length))
RNN.add(SpatialDropout1D(0.3))
RNN.add(Bidirectional(LSTM(250,dropout=0.1,recurrent_dropout=0.1)))
RNN.add(Dropout(0.2))
RNN.add(Dense(250,activation='relu'))
RNN.add(Dropout(0.1))
RNN.add(Dense(1,activation='sigmoid'))
RNN.summary()



RNN.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = RNN.fit(X_train,y_train,epochs=5,batch_size=32,validation_split=0.1)
loss,acc = RNN.evaluate(X_test,y_test)
pred = RNN.predict(X_test)

print(f"Testing Loss: {loss:.2f}")
print(f"Testing Accuracy: {acc*100:.2f}")
