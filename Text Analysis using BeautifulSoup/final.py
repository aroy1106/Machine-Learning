import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('cmudict')
from nltk.corpus import stopwords
from nltk.corpus import cmudict
import requests
from bs4 import BeautifulSoup
import string
import re
import os 
import pandas as pd
import numpy as np

for i in range (0, 115) :
    def fetchAndSaveToFile(url,path) :
        r = requests.get(url)
        with open (path,'w',encoding="utf-8",errors="ignore") as f :
            f.write(r.text)

    url = input("Enter the url : ")
    url_id = input("Enter url-id : ")
    path_data = input("Enter the path you want to save text file : ")
    path_html = input("Enter the path you want to save html file : ")

    text_filename = f"{url_id}.txt"
    html_filename = f"{url_id}.html"

    html_path = os.path.join(path_html,html_filename)
    data_path = os.path.join(path_data,text_filename)

    fetchAndSaveToFile(url,html_path)

    print(f"File '{html_filename}' has been created at '{html_path}'.")
    print(f"File '{text_filename}' has been created at '{data_path}'.")

    with open (html_path,'r') as f :
        html_data = f.read()

    soup = BeautifulSoup(html_data,'html.parser')

    with open (data_path,'w+') as f :
        title = soup.find('h1',class_="entry-title")
        if(title == None) :
            title = soup.find('h1',class_="tdb-title-text")
            f.write(title.get_text() + "\n")
        else :
            f.write(title.get_text() + "\n")
        div = soup.find('div',class_="td-post-content tagdiv-type")
        if (div == None) : 
            div = soup.find('div',class_="td_block_wrap tdb_single_content tdi_130 td-pb-border-top td_block_template_1 td-post-content tagdiv-type")
        p_tags = div.find_all('p')
        for p in p_tags :
            f.write(p.get_text())

    def cleanText(text, stop_words):
        words = text.split()
        cleaned_words = [word for word in words if word not in stop_words]
        cleanedText = ' '.join(cleaned_words)
        return cleanedText 

    files = ['StopWords_Auditor.txt','StopWords_Currencies.txt','StopWords_DatesandNumbers.txt','StopWords_Generic.txt','StopWords_GenericLong.txt','StopWords_Geographic.txt','StopWords_Names.txt']

    stop_words = []
    for file in files :
        with open(file,'r') as f :
            stop_words.extend(f.read().splitlines())

    with open(data_path,'r') as f :
        article = f.read()

    cleaned_article = cleanText(article,stop_words)

    with open(data_path,'w') as f :
        f.write(cleaned_article)

    def splitIntoSenetences(file) :
        with open (file, 'r') as f :
            text = f.read()

        sentences = text.split('. ')
        sentences = text.split('.')

        with open (file, 'w') as f :
            for sentence in sentences :
                f.write(sentence + '.\n')

    splitIntoSenetences(data_path)

    with open(data_path,'r') as f :
        content = f.read()

    sentences = nltk.sent_tokenize(content)

    with open ('positive-words.txt','r') as f :
        pos_words = f.read().splitlines()

    stop = set(stopwords.words('english'))

    filtered_positive = [word for word in pos_words if word not in stop]

    with open ('negative-words.txt','r') as f :
        neg_words = f.read().splitlines()

    filtered_negative = [word for word in neg_words if word not in stop]

    #CALCULATING POSITIVE SCORE
    positive_score = 0
    with open (data_path,'r') as f :
        content = f.read()

    words = content.split()

    for word in words :
        if (word in filtered_positive) :
            positive_score += 1

    print("Positive score : ",positive_score)

    #CALCULATING NEGATIVE SCORE
    negative_score = 0
    for word in words :
        if (word in filtered_negative) :
            negative_score -= 1
    negative_score *= -1

    print("Negative Score : ",negative_score)
    #CALCULATING POLARITY SCORE
    nume = (positive_score - negative_score)
    deno = (positive_score + negative_score + 0.000001)
    polarity_score = nume / deno
    print("Polarity Score : ",polarity_score)

    #CALCULATING SUBJECTIVITY SCORE
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    print("Subjectivity Score : ", subjectivity_score)

    sentences = content.split('.\n')
    no_of_sentences = len(sentences)

    #CALCULATING AVG SENTENCE LENGTH
    avg_sent_length = len(words) / no_of_sentences
    print("Average Sentence Length : ", avg_sent_length)

    #CALCULATING PERCENTAGE OF COMPLEX WORDS
    with open (data_path,'r') as f :
        text = f.read()
    cleaned_text = ''.join([i for i in text if not i.isdigit()])

    with open(data_path, 'w') as file:
        file.write(cleaned_text)

    d = cmudict.dict()

    def nsyl(word):
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]

    with open(data_path, 'r') as f:
        text = f.read()

    words = nltk.word_tokenize(text)

    num_complex_words = sum(1 for word in words if word.lower() in d and max(nsyl(word.lower())) >= 2)

    print("Complex Word Count : ", num_complex_words)
    words = text.split()

    complex_percentage = num_complex_words / len(words)

    print("Percentage of complex words : ", complex_percentage)

    # CALCULATING FOG INDEX

    fog_index = 0.4 * (avg_sent_length + complex_percentage)
    print("Fog Index : ", fog_index)

    # WORD COUNT
    with open (data_path,'r') as f :
        text = f.read()

    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    filtered_words = [word for word in words if word  not in string.punctuation]

    cleaned_article = ' '.join(filtered_words)

    with open (data_path,'w') as f :
        f.write(cleaned_article)

    with open (data_path,'r') as f :
        article = f.read()

    content = article.split()
    word_count = len(content)
    print("Word count : ", word_count)

    # SYLLABLE COUNT PER WORD
    d = cmudict.dict()

    def nsyl(word):
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]

    with open(data_path, 'r') as f:
        text = f.read()

    words = nltk.word_tokenize(text)

    syllable_counts = {word: nsyl(word)[0] if word.lower() in d else 0 for word in words}
    total_frequency = sum(syllable_counts.values())
    total_words = len(syllable_counts)
    avg_syllable_count_per_word = total_frequency / total_words
    print("Average Syllable count per word : ", avg_syllable_count_per_word)

    # COUNTING PERSONAL PRONOUNS
    pronouns = ['I', 'we', 'my', 'ours', 'us']

    with open(data_path, 'r') as f:
        text = f.read()

    num_pronouns = sum(1 for word in re.findall(r'\b\w+\b', text) if word in pronouns)

    print("Number of personal pronouns:", num_pronouns)

    # AVERAGE WORD LENGTH 
    with open (data_path,'r') as f :
        text = f.read()

    words = text.split()

    avg_word_length = sum(len(word) for word in words) / len(words)

    print("Average word length : ", avg_word_length)