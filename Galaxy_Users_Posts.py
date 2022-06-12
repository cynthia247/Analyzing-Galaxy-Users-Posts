# Shamse Tasnim Cynthia
# M.Sc. in CS
# University of Saskatchewan


# This script scrapes users posts from 
# https://help.galaxyproject.org/
# and some NLP techniques have been applied 
# to the created dataset.



# Importing all the necessary packages
import re
import random
import time
import pandas as pd
from pandas import DataFrame
from selenium import webdriver
from bs4 import BeautifulSoup
import gensim
import nltk
nltk.download('stopwords')
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim
import pickle
import pyLDAvis
from gensim.models import CoherenceModel
import warnings
warnings.filterwarnings('ignore')



def scroll(driver, timeout):
    """
        Purpose:
            This function handles the infinite scrolling of the web page
        Parameters:
            This function takes two parameters. The driver and a timeout. The driver is used for scrolling
            and the timeout is used to wait for the page to load.
        Returns:
            None.
    """
    scroll_pause_time = timeout

    # Scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(scroll_pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # If heights are the same it will exit the function
            break
        last_height = new_height



# New instance for Firefox driver
driver = webdriver.Firefox()

# Give the url address
k = driver.get("https://help.galaxyproject.org/")

# Using the scroll function to scroll the web page every 5 seconds
scroll(driver, random.randint(2,5))

# Using BeautifulSoup Library to interact with HTML
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Finding the specific HTML tag to scrape data
post_list = soup.find_all("tbody", "class" == "topic-list-body")
post_links = []

#   Getting all the posts link list
for post in post_list:
    tr = post.find_all("tr")
    for t in tr:
        links = t.find("td", {"class" : "main-link"})
        link = links.find('a')['href']
        post_links.append(link)

# Checking the total posts number
print(len(post_links))

# Keeping all the post links to a new CSV for future reference
post_urls = DataFrame(post_links)
post_urls.to_csv('url_links.csv', header = True)


def extract_data_from_webpage():
    """
        Purpose:
            This function scrapes the necessary data from the webpage.
        Parameters:
            None
        Returns:
            None
    """

    # Defining the baseurl
    baseurl = "https://help.galaxyproject.org/"

    # Initializing all the empty lists for storing data
    user_id = []
    title = []
    post = []
    data = []

    # For loop to iterate through all the post links
    for link in post_links:
        browser = webdriver.Firefox()   # New driver for each of the links
        browser.get(baseurl+link)
        html = browser.page_source
        soup = BeautifulSoup(html,'html.parser')

        # Storing the post title
        try:
            card = soup.find("div", "class" == "title-wrapper")
            title = card.find("a").text
        except:
            title = None

        # Storing the whole post data
        try:
            card = soup.find("div", "class" == "regular contents")
            c = card.find_all("p", "dir" == "ltr")
            c = str(c)
            post_text = BeautifulSoup(c, 'html.parser').text
        except:
            post_text = None

        # Storing the user id of the posts
        try:
            card = soup.find("div", "class" == "post-stream")
            user_id = card.find("article")["data-user-id"]
        except:
            user_id = None

        # Quiting the browser
        browser.quit()

        # Storing all the data as a dictionary
        all_data = {"user_id": user_id,
                "post_title" : title,
                "post_desc" : post_text
               }
        data.append(all_data)

    # Creating a dataframe and exporting the data as a CSV file for future use.
    data = DataFrame(data)
    data.to_csv('posts.csv', header=True)

# Calling the method to perform web scraping
extract_data_from_webpage()



"""
    APPLYING NLP MODEL TO THE CREATED DATASET
"""

# Reading the Posts.csv file
df = pd.read_csv('posts.csv')
corpus = df['post_desc']

# All the posts had this line mentioned below. So removing this line from all the posts.
content = "\nYou have selected 0 posts.\n\n, \n\n      select all\n    \n, \n\n    cancel selecting\n  \n, "


def utils_preprocess_text(text, lst_stopwords=None):
    """
        Purpose:
            This function does all the basic preprocessing of the text.
            (Removing line, cleaning, tokenizing, stemming and lemmitisation)
        Parameters:
            Takes four parameters according to the requirements of the corpus.
        Return:
            Returns the preprocessed text.
    """

    # Removing the common line
    try:
        text = text.split(content)
    except:
        text = text

    # Converting the text to lowercase and removing punctuations and characters and then stripping)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize
    lst_text = text.split()

    # Removing stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    # back to string from list
    text = " ".join(lst_text)
    return text


# Creating a new column in the dataframe of the preprocessed text
df['text_clean'] = df['post_desc'].apply(lambda x: utils_preprocess_text(x, lst_stopwords=None))
# Dropping the NaN values
df.dropna(inplace=True)

# Using wordcloud for the visual representation of the corpus
corpus = ','.join(list(df['text_clean'].values))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(corpus)
wordcloud.to_image()


def sent_to_words(sentences):
    """
        Purpose:
            This function converts a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
        Parameters:
            It takes one parameter -> the text
        Return:
            Returns the tokenized text
    """

    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

# Calling the sent_to_words() function
data = df.text_clean.values.tolist()
data_words = list(sent_to_words(data))


# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
        Purpose:
            This function lemmatizes the text
        Paramters:
            Takes two parameters -> the text and allowed post tags
        Returns:
            Returns the lemmatized tokens
    """
    texts_out = []
    for sent in texts:
        # Parse the sentence using the loaded 'en' model object `nlp`
        doc = nlp(" ".join(sent))

        # Extract the lemma for each token
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# Applying LDA Model
# Create Dictionary and corpus for the LDA Model
id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10,
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)



# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)



# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
LDAvis_prepared