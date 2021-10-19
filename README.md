# Docummend
![Docummend](https://user-images.githubusercontent.com/32643842/137021376-0dc06282-d912-4d2f-b75b-9bdf2fd25b5b.png)

Docummend is a proof-of-concept search method that uses advanced Natural Language Processing combined with a recommendation engine to make it possible to accurately search through large amounts of data without search-terms. Instead, users upload their entire research document to be compared against the documents the model was trained on and output the desired amount of closest matches. While this is a proof of concept built on a small dataset it is very easy to implement and scale up with any kind of academic data.

## Gathering Data
I obtained my training data from Westlaw. The data consisted of 1,077 individual judicial decisions in New York State from the last 3 years relating to personal injury cases. Using some custom functions I was able to convert the data into text form, do some light cleaning, and insert it into a Pandas dataframe that I saved as a CSV. To see these functions in detail, refer to the Functions.ipynb file in the repository.

My target data in this case was a Bill of Particulars, the document in every lawsuit that lays out the principle facts in a case. This was a BP from 2021 for an active case which was the best way for me to test my hypothesis with a real world example. 

The target data was in a .docx format so I wrote a simple function to convert the data into text form. I then performed a similar conversion to the train data in order to fit the target data into a Pandas dataframe and save it to a CSV. Once the data preparation was finished I loaded the train and target data back into their own dataframes. I also loaded a Stopwords instance for NLP analysis.

## EDA
My analysis consisted of standard NLP like the total number of unique words, normalized word frequencies, and a Word Cloud of top words. I also used an LDA model to cluster my data for a more in-depth analysis. To get the normalized frequencies I first had to flatten the words, get the total count, and finally print the top words with their normalized frequencies. Using some of the data from above I created a Word Cloud of the top 25 words in my train dataset.

A more in-depth analysis of the train data can be implemented with an LDA clustering model. Since I am not an expert in personal injury litigation I was only comfortable with a very general analysis.

![Analysis1](https://user-images.githubusercontent.com/32643842/137021552-8d29cf73-fcec-4b14-a1ba-533603f72cb0.png)
In the first cluster we see words like Plaintiff and Defendant, Complaint, Recover, Personal, and Damage which are all common terms associated with Negligence and Premises cases.

![Analysis2](https://user-images.githubusercontent.com/32643842/137021646-d9c6c909-607c-4f54-b6cc-b960c197a03f.png)
In the second cluster words like Injury, Limitation, Medical, Spine, Lumbar, and Cervical are usually associated with Medical Malpractice cases. 

By creating 2 clusters, the model was able to distinguish between Medical Malpractice Cases and general Premises or Negligence cases, which are the bulk of cases that personal injury attorneys take on. 

While my engine was able to vectorize with standard vectorizers like the Count Vectorizer and Tfidf, vectorizing with spaCy took significantly more time, especially considering that spaCy has several neural network trained libraries. Since I wanted to test several of these libraries I wrote a custom function to vectorize my data and save the output as a CSV to speed up the actual modeling.

## Modeling Results
I implemented a Nearest Neighbors model to get my recommendations. I started with a simple Countvectorizer and a Tfidf vectorizer. Then I ran the model on spaCy ‘medium’ and spaCy ‘large’ vectorization libraries.

Here we can see the results from the  best performing model based on the feedback from the attorney who provided the target document.

![Results1](https://user-images.githubusercontent.com/32643842/137021904-2928b993-4083-43a8-9088-f88def50afa7.png)
We get the names of the decisions and each decision is saved as an individual file into the results folder.
 
At the same time we can print a quick post modeling analysis displaying the percent difference between each decision and our input document, as well as the category that our document falls into based on the clustering analysis.

![Results2](https://user-images.githubusercontent.com/32643842/137022107-a364f97f-4379-4ab4-9d6f-65d181f24ea8.png)
![Results3](https://user-images.githubusercontent.com/32643842/137022174-92edd44a-e74e-4b26-95c3-f298d3a9692a.png)
 
We can see that the top 3 decisions are about 39% different while the last two are just over 40%. The case type of our target document can also be displayed. It is based on the LDA model that we saw in the Engine Analysis slide and we can see that our case type is over 80% in the Medical Malpractice category.

## Next Steps

There are many ways to improve the engine given enough time and resources. The quickest way, like with any model, is to feed it more data. More data means more accuracy and more chance to capture information potentially useful to the user. While this example is geared towards the legal field, the engine can be very simply trained on any data.
 
Another way to maximize the accuracy of the engine is to train a neural network vectorizer on the data the engine will be trained on. This requires a bit of setup but is very achievable, especially with larger datasets.

Finally, working with an expert in the field of the training dataset would allow for more granular and meaningful analysis. Docummend comes with pre and post modeling analytics built in, such as the LDA clustering we saw in the analysis slide, that someone who is well-versed in the field can easily use for analysis or deeper classification of the search results.

