import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import praw
import datetime
import time

# load roBERTa
device = torch.device("cuda")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# function to perform sentiment analysis on a single text string
def analyze_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
    return probs[1]  # return the probability of a positive sentimentc

# Define a function to run sentiment analysis on multiple Reddit posts and comments on the GPU
def analyze_reddit_sentiment(subs, ticker, company_name, time_period):
    reddit = praw.Reddit(client_id='', client_secret='', user_agent='')
    sentiments = []
    num_posts = 0
    num_comments = 0
    start_time = time.time()
    for sub in subs:
        subreddit = reddit.subreddit(sub)
        print ("Running sentiment analysis on subreddit: " + sub)
        for post in subreddit.search(ticker + " OR " + company_name, time_filter=time_period, limit=10):
            post_sentiment = analyze_sentiment(post.selftext)
            print("Post sentiment: " + str(post_sentiment) + ", Link: " + "https://www.reddit.com"+post.permalink)
            sentiments.append(post_sentiment)
            num_posts += 1
            post.comments.replace_more(limit=10)
            for comment in post.comments.list():
                comment_sentiment = analyze_sentiment(comment.body)
                sentiments.append(comment_sentiment)
                num_comments += 1
        print(f"Number of posts analyzed in r/{sub}: {num_posts}")
        print(f"Number of comments analyzed in r/{sub}: {num_comments}")
    end_time = time.time()
    time_taken = end_time - start_time  
    print(f"\nTotal time taken: {time_taken:.2f} seconds")  

    avg_sentiment = sum(sentiments) / len(sentiments)
    return avg_sentiment

# Example usage
subs = input("Enter subreddit(s) separated by comma: ").replace(" ", "").split(",")
ticker = input("Enter ticker symbol: ")
company_name = input("Enter company name: ")
time_period = input("Enter time period: hour, day, week, month, year, all: ")
sentiment = analyze_reddit_sentiment(subs, ticker, company_name, time_period)
print('Sentiment:', sentiment)
