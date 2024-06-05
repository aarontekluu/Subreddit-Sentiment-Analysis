import os
import requests
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
import numpy as np
import plotly.express as px

# Fetch environment variables
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USERNAME = os.getenv('REDDIT_USERNAME')
REDDIT_PASSWORD = os.getenv('REDDIT_PASSWORD')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

# Step 1: Authenticate and get access token
auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
data = {'grant_type': 'password', 'username': REDDIT_USERNAME, 'password': REDDIT_PASSWORD}
headers = {'User-Agent': REDDIT_USER_AGENT}

res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
if res.status_code == 200:
    token = res.json()['access_token']
    print("Access token:", token)
else:
    print("Failed to get access token:", res.json())
    exit()

# Step 2: Use the access token with PRAW
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)

# Fetch Data from Uniswap Subreddit
def fetch_data(subreddit_name, limit=1000):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.top(limit=limit):
        author_name = post.author.name if post.author else 'Unknown'
        post_url = f"https://www.reddit.com{post.permalink}"
        posts.append([post.title, post.selftext, post.score, post.num_comments, author_name, post.created_utc, post_url])
    df = pd.DataFrame(posts, columns=['Title', 'Text', 'Score', 'NumComments', 'Author', 'CreatedUTC', 'URL'])
    df['Date'] = pd.to_datetime(df['CreatedUTC'], unit='s')
    return df

# Fetch Data for a Specific Time Period
def fetch_data_period(subreddit_name, time_filter='week', limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.top(time_filter=time_filter, limit=limit):
        author_name = post.author.name if post.author else 'Unknown'
        post_url = f"https://www.reddit.com{post.permalink}"
        posts.append([post.title, post.selftext, post.score, post.num_comments, author_name, post.created_utc, post_url])
    df = pd.DataFrame(posts, columns=['Title', 'Text', 'Score', 'NumComments', 'Author', 'CreatedUTC', 'URL'])
    df['Date'] = pd.to_datetime(df['CreatedUTC'], unit='s')
    return df

# Feature Engineering for Bot Detection
def extract_features(data):
    data['post_frequency'] = data.groupby('Author')['Title'].transform('count')
    data['avg_post_interval'] = data.groupby('Author')['Date'].diff().dt.total_seconds().fillna(0).mean()
    data['content_similarity'] = data.groupby('Author')['Text'].transform(lambda x: x.duplicated().sum())
    data['account_age'] = (pd.Timestamp.now() - data['Date']).dt.days
    return data[['Author', 'post_frequency', 'avg_post_interval', 'content_similarity', 'account_age']].drop_duplicates()

# Detect Potential Bot Activity
def detect_bots(features):
    iso_forest = IsolationForest(contamination=0.1)
    iso_forest.fit(features[['post_frequency', 'avg_post_interval', 'content_similarity', 'account_age']])
    features['anomaly'] = iso_forest.predict(features[['post_frequency', 'avg_post_interval', 'content_similarity', 'account_age']])
    
    # Classify bot likelihood
    conditions = [
        (features['anomaly'] == -1) & (features['post_frequency'] > features['post_frequency'].quantile(0.75)),
        (features['anomaly'] == -1) & (features['post_frequency'] <= features['post_frequency'].quantile(0.75)),
        (features['anomaly'] == 1)
    ]
    choices = ['Highly Likely', 'Maybe', 'Unlikely']
    features['bot_likelihood'] = np.select(conditions, choices, default='Unlikely')
    
    return features

# Main Function to Run the Analysis
def run_analysis():
    st.title('Uniswap Subreddit Analyzer')
    
    # Sidebar
    st.sidebar.title('Data Fetching')
    data_limit = st.sidebar.slider('Select number of posts to fetch', 100, 2000, 1000)
    
    st.subheader('Fetching data...')
    data = fetch_data('uniswap', limit=data_limit)
    data_week = fetch_data_period('uniswap', time_filter='week')
    data_month = fetch_data_period('uniswap', time_filter='month')
    st.write('Data fetched successfully!')

    # Export data to CSV
    data.to_csv('reddit_data.csv', index=False)
    print("Data exported to reddit_data.csv")

    # Convert CreatedUTC to datetime
    data['Date'] = pd.to_datetime(data['CreatedUTC'], unit='s')
    data['Hour'] = data['Date'].dt.hour
    data['Month'] = data['Date'].dt.to_period('M')

    # Extract Features and Detect Bots
    features = extract_features(data)
    features = detect_bots(features)
    potential_bots = features[features['anomaly'] == -1]

    # Post Activity: Number of Posts Per Day in the Past Two Weeks
    st.subheader('Post Activity: Number of Posts Per Day in the Past Two Weeks')
    st.write('**Based on the number of posts made each day in the past two weeks.**')
    past_two_weeks = data[data['Date'] > (pd.Timestamp.now() - pd.Timedelta(weeks=2))]
    posts_per_day = past_two_weeks.groupby(past_two_weeks['Date'].dt.date).size()
    plt.figure(figsize=(10, 4))
    posts_per_day.plot(kind='line')
    plt.title('Number of Posts Per Day in the Past Two Weeks')
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    st.pyplot(plt)

    # Post Activity: Number of Comments Per Day in the Past Two Weeks
    st.subheader('Post Activity: Number of Comments Per Day in the Past Two Weeks')
    st.write('**Based on the total number of comments made on posts each day in the past two weeks.**')
    comments_per_day = past_two_weeks.groupby(past_two_weeks['Date'].dt.date)['NumComments'].sum()
    plt.figure(figsize=(10, 4))
    comments_per_day.plot(kind='line')
    plt.title('Number of Comments Per Day in the Past Two Weeks')
    plt.xlabel('Date')
    plt.ylabel('Number of Comments')
    st.pyplot(plt)

    # Post Activity: Average Score of Posts Per Day in the Past Two Weeks
    st.subheader('Post Activity: Average Score of Posts Per Day in the Past Two Weeks')
    st.write('**Based on the average score of posts each day in the past two weeks.**')
    avg_score_per_day = past_two_weeks.groupby(past_two_weeks['Date'].dt.date)['Score'].mean()
    plt.figure(figsize=(10, 4))
    avg_score_per_day.plot(kind='line')
    plt.title('Average Score of Posts Per Day in the Past Two Weeks')
    plt.xlabel('Date')
    plt.ylabel('Average Score')
    st.pyplot(plt)

    # Potential Bot Activity
    st.subheader('Potential Bot Activity')
    st.write('**Determined by high post frequency, regular post intervals, and repetitive content.**')
    total_bots = potential_bots.shape[0]
    total_sample_size = features.shape[0]
    st.write(f'Total Number of Potential Bots: {total_bots} out of {total_sample_size} posts analyzed.')

    # Bot Likelihood Pie Chart
    bot_counts = potential_bots['bot_likelihood'].value_counts()
    fig = px.pie(values=bot_counts, names=bot_counts.index, title='Bot Likelihood Distribution')
    st.plotly_chart(fig)

if __name__ == '__main__':
    run_analysis()
