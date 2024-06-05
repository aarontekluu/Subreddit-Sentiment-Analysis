import os
import requests
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
from sklearn.ensemble import IsolationForest
import numpy as np
import plotly.express as px

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Helvetica Neue', sans-serif;
        background-color: #f5f5f5;
    }
    .reportview-container .main .block-container{
        padding: 1rem;
        background: #ffffff;
    }
    .css-18e3th9 {
        padding: 1rem;
    }
    .stMarkdown p {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .reddit-post {
        border: 1px solid #e1e4e8;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background: #fff;
    }
    .reddit-post-title {
        font-size: 18px;
        font-weight: bold;
        color: #0079d3;
        text-decoration: none;
    }
    .reddit-post-meta {
        font-size: 12px;
        color: #878a8c;
    }
    </style>
    """, unsafe_allow_html=True)

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

# Fetch Data for the Past Two Weeks
def fetch_data_past_two_weeks(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.new(limit=None):
        if (pd.Timestamp.now() - pd.to_datetime(post.created_utc, unit='s')).days <= 14:
            author_name = post.author.name if post.author else 'Unknown'
            post_url = f"https://www.reddit.com{post.permalink}"
            posts.append([post.title, post.selftext, post.score, post.num_comments, author_name, post.created_utc, post_url])
    df = pd.DataFrame(posts, columns=['Title', 'Text', 'Score', 'NumComments', 'Author', 'CreatedUTC', 'URL'])
    df['Date'] = pd.to_datetime(df['CreatedUTC'], unit='s')
    return df

# Fetch Top 3 Most Commented Posts of the Past Week
def fetch_top_commented_posts(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.top(time_filter='week', limit=100):
        author_name = post.author.name if post.author else 'Unknown'
        post_url = f"https://www.reddit.com{post.permalink}"
        posts.append([post.title, post.selftext, post.score, post.num_comments, author_name, post.created_utc, post_url])
    df = pd.DataFrame(posts, columns=['Title', 'Text', 'Score', 'NumComments', 'Author', 'CreatedUTC', 'URL'])
    df['Date'] = pd.to_datetime(df['CreatedUTC'], unit='s')
    top_commented = df.nlargest(3, 'NumComments')
    return top_commented

# Feature Engineering for Bot Detection
def extract_features(data):
    data['post_frequency'] = data.groupby('Author')['Title'].transform('count')
    data['avg_post_interval'] = data.groupby('Author')['Date'].diff().dt.total_seconds().fillna(0).mean()
    data['content_similarity'] = data.groupby('Author')['Text'].transform(lambda x: x.duplicated().sum())
    data['account_age'] = (pd.Timestamp.now() - data['Date']).dt.days
    return data[['Author', 'post_frequency', 'avg_post_interval', 'content_similarity', 'account_age']].drop_duplicates()

# Detect Potential Bot Activity
def detect_bots(features):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(features[['post_frequency', 'avg_post_interval', 'content_similarity', 'account_age']])
    features['anomaly'] = iso_forest.predict(features[['post_frequency', 'avg_post_interval', 'content_similarity', 'account_age']])
    
    # Classify bot likelihood
    high_freq_threshold = features['post_frequency'].quantile(0.75)
    low_freq_threshold = features['post_frequency'].quantile(0.25)
    
    conditions = [
        (features['anomaly'] == -1) & (features['post_frequency'] > high_freq_threshold), # High post frequency
        (features['anomaly'] == -1) & (features['post_frequency'] <= high_freq_threshold) & (features['post_frequency'] > low_freq_threshold), # Medium post frequency
        (features['anomaly'] == -1) & (features['post_frequency'] <= low_freq_threshold), # Low post frequency
        (features['anomaly'] == 1) # Not an anomaly
    ]
    choices = ['Highly Likely', 'Maybe', 'Unlikely', 'Unlikely']
    features['bot_likelihood'] = np.select(conditions, choices, default='Unlikely')
    
    return features

# Main Function to Run the Analysis
def run_analysis():
    # Fetch data without displaying messages
    data = fetch_data('uniswap', limit=1000)
    data_past_two_weeks = fetch_data_past_two_weeks('uniswap')
    top_commented_posts = fetch_top_commented_posts('uniswap')

    # Export data to CSV
    data.to_csv('reddit_data.csv', index=False)

    # Convert CreatedUTC to datetime
    data['Date'] = pd.to_datetime(data['CreatedUTC'], unit='s')
    data_past_two_weeks['Date'] = pd.to_datetime(data_past_two_weeks['CreatedUTC'], unit='s')

    # Extract Features and Detect Bots
    features = extract_features(data)
    features = detect_bots(features)
    potential_bots = features[features['anomaly'] == -1]

    # Post Activity (Number of Posts per Day in the Past Two Weeks)
    st.subheader('Post Activity (Number of Posts per Day in the Past Two Weeks)')
    st.write('**Based on the number of posts created per day in the Uniswap subreddit over the past two weeks.**')
    posts_per_day = data_past_two_weeks.groupby(data_past_two_weeks['Date'].dt.date).size()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=posts_per_day.index, y=posts_per_day.values, color='#ff007a')
    plt.title('Post Activity (Number of Posts per Day in the Past Two Weeks)')
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Comment Activity per Day in the Past Two Weeks
    st.subheader('Top Active Users by Post Count in the Past Two Weeks')
    st.write('**Based on the number of posts created by each user in the Uniswap subreddit over the past two weeks.**')
    top_users = data_past_two_weeks['Author'].value_counts().head(10)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top_users.values, y=top_users.index, palette=['#ff007a'] * len(top_users))
    plt.title('Top Active Users by Post Count in the Past Two Weeks')
    plt.xlabel('Number of Posts')
    plt.ylabel('User')
    st.pyplot(plt)

    # Potential Bot Activity
    st.subheader('Potential Bot Activity')
    st.write('**Determined by high post frequency, regular post intervals, and repetitive content. Based on analysis of posting behavior and content similarity.**')
    total_bots = potential_bots.shape[0]
    total_sample_size = features.shape[0]
    st.write(f'Total Number of Potential Bots: {total_bots} out of {total_sample_size} posts analyzed.')

    # Bot Likelihood Pie Chart
    bot_counts = potential_bots['bot_likelihood'].value_counts()
    fig = px.pie(values=bot_counts, names=bot_counts.index, title='Bot Likelihood Distribution', color_discrete_sequence=['#ff007a', '#ff9999', '#ffc0cb'])
    st.plotly_chart(fig)

    # Top 3 Most Commented Posts of the Past Week
    st.subheader('Top 3 Most Commented Posts of the Past Week')
    st.write('**The top 3 most commented posts in the Uniswap subreddit over the past week.**')
    for index, row in top_commented_posts.iterrows():
        post_html = f"""
        <div class="reddit-post">
            <a class="reddit-post-title" href="{row['URL']}" target="_blank">{row['Title']}</a>
            <div class="reddit-post-meta">by {row['Author']} | {row['NumComments']} comments | Score: {row['Score']}</div>
            <div class="reddit-post-body">{row['Text'][:300]}...</div>
        </div>
        """
        st.markdown(post_html, unsafe_allow_html=True)

if __name__ == '__main__':
    run_analysis()
