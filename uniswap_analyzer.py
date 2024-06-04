import requests
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from sklearn.ensemble import IsolationForest
import numpy as np

# Step 1: Authenticate and get access token
auth = requests.auth.HTTPBasicAuth('ANte6ker2jgKNVYjACDZ8A', 'rIhtul0uIeY6WU7kDtukvxhYPlc2bA')
data = {'grant_type': 'password', 'username': 'Certain-Dot743', 'password': 'FATTYpatty12345'}
headers = {'User-Agent': 'uniswap-analysis-script/0.1'}

res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
if res.status_code == 200:
    token = res.json()['access_token']
    print("Access token:", token)
else:
    print("Failed to get access token:", res.json())
    exit()

# Step 2: Use the access token with PRAW
reddit = praw.Reddit(
    client_id='ANte6ker2jgKNVYjACDZ8A',
    client_secret='rIhtul0uIeY6WU7kDtukvxhYPlc2bA',
    user_agent='uniswap-analysis-script/0.1',
    username='Certain-Dot743',
    password='FATTYpatty12345'
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

    # Convert CreatedUTC to datetime
    data['Date'] = pd.to_datetime(data['CreatedUTC'], unit='s')
    data['Hour'] = data['Date'].dt.hour
    data['Month'] = data['Date'].dt.to_period('M')

    # Extract Features and Detect Bots
    features = extract_features(data)
    features = detect_bots(features)
    potential_bots = features[features['anomaly'] == -1]

    # Engagement Metrics
    st.subheader('Engagement Metrics')
    st.write('**Average metrics of posts on the Uniswap subreddit.**')
    avg_score = data['Score'].mean()
    avg_comments = data['NumComments'].mean()
    col1, col2 = st.columns(2)
    with col1:
        st.write(f'Average Post Score: {avg_score:.2f}')
    with col2:
        st.write(f'Average Number of Comments per Post: {avg_comments:.2f}')

    # Top 5 Posts
    st.subheader('Top 5 Posts by Score')
    st.write('**The highest scoring posts on the subreddit.**')
    top_posts = data.nlargest(5, 'Score')[['Title', 'Score', 'NumComments', 'Date', 'URL']]
    st.table(top_posts)

    # Top Posts in the Past Week
    st.subheader('Top Posts in the Past Week')
    st.write('**The highest scoring posts in the past week.**')
    top_posts_week = data_week.nlargest(5, 'Score')[['Title', 'Score', 'NumComments', 'Date', 'URL']]
    st.table(top_posts_week)

    # Top Posts in the Past Month
    st.subheader('Top Posts in the Past Month')
    st.write('**The highest scoring posts in the past month.**')
    top_posts_month = data_month.nlargest(5, 'Score')[['Title', 'Score', 'NumComments', 'Date', 'URL']]
    st.table(top_posts_month)

    # Daily Activity
    st.subheader('Daily Activity: Number of Posts and Comments')
    st.write('**The daily number of posts on the subreddit.**')
    daily_activity = data.groupby(data['Date'].dt.date).size()
    plt.figure(figsize=(10, 4))
    daily_activity.plot(kind='line')
    plt.title('Daily Activity: Number of Posts and Comments')
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    st.pyplot(plt)

    # Hourly Activity
    st.subheader('Hourly Activity: Number of Posts')
    st.write('**The distribution of posts throughout the day.**')
    hourly_activity = data['Hour'].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    hourly_activity.plot(kind='bar')
    plt.title('Hourly Activity: Number of Posts')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Posts')
    st.pyplot(plt)

    # Most Active Days
    st.subheader('Most Active Days of the Week')
    st.write('**The most active days for posting on the subreddit.**')
    data['DayOfWeek'] = data['Date'].dt.day_name()
    active_days = data['DayOfWeek'].value_counts()
    st.bar_chart(active_days)

    # Monthly Trends
    st.subheader('Monthly Trends: Number of Posts')
    st.write('**The monthly trend in the number of posts.**')
    monthly_trends = data.groupby(data['Month']).size()
    plt.figure(figsize=(10, 4))
    monthly_trends.plot(kind='line')
    plt.title('Monthly Trends: Number of Posts')
    plt.xlabel('Month')
    plt.ylabel('Number of Posts')
    st.pyplot(plt)

    # User Engagement
    st.subheader('User Engagement')
    st.write('**The top users by average post score and number of comments.**')
    user_engagement = data.groupby('Author').agg({'Score': 'mean', 'NumComments': 'mean'}).nlargest(10, 'Score')
    st.table(user_engagement)

    # Post Score Distribution
    st.subheader('Post Score Distribution')
    st.write('**The distribution of post scores on the subreddit.**')
    plt.figure(figsize=(10, 4))
    sns.histplot(data['Score'], bins=20, kde=False)
    plt.title('Post Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.ylim(0, 700)
    plt.xlim(0, 300)
    plt.xticks(range(0, 301, 50))
    st.pyplot(plt)

    # Word Cloud
    st.subheader('Word Cloud of Most Common Words')
    st.write('**A visual representation of the most common words in post texts.**')
    text = ' '.join(data['Text'].fillna(''))
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Potential Bot Activity
    st.subheader('Potential Bot Activity')
    st.write('**Accounts potentially exhibiting bot-like behavior based on their posting patterns and other metrics.**')
    st.table(potential_bots)

if __name__ == '__main__':
    run_analysis()