import os
import base64
import requests
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from wordcloud import WordCloud
import numpy as np
import plotly.express as px

# Function to load the image
def get_base64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error("Logo file not found. Please ensure the file path is correct.")
        return None

# Function to display the image
def display_logo():
    logo_path = os.getenv('UNISWAP_LOGO_PATH')
    if logo_path:
        logo_base64 = get_base64(logo_path)
        if logo_base64:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center;">
                    <img src="data:image/png;base64,{logo_base64}" style="width: 300px;"/>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.error("Logo path environment variable not set. Please set the UNISWAP_LOGO_PATH environment variable.")

# Fetch Data from Uniswap Subreddit
@st.cache_data(ttl=600)
def fetch_data(subreddit_name, limit=500):
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
@st.cache_data(ttl=600)
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
@st.cache_data(ttl=600)
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
    # Display Uniswap Logo
    display_logo()

    # Fetch data without displaying messages
    data = fetch_data('uniswap', limit=500)
    data_past_two_weeks = fetch_data_past_two_weeks('uniswap')
    top_commented_posts = fetch_top_commented_posts('uniswap')

    # Filter out posts from the Uniswap bot
    bot_username = 'UniswapBot'  # Replace with the actual username of the bot
    data = data[data['Author'] != bot_username]
    data_past_two_weeks = data_past_two_weeks[data_past_two_weeks['Author'] != bot_username]

    # Export data to CSV
    data.to_csv('reddit_data.csv', index=False)

    # Convert CreatedUTC to datetime
    data['Date'] = pd.to_datetime(data['CreatedUTC'], unit='s')
    data_past_two_weeks['Date'] = pd.to_datetime(data_past_two_weeks['CreatedUTC'], unit='s')

    # Calculate Baseline Engagement (Average Comments per Week)
    data['Week'] = data['Date'].dt.isocalendar().week
    baseline_comments_per_week = data.groupby('Week')['NumComments'].mean().mean()

    # Calculate Current Week's Engagement
    current_week = data['Week'].max()
    current_week_comments = data[data['Week'] == current_week]['NumComments'].sum()

    # Determine Sentiment Color
    sentiment_color = ''
    if current_week_comments > baseline_comments_per_week * 1.1:
        sentiment_color = 'green'
        sentiment_description = "This indicates a significant increase in engagement (more than 10% above the baseline)."
    elif baseline_comments_per_week * 0.9 <= current_week_comments <= baseline_comments_per_week * 1.1:
        sentiment_color = 'yellow'
        sentiment_description = "This indicates a stable engagement level (within 10% of the baseline)."
        # Display Sentiment Analysis
    st.markdown(f"""
        <div style="background-color:{sentiment_color};padding:10px;border-radius:5px;">
            <h2 style="color:white;text-align:center;">Weekly Engagement: {sentiment_color.capitalize()}</h2>
            <p style="color:white;text-align:center;">
                Based on the average comments per week ({baseline_comments_per_week:.2f} comments), the engagement this week is {sentiment_color.capitalize()}.
                {sentiment_description}
            </p>
        </div>
        """, unsafe_allow_html=True)

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
    st.subheader('Comment Activity per Day in the Past Two Weeks')
    st.write('**Based on the number of comments on posts created per day in the Uniswap subreddit over the past two weeks.**')
    comments_per_day = data_past_two_weeks.groupby(data_past_two_weeks['Date'].dt.date)['NumComments'].sum()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=comments_per_day.index, y=comments_per_day.values, color='#ff007a')
    plt.title('Comment Activity per Day in the Past Two Weeks')
    plt.xlabel('Date')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Top Active Users by Post Count in the Past Two Weeks
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
    st.write('**Comments are often a better proxy for popularity than upvotes because they indicate active engagement and discussions within the community. Here are the top 3 most commented posts in the Uniswap subreddit over the past week.**')
    for index, row in top_commented_posts.iterrows():
        post_html = f"""
        <div class="reddit-post">
            <a class="reddit-post-title" href="{row['URL']}" target="_blank">{row['Title']}</a>
            <div class="reddit-post-meta">by {row['Author']} | {row['NumComments']} comments | Score: {row['Score']}</div>
            <div class="reddit-post-body">{row['Text'][:300]}...</div>
        </div>
        """
        st.markdown(post_html, unsafe_allow_html=True)

    # Explanation of why comments are a better proxy for engagement than upvotes
    st.subheader('Why Comments are a Better Proxy for Engagement than Upvotes')
    st.write("""
        **Comments provide a more accurate measure of engagement because:**
        - **Active Participation**: Comments require more effort than upvotes and reflect active participation and discussion within the community.
        - **Depth of Interaction**: Comments often provide insights, feedback, and opinions, showcasing deeper interactions.
        - **Bot Mitigation**: Upvotes can be easily manipulated by bots, while comments (especially longer ones) are harder to automate convincingly.
    """)

if __name__ == '__main__':
    run_analysis()

