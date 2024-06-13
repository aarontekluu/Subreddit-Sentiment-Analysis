import os
import requests
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
from time import sleep

# Load secrets from Streamlit
REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
REDDIT_USER_AGENT = st.secrets["REDDIT_USER_AGENT"]
REDDIT_USERNAME = st.secrets["REDDIT_USERNAME"]
REDDIT_PASSWORD = st.secrets["REDDIT_PASSWORD"]

# Authenticate to Reddit
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)

def safe_reddit_request(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except prawcore.exceptions.RateLimitExceeded as e:
            sleep_time = int(re.search(r'\d+', str(e)).group())
            st.warning(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
            sleep(sleep_time)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            break

def fetch_data(subreddit_name, limit=1000):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in safe_reddit_request(subreddit.top, limit=limit):
        author_name = post.author.name if post.author else 'Unknown'
        post_url = f"https://www.reddit.com{post.permalink}"
        posts.append([post.title, post.selftext, post.score, post.num_comments, author_name, post.created_utc, post_url])
    df = pd.DataFrame(posts, columns=['Title', 'Text', 'Score', 'NumComments', 'Author', 'CreatedUTC', 'URL'])
    df['Date'] = pd.to_datetime(df['CreatedUTC'], unit='s')
    return df

def fetch_data_past_two_weeks(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in safe_reddit_request(subreddit.new, limit=None):
        if (pd.Timestamp.now() - pd.to_datetime(post.created_utc, unit='s')).days <= 14:
            author_name = post.author.name if post.author else 'Unknown'
            post_url = f"https://www.reddit.com{post.permalink}"
            posts.append([post.title, post.selftext, post.score, post.num_comments, author_name, post.created_utc, post_url])
    df = pd.DataFrame(posts, columns=['Title', 'Text', 'Score', 'NumComments', 'Author', 'CreatedUTC', 'URL'])
    df['Date'] = pd.to_datetime(df['CreatedUTC'], unit='s')
    return df

def fetch_top_commented_posts(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in safe_reddit_request(subreddit.top, time_filter='week', limit=100):
        author_name = post.author.name if post.author else 'Unknown'
        post_url = f"https://www.reddit.com{post.permalink}"
        posts.append([post.title, post.selftext, post.score, post.num_comments, author_name, post.created_utc, post_url])
    df = pd.DataFrame(posts, columns=['Title', 'Text', 'Score', 'NumComments', 'Author', 'CreatedUTC', 'URL'])
    df['Date'] = pd.to_datetime(df['CreatedUTC'], unit='s')
    top_commented = df.nlargest(3, 'NumComments')
    return top_commented

def fetch_popular_questions(subreddit_name, time_filter='week', limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in safe_reddit_request(subreddit.top, time_filter=time_filter, limit=limit):
        if '?' in post.title:
            author_name = post.author.name if post.author else 'Unknown'
            post_url = f"https://www.reddit.com{post.permalink}"
            posts.append([post.title, post.num_comments, post_url])
    df = pd.DataFrame(posts, columns=['Question', 'Number of Comments', 'URL'])
    return df.sort_values(by='Number of Comments', ascending=False).head(10)

def run_analysis():
    data = fetch_data('uniswap', limit=500)
    data_past_two_weeks = fetch_data_past_two_weeks('uniswap')
    top_commented_posts = fetch_top_commented_posts('uniswap')
    popular_questions_week = fetch_popular_questions('uniswap', time_filter='week')
    popular_questions_month = fetch_popular_questions('uniswap', time_filter='month')

    data.to_csv('reddit_data.csv', index=False)

    data['Date'] = pd.to_datetime(data['CreatedUTC'], unit='s')
    data_past_two_weeks['Date'] = pd.to_datetime(data_past_two_weeks['CreatedUTC'], unit='s')

    st.title("r/Uniswap Dashboard")

    st.header('Metrics on the Most Asked Questions in r/Uniswap')
    
    st.subheader('Top Questions of the Week')
    st.write('**The most popular questions asked on the Uniswap subreddit in the past week. A good metric to use to determine how to support the subreddit.**')
    popular_questions_week['Question'] = popular_questions_week.apply(lambda x: f'<a href="{x.URL}" target="_blank">{x.Question}</a>', axis=1)
    popular_questions_week = popular_questions_week[['Question', 'Number of Comments']]
    st.write(popular_questions_week.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.subheader('Top Questions of the Month')
    st.write('**The most popular questions asked on the Uniswap subreddit in the past month. A good metric to use to determine how to support the subreddit.**')
    popular_questions_month['Question'] = popular_questions_month.apply(lambda x: f'<a href="{x.URL}" target="_blank">{x.Question}</a>', axis=1)
    popular_questions_month = popular_questions_month[['Question', 'Number of Comments']]
    st.write(popular_questions_month.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.header('Other Community Engagement Metrics')

    st.subheader('Post Activity (Number of Posts per Day in the Past Two Weeks)')
    st.write('**The number of posts posted per day in the subreddit over the past two weeks.**')
    posts_per_day = data_past_two_weeks.groupby(data_past_two_weeks['Date'].dt.date).size()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=posts_per_day.index, y=posts_per_day.values, color='#ff007a')
    plt.title('Post Activity (Number of Posts per Day in the Past Two Weeks)')
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.subheader('Comment Activity per Day in the Past Two Weeks')
    st.write('**The comment activity per day in the subreddit over the past two weeks.**')
    comments_per_day = data_past_two_weeks.groupby(data_past_two_weeks['Date'].dt.date)['NumComments'].sum()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=comments_per_day.index, y=comments_per_day.values, color='#ff007a')
    plt.title('Comment Activity per Day in the Past Two Weeks')
    plt.xlabel('Date')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.subheader('Top Active Users on r/Uniswap in the Past Two Weeks')
    st.write('**The top active users based on the number of posts created by each user in the subreddit over the past two weeks.**')
    top_users = data_past_two_weeks['Author'].value_counts().head(10)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top_users.values, y=top_users.index, palette=['#ff007a'] * len(top_users))
    plt.title('Top Active Users by Post Count in the Past Two Weeks')
    plt.xlabel('Number of Posts')
    plt.ylabel('User')
    st.pyplot(plt)

    st.subheader('Average Upvotes per Post in the Past Two Weeks')
    st.write('**This metric shows the average upvotes of posts per day in the Uniswap subreddit over the past two weeks**')
    avg_score_per_day = data_past_two_weeks.groupby(data_past_two_weeks['Date'].dt.date)['Score'].mean()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=avg_score_per_day.index, y=avg_score_per_day.values, color='#ff007a')
    plt.title('Average Score per Post in the Past Two Weeks')
    plt.xlabel('Date')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    st.pyplot(plt)

if __name__ == '__main__':
    run_analysis()
