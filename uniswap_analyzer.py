import os
import tweepy
import requests
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

# Load secrets from Streamlit
TWITTER_API_KEY = st.secrets["TWITTER_API_KEY"]
TWITTER_API_SECRET_KEY = st.secrets["TWITTER_API_SECRET_KEY"]
TWITTER_BEARER_TOKEN = st.secrets["TWITTER_BEARER_TOKEN"]
TWITTER_ACCESS_TOKEN = st.secrets["TWITTER_ACCESS_TOKEN"]
TWITTER_ACCESS_TOKEN_SECRET = st.secrets["TWITTER_ACCESS_TOKEN_SECRET"]
REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]
REDDIT_USER_AGENT = st.secrets["REDDIT_USER_AGENT"]
REDDIT_USERNAME = st.secrets["REDDIT_USERNAME"]
REDDIT_PASSWORD = st.secrets["REDDIT_PASSWORD"]

# Authenticate to Twitter (Tweepy V2)
client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

def fetch_tweets_v2(username, count=5):
    try:
        user = client.get_user(username=username, user_auth=True)
        user_id = user.data.id
        tweets = client.get_users_tweets(id=user_id, max_results=count, tweet_fields=['id', 'text'], user_auth=True)
        return [{'text': tweet.text, 'url': f"https://twitter.com/{username}/status/{tweet.id}"} for tweet in tweets.data]
    except Exception as e:
        st.error(f"Failed to fetch tweets for {username}: {e}")
        return []

# Authenticate to Reddit
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD
)

def post_to_reddit(subreddit, title, url):
    reddit.subreddit(subreddit).submit(title, url=url)

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

def fetch_popular_questions(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.top(time_filter='month', limit=1000):
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
    popular_questions = fetch_popular_questions('uniswap')

    data.to_csv('reddit_data.csv', index=False)

    data['Date'] = pd.to_datetime(data['CreatedUTC'], unit='s')
    data_past_two_weeks['Date'] = pd.to_datetime(data_past_two_weeks['CreatedUTC'], unit='s')

    data['Week'] = data['Date'].dt.isocalendar().week
    baseline_comments_per_week = data.groupby('Week')['NumComments'].mean().mean()

    current_week = data['Week'].max()
    current_week_comments = data[data['Week'] == current_week]['NumComments'].sum()

    sentiment_color = ''
    if current_week_comments > baseline_comments_per_week * 1.1:
        sentiment_color = 'green'
        sentiment_description = "This indicates a significant increase in engagement (more than 10% above the baseline)."
    elif baseline_comments_per_week * 0.9 <= current_week_comments <= baseline_comments_per_week * 1.1:
        sentiment_color = 'yellow'
        sentiment_description = "This indicates a stable engagement level (within 10% of the baseline)."
    else:
        sentiment_color = 'red'
        sentiment_description = "This indicates a significant decrease in engagement (more than 10% below the baseline)."

    st.markdown(f"""
        <div style="background-color:{sentiment_color};padding:10px;border-radius:5px;" class="engagement-box">
            <p style="color:white;text-align:center;">
                Based on the average comments per week ({baseline_comments_per_week:.2f} comments), the engagement this week is {sentiment_color.capitalize()}.
                {sentiment_description}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    **Engagement color indicators:**
    - **Green**: Increase in engagement (more than 10% above the baseline)
    - **Yellow**: Stable engagement (within 10% of the baseline)
    - **Red**: Decrease in engagement (more than 10% below the baseline)
    """)

    st.subheader('Most Popular Questions on the Subreddit (Past Month)')
    st.write('**A spreadsheet showing the most popular questions asked on the Uniswap subreddit in the past month.**')
    popular_questions['Question'] = popular_questions.apply(lambda x: f'<a href="{x.URL}" target="_blank">{x.Question}</a>', axis=1)
    popular_questions = popular_questions[['Question', 'Number of Comments']]
    st.write(popular_questions.to_html(escape=False, index=False), unsafe_allow_html=True)

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

    st.subheader('Top Active Users by Post Count in the Past Two Weeks')
    st.write('**Based on the number of posts created by each user in the Uniswap subreddit over the past two weeks.**')
    top_users = data_past_two_weeks['Author'].value_counts().head(10)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top_users.values, y=top_users.index, palette=['#ff007a'] * len(top_users))
    plt.title('Top Active Users by Post Count in the Past Two Weeks')
    plt.xlabel('Number of Posts')
    plt.ylabel('User')
    st.pyplot(plt)

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

    # Display recent tweets and provide option to repost them
    st.subheader('Recent Tweets from Uniswap, Uniswap Foundation, and Hayden Adams')
    st.write('**Below are the most recent tweets fetched from the specified Twitter accounts. You can choose to repost these tweets to the Uniswap subreddit.**')

    usernames = ['Uniswap', 'UniswapFND', 'haydenzadams']
    recent_tweets = []
    for username in usernames:
        tweets = fetch_tweets_v2(username)
        recent_tweets.extend(tweets)

    for tweet in recent_tweets:
        tweet_html = f"""
        <div class="reddit-post">
            <a class="reddit-post-title" href="{tweet['url']}" target="_blank">{tweet['text']}</a>
        </div>
        """
        st.markdown(tweet_html, unsafe_allow_html=True)

    if st.button('Fetch and Post Tweets'):
        for tweet in recent_tweets:
            post_to_reddit('uniswap', tweet['text'], tweet['url'])
        st.write('Tweets fetched and posted successfully!')

if __name__ == '__main__':
    run_analysis()
