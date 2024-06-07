import os
import requests
import praw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

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
    .engagement-box {
        font-size: 1.5em;
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

# Fetch Most Popular Questions from the Past Month
def fetch_popular_questions(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.top(time_filter='month', limit=1000):
        if '?' in post.title:
            author_name = post.author.name if post.author else 'Unknown'
            post_url = f"https://www.reddit.com{post.permalink}"
            posts.append([post.title, post.num_comments, post_url])
    df = pd.DataFrame(posts, columns=['Question', 'NumComments', 'URL'])
    return df.sort_values(by='NumComments', ascending=False).head(10)

# Main Function to Run the Analysis
def run_analysis():
    # Fetch data without displaying messages
    data = fetch_data('uniswap', limit=500)
    data_past_two_weeks = fetch_data_past_two_weeks('uniswap')
    top_commented_posts = fetch_top_commented_posts('uniswap')
    popular_questions = fetch_popular_questions('uniswap')

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
    else:
        sentiment_color = 'red'
        sentiment_description = "This indicates a significant decrease in engagement (more than 10% below the baseline)."

    # Display Sentiment Analysis
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
    
    # Weekly Engagement Spreadsheet
    st.subheader('Weekly Engagement Spreadsheet')
    st.write('**A spreadsheet showing the engagement (comment frequency) over time on a weekly basis.**')
    weekly_comments = data.groupby('Week')['NumComments'].sum().reset_index()
    st.dataframe(weekly_comments)

    # Most Popular Questions
    st.subheader('Most Popular Questions on the Subreddit (Past Month)')
    st.write('**A spreadsheet showing the most popular questions asked on the Uniswap subreddit in the past month.**')
    popular_questions['Question'] = popular_questions.apply(lambda x: f'<a href="{x.URL}" target="_blank">{x.Question}</a>', axis=1)
    popular_questions = popular_questions[['Question', 'NumComments']]
    st.write(popular_questions.to_html(escape=False), unsafe_allow_html=True)

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

if __name__ == '__main__':
    run_analysis()
