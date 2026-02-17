import pandas as pd
import re
import numpy as np
from urlextract import URLExtract
import matplotlib.pyplot as plt
import streamlit as st
import emoji
from wordcloud import WordCloud , STOPWORDS
import plotly.express as px
import seaborn as sns

url_extracter = URLExtract()


def fetch_datas(selected_user , df):
    if selected_user == "Overall":
        message_len = df.shape[0]
        char_len = df['message'].apply(lambda x : len(x)).sum()
        media_omit = df[df['message'] == "<Media omitted>"].shape[0]
        url_count = df['message'].apply(lambda x: len(url_extracter.find_urls(x))).sum()
        percentage = 100.0
        return message_len , char_len , media_omit , url_count , percentage
    else:
        user_message_len = df[df['user'] == selected_user].shape[0]
        user_char_len = df[df['user'] == selected_user]['message'].apply(lambda x : len(x)).sum()
        user_media_omit = df[(df['user'] == selected_user) & (df['message'] == "<Media omitted>")].shape[0]
        user_url_count = df[df['user'] == selected_user]['message'].apply(lambda x: len(url_extracter.find_urls(x))).sum()
        user_percen_messag = round(df[df['user'] == selected_user].shape[0] / df.shape[0] * 100, 2)
        return user_message_len , user_char_len , user_media_omit , user_url_count , user_percen_messag

def active_user(df , n) -> pd.DataFrame:
    active_user = df['user'].value_counts().head(n).reset_index()
    active_user.columns = ["User" , "Message Count"]
    active_user = active_user.sort_values(by = "Message Count", ascending=False)
    message_percentage = (active_user['Message Count'] / df.shape[0]) * 100
    active_user['Percentage(%)'] = message_percentage.round(2)
    name = active_user['User'].tolist()
    count = active_user['Message Count'].tolist()
    plt.figure(figsize=(10, 6))
    plt.bar(name, count, color='skyblue')
    plt.xlabel('User')
    plt.ylabel('Message Count')
    plt.title(f'Top {n} Active Users')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    return active_user

def mostusesd_words(df, selected_user):

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    df = df[df['message'] != "<Media omitted>"]
    df = df[df['message'] != "This message was deleted"]

    df['message'] = df['message'].astype(str)

    # remove URLs
    df['message'] = df['message'].apply(
        lambda x: re.sub(r'http[s]?://\S+', '', x)
    )

    df['message'] = df['message'].apply(
        lambda x: re.sub(r'[^\w\s]', '', x)
    )

    stopwords = set(STOPWORDS)
    stopwords.update({"hai", "kya", "ho", "ka", "ki", "ke"})

    words = ' '.join(df['message']).strip()

    if not words:
        st.write("No words to display in the word cloud.")
        return

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords
    ).generate(words)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    st.pyplot(fig)

from collections import Counter
import emoji
import pandas as pd

def emoji_helper(selected_user, df) -> pd.DataFrame:

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    emojis_list = []

    for message in df['message']:
        for char in str(message):
            if emoji.is_emoji(char):
                emojis_list.append(char)

    if not emojis_list:
        return pd.DataFrame(columns=["Emoji", "Count"])

    emojis_dataframe = pd.DataFrame(
        Counter(emojis_list).most_common(),
        columns=["Emoji", "Count"]
    )

    return emojis_dataframe

import seaborn as sns
import matplotlib.pyplot as plt

def activity_heatmap(selected_user, df):

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    df['original_date'] = pd.to_datetime(df['original_date'])
    df['day_name'] = df['original_date'].dt.day_name()

    # Set correct weekday order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 
                 'Thursday', 'Friday', 'Saturday', 'Sunday']

    df['day_name'] = pd.Categorical(df['day_name'], 
                                     categories=day_order, 
                                     ordered=True)

    heatmap_data = df.pivot_table(
        index='day_name',     # Y-axis
        columns='hour',       # X-axis
        values='message',
        aggfunc='count'
    )



    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='g')

    plt.title(f'Activity Heatmap for {selected_user}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day')

    st.pyplot(plt)

def mostactive_monthly(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    monthly_activity = df.groupby(['year', 'month']).size().reset_index(name='Message Count')
    monthly_activity['date'] = pd.to_datetime(monthly_activity['year'].astype(str) + '-' + monthly_activity['month'].astype(str))
    monthly_activity = monthly_activity.sort_values('date')

    fig = px.line(
        monthly_activity,
        x='date',
        y='Message Count',
        title='Monthly Chat Activity',
        markers=True
    )

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Number of Messages'
    )
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Messages')
    st.plotly_chart(fig, use_container_width=True)