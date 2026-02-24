import streamlit as st
import pandas as pd
import re
from analysis import fetch_datas
import analysis
import matplotlib.pyplot as plt
import plotly.express as px
st.title("Whatsapp Chat Analysis")

def preprocessing(raw_text: str) -> pd.DataFrame:
    data = raw_text
    pattern = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)\s-\s"
    pattern_f = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)"
    message = re.split(pattern, data)[1:]
    dates = re.findall(pattern_f, data)
    df = pd.DataFrame({"user_message": message, "message_date": dates})
    df["message_date"] = pd.to_datetime(df["message_date"], format='%d/%m/%Y, %I:%M %p')
    df.rename(columns={"message_date": "date"}, inplace=True)
    df[["user", "message"]] = (
        df["user_message"]
        .str.split(":", n=1, expand=True)
        .apply(lambda x: x.str.strip())
    )
    df.drop(columns=["user_message"], inplace=True)
    df[['original_date' , 'time']] = df['date'].astype(str).str.split(' ' , expand = True)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df.dropna(inplace=True)
    return df

uploaded_media = st.sidebar.file_uploader("Upload your whatsapp chat file", type=["txt"])

def selectuser(df: pd.DataFrame) -> list:
    users = df['user'].dropna().unique().tolist()
    users.sort()
    users.insert(0 , "Overall")
    return users

if uploaded_media is not None:
    chat_text = uploaded_media.getvalue().decode('utf-8')
    df = preprocessing(chat_text)
    st.write("Uploaded chat parsed — showing first rows:")

    start_date = pd.to_datetime((df['date']).min())
    end_date = pd.to_datetime((df['date']).max())

    col1 , col2 = st.columns(2)
    
    col1.metric("Start Time" , str(start_date))
    col2.metric("End Time" , str(end_date))

    df = df.reset_index(drop=True)
    df.drop(columns=["date"] , inplace=True)
    st.dataframe(df.head())
    users = selectuser(df)

    @st.cache_resource
    def load_model():
        model = analysis.HybridToxicitymodel(
            model_a = "cardiffnlp/twitter-xlm-roberta-base-offensive",
            model_b = "l3cube-pune/hinglish-toxic-bert"
        )
        model.eval()
        return model

    toxicity_model = load_model()

    toxicity_dataframe = analysis.toxity_dataframe_creation(df, toxicity_model)

    

    selected_user = st.sidebar.selectbox("Select User" , users)
    st.write(f"Selected user: {selected_user}")

    with st.form("active user form"):
            head_count = st.number_input("Enter the number of top active users to display",
            min_value=1,
            max_value=len(users),
            value=5)

            submit = st.form_submit_button("Submit")

    if submit:
        col1 , col2 = st.columns(2)
        with col1:
            st.write(f"Top {head_count} Active Users:")
            active_users = analysis.active_user(df , n=head_count)
        with col2:
            st.write(f"Active Users Top {head_count}:" , active_users)

    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    if st.sidebar.button("Show Analysis"):
        st.session_state.show_analysis = True


    if st.session_state.show_analysis:

        col1 , col2  = st.columns(2)
        total_messages = fetch_datas(selected_user , df)
        with col1:
            st.header("All Messages")
            st.title(total_messages[0])

        with col2:
            st.header("All Characters")
            st.title(total_messages[1])

        col3 , col4 = st.columns(2)
        with col3:
            st.header("Media Omitted")
            st.title(total_messages[2])

        with col4:
            st.header("Link sended")
            st.title(total_messages[3])
    
        st.header("User Percentage of messages")
        st.title(f"{selected_user} : {total_messages[4]}%")

        st.header(f"Most Used Words in {selected_user}'s Messages", anchor=None)
        most_used_words = analysis.mostusesd_words(df, selected_user)

        emoji_df = analysis.emoji_helper(selected_user, df)

        st.header(f"Emoji Analysis for {selected_user}", anchor=None)

        if emoji_df.empty:
            st.warning("No emojis found for this user.")
        else:
            top_emoji = emoji_df.head(10)

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(top_emoji)

            with col2:
                fig = px.bar(
                    top_emoji,
                    x="Emoji",
                    y="Count",
                    title="Top 10 Emojis",
                    text="Count"
                )

                # Make it cleaner
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis_title="Emoji",
                    yaxis_title="Count",
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

        st.header(f"Activity Heatmap for {selected_user}", anchor=None)
        analysis.activity_heatmap(selected_user, df)

        st.header(f"Monthly chat activity for {selected_user}" , anchor=None)
        analysis.mostactive_monthly(selected_user , df)

        column1 , column2 = st.columns(2)
        with column1:
            st.header(f"Toxicity Analysis for {selected_user}", anchor=None)
            analysis.toxicity_analysis(selected_user, toxicity_dataframe)
        with column2:
            st.header(f"TOxicity Percentage for {selected_user}", anchor=None)
            st.title(analysis.toxicity_analysis(selected_user, toxicity_dataframe)[0])

    