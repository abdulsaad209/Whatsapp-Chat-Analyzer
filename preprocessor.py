import re
import pandas as pd
import nltk
nltk.download('vader_lexicon')
import numpy as np

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s[AP]M\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({'user_messages': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format="%m/%d/%y, %H:%M %p - ")

    # Separate user_names and messages
    users = []
    messages = []
    for message in df['user_messages']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(entry[2])

        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_messages'], inplace=True)

    df.rename(columns={'message_date': 'date'}, inplace=True)

    df['year'] = df['date'].dt.year
    df['only_date'] = df['date'].dt.date
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['min'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    ### for sentiment analysis
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sentiments = SentimentIntensityAnalyzer()
    df["positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]
    df["negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]
    df["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]

    df['positive'] = np.where(df['positive'] > 0.5, 1, 0)
    df['negative'] = np.where(df['negative'] > 0.5, 1, 0)
    df['neutral'] = np.where(df['neutral'] > 0.5, 1, 0)

    df = df.reset_index()
    df = df.iloc[:, 1:]

    sentiment_analysis = []
    for i in range(df.shape[0]):

        if df['positive'].values[i] == 1:
            sentiment_analysis.append('positive')
        elif df['negative'].values[i] == 1:
            sentiment_analysis.append('negative')
        elif df['neutral'].values[i] == 1:
            sentiment_analysis.append('neutral')
        else:
            sentiment_analysis.append('nothing')

    df['sentiment'] = sentiment_analysis

    return df
# end