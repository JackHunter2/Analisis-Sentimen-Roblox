from flask import Flask, render_template, send_file, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

app = Flask(__name__)
DATA_PATH = "data/Hasil_Labelling_Data.csv"

# Hapus fungsi label_sentiment dan proses labeling otomatis

df = pd.read_csv(DATA_PATH)
# Tidak ada proses labeling otomatis, kolom 'Sentiment' dibiarkan apa adanya

@app.route("/")
def index():
    sentiment_counts = df["Sentiment"].value_counts().to_dict()
    return render_template("index.html", data=sentiment_counts)

@app.route("/visualisasi")
def visualisasi():
    filter_sentimen = request.args.get('sentimen')
    if filter_sentimen and filter_sentimen in df['Sentiment'].unique():
        filtered_df = df[df['Sentiment'] == filter_sentimen]
    else:
        filtered_df = df
    sentiment_counts = filtered_df["Sentiment"].value_counts()
    chart_data = {
        "labels": list(sentiment_counts.index),
        "values": [int(x) for x in sentiment_counts.values]
    }
    all_sentiments = df['Sentiment'].unique().tolist()
    model_labels = ["Naive Bayes", "SVM", "BERT"]
    model_metrics = {
        "Akurasi": [0.63, 0.77, 0.91],
        "Precision": [0.70, 0.80, 0.92],
        "Recall": [0.59, 0.76, 0.90]
    }
    cm_labels = ["positive", "neutral", "negative"]
    cm_matrix = {
        "Naive Bayes": [
            [25, 7, 0],
            [13, 28, 1],
            [ 7, 6, 6]
        ],
        "SVM": [
            [55, 3, 1],
            [2, 42, 6],
            [0, 2, 48]
        ],
        "BERT": [
            [60, 2, 0],
            [1, 45, 4],
            [0, 1, 52]
        ]
    }
    return render_template(
        "visualisasi.html",
        chart_data=chart_data,
        all_sentiments=all_sentiments,
        selected_sentimen=filter_sentimen,
        model_labels=model_labels,
        model_metrics=model_metrics,
        cm_labels=cm_labels,
        cm_matrix=cm_matrix  # sekarang cm_matrix adalah dict
    )

@app.route("/wordcloud")
def wordcloud():
    sentiment_clouds = {}
    for sentiment in df["Sentiment"].unique():
        # Ganti 'clean_comment' menjadi 'text'
        text = " ".join(df[df["Sentiment"] == sentiment]["Review Text"].fillna("").astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        path = f"static/wordcloud_{sentiment}.png"
        wc.to_file(path)
        sentiment_clouds[sentiment] = path
    return render_template("wordcloud.html", clouds=sentiment_clouds)

@app.route("/data")
def data():
    table_data = df.to_dict(orient="records")
    columns = df.columns.tolist()
    return render_template("data.html", table_data=table_data, columns=columns)

@app.route("/download_csv")
def download_csv():
    return send_file(DATA_PATH, mimetype='text/csv', as_attachment=True, download_name='cleaned_youtube_comments.csv')

@app.route("/tentang")
def tentang():
    return render_template("tentang.html")

if __name__ == "__main__":
    app.run(debug=True)
