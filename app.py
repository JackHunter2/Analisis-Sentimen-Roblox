from flask import Flask, render_template, send_file, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

app = Flask(__name__)
DATA_PATH = "data/cleaned_youtube_comments.csv"

def label_sentiment(texts):
    POSITIVE_WORDS = [
        "bagus", "suka", "keren", "lucu", "seru", "mantap", "hebat", "cinta", "senang", "ngakak", "top", "asik", "puas", "terbaik", "recommended", "anjay", "mantul", "wow", "bagus banget", "positif", "support", "sukses", "terima kasih", "makasih", "good", "nice", "love", "amazing", "happy", "enak", "menarik", "oke", "yes", "setuju", "respect", "legend", "favorit", "favoritku", "favorit gw", "favorit gue"
    ]
    NEGATIVE_WORDS = [
        "jelek", "buruk", "benci", "parah", "gila", "aneh", "nggak suka", "marah", "sedih", "kecewa", "jelek banget", "parah banget", "gak suka", "sampah", "menyedihkan", "goblok", "anjir", "anjing", "negatif", "tidak suka", "tidak setuju", "payah", "lemah", "gak jelas", "gak guna", "gak penting", "gak paham", "gak ngerti", "gak enak", "gak bagus", "gak bener", "gak suka", "gak seru", "gak lucu", "gak asik", "gak mantap", "gak puas", "gak happy", "gak respect", "gak recommended", "gak legend", "gak favorit"
    ]
    results = []
    for text in texts:
        text_l = text.lower()
        if any(w in text_l for w in POSITIVE_WORDS):
            results.append("positive")
        elif any(w in text_l for w in NEGATIVE_WORDS):
            results.append("negative")
        else:
            results.append("neutral")
    return results

df = pd.read_csv(DATA_PATH)
if 'sentiment' not in df.columns:
    df['sentiment'] = label_sentiment(df['clean_comment'].fillna("").astype(str))

@app.route("/")
def index():
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    return render_template("index.html", data=sentiment_counts)

@app.route("/visualisasi")
def visualisasi():
    filter_sentimen = request.args.get('sentimen')
    if filter_sentimen and filter_sentimen in df['sentiment'].unique():
        filtered_df = df[df['sentiment'] == filter_sentimen]
    else:
        filtered_df = df
    sentiment_counts = filtered_df["sentiment"].value_counts()
    chart_data = {
        "labels": list(sentiment_counts.index),
        "values": [int(x) for x in sentiment_counts.values]
    }
    all_sentiments = df['sentiment'].unique().tolist()
    # Data dummy tren sentimen per bulan
    line_labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun"]
    line_data = {
        "positive": [20, 25, 30, 28, 35, 40],
        "neutral": [15, 18, 17, 20, 19, 21],
        "negative": [10, 12, 14, 13, 15, 16]
    }
    # Data dummy perbandingan model
    model_labels = ["Naive Bayes", "SVM", "BERT"]
    model_metrics = {
        "Akurasi": [0.82, 0.85, 0.91],
        "Precision": [0.80, 0.84, 0.92],
        "Recall": [0.81, 0.83, 0.90]
    }
    # Data dummy confusion matrix
    cm_labels = ["positive", "neutral", "negative"]
    cm_matrix = [
        [50, 5, 2],
        [4, 40, 6],
        [1, 3, 45]
    ]
    # Data dummy heatmap korelasi
    heatmap_labels = ["F1", "F2", "F3"]
    heatmap_matrix = [
        [1.0, 0.6, -0.2],
        [0.6, 1.0, 0.1],
        [-0.2, 0.1, 1.0]
    ]
    # Data dummy scatter plot
    scatter_data = [
        {"label": "positive", "data": [[1,2],[2,3],[3,4],[4,5],[5,6]]},
        {"label": "neutral", "data": [[1,5],[2,4],[3,3],[4,2],[5,1]]},
        {"label": "negative", "data": [[1,1],[2,2],[3,1.5],[4,1],[5,0.5]]}
    ]
    return render_template(
        "visualisasi.html",
        chart_data=chart_data,
        all_sentiments=all_sentiments,
        selected_sentimen=filter_sentimen,
        line_labels=line_labels,
        line_data=line_data,
        model_labels=model_labels,
        model_metrics=model_metrics,
        cm_labels=cm_labels,
        cm_matrix=cm_matrix,
        heatmap_labels=heatmap_labels,
        heatmap_matrix=heatmap_matrix,
        scatter_data=scatter_data
    )

@app.route("/wordcloud")
def wordcloud():
    sentiment_clouds = {}
    for sentiment in df["sentiment"].unique():
        text = " ".join(df[df["sentiment"] == sentiment]["clean_comment"].fillna("").astype(str))
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
