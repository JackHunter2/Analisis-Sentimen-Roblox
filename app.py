from flask import Flask, render_template, send_file, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import os
from sklearn.metrics import confusion_matrix
from collections import Counter

app = Flask(__name__)
DATA_PATH = "data/Hasil_Labelling_Data.csv"

# Hapus fungsi label_sentiment dan proses labeling otomatis

df = pd.read_csv(DATA_PATH)
# Tidak ada proses labeling otomatis, kolom 'Sentiment' dibiarkan apa adanya

cm_labels = ["positive", "neutral", "negative"]
cm_matrix = {
    "Naive Bayes": [
        [25, 7, 0],
        [13, 28, 1],
        [7, 6, 6]
    ],
    "SVM": [
        [23, 9, 0],
        [5, 35, 2],
        [1, 4, 14]
    ],
    "BERT": [
        [26, 4, 2],
        [10, 23, 9],
        [3, 1, 15]
    ]
}

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
        "Akurasi": [0.63, 0.77, 0.67],
        "Precision": [0.70, 0.80, 0.59],
        "Recall": [0.59, 0.76, 0.60]
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
    # Pastikan label sentimen konsisten dengan data (Negatif, Netral, Positif)
    for sentiment in df["Sentiment"].unique():
        # Ambil teks dari kolom 'steming_data' sesuai sentimen
        text = df[df["Sentiment"] == sentiment]["steming_data"].fillna("").astype(str).str.cat(sep=" ")
        wc = WordCloud(width=800, height=400, random_state=42, max_font_size=100, background_color="black").generate(text)
        path = f"static/wordcloud_{sentiment}.png"
        wc.to_file(path)
        sentiment_clouds[sentiment] = path

    # Hitung frekuensi kata untuk seluruh data
    text_all = " ".join(df["steming_data"].fillna("").astype(str))
    stopwords = set(STOPWORDS)
    stopwords.update(['https', 'yang', 'di', '...', 'dan', 'ya', 'ini', 'nya', 'buat', 'pas'])
    tokens = [word for word in text_all.split() if word not in stopwords]
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    words, counts = zip(*top_words) if top_words else ([], [])

    return render_template("wordcloud.html", clouds=sentiment_clouds, freq_words=words, freq_counts=counts)

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

@app.route("/confusion_matrix/<model>")
def confusion_matrix_img(model):
    # Data dummy, ganti dengan data asli jika ada
    if model not in cm_matrix:
        return "Model tidak ditemukan", 404
    cm = cm_matrix[model]
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title(f"Confusion Matrix {model}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    img_path = f"static/cm_{model}.png"
    plt.savefig(img_path)
    plt.close()
    from flask import send_file
    return send_file(img_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
