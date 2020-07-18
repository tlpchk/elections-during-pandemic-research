import fasttext as ft
import pandas as pd

from Main import clean_tweet


def get_preprocess_unlabeled_tweets(data):
    data = data[~data['Tweet'].str.startswith("b'RT")]
    return [clean_tweet(row['Tweet']) for index, row in data.iterrows()]


if __name__ == '__main__':
    model = ft.load_model("model/election_model.bin")
    test_data = pd.read_excel("data/final_test_dataset.xlsx")
    tweets = get_preprocess_unlabeled_tweets(test_data)
    labels = []
    for tweet in tweets:
        labels.append(model.predict(tweet)[0][0])
    print('Total', len(labels))
    print(0, labels.count('__label__negative'))
    print(1, labels.count('__label__positive'))
    print(2, labels.count('__label__neutral'))
    for i in range(len(labels)):
        if labels[i] == '__label__positive':
            print(labels[i], tweets[i])
