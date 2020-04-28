import fasttext as ft

import fasttext.util.util as ftu

from nltk import agreement

import preprocessor as p
import re

import pandas as pd


def calculate_stats_from_excel(path):
    print(path)
    excel_df = pd.read_excel(path)
    filip_ratings = excel_df['Filip'].values
    maksym_ratings = excel_df['Maksym'].values
    magda_ratings = excel_df['Magda'].values

    taskdata = [[0, str(i), str(filip_ratings[i])] for i in range(0, len(filip_ratings))] \
               + [[1, str(i), str(maksym_ratings[i])] for i in range(0, len(maksym_ratings))] \
               + [[2, str(i), str(magda_ratings[i])] for i in range(0, len(magda_ratings))]
    ratingtask = agreement.AnnotationTask(data=taskdata)
    print("kappa " + str(ratingtask.kappa()))
    print("fleiss " + str(ratingtask.multi_kappa()))
    print("alpha " + str(ratingtask.alpha()))
    print("scotts " + str(ratingtask.pi()))


def clean_tweet(tweet):
    tweet = tweet.replace('#', '')
    tweet = p.clean(tweet)
    tweet = "".join([re.sub(r"[^a-zA-ZżźćńółęąśŻŹĆĄŚĘŁÓŃ0-9]+", ' ', k) for k in tweet.split("\n")])
    tweet = tweet[2:]
    return tweet


def preprocess_tweets(data):
    data = data[~data['Tweet'].str.startswith("b'RT")]
    data = data[~data['Label'].str.startswith("Retweet", na=False)]
    data = data[~data['Label'].str.startswith("N/D", na=False)]
    data['Label'] = data['Label'].astype(int)

    lines = ""
    for index, row in data.iterrows():
        tweet = row['Tweet']
        data.at[index, 'Tweet'] = clean_tweet(tweet)

        lines += "__label__"
        if row["Label"] == 0:
            lines += "negative"
        elif row["Label"] == 1:
            lines += "positive"
        else:
            lines += "neutral"
        lines += " " + tweet + '\n'

    data.to_csv("data/preprocessed_data.csv")

    with open("data/wybory.txt", "w") as text_file:
        text_file.write(lines)

    return data


def save_to_txt(data, name):
    lines = ""
    for index, row in data.iterrows():

        lines += "__label__"
        if row["Label"] == 0:
            lines += "negative"
        elif row["Label"] == 1:
            lines += "positive"
        else:
            lines += "neutral"
        lines += " " + row["Tweet"] + '\n'

    with open(f"data/{name}", "w") as text_file:
        text_file.write(lines)


if __name__ == '__main__':
    # calculate_stats_from_excel("data/first_session.xlsx")
    # calculate_stats_from_excel("data/second_session_after_discussion.xlsx")

    # full_data = pd.read_excel("data/full_dataset.xlsx")
    # full_data = preprocess_tweets(full_data)
    #
    # split_ratio = 0.8
    # nrows = len(full_data)
    # nrows_train = int(nrows*split_ratio)
    # nrows_test = nrows - nrows_train
    # #
    # save_to_txt(full_data.head(nrows_train), "wybory_train.txt")
    # save_to_txt(full_data.tail(nrows_test), "wybory_test.txt")

    # line = "Epoch;dim;lr;wordNgrams;precision;recall\n"
    # for lr in (1e-4, 1e-3, 1e-2, 1e-1):
    #     for wordNgram in (1,2,3):
    #         model = ft.train_supervised(input="data/wybory_train.txt",epoch=500, dim=300, lr=lr, wordNgrams=wordNgram, pretrainedVectors="data/cc.pl.300.vec")
    #         result = model.test("data/wybory_test.txt")
    #         line += f"1000;300;{lr};{wordNgram};{result[1]};{result[2]}\n"
    #
    # with open('data/results.csv', 'w') as results_file:
    #     print(line, file=results_file)

    # model = ft.train_supervised(input="data/wybory.txt",epoch=1000, dim=300, lr=1e-2, wordNgrams=2, pretrainedVectors="data/cc.pl.300.vec")
    # model.save_model("election_model.bin")


    #USAGE OF MODEL
    model = ft.load_model("election_model.bin")
    tweet = "b'@michal_kolanko @bbudka Wybory zgodnie z Konstytucją RP powinny odbyć się 10 maja 2020 roku. Amen. #Konstytucja #demokracja #wybory'"
    tweet = clean_tweet(tweet)
    print(model.predict(tweet))
