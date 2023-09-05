import pandas as pd

data = pd.read_csv(r'D:\cuhk\23Spring\DDA4210\Project\cleaned_data.csv')

train = pd.DataFrame ()
star_5 = data[data["Star"] == 5]["Comment"].tolist()
star_4 = data[data["Star"] == 4]["Comment"].tolist()
star_2 = data[data["Star"] == 2]["Comment"].tolist()
star_1 = data[data["Star"] == 1]["Comment"].tolist()

train = train.assign(Star_5 = star_5[:20000])
train = train.assign(Star_4 = star_4[:20000])
train = train.assign(Star_2 = star_2[:20000])
train = train.assign(Star_1 = star_1[:20000])

dev = pd.DataFrame ()
dev = dev.assign(Star_5 = star_5[20001:25000])
dev = dev.assign(Star_4 = star_4[20001:25000])
dev = dev.assign(Star_2 = star_2[20001:25000])
dev = dev.assign(Star_1 = star_1[20001:25000])

train = train.astype(str)
dev = dev.astype(str)


train.to_csv(r'D:\cuhk\23Spring\DDA4210\Project\movie_4rating_20k_train_cleaned.csv', encoding='utf_8_sig', index = False)
dev.to_csv(r'D:\cuhk\23Spring\DDA4210\Project\movie_4rating_5k_dev_cleaned.csv', encoding='utf_8_sig', index = False)


train.to_csv(r'D:\cuhk\23Spring\DDA4210\Project\movie_4rating_20k_train_cleaned.tsv', sep = "\t", index = False)
dev.to_csv(r'D:\cuhk\23Spring\DDA4210\Project\movie_4rating_5k_dev_cleaned.tsv', sep = "\t", index = False)