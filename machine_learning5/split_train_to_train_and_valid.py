import random

train = open("../data/ml5/s3_train.jsonl", "w+")
valid = open("../data/ml5/s3_valid.jsonl", "w+")

for line in open("../data/ml5/train.jsonl"):
    if random.random() < 0.1:
        valid.write(line)
    else:
        train.write(line)


train.close()
valid.close()