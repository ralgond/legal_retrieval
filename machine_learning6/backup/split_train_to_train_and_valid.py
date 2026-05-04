import random

train = open("../data/ml6/train_raw.jsonl", "w+")
valid = open("../data/ml6/eval_raw.jsonl", "w+")

for line in open("../data/ml6/total.jsonl"):
    if random.random() < 0.1:
        valid.write(line)
    else:
        train.write(line)


train.close()
valid.close()