import random

of = open("../ft_data/train_sampled.jsonl", "w+")

with open("../ft_data/train.jsonl") as inf:
    for line in inf:
        if random.random() < 0.3:
            of.write(line)

of.close()