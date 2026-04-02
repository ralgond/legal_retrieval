import random

of = open("../ft_data/valid_sampled.jsonl", "w+")

with open("../ft_data/valid.jsonl") as inf:
    for line in inf:
        if random.random() < 0.1:
            of.write(line)

of.close()