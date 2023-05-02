from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split



def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    # print(tokens)
    out = int(tokens[0])
    # print(out)
    output = [0 if out == 1 else 0.5 if out == 2 else 1]
    # print(output)

    inpt = [float(x) for x in tokens[1:]]
    # print(inpt)
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data


with open("wine_data.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# for line in training_data:
#     print(line)

td = normalize(training_data)

train_data, test_data = train_test_split(td, test_size=.15)
print(len(train_data))
print(len(test_data))
# for line in td:
#     print(line)

nn = NeuralNet(13, 3, 1)

nn.train(train_data, learning_rate=.1)

for i in nn.test_with_expected(test_data):
    print(f"desired: {i[1]}, actual: {i[2]}")