import argparse
parser = argparse.ArgumentParser()


parser.add_argument("-td", "--train_dir",default="data/pizza_steak_sushi/train")
parser.add_argument("-tsd", "--test_dir", default="data/pizza_steak_sushi/test")
parser.add_argument("-lr", "--learning_rate", default=0.001)
parser.add_argument("-batch_size", "--batch_size", default=32)
parser.add_argument("-eps", "--epochs", default=5)
parser.add_argument("-hu", "--hidden_units", default=10)

