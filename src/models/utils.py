import pickle


def save_model_to_file(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load_model_from_file(filename):
    with open(filename, "rb") as file:
        obj = pickle.load(file)
    return obj
