from sklearn.preprocessing import LabelEncoder

from src.data.load_dataset import GESTURE_CLASSES


def encode_labels(labels):
    encoder = LabelEncoder()

    encoder.fit(GESTURE_CLASSES)

    encoded_labels = encoder.transform(labels)

    return encoded_labels, encoder


def decode_labels(encoded_labels, encoder):
    return encoder.inverse_transform(encoded_labels)


def get_label_mapping(encoder):
    return {
        label: int(index)
        for index, label in enumerate(encoder.classes_)
    }


if __name__ == "__main__":
    labels = GESTURE_CLASSES

    encoded_labels, encoder = encode_labels(labels)

    print("Categorical Data Encoding")
    print("-------------------------")

    print("\nOriginal Labels:")
    print(labels)

    print("\nEncoded Labels:")
    print(encoded_labels)

    print("\nLabel Mapping:")
    print(get_label_mapping(encoder))