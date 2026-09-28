import uuid


def generate_document_id():
    return str(uuid.uuid4())


if __name__ == "__main__":

    document_id = generate_document_id()

    print("Generated Document ID:")
    print(document_id)