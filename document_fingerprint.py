import hashlib


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


if __name__ == "__main__":
    file_path = "sample.pdf"

    document_hash = calculate_file_hash(file_path)

    print("Document SHA-256:")
    print(document_hash)