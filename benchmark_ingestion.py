import time

from vector_store import store_document


file_path = "textbook.pdf"

start_time = time.perf_counter()

result = store_document(file_path)

end_time = time.perf_counter()

processing_time = end_time - start_time


print("\n==============================")
print("DOCUMENT INGESTION BENCHMARK")
print("==============================")

print("\nDocument ID:")
print(result["document_id"])

print("\nFile Hash:")
print(result["file_hash"])

print("\nPages:")
print(result["pages"])

print("\nChunks:")
print(result["chunks"])

print("\nDuplicate:")
print(result["duplicate"])

print("\nProcessing Time:")
print(f"{processing_time:.2f} seconds")