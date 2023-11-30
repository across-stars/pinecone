import os

data_folder = "/n/fs/nlp-ap5697/projects/pinecone/data"

DATASETS = {
    "SAT": os.path.join(data_folder, "analogy_test_dataset", "sat", "test.jsonl"),
    "BATS": os.path.join(data_folder, "analogy_test_dataset", "bats", "test.jsonl"),
    "GOOGLE": os.path.join(data_folder, "analogy_test_dataset", "google", "test.jsonl"),
    "SCAN": os.path.join(data_folder, "SCAN_dataset.csv")
}

INDICES = {
    "glove": "glove40",
    "relbert": "relbert"
}