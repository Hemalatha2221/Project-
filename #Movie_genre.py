import re
import os
from collections import defaultdict

# Clean and normalize text
def clean_text(text):
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

# Load training data
def load_train_data(filename):
    genre_keywords = defaultdict(lambda: defaultdict(int))
    genres = set()

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) < 4:
                continue
            genre = parts[2]
            description = parts[3]
            genres.add(genre)
            words = clean_text(description)
            for word in words:
                genre_keywords[genre][word] += 1

    return genre_keywords, genres

# Predict genre for a description
def predict_genre(description, genre_keywords, genres):
    words = clean_text(description)
    scores = {genre: 0 for genre in genres}

    for genre in genres:
        for word in words:
            scores[genre] += genre_keywords[genre].get(word, 0)

    predicted = max(scores, key=scores.get)
    return predicted if scores[predicted] > 0 else "Unknown"

# Get valid file input
def get_valid_filename(prompt):
    while True:
        file_name = input(prompt)
        if os.path.isfile(file_name):
            return file_name
        else:
            print(f"❌ File '{file_name}' not found. Please try again.")

# Main function
def main():
    print("🎬 Movie Genre Predictor \n")

    train_file = get_valid_filename("Enter training file name : ")
    test_file = get_valid_filename("Enter test file name : ")
    output_file = input("Enter filename to save all predictions (e.g., all_predictions.txt): ")

    genre_keywords, genres = load_train_data(train_file)

    print("\n✅ Showing Predictions for First 50 Movies:\n")
    count = 0

    with open(test_file, 'r', encoding='utf-8') as file, \
         open(output_file, 'w', encoding='utf-8') as out:

        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) < 3:
                continue
            id_ = parts[0]
            title = parts[1]
            description = parts[2]
            genre = predict_genre(description, genre_keywords, genres)
            result = f"{id_} ::: {title} ::: {genre}"

            if count < 50:
                print(result)

            out.write(result + "\n")
            count += 1

    print(f"\n📁 All {count} predictions saved in '{output_file}'.")

if __name__ == "__main__":
    main()
