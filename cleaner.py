import pandas as pd

def clean_text(text):
    """
    Remove all double quotes except the first and last one in a string.
    """
    text = str(text)
    if text.startswith('"') and text.endswith('"') and len(text) > 2:
        return '"' + text[1:-1].replace('"', '') + '"'
    return text.replace('"', '')

def clean_csv(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Filter rows where the Sentence column has fewer than 20 characters
    cleaned_df = df[df['Sentence'].apply(lambda x: len(str(x)) >= 20)]
    
    # Remove all double quotes except the first and last in the Sentence column
    cleaned_df['Sentence'] = cleaned_df['Sentence'].apply(clean_text)
    
    # Save the cleaned data to the output CSV file
    cleaned_df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")

# Usage
if __name__ == "__main__":
    input_csv = 'dataset/dataset_sentences.csv'  # Replace with your input CSV file path
    output_csv = 'dataset/dataset_sentences_cleaned.csv'  # Replace with your desired output CSV file path
    clean_csv(input_csv, output_csv)