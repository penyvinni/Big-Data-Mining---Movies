# Importing necessary libraries
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting the dataset
import tensorflow_hub as hub  # For loading Universal Sentence Encoder (USE) model
from sklearn.manifold import TSNE  # For t-SNE dimensionality reduction
import matplotlib.pyplot as plt  # For data visualization

# Step 1: Loading and Preparing the SST-2 Dataset

# Load the dataset
dataset_path = "sst2-train.csv"
df = pd.read_csv(dataset_path)  # Reading data into a DataFrame

# Print DataFrame structure to check column names
print(df.head())  # Displaying the first few rows of the DataFrame

# Split into features (texts) and labels
texts = df['sentence'].tolist()  # Extracting text data
labels = df['label'].tolist()    # Extracting label data

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 2: Vectorize the Texts with the Universal Sentence Encoder

# Load Universal Sentence Encoder module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)  # Loading Universal Sentence Encoder model
print("Universal Sentence Encoder loaded")  # Printing confirmation message

# Define a function to embed texts
def embed_text(texts):
    embeddings = model(texts)  # Obtaining embeddings for texts using the USE model
    return embeddings

# Vectorize training and validation texts
train_embeddings = embed_text(train_texts)  # Embedding training texts
val_embeddings = embed_text(val_texts)      # Embedding validation texts

# Example: Print first 3 embeddings and corresponding texts
for i in range(3):
    print("Text:", train_texts[i])  # Displaying the text
    print("Embedding:", train_embeddings[i])  # Displaying the corresponding embedding
    print("Label:", train_labels[i])  # Displaying the corresponding label
    print()

# Step 3: Application of the t-SNE Method for Dimensionality Reduction

# Apply t-SNE to reduce dimensions of the embeddings
tsne = TSNE(n_components=2, random_state=42)  # Initializing t-SNE object
train_embeddings_tsne = tsne.fit_transform(train_embeddings)  # Applying t-SNE to training embeddings

# Step 4: Visualizing the Data

# Visualize the data in the new dimensional space created by t-SNE
plt.figure(figsize=(10, 6))  # Setting figure size
plt.scatter(train_embeddings_tsne[:, 0], train_embeddings_tsne[:, 1], c=train_labels, cmap=plt.cm.get_cmap('RdYlGn', 2))  # Creating scatter plot
plt.title('t-SNE Visualization of Universal Sentence Encoder Embeddings')  # Setting title
plt.xlabel('Component 1')  # Setting x-axis label
plt.ylabel('Component 2')  # Setting y-axis label
plt.colorbar(label='Label')  # Adding colorbar with label
plt.grid(True)  # Adding grid
plt.show()  # Displaying the plot
