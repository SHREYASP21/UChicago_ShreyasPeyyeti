# ===============================================================
# UChicago MS in Applied Data Science Application
# Author: Shreyas Peyyeti | GitHub:  https://github.com/SHREYASP21/UChicago_ShreyasPeyyeti.git
# Title: Quantifying and Visualizing Gender Bias in Word Embeddings
# ===============================================================

# ---- 1. Import Libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from scipy.stats import zscore

# ---- 2. Ingest Data: Load Pretrained Word2Vec Embeddings ----
print("Loading pretrained embeddings...")
model = KeyedVectors.load_word2vec_format(
    r"D:\UChicago_ShreyasPeyyeti\GoogleNews-vectors-negative300-SLIM.bin",
    binary=True
)
print("Embeddings loaded successfully.\n")

# ---- 3. Define Target Word Sets and Prepare Data ----
professions = [
    "engineer", "nurse", "doctor", "teacher", "scientist",
    "lawyer", "pilot", "chef", "artist", "accountant"
]
prof_available = [p for p in professions if p in model.key_to_index]

# ---- 4. Define Custom Functions ----
def cosine_similarity(a, b):
    """Compute cosine similarity between two n-dimensional vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_bias(word):
    """Bias = similarity(word, man) - similarity(word, woman)."""
    v = model[word]
    return cosine_similarity(v, model["man"]) - cosine_similarity(v, model["woman"])

def analyze_bias(words):
    """Compute bias metrics, z-scores, scaling, and classification."""
    bias = [compute_bias(w) for w in words]
    df = pd.DataFrame({"word": words, "bias_score": bias})
    df["bias_z"] = zscore(df["bias_score"])
    df["bias_scaled"] = df["bias_score"] / abs(df["bias_score"]).max()
    df["category"] = np.where(df["bias_score"] > 0, "Masculine-leaning", "Feminine-leaning")
    return df.sort_values("bias_score", ascending=False).reset_index(drop=True)

# ---- 5. Run Analysis ----
bias_df = analyze_bias(prof_available)
print("Gender Bias Scores (Positive = Closer to 'man', Negative = Closer to 'woman'):\n")
print(bias_df.to_string(index=False, float_format="%.3f"))

# ---- 6. Visualization 1: Directional Bias ----
plt.style.use("ggplot")
plt.figure(figsize=(8, 4))
colors = bias_df["bias_scaled"].apply(lambda x: "#2E86C1" if x > 0 else "#C0392B")
plt.barh(bias_df["word"], bias_df["bias_scaled"], color=colors, edgecolor="black", alpha=0.9)
plt.axvline(0, color="black", linewidth=1)
plt.title("Directional Gender Bias in Word Embeddings", fontsize=11, weight="bold")
plt.xlabel("Normalized Bias (Positive = Male, Negative = Female)")
plt.ylabel("Profession")
plt.tight_layout()
plt.savefig("bias_direction_plot.png", dpi=300)
print("\nSaved 'bias_direction_plot.png'.")

# ---- 7. Visualization 2: Magnitude of Bias ----
plt.figure(figsize=(8, 4))
plt.barh(bias_df["word"], abs(bias_df["bias_score"]),
         color="#6C3483", edgecolor="black", alpha=0.9)
plt.title("Bias Magnitude by Profession", fontsize=11, weight="bold")
plt.xlabel("Absolute Bias |Similarity(man) - Similarity(woman)|")
plt.ylabel("Profession")
plt.tight_layout()
plt.savefig("bias_magnitude_plot.png", dpi=300)
print("Saved 'bias_magnitude_plot.png'.\n")

# ---- 8. Summary Statistics ----
mean_bias = bias_df["bias_score"].mean()
std_bias = bias_df["bias_score"].std()
max_word = bias_df.loc[bias_df["bias_score"].idxmax(), "word"]
min_word = bias_df.loc[bias_df["bias_score"].idxmin(), "word"]

print("Summary Statistics:")
print(f"Average Bias Score: {mean_bias:.3f}")
print(f"Standard Deviation: {std_bias:.3f}")
print(f"Most Masculine-biased: {max_word}")
print(f"Most Feminine-biased: {min_word}")
print("\nInterpretation: Positive bias indicates male association; negative bias indicates female association.")
