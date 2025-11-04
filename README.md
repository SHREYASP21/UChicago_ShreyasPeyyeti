Gender Bias in Word Embeddings

Overview

This project checks whether AI word models show gender bias in how they represent jobs.
I used a pretrained Google News Word2Vec model and compared how close each profession word (like engineer, nurse, teacher) is to the words “man” and “woman.”
The difference gives a bias score — positive means the word is closer to man (masculine), and negative means closer to woman (feminine).

Two graphs are created:

Directional Bias Plot – shows whether each job leans male or female

Bias Magnitude Plot – shows how strong that bias is

Model File

The model file is too large to upload to GitHub.
You can download it here:
Word2Vec Slim 300k – Kaggle

File name:
GoogleNews-vectors-negative300-SLIM.bin

After downloading, put it in your project folder (example):

D:\UChicago_ShreyasPeyyeti\GoogleNews-vectors-negative300-SLIM.bin

How to Run

Install requirements:

pip install pandas numpy matplotlib gensim scipy


Run the program:

python main.py


The terminal will print bias scores and save two figures:

bias_direction_plot.png

bias_magnitude_plot.png

What It Shows

The results show that the model still carries small gender biases —
words like engineer lean masculine, and nurse or teacher lean feminine.
It’s a simple way to see how AI models can pick up real-world stereotypes from text.
