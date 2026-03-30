from textblob import TextBlob

def analyze_sentiment(text):

    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"


sentences = [
    "I love this programming language, it is amazing!",
    "This is a very bad and boring day.",
    "I am going to the market now."
]

print("--- AI Sentiment Analysis Result ---")
for s in sentences:
    result = analyze_sentiment(s)
    print(f"Text: {s}")
    print(f"Sentiment: {result}\n")

# User input option
user_input = input("Write something in English to check sentiment: ")
print(f"AI thinks this is: {analyze_sentiment(user_input)}")