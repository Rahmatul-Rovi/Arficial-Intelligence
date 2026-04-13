from deep_translator import GoogleTranslator

def ai_translator(text, target_lang):
    # Google Translator AI use 
    translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
    return translated

# Test korar jonno
original_text = "Artificial Intelligence is changing the world."

print("--- AI Language Translator ---")
print(f"Original (English): {original_text}")

# English to Bengali Translation
bengali_text = ai_translator(original_text, 'bn')
print(f"Translated (Bengali): {bengali_text}")

# English to Spanish Translation
spanish_text = ai_translator(original_text, 'es')
print(f"Translated (Spanish): {spanish_text}")

# User Input section
print("\n--- Try it yourself ---")
user_text = input("Enter a sentence in English: ")
output = ai_translator(user_text, 'bn')
print(f"AI Translated (Bengali): {output}")