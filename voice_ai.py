import speech_recognition as sr

def listen_and_convert():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak, I hearing.. (Speak now)")

        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
       
        print("AI analyze korche...")
        text = r.recognize_google(audio)
        print(f"I saying: {text}")
        
    except sr.UnknownValueError:
        print("AI not recognize what you say.")
    except sr.RequestError:
        print("Internet connection or API connection error.")

# Function call kora
if __name__ == "__main__":
    listen_and_convert()