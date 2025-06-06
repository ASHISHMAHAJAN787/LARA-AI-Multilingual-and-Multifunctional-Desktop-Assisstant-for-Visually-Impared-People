# import io
# import pygame
# import speech_recognition as sr
# from gtts import gTTS
# from keras.models import load_model # type: ignore
# from pickle import load
# import numpy as np
# import tensorflow as tf
# import eel
# from API_functionalities import get_news, get_weather, get_wikipedia_summary, get_ip_address, tell_joke, get_trending_movies, solve_math, check_internet_speed, ask_gemini, setup_gemini
# from system_operations import SystemTasks  # Import SystemTasks class
# from email_helper import send_email
# from deep_translator import GoogleTranslator
# from emergency_SMS import send_sos_whatsapp,get_user_location
# import re
# import webbrowser
# import cv2
# import torch
# from torchvision import transforms
# from PIL import Image

# # Initialize pygame mixer for audio playback
# pygame.mixer.init()

# # Initialize SystemTasks
# system_tasks = SystemTasks()

# # Load the trained model
# MODEL_PATH = r'D:/PYTHON/AI_VOICE/Data/chat_model.keras'
# TOKENIZER_PATH = r'D:/PYTHON/AI_VOICE/Data/tokenizer.pickle'
# LABEL_ENCODER_PATH = r'D:/PYTHON/AI_VOICE/Data/label_encoder.pickle'

# # Initialize Gemini
# gemini_available = setup_gemini()

# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# model = load_model(MODEL_PATH)

# with open(TOKENIZER_PATH, 'rb') as handle:
#     tokenizer = load(handle)

# with open(LABEL_ENCODER_PATH, 'rb') as enc:
#     label_encoder = load(enc)

# recognizer = sr.Recognizer()

# # Variable to store the user's name globally
# user_name = ""

# # Initialize Eel
# eel.init("D:/PYTHON/AI_VOICE/Proj/www")  # 'web' is the directory containing your HTML, JS, and CSS

# # Flag to control assistant status
# assistant_running = False

# def speak(text, language_code='en'):
#     """Use gTTS to speak the given text and display it in the response box."""
#     try:
#         eel.showResponding()
#         if user_name:
#             text = f"{user_name}, {text}"
        
#         # Translate the text to the selected language
#         translated_text = translate_text(text, language_code)
#         print(f"Assistant: {translated_text}")
        
#         # Update the responses box
#         eel.updateResponses(translated_text)  # Send the response to the frontend

#         # Generate speech in memory using gTTS
#         tts = gTTS(text=translated_text, lang=language_code)
#         audio_fp = io.BytesIO()
#         tts.write_to_fp(audio_fp)
#         audio_fp.seek(0)

#         # Load and play audio using pygame
#         pygame.mixer.music.load(audio_fp, 'mp3')
#         pygame.mixer.music.play()

#         # Wait for the audio to finish playing
#         while pygame.mixer.music.get_busy():
#             pygame.time.Clock().tick(10)
    
#     except Exception as e:
#         print(f"Error in speaking: {e}")
    
#     finally:
#         eel.resetAnimations()
 
# # def record_audio():
# #     """Record audio input from the user."""
# #     eel.showListening()
# #     with sr.Microphone() as mic:
# #         recognizer.adjust_for_ambient_noise(mic)
# #         print("Listening...")
# #         audio = recognizer.listen(mic)
# #         try:
# #             text = recognizer.recognize_google(audio)
# #             print(f"User: {text}")
# #             return text.lower()
# #         except:
# #             speak("Sorry, I didn't catch that.")
# #             return None
# def record_audio(language_code='en'):
#     """Record audio input from the user with language support"""
#     eel.showListening()
#     with sr.Microphone() as mic:
#         recognizer.adjust_for_ambient_noise(mic)
#         print("Listening...")
#         audio = recognizer.listen(mic)
#         try:
#             # Map language codes to recognizer language codes
#             recognizer_lang_map = {
#                 'en': 'en-IN',
#                 'hi': 'hi-IN',
#                 'es': 'es-ES',
#                 'fr': 'fr-FR',
#                 'pa': 'pa-IN',
#                 'ta': 'ta-IN',
#                 'te': 'te-IN'
#             }
            
#             # Get the recognizer language code
#             recognizer_lang = recognizer_lang_map.get(language_code, 'en-IN')
            
#             # Recognize speech in the specified language
#             text = recognizer.recognize_google(audio, language=recognizer_lang)
#             print(f"User (original): {text}")
            
#             # If input isn't English, translate to English for processing
#             if language_code != 'en':
#                 translated_text = translate_text(text, 'en')
#                 print(f"User (translated to English): {translated_text}")
#                 return translated_text.lower()
#             return text.lower()
#         except Exception as e:
#             print(f"Speech recognition error: {e}")
#             speak("Sorry, I didn't catch that.", language_code)
#             return None

# def select_language():
#     """Prompt user to select a language."""
#     speak("Please select a language: English, Spanish, French, Hindi, Punjabi, Tamil or Telugu.")
#     language = record_audio()  # Record the user's choice
    
#     if language:
#         language = language.lower()
#         if language in ['english', 'spanish', 'french', 'hindi', 'punjabi', 'tamil', 'telugu']:
#             return language
#         else:
#             speak("Language not supported. Defaulting to English.")
#             return "english"
#     else:
#         speak("No language selected. Defaulting to English.")
#         return "english"

# # def translate_text(text, target_language):
# #     """Translate text using deep-translator."""
# #     try:
# #         translated_text = GoogleTranslator(source='auto', target=target_language).translate(text)
# #         return translated_text
# #     except Exception as e:
# #         print(f"Translation error: {e}")
# #         return "Translation error occurred."
# def translate_text(text, target_language, source_language='auto'):
#     """Improved translation with better error handling"""
#     try:
#         if not text or text == "Translation error occurred.":
#             return text
            
#         # Don't translate if already in target language
#         if target_language == 'en' and text.isascii():
#             return text
            
#         translated = GoogleTranslator(
#             source=source_language,
#             target=target_language
#         ).translate(text)
        
#         return translated if translated else text
#     except Exception as e:
#         print(f"Translation error ({source_language}→{target_language}): {e}")
#         return text  # Return original text if translation fails

# def predict_intent(text):
#     """Predict intent from the user input using the chatbot model."""
#     seq = tokenizer.texts_to_sequences([text])
#     padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=20)  # Adjust maxlen as per your model
#     pred = model.predict(padded)
#     intent = label_encoder.inverse_transform([np.argmax(pred)])
#     return intent[0]



# def recognize_objects(language_code='en'):
#     """
#     Capture an image from the webcam, recognize objects, and describe them to the user.
#     """
#     # Initialize the webcam
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         speak("Unable to access the webcam.", language_code)
#         return

#     speak("Please show the object to the camera.", language_code)
#     ret, frame = cap.read()  # Capture a frame from the webcam
#     if not ret:
#         speak("Failed to capture image.", language_code)
#         return

#     # Save the captured frame as an image
#     image_path = "captured_image.jpg"
#     cv2.imwrite(image_path, frame)
#     cap.release()

#     # Perform object detection on the captured image using YOLOv5
#     results = yolo_model(image_path)  # YOLOv5 model inference
#     detected_objects = results.pandas().xyxy[0]  # Get detected objects in a DataFrame

#     if len(detected_objects) > 0:
#         object_names = detected_objects['name'].unique()  # Get unique object names
#         object_list = ", ".join(object_names)  # Convert to a comma-separated string
#         speak(f"I see: {object_list}", language_code)
#     else:
#         speak("I couldn't detect any objects.", language_code)

#     # Clean up the captured image
#     import os
#     os.remove(image_path)



# def extract_name(text):
#     """Extract name from user input."""
#     # Handle various ways of saying the name
#     name_patterns = [
#         r"my name is (\w+)",
#         r"i am (\w+)",
#         r"you can call me (\w+)",
#         r"(\w+)"
#     ]
    
#     for pattern in name_patterns:
#         match = re.search(pattern, text.lower())
#         if match:
#             name = match.group(1).capitalize()  # Capitalize the first letter of the name
#             return name
#     return None



# @eel.expose
# def start_assistant():
#     print("Assistant is now active.")
#     """Start the assistant and interact with the user."""
#     global assistant_running, user_name
#     assistant_running = True  # Start the assistant

#     # Ask the user to select a language first
#     selected_language = select_language()
#     language_code_map = {
#         "english": "en",
#         "spanish": "es",
#         "french": "fr",
#         "hindi": "hi",
#         "punjabi": "pa",
#         "tamil": "ta",
#         "telugu": "te"
#     }
#     language_code = language_code_map[selected_language]

#     speak(f" Hello! , I am Lara, your virtual assistant. Before we start, may I know your name?",language_code)
    
#     # Loop to handle user response for name
#     user_name = record_audio()
#     if user_name:
#         user_name = extract_name(user_name)  # Extract the name from the user input
#         if user_name:
#             speak(f"Nice to meet you",language_code)
#         else:
#             user_name = "Friend"
#             speak(f"Alright, I'll call you {user_name}.",language_code)
#     else:
#         user_name = "User"
#         speak(f"Alright, I'll call you {user_name}.")

#     speak(f"How can I assist you today?", language_code)

#     while assistant_running:  # Loop will run while assistant_running is True
#         text = record_audio(language_code)
#         if text is None:
#             continue
        
#         intent = predict_intent(text)
#         print(f"Predicted Intent: {intent}")  # Debugging line

#         if "wikipedia" in intent:
#             handle_wikipedia_intent(language_code, text)
            
       
        
#         elif "weather" in intent:
#             speak("Please tell me the country you want to know the weather for.", language_code)
#             country = record_audio(language_code)  # Get country input from the user
#             speak("Now please tell me the city.", language_code)
#             city = record_audio(language_code)  # Get city input from the user
            
#             # Check if the user asks for a 14-day forecast
#             if "next 14 days" in text or "14 days" in text:
#                 weather_report = get_weather(city, country, forecast_days=14)
#             else:
#                 weather_report = get_weather(city, country)  # Default to current weather

#             speak(weather_report, language_code)



#         elif "news" in intent:
#             headlines = get_news()  # Call the function to get the latest news
#             speak(f"Here are the latest headlines: {headlines}", language_code)
#             speak("Anything more to ask?")

#         elif "joke" in intent:
#             joke = tell_joke()  # Call the joke function
#             speak(f"Here's a joke: {joke}", language_code)

#         elif "ip" in intent:
#             ip = get_ip_address()  # Get the IP address
#             speak(f"Your IP address is: {ip}", language_code)

#         elif "movies" in intent:
#             trending_movies = get_trending_movies()  # Call the function to fetch trending movies
#             speak(f"Here are some trending movies: {trending_movies}", language_code)

#         elif intent == "open_website":
#             # Extract the website name from user input
#             website_name = text.split("open")[-1].strip()
#             response = system_tasks.open_website(website_name)
#             speak(response, language_code)

#         # Handle "open_app" intent
#         elif intent == "open_app":
#             # Extract the application name from user input
#             app_name = text.split("open")[-1].strip()
#             response = system_tasks.open_app(app_name)
#             speak(response, language_code)

#         elif "screenshot" in intent:  # Check for screenshot intent
#             system_tasks.take_screenshot()  # Take a screenshot
#             speak("Taking a screenshot.", language_code)

#         elif "close_window" in intent:
#             system_tasks.close_window()  # Close a window (e.g., Chrome)
#             speak("Closing the window.", language_code)

#         elif "email" in intent:
#             speak("Who is the recipient?", language_code)
#             recipient = input()
#             # recipient = input("Enter recipient email: ")
#             speak("What is the subject?", language_code)
#             subject = record_audio(language_code)
#             speak("What should the email say?", language_code)
#             body = record_audio(language_code)
#             success = send_email(recipient, subject, body)
#             if success:
#                 speak("Email sent successfully!", language_code)
#             else:
#                 speak("Failed to send email.", language_code)

        
        

#         elif "math" in intent:
#             speak("confirm the math problem?", language_code)
#             math_expression = record_audio(language_code)
#             math_solution = solve_math(math_expression)
#             speak(math_solution, language_code)

#         elif "internet_speedtest" in intent:
#             speed_report = check_internet_speed()
#             speak(speed_report, language_code)
        
#         elif "recognise" in intent:  # New intent for object recognition
#            speak("Starting object recognition.", language_code)
#            recognize_objects(language_code)  

#         elif "emergency" in intent:  # New intent for emergency assistance
#             speak("Activating emergency assistance.", language_code)
#             get_user_location()  # Get the user's location
#             send_sos_whatsapp()    
#             speak("Emergency alerts have been sent.", language_code)
         
#         elif "ask_gemini" in intent or "gemini" in intent or "ai" in intent:
#             if not gemini_available:
#                 speak("I'm sorry, the advanced AI features are currently unavailable.", language_code)
#                 continue
        
#             question = re.sub(r'ask gemini|gemini|ai', '', text, flags=re.IGNORECASE).strip()
           
#             if not question:
#                 speak("What would you like me to ask the AI assistant?", language_code)
#                 question = record_audio(language_code)
#                 if not question:
#                    speak("I didn't hear your question. Please try again.", language_code)
#                    continue
           
#             try:
                 
#                 speak("OK Let me construct the promt for that.", language_code) 
#                 response = ask_gemini(question, language_code)
        
#                 if response:
#             # Simple response cleaning
#                     clean_response = response.replace("**", "").replace("*", "")
#                     if len(clean_response) > 1000:  # For long responses
#                         clean_response = clean_response[:1000] + "... [response truncated]"
#                     speak(clean_response, language_code)

#                 else:
#                    speak("I couldn't get a response for that question. Please try asking differently.", language_code)
#             except Exception as e:
#                   print(f"Error in Gemini interaction: {e}")
#                   speak("I encountered an error while processing your request. Please try again later.", language_code)


#         elif "exit" in intent:
#             speak("Goodbye! Assistant is shutting down.", language_code)
#             break
        

#         else:
#             speak(get_phrase('not_understood', language_code), language_code)
    
#     assistant_running = False 

# def handle_wikipedia_intent(language_code, text):
#     """
#     Handle Wikipedia intent by directly fetching the summary of the topic.
#     Parameters:
#         language_code (str): Language for assistant communication.
#         text (str): The user's query.
#     """
#     # Extract the topic from the user input (assuming format like "Tell me about ...")
#     topic = text.replace("tell me about", "").strip()

#     if topic:
#         # Fetch the Wikipedia summary for the extracted topic
#         wikipedia_summary = get_wikipedia_summary(topic, language=language_code)
#         speak(wikipedia_summary, language_code)

    
#     else:
#         speak("Sorry, I couldn't understand the topic. Please try again.", language_code)


# LANGUAGE_PHRASES = {
#     'not_understood': {
#         'en': "Sorry, I didn't understand that.",
#         'hi': "क्षमा करें, मैं समझा नहीं।",
#         'es': "Lo siento, no entendí eso.",
#         'fr': "Désolé, je n'ai pas compris.",
#         'pa': "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਸਮਝ ਨਹੀਂ ਪਾਇਆ।",
#         'ta': "மன்னிக்கவும், அதை நான் புரிந்து கொள்ளவில்லை.",
#         'te': "క్షమించండి, నాకు అర్థం కాలేదు."
#     },
#     'try_again': {
#         # Add translations for "Please try again"
#     }
# }

# def get_phrase(phrase_key, language_code):
#     return LANGUAGE_PHRASES.get(phrase_key, {}).get(language_code, 
#            LANGUAGE_PHRASES[phrase_key]['en'])


# @eel.expose
# def end_assistant():
#     """
#     Function to stop the assistant gracefully.
#     """
#     global assistant_running
#     if assistant_running:
#         assistant_running = False
#         print("Assistant has been stopped.")
#     else:
#         print("Assistant is not running.")

# if __name__ == "__main__":
#     try:
        
#         eel.start("D:/PYTHON/AI_VOICE/Proj/www/index.html", size=(700, 600))  # Start the web page with Eel
#     except KeyboardInterrupt:
#         print("Assistant terminated.")





import io
import pygame
import speech_recognition as sr
from gtts import gTTS
from keras.models import load_model # type: ignore
from pickle import load
import numpy as np
import tensorflow as tf
import eel
from API_functionalities import get_news, get_weather, get_wikipedia_summary, get_ip_address, tell_joke, get_trending_movies, solve_math, check_internet_speed, ask_gemini, setup_gemini
from system_operations import SystemTasks  # Import SystemTasks class
from email_helper import send_email
from deep_translator import GoogleTranslator
from emergency_SMS import send_sos_whatsapp,get_user_location
import re
import webbrowser
import cv2
import torch
from torchvision import transforms
from PIL import Image
import noisereduce as nr
import sounddevice as sd
import soundfile as sf
from gps_service import GPSService
import geocoder

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Initialize SystemTasks
system_tasks = SystemTasks()

# Load the trained model
MODEL_PATH = r'D:/PYTHON/AI_VOICE/Data/chat_model.keras'
TOKENIZER_PATH = r'D:/PYTHON/AI_VOICE/Data/tokenizer.pickle'
LABEL_ENCODER_PATH = r'D:/PYTHON/AI_VOICE/Data/label_encoder.pickle'

# Initialize Gemini
gemini_available = setup_gemini()

# Initialize GPS Service
gps_service = GPSService()

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = load(handle)

with open(LABEL_ENCODER_PATH, 'rb') as enc:
    label_encoder = load(enc)

recognizer = sr.Recognizer()

# Configure recognizer settings for noise robustness
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8  # Longer pause threshold for better sentence detection
recognizer.phrase_threshold = 0.3  # Minimum audio energy to consider for recording
recognizer.non_speaking_duration = 0.5  # Duration of non-speaking audio to keep on both sides

# Variable to store the user's name globally
user_name = ""

# Initialize Eel
eel.init("D:/PYTHON/AI_VOICE/Proj/www")  # 'web' is the directory containing your HTML, JS, and CSS

# Flag to control assistant status
assistant_running = False

def apply_noise_reduction(audio_data, sample_rate):
    """Apply noise reduction to audio data using noisereduce library"""
    try:
        # Convert audio data to numpy array
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_np = np.array(audio_data.get_raw_data(), dtype=np.int16)
        
        # Convert to float32 for noise reduction
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        # Perform noise reduction
        reduced_noise = nr.reduce_noise(y=audio_float, sr=sample_rate, stationary=True)
        
        # Convert back to int16
        processed_audio = (reduced_noise * 32768).astype(np.int16)
        
        return processed_audio.tobytes()
    except Exception as e:
        print(f"Noise reduction error: {e}")
        return audio_data  # Return original if processing fails

def speak(text, language_code='en'):
    """Use gTTS to speak the given text and display it in the response box."""
    try:
        eel.showResponding()
        if user_name:
            text = f"{user_name}, {text}"
        
        # Translate the text to the selected language
        translated_text = translate_text(text, language_code)
        print(f"Assistant: {translated_text}")
        
        # Update the responses box
        eel.updateResponses(translated_text)  # Send the response to the frontend

        # Generate speech in memory using gTTS
        tts = gTTS(text=translated_text, lang=language_code)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        # Load and play audio using pygame
        pygame.mixer.music.load(audio_fp, 'mp3')
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    
    except Exception as e:
        print(f"Error in speaking: {e}")
    
    finally:
        eel.resetAnimations()

def record_audio_with_noise_reduction(language_code='en', duration=5):
    """Record audio with noise reduction and improved voice detection"""
    eel.showListening()
    try:
        with sr.Microphone() as source:
            print("Calibrating microphone for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Calibrate for 1 second
            
            recognizer.pause_threshold = 2

            # Record with voice activity detection
            print("Listening... (speak now)")
            audio = recognizer.listen(
                source, 
                timeout=3,
                phrase_time_limit=duration
            )

            # recognizer.pause_threshold = 0.8

            # Apply noise reduction
            try:
                # Save original audio for debugging
                with open("original_audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                # Apply noise reduction
                processed_audio = apply_noise_reduction(audio.get_wav_data(), audio.sample_rate)
                
                # Create new AudioData with processed audio
                audio = sr.AudioData(
                    processed_audio,
                    sample_rate=audio.sample_rate,
                    sample_width=audio.sample_width
                )
                
                # Save processed audio for debugging
                with open("processed_audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
            except Exception as e:
                print(f"Audio processing error: {e}")
                # Continue with original audio if processing fails

            # Map language codes to recognizer language codes
            recognizer_lang_map = {
                'en': 'en-IN',
                'hi': 'hi-IN',
                'es': 'es-ES',
                'fr': 'fr-FR',
                'pa': 'pa-IN',
                'ta': 'ta-IN',
                'te': 'te-IN',
                'gu': 'gu-IN',
                'mr': 'mr-IN'
            }
            
            # Get the recognizer language code
            recognizer_lang = recognizer_lang_map.get(language_code, 'en-IN')
            
            # Recognize speech in the specified language
            text = recognizer.recognize_google(audio, language=recognizer_lang)
            print(f"User (original): {text}")
            
            # If input isn't English, translate to English for processing
            if language_code != 'en':
                translated_text = translate_text(text, 'en')
                print(f"User (translated to English): {translated_text}")
                return translated_text.lower()
            return text.lower()
            
    except sr.WaitTimeoutError:
        print("Listening timed out while waiting for phrase to start")
        speak("I didn't hear anything. Please try again.", language_code)
        return None
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        speak("Sorry, I didn't catch that.", language_code)
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        speak("I'm having trouble with the speech service. Please try again later.", language_code)
        return None
    except Exception as e:
        print(f"Unexpected error in recording: {e}")
        speak("Something went wrong with the microphone. Please check your audio settings.", language_code)
        return None


    
def select_language():
    """Prompt user to select a language."""
    speak("Please select a language: English, Spanish, French, Hindi, Punjabi, Tamil, Telugu, Gujrati or Marathi.")
    language = record_audio_with_noise_reduction()  # Record the user's choice
    
    if language:
        language = language.lower()
        if language in ['english', 'spanish', 'french', 'hindi', 'punjabi', 'tamil', 'telugu', 'gujrati', 'marathi']:
            return language
        else:
            speak("Language not supported. Defaulting to English.")
            return "english"
    else:
        speak("No language selected. Defaulting to English.")
        return "english"

def translate_text(text, target_language, source_language='auto'):
    """Improved translation with better error handling"""
    try:
        if not text or text == "Translation error occurred.":
            return text
            
        # Don't translate if already in target language
        if target_language == 'en' and text.isascii():
            return text
            
        translated = GoogleTranslator(
            source=source_language,
            target=target_language
        ).translate(text)
        
        return translated if translated else text
    except Exception as e:
        print(f"Translation error ({source_language}→{target_language}): {e}")
        return text  # Return original text if translation fails

def predict_intent(text):
    """Predict intent from the user input using the chatbot model."""
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=20)  # Adjust maxlen as per your model
    pred = model.predict(padded)
    intent = label_encoder.inverse_transform([np.argmax(pred)])
    return intent[0]

def recognize_objects(language_code='en'):
    """
    Capture an image from the webcam, recognize objects, and describe them to the user.
    """
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Unable to access the webcam.", language_code)
        return

    speak("Please show the object to the camera.", language_code)
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        speak("Failed to capture image.", language_code)
        return

    # Save the captured frame as an image
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)
    cap.release()

    # Perform object detection on the captured image using YOLOv5
    results = yolo_model(image_path)  # YOLOv5 model inference
    detected_objects = results.pandas().xyxy[0]  # Get detected objects in a DataFrame

    if len(detected_objects) > 0:
        object_names = detected_objects['name'].unique()  # Get unique object names
        object_list = ", ".join(object_names)  # Convert to a comma-separated string
        speak(f"I see: {object_list}", language_code)
    else:
        speak("I couldn't detect any objects.", language_code)

    # Clean up the captured image
    import os
    os.remove(image_path)

def extract_name(text):
    """Extract name from user input."""
    # Handle various ways of saying the name
    name_patterns = [
        r"my name is (\w+)",
        r"i am (\w+)",
        r"you can call me (\w+)",
        r"(\w+)"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text.lower())
        if match:
            name = match.group(1).capitalize()  # Capitalize the first letter of the name
            return name
    return None

@eel.expose
def start_assistant():
    print("Assistant is now active.")
    """Start the assistant and interact with the user."""
    global assistant_running, user_name
    assistant_running = True  # Start the assistant

    # Ask the user to select a language first
    selected_language = select_language()
    language_code_map = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "hindi": "hi",
        "punjabi": "pa",
        "tamil": "ta",
        "telugu": "te",
        "gujrati": "gu",
        "marathi": "mr"
    }
    language_code = language_code_map[selected_language]

    speak(f" Hello! , I am Lara, your virtual assistant. Before we start, may I know your name?",language_code)
    
    # Loop to handle user response for name
    user_name = record_audio_with_noise_reduction()
    if user_name:
        user_name = extract_name(user_name)  # Extract the name from the user input
        if user_name:
            speak(f"Nice to meet you",language_code)
        else:
            user_name = "Friend"
            speak(f"Alright, I'll call you {user_name}.",language_code)
    else:
        user_name = "User"
        speak(f"Alright, I'll call you {user_name}.")

    speak(f"How can I assist you today?", language_code)

    while assistant_running:  # Loop will run while assistant_running is True
        text = record_audio_with_noise_reduction(language_code)
        if text is None:
            continue
        
        intent = predict_intent(text)
        print(f"Predicted Intent: {intent}")  # Debugging line

        if "wikipedia" in intent:
            handle_wikipedia_intent(language_code, text)
            
        elif "weather" in intent:
            speak("Please tell me the country you want to know the weather for.", language_code)
            country = record_audio_with_noise_reduction(language_code)  # Get country input from the user
            speak("Now please tell me the city.", language_code)
            city = record_audio_with_noise_reduction(language_code)  # Get city input from the user
            
            # Check if the user asks for a 14-day forecast
            if "next 14 days" in text or "14 days" in text:
                weather_report = get_weather(city, country, forecast_days=14)
            else:
                weather_report = get_weather(city, country)  # Default to current weather

            speak(weather_report, language_code)

        elif "news" in intent:
            headlines = get_news()  # Call the function to get the latest news
            speak(f"Here are the latest headlines: {headlines}", language_code)
            speak("Anything more to ask?")

        elif "joke" in intent:
            joke = tell_joke()  # Call the joke function
            speak(f"Here's a joke: {joke}", language_code)

        elif "ip" in intent:
            ip = get_ip_address()  # Get the IP address
            speak(f"Your IP address is: {ip}", language_code)

        elif "movies" in intent:
            trending_movies = get_trending_movies()  # Call the function to fetch trending movies
            speak(f"Here are some trending movies: {trending_movies}", language_code)

        elif intent == "open_website":
            # Extract the website name from user input
            website_name = text.split("open")[-1].strip()
            response = system_tasks.open_website(website_name)
            speak(response, language_code)

        elif intent == "open_app":
            # Extract the application name from user input
            app_name = text.split("open")[-1].strip()
            response = system_tasks.open_app(app_name)
            speak(response, language_code)

        elif "screenshot" in intent:  # Check for screenshot intent
            system_tasks.take_screenshot()  # Take a screenshot
            speak("Taking a screenshot.", language_code)

        elif "close_window" in intent:
            system_tasks.close_window()  # Close a window (e.g., Chrome)
            speak("Closing the window.", language_code)

        elif "email" in intent:
            speak("Who is the recipient?", language_code)
            recipient = input("Enter recipient email:")
            speak("What is the subject?", language_code)
            subject = record_audio_with_noise_reduction(language_code)
            speak("What should the email say?", language_code)
            body = record_audio_with_noise_reduction(language_code)
            success = send_email(recipient, subject, body)
            if success:
                speak("Email sent successfully!", language_code)
            else:
                speak("Failed to send email.", language_code)

        elif "math" in intent:
            speak("confirm the math problem?", language_code)
            math_expression = record_audio_with_noise_reduction(language_code)
            math_solution = solve_math(math_expression)
            speak(math_solution, language_code)

        elif "internet_speedtest" in intent:
            speed_report = check_internet_speed()
            speak(speed_report, language_code)
        
        elif "recognise" in intent:  # New intent for object recognition
           speak("Starting object recognition.", language_code)
           recognize_objects(language_code)  

        elif "emergency" in intent:  # New intent for emergency assistance
            speak("Activating emergency assistance.", language_code)
            get_user_location()  # Get the user's location
            send_sos_whatsapp()    
            speak("Emergency alerts have been sent.", language_code)
         
        elif "ask_gemini" in intent or "gemini" in intent or "ai" in intent:
            if not gemini_available:
                speak("I'm sorry, the advanced AI features are currently unavailable.", language_code)
                continue
        
            question = re.sub(r'ask gemini|gemini|ai', '', text, flags=re.IGNORECASE).strip()
           
            if not question:
                speak("What would you like me to ask the AI assistant?", language_code)
                question = record_audio_with_noise_reduction(language_code)
                
                if not question:
                   speak("I didn't hear your question. Please try again.", language_code)
                   continue
           
            try:
                speak("OK Let me construct the promt for that.", language_code) 
                response = ask_gemini(question, language_code)
        
                if response:
                    # Simple response cleaning
                    clean_response = response.replace("**", "").replace("*", "")
                    if len(clean_response) > 1000:  # For long responses
                        clean_response = clean_response[:1000] + "... [response truncated]"
                    speak(clean_response, language_code)

                else:
                   speak("I couldn't get a response for that question. Please try asking differently.", language_code)
            except Exception as e:
                  print(f"Error in Gemini interaction: {e}")
                  speak("I encountered an error while processing your request. Please try again later.", language_code)




        elif "gps" in intent:
            location = gps_service.get_current_gps_location()
            if location:
        # Speak only city and country for privacy
               response = f"You are currently in {location['city']}, {location['country']}, {location['address']}, {location['postal']}"
               speak(response, language_code)
        
        # For debugging/console
               print(f"Full GPS Info: {location}")
            else:
               speak("I couldn't access your GPS location. Please ensure location services are enabled.", language_code)



        elif "exit" in intent:
            speak("Goodbye! Assistant is shutting down.", language_code)
            break
        
        else:
            speak(get_phrase('not_understood', language_code), language_code)
    
    assistant_running = False 

def handle_wikipedia_intent(language_code, text):
    """
    Handle Wikipedia intent by directly fetching the summary of the topic.
    Parameters:
        language_code (str): Language for assistant communication.
        text (str): The user's query.
    """
    # Extract the topic from the user input (assuming format like "Tell me about ...")
    topic = text.replace("tell me about", "").strip()

    if topic:
        # Fetch the Wikipedia summary for the extracted topic
        wikipedia_summary = get_wikipedia_summary(topic, language=language_code)
        speak(wikipedia_summary, language_code)
    else:
        speak("Sorry, I couldn't understand the topic. Please try again.", language_code)

LANGUAGE_PHRASES = {
    'not_understood': {
        'en': "Sorry, I didn't understand that.",
        'hi': "क्षमा करें, मैं समझा नहीं।",
        'es': "Lo siento, no entendí eso.",
        'fr': "Désolé, je n'ai pas compris.",
        'pa': "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਸਮਝ ਨਹੀਂ ਪਾਇਆ।",
        'ta': "மன்னிக்கவும், அதை நான் புரிந்து கொள்ளவில்லை.",
        'te': "క్షమించండి, నాకు అర్థం కాలేదు."
    },
    'try_again': {
        # Add translations for "Please try again"
    }
}

def get_phrase(phrase_key, language_code):
    return LANGUAGE_PHRASES.get(phrase_key, {}).get(language_code, 
           LANGUAGE_PHRASES[phrase_key]['en'])

@eel.expose
def end_assistant():
    """
    Function to stop the assistant gracefully.
    """
    global assistant_running
    if assistant_running:
        assistant_running = False
        print("Assistant has been stopped.")
    else:
        print("Assistant is not running.")

if __name__ == "__main__":
    try:
        eel.start("D:/PYTHON/AI_VOICE/Proj/www/index.html", size=(700, 600))  # Start the web page with Eel
    except KeyboardInterrupt:
        print("Assistant terminated.")