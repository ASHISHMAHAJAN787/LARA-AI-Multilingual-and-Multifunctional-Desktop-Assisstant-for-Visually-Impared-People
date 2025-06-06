import requests
import os
import openai
import wikipedia
import sympy as sp
import speedtest
import wikipediaapi
import wikipedia

# APIs Setup
NEWS_API_KEY = '5ebf8d1578064bdc82828ca8c3ebdf9e'
WEATHER_API_KEY = '3ab8cb315e614da3830131614240112'
GMAIL_API_KEY = 'AIzaSyC_aBdu_sGgouib2wGhuMN9Nevuf1iw0GY'
TMDB_API_KEY = '88482e4f0655c832b93c047334edbbc2' 


def get_news():
    """Fetch the latest news headlines."""
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}'
    response = requests.get(url).json()
    
    if response['status'] == 'ok':
        articles = response['articles']
        headlines = [article['title'] for article in articles[:5]]
        return ' '.join(headlines)
    else:
        return "Sorry, I couldn't fetch the news."





def get_weather(city, country, forecast_days=1):
    """
    Fetch weather information for a given city and country.
    If forecast_days > 1, fetch the forecast for the specified number of days (up to 14).
    """
    if not city or not country:
        return "Please provide both city and country."

    try:
        # Construct the API URL with the city, country, and forecast_days
        url = f'http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city},{country}&days={forecast_days}'
        response = requests.get(url).json()

        # Handle errors from the API
        if 'error' in response:
            return f"Error fetching weather data: {response['error']['message']}"

        # Current Weather Report
        location = response['location']['name']
        country = response['location']['country']
        current_temp = response['current']['temp_c']
        condition = response['current']['condition']['text']

        # For single-day weather (current weather)
        if forecast_days == 1:
            return f"The current weather in {location}, {country} is {condition} with a temperature of {current_temp}°C."

        # For multi-day weather forecast
        forecast_data = response['forecast']['forecastday']
        forecast_report = f"Here is the 14-day weather forecast for {location}, {country}:\n"
        for day in forecast_data:
            date = day['date']
            max_temp = day['day']['maxtemp_c']
            min_temp = day['day']['mintemp_c']
            day_condition = day['day']['condition']['text']
            forecast_report += f"- {date}: {day_condition}, High: {max_temp}°C, Low: {min_temp}°C\n"

        return forecast_report.strip()

    except Exception as e:
        return f"An error occurred while fetching weather data: {e}"


# Function to get user's IP address
def get_ip_address():
    return requests.get('https://api.ipify.org').text

# # Function to generate an image from text using OpenAI's DALL·E
# def generate_image(text):
#     try:
#         response = openai.Image.create(
#             prompt=text,
#             n=1,
#             size="512x512"
#         )
#         image_url = response['data'][0]['url']
#         print(f"Image URL: {image_url}")
        
#         # Open the generated image (For MacOS/Linux use 'open', for Windows use 'os.startfile')
#         os.system(f'open {image_url}')  # For Windows, use os.startfile(image_url)
#         return image_url
#     except Exception as e:
#         print(f"Error generating image: {str(e)}")
#         return None


def get_trending_movies():
    """Fetch trending movies in India."""
    url = f'https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}&region=IN'
    response = requests.get(url).json()
    
    if response.get('results'):
        trending_movies = []
        for movie in response['results']:
            title = movie['title']
            overview = movie['overview']
            trending_movies.append(f"{title}: {overview}")
        
        return "\n".join(trending_movies[:5])  # Limit to the top 5 movies for brevity
    else:
        return "Sorry, I couldn't fetch the trending movies."



def tell_joke():
    """Fetch a random joke from the Official Joke API."""
    url = 'https://official-joke-api.appspot.com/random_joke'
    response = requests.get(url)
    
    if response.status_code == 200:
        joke_data = response.json()
        joke = f"{joke_data['setup']} - {joke_data['punchline']}"
        return joke
    else:
        return "Sorry, I couldn't fetch a joke at the moment."   
        
             


import wikipedia
import wikipediaapi

def get_wikipedia_summary(topic, language):
    """
    Fetch a summary of the given topic from Wikipedia.
    Parameters:
        topic (str): The topic to search for on Wikipedia.
        language (str): The language code for Wikipedia (default is 'en' for English).
    Returns:
        str: The summary of the topic, or an error message if the topic is not found.
    """
    try:
        # Initialize Wikipedia API with the specified language
        wiki_wiki = wikipediaapi.Wikipedia(language=language, user_agent="LaraVoiceAssistant/1.0")

        # Search for related topics using the wikipedia library
        search_results = wikipedia.search(topic)

        if search_results:
            # Use the first search result to fetch the summary
            closest_match = search_results[0]
            page = wiki_wiki.page(closest_match)

            # Check if the page exists
            if page.exists():
                return page.summary[:995]  # For brevity
            else:
                return f"Sorry, I couldn't find any information on '{closest_match}' on Wikipedia."
        else:
            return f"Sorry, I couldn't find any information on '{topic}' on Wikipedia."
    
    except Exception as e:
        return f"An error occurred while fetching information from Wikipedia: {e}"






def solve_math(expression):
    """
    Solve a mathematical expression.
    Handles expressions with various synonyms for basic arithmetic operations.
    """
    try:
        # Normalize the input
        expression = expression.lower()  # Convert to lowercase
        # Replace common terms with their mathematical symbols
        replacements = {
            "plus": "+",
            "minus": "-",
            "divided by": "/",
            "divide by": "/",
            "divide": "/",
            "times": "*",
            "cross": "*",
            "x": "*",  # Treat 'x' as multiplication
        }
        for word, symbol in replacements.items():
            expression = expression.replace(word, symbol)

        # Solve the expression
        result = sp.sympify(expression)
        return f"The result is {result}"
    except Exception as e:
        return f"Error solving the expression: {e}"


def check_internet_speed():
    """Check internet speed using speedtest-cli."""
    try:
        st = speedtest.Speedtest()
        download_speed = st.download() / 1_000_000  # Convert to Mbps
        upload_speed = st.upload() / 1_000_000  # Convert to Mbps
        return f"Your download speed is {download_speed:.2f} Mbps and your upload speed is {upload_speed:.2f} Mbps."
    except Exception as e:
        return f"Error checking internet speed: {e}"
    






from google.generativeai import configure, GenerativeModel
import google.generativeai as genai
import time

GEMINI_API_KEY = 'AIzaSyDEoRQ9wUsNwhGvphoo86LBDzZyXm8npTM'  # Replace with your actual API key

def setup_gemini():
    """Configure the Gemini API with proper settings."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Verify API connectivity
        try:
            models = genai.list_models()
            # print("Available models:", [m.name for m in models])
            return True
        except Exception as e:
            print(f"Error listing models: {e}")
            return False
            
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return False

def ask_gemini(prompt, language_code='en', max_retries=3):
    """
    Query Google Gemini with a prompt and get a response.
    
    Parameters:
        prompt (str): The question or prompt
        language_code (str): Language for response
        max_retries (int): Number of retry attempts
        
    Returns:
        str: Gemini's response or None if error
    """
    try:
        # Use one of the available models - choosing gemini-1.5-pro-latest
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        
        # Format the prompt with language context
        full_prompt = f"Respond in {language_code} language.\n\n{prompt}" if language_code != 'en' else prompt
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                response = model.generate_content(full_prompt)
                if response.text:
                    return response.text
                print(f"Empty response (attempt {attempt+1})")
                time.sleep(1)
            except Exception as e:
                print(f"API error (attempt {attempt+1}): {e}")
                time.sleep(1)
        
        return None
        
    except Exception as e:
        print(f"Error in ask_gemini: {e}")
        return None

## CHECK AVAILABLE GEMINI MODELS WITH THE API

# import google.generativeai as genai
# genai.configure(api_key='AIzaSyDEoRQ9wUsNwhGvphoo86LBDzZyXm8npTM')
# print([m.name for m in genai.list_models()])