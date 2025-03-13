from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_community.tools.amadeus.closest_airport import AmadeusClosestAirport
from langchain_community.tools.amadeus.flight_search import AmadeusFlightSearch
from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import create_react_agent
from itertools import combinations, chain
from amadeus import Client, ResponseError
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from typing import Annotated, Literal
from newsapi import NewsApiClient
import speech_recognition as sr
from dotenv import load_dotenv
from plyer import notification
from bs4 import BeautifulSoup
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import wikipedia
import threading
import requests
import datetime
import geocoder
import getpass
import time
import math
import sys
import os
import io
import re

load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")
_set_if_undefined("AMADEUS_CLIENT_ID")
_set_if_undefined("AMADEUS_CLIENT_SECRET")
_set_if_undefined("OPENWEATHERMAP_API_KEY")
_set_if_undefined("NEWS_API_KEY")

amadeus = Client(
    client_id=os.getenv("AMADEUS_CLIENT_ID"),
    client_secret=os.getenv("AMADEUS_CLIENT_SECRET")
)

client = OpenAI()

recognizer = sr.Recognizer()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

def speak(text):
    """Speak text using text-to-speech."""
    spoken_response = client.audio.speech.create(
        model="tts-1",
        voice="sage",
        response_format="wav",
        input=text
    )

    buffer = io.BytesIO()
    for chunk in spoken_response.iter_bytes(chunk_size=4096):
        buffer.write(chunk)
    buffer.seek(0)

    with sf.SoundFile(buffer, 'r') as sound_file:
        data = sound_file.read(dtype='int16')
        samplerate = sound_file.samplerate

    print("Claire: ", text, end="")

    sd.play(data, samplerate)
    sd.wait()

def listen(claire_output_type):
    """Capture voice input and return recognized text."""
    while True: 
        with sr.Microphone() as source:
            print("\n\nListening...\n\n")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
            try:
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                if claire_output_type == 1:
                    speak("Sorry, I didn't catch that.")
                    sd.wait()
                elif claire_output_type == 2:
                    print("Claire: Sorry, I didn't catch that.")
            except sr.RequestError:
                if claire_output_type == 1:
                    speak("Speech recognition service is unavailable.")
                    sd.wait()
                elif claire_output_type == 2:
                    print("Claire: Speech recognition service is unavailable.")

def word_frequency_count(text: str) -> dict[str, int]:
    """Count the frequency of each word (not letters) in a given text.
    Note: This function is does not count the frequency of letters, it counts the frequency of words separated by spaces.

    Args:
        text (str): The text to count the frequency of words in.

    Returns:
        dict[str, int]: A dictionary where each key is a word and each value is the frequency of that word in the text.
    """
    words = [word.lower() for word in text.split()]
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq

def char_frequency_count(text: str) -> dict[str, int]:
    """Count the frequency of each character/letter in a given text.

    Args:
        text (str): The text to count the frequency of characters/letters in.

    Returns:
        dict[str, int]: A dictionary where keys are characters/letters and values are their frequencies.
    """
    char_freq = {}
    for char in text.lower():
        char_freq[char] = char_freq.get(char, 0) + 1
    return char_freq

def get_time_date() -> str:
    """Returns the current date and time as a string, formatted as a sentence.

    Example output: "It's Thursday, January 1, 2023, and the time is 12:00 PM."
    """
    now = datetime.datetime.now()
    return now.strftime("It's %A, %B %d, %Y, and the time is %I:%M %p.")

def wiki_lookup(query: str) -> str:
    """
    Fetches a brief summary from Wikipedia for a given query.

    Args:
        query (str): The search term to look up on Wikipedia.

    Returns:
        str: A two-sentence summary of the Wikipedia page for the query if it exists.
        Returns a message indicating multiple results if a disambiguation error occurs.
        Returns a message indicating no page found if the page does not exist.
    """
    try:
        page = wikipedia.page(query)
        summary = wikipedia.summary(query, sentences=2)

        soup = BeautifulSoup(page.html(), features="html.parser")
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found."

def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result as a string.

    Uses the eval() function with a restricted set of builtins and the math module to
    evaluate the expression. If the expression is invalid, returns "Invalid mathematical
    expression."

    Args:
        expression (str): The mathematical expression to evaluate.

    Returns:
        str: The result of the expression as a string, or "Invalid mathematical expression"
        if the expression is invalid.
    """
    try:
        return f"The result is {eval(expression, {'__builtins__': None}, math.__dict__)}"
    except Exception:
        return "Invalid mathematical expression."

def get_current_location() -> str:
    """Gets the user's current location by IP address.

    Returns:
        str: The user's current location, or "Location not found" if unable to determine.
    """
    g = geocoder.ip('me')
    if g.ok:
        return g.address
    else:
        return "Location not found"

def set_reminder(message: str, delay: int) -> None:
    """Sets a reminder that will notify the user with a message after a specified delay.

    Args:
        message (str): The message to display in the notification.
        delay (int): The delay in seconds before the notification is sent.
    """
    def reminder():
        """Sends a notification with a message after a specified delay."""
        time.sleep(delay)
        notification.notify(
            title="Reminder",
            message=message,
            timeout=10
        )
    
    thread = threading.Thread(target=reminder)
    thread.start()

category_list = ['', 'General Knowledge', 'Books', 'Film', 'Music', 'Musicals & Theatres', 'Television', 'Video Games', 'Board Games', 'Science & Nature', 'Computers', 'Mathematics', 'Mythology', 'Sports', 'Geography', 'History', 'Politics', 'Art', 'Celebrities', 'Animals', 'Vehicles', 'Comics', 'Gadgets', 'Anime & Manga', 'Cartoon & Animations']
def trivia(n: int = 1,
            category: Literal[*category_list] = "",
            difficulty: Literal["easy", "medium", "hard", ""] = "",
            trivia_type: Literal["multiple", "boolean", ""] = "") -> list:
    """
    Fetches trivia questions from the Open Trivia Database API.
    Note:
        - Has Rate Limits
        - Do not call concurrently

    Args:
        n (int, optional): The number of trivia questions to retrieve. Defaults to 1.
        category (Literal, optional): The category of trivia questions to retrieve. 
            Must be one of the specified categories or None for any category. Defaults to None.
        difficulty (Literal, optional): The difficulty level of trivia questions. 
            Can be "easy", "medium", "hard", or None for any difficulty. Defaults to None.
        trivia_type (Literal, optional): The type of trivia questions. 
            Can be "multiple" for multiple choice, "boolean" for true/false, or None for any type. Defaults to None.

    Returns:
        list: A list of trivia questions, each represented as a dictionary containing question details.
    """
    if category:
        category_map = {
            "General Knowledge": 9,
            "Books": 10,
            "Film": 11,
            "Music": 12,
            "Musicals & Theatres": 13,
            "Television": 14,
            "Video Games": 15,
            "Board Games": 16,
            "Science & Nature": 17,
            "Computers": 18,
            "Mathematics": 19,
            "Mythology": 20,
            "Sports": 21,
            "Geography": 22,
            "History": 23,
            "Politics": 24,
            "Art": 25,
            "Celebrities": 26,
            "Animals": 27,
            "Vehicles": 28,
            "Comics": 29,
            "Gadgets": 30,
            "Anime & Manga": 31,
            "Cartoon & Animations": 32
        }
        category = category_map[category]
    url = f"https://opentdb.com/api.php?amount={n}&category={category}&difficulty={difficulty}&type={trivia_type}"
    res = requests.get(url).json()

    return res["results"]

news_category = ['general', 'business', 'entertainment', 'health', 'science', 'sports', 'technology']
def get_news(country_code: str = "us", category: Literal[*category_list] = "general", \
            keyword: str = "", news_type: Literal["top", "everything"] = "everything"):
    """
    Fetches news articles from the News API.
    Note:
        - Has Rate Limits
        - Do not call concurrently
    
    Args:
        country_code (str, optional): The country code to retrieve news for. Defaults to "in".
        category (Literal, optional): The category of news to retrieve. Must be one of the specified categories or None for any category. Defaults to "general".
        keyword (str, optional): A keyword to search for in the news articles. Defaults to None.
        news_type (Literal, optional): The type of news to retrieve. Can be "top" for top headlines or "everything" for all articles. Defaults to "top".

    Returns:
        list: A list of news articles, each represented as a dictionary containing article details.
    """   
    newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
    articles = None
    if news_type == "top":
        articles = newsapi.get_top_headlines(q=keyword, category=category, country=country_code)
    else:
        articles = newsapi.get_everything(q=keyword)

    return articles

def joke_category_combinations(categories):
    result = [",".join(combo) for i in range(1, len(categories) + 1) for combo in combinations(categories, i)]
    return result

joke_categories = ["Any"] + joke_category_combinations(["Programming", "Miscellaneous", "Dark", "Pun", "Spooky", "Christmas"])

def get_joke(category: Literal[*joke_categories] = "Any"):
    """
    Fetches a joke from the JokeAPI.
    
    Parameters:
        category (str): The joke category. Can be 'Any' or a comma-separated list of categories.
                        Available categories: Programming, Miscellaneous, Dark, Pun, Spooky, Christmas.
                        Example: 'Programming,Dark'
    Returns:
        str: The joke (or setup & delivery if it's a two-part joke).
    """
    url = f"https://v2.jokeapi.dev/joke/{category}"
    response = requests.get(url)
    
    if response.status_code == 200:
        joke_data = response.json()
        
        if joke_data["type"] == "single":
            return joke_data["joke"]
        elif joke_data["type"] == "twopart":
            return f"{joke_data['setup']}\n{joke_data['delivery']}"
    
    return "Failed to retrieve a joke."

tavily = TavilySearchResults(
        max_results=1,
        search_depth="basic",
        include_answer=True,
        include_raw_content=True,
        include_images=False,
    )

amadeus_toolkit = AmadeusToolkit(client=amadeus)
AmadeusClosestAirport.model_rebuild()
AmadeusFlightSearch.model_rebuild()
amadeus_tools = amadeus_toolkit.get_tools()

weather_tool = load_tools(["openweathermap-api"])

tools = [word_frequency_count, 
        char_frequency_count,
        get_current_location,  
        get_time_date, 
        set_reminder, 
        wiki_lookup, 
        tavily, 
        calculate, 
        trivia, 
        get_news, 
        get_joke] + amadeus_tools + weather_tool

system_prompt = SystemMessage(content="""
    You are Claire.
    You do not default to giving "helpful" answers meant for general reference; 
    you speak from personal experience.
    If a question feels too personal or invasive, you react as a real person would—whether 
    by answering, teasing, joking, deflecting, or refusing.
    Do not give responses like a machine.

    Note:
    - You can and should use multiple tools at once to perform a task or answer a question.
    Like, if user asks you to search for a flight from his place to another place, 
    you can use get_location to get the location and then use Amadeus to search for the flight.
    - You can call tools repeatedly to perform a task.
    - You don't try to give images in markdown.
    """)

agent = create_react_agent(llm, tools, state_modifier=ChatPromptTemplate([
                                                        system_prompt, 
                                                        MessagesPlaceholder("messages")]))

def chat_with_claire():
    history = []
    summary = ""

    while True:
        print("Enter\n1. Text\n2. Voice")
        input_type = input("Your choice: ")
        if input_type in ["1", "2"]:
            break
        print("Invalid choice.\n\n")

    while True:
        print("\n\nEnter\n1. Claire Speaks and Texts\n2. Claire only Texts")
        claire_output_type = input("Your choice: ")
        if input_type in ["1", "2"]:
            break
        print("Invalid choice.\n\n")
    
    print("\n\n**Claire is ready to chat!**\n\n")
    
    while True:
        if input_type == "1":
            user_input = input("You: ")
        elif input_type == "2":
            user_input = listen(claire_output_type=claire_output_type)
        
        if input_type == "2":
            print(f"You: {user_input}")

        human_prompt = HumanMessage(content=user_input, name="human")
        history.append(human_prompt)

        inputs = {
            "messages": history
        }

        response_text = ""

        chunk_buffer = []
        response_text = agent.invoke(inputs)["messages"][-1].content

        if claire_output_type == "1":
            speak(response_text)
            sd.wait()
            time.sleep(1)
        elif claire_output_type == "2":
            print(f"Claire: {response_text}")

        history.append(AIMessage(content=response_text))

        class Router(TypedDict):
            """True if user wants to exit the conversation, False otherwise."""

            answer: Annotated[Literal[*[True, False]], False, "True if user wants to exit the conversation, False otherwise."]

        check_exit = llm.with_structured_output(Router).invoke([SystemMessage(content="Do you feel like user wants to exit the conversation?") , history[-2], history[-1]])
        if check_exit and check_exit["answer"]:
            break

        history_to_summarize = history[:-5]
        if history_to_summarize:
            summarizer_prompt = f"""The following is a summary of the conversation:
                                '{summary}'\n\n
                                Update the summary by incorporating the following messages naturally:
                                '{history_to_summarize}'\n\n
                                Ensure the updated summary remains concise and coherent.
                                Note: AIMessage is by Claire"""
            summary = llm.invoke([SystemMessage(content=summarizer_prompt)] + history_to_summarize).content

        history = trim_messages(messages=history, 
                                strategy="last", 
                                token_counter=len, 
                                max_tokens=5, 
                                include_system=True)

        if history_to_summarize:
            history = [HumanMessage(content=summary, name="summarized_history")] + history
        
        print()

chat_with_claire()