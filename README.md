# Claire - The AI Assistant

Claire is an agentic AI designed for engaging conversations and performing a variety of tasks efficiently.

## Features

- **Voice Interaction**: Supports speech recognition and text-to-speech.
- **Flight Search**: Uses Amadeus API to find flights and closest airports.
- **News Fetching**: Retrieves the latest news by category and keyword.
- **Trivia & Jokes**: Engages users with fun facts and humor.
- **Weather Updates**: Provides real-time weather information.
- **Wikipedia Lookup**: Fetches brief Wikipedia summaries.
- **Reminders & Notifications**: Allows users to set reminders.
- **Math Calculations**: Evaluates mathematical expressions.
- **Web Search**: Uses Tavily Search for quick information lookup.
- **Location Services**: Determines user location based on IP.
- **Frequency Count**: Counts the frequency of words and characters in a text.

## Technologies Used

- **LangChain**: For managing AI interactions and tools.
- **OpenAI**: For AI language model, and text-to-speech.
- **SpeechRecognition**: For voice input.
- **Tavily Search**: For quick information lookup.
- **Amadeus API**: Flight and airport search.
- **NewsAPI**: Fetching latest news articles.
- **OpenWeatherMap**: Providing weather updates.
- **Wikipedia Library**: Fetching Wikipedia summaries.
- **Plyer**: For notifications and reminders.
- **Geocoder**: Fetching user location.
- **Open Trivia Database API**: Fetching trivia questions.
- **JokeAPI**: Fetching jokes.

## Installation

### Prerequisites (Python 3.13.1)

- API keys for [OpenAI](https://platform.openai.com/api-keys), [Tavily](https://app.tavily.com/home), [Amadeus](https://developers.amadeus.com/get-started/get-started-with-self-service-apis-335), [OpenWeatherMap](https://openweathermap.org/api), and [NewsAPI](https://newsapi.org/register)
- Store the API Keys in a `.env` file as shown below

    ```.env
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY
    TAVILY_API_KEY=YOUR_TAVILY_API_KEY
    AMADEUS_CLIENT_ID=YOUR_AMADEUS_CLIENT_ID
    AMADEUS_CLIENT_SECRET=YOUR_AMADEUS_CLIENT_SECRET
    OPENWEATHERMAP_API_KEY=YOUR_OPENWEATHERMAP_API_KEY
    NEWS_API_KEY=YOUR_NEWS_API_KEY
    ```

> **Note:** All the API keys required, apart from OpenAI, have a free tier.

### Setup

- Clone the repository
    ```sh
    git clone https://github.com/pratham-jaiswal/claire.git
    cd claire-ai
    ```

- Install dependencies
    ```
    pip install -r requirements.txt
    ```
- Run the assistant
    ```
    python main.py
    ```
- You can choose to give voice or text input.
- You can choose to receive voice and text or just text output.
- You can modify the code to use a different OpenAI model. ([Guide](https://platform.openai.com/docs/models))
    ```py
    llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5
        )
    ```
    
    > **Note**: If the budget allows, I'll highly recommended to utilize GPT-4o or a more advanced model. While GPT-4o mini offers decent performance, GPT-4o provides enhanced capabilities and more accurate responses, which significantly improve the quality of interactions. [Pricing guide](https://platform.openai.com/docs/pricing).
- You can modify the code to use a different voices, models, and audio formats. ([Guide](https://platform.openai.com/docs/guides/text-to-speech))
    ```py
    spoken_response = client.audio.speech.create(
            model="tts-1-hd", # HD costs more than standard
            voice="alloy",
            response_format="mp3",
            input=text
        )
    ```

## Usage

Once started, Claire will listen for voice commands or accept text input. Now simply interact with Claire as you would with a human assistant.

## Example Commands
- `What's the weather like in my city?`
- `Find me a flight from Kolkata to Mumbai.`
- `Tell me a dark programming joke.`
- `Set a reminder in 30 minutes.` (The code must be running for this to work)
- `Give me the latest news on technology.`
- `Who was the founder of reddit and what happend to him?`
- `How many times does the letter 'o' appear in the word 'pneumonoultramicroscopicsilicovolcanoconiosis'?`

## Contributing

Please read [CONTRIBUTING.md](https://github.com/pratham-jaiswal/claire/blob/main/CONTRIBUTING.md) for the process of submitting pull requests to us.