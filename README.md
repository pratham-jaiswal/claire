[![Read on Medium](https://img.shields.io/badge/Read_on-Medium-black?logo=medium&style=for-the-badge)](https://medium.com/@pratham52/heres-how-to-add-semantic-memory-to-ai-agents-f066b74de888)

# Claire - The AI Assistant

Claire is a agentic AI that features persistent semantic memory, allowing it to maintain context across interactions. It engages users in natural-sounding conversations by leveraging large language models. Additionally, Claire can perform certain tasks with the set of tools it has been provided with.

## Features

- **Voice Interaction**: Supports speech recognition and text-to-speech.
- **Semantic Memory**: Utilizes MongoDB Atlas Vector Search for semantic memory.
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
- **OpenAI**: For AI language model, text-to-speech, and vector embeddings.
- **MongoDB Atlas**: For semantic memory.
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
- [MongoDB Atlas URI](https://www.mongodb.com/products/platform/atlas-database) (not localhost).
- Store the API Keys in a `.env` file as shown below

    ```.env
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY
    TAVILY_API_KEY=YOUR_TAVILY_API_KEY
    AMADEUS_CLIENT_ID=YOUR_AMADEUS_CLIENT_ID
    AMADEUS_CLIENT_SECRET=YOUR_AMADEUS_CLIENT_SECRET
    OPENWEATHERMAP_API_KEY=YOUR_OPENWEATHERMAP_API_KEY
    NEWS_API_KEY=YOUR_NEWS_API_KEY
    MONGODB_URI=YOUR_MONGODB_ATLAS_URI
    ```

> **Note:** All the API keys required, apart from OpenAI, have a free tier.

### Setup

1. **Clone the Repository**  
    ```sh
    git clone https://github.com/pratham-jaiswal/claire.git
    cd claire-ai
    ```

2. **Install Dependencies**  
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Assistant**  
    ```sh
    python main.py
    ```

4. **Usage Options**  
    - You can choose to provide input via **voice or text**.  
    - You can choose to receive output as **voice, text, or both**.  

5. **Model Configuration**  
    You can modify the code to use a different OpenAI model. See the [Model Guide](https://platform.openai.com/docs/models) for available options.  
    ```python
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.5  # Controls response randomness
    )
    ```
    > **Note:** If your budget allows, it is highly recommended to use **GPT-4o** or a more advanced model. While **GPT-4o mini** offers decent performance, **GPT-4o** provides enhanced capabilities and more accurate responses, which significantly improve interaction quality. Refer to the [Pricing Guide](https://platform.openai.com/docs/pricing) for cost details.  

6. **Speech Configuration**  
    You can customize the code to use different voices, models, and audio formats. See the [Text-to-Speech Guide](https://platform.openai.com/docs/guides/text-to-speech).  
    ```python
    spoken_response = client.audio.speech.create(
        model="gpt-4o-mini-tts",  # 'gpt-4o-mini-tts' is costlier than 'tts-1' and 'tts-1-hd'
        voice="alloy",
        response_format="mp3",
        instructions="Use a warm and friendly tone",  # 'instructions' do not work with 'tts-1' or 'tts-1-hd'.
        input=text
    )
    ```

7. **Embedding Configuration**  
    You can switch to a different embedding model. Refer to the [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings).  
    ```python
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    ```
    Or, use a different embedding provider:  
    ```python
    from langchain_community.embeddings import FakeEmbeddings

    embeddings = FakeEmbeddings(size=1352)
    ```
    For detailed information on integrating various model providers with LangChain, refer to the [Integration Guide](https://python.langchain.com/docs/integrations/text_embedding/).

## Usage

Once started, Claire will listen for voice commands or accept text input. Now simply interact with Claire as you would with a human assistant.

## Example Commands
- `What's my name?`
- `What's the weather like in my city?`
- `Find me a flight from Kolkata to Mumbai.`
- `Tell me a dark programming joke.`
- `Set a reminder in 30 minutes.` (The code must be running for this to work)
- `Give me the latest news on technology.`
- `Who was the founder of reddit and what happend to him?`
- `How many times does the letter 'o' appear in the word 'pneumonoultramicroscopicsilicovolcanoconiosis'?`

## Contributing

Please read [CONTRIBUTING.md](https://github.com/pratham-jaiswal/claire/blob/main/CONTRIBUTING.md) for the process of submitting pull requests to us.

<!-- GitAds-Verify: 5GA61D76OFJQ15MFVIPMR13QPLE2SP3I -->
