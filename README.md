# GrokIsThisTrue Project

## Overview
GrokIsThisTrue is a Twitch bot built using the Twitchio library. The bot is designed to respond to specific commands in Twitch chat, utilizing Groq functionality for processing queries.

## Project Structure
```
GrokIsThisTrue
├── src
│   ├── bot.py            # Main implementation of the Twitchio bot
│   ├── groq_handler.py   # Implementation of Groq functionality
│   └── __init__.py      # Marks the src directory as a Python package
├── .env                  # Environment variables for the bot
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Setup Instructions
1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd GrokIsThisTrue
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   Create a `.env` file in the root directory and add your Twitch bot access token:
   ```
   AccessToken=your_access_token_here
   ```

## Usage
To run the bot, execute the following command:
```
python src/bot.py
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.