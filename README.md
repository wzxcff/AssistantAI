# Complete guide
1. Make sure you have python3 and pip3 installed.
```
python3 --version
```
```
pip3 --version
```

2. Download or clone repository to your local machine.
```bash
git clone https://github.com/wzxcff/AssistantAI.git
cd AssistantAI
```

3. Retrieve your OPENWEATHER_API_KEY from https://openweathermap.org/.
4. Create .env file in repository root folder, and put inside your API key. Should look like this:
```.env
OPENWEATHER_API_KEY=your_api_key
```

5. Create and activate virtual environment (if not activated already), open terminal and run:
```bash
python3 -m venv venv
```
Activate venv (run in the terminal):
  - MacOS/Linux: ```source venv/bin/activate```
  - Windows: ```venv\Scripts\activate```

6. Now using the same terminal, install project requirements:
```bash
pip install -r requirements.txt
```

7. Run main.py file, wait till model load, and then ask AI a question.
8. That's it, you're ready to go!

# Program showcase

### Weather
<img width="601" height="448" alt="image" src="https://github.com/user-attachments/assets/db9edabe-6db9-491f-819f-141d434efe00" />

### Joke
<img width="770" height="286" alt="image" src="https://github.com/user-attachments/assets/151aa54a-e7a1-4e07-a894-4b95ba3b05be" />

### Math
<img width="299" height="134" alt="image" src="https://github.com/user-attachments/assets/0c9ca5a1-416b-43f9-b4ce-269ade0fd74d" />


