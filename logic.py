user_info = {"city": None}


def fetch_weather(city):
    pass


def weather():
    if user_info["city"]:
        print(f"I remember that last time you used: {user_info["city"]}, should I use it for this result?")
        decision = input("y/n: ").lower()
        if decision == "y":
            fetch_weather(user_info["city"])
            return f"Fetching your weather now..."

    print("So where do you live?")
    city = input("City: ")
    user_info["city"] = city

    fetch_weather(city)

    return f"Fetching your weather now..."


def reminder():
    print("Sure, I got you covered.\nAbout what do I need to remind you?")
    topic = input("Topic: ")
    print("For what time do I set the reminder?")
    time = input("Time: ")
    return f"\nSo that's your reminder:\nTopic: {topic}\nTime: {time}"


def math():
    print("What do I need to solve?")
    problem = input("Problem: ")
    return f"Here's your answer: {eval(problem)}"


