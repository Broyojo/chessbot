import os

files = os.listdir("raw_data")

for file in files:
    games = []

    with open("raw_data/" + file, "r") as f:
        text = f.read()
        sections = text.split("\n\n")

        for i, section in enumerate(sections):
            if i % 2 == 1:
                games.append(section.replace("\n", " "))

    with open("data/" + file, "a+") as f:
        for game in games:
            f.write(game + "\n")
        print(f"wrote: {file}")
