def greet(name):
    if name == "":
        return "Hello, World!"
    return "Hello, " + name + "!"


def main():
    print(greet("World"))
    print(greet("Alice"))
    print(greet(""))


if __name__ == "__main__":
    main()
