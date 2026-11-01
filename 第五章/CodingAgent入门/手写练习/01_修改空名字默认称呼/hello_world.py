def greet(name):
    if name == "":
        return "Hello, 匿名用户!"
    return "Hello, " + name + "!"


def main():
    print(greet("World"))
    print(greet("Alice"))
    print(greet(""))


if __name__ == "__main__":
    main()
