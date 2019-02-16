class MyClass:
    myvar = "blah"

    def __init__(self):
        self.myvar = "initiated"

    def myfun(self):
        print("This is a message inside the clss.")

    def printMyvar(self):
        print(self.myvar)