class Base:
    def __init__(self):
        print("In Base.__init__:")
        print("self.step =", self.step)  # bound method
        print("self.step.__func__ =", self.step.__func__)  # unbound function (who really owns it)
        self._user_step = self.step
        self.step = self._wrapped_step

    def _wrapped_step(self):
        print("Calling wrapped step:")
        self._user_step()

    def step(self):
        print("Base step")


class Child(Base):
    def __init__(self):
        super().__init__()

    def step(self):
        print("Child step")

class Test:
    print("class is being defined")

    def method(self):
        print("method defined")

model = Child()
model.step()
