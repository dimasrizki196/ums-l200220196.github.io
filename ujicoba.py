from metaflow import FlowSpec, step

class HelloFlow(FlowSpec):

    @step
    def start(self):
        print("Hello, Metaflow!")
        self.next(self.end)

    @step
    def end(self):
        print("Flow Complete!")

if __name__ == "__main__":
    HelloFlow() 
