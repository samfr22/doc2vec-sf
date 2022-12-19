class Doc:
    """
    Class represents a singular document and contains the relevant information
    """

    def __init__(self, text: list, labels: list = []):
        self.text = text
        self.labels = labels

    def __str__(self):
        text = " ".join(self.text)
        labels = " ".join(self.labels)
        return "[" + text + "]," + "[" + labels + "]"