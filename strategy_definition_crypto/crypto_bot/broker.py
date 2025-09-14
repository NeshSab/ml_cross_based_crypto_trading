from datetime import datetime, timezone, timedelta


class Broker:
    def __init__(self, connection):
        self.connection = connection


    def just_an_idea(self):
        print("Broker class")