import requests

class BaseClient:
    API_URL = 'https://api.beta.ons.gov.uk/v1/datasets'

    def __init__(self):
        self.API_URL = API_URL
