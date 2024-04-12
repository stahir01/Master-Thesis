import requests
from bs4 import BeautifulSoup
import pandas as pd
from googlesearch import search




def extract_aircraft_data(aircraft_name):

    url = f"https://en.wikipedia.org/wiki/{aircraft_name}"

    response = requests.get(url)
    html_content = response.text
    ##print(response.text)

    soup = BeautifulSoup(html_content, 'html.parser')
    
    columns = [th.text.strip() for th in soup.find_all('th')]
    #print(columns)

    aircraft_data = {}


def search_parameters(aircraft_model, parameters):
    search_query = f"What is the {parameters} of {aircraft_model}"
    print(search_query)

    # Encode the search query for use in the URL
    encoded_query = requests.utils.quote(search_query)
    
    # URL of the Google search results page
    url = f"https://www.google.com/search?q={encoded_query}"
    
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find the answer box or relevant search results
        answer_box = soup.find("div", class_="Z0LcW XcVN5d")
        
        # Extract the answer text
        if answer_box:
            answer_text = answer_box.text.strip()
            print(f"{parameter} of {aircraft_model}: {answer_text}")
        else:
            print("No answer found in search results.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")









if __name__ == "__main__":
    aircraft_name = "Boeing_747"
    parameter = "Height"
    search_parameters(aircraft_name, parameter)
   # extract_aircraft_data(aircraft_name)
    
    