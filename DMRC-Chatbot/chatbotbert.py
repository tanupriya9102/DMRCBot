# -*- coding: utf-8 -*-
"""ChatBotBert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/tanupriya9102/metro_cc/blob/main/ChatBotBert.ipynb
"""


import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer

import pandas as pd
import torch

import requests
import json
import pprint
import requests
import difflib
from scipy.spatial.distance import cosine
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)


df = pd.read_excel("newchatbot.xlsx")
faq_dict = {}
for index, row in df.iterrows():
    question = row["Question"]
    answer = row["Answer"]
    faq_dict[question] = answer

def get_contextual_embeddings(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    inputs = tf.constant([input_ids])
    outputs = model(inputs)
    hidden_states = outputs.last_hidden_state
    contextual_embeddings = tf.reduce_mean(hidden_states, axis=1).numpy()
    return contextual_embeddings

def find_most_similar_question(input_question, faq_dict):
    input_embeddings = get_contextual_embeddings(input_question)

    max_similarity = -1
    most_similar_question = None

    for question in faq_dict:
        question_embeddings = get_contextual_embeddings(question)

        similarity = 1 - cosine(input_embeddings.flatten(), question_embeddings.flatten())

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_question = question

    return most_similar_question, max_similarity





def find_closest_match(input_str, options):
    # print(options)
    closest_match = difflib.get_close_matches(
        input_str.title(), options, n=1, cutoff=0.8)
    # print(closest_match)
    indexOfDataFound = options.index(closest_match[0])
    # print("Index: ", indexOfDataFound)
    if indexOfDataFound is not None:
        return indexOfDataFound
    else:
        return None
    

def routeInfo():
    station = pd.read_excel("newstation.xlsx")
    d = {}
    for index, row in station.iterrows():
        name = row["NAME"]
        code= row["CODE"]
        d[name]=code

    url = "http://139.59.31.166:8000/api/v2/en/station_route/*source*/*destination*/least-distance/2023-07-12%2010:36:00.000000"
    def find_closest_match(input_str, options):
        # print(options)
        closest_match = difflib.get_close_matches(
            input_str.title(), options, n=1, cutoff=0.8)
        # print(closest_match)
        indexOfDataFound = options.index(closest_match[0])
        # print("Index: ", indexOfDataFound)
        if indexOfDataFound is not None:
            return indexOfDataFound
        else:
            return None


    fromStn = input("From: ")
    toStn = input("To: ")


    fromStnIndex = find_closest_match(fromStn, list(
        map(lambda x: x.title(), list(d.keys()))))
    # print("From Index: ", fromStnIndex)
    toStnIndex = find_closest_match(toStn, list(
        map(lambda x: x.title(), list(d.keys()))))
    # print("To Index: ", toStnIndex,type(fromStnIndex), type(toStnIndex))


    if fromStnIndex is not None:
        fromStn = list(d.values())[fromStnIndex]
    else:
        print(f"Could not find a matching station for '{fromStn}'")

    if toStnIndex is not None:
        toStn = list(d.values())[toStnIndex]
    else:
        print(f"Could not find a matching station for '{toStn}'")

    url = url.replace("*destination*", toStn)

    url = url.replace("*source*", fromStn)

    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        # pprint.pprint(data)

        stations = data['stations']
    
        total_time = data['total_time']
        fare = data['fare']
        line = (data['route'][0]['line'])
        line_number = (data['route'][0]['line_no'])
        towards_station = (data['route'][0]['towards_station'])
    
        print(
            f"No. of stations: {stations}, Total time: {total_time}, Total Fare: {fare} .")

    
    # data['route'][0]['start']
    # Region Displaying boarding station info
        input_boarding_info_askuser = input(
            'Do you want boarding station information: (yes/no) ? ')
        if input_boarding_info_askuser.lower() == 'yes':
            output_boarding_info = f"Board at station: {data['route'][0]['start']}, Towards Station: {data['route'][0]['towards_station']}, in Platform : {data['route'][0]['platform_name']} "
            print(output_boarding_info)  # output boarding information
        elif input_boarding_info_askuser.lower() == 'no':
            pass
        else:
            print('Did not match your response')
    # end region


        if int(len(data['route'])) > 1:
            input_interchange_info_askuser = input(
                'Do you want interchange station information: (yes/no) ? ')  # Displaying interchange station information
            if input_interchange_info_askuser.lower() == 'yes':
                
                outputStations=[]
                interchange_list_stationsname = "" 
                for i in range(1, len(data['route'])):
                    interchange_list_stationsname = data['route'][i]['start'] 
                    outputStations.append(interchange_list_stationsname)
    
                output_interchange_info = "No of Interchange stations are: " + \
                    str(len(data['route'])-1)+'\n'+'Namely:'
                print(output_interchange_info, outputStations)
            
                
                # print(outputStations)
            elif input_interchange_info_askuser.lower() == 'no':
                pass
            else:
                print('Did not match your response')

        # print(f"Board at station: {data['route'][0]['start']}, Towards Station:{data['route'][0]['end']}, in platform :{data['route'][0]['platform_name']} ")


    else:
        print(f"Error: {response.status_code}")
   

def main():
    similarity_threshold = 0.85

    while True:
        # Ask the user a question
        question = input("Hi! I'm DMRC bot. How May I help you? \n If query related to metro route or fare of journey type \"route\" \n To exit chatbot type \"exit\" \n else: \n Enter your question: ")

        
        if question.lower()=='route':
          print("Please Wait!!")
          print(routeInfo())
          continue


        elif question.lower() == "exit":
            print("Goodbye!")
            break



        # Find the most similar question in the dataset
        print("Please Wait!!")
        most_similar_question, similarity_score = find_most_similar_question(question, faq_dict)
        

        if most_similar_question is not None:
            
            if similarity_score < similarity_threshold:
                print("Sorry, I couldn't find a similar question in the dataset.")
            else:
                # Get the answer for the most similar question
                answer = faq_dict[most_similar_question]
                
                print(answer)
        else:
            print("Sorry, I couldn't find a similar question in the dataset.")

if __name__ == "__main__":
    main()



