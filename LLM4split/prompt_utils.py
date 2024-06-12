from openai import OpenAI

import os

def prompt_crepe_training():
    prompt_training = "You are a query decomposition assistant. Please decompose one long query Q into semantically coherent sub-queries, \
    each of which includes the positional information phrase of the objects. The decomposition requirements are to ensure that each sub-query after decomposition \
    contains keyword groups that reflect the original sentence's information; if the original sentence has multiple commas, \
    the nouns in the sub-queries should reflect the semantic relationship with the preceding and following objects. \
    Below is an example of Q&A pairing, where each long Q is decomposed into semantically coherent sub-queries in A (sub-queries are separated by vertical lines \"|\").  \
    Q: \"keyboard, computer monitor, printer, and fax machine on a desk, with a chair against the wall.\", \
        A: \"keyboard on a desk| computer monitor on a desk| printer on a desk| fax machine on a desk| a chair against the wall\"| \
        Q: \"woman wearing a sweater with a wrist pad on her keyboard in front of a monitor. there is a sticky note taped to the monitor and a juice bottle next to it.\", \
        A: \"woman wearing a sweater| a wrist pad on her keyboard| her keyboard in front of a monitor| a sticky note taped to the monitor| a juice bottle next to monitor.\", \
        Q: \"man with sleeves and a hand on a keyboard in front of a cpu. there are books on the cpu and a mouse next to the keyboard.\", \
        A: \"man with sleeves| a hand on a keyboard| a keyboard in front of a cpu| books on the cpu| a mouse next to the keyboard.\", \
        Q: \"shelves with books and a baby on them, against a wall with a corkboard and posters\", \
        A: \"shelves with books| a baby on books| a bady against a wall| a wall with a corkboard| a wall with posters\", \
        Q: \"computer on a desk with a mouse and cup. there is a computer tower below the desk and lint on the floor.\", \
        A: \"computer on a desk| a desk with a mouse| a desk with a cup| a computer tower below the desk|  and a computer tower below the lint| lint on the floor.\", \
        Q: \"a rolodex, desk, chair, monitor, and book on a desk. there is a picture of a baby propped up against the monitor.\", \
        A: \"a rolodex on a desk| chair on a desk| monitor on a desk| book on a desk| a picture of a baby propped up against the monitor.\", \
        Q: \"envelope, soda cup, book, and pen on a desk next to a computer monitor. there is a picture on the desk.\", \
        A: \"envelope on a desk| soda cup on a desk| book on a desk| pen on a desk| a desk next to a computer monitor| a picture on the desk.\", \
        Q: \"sign on a pole next to a man on a sidewalk. there is a parking meter and people on the sidewalk. there is a boy by the parking meter.\", \
        A: \"sign on a pole| a pole next to a man| a man on a sidewalk| a parking meter on the sidewalk| people on the sidewalk| a boy by the parking meter.\", \
        Q: \"keyboard and telephone on top of a desk made of books with a computer and mouse next to it. there is a bookshelf with books on it.\", \
        A: \"keyboard on top of a desk| telephone on top of a desk| a desk made of books| books with a computer| books with mouse. a bookshelf with books\", \
        Q: \"a man with two eyes and a torso. the man is standing in a room and has a hand.\", \
        A: \"a man with two eyes| a man with a torso| a man is standing in a room| a man has a hand.\", \
        Q: \"sign on a building with a window and balcony. the door has a frame around it, and there is a second window on the building.\", \
        A: \"sign on a building| a building with a window| a building with balcony| the door has a frame around it| a second window on the building.\", \
        Q: \"light on a car in front of a house. the car has a left brake-light, and the house has a window. there is an emblem on the back of the car, and a log on the house.\", \
        A: \"light on a car| a car in front of a house| the car has a left brake-light| the house has a window| an emblem on the back of the car| a log on the house.\", \
        Q: \"there is a tree, another tree, and another tree on the lawn. there is a person hunched over on the lawn, and a light pole in front of the building.\", \
        A: \"a tree on the lawn| another tree on the lawn| another tree on the lawn| a person hunched over on the lawn| a light pole in front of the building.\", \
        Q: \"a recliner facing a television that is next to a wall. the television is on a stand in the corner and has a sign atop it.\", \
        A: \"a recliner facing a television| a television is next to a wall| the television is on a stand| a stand in the corner| the television has a sign atop it.\", \
        "
    

    # print(prompt_training)
    return prompt_training

def prompt_flickr_training_two():
    prompt_training = "You are a query decomposition assistant. Please decompose one long query Q into semantically coherent sub-queries, \
    each of which comprises of a phrase including one subject and its action and optionally one object. The decomposition requirements are to ensure that each sub-query after decomposition \
    contains keyword groups that reflect the original sentence's information; if the original sentence has multiple commas, \
    the nouns in the sub-queries should reflect the semantic relationship with the preceding and following objects. \
    Below is an example of Q&A pairing, where each long Q is decomposed into semantically coherent sub-queries in A (sub-queries are separated by vertical lines \"|\").  \
    Q: \"keyboard, computer monitor, printer, and fax machine on a desk, with a chair against the wall.\", \
        A: \"keyboard on a desk| computer monitor on a desk| printer on a desk| fax machine on a desk| a chair against the wall\"| \
        Q: \"woman wearing a sweater with a wrist pad on her keyboard in front of a monitor. there is a sticky note taped to the monitor and a juice bottle next to it.\", \
        A: \"woman wearing a sweater| a wrist pad on her keyboard| her keyboard in front of a monitor| a sticky note taped to the monitor| a juice bottle next to monitor.\", \
        Q: \"man with sleeves and a hand on a keyboard in front of a cpu. there are books on the cpu and a mouse next to the keyboard.\", \
        A: \"man with sleeves| a hand on a keyboard| a keyboard in front of a cpu| books on the cpu| a mouse next to the keyboard.\", \
        Q: \"shelves with books and a baby on them, against a wall with a corkboard and posters\", \
        A: \"shelves with books| a baby on books| a bady against a wall| a wall with a corkboard| a wall with posters\", \
        Q: \"computer on a desk with a mouse and cup. there is a computer tower below the desk and lint on the floor.\", \
        A: \"computer on a desk| a desk with a mouse| a desk with a cup| a computer tower below the desk|  and a computer tower below the lint| lint on the floor.\", \
        Q: \"a rolodex, desk, chair, monitor, and book on a desk. there is a picture of a baby propped up against the monitor.\", \
        A: \"a rolodex on a desk| chair on a desk| monitor on a desk| book on a desk| a picture of a baby propped up against the monitor.\", \
        Q: \"envelope, soda cup, book, and pen on a desk next to a computer monitor. there is a picture on the desk.\", \
        A: \"envelope on a desk| soda cup on a desk| book on a desk| pen on a desk| a desk next to a computer monitor| a picture on the desk.\", \
        Q: \"sign on a pole next to a man on a sidewalk. there is a parking meter and people on the sidewalk. there is a boy by the parking meter.\", \
        A: \"sign on a pole| a pole next to a man| a man on a sidewalk| a parking meter on the sidewalk| people on the sidewalk| a boy by the parking meter.\", \
        Q: \"keyboard and telephone on top of a desk made of books with a computer and mouse next to it. there is a bookshelf with books on it.\", \
        A: \"keyboard on top of a desk| telephone on top of a desk| a desk made of books| books with a computer| books with mouse. a bookshelf with books\", \
        Q: \"a man with two eyes and a torso. the man is standing in a room and has a hand.\", \
        A: \"a man with two eyes| a man with a torso| a man is standing in a room| a man has a hand.\", \
        Q: \"sign on a building with a window and balcony. the door has a frame around it, and there is a second window on the building.\", \
        A: \"sign on a building| a building with a window| a building with balcony| the door has a frame around it| a second window on the building.\", \
        Q: \"light on a car in front of a house. the car has a left brake-light, and the house has a window. there is an emblem on the back of the car, and a log on the house.\", \
        A: \"light on a car| a car in front of a house| the car has a left brake-light| the house has a window| an emblem on the back of the car| a log on the house.\", \
        Q: \"there is a tree, another tree, and another tree on the lawn. there is a person hunched over on the lawn, and a light pole in front of the building.\", \
        A: \"a tree on the lawn| another tree on the lawn| another tree on the lawn| a person hunched over on the lawn| a light pole in front of the building.\", \
        Q: \"a recliner facing a television that is next to a wall. the television is on a stand in the corner and has a sign atop it.\", \
        A: \"a recliner facing a television| a television is next to a wall| the television is on a stand| a stand in the corner| the television has a sign atop it.\", \
        "
    

    # print(prompt_training)
    return prompt_training


def prompt_crepe_testing(query = None):
    # prompt_inferring_test = 'Q&A示例展示完毕，请给出以下Q的分解结果；输出格式：对于每一个Q，各回复一行分解后的子句"A=分解后的句子"，不要输出原句Q，每两行之间不要有空行。'
    prompt_inferring_test = " The Q&A example display is complete. Please provide the decomposition result for the following Q; The output format should be: for each Q, \
    reply with one line for all decomposed sub-queries which are seperated by the vertical line. "
    #reply with multiple lines for all decomposed sub-queries in which each line contains one sub-query " \"|\" "\
    

    # "\"A=decomposed sentence\", without outputting the original Q sentence, and without any blank lines between the lines. "

    # input for testing
    Q7 = "Q = \"table with photos, a vase, and a bow on it, and a chair by the table\" "
    Q8 = "Q = \"lamp stand with a desk lamp on it and duct tape. there are books on a desk and a keyboard on the table.\""
    Q9 = "Q = \"a fence surrounding a building, with a bird fountain in front of it that is covered in snow, and a gate leading into the building	\""
    Q10 = "Q = \"airplane with wheels in the middle of it and light under it, in the sky with clouds\""
    if query is None:
        query = Q7

    # prompt_inferring = 'Give the answer : Q: "%s", Q: "%s" (only give the A of each Q divided by &&)' % (Q5, Q6)
    prompt_inferring_test = prompt_inferring_test + query # + Q8 + Q9 + Q10
    
    return prompt_inferring_test

def prompt_crepe(query=None):
    prompt_training = prompt_crepe_training()
    prompt_inferring_test = prompt_crepe_testing(query=query)
    prompt_test = prompt_training + prompt_inferring_test
    return prompt_test




def prompt_flickr_training():
    prompt_training = "You are a query decomposition assistant. Please decompose one long query Q into semantically coherent sub-queries, \
    The decomposition rule is that if the original sentences express the quantities of one object, then split them into individual sub-queries by those quantity numbers \
    Below is an example of Q&A pairing, where each long Q is decomposed into semantically coherent sub-queries in A (sub-queries are separated by vertical lines \"|\").  \
    Q: \"Two young children eating a snack and playing in the grass. \" \
        A: \"One young children eating a snack and playing in the grass. | One young children eating a snack and playing in the grass. \" \
    Q: \"Two men sitting on the roof of a house while another one stands on a ladder\"\
        A: \"One men sitting on the roof of a house | One men sitting on the roof of a house | another one stands on a ladder \"\
    Q: \"Three young , White males are outside near many bushes. \" \
        A: \" One young , White males is outside near many bushes. | One young , White males is outside near many bushes. | One young , White males is outside near many bushes. \" \
    Q: \"Several men in hard hats are operating a giant pulley system .\"\
        A: \"One men in hard hats are operating a giant pulley system | One men in hard hats are operating a giant pulley system .\" \
    "
    #     each of which expresses some phrases of the objects. The decomposition requirements are to ensure that each sub-query after decomposition \
    # contains keyword groups that reflect the original sentence's information; \
    # if the original sentence has multiple commas, \
    # the nouns in the sub-queries should reflect the semantic relationship with the preceding and following objects.
    # 
    # Q: \"A child in a pink dress is climbing up a set of stairs in an entry way.\", \
    #     A: \"A child in a pink dress| A child is climbing up a set of stairs in an entry way,A child in a pink dress| A child is climbing up a set of stairs in an entry way. \", \
    # Q: \"Someone in a blue shirt and hat is standing on stair and leaning against a window\", \
    #     A: \"Someone in a blue shirt and hat| Someone is standing on stair| Someone is leaning against a window,Someone in a blue shirt and hat | Someone is standing on stair | Someone is leaning against a window \", \
    # Q: \"A man in green holds a guitar while the other man observes his shirt\", \
    #     A: \"A man in green| A man holds a guitar| a man observes another man's shirt,A man in green| A man holds a guitar| a man observes another man's shirt\", \
    # Q: \"A girl is on rollerskates talking on her cellphone standing in a parking lot\" , \
    #     A: \"A girl is on rollerskates | A girl is talking on her cellphone | A girl is standing in a parking lot,A girl is on rollerskates | A girl is talking on her cellphone | A girl is standing in a parking lot\", \
    # Q: \"A woman is sorting white tall candles as a man in a green shirt stands behind her\", \
    #     A: \"A woman is sorting white tall candles | a man in a green shirt | a man stands behind a woman,A woman is sorting white tall candles | a man in a green shirt | a man stands behind a woman\",\

    # print(prompt_training)
    return prompt_training


def prompt_flickr_testing(query = None):
    # prompt_inferring_test = 'Q&A示例展示完毕，请给出以下Q的分解结果；输出格式：对于每一个Q，各回复一行分解后的子句"A=分解后的句子"，不要输出原句Q，每两行之间不要有空行。'
    prompt_inferring_test = " The Q&A example display is complete. Please provide the decomposition result for the following Q; The output format should be: for each Q, \
    reply with one line for all decomposed sub-queries which are seperated by the vertical line. "
    #reply with multiple lines for all decomposed sub-queries in which each line contains one sub-query " \"|\" "\
    

    # "\"A=decomposed sentence\", without outputting the original Q sentence, and without any blank lines between the lines. "

    # input for testing
    Q7 = "Q = \"Two children are sitting at a table eating food\" "
    # Q8 = "Q = \"lamp stand with a desk lamp on it and duct tape. there are books on a desk and a keyboard on the table.\""
    # Q9 = "Q = \"a fence surrounding a building, with a bird fountain in front of it that is covered in snow, and a gate leading into the building	\""
    # Q10 = "Q = \"airplane with wheels in the middle of it and light under it, in the sky with clouds\""
    if query is None:
        query = Q7

    # prompt_inferring = 'Give the answer : Q: "%s", Q: "%s" (only give the A of each Q divided by &&)' % (Q5, Q6)
    prompt_inferring_test = prompt_inferring_test + query # + Q8 + Q9 + Q10
    
    return prompt_inferring_test

def prompt_flickr(query=None):
    prompt_training = prompt_flickr_training()
    prompt_inferring_test = prompt_flickr_testing(query=query)
    prompt_test = prompt_training + prompt_inferring_test
    return prompt_test

def prompt_flickr_two(query=None):
    prompt_training = prompt_flickr_training_two()
    prompt_inferring_test = prompt_flickr_testing(query=query)
    prompt_test = prompt_training + prompt_inferring_test
    return prompt_test


def obtain_response_from_gpt_utils(prompt_test):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt_test,
            }
        ],
        model="gpt-3.5-turbo",
    )

    # print(prompt_test)
    response = chat_completion.choices[0].message.content
    return response

def obtain_response_from_openai(dataset_name="crepe", query=None):

    if dataset_name == "crepe":
        prompt_test = prompt_crepe(query=query)
    elif dataset_name == "flickr":
        prompt_test = prompt_flickr(query=query)
    elif dataset_name == "flickr_two":
        prompt_test = prompt_flickr_two(query=query)

    response = obtain_response_from_gpt_utils(prompt_test)

    return response

def prompt_check_correctness(query, sub_queries_str):
    prompt_training = "You are a query decomposition assistant. For the following query Q (starting with Q), please check whether the following decomposed sub-queries \
    (starting with A and sub-queries are seperated by vertical lines \"|\") 1) can be combined to express the same meaning of Q; 2) are not redundant or unnecessary. \n \
    Please answer True or False \n \
    "
    prompt = "Q: " + query + " \n \
    A: " + sub_queries_str + " \n \
    True or False: \
    "
    
    prompt = prompt_training + prompt

    response = obtain_response_from_gpt_utils(prompt)
    
    response = bool(response.strip() == "True")
    
    return response

def update_decomposed_queries(query, sub_queries_str):
    prompt_training = "You are a query decomposition assistant. For the following query Q (starting with Q), please check whether the following decomposed sub-queries \
    (starting with A and sub-queries are seperated by vertical lines \"|\") can be combined to express the same meaning of Q. \n \
    If these sub-queries cannot collectively express the same meaning of Q, please modify those sub-queries or drop those unnecessary or redundant or incorrect sub-queries \
    such that the new set of sub-queries can collectively express the same meaning of Q. Finally only output the modified sub-queries seperated them by vertical lines \"|\" \n \
    "
    prompt = "Q: " + query + " \n \
    A: " + sub_queries_str + " \n \
    new A: \
    "
    
    prompt = prompt_training + prompt

    response = obtain_response_from_gpt_utils(prompt)
    
    return response