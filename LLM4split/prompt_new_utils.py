import openai
import csv
import time
import re

def prompt_crepe_training():
    prompt_training = "You are a query decomposition assistant. Please decompose one long query Q into semantically coherent sub-queries, \
    each of which includes the positional information phrase of the objects. The decomposition requirements are to ensure that each sub-query after decomposition \
    contains keyword groups that reflect the original sentence's information; if the original sentence has multiple commas, \
    the nouns in the sub-queries should reflect the semantic relationship with the preceding and following objects. \
        **For each subquery (the subquery number starts from 0), if the nouns of the decomposed subqueries have overlapping and dependent relationships, \
        they are divided into the same group, otherwise they are divided into other groups. Use \[number1, number2, ... \] to indicate the grouping.**\
    Below is an example of Q&A pairing, where each long Q is decomposed into semantically coherent sub-queries in A (sub-queries are separated by vertical lines \"|\").  \
    Q: \"keyboard, computer monitor, printer, and fax machine on a desk, with a chair against the wall.\", \
        A: \"keyboard on a desk| computer monitor on a desk| printer on a desk| fax machine on a desk| a chair against the wall\"| ## [0,1,2,3],[4] ##\
    Q: \"woman wearing a sweater with a wrist pad on her keyboard in front of a monitor. there is a sticky note taped to the monitor and a juice bottle next to it.\", \
        A: \"woman wearing a sweater| a wrist pad on her keyboard| her keyboard in front of a monitor| a sticky note taped to the monitor| a juice bottle next to monitor. \" ## [0,1,2,3,4] ##, \
    Q: \"man with sleeves and a hand on a keyboard in front of a cpu. there are books on the cpu and a mouse next to the keyboard.\", \
        A: \"man with sleeves| a hand on a keyboard| a keyboard in front of a cpu| books on the cpu| a mouse next to the keyboard.\" ## [0,1,2,3,4] ##, \
    Q: \"shelves with books and a baby on them, against a wall with a corkboard and posters\", \
        A: \"shelves with books| a baby on books| a bady against a wall| a wall with a corkboard| a wall with posters\" ## [0,1,2,3,4] ##, \
    Q: \"computer on a desk with a mouse and cup. there is a computer tower below the desk and lint on the floor.\", \
        A: \"computer on a desk| a desk with a mouse| a desk with a cup| a computer tower below the desk|  and a computer tower below the lint| lint on the floor.\" ## [0,1,2,3,4],[5] ##, \
    Q: \"a rolodex, desk, chair, monitor, and book on a desk. there is a picture of a baby propped up against the monitor.\", \
        A: \"a rolodex on a desk| chair on a desk| monitor on a desk| book on a desk| a picture of a baby propped up against the monitor.\" ## [0,1,2,3,4] ##, \
    Q: \"Two children are playing in a pile of hay with the sun shining overhead.\", \
        A: \"Two children are playing| playing in a pile of hay| hay with the sun shining overhead.\" ## [0,1,2] ##, \
    Q: \"Two guys one in red and the other in black sitting at computers\", \
        A: \"Two guys sitting at computers| one in red| the other in black.\" ## [0,1,2] ##, \
    "
    

    # print(prompt_training)
    return prompt_training

def prompt_crepe_testing(query = None):
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

def prompt_count_training():
    prompt_training = "You are a query decomposition assistant. Please decompose one long query Q into semantically coherent sub-queries, \
    The decomposition rule is that if the original sentences express the quantities of one object, then split them into individual sub-queries by those quantity numbers.\
    ** When the long query has multiple quantifiers, it is decomposed into subqueries where the first quantifier is 1, and keep the other quantifier.** \
    When the long query represents quantity words such as some, lots of, or other approximate numbers, it is decomposed into dentical subqueries where the quantifier is 1, **and the number of subqueries is 2.**\
    Below is the examples of Q&A pairing, where each long Q is decomposed into semantically coherent sub-queries in A. \
    The input format is: Q: sentence [number1], where number1 represents the number of identical clauses, and the output is: sentence [number2], where number2 represents the number of identical subqueries. \
    Noted that number2 has no relationship with number1, number2 equals quantity numbers when decomposing \
    Q: \"Two young children eating a snack and playing two balls in the grass. [1] \" \
        A: \"One young children eating a snack and playing two balls in the grass. [2] \" \
    Q: \"A boy is playing two footballs. [3]\"\
        A: \"A boy is playing one football [2]\" \
    Q: \"Two men sitting on the roof of a house while another one stands on a ladder. [1]\"\
        A: \"One man sitting on the roof of a house while another one stands on a ladder. [2]\"\
    Q: \"Three young , White males are outside near many bushes. [1]\" \
        A: \" One young , White males is outside near many bushes. [3]\" \
    Q: \"Some men in hard hats are operating a giant pulley system. [1]\"\
        A: \"One man in hard hats are operating a giant pulley system [2]\" \
    Q: \"men dressed in cowboy boots and hats are hanging around. [1]\"\
        A: \"one man dressed in cowboy boots and hats is hanging around [2]\" \
    Q: \"People on the beach look up to the sky. [1]\"\
        A: \"a person on the beach look up to the sky [2]\" \
    Q: \"Several People on the beach look up to the sky. [1]\"\
        A: \"a person on the beach look up to the sky [2]\" \
    "
    return prompt_training

def prompt_count_testing(query = None):
    prompt_inferring_test = " The Q&A example display is complete. Please provide the decomposition result for the following Q; The output format should be: for each Q, \
    reply with one line for decomposed sub-queries with the format of : sentence [number2], where number2 represents the number of identical subqueries.. "
    #reply with multiple lines for all decomposed sub-queries in which each line contains one sub-query " \"|\" "\
    
    # input for testing
    QX1 = "Q = \"Two children are sitting at a table eating food. [1]\" " # ok
    QX2 = "Q = \"Two girls and a man were peeling corn.[1]\"" # ok 
    QX3 = "Q = \"A boy jump kicking over three kids kicking wood during a tae kwon do competition.[1]\"" #ok
    QX4 = "Q = \"Lots of people in different costumes at some kind of party.[1] \"" # ok 2 times
    QX5 = "Q = \"Some people in different costumes at some kind of party.[1] \"" # ok 2 times
    QX6 = "Q = \"four people in different costumes at some kind of party.[1] \"" # ok 4 times
    QX7 = "Q = \"A woman standing with 3 other people in a store with two tables, some shelves with coffee and tea for sale, and a refrigerated drink case.[1]\"" # not so good , 4 times
    QX8 = "Q = \"A couple of several people sitting on a ledge overlooking the beach [1]\"" # ok 2 times
    QX9 = "Q = \"Many shoppers are walking around in a mall. [1]\"" # ok 2 times
    QX10 = "Q = \"A group of people getting ready to rollerskate. [1]\"" # ok 2 times
    QX11 = "Q = \"A child practices hitting a baseball into a net, while two adult men watch. [1]\"" # ok
    QX12 = "Q = \"Doctors wearing green scrubs are performing surgery on a patient [1]\"" 
    if query is None:
        query = QX12

    # prompt_inferring = 'Give the answer : Q: "%s", Q: "%s" (only give the A of each Q divided by &&)' % (Q5, Q6)
    prompt_inferring_test = prompt_inferring_test + query
    
    return prompt_inferring_test

def prompt_seg_training():
    prompt_training = "You are a query decomposition assistant. Please decompose one long query Q into semantically coherent sub-queries, \
    each of which includes the situations information phrase of the objects. \
    The decomposition requirements are to ensure that each sub-query after decomposition \
    contains keyword groups that reflect the original sentence's information; if the original sentence has multiple commas, \
    the nouns in the sub-queries should reflect the semantic relationship with the preceding and following objects. \
    **When decomposing, replace pronouns such as it and them with semantically appropriate noun**\
    **Important rules: If a noun is not singular (e.g., \"two women\", \"some guys\"), directely extract the single sentence related to the noun without any changes of the clause (excluding clauses and nouns that are parallel to it).\
    Encapsulate the extracted sentence with ## symbols at the beginning and end.\
    For singular nouns (e.g., \"A man\", \"desk\", \"table\"), decompose them into multiple clauses that describe different actions of a person or states of an object.**\
    **If the following situation appears in the original query sentence: the following singular/plural quantity noun is an explanation of the preceding plural quantity noun\
    , you can directly decompose it according to the following singular/plural quantity noun, see the example for details.**\
    Below is an example of Q&A pairing, where each long Q is decomposed into semantically coherent sub-queries in A (sub-queries are separated by vertical lines \"|\").  \
    The number [number] after each clause in A is the same as the number [number] after the query in input Q.\
    Q: \"a woman and three men wearing sweaters with a wrist pad on their keyboards in front of a monitor. \", \
        A: \"a woman wearing a sweater [1]| a woman whit a wrist pad on her keyboard[1]| keyboards in front of a monitor [1]| ## three men wearing sweaters with a wrist pad on their keyboards in front of a monitor.[1] ##\", \
    Q: \"man with sleeves and a hand on a keyboard in front of a cpu. there are books on the cpu and a mouse next to the keyboard. [1]\", \
        A: \"man with sleeves [1]| a hand on a keyboard [1]| a keyboard in front of a cpu [1]| ## books on the cpu [1] ## | a mouse next to the keyboard. [1]\", \
    Q: \"Two kids with helmets playing with Nerf swords while one looks on drinking from a plastic cup and a little toddler boy with a flower on hands is crying while walking.[1]\", \
        A: \"## Two kids with helmets playing with Nerf swords [2] ## | one kid looks on drinking from a plastic cup [1] | a little toddler boy is crying [1]| a little toddler boy is walking [1]| a little toddler boy with a flower on hands [1]\", \
    Q: \"Two boys wearing matching blue polo shirts strike a goofy pose. [1]\", \
        A: \"## Two boys wearing matching blue polo shirts strike a goofy pose. [1] ##\", \
    Q: \"Two boys, one with a yellow and orange ball, play in some water in front of a field.\", \
        A: \"## Two boys play in some water in front of a field [1] ## | one boy with a yellow and orange ball [1]\", \
    Q: \"two men, one playing games, the other watching.[2]\", \
        A: \"One man is playing games. [1] | One man is watching. [1]\", \
    Q: \"two men on the grass while the left one if singing and the right one is dancing.[2]\", \
        A: \"## two men on the grass [1] ## | One man on the left is singing. [1] | One man on the right is dancing. [1]\", \
    **Q: \"Three small dogs, two white and one black and white, on a sidewalk.[1]\", \
        A: \"## Three small dogs on a sidewalk. [1] ## | ## two white small dogs on a sidewalk [1] ## | one black and white small dog on a sidewalk. [1] \", **\
    **Q: \"two boys running while one in blue shirt.[2]\", \
        A: \"## two boys running [1] ## | one boy in blue shirt. [1] \", **\
    Q: \"Two young girls are playing; the younger one is climbing.\", \
        A: \"## Two young girls are playing [1] ## | the younger girl is climbing [1]\", \
    Q: \"Three young Afro-American ladies on a subway train, while one is on the phone.\", \
        A: \"## Three young Afro-American ladies on a subway train [1] ## |  one young Afro-American lady is on the phone\", \
    Q: \"a group of three men and one woman are playing on the playground.\", \
        A: \"## three men are playing on the playground [1] ## | one woman is playing on the playground.[1]\", \
    Q: \"Four men, two of them with guitars, perform on a stage while several people with their hands in the air watch.\", \
        A: \"## Four men perform on a stage [1] ## | ## two men with guitars [1]##| ## several people with their hands in the air watch [1]## \", \
    Q: \"There are people mulling around and going about their business at what appears to be a street fair, or a ballgame.\", \
        A: \"## people mulling around and going about their business [1] ## | person at a street fair or a ballgame [1] \", \
    Q: \"Several men dressed in cowboy boots and hats are hanging around.\", \
        A: \"## men dressed in cowboy boots and hats are hanging around [1] ## \", \
    Q: \"People on the beach look up to the sky while a boy is crying.\", \
        A: \"## People on the beach look up to the sky [1] ## | a boy is crying [1] \", \
        "

# def prompt_seg_training():
#     prompt_training = "You are a query decomposition assistant. Please decompose one long query Q into semantically coherent sub-queries, \
#     each of which includes the situations information phrase of the objects. The decomposition requirements are to ensure that each sub-query after decomposition \
#     contains keyword groups that reflect the original sentence's information; if the original sentence has multiple commas, \
#     the nouns in the sub-queries should reflect the semantic relationship with the preceding and following objects. \
#     **Extract complete sentences with numerals or subjects indicating quantity**\
#     Below is an example of Q&A pairing, where each long Q is decomposed into semantically coherent sub-queries in A (sub-queries are separated by vertical lines \"|\").  \
#     The number [number] after each clause in A is the same as the number [number] after the query in input Q.\
#     Q: \"keyboard, computer monitor, printer, and fax machine on a desk, with a chair against the wall. [1]\", \
#         A: \"keyboard on a desk [1]| computer monitor on a desk [1]| printer on a desk [1]| fax machine on a desk [1]| a chair against the wall [1]\"| \
#     Q: \"woman wearing a sweater with a wrist pad on her keyboard in front of a monitor. there is a sticky note taped to the monitor and a juice bottle next to it.[1]\", \
#         A: \"woman wearing a sweater [1]| a wrist pad on her keyboard [1]| her keyboard in front of a monitor [1]| a sticky note taped to the monitor [1]| a juice bottle next to monitor.[1]\", \
#     Q: \"man with sleeves and a hand on a keyboard in front of a cpu. there are books on the cpu and a mouse next to the keyboard. [1]\", \
#         A: \"man with sleeves [1]| a hand on a keyboard [1]| a keyboard in front of a cpu [1]| books on the cpu [1]| a mouse next to the keyboard. [1]\", \
#     Q: \"shelves with books and a baby on them, against a wall with a corkboard and posters [2]\", \
#         A: \"shelves with books [2]| a baby on books [2]| a bady against a wall [2]| a wall with a corkboard [2]| a wall with posters [2]\", \
#     Q: \"computer on a desk with a mouse and cup. there is a computer tower below the desk and lint on the floor. [3]\", \
#         A: \"computer on a desk [3]| a desk with a mouse [3]| a desk with a cup [3]| a computer tower below the desk [3]|  and a computer tower below the lint [3]| lint on the floor. [3]\", \
#         "    

    # print(prompt_training)
    return prompt_training

def prompt_seg_testing(query = None):
    prompt_inferring_test = " The Q&A example display is complete. Please provide the decomposition result for the following Q; The output format should be: for each Q, \
    reply with one line for all decomposed sub-queries which are seperated by the vertical line. \
    ** The number [number] after each clause in A is the same as the number [number] after the query in input Q. **"
    #reply with multiple lines for all decomposed sub-queries in which each line contains one sub-query " \"|\" "\
    

    # "\"A=decomposed sentence\", without outputting the original Q sentence, and without any blank lines between the lines. "

    # input for testing
    Q7 = "Q = \"table with photos, a vase, and a bow on it, and a chair by the table\" "
    Q8 = "Q = \"lamp stand with a desk lamp on it and duct tape. there are books on a desk and a keyboard on the table.\""
    Q9 = "Q = \"a fence surrounding a building, with a bird fountain in front of it that is covered in snow, and a gate leading into the building	\""
    Q10 = "Q = \"airplane with wheels in the middle of it and light under it, in the sky with clouds\""
    if query is None:
        query = Q7

    prompt_inferring_test = prompt_inferring_test + query 
    
    return prompt_inferring_test


def decompose_sentence(prompt_training, prompt_testing):
    messages = [
        {"role": "system", "content": prompt_training},
        {"role": "user", "content": prompt_testing}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    
    return response['choices'][0]['message']['content']



QX1 = "Q = \"Two young guys with shaggy hair look at their hands while hanging out in the yard. [1]\" " # ok
QX2 = "Q = \"Two girls and a man were peeling corn.[1]\"" # ok 
QX3 = "Q = \"A boy jump kicking over three kids kicking wood during a tae kwon do competition.[1]\"" #ok
QX4 = "Q = \"Lots of people in different costumes at some kind of party.[1] \"" # ok 2 times
QX5 = "Q = \"Some people in different costumes at some kind of party.[1] \"" # ok 2 times
QX6 = "Q = \"four people in different costumes at some kind of party.[1] \"" # ok 4 times
QX7 = "Q = \"A woman standing with 3 other people in a store with two tables, some shelves with coffee and tea for sale, and a refrigerated drink case.[1]\"" # not so good , 4 times
QX8 = "Q = \"A couple of several people sitting on a ledge overlooking the beach [1]\"" # ok 2 times
QX9 = "Q = \"Many shoppers are walking around in a mall. [1]\"" # ok 2 times
QX10 = "Q = \"A group of people getting ready to rollerskate. [1]\"" # ok 2 times
QX11 = "Q = \"A child practices hitting a baseball into a net, while two adult men watch. [1]\"" # ok
QX12 = "Q = \"Doctors wearing green scrubs are performing surgery on a patient [1]\"" 

# results = []

def run_dataset_test_origin():
    with open('flickr_1000.csv', 'r', encoding='utf-8') as infile:
        
        reader = csv.DictReader(infile)
        
        # 读取前500条数据并传入test函数
        for i, row in enumerate(reader):
            print(i)
            if i <= 10: continue
            if i >= 500:
                break
            
            # step1 : count split
            input = row['first_sentence']
            prompt_training_s1 = prompt_count_training()
            prompt_testing_s1 = prompt_count_testing(input)
            result1 = decompose_sentence(prompt_training_s1,prompt_testing_s1)
            print("split1:")
            print(result1)

            # step2 : phrase split
            prompt_training_s2 = prompt_seg_training()
            prompt_testing_s2 = prompt_seg_testing(result1)
            result2 = decompose_sentence(prompt_training_s2,prompt_testing_s2)
            print("split2:")
            print(result2)

            with open('flickr_result.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(result2 + '\n')

            time.sleep(10) # reduce the frequency of asking GPT


def extract_sentences(sentence):
    """get count-parts and non-count-part from sentence after seg-decomposition(step1).

    :param sentence: sentence after seg-decomposition(step1).
    :return result1: count-parts (list), 'no sentence' for no count-parts
    :return result2: non-count-part, 
    """
    result1 = []
    result2 = sentence

    while '##' in result2:
        start = result2.find('##')
        end = result2.find('##', start + 2)
        if end == -1:
            break
        result1.append(result2[start + 2:end])       
        result2 = result2[:start] + result2[end + 2:]

    if not result1:
        result1.append('no sentence')

    return result1, result2

# final_sub_query = ""
def decompose_framework(q_input, node_num, final_sub_query):
    """decompose input sentence into sub-query.

    :param q_input: sentence to be decomposed.
    :return final_sub_query: final sub-query after whole decomposing process
    """
    # node_num  = node_num + 1
    child_list = []
    query_prefix = "Q = " 
    query = query_prefix + q_input

    # decompose 1 - seg-decompose
    prompt_training = prompt_seg_training()
    prompt_testing = prompt_seg_testing(query)
    seg_result = decompose_sentence(prompt_training,prompt_testing)
    # if '##' in seg_result:
    #     start = seg_result.find('##')
    #     seg_result = "##"+"(" + str(node_num) + ")"+ seg_result[start+2:]
    # else:
    #     seg_result = "(" + str(node_num) + ")" + seg_result
    # node_num = node_num + 1
    print('decompose 1 ---')
    print(seg_result)

    # extract 1 - get cnt and non-cnt part
    cnt_part, non_cnt_part = extract_sentences(seg_result)
    cleaned = re.sub(r'[ \"#(){}<>\[\]]', '', non_cnt_part) # cleaning
    if cleaned != "":
        non_cnt_part = "(" + str(node_num) + ")" + ' { ' + non_cnt_part
        child_list.append(node_num)
        node_num = node_num + 1
        final_sub_query = final_sub_query + non_cnt_part + ' } '
    print('extract 1 ---')
    print(cnt_part)
    print(non_cnt_part)


    if(cnt_part[0] == 'no sentence'):
        print("in no sentence")
    # decompose 2 or more
    else:
        # final_sub_query_t = ""
        print('decompose 2 ---')
        for cnt_query in cnt_part:
            prompt_training = prompt_count_training()
            prompt_testing = prompt_count_testing(cnt_query)
            cnt_result = decompose_sentence(prompt_training,prompt_testing)
            cnt_result = "(" + str(node_num) + ")" + '{' + cnt_result
            child_list.append(node_num)
            node_num = node_num + 1
            
            print(cnt_result)
            # recursively
            cur_child_list = []
            final_sub_query, cur_child_list= decompose_framework(cnt_result, node_num, final_sub_query)

            tmp_sub_query =  cnt_result + '-><'
            for num in cur_child_list:
                tmp_sub_query = tmp_sub_query + str(num) + ','
            tmp_sub_query = tmp_sub_query + '> } '

            final_sub_query = tmp_sub_query + final_sub_query

            # final_sub_query_t = final_sub_query + tmp_sub_query
        
        # final_sub_query = final_sub_query_t + non_cnt_part
        
    
    return final_sub_query, child_list

def run_simple_seg_split_test():
    query_prefix = "Q = "

    q1 = query_prefix + "\"A man and two women are playing a boardgame at a wooden table while drinking alcohol.\""
    q2 = query_prefix + "\"A young girl wearing a blue dress runs down a gravel road with a teddy bear.\""
    q3 = query_prefix + "\"Three women dressed in plain clothes are cooking a meal in their kitchen.\""
    q4 = query_prefix + "Several women are gather around a table in a corner surrounded by bookshelves."


    prompt_training = prompt_crepe_training()
    prompt_testing = prompt_crepe_testing(q4)
    result = decompose_sentence(prompt_training,prompt_testing)

    result1, result2 = extract_dependency(result)
    return result1, result2

def run_single_query_test():
    query_prefix = "Q = "

    q1 = query_prefix + "\"A man and two women are playing a boardgame at a wooden table while drinking alcohol.\""
    q2 = query_prefix + "\"A young girl wearing a blue dress runs down a gravel road with a teddy bear.\""
    q3 = query_prefix + "\"Three women dressed in plain clothes are cooking a meal in their kitchen.\""
    q4 = query_prefix + "three man are playing with two dogs"

    q5 = query_prefix + "Several women are gather around a table in a corner surrounded by bookshelves."
    q6 = query_prefix + "Three young Afro-American ladies on a subway train, one of them is on the phone."
    q7 = query_prefix + "Four men, two of them with guitars, perform on a stage while several people with their hands in the air watch."

    query = q7
    final_sub_query, child_list = decompose_framework(query, 1, "")

    tmp_sub_query = "(0){" + query + '-><'
    for num in child_list:
        tmp_sub_query = tmp_sub_query + str(num) + ','
    tmp_sub_query = tmp_sub_query + '> } '

    final_sub_query = tmp_sub_query + final_sub_query

    return final_sub_query

def run_dataset_test():
    with open('flickr_500_choose.csv', 'r', encoding='utf-8') as infile:
        
        reader = csv.DictReader(infile)
        
        for i, row in enumerate(reader):
            print(i)
            if i < 126: continue
            if i >= 127:
                break
        
            query = row['choose']
            print(query)
            final_sub_query, child_list = decompose_framework(query, 1, "")

            tmp_sub_query = "(0){" + query + '-><'
            for num in child_list:
                tmp_sub_query = tmp_sub_query + str(num) + ','
            tmp_sub_query = tmp_sub_query + '> } '

            final_sub_query = tmp_sub_query + final_sub_query
            print(final_sub_query)

            with open('flickr_100_300_result.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(final_sub_query + '\n')

            time.sleep(4) # reduce the frequency of asking GPT

def extract_dependency(sentence):
    """get count-parts and non-count-part from sentence after seg-decomposition(step1).

    :param sentence: sentence after seg-decomposition(step1).
    :return result1: count-parts (list), 'no sentence' for no count-parts
    :return result2: non-count-part, 
    """
    result1 = ""
    result2 = sentence

    while '##' in result2:
        start = result2.find('##')
        end = result2.find('##', start + 2)
        if end == -1:
            break
        result1 = result2[start + 2:end]     
        result2 = result2[:start] + result2[end + 2:]

    return result1, result2

def run_dataset_simple_seg_split():

    with open('flickr_500_choose.csv', 'r', encoding='utf-8') as infile:
        
        reader = csv.DictReader(infile)
        for i, row in enumerate(reader):
            print(i)
            if i < 0: continue
            if i >= 500:
                break
        
            query = row['choose']
            print(query)

            prompt_training = prompt_crepe_training()
            prompt_testing = prompt_crepe_testing(query)
            result = decompose_sentence(prompt_training,prompt_testing)

            result1, result2 = extract_dependency(result)

            print(result1)
            print(result2)

            with open('flickr_simple_decompose_dependency_240928.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(result1 + '\n')
            
            with open('flickr_simple_decompose_result_240928.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(result2 + '\n')

            time.sleep(4) # reduce the frequency of asking GPT

def run_dataset_adjust_test():
    i = 0
    with open('adjust.txt', 'r', encoding='utf-8') as infile:
        
        for line in infile:
            print(i)
            if i < 0: 
                i = i + 1
                continue
            if i >= 1:
                break
            query = line.strip()
            print(query)
            final_sub_query, child_list = decompose_framework(query, 1, "")

            tmp_sub_query = "(0){" + query + '-><'
            for num in child_list:
                tmp_sub_query = tmp_sub_query + str(num) + ','
            tmp_sub_query = tmp_sub_query + '> } '

            final_sub_query = tmp_sub_query + final_sub_query
            print(final_sub_query)

            with open('adjust_tmp_result.txt', 'a', encoding='utf-8') as outfile:
                outfile.write(final_sub_query + '\n')

            time.sleep(3) # reduce the frequency of asking GPT
            i = i + 1


# r1,r2 = run_simple_seg_split_test()
# print(r1)
# print(r2)

# run_dataset_simple_seg_split()

run_dataset_test()
