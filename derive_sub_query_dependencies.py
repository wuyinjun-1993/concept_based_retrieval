import nest_asyncio
import asyncio
import aiohttp
import ssl
import certifi
import time
import pandas as pd
import pickle
import os
from tqdm import tqdm
nest_asyncio.apply()
import requests
from retrieval_utils import decompose_single_query_parition_groups
import utils


# Settings
MODEL = "gpt-3.5-turbo"
OPENAI_SECRET_KEY = os.environ.get("OPENAI_API_KEY") #"sk-proj-X...X"
MAX_RETRIES = 5  # Maximum number of retries
BATCH_SIZE = 25

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])  # Path compression
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            # Union by rank
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

# Asynchronous function to check dependency
async def check_dependency_async(session, sentence, segment1, segment2, retries=MAX_RETRIES):
    payload = {
        'model': MODEL,
        'messages': [
          {"role": "system", "content": "Example 1:\nOverall sentence: An institutional looking room holds an exit sign, chairs, recording equipment and a man with one arm aloft that appears to be demonstrating something with a device in the held-up hand for the benefit of a woman behind him that is watching closely.\nDecomposed sentence: \"An exit sign.\"|\"Chairs.\"|\"Recording equipment.\"|\"A man with one arm aloft appears to be demonstrating with a device in the held-up hand.\"|\"A woman behind a man is watching him closely.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3,4\nExample 2:\nOverall sentence: Black night, a blurry shot shows a tall fence that angles, in a kind of Z shape, around what appears to be a parking area with a large building fronting it, as a kid with a red helmet takes a fierce swing in the batting area, inside the fence.\nDecomposed sentence: \"In a black night, a blurry shot shows a tall fence that angles, in a kind of Z shape.\"|\"Around appears to be a parking area with a large building fronting it.\"|\"A kid with a red helmet takes a fierce swing in the batting area, inside the fence.\"\nDependencies of decomposed sentences: 0,1,2\nExample 3:\nOverall sentence: An older looking model computer components; two disk drive boxes stacked, a keyboard unit, and a more modern looking computer monitor with apple logo center bottom, and on screen at top centered in light blue font letters is printed APPLE II\nDecomposed sentence: \"An older looking model computer components.\"|\"Computer disk drive boxes stacked.\"|\"A keyboard unit.\"|\"A modern computer monitor with apple logo at the center bottom.\"|\"On the screen of a modern computer monitor, at the top center, light blue font letters print the words APPLE II.\"\nDependencies of decomposed sentences: 0,1,2 | 3,4\nExample 4:\nOverall sentence: A motel-like room features all of its furniture in rows, including a sofa, double bed, stuffed chair, desk chair, desk, bureau and television, all of which, like the one piece of artwork, carpet, linen and window drapes, are all in neutrals.\nDecomposed sentence: \"A motel room.\"|\"A neutral colored sofa.\"|\"A neutral colored double bed.\"|\"A neutral colored stuffed chair.\"|\"A desk chair.\"|\"A desk.\"|\"A bureau.\"|\"A television.\"|\"One piece of artwork.\"|\"Neutral colored linen.\"|\"Neutral colored window drapes.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3 | 4,5 | 6 | 7 | 8 | 9 | 10\nExample 5:\nOverall sentence: A room shows a twin bed, crib, double bed, and a round bed, all featuring checks and pillows, either white, or checked, high windows, a brick wall, a piece of art against a white wall, and a separate vertical surface with framed items on it.\nDecomposed sentence: \"A room.\"|\"A twin bed featuring checks and pillows, either white, or checked.\"|\"A crib.\"|\"A double bed featuring checks and pillows, either white, or checked.\"|\"A round bed featuring checks and pillows, either white, or checked.\"|\"High windows.\"|\"A brick wall.\"|\"A piece of art against a white wall.\"|\"A vertical surface with framed items on it.\"\nDependencies of decomposed sentences: 0 | 1| 2 | 3 | 4 | 5 | 6 | 7 | 8\nExample 6:\nOverall sentence: A street view shows a building entrance and traffic, including a bus, in the background, and to the foreground, a black iron fence, and a bench with an older, seated person, looking at a flock of pigeons scattered on the ground and bench.\nDecomposed sentence: \"A street view shows a building entrance.\"|\"A street view shows traffic, including a bus in the background.\"|\"In the foreground is a black iron fence.\"|\"In the foreground is a bench with an older person sitting on it\"|\"A flock of pigeons scattered on the ground and bench.\"\nDependencies of decomposed sentences: 0,1 | 2 | 3,4\nExample 7:\nOverall sentence: Off in the distance, a few people dot the stands and the fenced in area bordering a ball field, while an umpire and three uniformed baseball players, including a batter, catcher and outfielder, strike playing poses in the foreground.\nDecomposed form: \"A few people dot the stands bordering a ball field.\"|\"A few people dot the fenced-in area bordering a ball field.\"|\"A baseball umpire.\"|\"A baseball batter.\"|\"A baseball catcher.\"|\"An outfielder.\"\nDependencies of decomposed sentences: 0 | 1 | 2,3,4,5\nExample 8:\nOverall sentence: A counter top with a plate with a fork and few scraps of food and a teddy bear lying on side with arm outstretched on plate near fork, with another plate with an apple and two bowls with produce, a canister and some metal objects.\nDecomposed sentence: \"A counter top.\"|\"A plate with a fork and few scraps of food.\"|\"A teddy bear lying with arm outstretched on plate near fork.\"|\"A plate with an apple.\"|\"Bowls with apples.\"|\"A canister.\"|\"Metal objects.\"\nDependencies of decomposed sentences: 0 | 1,2 | 3 | 4 | 5 | 6\nExample 9:\nOverall sentence: Indirect lighting in a darkened room plays across a table covered with the remains of a meal, thereby giving everything, including the table, a basket, plates with uneaten food, utensils and glasses with liquid, a reddish patina.\nDecomposed sentence: \"Indirect lighting in a darkened room shines across a table covered with the remains of a meal.\"|\"The table has a reddish patina.\"|\"The basket has a reddish patina.\"|\"The plates with uneaten food have a reddish patina.\"|\"Utensils have a reddish patina.\"|\"Glasses with liquid have a reddish patina.\"\nDependencies of decomposed sentences: 0,1,2,3,4,5\nExample 10:\nOverall sentence: the long view of a parking lot shows a row of parked vehicles, including a bus, which appears to be the destination of a number of similarly dressed women with luggage that are moving towards it from the immediate foreground.\nDecomposed sentence: \"A parking lot shows a row of parked vehicles.\"|\"A bus.\"|\"Similarly dressed women with luggages.\"\nDependencies of decomposed sentences: 0,1 | 2\nExample 11:\nOverall sentence: Corner view of a table with plaid cloth, red and white checkerboard napkins, a pizza, a container of green leafy vegetable, and some bottles, with the hips of a person wearing an apron standing beside table in green grass.\nDecomposed sentence: \"Corner view of a table with plaid cloth.\"|\"Red and white checkerboard napkins.\"|\"A pizza.\"|\"A container of green leafy vegetables.\"|\"Some bottles.\"|\"The hips of a person wearing an apron standing beside a table in green grass.\"\nDependencies of decomposed sentences: 0,5| 1 | 2 | 3 | 4\nExample 12:\nOverall sentence: A room with natural, unfinished, pale wood walls and built in shelves, also has curtained sliding doors that open onto a patio with a lounge chair, a big double bed with a blue blanket, and various framed photos and books.\nDecomposed sentence: \"A room.\"|\"Natural, unfinished, pale wood walls.\"|\"Built-in shelves.\"|\"Curtained sliding doors.\"|\"Sliding doors that open onto a patio with a lounge chair.\"|\"A big double bed with a blue blanket.\"|\"Framed photos.\"|\"Books.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3,4 | 5 | 6 | 7\nExample 13:\nOverall sentence: A minimally furnished room shows small window pairs with blinds and no curtains, an unmade double bed with a heavy, taupe duvet, and a white border near the ceiling with colorful birds trailing in both directions.\nDecomposed sentence: \"A minimally furnished room.\"|\"Small window pairs with blinds and no curtains.\"|\"An unmade double bed with a heavy, taupe duvet.\"|\"A white border near the ceiling with colorful birds trailing in both directions.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3\nExample 14:\nOverall sentence: A blurry night-time street scene shows a higher and lower roadway with overpasses and lots of traffic going the same way, while to the left an area of red glare with an arrow and number, suggests a caution area.\nDecomposed sentence: \"A blurry night-time street scene shows a higher and lower roadway with overpasses and lots of traffic going the same way.\"|\"An area of red glare with an arrow and number, suggests a caution area.\"\nDependencies of decomposed sentences: 0,1\nExample 15:\nOverall sentence: Several people are standing, or sitting, in the vicinity of a scoreboard, to the front of which lies a tennis court, featuring a player with his knees bent, who is also grasping a racquet with both hands,\nDecomposed sentence: \"Several people are standing, or sitting, in the vicinity of a scoreboard.\"|\"In front of the scoreboard lies a tennis court.\"|\"A tennis player with his knees bent, who is grasping a racquet with both hands.\"\nDependencies of decomposed sentences: 0,1 | 2\nExample 16:\nOverall sentence: Black and white of a blacksmith interior with a horse, a man standing with arm out to it, and a man ducking under it's head and holding a long metal file in both hands, with horses on poles on the wall.\nDecomposed sentence: \"Black and white photo of a blacksmith interior with a horse.\"|\"Black and white photo of a man standing with arm out to a horse.\"|\"Black and white photo of a man ducking under a horses head and holding a long metal file in both hands.\"|\"Black and white photo with horse poles on the wall.\"\nDependencies of decomposed sentences: 0,1,2 | 3\nExample 17:\nOverall sentence: A blurry living room scene suggests a wide screen T,V. with a game playing, bookshelves, a low cluttered table, a couch, and closest and clearest, a sports t-shirt with the number thirty-five on it.\nDecomposed sentence: \"A blurry living room scene.\"|\"A wide screen TV with a sports game playing.\"|\"Bookshelves.\"|\"A low cluttered table.\"|\"A couch.\"|\"A sports t-shirt with the number 35 on it.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3 | 4 | 5\nExample 18:\nOverall sentence: In the first picture a boy is flipping his skateboard, in the second picture is skating and lifting the board slightly off the ground, and in the third he is jumping into the air onto the skateboard.\nDecomposed sentence: \"A boy is flipping his skateboard.\"|\"A boy is skating and lifting the skateboard slightly off the ground.\"|\"A boy is jumping into the air onto the skateboard.\"\nDependencies of decomposed sentences: 0 | 1 | 2\nExample 19:\nOverall sentence: A table with a person holding a sandwich in hand over a plate with a knife on it, and another plate with a knife, napkin, and french bread sandwich, along with a bottle and a glass with beverages.\nDecomposed sentence: \"A table with a person holding a sandwich in hand.\"|\"A sandwich in hand over a plate with a knife on it.\"|\"A plate with a knife, napkin, and french bread sandwich.\"|\"A bottle.\"|\"A glass with beverages.\"\nDependencies of decomposed sentences: 0,1 | 2,3,4\nExample 20:\nOverall sentence: An impressive array of motorcycles recedes into the distance between a driveway and a building fronted with cars and a grassy area, as lots of people mill about the driveway, around the bikes.\nDecomposed sentence: \"An array of motorcycles between a driveway and a building.\"|\"A building fronted with cars and a grassy area.\"|\"Lots of people mill about the driveway, around the bikes.\"\nDependencies of decomposed sentences: 0,1,2\nLearning from how the provided 20 examples of overall sentences have their decomposed phrases, each represented by their respective index, starting from 0, are decomposed, then the decomposed phrases are grouped together by commas to indicate phrases are dependent, while bars separate independent phrases or phrase groups, output whether two decomposed phrases are dependent given the overall sentence. You may answer in only one digit, 1 meaning yes, 0 meaning no."},
          {"role": "user", "content": f"Overall sentence:{sentence}\n And two decomposed phrases: {segment1}|{segment2}\nPlease output 1 or 0, 1 meaning they are dependent phrases, 0 meaning they are independent phraes."}
        ],
        'max_tokens': 1,
        'n': 1,
        'temperature': 0
    }
    for attempt in range(retries):
        try:
            async with session.post(
                url='https://api.openai.com/v1/chat/completions',
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_SECRET_KEY}"},
                json=payload,
                ssl=ssl.create_default_context(cafile=certifi.where())
            ) as response:
                response_json = await response.json()
            if "error" in response_json:
                print(f"OpenAI request failed with error {response_json['error']}")
                if response_json['error']['code'] == 'rate_limit_exceeded':
                    retry_after = 10
                    await asyncio.sleep(retry_after)  # Wait for the retry-after duration before retrying
                continue
            result = response_json['choices'][0]['message']['content']
            if result in {'1', '0'}:
                print(result)
                return int(result)
            else:
                print("errors")
                return 0
        except Exception as e:
            print(f"Request failed: {e}")
    print("errors")
    return 0

# Asynchronous function to check dependency
def check_dependency_seq(sentence, segment1, segment2, retries=MAX_RETRIES):
    payload = {
        'model': MODEL,
        'messages': [
          {"role": "system", "content": "Example 1:\nOverall sentence: An institutional looking room holds an exit sign, chairs, recording equipment and a man with one arm aloft that appears to be demonstrating something with a device in the held-up hand for the benefit of a woman behind him that is watching closely.\nDecomposed sentence: \"An exit sign.\"|\"Chairs.\"|\"Recording equipment.\"|\"A man with one arm aloft appears to be demonstrating with a device in the held-up hand.\"|\"A woman behind a man is watching him closely.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3,4\nExample 2:\nOverall sentence: Black night, a blurry shot shows a tall fence that angles, in a kind of Z shape, around what appears to be a parking area with a large building fronting it, as a kid with a red helmet takes a fierce swing in the batting area, inside the fence.\nDecomposed sentence: \"In a black night, a blurry shot shows a tall fence that angles, in a kind of Z shape.\"|\"Around appears to be a parking area with a large building fronting it.\"|\"A kid with a red helmet takes a fierce swing in the batting area, inside the fence.\"\nDependencies of decomposed sentences: 0,1,2\nExample 3:\nOverall sentence: An older looking model computer components; two disk drive boxes stacked, a keyboard unit, and a more modern looking computer monitor with apple logo center bottom, and on screen at top centered in light blue font letters is printed APPLE II\nDecomposed sentence: \"An older looking model computer components.\"|\"Computer disk drive boxes stacked.\"|\"A keyboard unit.\"|\"A modern computer monitor with apple logo at the center bottom.\"|\"On the screen of a modern computer monitor, at the top center, light blue font letters print the words APPLE II.\"\nDependencies of decomposed sentences: 0,1,2 | 3,4\nExample 4:\nOverall sentence: A motel-like room features all of its furniture in rows, including a sofa, double bed, stuffed chair, desk chair, desk, bureau and television, all of which, like the one piece of artwork, carpet, linen and window drapes, are all in neutrals.\nDecomposed sentence: \"A motel room.\"|\"A neutral colored sofa.\"|\"A neutral colored double bed.\"|\"A neutral colored stuffed chair.\"|\"A desk chair.\"|\"A desk.\"|\"A bureau.\"|\"A television.\"|\"One piece of artwork.\"|\"Neutral colored linen.\"|\"Neutral colored window drapes.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3 | 4,5 | 6 | 7 | 8 | 9 | 10\nExample 5:\nOverall sentence: A room shows a twin bed, crib, double bed, and a round bed, all featuring checks and pillows, either white, or checked, high windows, a brick wall, a piece of art against a white wall, and a separate vertical surface with framed items on it.\nDecomposed sentence: \"A room.\"|\"A twin bed featuring checks and pillows, either white, or checked.\"|\"A crib.\"|\"A double bed featuring checks and pillows, either white, or checked.\"|\"A round bed featuring checks and pillows, either white, or checked.\"|\"High windows.\"|\"A brick wall.\"|\"A piece of art against a white wall.\"|\"A vertical surface with framed items on it.\"\nDependencies of decomposed sentences: 0 | 1| 2 | 3 | 4 | 5 | 6 | 7 | 8\nExample 6:\nOverall sentence: A street view shows a building entrance and traffic, including a bus, in the background, and to the foreground, a black iron fence, and a bench with an older, seated person, looking at a flock of pigeons scattered on the ground and bench.\nDecomposed sentence: \"A street view shows a building entrance.\"|\"A street view shows traffic, including a bus in the background.\"|\"In the foreground is a black iron fence.\"|\"In the foreground is a bench with an older person sitting on it\"|\"A flock of pigeons scattered on the ground and bench.\"\nDependencies of decomposed sentences: 0,1 | 2 | 3,4\nExample 7:\nOverall sentence: Off in the distance, a few people dot the stands and the fenced in area bordering a ball field, while an umpire and three uniformed baseball players, including a batter, catcher and outfielder, strike playing poses in the foreground.\nDecomposed form: \"A few people dot the stands bordering a ball field.\"|\"A few people dot the fenced-in area bordering a ball field.\"|\"A baseball umpire.\"|\"A baseball batter.\"|\"A baseball catcher.\"|\"An outfielder.\"\nDependencies of decomposed sentences: 0 | 1 | 2,3,4,5\nExample 8:\nOverall sentence: A counter top with a plate with a fork and few scraps of food and a teddy bear lying on side with arm outstretched on plate near fork, with another plate with an apple and two bowls with produce, a canister and some metal objects.\nDecomposed sentence: \"A counter top.\"|\"A plate with a fork and few scraps of food.\"|\"A teddy bear lying with arm outstretched on plate near fork.\"|\"A plate with an apple.\"|\"Bowls with apples.\"|\"A canister.\"|\"Metal objects.\"\nDependencies of decomposed sentences: 0 | 1,2 | 3 | 4 | 5 | 6\nExample 9:\nOverall sentence: Indirect lighting in a darkened room plays across a table covered with the remains of a meal, thereby giving everything, including the table, a basket, plates with uneaten food, utensils and glasses with liquid, a reddish patina.\nDecomposed sentence: \"Indirect lighting in a darkened room shines across a table covered with the remains of a meal.\"|\"The table has a reddish patina.\"|\"The basket has a reddish patina.\"|\"The plates with uneaten food have a reddish patina.\"|\"Utensils have a reddish patina.\"|\"Glasses with liquid have a reddish patina.\"\nDependencies of decomposed sentences: 0,1,2,3,4,5\nExample 10:\nOverall sentence: the long view of a parking lot shows a row of parked vehicles, including a bus, which appears to be the destination of a number of similarly dressed women with luggage that are moving towards it from the immediate foreground.\nDecomposed sentence: \"A parking lot shows a row of parked vehicles.\"|\"A bus.\"|\"Similarly dressed women with luggages.\"\nDependencies of decomposed sentences: 0,1 | 2\nExample 11:\nOverall sentence: Corner view of a table with plaid cloth, red and white checkerboard napkins, a pizza, a container of green leafy vegetable, and some bottles, with the hips of a person wearing an apron standing beside table in green grass.\nDecomposed sentence: \"Corner view of a table with plaid cloth.\"|\"Red and white checkerboard napkins.\"|\"A pizza.\"|\"A container of green leafy vegetables.\"|\"Some bottles.\"|\"The hips of a person wearing an apron standing beside a table in green grass.\"\nDependencies of decomposed sentences: 0,5| 1 | 2 | 3 | 4\nExample 12:\nOverall sentence: A room with natural, unfinished, pale wood walls and built in shelves, also has curtained sliding doors that open onto a patio with a lounge chair, a big double bed with a blue blanket, and various framed photos and books.\nDecomposed sentence: \"A room.\"|\"Natural, unfinished, pale wood walls.\"|\"Built-in shelves.\"|\"Curtained sliding doors.\"|\"Sliding doors that open onto a patio with a lounge chair.\"|\"A big double bed with a blue blanket.\"|\"Framed photos.\"|\"Books.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3,4 | 5 | 6 | 7\nExample 13:\nOverall sentence: A minimally furnished room shows small window pairs with blinds and no curtains, an unmade double bed with a heavy, taupe duvet, and a white border near the ceiling with colorful birds trailing in both directions.\nDecomposed sentence: \"A minimally furnished room.\"|\"Small window pairs with blinds and no curtains.\"|\"An unmade double bed with a heavy, taupe duvet.\"|\"A white border near the ceiling with colorful birds trailing in both directions.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3\nExample 14:\nOverall sentence: A blurry night-time street scene shows a higher and lower roadway with overpasses and lots of traffic going the same way, while to the left an area of red glare with an arrow and number, suggests a caution area.\nDecomposed sentence: \"A blurry night-time street scene shows a higher and lower roadway with overpasses and lots of traffic going the same way.\"|\"An area of red glare with an arrow and number, suggests a caution area.\"\nDependencies of decomposed sentences: 0,1\nExample 15:\nOverall sentence: Several people are standing, or sitting, in the vicinity of a scoreboard, to the front of which lies a tennis court, featuring a player with his knees bent, who is also grasping a racquet with both hands,\nDecomposed sentence: \"Several people are standing, or sitting, in the vicinity of a scoreboard.\"|\"In front of the scoreboard lies a tennis court.\"|\"A tennis player with his knees bent, who is grasping a racquet with both hands.\"\nDependencies of decomposed sentences: 0,1 | 2\nExample 16:\nOverall sentence: Black and white of a blacksmith interior with a horse, a man standing with arm out to it, and a man ducking under it's head and holding a long metal file in both hands, with horses on poles on the wall.\nDecomposed sentence: \"Black and white photo of a blacksmith interior with a horse.\"|\"Black and white photo of a man standing with arm out to a horse.\"|\"Black and white photo of a man ducking under a horses head and holding a long metal file in both hands.\"|\"Black and white photo with horse poles on the wall.\"\nDependencies of decomposed sentences: 0,1,2 | 3\nExample 17:\nOverall sentence: A blurry living room scene suggests a wide screen T,V. with a game playing, bookshelves, a low cluttered table, a couch, and closest and clearest, a sports t-shirt with the number thirty-five on it.\nDecomposed sentence: \"A blurry living room scene.\"|\"A wide screen TV with a sports game playing.\"|\"Bookshelves.\"|\"A low cluttered table.\"|\"A couch.\"|\"A sports t-shirt with the number 35 on it.\"\nDependencies of decomposed sentences: 0 | 1 | 2 | 3 | 4 | 5\nExample 18:\nOverall sentence: In the first picture a boy is flipping his skateboard, in the second picture is skating and lifting the board slightly off the ground, and in the third he is jumping into the air onto the skateboard.\nDecomposed sentence: \"A boy is flipping his skateboard.\"|\"A boy is skating and lifting the skateboard slightly off the ground.\"|\"A boy is jumping into the air onto the skateboard.\"\nDependencies of decomposed sentences: 0 | 1 | 2\nExample 19:\nOverall sentence: A table with a person holding a sandwich in hand over a plate with a knife on it, and another plate with a knife, napkin, and french bread sandwich, along with a bottle and a glass with beverages.\nDecomposed sentence: \"A table with a person holding a sandwich in hand.\"|\"A sandwich in hand over a plate with a knife on it.\"|\"A plate with a knife, napkin, and french bread sandwich.\"|\"A bottle.\"|\"A glass with beverages.\"\nDependencies of decomposed sentences: 0,1 | 2,3,4\nExample 20:\nOverall sentence: An impressive array of motorcycles recedes into the distance between a driveway and a building fronted with cars and a grassy area, as lots of people mill about the driveway, around the bikes.\nDecomposed sentence: \"An array of motorcycles between a driveway and a building.\"|\"A building fronted with cars and a grassy area.\"|\"Lots of people mill about the driveway, around the bikes.\"\nDependencies of decomposed sentences: 0,1,2\nLearning from how the provided 20 examples of overall sentences have their decomposed phrases, each represented by their respective index, starting from 0, are decomposed, then the decomposed phrases are grouped together by commas to indicate phrases are dependent, while bars separate independent phrases or phrase groups, output whether two decomposed phrases are dependent given the overall sentence. You may answer in only one digit, 1 meaning yes, 0 meaning no."},
          {"role": "user", "content": f"Overall sentence:{sentence}\n And two decomposed phrases: {segment1}|{segment2}\nPlease output 1 or 0, 1 meaning they are dependent phrases, 0 meaning they are independent phraes."}
        ],
        'max_tokens': 1,
        'n': 1,
        'temperature': 0
    }
    for attempt in range(retries):
        try:
            with requests.post(
                url='https://api.openai.com/v1/chat/completions',
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_SECRET_KEY}"},
                json=payload,
                # ssl=ssl.create_default_context(cafile=certifi.where())
            ) as response:
                response_json = response.json()
            # if "error" in response_json:
            #     print(f"OpenAI request failed with error {response_json['error']}")
            #     if response_json['error']['code'] == 'rate_limit_exceeded':
            #         retry_after = 10
            #         # await asyncio.sleep(retry_after)  # Wait for the retry-after duration before retrying
            #     continue
            result = response_json['choices'][0]['message']['content']
            if result in {'1', '0'}:
                # print(result)
                return int(result)
            else:
                print("errors")
                return 0
        except Exception as e:
            print(f"Request failed: {e}")
    print("errors")
    return 0

# Asynchronous function to group dependent segments for a single sentence
async def group_dependent_segments_async(session, sentence, segments):
    if pd.isnull(segments):
        return "", 0
    start_time = time.time()  # Start time
    segments_list = segments.split('|')
    n = len(segments_list)

    # Initialize Union-Find
    uf = UnionFind(n)

    # Perform pairwise comparisons sequentially
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) != uf.find(j):  # Check if i and j are already in the same group
                result = await check_dependency_async(session, sentence, segments_list[i], segments_list[j])
                if result == 1:
                    uf.union(i, j)
                await asyncio.sleep(2)  # Small delay to avoid hitting limits too quickly
            # else:
            #   print('already in same group')

    # Group segments based on union-find results
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Convert indices back to segments
    grouped_indices = [",".join(map(str, sorted(group))) for group in groups.values()]
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    return " | ".join(grouped_indices), elapsed_time


def group_dependent_segments_seq(sentence, segments):
    if pd.isnull(segments):
        return "", 0
    start_time = time.time()  # Start time
    segments_list = segments.split('|')
    n = len(segments_list)

    # Initialize Union-Find
    uf = UnionFind(n)

    # Perform pairwise comparisons sequentially
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) != uf.find(j):  # Check if i and j are already in the same group
                result = check_dependency_seq(sentence, segments_list[i], segments_list[j])
                if result == 1:
                    uf.union(i, j)
                time.sleep(2)
                # await asyncio.sleep(2)  # Small delay to avoid hitting limits too quickly
            # else:
            #   print('already in same group')

    # Group segments based on union-find results
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Convert indices back to segments
    grouped_indices = [",".join(map(str, sorted(group))) for group in groups.values()]
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    return " | ".join(grouped_indices), elapsed_time

def group_dependent_segments_seq_all(sentence_mappings, segments_mappings, data_path, query_hash=None):
    sentence_ls = [sentence_mappings[str(idx + 1)] for idx in range(len(sentence_mappings))]
    segments_str_ls = ["|".join(segments_mappings[str(idx + 1)][0]) for idx in range(len(segments_mappings))]
    segments_ls = [segments_mappings[str(idx + 1)] for idx in range(len(segments_mappings))]
    
    if query_hash is None:
        grouped_sub_q_ids_file = os.path.join(data_path, "grouped_sub_q_ids_ls.pkl")
    else:
        grouped_sub_q_ids_file = os.path.join(data_path, f"grouped_sub_q_ids_ls_{query_hash}.pkl")
    if os.path.exists(grouped_sub_q_ids_file):
        grouped_sub_q_ids_ls= utils.load(grouped_sub_q_ids_file)
    
    else:
        grouped_sub_q_ids_ls = []
        for (sentence, segments, segments_str) in tqdm(zip(sentence_ls, segments_ls, segments_str_ls)):
            curr_sub_q_ids_str, _ = group_dependent_segments_seq(sentence, segments_str)
            curr_sub_q_ids_ls = decompose_single_query_parition_groups(segments, curr_sub_q_ids_str)
            # curr_sub_q_ids_ls = decompose_single_query_ls(curr_sub_q_ids_str)
            grouped_sub_q_ids_ls.append(curr_sub_q_ids_ls)
            time.sleep(10)
        
        utils.save(grouped_sub_q_ids_ls, grouped_sub_q_ids_file)
        
    return grouped_sub_q_ids_ls
        
        

# Asynchronous function to process multiple sentences in parallel
async def process_sentences_in_bulk(sentences_and_segments):
    async with aiohttp.ClientSession() as session:
        results = []
        for i in tqdm(range(0, len(sentences_and_segments), BATCH_SIZE)):
            batch = sentences_and_segments[i:i+BATCH_SIZE]
            tasks = [group_dependent_segments_async(session, sentence, segments) for sentence, segments in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            if i + BATCH_SIZE < len(sentences_and_segments):
                await asyncio.sleep(10)  # Wait for 60 seconds between batches to respect rate limits
    return results

def process_sentences_in_bulk_seq(sentences_and_segments):
    # with requests.ClientSession() as session:
    results = []
    for i in tqdm(range(0, len(sentences_and_segments))):
        batch = sentences_and_segments[i:i+1]
        batch_results = [group_dependent_segments_seq(sentence, segments) for sentence, segments in batch]
        # batch_results = asyncio.gather(*tasks)
        results.extend(batch_results)
            # if i + BATCH_SIZE < len(sentences_and_segments):
            #     await asyncio.sleep(10)  # Wait for 60 seconds between batches to respect rate limits
    return results

def apply_to_dataframe(df, sentence_col, segments_col):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_copy = df.copy()
    sentences_and_segments = list(zip(df_copy[sentence_col], df_copy[segments_col]))
    # results = asyncio.run(process_sentences_in_bulk_seq(sentences_and_segments))
    results = process_sentences_in_bulk_seq(sentences_and_segments)
    df_copy['groups'] = [result[0] for result in results]
    df_copy['elapsed_time'] = [result[1] for result in results]
    return df_copy

def pre_process_caption_triples_func(x):
    if pd.isnull(x):
        return x
    else:
        return '"' + x.split('"', 1)[-1].rsplit('"', 1)[0] + '"'

if __name__ == "__main__":
    begin = time.time()
    query_path = "/home/wuyinjun/concept_based_retrieval/"
    img_caption_file_name_output= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all_output3.csv")
    df = pd.read_csv(img_caption_file_name_output)
    # df = intermediate_df.copy() #can change according to user
    # Filter rows that do not start and end with "
    # filtered_df = df[~(df['caption_triples_ls'].str.startswith('"') & df['caption_triples_ls'].str.endswith('"'))].copy()
    # Apply the lambda function to those rows only
    # filtered_df['caption_triples_ls'] = filtered_df['caption_triples_ls'].apply(lambda x: pre_process_caption_triples_func(x)) # '"' + x.split('"', 1)[-1].rsplit('"', 1)[0] + '"')
    # Combine the processed rows back with the original DataFrame
    result_df = pd.concat([df[(df['caption_triples_ls'].str.startswith('"') & df['caption_triples_ls'].str.endswith('"'))], df]).sort_index()
    # Apply function to DataFrame
    final_df = apply_to_dataframe(result_df, 'caption', 'caption_triples_ls')
    end = time.time()
    total_time = end - begin
    print(f"Operation took a total of {total_time} seconds.")
    img_caption_file_name_output= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all_output_with_dependency3.csv")
    final_df.to_csv(img_caption_file_name_output, index=False)
    '''
    pickle_file_path = '/content/drive/MyDrive/mscoco_500decomposed_dependencies.pkl'
    with open(pickle_file_path, 'wb') as f:
      pickle.dump(final_df, f)
    '''