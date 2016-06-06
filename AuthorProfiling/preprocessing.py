import os
import xml.etree.ElementTree as ET
from collections import defaultdict

class Preprocess:

    def __init__(self, folder):
        self.folder = folder

    def load_data(self):
        users = defaultdict(list)
        files = os.listdir(self.folder)
        truth = files.pop(0)

        path =  os.getcwd()
        for xml in files:
            text = []
            tree = ET.parse(path + '/' + xml)
            root = tree.getroot()
            user_id = root.attrib['id']
            for child in root:
                text.append(child.text.strip())
            users[user_id] = text

        self.users = users
        self.truth_file = truth

    # splitting truth data per user to tokens (age, gender, big five classes)
    def truth_data(self):
        tmp = defaultdict(list)
        with open(self.truth_file) as f:
            data = f.readlines()
            for user in data:
                token = user.split(":::")
                tmp[token[0]] = [token[1], token[2], token[3]]
                # token[3] = extroverted
                # Big five classification labels is in token 3-7
        self.truth_users = tmp

    def get_data(self):
        return self.users, self.truth_users

