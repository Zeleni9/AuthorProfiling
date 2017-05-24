import os
import xml.etree.ElementTree as ET
from collections import defaultdict

class Preprocess:

    def __init__(self, folder):
        self.folder = folder + '/english_dataset'

    def load_data(self):
        users = defaultdict(list)
        files = os.listdir(self.folder)
        truth = files.pop(0)

        path =  os.getcwd() + '/english_dataset'
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
        path = os.path.join(self.folder, self.truth_file)
        with open(path) as f:
            data = f.readlines()
            for user in data:
                token = user.split(":::")
                tmp[token[0]] = [token[1], token[2], token[3], token[4], token[5], token[6], token[7]]
                # token[3] = extroverted
                # token[4] = stable
                # token[5] = agreeable
                # token[6] = conscientious
                # token[7] = open
                # Big five classification labels is in token 3-7
        self.truth_users = tmp

    def get_data(self):
        return self.users, self.truth_users
