import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import re

class Preprocessing:

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
                text.append(child.text)
            users[user_id] = text

        self.users = users
        self.truth_file = truth

    def clean_data(self):

        for key, value in self.users.iteritems():
            clean_lines = []

            for line in value:
                # removing -> This is a tweet with a url: http://t.co/0DlGChTBIx
                result = re.sub(r"http\S+", "", line)
                # other cleaning
                clean_lines.append(result)


            self.users[key] = clean_lines

    def truth_data(self):
        tmp = defaultdict(list)
        with open(self.truth_file) as f:
            data = f.readlines()
            for user in data:
                token = user.split(":::")
                tmp[token[0]] = [token[1], token[2]]
        self.truth_users = tmp


    def get_data(self):
        return self.users, self.truth_users