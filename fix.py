from os import listdir
from os.path import isfile, join
import os

directory = './data'
new_directory = './data_fixed2'
os.makedirs(new_directory)

files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.csv')]

for file in files:
    with open(os.path.join(directory, file), 'r') as infile:
        with open(os.path.join(new_directory, file), 'w') as outfile:
            for line in infile.readlines():
                fixed = []
                for column in line.replace(',""', ',"').replace('"""', '"').replace('""', '"').replace('\r\n', '').split(','):
                    '''
                    if column == '"':
                        continue
                    if column.startswith('"') and not column.endswith('"'):
                        column = column + '"'
                    elif column.endswith('"') and not column.startswith('"'):
                        column = '"' + column
                    '''
                    fixed.append(column)
                outfile.write(','.join(fixed) + '\n')
