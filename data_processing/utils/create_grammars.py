import os
import json

captions_path = '../data/captions/'
grammars_path = '../data/grammars/'
if not os.path.isdir(grammars_path):
    os.mkdir(grammars_path)

#template for an example sentence
#we use this as the only rule in the grammar, forcing the tool to align only given these words

'''#JSGF V1.0;
grammar forcing;
public <s> = gezelschap die aan het eten is of die in een restaurant zit en iets willen gaan bestellen;
'''

count = 0

for root, dirs, files in os.walk(captions_path):

    for f in files:
        #print(f)

        file_path = os.path.join(root, f)
        with open(file_path, 'r') as c_file:
            caption_lines = c_file.readlines()

        caption_string = ''

        for c in range(len(caption_lines)):

            line = caption_lines[c].strip()
            #print(line, len(line))

            if '<' in line:
                #special tags from annotations
                pass
            else:
                caption_string += line

                if c < len(caption_lines)-1:
                    caption_string += ' '


            for s in line:
                if s.isdigit():
                    print(f, line)
                    print(caption_string)

            #also check multi-digit numbers and punc. marks

        #CHECK NUMERIC NUMBERS AND PUNCTUATION MARKS in CMUSPHINX

        #CMUSPHINX can't process these because:
        #Numeric symbols are not in the dictionary
        #Punctuation marks are also not in the dictionary

        #CHECK OUT OF VOCAB WORDS

        #Uh words

        grammar_file = f.split('.')[0] + '.jsgf'
        grammar_path= os.path.join(grammars_path, grammar_file)

        #print(caption_string)

        with open(grammar_path, 'w') as g_file:
            g_file.write('#JSGF V1.0;\n')
            g_file.write('grammar forcing;\n')

            caption_rule = 'public <s> = ' + caption_string
            g_file.write(caption_rule)
            g_file.write(';')

            count += 1


print(count)