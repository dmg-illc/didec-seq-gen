import subprocess
import json

#gets a list of CMU phonemes, eSpeak phonemes and a dict of out-of-vocabulary word to eSpeak phonetic transcription

#list of out-of-vocab words
with open('../oov_word_list.txt', 'r') as file:
    oov_words = file.readlines()

#list of phonemes in CMU Sphinx dict
with open('../voxforge_nl_sphinx.phone', 'r') as file:
    cmu_phoneme_lines = file.readlines()

oov2phone = dict()

cmu_phonemes = []

for p in cmu_phoneme_lines:
    cmu_phonemes.append(p.split()[0]) #remove \n

print('CMU', len(cmu_phonemes), cmu_phonemes)

IPA_symbols = []

count = 0

for w in oov_words:

    count += 1
    if count % 20 == 0:
        pass
        #print(count)

    #run espeak with dutch settings to obtain the IPA transcriptions

    split_word = w.split()[0]

    command = "espeak -q -v nl --ipa=3 " + split_word #remove \n

    list_command = command.split()

    #subprocess.call(list_command)

    printout = subprocess.Popen(list_command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

    stdout, stderr = printout.communicate()

    phonetic_word = stdout.decode('utf8').split('\n')[0]

    #print(phonetic_word)

    oov2phone[split_word] = phonetic_word #NOT UTF8

    #print(split_word, stdout.decode('utf8').split('\n')[0])

    ipa_tr = ''

    for s in phonetic_word.split('_'):
        if s != '' and s != ' ':
            if ' ' in s:
                sym = s.split()

                for c in sym:
                    if c !='' and c!= ' ':
                        #print(c)

                        IPA_symbols.append(c)
                        ipa_tr += c + '_'
            else:
                #print(s)
                IPA_symbols.append(s)

                ipa_tr += s  + '_'

    print(split_word, ipa_tr)


IPA_symbols = list(set(IPA_symbols))
#print('ESP', len(IPA_symbols), IPA_symbols)

for i in IPA_symbols:
    print(i)

with open('oov2phonemes.json', 'w') as file:
    json.dump(oov2phone, file)

with open('cmu_phonemes.json', 'w') as file:
    json.dump(cmu_phonemes, file)

with open('espeak_phonemes.json', 'w') as file:
    json.dump(IPA_symbols, file)