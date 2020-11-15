import json

# add the contents of the output file to the Dutch phonetic dictionary, .dic file of CMU

# also had manual additions to the dic

with open('../data/mapping_symbols.txt', 'r') as file:
    mapping_lines = file.readlines()

with open('../data/ipa_oov_separated.txt', 'r') as file:
    oov_words = file.readlines()

stress_symbols= ['ˈ', 'ˌ']

ipa2cmu = dict()

for l in mapping_lines:

    ipa, cmu = l.split()

    if '+' in cmu:
        #ipa maps to a combination of cmu symbols, diphthongs
        cmu = cmu.split('+')

    ipa2cmu[ipa] = cmu


oov2cmu = dict()


for w in oov_words:

    actual_w, ipa_w = w.split()

    cmu_phonetic = ''

    phonemes = ipa_w.split('_')


    for p in range(len(phonemes)):

        pp = phonemes[p]

        '''pick-up p_ˈɪ_k_(en)_ˈʌ_p_(nl)_
        pick-ups p_ˈɪ_k_(en)_ˈʌ_p_s_(nl)_
        make-up m_ˈaː_k_ə_(en)_ˈʌ_p_(nl)_
        warming-up ʋ_ˈɑ_r_m_ɪ_ŋ_(en)_ˈʌ_p_(nl)_
        time-lapse (en)_t_ˈaɪ_m_(nl)_l_ˈɑ_p_s_ə_
        close-up k_l_ˈoː_s_ə_(en)_ˈʌ_p_(nl)_'''

        if len(pp) > 0 and pp != '(en)' and pp != '(nl)': #skip \n and language tags

            for st in stress_symbols:
                if st in pp:
                    pp_split = pp.split(st)

                    for i in pp_split:
                        if i != st:
                            pp = i


            cmu_p = ipa2cmu[pp]

            if type(cmu_p) != list:
                cmu_phonetic += cmu_p

            else:

                for c in range(len(cmu_p)):

                    cmu_phonetic += cmu_p[c]

                    if c < len(cmu_p) - 1:
                        cmu_phonetic += ' '

            if p < len(phonemes)-1:
                cmu_phonetic += ' '

    oov2cmu[actual_w] = cmu_phonetic

with open('../data/oov2cmu.json', 'w') as file:
    json.dump(oov2cmu, file)

with open('../data/oov2cmu.txt', 'w') as file:

    count = 0

    for w in oov2cmu:

        entry_str = w + ' ' + oov2cmu[w]
        file.write(entry_str)

        if count < len(oov2cmu) - 1:
            file.write('\n')

