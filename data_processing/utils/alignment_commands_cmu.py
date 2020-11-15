import json
import os

#we don't need to read the output from the shell, because the alignment is already written into text

#creates the shell script for the commands to be fed into the cmu alignment tool
#for all the captions and the corresponding audio files

with open('../data/dict_caption_audio_wav.json', 'r') as file:
    dict_caption_audio = json.load(file) #caption path -> wav audio path

main_folder = 'data_processing'  # put the full path here

alignment_relative = 'data/alignments'

alignment_path = os.path.join(main_folder, alignment_relative)

if not os.path.isdir(alignment_path):
    os.mkdir(alignment_path)

grammars_relative = 'data/grammars/'

min_head_duration = 0 #in seconds

count = 0

#run this .sh in the cmusphinx folder
with open('align_commands.sh', 'w') as file:

    file.write('export LD_LIBRARY_PATH=/usr/local/lib\n')
    file.write('export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig\n')

    converted_audio_file = ''
    grammar_file = ''

    alignment_output_file = ''

    #example command below

    '''  pocketsphinx_continuous -dict nl-nl/voxforge_nl_sphinx.dic  -hmm nl-nl/nl-nl/ 
     -infile Scan_Path_ppn46_1593035.wav -jsgf with-word.jsgf -time yes -backtrace yes 
     -fsgusefiller yes -bestpath yes 2>&1 > try_out.txt -samprate 16000
      -remove_noise yes -remove_silence no -fillprob 0.1 ?
    '''

    for caption_path in dict_caption_audio:

        audio_path = dict_caption_audio[caption_path]
        print(caption_path, audio_path)

        audio_subsplit = audio_path.split('/')[3].split('_')
        ppn = audio_subsplit[2][3:]
        img_id = audio_subsplit[3].split('.')[0]

        rel_audio_path = audio_path[3:] #to remove ../
        abs_audio_path = os.path.join(main_folder, rel_audio_path)

        grammar_file_name = 'raw_caption_' + ppn + '_' + img_id + '.jsgf'
        grammar_file = os.path.join(main_folder, grammars_relative, grammar_file_name)

        mapped_file_name = "cmu_mapped_" + ppn + "_" + img_id + ".txt"
        mapped_path = os.path.join(alignment_path, mapped_file_name)

        #to be run in the cmusphinx folder, all paths absolute

        # fillprob parameter?
        cmusphinx_command = 'pocketsphinx_continuous -dict nl-nl/voxforge_nl_sphinx.dic  -hmm nl-nl/nl-nl/ ' +\
                            '-infile ' + abs_audio_path + ' -jsgf ' + grammar_file +\
                            ' -time yes -backtrace yes -fsgusefiller yes -bestpath yes 2>&1 > ' +\
                            mapped_path + ' -samprate 16000 -remove_noise no -remove_silence no ' +\
                            '-lw 1.0 -beam 1e-80 -wbeam 1e-60 -pbeam 1e-80\n'

        #to get good recognition results added the last line
        '''Nickolay V. Shmyrev - 2014-09-06
        To get good recognition results with your data you can add the following arguments:

   -lw 1.0 -beam 1e-80 -wbeam 1e-60 -pbeam 1e-80'''
        '''it says you that decoding result does not match the grammar so you should update your grammar 
        to make it more flexible. To use older models in trunk you need to add '-remove_noise no'...
        per https://sourceforge.net/p/cmusphinx/discussion/help/thread/0dcf09b5/#75db/9e96'''

        #print(cmusphinx_command)

        file.write(cmusphinx_command)
        count += 1

    print(count)
