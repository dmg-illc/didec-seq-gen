# from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py
#
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils.DidecDatasetCOCONL import DidecDatasetCOCONL

from models.model_image import DecoderWithAttention

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import NLGEval

import json


def evaluate(beam_size, decoder, loader, device, word_map, vocab_size, rev_word_map, nlgeval, split):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    smoothing_method = SmoothingFunction().method1  # epsilon method

    references = list()
    hypotheses = list()

    # corpus bleu from nltk
    refs_nltk = list()
    hyps_nltk = list()

    empty_count = 0

    with torch.no_grad():
        with open('image_COCONL_orig_cider.txt', 'w') as hyp_file:

            # For each image
            for i, (image_features, caps, caplens, allcaps) in enumerate(tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

                k = beam_size

                # Move to GPU device, if available
                image_features = image_features.to(device)  # (1, 3, 256, 256)

                image_features_mean = image_features.mean(1)
                image_features_mean = image_features_mean.expand(k, 2048)

                # Tensor to store top k previous words at each step; now they're just <start>
                k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

                # Tensor to store top k sequences; now they're just <start>
                seqs = k_prev_words  # (k, 1)

                # Tensor to store top k sequences' scores; now they're just 0
                top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

                # Lists to store completed sequences and scores
                complete_seqs = list()
                complete_seqs_scores = list()

                # Start decoding
                step = 1

                h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
                h2, c2 = decoder.init_hidden_state(k)

                # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
                while True:

                    embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                    h1, c1 = decoder.top_down_attention(
                        torch.cat([h2, image_features_mean, embeddings], dim=1),
                        (h1, c1))  # (batch_size_t, decoder_dim)
                    attention_weighted_encoding = decoder.attention(image_features, h1)
                    h2, c2 = decoder.language_model(
                        torch.cat([attention_weighted_encoding, h1], dim=1), (h2, c2))

                    scores = decoder.fc(h2)  # (s, vocab_size)
                    scores = F.log_softmax(scores, dim=1)

                    # Add
                    scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                    # For the first step, all k points will have the same scores (since same k previous words, h, c)
                    if step == 1:
                        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                    else:
                        # Unroll and find top scores, and their unrolled indices
                        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                    # Convert unrolled indices to actual indices of scores
                    prev_word_inds = top_k_words / vocab_size  # (s)
                    next_word_inds = top_k_words % vocab_size  # (s)

                    # Add new words to sequences
                    seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                    # Which sequences are incomplete (didn't reach <end>)?
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != word_map['<end>']]
                    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                    top_k_scores_temp = top_k_scores

                    # Set aside complete sequences
                    if len(complete_inds) > 0:
                        complete_seqs.extend(seqs[complete_inds].tolist())
                        complete_seqs_scores.extend(top_k_scores[complete_inds])
                    k -= len(complete_inds)  # reduce beam length accordingly

                    # Proceed with incomplete sequences
                    if k == 0:
                        break
                    seqs = seqs[incomplete_inds]

                    h1 = h1[prev_word_inds[incomplete_inds]]
                    c1 = c1[prev_word_inds[incomplete_inds]]
                    h2 = h2[prev_word_inds[incomplete_inds]]
                    c2 = c2[prev_word_inds[incomplete_inds]]

                    image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]

                    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                    k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                    # Break if things have been going on too long
                    if step > 45:  # max val
                        break
                    step += 1

                if len(complete_seqs_scores) > 0:

                    max_score = max(complete_seqs_scores)

                    i = complete_seqs_scores.index(max_score)
                    seq = complete_seqs[i]

                    '''# References
                    img_caps = allcaps[0].tolist()
                    img_captions = list(
                        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                            img_caps))  # remove <start> and pads
                    references.append(img_captions)

                    # Hypotheses
                    hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            '''
                    # References
                    img_caps = allcaps[0].tolist()

                    img_captions = list(
                        map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                            img_caps))  # remove <start> and pads

                    img_captions = [c for c in img_captions if len(c) > 0]
                    refs_nltk.append(img_captions)

                    img_caps = [' '.join(c) for c in img_captions]
                    # print(img_caps)
                    references.append(img_caps)

                    # Hypotheses
                    hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<pad>']}])
                    hyps_nltk.append(hypothesis)

                    hypothesis = ' '.join(hypothesis)
                    hypotheses.append(hypothesis)

                    assert len(references) == len(hypotheses)
                    assert len(refs_nltk) == len(hyps_nltk)

                else:
                    empty_count += 1
                    print('emptyseq', empty_count)

                    # all incomplete here

                    complete_seqs.extend((seqs[incomplete_inds].tolist()))
                    complete_seqs_scores.extend(top_k_scores[incomplete_inds])

                    max_score = max(complete_seqs_scores)

                    i = complete_seqs_scores.index(max_score)
                    seq = complete_seqs[i]

                    # References
                    img_caps = allcaps[0].tolist()

                    img_captions = list(
                        map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                            img_caps))  # remove <start> and pads

                    img_captions = [c for c in img_captions if len(c) > 0]
                    refs_nltk.append(img_captions)

                    img_caps = [' '.join(c) for c in img_captions]
                    # print(img_caps)
                    references.append(img_caps)

                    # Hypotheses
                    hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<pad>']}])
                    hyps_nltk.append(hypothesis)

                    hypothesis = ' '.join(hypothesis)
                    hypotheses.append(hypothesis)

                    assert len(references) == len(hypotheses)
                    assert len(refs_nltk) == len(hyps_nltk)

                hyp_file.write(hypothesis)
                hyp_file.write('\n')

    # Calculate scores
    metrics_dict = nlgeval.compute_metrics(references, hypotheses)

    nltk_bleu4 = corpus_bleu(refs_nltk, hyps_nltk, smoothing_function=smoothing_method)

    metrics_dict.update({'nltk_bleu4': nltk_bleu4})

    # from https://github.com/Maluuba/nlg-eval
    # where references is a list of lists of ground truth reference text strings and hypothesis is a list of
    # hypothesis text strings. Each inner list in references is one set of references for the hypothesis
    # (a list of single reference strings for each sentence in hypothesis in the same order).

    return metrics_dict


if __name__ == '__main__':

    beam_size = 5

    # Parameters
    data_folder = 'final_dataset'  # folder with data files saved by create_input_files.py
    data_name = 'coconl'  # base name shared by data files

    checkpoint_file = 'model_COCONLimage_cider_2019-10-31-13-53-9.pkl'  # model checkpoint

    model_type = 'image'

    print(model_type)

    word_map_file = 'final_dataset/WORDMAP_union.json'  # word map, ensure it's the same the data was encoded with and the model was trained with

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    # DataLoader
    test_set = DidecDatasetCOCONL(data_folder, data_name, 'test', model_type)

    loader = torch.utils.data.DataLoader(test_set,
                                         batch_size=1, shuffle=False, num_workers=4,
                                         pin_memory=torch.cuda.is_available())

    # Load model

    torch.nn.dump_patches = True
    checkpoint = torch.load(checkpoint_file, map_location=device)

    print(checkpoint['args'])
    print('Metrics', checkpoint['metrics_dict'], 'Loss', checkpoint['loss'], 'Epoch', checkpoint['epoch'])

    cp_args = checkpoint['args']

    # Load word map (word2id)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    rev_word_map = {v: k for k, v in word_map.items()}

    vocab_size = len(word_map)
    print('vs', vocab_size)

    embedding_dim = cp_args.emb_dim
    hidden_dim = cp_args.decoder_dim
    attention_dim = cp_args.attention_dim
    features_dim = cp_args.features_dim
    dropout = cp_args.dropout

    decoder = DecoderWithAttention(vocab_size, embedding_dim, hidden_dim, attention_dim, features_dim, dropout, device)

    decoder.load_state_dict(checkpoint['model_state_dict'])

    decoder = decoder.to(device)

    decoder.eval()

    nlgeval = NLGEval()  # loads the evaluator

    metrics_dict = evaluate(beam_size, decoder, loader, device, word_map, vocab_size, rev_word_map, nlgeval, 'test')

    print(metrics_dict)