import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.utils.data

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nlgeval import NLGEval

from gensim.models import KeyedVectors

import time

import os
import json

import argparse

import datetime

from models.model_pret_image import DecoderWithAttention

from utils.DidecDataset import DidecDataset
import eval_pret_image_NWMT
from utils.util_funcs import AverageMeter, accuracy, adjust_learning_rate

from utils.Metric_New import Metric

def save_model(model, epoch, scores, metrics_dict, losses, optimizer, args, metric, timestamp, seed):

    file_name = 'model_pret_image__NWMT' + str(seed) + '_' + metric + '_' + timestamp + '.pkl'

    print(file_name)

    torch.save({
        'metrics_dict' : metrics_dict,
        'args' : args,
        'scores' : scores,
        'epoch': epoch,
        'loss' : losses,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, file_name)


def print_predictions(scores, targets, i2w):

    target_sentences = []
    predicted_sentences = []

    for tg in targets:
        target_sentence = ''
        for t in tg:
            ww = i2w[int(t)]

            if ww != '<pad>':
                target_sentence += i2w[int(t)] + ' '

        target_sentences.append(target_sentence)

    for scr in scores:

        values, indices = torch.max(scr, 1)
        predicted_sentence = ''

        for sc_ind in indices:

            w_a = i2w[int(sc_ind)]
            if w_a != '<pad>':
                predicted_sentence += w_a + ' '

        predicted_sentences.append(predicted_sentence)

    for b_i in range(len(target_sentences)):
        print('T', target_sentences[b_i])
        print('P', predicted_sentences[b_i])
        print()

def validate(val_loader, decoder, criterion_ce, i2w, device, print_freq, word_map, current_epoch, break_flag, top_x, smoothing_method, print_flag):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad():
        for i, data in enumerate(val_loader):

            if break_flag and i == 5:
                break  # only 5 batches

            print('val i', i)
            imgs, caps, caplens, allcaps = data

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            if print_flag:
                print_predictions(scores, targets, i2w)

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion_ce(scores, targets)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, top_x)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-{topx} Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, topx=top_x, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            # DIDEC caps of other participants come here

            # print(allcaps.shape)

            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()

                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads

                refs_per_img = []

                for ic in img_captions:
                    if len(ic) > 0:
                        refs_per_img.append(ic)

                references.append(refs_per_img)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            # print('scr', scores_copy)
            # print('preds', preds)

            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads #SUPERFLUOUS ENDS stay for teacher forcing
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores

    #print('refshyps')
    #print(references)
    #print(hypotheses)

    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothing_method)
    bleu4 = round(bleu4,4)

    print('\n * LOSS - {loss.avg:.3f}, TOP-{topx} ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            topx=top_x,
            top5=top5accs,
            bleu=bleu4))

    return bleu4, losses


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_folder", type=str, default='final_dataset')
    parser.add_argument("-data_name", type=str, default='didec')
    parser.add_argument("-emb_dim", type=int, default=2048)
    parser.add_argument("-attention_dim", type=int, default=1024)
    parser.add_argument("-decoder_dim", type=int, default=1024)
    parser.add_argument("-features_dim", type=int, default=2048)
    parser.add_argument("-dropout", type=float, default=0.0)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-break_flag", action='store_true')
    parser.add_argument("-shuffle_flag", action='store_true')
    parser.add_argument("-num_workers", type=int, default=0)
    parser.add_argument("-print_flag", action='store_true')
    parser.add_argument("-print_freq", type=int, default=10)
    parser.add_argument("-checkpoint", type=str, default=None)
    parser.add_argument("-learning_rate", type=float, default=0.02)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-top_x", type=int, default=5)
    parser.add_argument("-metric", type=str, default='ours')

    args = parser.parse_args()

    print(args)

    model_type = 'image'
    metric = args.metric #loss or cider

    t = datetime.datetime.now()
    timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data_folder = args.data_folder
    data_name = args.data_name

    batch_size = args.batch_size
    break_flag = args.break_flag  # to break after 5 batches, for debugging reasons
    shuffle_flag = args.shuffle_flag  # both train and val
    print_flag = args.print_flag  # prints target and hypotheses

    workers = args.num_workers  # for data-loading; right now, only 1 works with h5py

    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True) # evaluator for multiple metrics

    embedding_model = KeyedVectors.load_word2vec_format("data/cow-big/big/cow-big.txt", binary=False)
    metric_obj = Metric(embedding_model)

    train_set = DidecDataset(data_folder, data_name, 'train', model_type)
    val_set = DidecDataset(data_folder, data_name, 'val', model_type)
    test_set = DidecDataset(data_folder, data_name, 'test', model_type)

    load_params = {'batch_size': batch_size, 'shuffle': shuffle_flag, 'num_workers': workers, 'pin_memory': True}
    load_params_test = {'batch_size': batch_size, 'shuffle': False, 'num_workers': workers, 'pin_memory': True}

    training_loader = torch.utils.data.DataLoader(train_set, **load_params)
    val_loader = torch.utils.data.DataLoader(val_set, **load_params_test)
    test_loader = torch.utils.data.DataLoader(test_set, **load_params_test)

    load_params_beam = {'batch_size': 1, 'shuffle': False, 'num_workers': workers, 'pin_memory': True}
    val_loader_beam = torch.utils.data.DataLoader(val_set, **load_params_beam)

    print('train len', len(train_set))
    print('val len', len(val_set))
    print('test len', len(test_set))


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    cudnn.benchmark = True

    emb_dim = args.emb_dim  # dimension of word embeddings
    attention_dim = args.attention_dim  # dimension of attention linear layers
    decoder_dim = args.decoder_dim  # dimension of decoder RNN
    features_dim = args.features_dim #img dim
    learning_rate = args.learning_rate  # learning rate for the optimizer
    dropout = args.dropout

    top_x = args.top_x

    start_epoch = 0
    epochs = args.epochs  # number of epochs to train for (if early stopping is not triggered)
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU

    best_bleu4 = 0.0  # to store the best BLEU-4 score
    best_cider = 0.0 # to store the best CIDER score
    best_loss = 0.0 # to store the best Cross-Entropy loss

    best_ours = 100.0 # to store the best metric (ours)

    smoothing_method = SmoothingFunction().method1  # epsilon method for bleu

    print_freq = args.print_freq  # print training/validation stats every __ batches
    checkpoint = args.checkpoint  # path to checkpoint, None if none

    # Read word map
    # WARNING union vocab from pretrained + didec itself
    word_map_file = os.path.join(data_folder, 'WORDMAP_union.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    rev_word_map = {v: k for k, v in word_map.items()}

    print('vocab len', len(word_map))

    i2w = dict()

    for w in word_map:
        i2w[word_map[w]] = w

    print('using a pretrained checkpoint')

    checkpoint = torch.load(checkpoint, map_location=device)

    #args = checkpoint['args']

    # construct the model and optimizer with values from args

    decoder = DecoderWithAttention(vocab_size=len(word_map), embedding_dim=emb_dim, hidden_dim=decoder_dim,
                                   attention_dim=attention_dim, features_dim=features_dim, dropout=dropout, device=device)

    # fill in state dicts
    decoder.load_state_dict(checkpoint['model_state_dict'])

    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    decoder = decoder.to(device)

    losses = []

    best_loss = 100000
    best_accuracy = -1
    prev_accuracy = -1
    prev_loss = 100000

    t = datetime.datetime.now()
    begin_timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)

    for epoch in range(epochs):
         print('Epoch', epoch)
         print('Train')

         # Decay learning rate if there is no improvement for 5 consecutive epochs
         # halved
         # Terminate training after 50
         if epochs_since_improvement == 50:

             duration = datetime.datetime.now() - t

             print('model ending duration', duration)

             break

         #if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
         #    adjust_learning_rate(decoder_optimizer, 0.5)

         decoder.train()
         torch.enable_grad()

         batch_time = AverageMeter()  # forward prop. + back prop. time
         data_time = AverageMeter()  # data loading time
         losses = AverageMeter()  # loss (per word decoded)
         top5accs = AverageMeter()  # top5 accuracy

         start = time.time()

         count = 0

         for i, data in enumerate(training_loader):

            if break_flag and count == 1:
                break

            count += 1

            image_set, caption_texts, caption_lengths = data

            # Move to GPU, if available
            imgs = image_set.to(device)
            caps = caption_texts.to(device)
            caplens = caption_lengths.to(device)

            # Forward prop.
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            if print_flag:
                print_predictions(scores, targets, i2w)


            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Backprop.
            decoder_optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            # Clip gradients when they are getting too large
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 5)

            # Update weights
            decoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, top_x)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-{topx} Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(training_loader),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              topx=top_x, top5=top5accs))

         decoder.eval()

         # One epoch's validation
         recent_bleu4, recent_loss = validate(val_loader=val_loader,
                                decoder=decoder,
                                criterion_ce=criterion, i2w=i2w, device=device, print_freq=print_freq,
                                word_map=word_map, current_epoch=epoch, break_flag=break_flag, top_x=top_x, smoothing_method=smoothing_method, print_flag=print_flag)

         # also eval beam search
         beam_size = 5

         metrics_dict, recent_ours_beam = eval_pret_image_NWMT.evaluate(beam_size, decoder, val_loader_beam, device, word_map, len(word_map), rev_word_map, nlgeval, 'val', metric_obj)

         print(metrics_dict, 'Ours:', recent_ours_beam)

         recent_cider = metrics_dict['CIDEr']
         recent_bleu4_bm = metrics_dict['Bleu_4'] # USING THIS

         if metric == 'cider':

             # Check if there was an improvement
             is_best = recent_cider > best_cider
             best_cider = max(recent_cider, best_cider)

             recent_metric = recent_cider

             print('Best CIDEr', best_cider)

         elif metric == 'bleu':

             # Check if there was an improvement
             is_best = recent_bleu4_bm > best_bleu4
             best_bleu4 = max(recent_bleu4_bm, best_bleu4)

             recent_metric = recent_bleu4_bm

             print('Best BLEU-4', best_bleu4)

         elif metric == 'loss':

             # Check if there was an improvement
             recent_loss = recent_loss.avg # AverageMeter object

             is_best = recent_loss < best_loss
             best_loss = min(recent_loss, best_loss) # min here

             recent_metric = recent_loss

             print('Best Cross-Entropy Loss', best_loss)

         elif metric == 'ours':

             recent_ours_beam = recent_ours_beam[0] # avg, gen2ref, ref2gen, exp, harm
             is_best = recent_ours_beam < best_ours
             best_ours = min(recent_ours_beam, best_ours)  # min here

             recent_metric = recent_ours_beam

             print('Best (our) Metric score', best_ours)


         if not is_best:
             epochs_since_improvement += 1
             print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
         else:
             epochs_since_improvement = 0

             # Save checkpoint
             print('Saving current best model', timestamp)

             save_model(decoder, epoch, recent_metric, metrics_dict, losses, decoder_optimizer, args, metric, timestamp, seed)

             duration = datetime.datetime.now() - t

             print('model saving duration', duration)

