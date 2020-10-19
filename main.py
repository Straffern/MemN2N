import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model import MemN2N
from helpers import dataloader, get_fname, get_params


def train(train_iter, model, optimizer, epochs, max_clip, valid_iter=None):
    total_loss = 0
    valid_data = list(valid_iter)
    valid_loss = None
    next_epoch_to_report = 5
    pad = model.vocab.stoi['<pad>']

    for _, batch in enumerate(train_iter, start=1):
        story = batch.story
        query = batch.query
        answer = batch.answer

        optimizer.zero_grad()
        outputs = model(story, query)
        loss = F.nll_loss(outputs, answer.view(-1), ignore_index=pad, reduction='sum')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_clip)
        optimizer.step()
        total_loss += loss.item()

        # linear start
        if model.use_ls:
            loss = 0
            for k, batch in enumerate(valid_data, start=1):
                story = batch.story
                query = batch.query
                answer = batch.answer
                outputs = model(story, query)
                loss += F.nll_loss(outputs, answer.view(-1), ignore_index=pad, reduction='sum').item()
            loss = loss / k
            if valid_loss and valid_loss <= loss:
                model.use_ls = False
            else:
                valid_loss = loss

        if train_iter.epoch == next_epoch_to_report:
            print("#! epoch {:d} average batch loss: {:5.4f}".format(
                int(train_iter.epoch), total_loss / len(train_iter)))
            next_epoch_to_report += 5
        if int(train_iter.epoch) == train_iter.epoch:
            total_loss = 0
        if train_iter.epoch == epochs:
            break
        if ((train_iter.epoch + 1) % 25 == 0):
            for g in optim.param_groups:
                g['lr'] = g['lr']/2


def eval(test_iter, model, task):
    total_error = 0
    failed_tasks = []

    story_vocab = test_iter.dataset.fields['story'].vocab
    query_vocab = test_iter.dataset.fields['query'].vocab
    answer_vocab = test_iter.dataset.fields['answer'].vocab

    failed_tests = ""


    for k, batch in enumerate(test_iter, start=1):
        story = batch.story
        query = batch.query
        answer = batch.answer
        outputs = model(story, query)
        _, outputs = torch.max(outputs, -1)
        # log failed tasks
        # failed_tasks = failed_tasks + [    ([story_vocab.itos[s] for sublist in stry[0:3] for s in sublist if s != 0] ,
        #                     [query_vocab.itos[s] for s in qry], 
        #                     [answer_vocab.itos[s] for s in ans]) for i, (stry,qry,ans) 
        #                 in enumerate(zip(story,query,answer)) 
        #                 if outputs[i] != answer[i]]
        failed_tasks = np.where(outputs != answer.view(-1))[0]

        if len(failed_tasks) > 0:
            failed_stories = [[' '.join([story_vocab.itos[x] for x in sublist if x != 0]) for sublist in astory if not all(i == 0 for i in sublist) ] for astory in story[failed_tasks] ]
            failed_queries = [' '.join([query_vocab.itos[x] for x in question]) for question in query[failed_tasks] ]
            failed_answers = [answer_vocab.itos[question]  for question in answer[failed_tasks] ]
            failed_predictions = [answer_vocab.itos[question] for question in outputs[failed_tasks] ]

            for i in range(len(failed_tasks)):
                context = '\n'.join(failed_stories[i])
                _query = failed_queries[i] + '?'
                _answer = failed_answers[i]
                _prediction = failed_predictions[i]

                failed_tests += "\t\t||STORY||\n"
                failed_tests += context + '\n'
                failed_tests += "\t\t||QUESTION||\n"
                failed_tests += _query + '\n'
                failed_tests += "\t\t||ANSWER||\n"
                failed_tests += _answer + '\n'
                failed_tests += "\t\t||PREDICTED||\n"
                failed_tests += _prediction + '\n'


        total_error += torch.mean((outputs != answer.view(-1)).float()).item()
    
    # save failed_tasks to file:
    if failed_tests != "":
        with open('.failed_tasks/'+f'task_{task}.txt', 'w') as f:
            f.write(failed_tests)

    print("#! average error: {:5.1f}".format(total_error / k * 100))


def run(config):
    print("#! preparing data...")
    train_iter, valid_iter, test_iter, vocab = dataloader(config.batch_size, config.memory_size,
                                                          config.task, config.joint, config.tenk)

    print("#! instantiating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemN2N(get_params(config), vocab).to(device)

    if config.file:
        with open(os.path.join(config.save_dir, config.file), 'rb') as f:
            if torch.cuda.is_available():
                state_dict = torch.load(f, map_location=lambda storage, loc: storage.cuda())
            else:
                state_dict = torch.load(f, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

    if config.train:
        print("#! training...")
        optimizer = optim.Adam(model.parameters(), config.lr)
        train(train_iter, model, optimizer, config.num_epochs, config.max_clip, valid_iter)
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        torch.save(model.state_dict(), os.path.join(config.save_dir, get_fname(config)))

    print("#! testing...")
    with torch.no_grad():
        eval(test_iter, model, config.task)
