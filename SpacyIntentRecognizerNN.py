import spacy
import random
import json
import warnings
import time
from pathlib import Path
import numpy as np
from spacy.util import minibatch, compounding
from sklearn.metrics.pairwise import cosine_similarity

from spacy.util import minibatch, compounding, decaying
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# convert Chatette trainingdata to be analysed
def get_data(trainingData):
    DATA = []
    with open(trainingData) as json_file:
        data = json.load(json_file)

    random.shuffle(data)
    examples = data['rasa_nlu_data']['common_examples'][-10:]
    
    print("length examples: ",len(examples))
    for example in examples:
        sentence = example["text"]
        intent = example["intent"]
        #print("intent:",intent) 
        for entity in example["entities"]:
            label = entity["entity"]
            label_start = entity["start"]
            label_end = entity["end"]
            #puts the labels in the right format into TRAINING_DATA
            DATA = DATA + [(sentence, {"entities":[(label_start, label_end, label)]} ,{"intent":[intent]}),]

    return DATA

# Trains the model for entities, with parameters Training Data, labels also called Entities and amount of times to to be trained
def train_entity_recognition(model, output_dir , n_iter, TRAIN_DATA):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.load("en_core_web_lg")  # create blank Language class !! error !!
        print("loaded 'en_core_web_lg' model")
    
    # Entity Classification: 
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    # add labels
    for _, annotations, _ in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                #print("batch",batch)
                texts, annotations, _= zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    sgd=optimizer,
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            #print("Losses", losses)
        
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    return nlp


def train_intent(model, output_dir, iterator, TEST_DATA):
    """Train the entity recognizer using cosine_similarity"""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s' to train intent recognition " % model)
    else:
        # create blank Language class !! error !!
        nlp = spacy.load("en_core_web_lg")
        print("loaded blank language class !!error!!")

    # Calculate the length of sentences
    n_text = (len(TEST_DATA))
    # Calculate the dimensionality of nlp
    embedding_dim = nlp.vocab.vectors_length
    # Initialize the array with zeros: X
    X = np.zeros((n_text, embedding_dim))

    # Iterate over the sentences
    for idx, element in enumerate(TEST_DATA):
        # Pass each each sentence to the nlp object to create a document
        doc = nlp(element[0])
        # Save the document's .vector attribute to the corresponding row in X
        X[idx,:] = doc.vector

    """Test the intent recognizer"""
    print("Testing the intent recognizer")
    for text, _, _ in TEST_DATA:
        vector = nlp(text).vector
        #print('text:',text)
        scores = [cosine_similarity(X[i,:].reshape(-1,1), vector.reshape(-1,1)) for i in range (len(TEST_DATA))]
        #print("np.argmax(scores):", TEST_DATA[np.argmax(scores)])

     # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    return nlp


def intent_confusion_matrix(list):
    # builds an intent confusion matrix
    # format of list is:
    # [(predicted, actual Value),...,]
    # returns precision, recall and F1 score
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for elem in list:
        if elem[0] is elem[1]:
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1

    prec = true_positive / (true_positive + false_positive)
    rec = true_positive / (true_positive + false_negative)
    if  (prec + rec != 0):
        f1 = 2 * ((rec * prec) / (prec + rec))
    else:
        f1 = 0.0

    return prec, rec, f1

# test the model, with "TrainingData" as examples
# iterates for every example in TrainingData and extracts
# every sentence and entities out of it
# compares with GoldParse and scorer.score the predicted entities and 
# the annotated entities in the training data
def get_key(my_dict): 
    for key, _ in my_dict.items(): 
        return key 


def evaluate(model, output_dir, trainingData):
    start_time = time.perf_counter()
    print("evaluate: loading model ..")
    nlp = spacy.load(output_dir)
    scorer = []
    GOLDPARSE_DATA = []
    with open(trainingData) as json_file:
        data = json.load(json_file)

    #access dictionary and save into variables
    print("evaluate: Goldparse Data ..")
    examples = data['rasa_nlu_data']['common_examples'][-200:]
    print("examples:")
    print(examples)
    for example in examples:
        sentence = example["text"]
        cat = example["intent"]
        GOLDPARSE_DATA = GOLDPARSE_DATA + [(sentence, {"cats": {cat: 1}}), ]

    print(GOLDPARSE_DATA)
    for text, annotations in GOLDPARSE_DATA:
        doc_gold = nlp.make_doc(text)
        cat = annotations.get("cats")
        gold = GoldParse(doc_gold, cats=cat)
        print("debug cats 0\n")
        print(nlp(text))
        print("debug cats 1\n")
        print(nlp(text).cats)
        print("debug cats 2\n")
        print(nlp(text).cats.get)
        predicted = max(nlp(text).cats, key=nlp(text).cats.get)
        tupel = (predicted, get_key(gold.cats))
        scorer.append(tupel)
        p, r, f = intent_confusion_matrix(scorer)

    print("Testing data took ", time.perf_counter() - start_time, "to run")
    quickSave = open('quickSaveIntents.txt', 'w') 
    print('These are the values for Spacy', file = quickSave)
    print("Precision: ", prec, file = quickSave)
    print("Recall: ", rec, file = quickSave)
    print("F1: ",f1, file = quickSave)
    print("Performance: ", timing, file = quickSave)    
    quickSave.close() 
    
    return p, r, f


#train data here model here
if __name__ == "__main__":
    output_dir = "/home/mehcih/Desktop/NaturalLanguageProcessing/example spacy code/spacy-env/model"
    TRAINING_DATA = get_data("trainingData.json")
    TEST_DATA = get_data("chatette.json")
    
    print("TRAIN DATA SIZE:",np.size(TRAINING_DATA))
    print("TEST DATA SIZE:",np.size(TEST_DATA))
    
    model = None

    prec = []
    rec = []
    f1 = []
    timing = []
    iterations = range(1, 5, 1)

    # using these to increase performance by not repeating iterations 
    # but building on top of already trained model
    trueIter, tempTime = 0, 0
    print("Computing ..")
    for iter in iterations:
        trueIter = iter - trueIter
        start_time = time.perf_counter()
        print("Entities")
        model = train_entity_recognition(model, output_dir, 2, TRAINING_DATA)
        print("intents")
        model = train_intent(model, output_dir, trueIter, TRAINING_DATA)
        print("Training took", time.perf_counter() - start_time, "to run")
        currTime = time.perf_counter() - start_time
        tempTime = currTime + tempTime
        trueIter = iter
        p, r, f = evaluate(model, output_dir, "chatette.json")
        prec.append(p)
        rec.append(r)
        f1.append(f)
        timing.append(currTime + tempTime)

    #plots precision, recall, f1 and time to execute in 4 different graphs
    fig=plt.figure()
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    ax1.title.set_text('Precision')
    ax2.title.set_text('Recall')
    ax3.title.set_text('F1')
    ax4.title.set_text('Performance')

    ax1.set_ylabel('Precision in %')
    ax2.set_ylabel('Recall in %')
    ax3.set_ylabel('F1 in %')
    ax4.set_ylabel('time in s')

    ax1.set_xlabel('amount of iterations')
    ax2.set_xlabel('amount of iterations')
    ax3.set_xlabel('amount of iterations')
    ax4.set_xlabel('amount of iterations')

    ax1.plot(iterations, prec)
    ax2.plot(iterations, rec)
    ax3.plot(iterations,f1)
    ax4.plot(iterations,timing)

    #plt.ylim((95, 100))
    plt.show() 
