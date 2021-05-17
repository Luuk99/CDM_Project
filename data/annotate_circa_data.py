from nltk import pos_tag, word_tokenize, download
from nltk.corpus import wordnet as wn, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="Discarded redundant search")

# Values below are fixed default values, but can be specified by cmd arguments too

WORDNET_TOPIC_DEPTH   = 6   # top-down (most start with 'entity') - empirically set
                            # only used for naive case without tree traversing
MAXIMUM_LABEL_DENSITY = 100 # amount of most common occurring labels every QA can 'choose' from
                            # the higher this amount, the more specific labels will be
                            # note that the number of total labels <= MAXIMUM_LABEL_DENSITY
                        
FILENAME_IMPORTANT_WORD_ANNOTATIONS = "data/annotations/important_word_annotations_tfidf.dat" # plaintext (format QA::annotation)
FILENAME_TOPIC_ANNOTATIONS          = "data/annotations/topic_annotations_mld50.dat"          # plaintext (format QA::annotation)
FILENAME_TOPIC_ANNOTATIONS_LABELS   = "data/annotations/topic_annotations_labels_mld50.p"     # pickle

# lambda calculus func for easy recursive traversing tree
hypernyms = lambda l:l.hypernyms()

def annotateImportantWords(dataset, preload = True, context = False, hybrid = False, annotationFileName = None):
    """
    Function that generates a column of the most important words for every answer
    present in a given dataset. Dumps automatically generated results in a .dat
    file which can be manually corrected and pre-loaded later. Works by simply
    extracting the last noun in the answer (context = False), or by extracting
    the word with the highest TF-IDF value (context = True)
    
    Inputs:
        dataset - circa dataset loaded from huggingface
        preload - if set to True, data will be preloaded and not generated
                        (from file specified in top of this document)
        context - if set to True, TF-IDF will be used (measuring the whole set as "context") to extract the noun with highest TF-IDF (otherwise nothing)
        hybrid  - if set to True, TF-IDF will ONLY be used if there is no last noun (and any PoS tagged word will be extracted)
    Outputs:
        importantWordsColumn - column of same dimensions of len(dataset) with respective words
    """
    
    importantWordsColumn = []
    
    if not annotationFileName:
        annotationFileName = FILENAME_IMPORTANT_WORD_ANNOTATIONS

    if not preload:
        print("Starting automatic annotation of most important words\r\nErrors can be corrected manually afterwards and re-(pre-)loaded.")
        
        # automatic annotations are automatically saved (and should be corrected afterwards)
        annotationFile = open(annotationFileName, 'w') # note: overwriting!
        
        # prepare TF-IDF
        if context or hybrid:
            vectorizer   = TfidfVectorizer(sublinear_tf=True, stop_words='english')
            
            if not hybrid: # we need to remove all non-nouns to get a better TF-IDF estimate
                fullDocumentSet = []
                
                for i in range(len(dataset['answer-Y'])):
                    pos_tags = pos_tag(word_tokenize(dataset[i]['answer-Y']))
                    fullDocumentSet.append(' '.join([pos_tag[0] for pos_tag in pos_tags if pos_tag[1][0:2] == 'NN']))
            else:
                fullDocumentSet = dataset['answer-Y']
            
            TFIDFMatrix  = vectorizer.fit_transform(fullDocumentSet)
            wordMappings = vectorizer.get_feature_names()
        
        # looping over all samples
        for i in range(len(dataset)):
            noun = "" # base
            
            if context:
                wordMappingsIndex = TFIDFMatrix[i].A.argmax(axis=1)
                
                if wordMappingsIndex:
                    noun = wordMappings[wordMappingsIndex[0]]
                
            else: # extract last noun
                pos_tags = pos_tag(word_tokenize(dataset[i]['answer-Y']))
                nouns = [pos_tag[0] for pos_tag in pos_tags if pos_tag[1][0:2] == 'NN'] # taking both singular and plural
                
                if nouns:
                    noun = nouns[-1] # last noun
                elif hybrid:
                    wordMappingsIndex = TFIDFMatrix[i].A.argmax(axis=1)
                
                    if wordMappingsIndex:
                        noun = wordMappings[wordMappingsIndex[0]]
                
            annotationFile.write(dataset[i]['answer-Y'] + '::' + noun + "\n") # sequence :: doesn't occur in the answers
            
            importantWordsColumn.append(noun)
        
        annotationFile.close()
        
    else:
        print("Pre-loading annotations for most important word in answer")
        
        with open(annotationFileName) as annotationFile:
            for annotation in annotationFile:
                importantWordsColumn.append(annotation.split('::')[1].strip('\n'))
    
    return importantWordsColumn
    
def annotateWordNetTopics(dataset, importantWordsColumn, preload = None, traverseAll = False, topic_depth = None, label_density = None, annotationFileName = None, annotationLabelsFileName = None):
    """
    Function that generates a high-level topic for every answer present in a given
    dataset. Note that the column with the most important word in the answer must
    be present. Dumps automatically generated results in a .dat file. Works by
    constructing a WordNet hierarchy from bottom (import word) to top ('entity'),
    from which then (top-down) a node is chosen as the 'topic' (node level specified
    at the top of this document).
    
    NOTE: synsets (including synonyms) are recursively parsed - classes are then taken
          based on total number of output labels
    
    Inputs:
        dataset     - circa dataset loaded from huggingface
        preload     - if set to True, data will be preloaded and not generated
                        (from file specified in top of this document)
        traverseAll - if set to True, all possible lemmas will be recursively
                      loaded from all hypernyms and synonyms
    Outputs:
        importantWordsColumn - column of same dimensions of len(dataset) with respective words
    """
    download('wordnet') # WordNet data must be downloaded locally

    if not topic_depth:
        topic_depth   = WORDNET_TOPIC_DEPTH
        
    if not label_density:
        label_density = MAXIMUM_LABEL_DENSITY
        
    if not annotationFileName:
        annotationFileName = FILENAME_TOPIC_ANNOTATIONS
        
    if not annotationLabelsFileName:
        annotationLabelsFileName = FILENAME_TOPIC_ANNOTATIONS_LABELS

    topicsColumn = []

    if not preload:
        print("Starting automatic annotation of high-level topics based on most important word")
        
        # automatic annotations are automatically saved
        annotationFile = open(annotationFileName, 'w')
        
        if traverseAll: # after traversing, 'sweet spot' of number of labels need to be found
            allLemmasColumn = []
        
        for i in range(len(dataset)):
            topic = "" # base
            
            if importantWordsColumn[i]:
                hierarchicalHypernyms = []
                
                baseSynsets = wn.synsets(importantWordsColumn[i], pos=wn.NOUN) # only nouns
                
                if baseSynsets:
                    if not traverseAll: # naive case (only first/left nodes)
                        synsetHypernyms = baseSynsets[0].hypernyms()
                        
                        while synsetHypernyms:
                            hierarchicalHypernyms.append(synsetHypernyms[0])
                            synsetHypernyms = synsetHypernyms[0].hypernyms()
                            
                        if len(hierarchicalHypernyms) > topic_depth:
                            topic = hierarchicalHypernyms[-topic_depth].lemmas()[0].name()
                        elif hierarchicalHypernyms:
                            topic = hierarchicalHypernyms[0].lemmas()[0].name() # a synset can have multiple lemma's
                
                    else: # recursively collecting all synset lemmas
                        topicLemmas     = set()
                        synsetQueue     = set(baseSynsets)
                        synsetQueueSeen = set()
                        
                        while synsetQueue:
                            synsetCurrent = synsetQueue.pop()
                            synsetCurrentHypernyms = synsetCurrent.closure(hypernyms)
                            
                            # add lemmas to lemma set
                            for lemma in synsetCurrent.lemmas():
                                topicLemmas.add(lemma.name())
                                
                            # add synsets to queue
                            for synsetCurrentHypernym in synsetCurrentHypernyms:
                                if synsetCurrentHypernym not in synsetQueueSeen:
                                    synsetQueue.add(synsetCurrentHypernym)
                                    synsetQueueSeen.add(synsetCurrentHypernym)
                            
                        allLemmasColumn.append((dataset[i]['answer-Y'], list(topicLemmas)))
                
                elif traverseAll: # we have to write an empty list for now
                    allLemmasColumn.append((dataset[i]['answer-Y'], []))
            elif traverseAll:
                allLemmasColumn.append((dataset[i]['answer-Y'], []))
                
            if not traverseAll: # then we can immediately write to annotation file and column
                annotationFile.write(dataset[i]['answer-Y'] + '::' + topic + "\n")
                topicsColumn.append(topic)
        
        if traverseAll:
            # the procedure is as follows:
            # 1. occurring lemmas will be counted and descendingly sorted
            # -> expectation is that the more specific a lemma is, the less counts it gets
            # 2. from all lists of lemmas (for each QA pair) the LEAST occurring lemma will
            #    be chosen from the top X of possible lemmas, where X is some sort of
            #    threshold value that controls the maximum amount of classes that can be
            #    generated. Note that there is ALWAYS a possibility within this ordered
            #    list of lemma's, as all synsets are at least connected to 'entity'.
            # -> these are the class labels! :)
            
            labelCounts = Counter(x for _, xs in allLemmasColumn for x in set(xs))
            mostCommonLabels = [x for x, _ in labelCounts.most_common(label_density)][::-1] # reversed
            
            for answer_y, lemmas in allLemmasColumn:
                topic = ""
                most_specific = label_density
                if lemmas:
                    for lemma in lemmas:
                        if lemma in mostCommonLabels:
                            lemmaIndex = mostCommonLabels.index(lemma)
                            if lemmaIndex < most_specific:
                                most_specific = lemmaIndex
                                topic = lemma
                annotationFile.write(answer_y + '::' + topic + "\n")
                topicsColumn.append(topic)
        
        annotationFile.close()
        
        usedLabels = set(topicsColumn)
        
        print('Total amount of distinctive topic labels: ' + str(len(usedLabels)) + ' (' + ('NOT ' if '' not in usedLabels else '') + 'including empty label)')
        pickle.dump(usedLabels, open(annotationLabelsFileName, "wb"))
        
    else:
        print("Pre-loading annotations for topics in answer")
        
        with open(annotationFileName) as annotationFile:
            for annotation in annotationFile:
                topicsColumn.append(annotation.split('::')[1].strip('\n'))
                
        usedLabels = pickle.load(open(annotationLabelsFileName, "rb" ))
    
    usedLabels = list(usedLabels)
    
    if '' in usedLabels:
        # move to end so there is no gap in label numbers if used as a task later
        usedLabels.append(usedLabels.pop(usedLabels.index('')))
    
    labelIndices = [usedLabels.index(label) for label in topicsColumn]
    
    return topicsColumn, labelIndices, usedLabels