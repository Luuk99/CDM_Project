from nltk import pos_tag, word_tokenize, download
from nltk.corpus import wordnet as wn

WORDNET_TOPIC_DEPTH = 6 # top-down (most start with 'entity') - empirically set
                        # only used for naive case without tree traversing
                        
FILENAME_IMPORTANT_WORD_ANNOTATIONS = "data/annotations/important_word_annotations.dat"
FILENAME_TOPIC_ANNOTATIONS          = "data/annotations/topic_annotations.dat"

# lambda calculus func for easy recursive traversing tree
hypernyms = lambda l:l.hypernyms()

def annotateImportantWords(dataset, preload = True):
    """
    Function that generates a column of the most important words for every answer
    present in a given dataset. Dumps automatically generated results in a .dat
    file which can be manually corrected and pre-loaded later. Works by simply
    extracting the last noun in the answer (NOTE: could definitely be improved!)
    
    Inputs:
        dataset - circa dataset loaded from huggingface
        preload - if set to True, data will be preloaded and not generated
                        (from file specified in top of this document)
    Outputs:
        importantWordsColumn - column of same dimensions of len(dataset) with respective words
    """
    
    importantWordsColumn = []

    if not preload:
        print("Starting automatic annotation of most important words\r\nErrors can be corrected manually afterwards and re-(pre-)loaded.")
        
        # automatic annotations are automatically saved (and should be corrected afterwards)
        annotationFile = open(FILENAME_IMPORTANT_WORD_ANNOTATIONS, 'w') # note: overwriting!
        
        for i in range(len(dataset)):
            noun = "" # base
            pos_tags = pos_tag(word_tokenize(dataset[i]['answer-Y']))
            nouns = [pos_tag[0] for pos_tag in pos_tags if pos_tag[1][0:2] == 'NN'] # taking both singular and plural
            
            if nouns:
                noun = nouns[-1] # last noun
                
            annotationFile.write(dataset[i]['answer-Y'] + '::' + noun + "\n") # sequence :: doesn't occur in the answers
            
            importantWordsColumn.append(noun)
        
        annotationFile.close()
        
    else:
        print("Pre-loading annotations for most important word in answer")
        
        with open(FILENAME_IMPORTANT_WORD_ANNOTATIONS) as annotationFile:
            for annotation in annotationFile:
                importantWordsColumn.append(annotation.split('::')[1].strip('\n'))
    
    return importantWordsColumn
    
def annotateWordNetTopics(dataset, preload = None, traverseAll = True):
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

    topicsColumn = []

    if not preload:
        print("Starting automatic annotation of high-level topics based on most important word")
        
        # automatic annotations are automatically saved
        annotationFile = open(FILENAME_TOPIC_ANNOTATIONS, 'w')
        
        if traverseAll: # after traversing, 'sweet spot' of number of labels need to be found
            allLemmasColumn = []
        
        for i in range(len(dataset)):
            topic = "" # base
            
            if dataset[i]['most_important_word_in_answer']:
                hierarchicalHypernyms = []
                
                baseSynsets = wn.synsets(dataset[i]['most_important_word_in_answer'], pos=wn.NOUN) # only nouns
                
                if baseSynsets:
                    if not traverseAll: # naive case (only first/left nodes)
                        synsetHypernyms = baseSynsets[0].hypernyms()
                        
                        while synsetHypernyms:
                            hierarchicalHypernyms.append(synsetHypernyms[0])
                            synsetHypernyms = synsetHypernyms[0].hypernyms()
                            
                        if len(hierarchicalHypernyms) > WORDNET_TOPIC_DEPTH:
                            topic = hierarchicalHypernyms[-WORDNET_TOPIC_DEPTH].lemmas()[0].name()
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
            # here, some heuristic tool should be used to choose the best entries from each list in allLemmasColumn
            
            for answer_y, lemmas in allLemmasColumn:
                topic = ""
                if lemmas:
                    topic = lemmas[0] # this obviously is a super naive heuristic
            
                annotationFile.write(answer_y + '::' + topic + "\n")
                topicsColumn.append(topic)
        
        annotationFile.close()
        
    else:
        print("Pre-loading annotations for most important word in answer")
        
        with open(FILENAME_TOPIC_ANNOTATIONS) as annotationFile:
            for annotation in annotationFile:
                topicsColumn.append(annotation.split('::')[1].strip('\n'))
    
    return topicsColumn