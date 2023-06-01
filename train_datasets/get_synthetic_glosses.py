"""
To run this you may need to download the model, paste the following into terminal:
    python3 -m spacy download de_core_news_lg
"""

import spacy
import numpy as np
import pandas as pd
import pdb
from torchmetrics.functional import word_error_rate

class GlossPredicter:
    def __init__(self, df) -> None:
        ### for POS
        self.NOUNS = ['NOUN', 'NE', 'NNE', 'PROPN']
        self.VERBS = ['VERB', 'VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP', 'VVIZU', 'VVPP', 'AUX', 'ROOT'] # For POS and DEP
        self.ADJECTIVES = ['ADJ', 'ADJA', 'ADJD']
        self.ADVERBS = ['ADV', 'PAV', 'PROAV']
        self.NUMERALS = ['NUM', 'CARD']

        ### for DEP
        self.SUBJECTS = ['sb', 'sbp', 'sp']
        self.OBJECTS = ['op', 'og', 'oa', 'oc', 'da', 'nk']
        self.NEGATIONS = ['ng'] # DEP

        self.VALID_POS = self.NOUNS + self.VERBS + self.ADJECTIVES + self.NUMERALS + self.ADVERBS
        self.model = spacy.load("de_core_news_lg")

        ### for data cleaning
        self.numbers = ['null', 'ein', 'zwei', 'drei', 'vier', 'funf', 'sechs', 'sieben', 'acht', 'neun', 'zehn', 'elf', 'zwölf', 'dreizehn', 'vierzehn', 'funfzehn', 'sechzehn', 'siebzehn', 'achtzehn', 'neunzehn', 'zwanzig', 'einundzwanzig', 'zweiundzwanzig', 'dreiundzwanzig', 'vierundzwanzig', 'funfundzwanzig', 'sechsundzwanzig', 'siebenundzwanzig', 'achtundzwanzig', 'neunundzwanzig', 'dreissig', 'einunddreissig']
        self.stopwords = ['dazu', 'dort', 'sein', 'und', 'ein', 'an', 'muss', 'bö', 'geben', 'Emsland', 'Linie', 'Uckermark', 'stellenweise', 'hochnebelfelder']
        self.clean_dict = {'norden' : 'nord',
                           'Nordhälfte' : 'nord',
                           'osten' : 'ost',
                           'süden' : 'sued',
                           'südosten' : 'suedost',
                           'südwesten' : 'suedwest',
                           'südhälfte' : 'sued',
                           'westhälfte' : 'west',
                           'westen' : 'west',
                           'nordwesten' : 'nordwest',
                           'nordosten' : 'nordost',
                           'kühl' : 'kuehl',
                           'kommend' : 'kommen',
                           'teilweise' : 'teil',
                           'sturmböen' : 'sturm',
                           }
        self.df = df

    def get_glosses(self, sent):
        properties = ['dep', 'lem', 'ent', 'pos']
        model_out = self.model(sent)
        sent_info = {i : {key : None for key in properties} for i in range(len(sent.strip().split(' ')))}
        final_indicies = list(range(len(sent.strip().split(' '))))
        # get sentence info
        for i, token in enumerate(model_out):
            if token.pos_ in self.VALID_POS: # check word type 
                sent_info[i]['pos'] = token.pos_
                sent_info[i]['ent'] = token.ent_type_
                sent_info[i]['dep'] = token.dep_
                sent_info[i]['lem'] = token.lemma_
            else:
                sent_info[i] = None
                final_indicies.remove(i)
        
        # Svap SVO --> SOV
        
        SOV = self.SVO2SOV(sent_info)
        for elem in SOV:
            O = elem[1]
            V = elem[2]
            final_indicies[O] = final_indicies[V]
            final_indicies[V] = final_indicies[O]
        
        # process sent info
        for key in final_indicies:
            
            #if sent_info[key]['pos'] == 'ADV': # add adverb to BOS
            #    final_indicies.remove(key)
            #    final_indicies = [key] + final_indicies
            
            if sent_info[key]['ent'] == 'LOC': # add location to BOS
                final_indicies.remove(key)
                final_indicies = [key] + final_indicies
            
            if sent_info[key]['dep'] in self.NEGATIONS: # move negation to EOS
                final_indicies.remove(key)
                final_indicies.append(key)

            # remove all nouns after c_1 in case of compound noun
            comp_keys = self.detect_compound_noun(sent_info)
            if len(comp_keys) > 0:
                for j in comp_keys:
                    try:
                        final_indicies.remove(comp_keys) 
                    except ValueError: # element has been previously removed
                        continue
            
        # generate final output
        out = []
        for idx in final_indicies:
            out.append(sent_info[idx]['lem'])
        
        # clean output
        for i, elem in enumerate(out):
            if elem.isdigit():
                out[i] = self.numbers[int(elem)]
            elif elem.lower() in self.stopwords:
                out.remove(elem)

            elif elem.lower() in self.clean_dict.keys():
                out[i] = self.clean_dict[elem.lower()]

        return ' '.join(out)

    def detect_compound_noun(self, sent_info):
        compound_keys = []
        for key in range(len(sent_info.keys()) - 1):
            try:
                if sent_info[key]['pos'] == 'NOUN' and sent_info[key + 1]['pos'] == 'NOUN':
                    compound_keys.append(key + 1)
            except TypeError: # Handle None values
                continue
        return compound_keys

    def SVO2SOV(self, sent_info):
        S, V, O = [], [], []
        idx = 0
        SOV = []
        for key in sent_info:
            if not sent_info[key] == None:
                if sent_info[key]['dep'] in self.SUBJECTS:
                    S.append(key)
                elif sent_info[key]['dep'] in self.VERBS or sent_info[key]['pos'] in self.VERBS:
                    V.append(key)
                elif sent_info[key]['dep'] in self.OBJECTS:
                    O.append(key)
                
                if len(S) + len(V) + len(O) % 3 == 0:
                    try:
                        SOV.append((S[idx], O[idx], V[idx]))
                        idx += 1
                    except:
                        continue
        return SOV
    
    def getSyntheticGlosses(self, save_path=None):
        glosses = []
        sents = self.df['translation']
        print("Generating synthetic glosses")
        for i, sent in enumerate(sents):
            glosses.append(self.get_glosses(sent))

            #if i % len(self.df) / 10 == 0:
            if i % 500 == 0:
                print(f"Iter: {i}/{len(self.df)}")

        self.df['synthetic glosses'] = glosses
        if save_path is not None:
            self.df.to_csv(save_path)
        
        return self.df
    
    def getPOSDistribution(self):
        dist = {}
        print("Checking ground truth POS distribtion")
        for i in range(len(self.df)):
            model_out = self.model(self.df.iloc[i]['orth'])
            for token in model_out:
                if token.pos_ not in dist.keys():
                    dist[token.pos_] = 1
                else:
                    dist[token.pos_] += 1
        return dist
    
    def getPOSOrderDist(self):
        dist = {33 : {},
                67 : {},
                100: {}}
        print("Checking ground truth POS placement distribtion")
        for i in range(len(self.df)):
            model_out = self.model(self.df.iloc[i]['orth'])
            N = len(model_out.text.split(' '))

            for i, token in enumerate(model_out):
                if (i+1) / N <= 0.33:
                    if token.pos_ not in dist[33].keys():
                        dist[33][token.pos_] = 1
                    else:
                        dist[33][token.pos_] += 1
                
                elif (0.33 <= (i+1) / N) and((i+1) / N <= 0.67):
                    if token.pos_ not in dist[67].keys():
                        dist[67][token.pos_] = 1
                    else:
                        dist[67][token.pos_] += 1
                
                elif (i+1) / N > 0.67:
                    if token.pos_ not in dist[100].keys():
                        dist[100][token.pos_] = 1
                    else:
                        dist[100][token.pos_] += 1

                    if token.pos_ not in dist.keys():
                        dist[token.pos_] = 1
                    else:
                        dist[token.pos_] += 1
        return dist  
        

def testWER(data):
    WERS = []
    for i in range(len(data)):
        #WERS.append(word_error_rate(data.iloc[i]['orth'].lower().split(' '), data.iloc[i]['synthetic glosses'].lower().split(' ')).item())
        WERS.append(word_error_rate(data.iloc[i]['orth'].lower().split(' '), data.iloc[i]['synthetic glosses'].lower().split(' ')).item())
        if np.mean(WERS) > 1:
            print(data.iloc[i]['orth'].lower())
            print(data.iloc[i]['synthetic glosses'].lower())
            pdb.set_trace()
    return WERS

def showSample(data, idx):
    print(f"Ground truth: {data.iloc[idx]['orth']}\nPred: {data.iloc[idx]['synthetic glosses']}")    

if __name__ == '__main__':
    SAVE_PATH = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.synthetic.glosses.csv'
    
    #DATA_PATH = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv' 
    #DATA_PATH = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv'
    DATA_PATH = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv'
    df = pd.read_csv(DATA_PATH, delimiter='|')
       
    GP = GlossPredicter(df)
    data = GP.getSyntheticGlosses(SAVE_PATH)
    WERS = testWER(data)
    print(f'mean WER: {np.mean(WERS)}')
    """
    dist = GP.getPOSDistribution()
    orderDist = GP.getPOSOrderDist()
    print("------------------- General distribution -------------------")
    print(f'{dist}\n\n')
    print("------------------- 33 distribution -------------------")
    print(f'{orderDist[33]}\n\n')
    print("------------------- 67 distribution -------------------")
    print(f'{orderDist[67]}\n\n')
    print("------------------- 100 distribution -------------------")
    print(f'{orderDist[100]}\n\n')
    """
    #pdb.set_trace()


"""
#### Preliminary examples

sent = "Hallo, Bigom habe einen kleinen wurst"
sent = "Ich liebe essen."
sent = 'Du essen würst'
#sent = 'Die apfel baum ist nicht sehr schön.'
sent = "Andreas fahren nach Dänemark. Ich liebe Tivoli"
nlp = spacy.load('de_core_news_lg')
doc = nlp(sent)
print(doc.text)
print(type(doc))
for token in doc:
    print(f"Word: {token.text} is of type: {token.pos_}")
    print("Token: ", token.lemma_)
    print("DEP: ", token.dep_)
    print("NER: ", token.ent_type_)
print(doc.ents) 
#print(doc.rights)

#print(spacy.info('de_core_news_lg'))
#print(token.text, token.pos_, token.dep_)

L1 = ['a', 'c', 'b']
L2 = 'a'#, 'b']
if 'a' in L1  and 'b' in L1 and 'c' in L1:
    print("HII")

    
explain = ['ROOT', 'ac', 'adc', 'ag', 'ams', 'app', 'avc', 'cc', 'cd', 'cj', 'cm', 
           'cp', 'cvc', 'da', 'dep', 'dm', 'ep', 'ju', 'mnr', 'mo', 'ng', 'nk', 'nmc', 
           'oa', 'oc', 'og', 'op', 'par', 'pd', 'pg', 'ph', 'pm', 'pnc', 'punct', 'rc', 
           're', 'rs', 'sb', 'sbp', 'svp', 'uc', 'vo']
for item in explain:
    print(f'Explanation for: {item} is:\n{spacy.explain(item)}')    
"""