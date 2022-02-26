import spacy
import unidecode
import contractions
import numpy as np

class TextPreProcessing():
    def __init__(self):
        super(TextPreProcessing, self).__init__()

    def __remove_accented_chars(self, text):
        """remove accented characters from text, e.g. café"""
        text = unidecode.unidecode(text)
        return text

    def __expand_contractions(self, text):
        """expand shortened words, e.g. don't to do not"""
        text = contractions.fix(text)
        return text

    def __to_lowercase(self, text):
        """to lowercase"""
        return text.lower()
    
    def __remove_whitespace(self, text):
        """remove extra whitespaces from text"""
        text = text.strip()
        return " ".join(text.split())

    def preprocess(self, doc):
        for i in range(len(doc)):
            text = doc[i]
            text = self.__to_lowercase(text)
            text = self.__remove_whitespace(text)
            text = self.__expand_contractions(text)
            text = self.__remove_accented_chars(text)

            doc[i] = text
        
        return doc

class TokenPreProcessing():
    def __init__(self):
        super(TokenPreProcessing, self).__init__()
        self.__language_model = spacy.load('en_core_web_md')
        self.__deselect_stopwords()
    
    def __deselect_stopwords(self, stopwords_list=['no', 'not']):
        """exclude words from spacy stopwords list"""
        deselect_stop_words = stopwords_list
        for w in deselect_stop_words:
            self.__language_model.vocab[w].is_stop = False

    def __tokenization(self, text):
        """tokenization step"""
        tokens = self.__language_model(text)
        return tokens

    def __remove_punctuations(self, tokens):
        """Remove punctuations"""
        new_tokens = []
        for token in tokens:
            if token.pos_ != 'PUNCT':
                new_tokens.append(token)  
        
        return new_tokens

    def __remove_special_characters(self, tokens):
        """Remove special characters"""
        new_tokens = []
        for token in tokens:
            if token.pos_ != 'SYM':
                new_tokens.append(token)  
        
        return new_tokens
    
    def __lemmatization(self, tokens):
        """Lemmatization"""
        new_tokens = []
        for token in tokens:
            if token.lemma_ != "-PRON-":
                new_tokens.append(token.lemma_)  
        
        return new_tokens

    def preprocess(self, doc):
        new_doc = []
        for i in range(len(doc)):
            text = doc[i]

            tokens = self.__tokenization(text)
            tokens = self.__remove_punctuations(tokens)
            tokens = self.__remove_special_characters(tokens)
            tokens = self.__lemmatization(tokens)

            new_doc.append(tokens)
        
        return np.array(new_doc, dtype = 'object')

"""This class joins TextPreprocessing and TokenPreprocessing"""
class PreProcessing():
    def __init__(self):
        super(PreProcessing, self).__init__()
        self.__text_preprocess = TextPreProcessing()
        self.__token_preprocess = TokenPreProcessing()
        
    def getDoc(self):
        return self.__doc

    def run(self, doc):
        self.__doc = self.__text_preprocess.preprocess(doc)
        self.__doc = self.__token_preprocess.preprocess(doc)



    

    
    


