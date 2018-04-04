# coding: utf-8

from nltk.parse.stanford import StanfordParser
parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
a=list(parser.raw_parse("Please edit your question to be more clear about what you need"))[0]
a=list(parser.raw_parse("Please edit your question to be more clear about what you need"))[0]
print(a)
