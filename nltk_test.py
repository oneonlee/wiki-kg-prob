from nltk import pos_tag
from nltk.tokenize import sent_tokenize, NLTKWordTokenizer


paragraph = "The communications team, including driver and mechanic Eyjo Furteitsson, communication manager Nathan Hambrook-Skinner, and cinematographer Paddy Scott trailed the team in the 6x6 expedition truck built especially for the expedition by Arctic Trucks.The expedition consisted of three main components: Science."

sentence_candidates = sent_tokenize(paragraph)
sentence_candidates = [candidate.strip() for candidate in sentence_candidates]
sentence_candidates = [candidate for candidate in sentence_candidates if len(candidate.split(" ")) >= 4]
sentences = [candidate for candidate in sentence_candidates if candidate[-1] == "."]

print(sentences)