import re
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, NLTKWordTokenizer

from datasets import load_dataset
from datasets import Dataset

from tqdm import tqdm

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

LENGTH = 256
dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")


def count_word_in_sentence(word, sentence):
    word = word.lower()
    sentence = sentence.lower()

    count = 0
    for sentence in re.finditer(word, sentence):
         count += 1
    
    return count


def get_target_indices_in_pos_tags(pos_tags, target_pos_tag, symbol_pos_tag, end_acceptable_pos_tag):
    allowed_pos_tag = target_pos_tag + symbol_pos_tag

    i = len(pos_tags)-1
    start_index = None
    end_index = None
    
    while i>=0:
        if pos_tags[i][1] not in allowed_pos_tag:
            break
        if end_index is None:
            end_index = i
        start_index = i
    
        i = i-1
    
    if start_index is not None and end_index is not None:
        
        while pos_tags[start_index][1] in symbol_pos_tag:
            start_index += 1
            
        if start_index > end_index:
            return None, None
        elif start_index >= len(pos_tags) or end_index >= len(pos_tags):
            return None, None
        elif pos_tags[end_index][1] not in end_acceptable_pos_tag:
            return None, None
        
        assert start_index <= end_index 
        assert start_index < len(pos_tags) and end_index<len(pos_tags)
        assert pos_tags[end_index][1] in end_acceptable_pos_tag
    
    return start_index, end_index


def split_sentence_into_input_and_output(sentence, pos_tags, span_list, start_index, end_index, symbol_pos_tag):
    input_sentence = None
    output_phrase = None
    
    if start_index is not None and end_index is not None:
        start_token = pos_tags[start_index][0]
        if start_token == "``" or start_token == "''":
            start_token = '"'
        end_token = pos_tags[end_index][0]
        if end_token == "``" or end_token == "''":
            end_token = '"'
        
        assert sentence.endswith(end_token)
        
        input_pos_tags = pos_tags[:start_index]
        
        if start_index == end_index:
            assert start_token == end_token
            
            if pos_tags[start_index][1] in symbol_pos_tag:
                return None, None
                
            # context에 phrase의 subset이 이미 있는지 검사
            for tok, pos_tag in input_pos_tags:
                if tok == start_token:
                    return None, None
            
            if count_word_in_sentence(start_token, sentence) != 1:
                return None, None
            
            start_span = span_list[start_index][0]
            input_sentence = sentence[:start_span]
            output_phrase = start_token
        
        else:
            assert start_index < end_index
            assert pos_tags[start_index][1] not in symbol_pos_tag

            # context에 phrase의 subset이 이미 있는지 검사
            for index in range(start_index, end_index+1):
                if pos_tags[index][1] in symbol_pos_tag:
                    continue
                for tok, pos_tag in input_pos_tags:
                    if tok == pos_tags[index][0]:
                        return None, None
                   
            ##### input_sentence 구하기 #####
            count = count_word_in_sentence(start_token, sentence)  
            assert count >= 1
            
            if count > 1:
                # count가 input에서 잡혔는지 확인
                for tok, pos_tag in input_pos_tags:
                    if tok == start_token:
                        return None, None
                    
            start_span = span_list[start_index][0]
            input_sentence = sentence[:start_span]
            output_phrase = sentence[start_span:]

    return input_sentence, output_phrase


def get_inputs_and_outputs_from_paragraph(paragraph, target_pos_tag, symbol_pos_tag, end_acceptable_pos_tag):
    # Tokenize the paragraph into sentences
    sentence_candidates = sent_tokenize(paragraph)
    sentence_candidates = [candidate.strip() for candidate in sentence_candidates]
    sentence_candidates = [candidate for candidate in sentence_candidates if len(candidate.split(" ")) >= 4]
    sentences = [candidate for candidate in sentence_candidates if candidate[-1] == "."]

    input_text_list = []
    output_answer_list = []

    for i in range(1, len(sentences)):
    # for sentence in sentences:
        sentence = sentences[i]
        sentence = sentence.rstrip(".")

        # Tokenize the sentence into words
        words = NLTKWordTokenizer().tokenize(sentence)

        # Get the span of each word
        span_list = list(NLTKWordTokenizer().span_tokenize(sentence))

        # Get the part of speech for each word
        pos_tags = pos_tag(words)

        assert len(pos_tags) == len(span_list)

        start_index, end_index = get_target_indices_in_pos_tags(pos_tags, target_pos_tag, symbol_pos_tag, end_acceptable_pos_tag)
        if start_index is None or end_index is None:
            continue

        input_sentence, output_phrase = split_sentence_into_input_and_output(sentence, pos_tags, span_list, start_index, end_index, symbol_pos_tag)
        if input_sentence is None or output_phrase is None:
            continue

        input_text = f"{sentences[i-1].strip()} {input_sentence}"

        continue_flag = False
        for subset in output_phrase.split():
            if count_word_in_sentence(subset, input_text) >= 1:
                continue_flag = True
                break
        if continue_flag:
            continue

        input_text_list.append(input_text)
        output_answer_list.append(output_phrase)

    return input_text_list, output_answer_list


def transform_dataset(target_pos_tag, symbol_pos_tag, end_acceptable_pos_tag):
    x_list = []
    y_list = []
    source_list = []

    for index, label in tqdm(enumerate(dataset["label"])):
        if label == 1:
            paragraph = dataset["input"][index]
            input_text_list, output_answer_list = get_inputs_and_outputs_from_paragraph(paragraph, target_pos_tag, symbol_pos_tag, end_acceptable_pos_tag)
            assert len(input_text_list) == len(output_answer_list)
            source_text_list = [paragraph] * len(input_text_list)
            
            x_list.extend(input_text_list)
            y_list.extend(output_answer_list)
            source_list.extend(source_text_list)

        elif label == 0:
            continue

    assert len(x_list) == len(y_list)
    assert len(x_list) == len(source_list)

    paragraph_list = []
    for i in range(len(x_list)):
        paragraph = f"{x_list[i].strip()} {y_list[i].rstrip()}."
        paragraph_list.append(paragraph)

    # Create a dictionary to store the data
    data_dict = {
        "input": x_list,
        "output": y_list,
        "paragraph": paragraph_list,
        "source": source_list
    }

    # Create a Hugging Face dataset
    kg_prob_dataset = Dataset.from_dict(data_dict)

    return kg_prob_dataset


if __name__ == "__main__":
    target_pos_tag = ["NN", "NNS", "NNP", "NNPS", "CD"]
    symbol_pos_tag = ["''", '--', ':', '(', ')', ',', '``', '$']
    end_acceptable_pos_tag = target_pos_tag + ['$']

    kg_prob_dataset = transform_dataset(target_pos_tag, symbol_pos_tag, end_acceptable_pos_tag)
    kg_prob_dataset.push_to_hub("oneonlee/wiki-kg-prob")