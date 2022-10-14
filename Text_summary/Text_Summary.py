# Summarize the text using Tf-IDF Algorithm.
'''
Term Frequency * Inverse Document Frequency
In a simple language, TF-IDF can be defined as follows:
A High weight in TF-IDF is reached by a high term frequency(in the given document) 
and a low document frequency of the term in the whole collection of documents.
TF-IDF algorithm is made of 2 algorithms multiplied together.
Term Frequency
Term frequency (TF) is how often a word appears in a document, divided by how many words there are.
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
Inverse document frequency
Term frequency is how common a word is, inverse document frequency (IDF) is how unique or rare a word is.
IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
Example,
Consider a document containing 100 words wherein the word apple appears 5 times. 
The term frequency (i.e., TF) for apple is then (5 / 100) = 0.05.
Now, assume we have 10 million documents and the word apple appears in one thousand of these. 
Then, the inverse document frequency (i.e., IDF) is calculated as log(10,000,000 / 1,000) = 4.
Thus, the TF-IDF weight is the product of these quantities: 0.05 * 4 = 0.20.
'''
import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

# to replace this code or text i need to create a function
#Tokenize the sentences
'''text = """
Those Who Are Resilient Stay In The Game Longer
“On the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow."-Friedrich Nietzsche
Challenges and setbacks are not meant to defeat you, but promote you. However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments. Have you experienced this before? To be honest, I don't have the answers. I can't tell you what the right course of action is; only you will know. However, it's important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people. To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, it's an opportunity to improve and find new ways to overcome their obstacles. Same failure, yet different responses. Who is right and who is wrong? Neither. Each person has a different mindset that decides their outcome. Those who are resilient stay in the game longer and draw on their inner means to succeed.
I've coached many clients who gave up after many years toiling away at their respective goal or dream. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It was the 19th Century's minister Henry Ward Beecher who once said: “One's best success comes after their greatest disappointments.” No one knows what the future holds, so your only guide is whether you can endure repeated defeats and disappointments and still pursue your dream. Consider the advice from the American academic and psychologist Angela Duckworth who writes in Grit: The Power of Passion and Perseverance: “Many of us, it seems, quit what we start far too early and far too often. Even more than the effort a gritty person puts in on a single day, what matters is that they wake up the next day, and the next, ready to get on that treadmill and keep going.”
I know one thing for certain: don't settle for less than what you're capable of, but strive for something bigger. Some of you reading this might identify with this message because it resonates with you on a deeper level. For others, at the end of their tether the message might be nothing more than a trivial pep talk. What I wish to convey irrespective of where you are in your journey is: NEVER settle for less. If you settle for less, you will receive less than you deserve and convince yourself you are justified to receive it.
“Two people on a precipice over Yosemite Valley” by Nathan Shipps on Unsplash
Develop A Powerful Vision Of What You Want
“Your problem is to bridge the gap which exists between where you are now and the goal you intend to reach."-Earl Nightingale
I recall a passage my father often used growing up in 1990s: “Don't tell me your problems unless you've spent weeks trying to solve them yourself.” That advice has echoed in my mind for decades and became my motivator. Don't leave it to other people or outside circumstances to motivate you because you will be let down every time. It must come from within you. Gnaw away at your problems until you solve them or find a solution. Problems are not stop signs, they are advising you that more work is required to overcome them. Most times, problems help you gain a skill or develop the resources to succeed later. So embrace your challenges and develop the grit to push past them instead of retreat in resignation. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? Are you willing to play bigger even if it means repeated failures and setbacks? You should ask yourself these questions to decide whether you're willing to put yourself on the line or settle for less. And that's fine if you're content to receive less, as long as you're not regretful later.
If you have not achieved the success you deserve and are considering giving up, will you regret it in a few years or decades from now? Only you can answer that, but you should carve out time to discover your motivation for pursuing your goals. It's a fact, if you don't know what you want you'll get what life hands you and it may not be in your best interest, affirms author Larry Weidel: “Winners know that if you don't figure out what you want, you'll get whatever life hands you.” The key is to develop a powerful vision of what you want and hold that image in your mind. Nurture it daily and give it life by taking purposeful action towards it.
Vision + desire + dedication + patience + daily action leads to astonishing success. Are you willing to commit to this way of life or jump ship at the first sign of failure? I'm amused when I read questions written by millennials on Quora who ask how they can become rich and famous or the next Elon Musk. Success is a fickle and long game with highs and lows. Similarly, there are no assurances even if you're an overnight sensation, to sustain it for long, particularly if you don't have the mental and emotional means to endure it. This means you must rely on the one true constant in your favour: your personal development. The more you grow, the more you gain in terms of financial resources, status, success-simple. If you leave it to outside conditions to dictate your circumstances, you are rolling the dice on your future.
So become intentional on what you want out of life. Commit to it. Nurture your dreams. Focus on your development and if you want to give up, know what's involved before you take the plunge. Because I assure you, someone out there right now is working harder than you, reading more books, sleeping less and sacrificing all they have to realise their dreams and it may contest with yours. Don't leave your dreams to chance.
"""
'''
def text_sum_alg(text):
    sentences = sent_tokenize(text) # NLTK function
    total_documents = len(sentences)
    def _create_frequency_matrix(sentences):
        frequency_matrix = {}
        stopWords = set(stopwords.words("english"))
        ps = PorterStemmer()

        for sent in sentences:
            freq_table = {}
            words = word_tokenize(sent)
            for word in words:
                word = word.lower()
                word = ps.stem(word)
                if word in stopWords:
                    continue

                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1

            frequency_matrix[sent[:15]] = freq_table

        return frequency_matrix
    '''
    Calculate TermFrequency and generate a matrix
    We'll find the TermFrequency for each word in a paragraph.
    Now, remember the definition of TF,
    TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
    Here, the document is a paragraph, the term is a word in a paragraph.
    '''
    def _create_tf_matrix(freq_matrix):
        tf_matrix = {}

        for sent, f_table in freq_matrix.items():
            tf_table = {}

            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence

            tf_matrix[sent] = tf_table

        return tf_matrix
    '''
    Creating a table for documents per words
    This again a simple table which helps in calculating IDF matrix.
    we calculate, “how many sentences contain a word”, Let's call it Documents per words matrix.
    '''
    def _create_documents_per_words(freq_matrix):
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        return word_per_doc_table
    '''
    Calculate IDF and generate a matrix
    We'll find the IDF for each word in a paragraph.
    Now, remember the definition of IDF,
    IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
    Here, the document is a paragraph, the term is a word in a paragraph.
    '''
    def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}

        for sent, f_table in freq_matrix.items():
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

            idf_matrix[sent] = idf_table

        return idf_matrix
    '''
    6.Calculate TF-IDF and generate a matrix
    Now we have both the matrix and the next step is very easy.
    TF-IDF algorithm is made of 2 algorithms multiplied together.
    In simple terms, we are multiplying the values from both the matrix and generating new matrix.
    '''
    def _create_tf_idf_matrix(tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                        f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        return tf_idf_matrix
    '''
    7. Score the sentences
    Scoring a sentence is differs with different algorithms. 
    Here, we are using Tf-IDF score of words in a sentence to give weight to the paragraph.
    '''
    def _score_sentences(tf_idf_matrix) -> dict:
        """
        score a sentence by its word's TF
        Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """

        sentenceValue = {}

        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0

            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score

            sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

        return sentenceValue
    '''
    8. Find the threshold
    Similar to any summarization algorithms, there can be different ways to calculate a threshold value. 
    We're calculating the average sentence score.
    '''
    def _find_average_score(sentenceValue) -> int:
        """
        Find the average score from the sentence value dictionary
        :rtype: int
        """
        sumValues = 0
        for entry in sentenceValue:
            sumValues += sentenceValue[entry]

        # Average value of a sentence from original summary_text
        average = (sumValues / len(sentenceValue))

        return average
    '''
    9. Generate the summary
    Algorithm: Select a sentence for a summarization if the sentence score is more than the average score.
    '''
    def _generate_summary(sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''

        for sentence in sentences:
            if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
                summary += " " + sentence
                sentence_count += 1

        return summary
    '''
    Everything in one place: Algorithms Assemble
    For the threshold, we’ve used 1.3x of the average score. You can play with such variables to generate the summary as you like.
    '''
    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 1 Sentence Tokenize
    '''sentences = sent_tokenize(text)
    total_documents = len(sentences)
    #print(sentences)'''

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)
    #print(freq_matrix)

    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    #print(tf_matrix)

    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)

    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)

    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)

    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix)
    #print(sentence_scores)

    # 8 Find the threshold
    threshold = _find_average_score(sentence_scores)
    #print(threshold)

    # 9 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
    return summary