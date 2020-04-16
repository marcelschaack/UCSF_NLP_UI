import sys
sys.path.append('/home/TheLumino/UCSF_NLP_UI/flaskr')

import csv
import os
from zipfile import ZipFile
import time
import numpy as np
import mpld3
from mpld3 import plugins
import pandas as pd


import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk.stem import WordNetLemmatizer
from flask import (Blueprint, flash, redirect, render_template, request, url_for, Flask, send_from_directory)
from flask_table import Table, Col, LinkCol
from gensim.models import Word2Vec
from scipy import spatial
from werkzeug.utils import secure_filename
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from db import get_db

app = Flask(__name__)

bp = Blueprint('note', __name__)

lemmatizer = WordNetLemmatizer()

UPLOAD_FOLDER = '/home/TheLumino/UCSF_NLP_UI/Uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

currentmodel = Word2Vec.load('/home/TheLumino/UCSF_NLP_UI/flaskr/cosine_model/cosine_similarity_metric')
rf_model = pickle.load(open('/home/TheLumino/UCSF_NLP_UI/flaskr/cosine_model/random_forest_confidence_score.sav', 'rb'))

rx_url = 'https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm='
snomed_url = 'https://browser.ihtsdotools.org/?perspective=full&conceptId1='

abbreviation_path = '/home/TheLumino/UCSF_NLP_UI/flaskr/cosine_model/concept_abbreviation.txt'
with open(abbreviation_path, 'r') as dictionary:
    abbreviations_dic = {}
    for line in dictionary:
        if len(line)>1:
            line=line.split('-',1)
            abbreviations_dic[line[0].rstrip().lower()] = line[1].lstrip().lower()


def allowed_file(filename):
    allowed_extensions = {'csv', 'txt', 'zip'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def stripword(word):
    word = word.replace("[", "")
    word = word.replace("]", "")
    word = word.replace("'", "")
    word = word.replace("'", "")
    word = word.replace(" ", "")
    return word


def stripword2(word):
    word = word.replace("[", "")
    word = word.replace("]", "")
    word = word.replace("'", "")
    word = word.replace("'", "")
    word = word.replace(",", "")
    word = word.replace(".", "")
    word = word.replace("_", "")
    word = word.replace(":", "")
    word = word.replace("-", "")
    word = word.replace("*", "")
    word = word.replace("/", "")
    word = word.replace("(", "")
    word = word.replace(")", "")
    word = word.replace("´", "")
    word = word.replace("`", "")
    word = word.replace(";", "")
    return word


def levensteindistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def cosine_similarity(word1, word2):
    return 1 - spatial.distance.cosine(currentmodel.wv[word1], currentmodel.wv[word2])


def combined_cosine_similarity(concept, in_txt):
    concept_words = concept.split()
    concept_words = [stripword2(i) for i in concept_words]
    range_words = in_txt.split()
    range_words = [stripword2(i) for i in range_words]
    cosine_average = 0
    count = 0
    for m in range(0, len(concept_words)):
        try:
            word1 = '_'.join(concept_words[m:m + 3])
            word2 = '_'.join(range_words[m:m + 3])
            cosine_average += cosine_similarity(word1, word2)
            count += 1
        except:
            pass
    for m in range(0, len(concept_words)):
        try:
            word1 = '_'.join(concept_words[m:m + 2])
            word2 = '_'.join(range_words[m:m + 2])
            cosine_average += cosine_similarity(word1, word2)
            count += 1
        except:
            pass
    if count == 0:
        for word3 in concept_words:
            for word4 in range_words:
                try:
                    cosine_average += cosine_similarity(word3, word4)
                    count += 1
                except:
                    pass
    if count == 0:
        return 0
    else:
        return cosine_average / count


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def str2int(v):
    if v.lower() in ("yes", "true", "t", "1"):
        return int(1)
    elif v.lower() in ("no", "false", "f", "0"):
        return int(0)
    else:
        return None

def create_figure(confidence_scores, concepts, filename):
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)
    xs = range(len(confidence_scores))
    ys, concepts = zip(*sorted(zip(confidence_scores, concepts)))
    df = pd.DataFrame(index=xs)
    df['Score'] = ys

    labels = list()
    for i in range(len(xs)):
        label = df.iloc[[i], :].T
        label.columns = [str(concepts[i])]
        label2 = '<a class="action" href=' + str(url_for('note.sentence', filename=filename, conc=concepts[i])) + ">View concept</a>"
        # .to_html() is unicode; so make leading 'u' go away with str()
        labels.append(str(label.to_html()) + label2)

    points = ax.plot(xs, ys, 'o', color='b',
                     mec='k', ms=8, mew=1, alpha=.6)

    ax.set_ylabel('Confidence Score')
    ax.set_title('Confidence Score Distribution', size=20)

    tooltip = plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10)
    plugins.connect(fig, tooltip)

    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)
    # xs = range(len(confidence_scores))
    # ys = np.sort(confidence_scores)
    # axis.plot(xs, ys, 'ro')
    return fig


@bp.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@bp.route('/', methods=('GET', 'POST'))
def index():
    now = time.time()

    for f in os.listdir(UPLOAD_FOLDER):
        if os.stat(os.path.join(UPLOAD_FOLDER, f)).st_mtime < now - 3*3600:
            if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)):
                os.remove(os.path.join(UPLOAD_FOLDER, f))

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        txt = request.files['txt']
        if txt:
            txtname = secure_filename(txt.filename)
            txt.save(os.path.join('/home/TheLumino/UCSF_NLP_UI/Uploads', txtname))
            if 'zip' in txtname:
                with ZipFile(os.path.join('UCSF_NLP_UI/Uploads', txtname), 'r') as zipObj:
                    zipObj.extractall('UCSF_NLP_UI/Uploads')

        if file.filename == '' or txt.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if not allowed_file(file.filename) or not allowed_file(txt.filename):
            flash('Please upload .csv, .txt or .zip files only')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file.save(os.path.join('UCSF_NLP_UI/Uploads', filename))

        return redirect(url_for('note.open_file', filename=filename))

    return render_template('note/index.html')


@bp.route('/<filename>', methods=['GET', 'POST'])
def open_file(filename):
    sort = request.args.get('sort', 'concept_amount')
    reverse = (request.args.get('direction', 'desc') == 'desc')


    with open(os.path.join('UCSF_NLP_UI/Uploads', filename)) as csvfile:
        csvdata = csv.reader(csvfile)
        next(csvdata, None)

        concepts = list()
        txt_files = list()
        similarity_list = list()

        for row in csvdata:
            concept_txt = row[3]
            range_txt = row[13]

            if concept_txt.lower() in abbreviations_dic:
                concept_calc = abbreviations_dic[concept_txt.lower()]
            else:
                concept_calc = concept_txt

            if range_txt.lower() in abbreviations_dic:
                range_calc = abbreviations_dic[range_txt.lower()]
            else:
                range_calc = range_txt

            bigger = max(len(concept_calc), len(range_calc))
            levenstein_sim = (bigger - levensteindistance(concept_calc.lower(), range_calc.lower()))/bigger
            cosine_similarity = combined_cosine_similarity(concept_calc.lower(), range_calc.lower())
            jaccard_sim = 1 - nltk.jaccard_distance(set(concept_calc.lower()), set(range_calc.lower()))

            similarity_list.append([levenstein_sim, cosine_similarity, jaccard_sim])


            concepts.append(concept_txt)
            txt_files.append(range_txt)

        confidence_scores = rf_model.predict_proba(np.array(similarity_list))[:,1]

        fig = create_figure(confidence_scores, concepts, filename)

        if os.path.exists('/home/TheLumino/UCSF_NLP_UI/flaskr/static/plot.html'):
            os.remove('/home/TheLumino/UCSF_NLP_UI/flaskr/static/plot.html')


        mpld3.save_html(fig, '/home/TheLumino/UCSF_NLP_UI/flaskr/static/plot.html')
        #fig.savefig('/home/TheLumino/UCSF_NLP_UI/flaskr/static/plot.png')

        # class LangCol(Col):  # This lets me get a webaddress into the table
        # def td_format(self, content, attr):
        # return element('a', attrs=dict(href=content), content=attr)

        class ItemTable(Table):
            allow_sort = True
            # concept_name = Col('Concept Ontology')
            # note_file = Col('Clincial Note File')
            concept_amount = Col('Amount')
            conc_confidence = Col('Confidence Score')
            # link = LinkCol('')

            def sort_url(self, col_key, reverse=False, filename=filename):
                #filename = 'combined_diseases.csv'
                if reverse:
                    direction =  'desc'
                else:
                    direction = 'asc'
                return url_for('note.open_file', sort=col_key, direction=direction, filename=filename)

        # Get some objects
        class Item(object):
            def __init__(self, concept_name, conc_confidence):
                self.concept_name = concept_name
                self.concept_amount = 1
                self.conc_confidence = conc_confidence
                self.filename = filename

            def obtain_name(self):
                return self.concept_name

            def obtain_txt(self):
                return self.note_file[0]

            def obtain_txt_list(self):
                return self.note_file

            def increase_number(self):
                self.concept_amount += 1

            def add_txt(self, filename):
                self.note_file.append(filename)

            def average_conf_score(self, confidence_j):
                self.confidence = (self.conc_confidence * self.concept_amount + confidence_j)/(self.concept_amount + 1)

            def get_confidence(self):
                return self.conc_confidence

        items = list()
        for i in range(0, len(concepts)):
            # attrs = url_for('note.txt_files', conc=concepts[i], filename=filename))
            # items.append(Item(concepts[i], txt_files[i], element('a', attrs=attrs, content='H')))
            items.append(Item(concepts[i], round(confidence_scores[i],4)))


        def remove_duplicates(items):
            for k in range(0, len(items) - 1):
                for j in range(k + 1, len(items)):
                    while k < len(items) and j < len(items) and items[k].obtain_name() == items[j].obtain_name():
                        items[k].increase_number()
                        items[k].average_conf_score(items[j].get_confidence())
                        items.remove(items[j])

            return items

        items = remove_duplicates(items)

        items.sort(key=lambda x: getattr(x, sort), reverse=reverse)

        # Populate the table
        table = ItemTable(items, sort_by=sort, sort_reverse=reverse)
        for item in items:
            url = 'note.sentence'
            # conc=item.obtain_name(), filename=filename)
            table.add_column('concept_name', LinkCol(name='Concept Ontology', endpoint=url,
                                         url_kwargs=dict(conc='concept_name', filename='filename'),
                                         attr='concept_name'))
        table.border = True

    return render_template('note/concepts.html', table=table, filename=filename, confidence=confidence_scores, mean=round(np.mean(confidence_scores),3))


# @app.route('/plot-<confidence_scores>.png')


# @bp.route('/<filename>/hello/<conc>', methods=['GET', 'POST'])
# def txt_files(filename, conc):
#     with open(os.path.join('UCSF_NLP_UI/Uploads', filename)) as csvfile:
#         csvdata = csv.reader(csvfile)
#         next(csvdata, None)
#         concept_locations = []
#         for row in csvdata:
#             if row[3] == conc and row[1] not in concept_locations:
#                 concept_locations.append(stripword(row[1]))


#     return render_template('note/txt_files.html', txt=concept_locations, conc=conc, filename=filename)


@bp.route('/<filename>/<conc>', methods=('GET', 'POST'))
def sentence(filename, conc):
    with open(os.path.join('UCSF_NLP_UI/Uploads', filename)) as csvfile:
        csvdata = csv.reader(csvfile)
        next(csvdata, None)
        txt_files = list()
        conc_beginning = list()
        conc_end = list()
        range_txt = list()
        for row in csvdata:
            if row[3] == conc:
                txt_files.append(row[1])
                conc_beginning.append(int(row[12]))
                conc_end.append(int(row[11]))
                range_txt.append(row[13])

    class ConceptsToDisplay:
        def __init__(self,  concept_name, highlight, sentence, beginning, highlight_beginning, other_concepts, txt_file):
            self.concept_name = concept_name
            self.highlight = highlight
            self.other_concepts = other_concepts
            self.sentence = sentence
            self.beginning = beginning
            self.highlight_beginning = highlight_beginning
            self.highlight_broken = list()
            self.txt_file = txt_file

            for high in self.highlight:
                words = high.split(' ')
                for one_word in words:
                    self.highlight_broken.append(one_word.lower())

    #all_txt = list()
    concepts_display = list()
    for i,txt in enumerate(txt_files):
        with open(os.path.join('UCSF_NLP_UI/Uploads', txt), 'r') as file:
            whole_txt = list()
            for l in file:
                whole_txt.extend([i for j in l.split() for i in (j, ' ')][:-1])
                whole_txt.extend(['\n'])
            #all_txt.append(whole_txt)

    #for i in range(0, len(conc_beginning)):
        txt_length = 0
        end = conc_end[i] + 150
        beginning = conc_beginning[i] - 150
        concept_name = list()
        concept_names = range_txt[i].split(' ')
        for name in concept_names:
            concept_name.extend(name.lower())

        highlight = list()
        highlight_beginning = list()
        other_concepts = list()
        sentence = list()
        with open(os.path.join('UCSF_NLP_UI/Uploads', filename)) as csvfile:
            csvdata = csv.reader(csvfile)
            next(csvdata, None)
            for row in csvdata:
                if int(row[12]) >= beginning and int(row[11]) <= end and row[1] == txt:
                    highlight.append(row[13])
                    highlight_beginning.append(row[12])
                    other_concepts.append(row[3])

        for word in whole_txt:
            if beginning <= txt_length <= end:
                sentence.append(word)
            txt_length += len(word)

        concepts_display.append(ConceptsToDisplay(concept_name, highlight, sentence, beginning, highlight_beginning,
                                                  other_concepts, txt))

    return render_template('note/sentence.html', concept_chosen=conc, filename=filename, concepts_display=concepts_display)



@bp.route('/<filename>/<concept_chosen>/<i_txt_file>/<beginning>', methods=('GET', 'POST'))
def display_concept(i_txt_file, concept_chosen, beginning, filename):
    with open(os.path.join('UCSF_NLP_UI/Uploads', filename)) as csvfile:
        csvdata = csv.reader(csvfile)
        next(csvdata, None)
        for row in csvdata:
            if row[12] == beginning and row[1] == i_txt_file:
                hof = str2bool(row[6])
                negated = str2bool(row[10])
                location = row[8]
                concept_to_display = row[3]
                range_txt_display = row[13]
                concept_id = row[14]
                database = row[15]

                if "SNOMEDCT" in database:
                    url = snomed_url + concept_id
                else:
                    url = rx_url + concept_id

                if concept_to_display.lower() in abbreviations_dic:
                    concept_calc = abbreviations_dic[concept_to_display.lower()]
                else:
                    concept_calc = concept_to_display

                if range_txt_display.lower() in abbreviations_dic:
                    range_calc = abbreviations_dic[range_txt_display.lower()]
                else:
                    range_calc = range_txt_display

                if range_calc.lower() == concept_calc.lower():
                    concept_tokenized = nltk.word_tokenize(concept_calc.lower())
                    range_tokenized = nltk.word_tokenize(range_calc.lower())
                    concept_lemmatized = [lemmatizer.lemmatize(n).lower() for n in concept_tokenized]
                    range_lemmatized = [lemmatizer.lemmatize(n).lower() for n in range_tokenized]
                    reference_list = [concept_lemmatized]
                    blue_score = sentence_bleu(reference_list, range_lemmatized, weights=(1, 0, 0, 0))
                else:
                    blue_score = single_meteor_score(concept_calc, range_calc)

                bigger = max(len(concept_calc), len(range_calc))
                levenstein_sim = (bigger - levensteindistance(concept_calc.lower(), range_calc.lower()))/bigger
                cosine_similarity_value = combined_cosine_similarity(concept_calc.lower(), range_calc.lower())
                jaccard_sim = 1 - nltk.jaccard_distance(set(concept_calc.lower()), set(range_calc.lower()))
                confidence_score = rf_model.predict_proba(np.array([levenstein_sim, cosine_similarity_value, jaccard_sim]).reshape(1, -1))[0,1]

    if request.method == 'POST':

        if not request.form['correct'] or not request.form['location'] or not request.form['hof'] \
                or not request.form['negation']:
            flash('Please insert yes or no.')
            return redirect(request.url)
        else:
            correct_answ = request.form['correct']
            location_answ = request.form['location']
            hof_answ = request.form['hof']
            negation_answ = request.form['negation']
            db = get_db()
            db.execute(
                'INSERT INTO feedback (range_txt, concept, negation, hof, location, correct_answ,\
                 hof_answ, location_answ, negation_answ, bleu_score, levenstein_sim, cosine_sim, jaccard_sim)'
                ' VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (str(range_txt_display), str(concept_chosen), int(negated), int(hof), str(location), str2int(correct_answ),
                 str2int(hof_answ), str2int(location_answ), str2int(negation_answ), float(blue_score),
                 float(levenstein_sim), float(cosine_similarity_value), float(jaccard_sim))
            )
            db.commit()
            return redirect(url_for('note.sentence', filename=filename, conc=concept_chosen))

    return render_template('note/concept_display.html', hof=hof, negated=negated, range_txt=range_txt_display,
                           location=location, concept=concept_to_display, filename=filename, blue=round(blue_score,3),
                           levenstein=round(levenstein_sim,3), cosine=round(cosine_similarity_value,3),
                           jaccard=round(jaccard_sim,3), confidence=round(confidence_score,3), i_txt_file=i_txt_file, concept_chosen=concept_chosen, url=url)
