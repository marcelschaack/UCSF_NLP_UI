import sys
# sys.path.append('/home/TheLumino/UCSF_NLP_UI/flaskr')

import time
import numpy as np
import mpld3
from mpld3 import plugins
import pandas as pd

from flaskr.db import get_db
from flask import (Blueprint, flash, redirect, render_template, request, url_for, Flask, send_from_directory)
from flask_table import Table, Col, LinkCol
from werkzeug.utils import secure_filename
import pickle
import csv
import os
from zipfile import ZipFile


import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from scipy import spatial
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import stanfordnlp
import re

# Create Flask application here
app = Flask(__name__)

# Load blueprint for flask
bp = Blueprint('note', __name__)

STOP_WORDS = set(ENGLISH_STOP_WORDS)
STOP_WORDS.remove('no')
STOP_WORDS.remove('not')

# Loading lemmatizer for BLEU-socre and Stanford NLP Pipeline
lemmatizer = WordNetLemmatizer()
nlp = stanfordnlp.Pipeline(processors="tokenize,mwt,pos,depparse")
embedding = ELMoEmbeddings('pubmed')

# Define upload folder location
# UPLOAD_FOLDER = '/home/TheLumino/UCSF_NLP_UI/Uploads'
UPLOAD_FOLDER = 'Uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load word2vec and random forest model
# currentmodel = Word2Vec.load('/home/TheLumino/UCSF_NLP_UI/flaskr/cosine_model/cosine_similarity_metric')
# rf_model = pickle.load(open('/home/TheLumino/UCSF_NLP_UI/flaskr/cosine_model/random_forest_confidence_score.sav', 'rb'))
currentmodel = Word2Vec.load('flaskr/cosine_model/bestmodel')
rf_model = pickle.load(open('flaskr/cosine_model/random_forest_confidence_score.sav', 'rb'))

# set UMLS database links for quick access from UI
rx_url = 'https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm='
snomed_url = 'https://browser.ihtsdotools.org/?perspective=full&conceptId1='

# load abbreviation dictionary to replace common abbreviations
# abbreviation_path = '/home/TheLumino/UCSF_NLP_UI/flaskr/cosine_model/concept_abbreviation.txt'
abbreviation_path = 'flaskr/cosine_model/concept_abbreviation.txt'
with open(abbreviation_path, 'r') as dictionary:
    abbreviations_dic = {}
    for line in dictionary:
        if len(line)>1:
            line=line.split('-', 1)
            abbreviations_dic[line[0].rstrip().lower()] = line[1].lstrip().lower()

# important patterns to remove from health records string
digit_pattern = '\d+'
date_pattern = r"[\d]{1,2}/[\d]{1,2}/[\d]{2,4}"
time_pattern = r"[\d]{1,2}:[\d]{1,2}:[\d]{1,2}"
DIGIT_SIGN = '/DIGIT'
DATE_SIGN = 'DATE'
TIME_SIGN = 'TIME'
apas_error = '&apos;'

# create Plugin subclass to create clickable graph
class PluginBase(object):
    def get_dict(self):
        return self.dict_

    def javascript(self):
        if hasattr(self, "JAVASCRIPT"):
            if hasattr(self, "js_args_"):
                return self.JAVASCRIPT.render(self.js_args_)
            else:
                return self.JAVASCRIPT
        else:
            return ""

    def css(self):
        if hasattr(self, "css_"):
            return self.css_
        else:
            return ""

class PointClickableHTMLTooltip(PluginBase):
    JAVASCRIPT="""
    mpld3.register_plugin("clickablehtmltooltip", PointClickableHTMLTooltip);
    PointClickableHTMLTooltip.prototype = Object.create(mpld3.Plugin.prototype);
    PointClickableHTMLTooltip.prototype.constructor = PointClickableHTMLTooltip;
    PointClickableHTMLTooltip.prototype.requiredProps = ["id"];
    PointClickableHTMLTooltip.prototype.defaultProps = {labels:null,
                                                 targets:null,
                                                 hoffset:0,
                                                 voffset:10};
    function PointClickableHTMLTooltip(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    PointClickableHTMLTooltip.prototype.draw = function(){
       var obj = mpld3.get_element(this.props.id);
       var labels = this.props.labels;
       var targets = this.props.targets;

       var tooltip = d3.select("body").append("div")
                    .attr("class", "mpld3-tooltip")
                    .style("position", "absolute")
                    .style("z-index", "10")
                    .style("visibility", "hidden");

       obj.elements()
           .on("mouseover", function(d, i){
                  if ($(obj.elements()[0][0]).css( "fill-opacity" ) > 0 || $(obj.elements()[0][0]).css( "stroke-opacity" ) > 0) {
                              tooltip.html(labels[i])
                                     .style("visibility", "visible");
                              } })

           .on("mousedown", function(d, i){
                              window.open().document.write(targets[i]);
                               })
           .on("mousemove", function(d, i){
                  tooltip
                    .style("top", d3.event.pageY + this.props.voffset + "px")
                    .style("left",d3.event.pageX + this.props.hoffset + "px");
                 }.bind(this))
           .on("mouseout",  function(d, i){
                           tooltip.style("visibility", "hidden");});
    };
    """

    def __init__(self, points, labels=None, targets=None,
                 hoffset=2, voffset=-6, css=None):
        self.points = points
        self.labels = labels
        self.targets = targets
        self.voffset = voffset
        self.hoffset = hoffset
        self.css_ = css or ""
        if targets is not None:
            styled_targets = list(map(lambda x: self.css_ + x, targets))
        else:
            styled_targets = None

        if isinstance(points, matplotlib.lines.Line2D):
            suffix = "pts"
        else:
            suffix = None
        self.dict_ = {"type": "clickablehtmltooltip",
                      "id": mpld3.utils.get_id(points, suffix),
                      "labels": labels,
                      "targets": styled_targets,
                      "hoffset": hoffset,
                      "voffset": voffset}


class ConceptsToDisplay:
    def __init__(self, concept_name, highlight, sentence, beginning, highlight_beginning, other_concepts,
                 txt_file):
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


def combined_cosine_similarity_flair(concept, in_txt):
    if in_txt.lower() in abbreviations_dic:
        in_txt = abbreviations_dic[in_txt.lower()]

    concept = concept.split(' ')
    in_txt = in_txt.split(' ')
    concept = [stripword(i.lower()) for i in concept]
    in_txt = [stripword(i.lower()) for i in in_txt]

    concept = '_'.join(concept)
    range_words = '_'.join(in_txt)
    concept_words = Sentence('the disease and illness ' + concept)
    embedding.embed(concept_words)
    range_words = Sentence(range_words)
    embedding.embed(range_words)

    cosine_sim = (1 - spatial.distance.cosine(concept_words[-1].embedding.cpu(), range_words[0].embedding.cpu()))

    return cosine_sim


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
    labels2 = list()
    for i in range(len(xs)):
        label = df.iloc[[i], :].T
        label.columns = [str(concepts[i])]
        label2 = '<a class="action" href=' + str(url_for('note.sentence', filename=filename, conc=concepts[i])) + ">View concept</a>"
        # .to_html() is unicode; so make leading 'u' go away with str()
        labels.append(str(label.to_html()))
        labels2.append(str(label2))

    points = ax.plot(xs, ys, 'o', color='b',
                     mec='k', ms=8, mew=1, alpha=.6)

    ax.set_ylabel('Confidence Score')
    ax.set_title('Confidence Score Distribution', size=20)

    tooltip = PointClickableHTMLTooltip(points[0], labels=labels, targets=labels2, voffset=10, hoffset=10)
    plugins.connect(fig, tooltip)
    plugins.connect(fig, plugins.MouseXPosition())
    plugins.connect(fig, plugins.PointHTMLTooltip(points[0], labels=labels, voffset=10, hoffset=10))

    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)
    # xs = range(len(confidence_scores))
    # ys = np.sort(confidence_scores)
    # axis.plot(xs, ys, 'ro')
    return fig


def remove_duplicates(items):
    for k in range(0, len(items) - 1):
        for j in range(k + 1, len(items)):
            while k < len(items) and j < len(items) and items[k].obtain_name() == items[j].obtain_name():
                items[k].increase_number()
                items[k].average_conf_score(items[j].get_confidence())
                items.remove(items[j])

    return items


def analyse_deps(doc, index):
    dependencies = list()
    sent_length = 0
    for i in index:
        if i is None:
            continue
        for word in doc.sentences[i].words:
            if word.governor == 0:
                dependencies.append(['root', word.text])
            else:
                dependencies.append([word.governor-1+sent_length, word.text])
        sent_length = len(doc.sentences[i].words)
    return dependencies


def create_sent(i, token_sent):
    if 0 < i < len(token_sent) - 1:
        sentence = ' '.join([word for word in word_tokenize(token_sent[i - 1])] +
                            [word for word in word_tokenize(token_sent[i])] +
                            [word for word in word_tokenize(token_sent[i + 1])])
        j = [i - 1, i, i + 1]
    elif i == 0:
        sentence = ' '.join([word for word in word_tokenize(token_sent[i])] +
                            [word for word in word_tokenize(token_sent[i + 1])])
        j = [i, i + 1]
    else:
        sentence = ' '.join([word for word in word_tokenize(token_sent[i - 1])] +
                            [word for word in word_tokenize(token_sent[i])])
        j = [i - 1, i]
    return sentence, j


def create_sent_stanford(i, doc):
    if 0 < i < len(doc.sentences) - 1:
        sentence = ' '.join([word.text for word in doc.sentences[i - 1].words] +
                            [word.text for word in doc.sentences[i].words] +
                            [word.text for word in doc.sentences[i + 1].words])
        j = [i - 1, i, i + 1]
    elif i == 0:
        sentence = ' '.join([word.text for word in doc.sentences[i].words] +
                            [word.text for word in doc.sentences[i + 1].words])
        j = [i, i + 1]
    else:
        sentence = ' '.join([word.text for word in doc.sentences[i - 1].words] +
                            [word.text for word in doc.sentences[i].words])
        j = [i - 1, i]
    return sentence, j


def get_sentence(text, range_txt, conc_beginning, conc_end):
    txt_length = 0
    token_sent = sent_tokenize(text)
    for i, sent in enumerate(token_sent):
        for word in word_tokenize(sent):
            if conc_beginning - 20 <= txt_length:
                sentence, j = create_sent(i, token_sent)

                if re.sub("\s*", "", range_txt.lower()) not in re.sub("\s*", "", sentence.lower()):
                    continue
                else:
                    return j, sentence

            elif conc_end <= txt_length:
                sentence, j = create_sent(i, token_sent)
                return j, sentence
            txt_length += len(word) + 1


def get_sentence_stanford(doc, range_txt, conc_beginning, conc_end):
    txt_length = 0
    for i, sent in enumerate(doc.sentences):
        for word in sent.words:
            if conc_beginning - 20 <= txt_length:
                sentence, j = create_sent_stanford(i, doc)
                if re.sub("\s*", "", range_txt.lower()) not in re.sub("\s*", "", sentence.lower()):
                    continue
                else:
                    return j, sentence
            elif conc_end <= txt_length:
                sentence, j = create_sent_stanford(i, doc)
                return j, sentence
            txt_length += len(word.text) + 1


def get_word_dependents(text, range_txt, conc_beginning, conc_end):
    doc = nlp(text)
    index, sentence = get_sentence_stanford(doc, range_txt, conc_beginning, conc_end)
    depend = analyse_deps(doc, index)
    words = [word.text for i in index for word in doc.sentences[i].words if i is not None]
    pos = [word.pos for i in index for word in doc.sentences[i].words if i is not None]
    range_words = range_txt.replace('-', ' - ').split(' ')
    for i in range(len(words)):
        flag = 0
        for j in range(len(range_words)):
            if range_words[j].lower() == words[i + j].lower():
                flag += 1
                if flag >= len(range_words):
                    dep_indices = [k for k, value in enumerate(depend) if value[0] in range(i, i + j + 1)]
                    dep_indices2 = [depend[k][0] for k in range(i, i + j + 1) if depend[k][0] != 'root']
                    dep_indices += dep_indices2
                    return [[words[l] for l in dep_indices if words[l] not in range_words],
                            [pos[l] for l in dep_indices if words[l] not in range_words]]


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
            # txt.save(os.path.join('/home/TheLumino/UCSF_NLP_UI/Uploads', txtname))
            txt.save(os.path.join('Uploads', txtname))
            if 'zip' in txtname:
                # with ZipFile(os.path.join('UCSF_NLP_UI/Uploads', txtname), 'r') as zipObj:
                    # zipObj.extractall('UCSF_NLP_UI/Uploads')
                with ZipFile(os.path.join('Uploads', txtname), 'r') as zipObj:
                    zipObj.extractall('Uploads')

        if file.filename == '' or txt.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if not allowed_file(file.filename) or not allowed_file(txt.filename):
            flash('Please upload .csv, .txt or .zip files only')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        # file.save(os.path.join('UCSF_NLP_UI/Uploads', filename))
        file.save(os.path.join('Uploads', filename))

        return redirect(url_for('note.open_file', filename=filename))

    return render_template('note/index.html')


@bp.route('/<filename>', methods=['GET', 'POST'])
def open_file(filename):
    sort = request.args.get('sort', 'concept_amount')
    reverse = (request.args.get('direction', 'desc') == 'desc')


    # with open(os.path.join('UCSF_NLP_UI/Uploads', filename)) as csvfile:
    with open(os.path.join('Uploads', filename)) as csvfile:
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
            elmo = combined_cosine_similarity_flair(concept_calc.lower(), range_calc.lower())

            similarity_list.append([levenstein_sim, jaccard_sim, cosine_similarity, elmo])


            concepts.append(concept_txt)
            txt_files.append(range_txt)

        confidence_scores = rf_model.predict_proba(np.array(similarity_list))[:,1]

        fig = create_figure(confidence_scores, concepts, filename)

        # if os.path.exists('/home/TheLumino/UCSF_NLP_UI/flaskr/static/plot.html'):
        #     os.remove('/home/TheLumino/UCSF_NLP_UI/flaskr/static/plot.html')
        # if os.path.exists('flaskr/static/plot.html'):
            # os.remove('flaskr/static/plot.html')

        # mpld3.save_html(fig, '/home/TheLumino/UCSF_NLP_UI/flaskr/static/plot.html')
        fig_html = mpld3.fig_to_html(fig)
        #fig.savefig('/home/TheLumino/UCSF_NLP_UI/flaskr/static/plot.png')

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
                    direction = 'desc'
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

            def increase_number(self):
                self.concept_amount += 1

            def average_conf_score(self, confidence_j):
                self.conc_confidence = (self.conc_confidence * self.concept_amount + confidence_j)/(self.concept_amount + 1)

            def get_confidence(self):
                return self.conc_confidence

        items = list()
        for i in range(0, len(concepts)):
            # attrs = url_for('note.txt_files', conc=concepts[i], filename=filename))
            # items.append(Item(concepts[i], txt_files[i], element('a', attrs=attrs, content='H')))
            items.append(Item(concepts[i], round(confidence_scores[i], 4)))

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

    return render_template('note/concepts.html', table=table, filename=filename,
                           mean=round(float(np.mean(confidence_scores)), 3), fig=fig_html)


# @app.route('/plot-<confidence_scores>.png')


@bp.route('/<filename>/<conc>', methods=('GET', 'POST'))
def sentence(filename, conc):
    # with open(os.path.join('UCSF_NLP_UI/Uploads', filename)) as csvfile:
    with open(os.path.join('Uploads', filename)) as csvfile:
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


    concepts_display = list()

    for i, txt in enumerate(txt_files):
        # with open(os.path.join('UCSF_NLP_UI/Uploads', txt), 'r') as file:
        with open(os.path.join('Uploads', txt), 'r') as file:
            whole_txt = file.read()
            whole_txt = re.sub(apas_error, "'", whole_txt)

        index, sentence = get_sentence(whole_txt, range_txt[i], conc_beginning[i], conc_end[i])
        position = sentence.index(range_txt[i])
        end = conc_beginning[i] - position + len(sentence)
        beginning = conc_beginning[i] - position -1
        concept_name = [word.lower() for word in range_txt[i].split(' ')]

        highlight = list()
        highlight_beginning = list()
        other_concepts = list()

        # with open(os.path.join('UCSF_NLP_UI/Uploads', filename)) as csvfile:
        with open(os.path.join('Uploads', filename)) as csvfile:
            csvdata = csv.reader(csvfile)
            next(csvdata, None)
            for row in csvdata:
                if int(row[12]) >= beginning and int(row[11]) <= end and row[1] == txt:
                    highlight.append(row[13])
                    highlight_beginning.append(row[12])
                    other_concepts.append(row[3])
        sentence = sentence.split(' ')
        concepts_display.append(ConceptsToDisplay(concept_name, highlight, sentence, beginning, highlight_beginning,
                                                  other_concepts, txt))

    return render_template('note/sentence.html', concept_chosen=conc, filename=filename, concepts_display=concepts_display)


@bp.route('/<filename>/<concept_chosen>/<i_txt_file>/<beginning>', methods=('GET', 'POST'))
def display_concept(i_txt_file, concept_chosen, beginning, filename):
    # with open(os.path.join('UCSF_NLP_UI/Uploads', filename)) as csvfile:
    with open(os.path.join('Uploads', filename)) as csvfile:
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
                end = int(row[11])
                break


    if "SNOMEDCT" in database:
        url = snomed_url + concept_id
    else:
        url = rx_url + concept_id

    with open(os.path.join('Uploads', i_txt_file), 'r') as file:
        whole_txt = file.read()
        whole_txt = re.sub(apas_error, "'", whole_txt)
    dependents, poss = get_word_dependents(whole_txt, range_txt_display, int(beginning), end)

    if dependents is not None:
        has_deps = 1
        dependencies = list()
        removal = list()
        for i, (wrd, pos) in enumerate(zip(dependents, poss)):
            if pos in ['VBD', 'VB', 'VBG', 'VBP', 'VBN', 'EX', 'CC', 'HYPH', '.', ',', 'LS', 'PRP', 'PRP$', 'WDT',
                       'WP', 'WP$', 'WRB', '-RRB-']:
                removal.append([wrd, pos])
            elif wrd.lower() in STOP_WORDS:
                removal.append([wrd, pos])
        for j in removal:
            dependents.remove(j[0])
            poss.remove(j[1])
        for dep, pos in zip(dependents, poss):
            dependencies.append([dep, pos])
    else:
        has_deps = 0

    if concept_to_display.lower() in abbreviations_dic:
        concept_calc = abbreviations_dic[concept_to_display.lower()].lower()
    else:
        concept_calc = concept_to_display.lower()

    if range_txt_display.lower() in abbreviations_dic:
        range_calc = abbreviations_dic[range_txt_display.lower()].lower()
    else:
        range_calc = range_txt_display.lower()

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
    levenstein_sim = (bigger - levensteindistance(concept_calc, range_calc))/bigger
    cosine_similarity_value = combined_cosine_similarity(concept_calc, range_calc)
    cosine_flair = combined_cosine_similarity_flair(concept_calc, range_calc)
    jaccard_sim = 1 - nltk.jaccard_distance(set(concept_calc), set(range_calc))
    confidence_score = rf_model.predict_proba(np.array([levenstein_sim, jaccard_sim, cosine_similarity_value, cosine_flair]).reshape(1, -1))[0,1]

    if request.method == 'POST':
        if has_deps == 1:
            for dep in dependencies:
                try:
                    deps_feedback = request.form[str(dep[0])]
                    db = get_db()
                    db.execute(
                        'INSERT INTO dependencies (word, pos, feedback)'
                        ' VALUES (?,?,?)',
                        (str(dep[0]), str(dep[1]), int(deps_feedback))
                    )
                    db.commit()
                except:
                    pass


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
                 hof_answ, location_answ, negation_answ, bleu_score, levenstein_sim, cosine_sim, jaccard_sim, elmo)'
                ' VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (str(range_txt_display), str(concept_chosen), int(negated), int(hof), str(location), str2int(correct_answ),
                 str2int(hof_answ), str2int(location_answ), str2int(negation_answ), float(blue_score),
                 float(levenstein_sim), float(cosine_similarity_value), float(jaccard_sim), float(cosine_flair))
            )
            db.commit()
            return redirect(url_for('note.sentence', filename=filename, conc=concept_chosen))

    return render_template('note/concept_display.html', hof=hof, negated=negated, range_txt=range_txt_display,
                           location=location, concept=concept_to_display, filename=filename, blue=round(blue_score,3),
                           levenstein=round(levenstein_sim,3), cosine=round(cosine_similarity_value,3),
                           flair=round(cosine_flair,3), jaccard=round(jaccard_sim,3),
                           confidence=round(confidence_score,3), i_txt_file=i_txt_file, concept_chosen=concept_chosen, url=url,
                           has_deps=has_deps, dependencies=dependencies)
