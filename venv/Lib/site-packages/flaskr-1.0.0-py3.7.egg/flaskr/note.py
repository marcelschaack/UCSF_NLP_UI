import csv
import os
from zipfile import ZipFile

import nltk
from flask import (Blueprint, flash, redirect, render_template, request, url_for, Flask)
from flask_table import Table, Col, LinkCol
from gensim.models import Word2Vec
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

from flaskr.db import get_db

app = Flask(__name__)

bp = Blueprint('note', __name__)

UPLOAD_FOLDER = 'Uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

currentmodel = Word2Vec.load("cosine_model/cosine_similarity_metric")


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


def stripwordlist(listing):
    for ii in range(len(listing)):
        listing[ii] = listing[ii].replace("[", "")
        listing[ii] = listing[ii].replace("]", "")
        listing[ii] = listing[ii].replace("'", "")
        listing[ii] = listing[ii].replace(",", "")
        listing[ii] = listing[ii].replace('"', "")
        listing[ii] = listing[ii].replace(".", "")
        listing[ii] = listing[ii].replace("-", "")
        listing[ii] = listing[ii].replace("_", "")
        listing[ii] = listing[ii].replace(":", "")
        listing[ii] = listing[ii].replace(";", "")
        listing[ii] = listing[ii].replace("*", "")
        listing[ii] = listing[ii].replace("(", "")
        listing[ii] = listing[ii].replace(")", "")
        listing[ii] = listing[ii].replace("´", "")
        listing[ii] = listing[ii].replace("`", "")
        listing[ii].strip()
    return listing


def cosine_similarity(word1, word2):
    return 1 - spatial.distance.cosine(currentmodel.wv[word1], currentmodel.wv[word2])


def combined_cosine_similarity(concept, in_txt):
    concept_words = concept.split()
    concept_words = stripwordlist(concept_words)
    range_words = in_txt.split()
    range_words = stripwordlist(range_words)
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
    if count == 0:
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



@bp.route('/', methods=('GET', 'POST'))
def index():
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
            txt.save(os.path.join(app.config["UPLOAD_FOLDER"], txtname))
            if 'zip' in txtname:
                with ZipFile(os.path.join('Uploads', txtname), 'r') as zipObj:
                    zipObj.extractall('Uploads')
        if file.filename == '' or txt.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename) or not allowed_file(txt.filename):
            flash('Please upload .csv, .txt or .zip files only')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return redirect(url_for('note.open_file', filename=filename))
    return render_template('note/index.html')


@bp.route('/<filename>', methods=['GET', 'POST'])
def open_file(filename):
    with open(os.path.join('Uploads', filename)) as csvfile:
        csvdata = csv.reader(csvfile)
        next(csvdata, None)

        concepts = []
        txt_files = []

        for row in csvdata:
            concepts.append(row[3])
            txt_files.append(row[1])

        # class LangCol(Col):  # This lets me get a webaddress into the table
        # def td_format(self, content, attr):
        # return element('a', attrs=dict(href=content), content=attr)

        class ItemTable(Table):
            # concept_name = Col('Concept Ontology')
            # note_file = Col('Clincial Note File')
            concept_amount = Col('Amount')
            # link = LinkCol('')

        # Get some objects
        class Item(object):
            def __init__(self, concept_name, note_file, link):
                self.concept_name = concept_name
                self.note_file = []
                self.concept_amount = 1
                self.link = link
                self.note_file.append(note_file)
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

        items = []
        for i in range(0, len(concepts)):
            # attrs = url_for('note.txt_files', conc=concepts[i], filename=filename))
            # items.append(Item(concepts[i], txt_files[i], element('a', attrs=attrs, content='H')))
            items.append(Item(concepts[i], txt_files[i], 'View concept here'))

        def remove_duplicates(items):
            for k in range(0, len(items) - 1):
                for j in range(k + 1, len(items)):
                    while k < len(items) and j < len(items) and items[k].obtain_name() == items[j].obtain_name():
                        items[k].increase_number()
                        flag = True
                        for n in items[k].obtain_txt_list():
                            if n == items[j].obtain_txt():
                                flag = False
                        if flag:
                            items[k].add_txt(items[j].obtain_txt())
                        items.remove(items[j])

            return items

        items = remove_duplicates(items)

        items.sort(key=lambda x: x.concept_amount, reverse=True)

        # Populate the table
        table = ItemTable(items)
        for item in items:
            url = 'note.txt_files'
            # conc=item.obtain_name(), filename=filename)
            table.add_column('', LinkCol(name='Concept Ontology', endpoint=url,
                                         url_kwargs=dict(conc='concept_name', filename='filename'),
                                         attr='concept_name'))
        table.border = True

    return render_template('note/concepts.html', table=table, filename=filename)


@bp.route('/<filename>/<conc>', methods=['GET', 'POST'])
def txt_files(filename, conc):
    with open(os.path.join('Uploads', filename)) as csvfile:
        csvdata = csv.reader(csvfile)
        next(csvdata, None)
        concept_locations = []
        for row in csvdata:
            if row[3] == conc and row[1] not in concept_locations:
                concept_locations.append(stripword(row[1]))

    # if request.method == 'POST':
    # txt_chosen = request.form['txt_chosen']
    # error = 'No matching txt file found'
    # for n in concept_locations:
    # if txt_chosen in n:
    # error = None
    # txt_file = n
    # if error is not None:
    # flash(error)
    # return redirect(request.url)
    # return redirect(url_for('note.sentence', filename=filename, i_txt_file=txt_file, conc=conc))

    return render_template('note/txt_files.html', txt=concept_locations, conc=conc, filename=filename)


@bp.route('/<filename>/<conc>/<i_txt_file>/sentence', methods=('GET', 'POST'))
def sentence(i_txt_file, filename, conc):
    with open(os.path.join('Uploads', filename)) as csvfile:
        csvdata = csv.reader(csvfile)
        next(csvdata, None)
        conc_beginning = []
        conc_end = []
        range_txt = []
        for row in csvdata:
            if row[3] == conc and row[1] == i_txt_file:
                conc_beginning.append(int(row[12]))
                conc_end.append(int(row[11]))
                range_txt.append(row[13])

    with open(os.path.join('Uploads', i_txt_file), 'r') as file:
        whole_txt = []
        for l in file:
            whole_txt.extend([i for j in l.split() for i in (j, ' ')][:-1])
            whole_txt.extend(['\n'])

    class ConceptsToDisplay:
        def __init__(self,  concept_name, highlight, sentence, beginning, highlight_beginning, other_concepts):
            self.concept_name = concept_name
            self.highlight = highlight
            self.other_concepts = other_concepts
            self.sentence = sentence
            self.beginning = beginning
            self.highlight_beginning = highlight_beginning
            self.highlight_broken = []
            for high in self.highlight:
                words = high.split(' ')
                for one_word in words:
                    self.highlight_broken.append(one_word.lower())

    concepts_display = []
    for i in range(0, len(conc_beginning)):
        txt_length = 0
        end = conc_end[i] + 150
        beginning = conc_beginning[i] - 150
        concept_name = []
        concept_names = range_txt[i].split(' ')
        for name in concept_names:
            name = stripword2(name)
            concept_name.append(name.lower())
        highlight = []
        highlight_beginning = []
        other_concepts = []
        sentence = []
        with open(os.path.join('Uploads', filename)) as csvfile:
            csvdata = csv.reader(csvfile)
            next(csvdata, None)
            for row in csvdata:
                if int(row[12]) >= beginning and int(row[11]) <= end and row[1] == i_txt_file:
                    highlight.append(row[13])
                    highlight_beginning.append(row[12])
                    other_concepts.append(row[3])

        # highlight = range_txt[i]
        for word in whole_txt:
            if beginning <= txt_length <= end:
                sentence.append(word)
            txt_length += len(word)

        concepts_display.append(ConceptsToDisplay(concept_name, highlight, sentence, beginning, highlight_beginning,
                                                  other_concepts))

    # environment = jinja2.Environment('a')
    # environment.filters['b_any'] = b_any
    # note/sentence.html.render(b_any)
    return render_template('note/sentence.html', concept_chosen=conc, i_txt_file=i_txt_file, filename=filename,
                           concepts_display=concepts_display)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def str2int(v):
    if v.lower() in ("yes", "true", "t", "1"):
        return int(1)
    else:
        return int(0)


@bp.route('/<filename>/<concept_chosen>/<i_txt_file>/<beginning>', methods=('GET', 'POST'))
def display_concept(i_txt_file, concept_chosen, beginning, filename):
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
                cosine_similarity_value = combined_cosine_similarity(concept_to_display.lower(), range_txt_display.lower())
                jaccard_dist = nltk.jaccard_distance(set(concept_to_display), set(range_txt_display))

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
                'INSERT INTO feedback (concept, negation, hof, location, correct_answ,\
                 hof_answ, location_answ, negation_answ, cosine_sim, jaccard_dist)'
                ' VALUES (?,?,?,?,?,?,?,?,?,?)',
                (str(concept_chosen), int(negated), int(hof), str(location), str2int(correct_answ),
                 str2int(hof_answ), str2int(location_answ), str2int(negation_answ), float(cosine_similarity_value),
                 float(jaccard_dist))
            )
            db.commit()
            return redirect(url_for('note.sentence', i_txt_file=i_txt_file, filename=filename,
                                    conc=concept_chosen))

    return render_template('note/concept_display.html', hof=hof, negated=negated, range_txt=range_txt_display,
                           location=location, concept=concept_to_display, filename=filename,
                           cosine=cosine_similarity_value, jaccard=jaccard_dist, i_txt_file=i_txt_file,
                           concept_chosen=concept_chosen)
