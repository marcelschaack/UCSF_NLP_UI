import csv
import os

from flask import (Blueprint, flash, redirect, render_template, request, url_for, send_from_directory, Flask)
from werkzeug.utils import secure_filename
from flask_table import Table, Col
import string

from flaskr.db import get_db

app = Flask(__name__)

bp = Blueprint('note', __name__)

UPLOAD_FOLDER = 'Uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        #check if the post request has the file part
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
        if file.filename == '' or txt.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename) or not allowed_file(txt.filename):
            flash('Please upload .csv and .txt files only')
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return redirect(url_for('note.open_file', filename=filename))
    return render_template('note/index.html')




@bp.route('/uploads/<filename>', methods=['GET', 'POST'])
def open_file(filename):
    with open(os.path.join('Uploads', filename)) as csvfile:
        csvdata = csv.reader(csvfile)
        next(csvdata, None)



        concepts = []
        txt_files = []

        for row in csvdata:
            concepts.append(string.capwords(row[3]))
            txt_files.append(row[1])

        class ItemTable(Table):
            concept_name = Col('Concept Ontology')
            #note_file = Col('Clincial Note File')
            concept_amount = Col('Amount')
            #link = Col('')


        # Get some objects
        class Item(object):
            def __init__(self, concept_name, note_file, link):
                self.concept_name = concept_name
                self.note_file = []
                self.concept_amount = 1
                self.link = link
                self.note_file.append(note_file)

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
        for i in range(0,len(concepts)):
            items.append(Item(concepts[i], txt_files[i], 'View Concept'))

        def remove_duplicates(items):
            for k in range(0,len(items)-1):
                for j in range(k+1,len(items)):
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
        table.border = True

    if request.method == 'POST':
        concept_chosen = string.capwords(request.form['concept_chosen'])
        error = 'No matching concept found'
        for n in items:
            if concept_chosen == n.obtain_name():
                error = None
                txt = n.obtain_txt_list()
        if error is not None:
            flash(error)
            return redirect(request.url)
        return redirect(url_for('note.txt_files', txt=txt))


    return render_template('note/concepts.html', table=table)


def stripword(word):
    word = word.replace("[", "")
    word = word.replace("]", "")
    word = word.replace("'", "")
    word = word.replace("'", "")
    return word


@bp.route('/<txt>', methods=['GET', 'POST'])
def txt_files(txt):
    txt_str = ''.join(txt)
    txt_str = stripword(txt_str)
    txt = txt_str.split(',')

    if request.method == 'POST':
        txt_chosen = request.form['txt_chosen']
        error = 'No matching txt file found'
        for n in txt:
            if txt_chosen in n:
                error = None
                txt_file = n
        if error is not None:
            flash(error)
            return redirect(request.url)
        return redirect(url_for('note.sentence', i_txt_file=txt_file))

    return render_template('note/txt_files.html', txt=txt)

@bp.route('/<i_txt_file>/sentence')
def sentence(i_txt_file):
    with open(os.path.join('Uploads', i_txt_file), 'r') as file:
        sentence = []
        for l in file:
            sentence.append(l.split())
    #sentence = ['Tommy came to the hospital on July 21st 2019, \
    #diagnosed as', 'common cold', 'no ', 'mononucleosis']

    #if request.method == 'POST':
        #concept_chosen = request.form['concept']
        #error = 'No matching concept found'
        #for n in sentence:
            #if concept_chosen in n:
                #error = None
                #concept = n
        #if error is not None:
            #flash(error)
            #return redirect(request.url)
        #return redirect(url_for('note.concept', txt_file=txt_file, concept=concept))

    return render_template('note/sentence.html', sentence=sentence)



@bp.route('/<i_txt_file>/<concept>', methods=('GET', 'POST'))
def concept(i_txt_file, concept):
    if request.method == 'POST':
        error = None
        correct = request.form['correct']
        correct = correct.lower()
        if correct != 'yes' and correct != 'no':
            error = 'Please insert yes or no.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                'INSERT INTO feedback (correct)'
                ' VALUES (?)',
                ([correct])
            )
            db.commit()
            return redirect(url_for('note.sentence', i_txt_file=i_txt_file))

    file = str(os.path.join('note', concept)) + '.html'
    return render_template('note/concept1.html', concept=concept)