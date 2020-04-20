DROP TABLE IF EXISTS feedback;
DROP TABLE IF EXISTS dependencies;

CREATE TABLE feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  range_txt TEXT NOT NULL,
  concept TEXT NOT NULL,
  negation INTEGER NOT NULL,
  hof INTEGER NOT NULL,
  location TEXT,
  correct_answ INTEGER,
  hof_answ INTEGER,
  location_answ INTEGER,
  negation_answ INTEGER,
  bleu_score REAL NOT NULL,
  levenstein_sim REAL NOT NULL,
  cosine_sim REAL NOT NULL,
  jaccard_sim REAL NOT NULL,
  elmo REAL NOT NULL
  );

CREATE TABLE dependencies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  word TEXT NOT NULL,
  pos TEXT NOT NULL,
  feedback TEXT NOT NULL
  );