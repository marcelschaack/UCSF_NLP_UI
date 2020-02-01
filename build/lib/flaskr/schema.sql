DROP TABLE IF EXISTS user;

CREATE TABLE feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  concept TEXT NOT NULL,
  negation INTEGER NOT NULL,
  hof INTEGER NOT NULL,
  location TEXT,
  correct_answ INTEGER NOT NULL,
  hof_answ INTEGER NOT NULL,
  location_answ INTEGER NOT NULL,
  negation_answ INTEGER NOT NULL,
  cosine_sim REAL,
  jaccard_dist REAL
  );