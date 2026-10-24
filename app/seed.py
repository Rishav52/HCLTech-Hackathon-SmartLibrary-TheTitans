"""Synthetic seed data: a small but realistic catalog, users and circulation history.

Dates are relative to "today" so demo scenarios (e.g. "due this week", overdue) always work.
Run:  python -m app.seed [--reset]
"""
import argparse
import os
import random
from datetime import date, timedelta

from .db import DB_PATH, connect, init_db

# (type, title, authors, year, isbn, abstract, subjects, level, course_tags, source, access_url, n_copies)
RESOURCES = [
    ("book", "Artificial Intelligence: A Modern Approach", "Stuart Russell, Peter Norvig", 2021, "9780134610993",
     "Comprehensive introduction to AI covering search, knowledge, planning, learning and robotics.",
     "artificial intelligence,machine learning,search", "intermediate", "CS301,DS201", "catalog", None, 3),
    ("book", "Python Crash Course", "Eric Matthes", 2023, "9781718502703",
     "Hands-on, project-based introduction to Python programming for beginners.",
     "python,programming", "beginner", "CS101", "catalog", None, 4),
    ("book", "Automate the Boring Stuff with Python", "Al Sweigart", 2019, "9781593279929",
     "Practical Python programming for beginners: files, web scraping, spreadsheets and automation.",
     "python,programming,automation", "beginner", "CS101", "catalog", None, 2),
    ("book", "Fluent Python", "Luciano Ramalho", 2022, "9781492056355",
     "Deep dive into idiomatic Python: data model, concurrency, metaprogramming.",
     "python,programming", "advanced", "CS201", "catalog", None, 2),
    ("book", "Introduction to Machine Learning with Python", "Andreas Mueller, Sarah Guido", 2016, "9781449369415",
     "Beginner-friendly guide to machine learning using scikit-learn.",
     "machine learning,python,data science", "beginner", "DS201,CS301", "catalog", None, 3),
    ("book", "Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow", "Aurelien Geron", 2022, "9781098125974",
     "Practical machine learning and deep learning with real projects.",
     "machine learning,deep learning,python", "intermediate", "DS201,CS301", "catalog", None, 2),
    ("book", "Pattern Recognition and Machine Learning", "Christopher Bishop", 2006, "9780387310732",
     "Rigorous Bayesian treatment of pattern recognition and machine learning.",
     "machine learning,statistics,probability", "advanced", "DS301", "catalog", None, 1),
    ("book", "Deep Learning", "Ian Goodfellow, Yoshua Bengio, Aaron Courville", 2016, "9780262035613",
     "Foundational textbook on deep neural networks and optimization.",
     "deep learning,neural networks", "advanced", "DS301,CS401", "catalog", None, 2),
    ("book", "Python for Data Analysis", "Wes McKinney", 2022, "9781098104030",
     "Data wrangling with pandas, NumPy and Jupyter for data science.",
     "data science,python,pandas", "beginner", "DS101,DS201", "catalog", None, 3),
    ("book", "Data Science from Scratch", "Joel Grus", 2019, "9781492041139",
     "First principles introduction to data science, statistics and machine learning in Python.",
     "data science,python,statistics", "beginner", "DS101", "catalog", None, 2),
    ("book", "The Elements of Statistical Learning", "Trevor Hastie, Robert Tibshirani, Jerome Friedman", 2009, "9780387848570",
     "Classic reference on statistical learning methods.",
     "statistics,machine learning", "advanced", "DS301", "catalog", None, 1),
    ("book", "Introduction to Algorithms", "Thomas Cormen et al.", 2022, "9780262046305",
     "Comprehensive textbook on algorithms and data structures.",
     "algorithms,data structures", "intermediate", "CS202", "catalog", None, 4),
    ("book", "Database System Concepts", "Abraham Silberschatz et al.", 2019, "9780078022159",
     "Core database textbook: relational model, SQL, transactions, indexing.",
     "databases,sql", "intermediate", "CS204", "catalog", None, 3),
    ("book", "Computer Networking: A Top-Down Approach", "James Kurose, Keith Ross", 2021, "9780136681557",
     "Introduction to computer networks from the application layer down.",
     "networking", "intermediate", "CS305", "catalog", None, 2),
    ("book", "Designing Data-Intensive Applications", "Martin Kleppmann", 2017, "9781449373320",
     "Principles behind reliable, scalable and maintainable data systems.",
     "distributed systems,databases", "advanced", "CS401", "catalog", None, 1),
    ("book", "Clean Code", "Robert C. Martin", 2008, "9780132350884",
     "Handbook of agile software craftsmanship and readable code.",
     "software engineering,programming", "intermediate", "CS210", "catalog", None, 2),
    ("book", "Linear Algebra and Its Applications", "David Lay", 2015, "9780321982384",
     "Linear algebra with applications to science and engineering.",
     "mathematics,linear algebra", "intermediate", "MA102,DS201", "catalog", None, 3),
    ("book", "Probability and Statistics for Engineers", "Richard Johnson", 2016, "9780321640772",
     "Probability, inference and regression for engineering students.",
     "statistics,probability", "beginner", "MA201,DS101", "catalog", None, 2),
    ("ebook", "Generative AI with Large Language Models: A Beginner's Guide", "Library Digital Press", 2024, "9781000000011",
     "Beginner introduction to generative AI, prompting and large language models.",
     "generative ai,llm,artificial intelligence", "beginner", "CS301,DS201", "digital_repository",
     "https://digital.library.example/ebooks/genai-beginners", 0),
    ("ebook", "Natural Language Processing with Transformers", "Lewis Tunstall et al.", 2022, "9781098136796",
     "Building NLP applications with Hugging Face transformers.",
     "nlp,deep learning,llm", "intermediate", "CS401,DS301", "digital_repository",
     "https://digital.library.example/ebooks/nlp-transformers", 0),
    ("ebook", "Introduction to Data Science: A Python Approach", "Library Digital Press", 2023, "9781000000028",
     "Gentle introduction to data science workflows for first-year students.",
     "data science,python", "beginner", "DS101", "digital_repository",
     "https://digital.library.example/ebooks/intro-data-science", 0),
    ("learning_path", "Generative AI Learning Path (Beginner)", "Department of Computer Science", 2024, None,
     "Curated sequence of readings and exercises introducing generative AI and prompt engineering.",
     "generative ai,llm,artificial intelligence", "beginner", "CS301", "digital_repository",
     "https://digital.library.example/paths/genai-beginner", 0),
    ("learning_path", "Data Science Foundations Learning Path", "Department of Statistics", 2024, None,
     "Step-by-step path: Python, statistics, visualization and first machine learning models.",
     "data science,python,statistics,machine learning", "beginner", "DS101,DS201", "digital_repository",
     "https://digital.library.example/paths/ds-foundations", 0),
    ("paper", "Attention Is All You Need", "Vaswani et al.", 2017, None,
     "Introduces the Transformer architecture based solely on attention mechanisms.",
     "deep learning,nlp,transformers", "advanced", "CS401,DS301", "research_archive",
     "https://archive.library.example/papers/attention-is-all-you-need", 0),
    ("paper", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "Lewis et al.", 2020, None,
     "Combines parametric and non-parametric memory: a retriever feeds documents to a seq2seq generator (RAG).",
     "retrieval-augmented generation,rag,nlp,llm", "advanced", "CS401", "research_archive",
     "https://archive.library.example/papers/rag-lewis-2020", 0),
    ("paper", "Dense Passage Retrieval for Open-Domain Question Answering", "Karpukhin et al.", 2020, None,
     "Learns dense dual-encoder representations for passage retrieval, improving open-domain QA.",
     "retrieval,rag,nlp,question answering", "advanced", "CS401", "research_archive",
     "https://archive.library.example/papers/dpr-2020", 0),
    ("paper", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "Asai et al.", 2023, None,
     "Trains an LM to adaptively retrieve passages and critique its own generations.",
     "retrieval-augmented generation,rag,llm", "advanced", "CS401", "research_archive",
     "https://archive.library.example/papers/self-rag-2023", 0),
    ("paper", "BERT: Pre-training of Deep Bidirectional Transformers", "Devlin et al.", 2019, None,
     "Bidirectional pre-training of transformers for language understanding.",
     "nlp,deep learning,transformers", "advanced", "CS401", "research_archive",
     "https://archive.library.example/papers/bert-2019", 0),
    ("journal", "Journal of Machine Learning Research", "JMLR Inc.", 2024, None,
     "Peer-reviewed open journal covering all areas of machine learning.",
     "machine learning,statistics", "advanced", "DS301", "journal_subscription",
     "https://journals.library.example/jmlr", 0),
    ("journal", "IEEE Transactions on Pattern Analysis and Machine Intelligence", "IEEE", 2024, None,
     "Leading journal on computer vision, pattern analysis and machine intelligence.",
     "computer vision,machine learning", "advanced", "CS401", "journal_subscription",
     "https://journals.library.example/tpami", 0),
    ("dissertation", "Semantic Search for University Digital Libraries", "A. Sharma", 2023, None,
     "PhD thesis on embedding-based retrieval and metadata enrichment for academic repositories.",
     "information retrieval,semantic search,libraries", "advanced", "", "research_archive",
     "https://archive.library.example/theses/sharma-2023", 0),
    ("dissertation", "Recommender Systems for Academic Reading Lists", "R. Iyer", 2022, None,
     "Master's thesis on course-aware recommendation of reading material to students.",
     "recommender systems,education", "advanced", "", "research_archive",
     "https://archive.library.example/theses/iyer-2022", 0),
]

USERS = [
    ("Aarav Mehta", "aarav@univ.example", "student", "B.Tech Data Science", "DS101,DS201,MA201", "machine learning,python", 1),
    ("Diya Nair", "diya@univ.example", "student", "B.Tech Computer Science", "CS101,CS202,CS204", "algorithms,web", 1),
    ("Rohan Gupta", "rohan@univ.example", "student", "B.Tech Computer Science", "CS301,CS305", "artificial intelligence", 0),
    ("Ishita Rao", "ishita@univ.example", "student", "M.Tech AI", "CS401,DS301", "nlp,llm", 1),
    ("Kabir Singh", "kabir@univ.example", "student", "B.Tech Data Science", "DS101,MA102", "statistics", 1),
    ("Meera Pillai", "meera@univ.example", "student", "B.Tech Computer Science", "CS101,CS210", "programming", 0),
    ("Dr. Anil Kapoor", "anil@univ.example", "faculty", "Computer Science", "CS301,CS401", "deep learning,nlp", 1),
    ("Dr. Sunita Verma", "sunita@univ.example", "researcher", "Information Retrieval Lab", "", "rag,semantic search", 1),
    ("Lena Fernandes", "lena@univ.example", "librarian", "Library Services", "", "", 0),
]

BRANCHES = [("Central Library", "Main Campus"), ("Engineering Library", "Engineering Block"), ("Science Library", "Science Block")]


def seed(conn, today: date | None = None, rng_seed: int = 42) -> dict:
    rng = random.Random(rng_seed)
    today = today or date.today()
    iso = lambda d: d.isoformat()

    for r in RESOURCES:
        cur = conn.execute(
            "INSERT INTO resources (type,title,authors,year,isbn,abstract,subjects,level,course_tags,source,access_url)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)", r[:11])
        rid, n_copies = cur.lastrowid, r[11]
        for i in range(n_copies):
            branch, area = BRANCHES[(rid + i) % len(BRANCHES)]
            shelf = f"{area} / Shelf {chr(65 + rid % 8)}{(rid * 7 + i) % 40 + 1:02d}"
            conn.execute("INSERT INTO copies (resource_id,branch,shelf_location) VALUES (?,?,?)", (rid, branch, shelf))

    for u in USERS:
        conn.execute("INSERT INTO users (name,email,role,program,courses,interests,history_consent) VALUES (?,?,?,?,?,?,?)", u)

    users = {row["email"]: row["id"] for row in conn.execute("SELECT id,email FROM users")}
    title_copies = {}
    for row in conn.execute("SELECT c.id, r.title FROM copies c JOIN resources r ON r.id=c.resource_id ORDER BY c.id"):
        title_copies.setdefault(row["title"], []).append(row["id"])

    def lend(copy_id, email, borrowed_days_ago, due_in_days, returned=False, renewals=0):
        borrowed = today - timedelta(days=borrowed_days_ago)
        due = today + timedelta(days=due_in_days)
        conn.execute(
            "INSERT INTO loans (copy_id,user_id,borrowed_at,due_at,returned_at,renewals) VALUES (?,?,?,?,?,?)",
            (copy_id, users[email], iso(borrowed), iso(due), iso(borrowed + timedelta(days=rng.randint(5, 20))) if returned else None, renewals))
        if not returned:
            conn.execute("UPDATE copies SET status='on_loan' WHERE id=?", (copy_id,))

    # --- Scenario fixtures for the minimum demo ------------------------------------------------
    aima = title_copies["Artificial Intelligence: A Modern Approach"]
    # Aarav: one due this week, one overdue (fine), one comfortable
    lend(aima[0], "aarav@univ.example", 12, 2)
    lend(title_copies["Python for Data Analysis"][0], "aarav@univ.example", 30, -4)
    lend(title_copies["Data Science from Scratch"][0], "aarav@univ.example", 5, 9)
    # Another user's loans (privacy-refusal scenario: Aarav must not see Diya's)
    lend(title_copies["Database System Concepts"][0], "diya@univ.example", 8, 6)
    lend(title_copies["Introduction to Algorithms"][0], "diya@univ.example", 20, -1, renewals=2)

    # Fully checked-out title with a waitlist (the "reserve" and "longest waitlist" scenarios)
    for cid, email in zip(title_copies["Python Crash Course"], ["rohan@univ.example", "meera@univ.example", "kabir@univ.example", "ishita@univ.example"]):
        lend(cid, email, rng.randint(3, 15), rng.randint(3, 14))
    for i, email in enumerate(["aarav@univ.example", "diya@univ.example", "rohan@univ.example"]):
        conn.execute("INSERT INTO reservations (resource_id,user_id,created_at) VALUES ((SELECT id FROM resources WHERE title='Python Crash Course'),?,?)",
                     (users[email], iso(today - timedelta(days=3 - i))))

    # Single-copy advanced titles: on loan + waitlist
    for title, holder, waiters in [
        ("Pattern Recognition and Machine Learning", "ishita@univ.example", ["kabir@univ.example", "aarav@univ.example"]),
        ("Designing Data-Intensive Applications", "anil@univ.example", ["diya@univ.example"]),
    ]:
        lend(title_copies[title][0], holder, 10, 5)
        for i, email in enumerate(waiters):
            conn.execute("INSERT INTO reservations (resource_id,user_id,created_at) VALUES ((SELECT id FROM resources WHERE title=?),?,?)",
                         (title, users[email], iso(today - timedelta(days=2 - i))))

    # --- Historical circulation (feeds analytics + recommendations) ---------------------------
    student_emails = [e for e in users if e != "lena@univ.example"]
    physical = [t for t in title_copies]
    weights = [3 if t.startswith(("Python", "Introduction to Machine", "Artificial")) else 1 for t in physical]
    for _ in range(140):
        title = rng.choices(physical, weights)[0]
        cid = rng.choice(title_copies[title])
        days_ago = rng.randint(20, 330)
        email = rng.choice(student_emails)
        borrowed = today - timedelta(days=days_ago)
        returned = borrowed + timedelta(days=rng.randint(5, 35))
        due = borrowed + timedelta(days=21)
        conn.execute("INSERT INTO loans (copy_id,user_id,borrowed_at,due_at,returned_at,renewals) VALUES (?,?,?,?,?,?)",
                     (cid, users[email], iso(borrowed), iso(due), iso(returned), rng.choice([0, 0, 0, 1, 2])))

    # --- Digital usage + search history (separate RNG so earlier seed data is unchanged) ------------
    r2 = random.Random(rng_seed + 1)
    digital = [row["id"] for row in conn.execute("SELECT id FROM resources WHERE access_url IS NOT NULL ORDER BY id")]
    heavy = set(digital[:4])                          # a few popular digital items; the rest stay quiet, a couple unused
    for rid in digital[: len(digital) - 3]:
        for _ in range(r2.randint(25, 60) if rid in heavy else r2.randint(1, 8)):
            conn.execute("INSERT INTO usage_events (resource_id,user_id,kind,ts) VALUES (?,?,?,?)",
                         (rid, users[r2.choice(student_emails)], r2.choice(["view", "view", "download"]),
                          iso(today - timedelta(days=r2.randint(0, 300)))))
    gaps = [("quantum computing", 6), ("rust programming language", 4), ("kubernetes in action", 3),
            ("ethics of artificial intelligence", 2), ("medieval history", 1)]
    served = ["python programming", "machine learning", "data science", "deep learning", "statistics", "algorithms", "databases"]
    searches = [(q, "not_found") for q, n in gaps for _ in range(n)] + [(r2.choice(served), "ok") for _ in range(45)]
    r2.shuffle(searches)
    for q, outcome in searches:
        when = f"{iso(today - timedelta(days=r2.randint(0, 80)))} {r2.randint(8, 21):02d}:{r2.randint(0, 59):02d}:00"
        conn.execute("INSERT INTO audit_log (trace_id,ts,user_id,tool,params,outcome,latency_ms) VALUES (?,?,?,?,?,?,?)",
                     (f"seed{r2.getrandbits(40):010x}", when, users[r2.choice(student_emails)], "search_resources",
                      f'{{"query": "{q}", "top_k": 5}}', outcome, round(r2.uniform(8, 60), 2)))

    conn.commit()
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("resources", "copies", "users", "loans", "reservations")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete the existing database first")
    args = ap.parse_args()
    if args.reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = connect()
    init_db(conn)
    if conn.execute("SELECT COUNT(*) FROM resources").fetchone()[0]:
        print("Database already seeded; use --reset to rebuild.")
        return
    print("Seeded:", seed(conn))


if __name__ == "__main__":
    main()
