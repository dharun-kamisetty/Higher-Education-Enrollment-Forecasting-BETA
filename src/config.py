import os

RAW_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NDAP_REPORT_9259.csv")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
CLEANED_PATH = os.path.join(OUTPUTS_DIR, "cleaned_dataset.csv")
ML_DATASET_PATH = os.path.join(OUTPUTS_DIR, "ml_dataset.csv")

NUMERIC_COLUMNS = [
    "enrolment",
    "university_enrolment",
    "foreign_students",
    "stand_alone_institutions",
    "teachers",
    "colleges",
    "universities",
    "GPI",
    "GER",
    "PTR_all",
    "PTR_university_colleges",
    "PTR_university_units",
]

COLUMN_RENAME = {
    "Number of enrolment in higher education": "enrolment",
    "Number of foreign students": "foreign_students",
    "Number of stand alone institutions": "stand_alone_institutions",
    "Number of teachers": "teachers",
    "Number of colleges": "colleges",
    "Number of universities": "universities",
    "Gender Parity Index (GPI)": "GPI",
    "Gross Enrolment Ratio  (GER)": "GER",
    "All Institutions": "PTR_all",
    "University and Colleges": "PTR_university_colleges",
    "University and its units": "PTR_university_units",
    "Enrolment in Universities Teaching Departments": "university_enrolment",
}

TEACHER_RANK_ITEMS = [
    "Professor & Equivalent",
    "Reader & Associate Professor",
    "Lecturer / Assistant Professor",
    "Demonstrator / Tutor",
    "Temporary Teacher etc",
]

MISSING_MARKERS = ["<br>", "", "NA", "N/A", "null", "NaN", "nan"]

YEARS_BEFORE_PREDICTION = 1
FIRST_MODEL_YEAR = 2014
TEST_YEARS = [2018, 2019]
