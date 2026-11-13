# === Configuration partagée par toutes les étapes ===

RANDOM_STATE = 26

TARGET = "Mental_Health_Risk"
ID_COL = "Student_ID"

NUMERIC_COLS = [
    "Age",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]

NOMINAL_COLS = [
    "Gender",
    "Part_Time_Job",
    "Previous_Mental_Health_History",
]

ORDINAL_COLS = [
    "Physical_Activity",
    "Diet_Quality",
]

ORDINAL_CATEGORIES = [
    ["Low", "Moderate", "High"],
    ["Poor", "Average", "Good"],
]

OTHER_NOMINAL_COLS = [
    "Year_of_Study",
]
