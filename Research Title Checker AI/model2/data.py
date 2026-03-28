# import pandas as pd
# import random

# def generate_research_dataset(samples_per_class=500):
#     # Defining keyword pools for each review type
#     keywords = {
#         "Full": {
#             "topics": ["Clinical Trial of", "Efficacy of New Drug", "Invasive Surgery for", "High-Risk Intervention in", "Psychological Stress and"],
#             "subjects": ["Patients with Stage IV Cancer", "Prisoners", "Minors with Trauma", "Emergency Room Patients"],
#             "methods": ["Double-Blind Study", "Randomized Controlled Trial", "Longitudinal Physiological Monitoring"]
#         },
#         "Expedited": {
#             "topics": ["Survey of", "Behavioral Observation of", "Analysis of", "Focus Group on", "Educational Impact of"],
#             "subjects": ["Healthy Adults", "University Students", "Office Workers", "Nursing Staff"],
#             "methods": ["Non-Invasive Collection", "Standardized Testing", "Video Recording Analysis", "Strength Testing"]
#         },
#         "Exempt": {
#             "topics": ["De-identified Data Analysis of", "Retrospective Review of", "Public Observation of", "Secondary Data Study on"],
#             "subjects": ["Public Records", "Anonymized Medical Records", "Standard Educational Practices", "Publicly Available Social Media"],
#             "methods": ["Data Mining", "Historical Archive Review", "Curriculum Evaluation"]
#         }
#     }

#     data = []

#     for review_type, pools in keywords.items():
#         for _ in range(samples_per_class):
#             # Randomly assemble a title
#             topic = random.choice(pools["topics"])
#             subject = random.choice(pools["subjects"])
#             method = random.choice(pools["methods"])
            
#             # Mix up the structures for variety
#             structures = [
#                 f"{topic} {subject} using {method}",
#                 f"{method}: A {topic} {subject}",
#                 f"{topic} {subject}"
#             ]
#             title = random.choice(structures)
            
#             data.append({"Research Title": title, "Review Type": review_type})

#     # Create DataFrame and shuffle
#     df = pd.DataFrame(data)
#     df = df.sample(frac=1).reset_index(drop=True)
#     return df

# # Generate the data
# df_research = generate_research_dataset(500)

# # Show a glimpse and save
# print(df_research["Review Type"].value_counts())
# print("\nFirst 5 rows:")
# print(df_research.head())

# # Optional: Save to CSV
# df_research.to_csv("research_reviews.csv", index=False)





import pandas as pd
import random

def generate_research_dataset(samples_per_class=500):
    # Defining keyword pools for each review type - tailored to match the original dataset style
    keywords = {
        "Full": {
            "topics": [
                "The Effect of", "Exploring the Impact of", "Assessment of", "Evaluation of", 
                "Comparative Study of", "Development of", "Analysis of", "Influence of",
                "Effectiveness of", "Relationship Between", "Correlational Analysis of",
                "A Mixed-Methods Study on", "A Quantitative Analysis of", "A Qualitative Study of",
                "Investigating the Factors of", "Determinants of", "Predictors of", "Modeling the",
                "Implementation of", "Integration of", "Optimization of", "Formulation and",
                "Phytochemical Analysis of", "Anti-Urolithiatic Potential of", "Diversity Assessment of",
                "Habitat Assessment of", "Growth Performance of", "Response of", "Performance of",
                "Somatic Embryogenesis of", "Utilization of", "Adsorption of", "Removal of",
                "Geospatial Multi-Criteria Analysis of", "Site Suitability Analysis for"
            ],
            "subjects": [
                "Teacher Aspirants", "Pre-Service Teachers", "Junior High School Students", 
                "Senior High School Students", "College Students", "Indigenous Learners",
                "Zampen Native Chicken", "Itik Pinas Ducklings", "Black Soldier Fly Larvae",
                "Cacao Cuttings", "Coffee Cuttings", "Hybrid Yellow Corn", "Tomato Production",
                "Bell Pepper Production", "Eggplant Production", "Cucumber Production",
                "Coptotermes gestroi Termites", "Philippine Milk Termites", "Bamboo Species",
                "Mangrove Ecosystem", "Coral Diversity", "Seagrasses and Macroinvertebrates",
                "Invasive Alien Plant Species", "Medicinal Plants", "Herbal Formulations",
                "Activated Carbon from Agricultural Waste", "Phosphate Removal", "Heavy Metal Adsorption",
                "Biodegradable Films", "Bioplastic Development", "Semen Extender Development",
                "Araling Panlipunan Instruction", "Mathematics Performance", "Science Achievement",
                "Reading Comprehension", "Writing Skills", "Vocabulary Development", "Grammar Proficiency",
                "Student Engagement", "Academic Performance", "Learning Outcomes", "Teaching Competence",
                "ICT Integration", "AI Literacy", "Digital Competence", "Technological Pedagogical Knowledge"
            ],
            "methods": [
                "Quasi-Experimental Design", "Structural Equation Modeling", "Mixed-Methods Approach",
                "Randomized Controlled Trial", "Descriptive-Correlational Design", "Phenomenological Study",
                "Grounded Theory Approach", "Case Study Design", "Action Research", "Developmental Research",
                "Experimental Study", "Comparative Analysis", "Longitudinal Analysis", "Cross-Sectional Survey",
                "In Vitro Assay", "Phytochemical Screening", "GC-MS Analysis", "Atomic Absorption Spectroscopy",
                "Kinetic and Isotherm Modeling", "Response Surface Methodology", "Artificial Neural Network",
                "GIS-Based Multi-Criteria Analysis", "Geospatial Analysis", "Predictive Modeling",
                "Focus Group Discussion", "Document Analysis", "Content Analysis", "Discourse Analysis",
                "Semantic Analysis", "Stylistic Analysis", "Critical Discourse Analysis", "Ethnographic Study",
                "Tracer Study", "Employability Assessment", "Policy Analysis", "Program Evaluation"
            ]
        },
        "Expedited": {
            "topics": [
                "Lived Experiences of", "Perceptions of", "Attitudes Towards", "Awareness of",
                "Challenges Encountered by", "Coping Strategies of", "Factors Influencing",
                "Barriers to", "Effectiveness of", "Implementation of", "Assessment of",
                "Understanding the", "Exploring the", "Investigating the", "Analysis of",
                "Correlational Study on", "Relationship Between", "Impact of", "Role of",
                "Community-Based", "Stakeholders' Perspective on", "Evaluation of", "Status of",
                "Pagsusuri sa", "Antas ng", "Epekto ng", "Pananaw ng mga", "Kakayahan ng",
                "Pagbabagong Morpoponemiko ng", "Sipat-Suri sa", "Wika sa Mundo ng"
            ],
            "subjects": [
                "Student-Athletes", "Student Leaders", "Student Mothers", "Working Students",
                "Out-of-School Youth", "Juvenile Delinquents", "Children in Conflict with the Law",
                "Persons with Disabilities", "Visually Impaired Individuals", "Senior Citizens",
                "Barangay Officials", "Barangay Tanods", "Barangay Health Workers", "Police Officers",
                "Social Workers", "Teachers", "SPED Teachers", "Physical Education Teachers",
                "Nursing Students", "Criminology Students", "Accountancy Students", "Psychology Students",
                "OFW Families", "4Ps Beneficiaries", "TUPAD Beneficiaries", "Fisherfolk", "Fish Vendors",
                "Sardine Factory Workers", "Street Food Vendors", "Sari-Sari Store Owners",
                "Chabacano Speakers", "Tausug Community", "Muslim Students", "Indigenous Peoples",
                "Badjao Community", "Online Gamers", "Social Media Users", "K-Drama Fans",
                "TikTok Users", "Facebook Users", "Motorcycle Riders", "Public Utility Drivers"
            ],
            "methods": [
                "Phenomenological Study", "Qualitative Descriptive Design", "Focus Group Discussion",
                "Semi-Structured Interviews", "Key Informant Interviews", "Survey Questionnaire",
                "Descriptive Survey", "Correlational Analysis", "Comparative Analysis", "Case Study",
                "Exploratory Study", "Descriptive Design", "Mixed-Methods Design", "Document Analysis",
                "Content Analysis", "Thematic Analysis", "Narrative Analysis", "Discourse Analysis",
                "Descriptive-Comparative Design", "Cross-Sectional Survey", "Purposive Sampling",
                "Stratified Random Sampling", "Convenience Sampling", "Snowball Sampling"
            ]
        },
        "Exempt": {
            "topics": [
                "An Exposition on", "A Study of", "Analysis of", "Review of", "Exploration of",
                "On the Properties of", "On Certain", "An Analysis of", "De-identified Data Analysis of",
                "Retrospective Review of", "Secondary Data Study on", "Public Observation of",
                "Forecasting", "Predictive Modeling of", "Time Series Analysis of", "Determination of",
                "Assessment of", "Evaluation of", "Exposition of", "An Overview of", "Computational Analysis of",
                "Statistical Analysis of", "Mathematical Modeling of", "Graph Theoretical Analysis of",
                "Development of", "Formulation of", "Characterization of", "Optimization of",
                "Phytochemical Screening of", "In Vitro Evaluation of", "Anti-Diabetic Potential of",
                "Antioxidant Activity of", "Removal of", "Adsorption of", "Synthesis of",
                "Pagsusuri sa", "Isang Pagsusuri", "Pahambing na Pagsusuri", "Pagbabagong Morpoponemiko"
            ],
            "subjects": [
                "Graphs and Graph Theory", "Prime Labeling", "Domination Number", "Vertex Switching",
                "Perfect Numbers", "Lucas Numbers", "Pythagorean Triples", "Fibonacci Sequence",
                "Mathematical Proofs", "Statistical Methods", "R Packages", "Meta-Analysis",
                "Time Series Data", "Forecasting Models", "ARIMA Modeling", "Benford's Law",
                "Gross Domestic Product", "Inflation Rates", "Retail Prices", "Temperature Forecasting",
                "Public Records", "Anonymized Medical Records", "Standard Educational Practices",
                "Publicly Available Data", "Historical Archives", "Curriculum Evaluation",
                "Lead and Antioxidant Levels", "Arsenic and Vitamin D3", "Heavy Metal Analysis",
                "Atomic Absorption Spectroscopy", "Phytochemical Properties", "Herbal Formulations",
                "Activated Carbon", "Biosorbent Materials", "Biodegradable Films", "Bioplastic Development",
                "Water Quality Assessment", "Air Quality Monitoring", "Flood Susceptibility",
                "Land Suitability Analysis", "Geospatial Mapping", "Machine Learning Models",
                "Deep Learning Applications", "Convolutional Neural Networks", "YOLOv5 Detection",
                "IoT-Based Systems", "Mobile Application Development", "Blockchain Technology"
            ],
            "methods": [
                "Mathematical Proof", "Graph Theoretical Approach", "Statistical Modeling", "Time Series Analysis",
                "Expository Method", "Literature Review", "Document Analysis", "Data Mining",
                "Secondary Data Analysis", "Public Database Review", "Historical Archive Review",
                "Computational Analysis", "Algorithmic Approach", "Machine Learning Modeling",
                "Deep Learning Architecture", "Convolutional Neural Network", "Random Forest Algorithm",
                "Support Vector Machine", "ARIMA Modeling", "SARIMA Forecasting", "Holt-Winters Method",
                "Descriptive Statistics", "Inferential Statistics", "Correlation Analysis", "Regression Analysis",
                "FAAS Analysis", "AAS Analysis", "GC-MS Analysis", "Phytochemical Screening",
                "In Vitro Assay", "Kinetic Modeling", "Isotherm Modeling", "Response Surface Methodology",
                "Artificial Neural Network", "GIS-Based Analysis", "Multi-Criteria Decision Analysis",
                "Geospatial Analysis", "Remote Sensing", "Survey of Literature", "Systematic Review"
            ]
        }
    }

    data = []

    for review_type, pools in keywords.items():
        for _ in range(samples_per_class):
            # Randomly assemble a title with various structures
            structure_choice = random.choice([1, 2, 3, 4, 5])
            
            if structure_choice == 1:
                title = f"{random.choice(pools['topics'])} {random.choice(pools['subjects'])}"
            elif structure_choice == 2:
                title = f"{random.choice(pools['topics'])} {random.choice(pools['subjects'])}: {random.choice(pools['methods'])}"
            elif structure_choice == 3:
                title = f"{random.choice(pools['methods'])}: {random.choice(pools['topics'])} {random.choice(pools['subjects'])}"
            elif structure_choice == 4:
                title = f"{random.choice(pools['topics'])} {random.choice(pools['subjects'])} Using {random.choice(pools['methods'])}"
            else:
                title = f"{random.choice(pools['subjects'])}: {random.choice(pools['topics'])} {random.choice(pools['methods'])}"
            
            # Add some variety with colons, commas, and subtitles like original dataset
            if random.random() > 0.7:
                subtitle_options = [
                    f" A {random.choice(pools['methods'])}",
                    f" Basis for {random.choice(['Policy Development', 'Program Enhancement', 'Curriculum Framework', 'Action Plan'])}",
                    f" Inputs to {random.choice(['Instructional Model', 'Capacity Development', 'Strategic Framework', 'Management System'])}",
                    f" An {random.choice(['Exposition', 'Analysis', 'Evaluation', 'Assessment'])}"
                ]
                title = title + random.choice(subtitle_options)
            
            data.append({"Research Title": title, "Review Type": review_type})

    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# Generate the data
df_research = generate_research_dataset(500)

# Show counts and sample
print("=== Counts by Review Type ===")
print(df_research["Review Type"].value_counts())
print(f"\nTotal records: {len(df_research)}")
print("\n=== First 10 rows ===")
print(df_research.head(10).to_string(index=False))

# Save to CSV without quotes for clean format
df_research.to_csv("research_reviews2.csv", index=False, quoting=1)  # quoting=1 means minimal quoting

print("\n✅ Dataset saved to 'research_reviews2.csv'")