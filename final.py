import streamlit as st
import pandas as pd
import joblib
import numpy as np
from openai import OpenAI
import json
import base64

# ======= Utility Functions =========
def decode_api_key(encoded_api_key):
    decoded_bytes = base64.b64decode(encoded_api_key.encode('utf-8'))
    decoded_str = str(decoded_bytes, 'utf-8')
    return decoded_str

# Replace with your encoded API key
encoded_api_key = "your-api-key"

# Initialize OpenAI client
client = OpenAI(api_key=decode_api_key(encoded_api_key=encoded_api_key))

# ======= Load ML Model Artifacts =========
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, label_encoder, scaler

model, label_encoder, scaler = load_model_artifacts()

# ======= Feature Configuration (EXACT MATCH WITH TRAINING DATA) =========
COLLECTED_FEATURES = {
    'Subjects_before_university': [
        'Chemistry', 'Physics', 'Mathematics', 'Biology',
        'Computer Science', 'Engineering', 'Psychology',
        'Sociology', 'Geography', 'Geology', 
        'Environmental Science', 'Computer Engineering',
        'Information Technology', 'Software Engineering',
        'Unknown', 'Other'
    ],
    'Career_prospects_Importance': [
        'Extremely not important', 'Somewhat not important',
        'Neutral', 'Somewhat important', 'Extremely important'
    ],
    'Post_course_goals': [
        'Secure a job in a specific industry or field', 
        'Develop transferable skills for various job sectors',
        'Pursue further study or research in the field',
        'Start my own business or entrepreneurial venture',
        'Gain a broad knowledge base and personal growth'
    ],
    'Workload_assessment_method': [
        "I didn't think about it much; I just went for it",
        'I assumed it would be manageable based on my interests and strengths',
        'I considered the course prerequisites and my academic background',
        'I researched online and talked to current students',
        'I spoke with academic advisors or professors'
    ],
    'Preferred_learning_environment': [
        'A hands-on, practical approach with real-world applications',
        'A theoretical, lecture-based approach with in-depth learning',
        'A mix of both theory and practice',
        'Self-paced, independent learning',
        'A collaborative, group-based learning experience'
    ],
    'Alignment_with_career_goals': [
        'Neutral', 
        'Somewhat interested',
        'Extremely interested',
        'Extremely not interested',
        'Somewhat not interested'
    ],
    'Confident_skills': [
        'Problem-solving',
        'Researching skills',
        'Mathematical skills',
        'Technical skills'
    ],
    'Your_strengths': [
        'Data analysis',
        'Lab work',
        'Ethical reasoning',
        'Leadership skills'
    ]
}

# Default values from training
DEFAULTS = {
    'Reason_for_current_course': 'Passion for the subject',
    'First_interest_for_current_course': 'Personal experience or hobby',
    'previous_course_prepared_how_much_for_current_course': 5.71,
    'Would_you_recommend': 'Yes',
    'Alternative_choice': 'No',
    'satisfaction_with_current_course': 5.65
}

def sanitize(name):
    """EXACT same sanitization as training"""
    return (name.replace(" ", "_")
              .replace(",", "")
              .replace(";", "")
              .replace("'", "")
              .replace("-", "_")
              .lower())

def preprocess_input(form_data):
    """Process form data to match model requirements"""
    df = pd.DataFrame([form_data])
    
    # Add defaults for missing features
    for feature, value in DEFAULTS.items():
        df[feature] = value
    
    # Ordinal encoding with training order
    ordinal_map = {
        'Career_prospects_Importance': COLLECTED_FEATURES['Career_prospects_Importance'],
        'Alignment_with_career_goals': COLLECTED_FEATURES['Alignment_with_career_goals'],
        'Workload_assessment_method': COLLECTED_FEATURES['Workload_assessment_method']
    }
    for col, categories in ordinal_map.items():
        if col in form_data:
            df[col] = categories.index(form_data[col])
    
    # One-hot encoding with exact training names
    categorical_cols = [
        'Subjects_before_university', 'Post_course_goals',
        'Preferred_learning_environment', 'Confident_skills',
        'Your_strengths', 'Reason_for_current_course',
        'First_interest_for_current_course', 'Would_you_recommend',
        'Alternative_choice'
    ]
    
    for col in categorical_cols:
        options = COLLECTED_FEATURES.get(col, [DEFAULTS.get(col, '')])
        for option in options:
            clean_option = sanitize(option)
            df[f"{col}_{clean_option}"] = 0
        selected = form_data.get(col, DEFAULTS.get(col, ''))
        clean_selected = sanitize(selected)
        df[f"{col}_{clean_selected}"] = 1
    
    # Scale ONLY features scaled during training
    numerical_cols = ['previous_course_prepared_how_much_for_current_course']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Add satisfaction WITHOUT scaling
    df['satisfaction_with_current_course'] = DEFAULTS['satisfaction_with_current_course']
    
    # Match EXACT feature names
    expected_features = model.feature_names_in_
    missing = set(expected_features) - set(df.columns)
    for feature in missing:
        df[feature] = 0
    
    return df[expected_features]

# ======= LLM Prompt =========
LLM_PROMPT = """
You are a university course recommendation expert. Based on a student's answers to the following assessment questions, determine the most suitable university courses for them.

Courses available include: Zoology, Bio Chemistry, Bio-Medical Sciences, Biology, Chemical Engineering, Chemistry, 
Civil Engineering, Mathematics, Mathematics with Finance, Mechanical Engineering, Mechatronics & Automated Systems, 
Pharmacy, Physics, Psychology, Software Engineering, Computer Science, Cosmetics Science, Data Science, 
Artificial Intelligence.

The student has answered these questions:
{user_responses}

The machine learning model has recommended: {model_prediction} with {model_confidence}% confidence.

**Important Considerations:**
- Take into account the subjects the student has studied, including any non-STEM subjects like Sociology.
- Consider how the student's interests and motivations align with the recommended courses.
- Provide a balanced recommendation that reflects both the student's academic background and their career aspirations.
- Eventually recommend most relevant STEM subject.

Please provide your analysis in the following structured format:

Student Profile Overview
[Provide a brief, focused summary of the student's key characteristics, preferences, and strengths]

Top 3 Recommended Courses
1. [Course Name]: [One-line explanation]
2. [Course Name]: [One-line explanation]
3. [Course Name]: [One-line explanation]

Comments on the ML Model's Recommendation
[Brief analysis of whether you agree with the ML model's suggestion and why/why not]

Additional Advice for the Student
[Short, actionable advice for next steps]

Keep each section concise and focused. Use bullet points where appropriate.
"""

# ======= Question Definitions =========
QUESTIONS = [
    {
        "id": 1,
        "text": "1️⃣ Most Relevant Subject? (Pick 2 options)",
        "options": [
            "Mathematics",
            "Physics",
            "Biology",
            "Chemistry",
            "Computer Science",
            "Engineering",
            "Psychology",
            "Data Science",
            "Information Technology",
            "Software Engineering",
            "Environmental Science",
            "Pharmacy",
            "Biomedical Sciences",
            "Geography",
            "Geology",
            "Sociology",
            "Other"
        ],
        "multiple": True,
        "max_selections": 2,
        "model_field": "Subjects_before_university",
        "mapping": {
            "Mathematics": "Mathematics",
            "Physics": "Physics",
            "Biology": "Biology",
            "Chemistry": "Chemistry",
            "Computer Science": "Computer Science",
            "Engineering": "Engineering",
            "Psychology": "Psychology",
            "Data Science": "Computer Science",
            "Information Technology": "Information Technology",
            "Software Engineering": "Software Engineering",
            "Environmental Science": "Environmental Science",
            "Pharmacy": "Chemistry",
            "Biomedical Sciences": "Biology",
            "Geography": "Geography",
            "Geology": "Geology",
            "Sociology": "Sociology",
            "Other": "Other"
        }
    },
    {
        "id": 2,
        "text": "2️⃣ Motivation for Picking the Course?",
        "options": [
            "Passion for the subject",
            "Pursue further research",
            "Job prospects",
            "Earning potential",
            "Family influence"
        ],
        "model_field": "Reason_for_current_course",
        "mapping": {
            "Passion for the subject": "Passion for the subject",
            "Pursue further research": "Pursue further research",
            "Job prospects": "Strong job prospects",
            "Earning potential": "High earning potential",
            "Family influence": "Family influence"
        }
    },
    {
        "id": 3,
        "text": "3️⃣ Career Priority?",
        "options": [
            "High earnings",
            "Job security",
            "Work satisfaction",
            "Helping others",
            "Not a priority"
        ],
        "model_field": "Career_prospects_Importance",
        "mapping": {
            "High earnings": "Extremely important",
            "Job security": "Extremely important",
            "Work satisfaction": "Somewhat important",
            "Helping others": "Neutral",
            "Not a priority": "Somewhat not important"
        }
    },
    {
        "id": 4,
        "text": "4️⃣ School Grades?",
        "options": [
            "A*",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "N/A"
        ],
        "model_field": "previous_course_prepared_how_much_for_current_course",
        "mapping": {
            "A*": 10.0,
            "A": 8.5,
            "B": 7.5,
            "C": 6.5,
            "D": 5.5,
            "E": 4.5,
            "F": 3.5,
            "N/A": 3.0
        }
    },
    {
        "id": 5,
        "text": "5️⃣ Preferred Skill Set?",
        "options": [
            "Math skills",
            "Problem-solving",
            "Technical skills",
            "Research skills"
        ],
        "model_field": "Confident_skills",
        "mapping": {
            "Math skills": "Mathematical skills",
            "Problem-solving": "Problem-solving",
            "Technical skills": "Technical skills",
            "Research skills": "Researching skills"
        }
    },
    {
        "id": 6,
        "text": "6️⃣ Work Style Preference?",
        "options": [
            "Independent problem-solving",
            "Collaborative teamwork"
        ],
        "model_field": "Preferred_learning_environment",
        "mapping": {
            "Independent problem-solving": "Self-paced, independent learning",
            "Collaborative teamwork": "A collaborative, group-based learning experience"
        }
    },
    {
        "id": 7,
        "text": "7️⃣ Practical vs. Theoretical Learning?",
        "options": [
            "Hands-on, practical learning",
            "Theoretical, lecture-based learning",
            "Mix of both"
        ],
        "model_field": "Preferred_learning_environment",
        "mapping": {
            "Hands-on, practical learning": "A hands-on, practical approach with real-world applications",
            "Theoretical, lecture-based learning": "A theoretical, lecture-based approach with in-depth learning",
            "Mix of both": "A mix of both theory and practice"
        }
    },
    {
        "id": 8,
        "text": "8️⃣ Preferred Work Environment?",
        "options": [
            "Data-driven analysis",
            "Creating projects & designs",
            "Lab-based work",
            "Field work"
        ],
        "model_field": "Your_strengths",
        "mapping": {
            "Data-driven analysis": "Data analysis",
            "Creating projects & designs": "Leadership skills",
            "Lab-based work": "Lab work",
            "Field work": "Ethical reasoning"
        }
    },
    {
        "id": 9,
        "text": "9️⃣ Communication vs. Research Skills?",
        "options": [
            "Communicating & explaining",
            "Technical Reasoning"
        ],
        "model_field": "Post_course_goals",
        "mapping": {
            "Communicating & explaining": "Develop transferable skills for various job sectors",
            "Technical Reasoning": "Pursue further study or research in the field"
        }
    },
    {
        "id": 10,
        "text": "🔟 Individual or Teamwork?",
        "options": [
            "Individual projects",
            "Team projects"
        ],
        "model_field": "Workload_assessment_method",
        "mapping": {
            "Individual projects": "I assumed it would be manageable based on my interests and strengths",
            "Team projects": "I researched online and talked to current students"
        }
    },
    {
        "id": 11,
        "text": "1️⃣1️⃣ Do You Like Technology?",
        "options": [
            "Yes",
            "No",
            "Somewhat"
        ],
        "llm_only": True
    },
    {
        "id": 12,
        "text": "1️⃣2️⃣ Industry vs. Academia?",
        "options": [
            "Industry job",
            "Academic / Research career",
            "Not sure yet"
        ],
        "model_field": "Post_course_goals",
        "mapping": {
            "Industry job": "Secure a job in a specific industry or field",
            "Academic / Research career": "Pursue further study or research in the field",
            "Not sure yet": "Gain a broad knowledge base and personal growth"
        }
    },
    {
        "id": 13,
        "text": "1️⃣3️⃣ Clear Career Path vs. Flexibility?",
        "options": [
            "Clear job roles",
            "Flexible career paths"
        ],
        "model_field": "Alignment_with_career_goals",
        "mapping": {
            "Clear job roles": "Extremely interested",
            "Flexible career paths": "Somewhat interested"
        }
    },
    {
        "id": 14,
        "text": "1️⃣4️⃣ Work-Life Balance Importance?",
        "options": [
            "Very important",
            "Somewhat important",
            "Not a priority"
        ],
        "llm_only": True
    },
    {
        "id": 15,
        "text": "1️⃣5️⃣ Degree Alignment to Career (Scale 1-10)?",
        "options": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "model_field": "Alignment_with_career_goals",
        "mapping": {
            "1": "Extremely not interested",
            "2": "Extremely not interested",
            "3": "Extremely not interested",
            "4": "Somewhat not interested",
            "5": "Somewhat not interested",
            "6": "Neutral",
            "7": "Neutral",
            "8": "Somewhat interested",
            "9": "Extremely interested",
            "10": "Extremely interested"
        }
    }
]

def generate_llm_response(user_responses, model_prediction, model_confidence):
    """Generate a response from the LLM based on user responses and model prediction"""
    # Format user responses for the prompt
    formatted_responses = "\n".join([f"Q: {q['text']}\nA: {user_responses.get(q['id'], 'Not answered')}" for q in QUESTIONS])
    
    # Create prompt
    prompt = LLM_PROMPT.format(
        user_responses=formatted_responses,
        model_prediction=model_prediction,
        model_confidence=model_confidence
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can change to another model if needed
            messages=[
                {"role": "system", "content": "You are a university course recommendation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating recommendation: {str(e)}"

def main():
    # Initialize session state
    if "user_responses" not in st.session_state:
        st.session_state.user_responses = {}
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "questionnaire"
    
    if "model_prediction" not in st.session_state:
        st.session_state.model_prediction = None
        
    if "llm_recommendation" not in st.session_state:
        st.session_state.llm_recommendation = None
    
    # Questionnaire Page
    if st.session_state.current_page == "questionnaire":
        # Show title and description only on questionnaire page
        st.title("🎓 University Course Recommendation System")
        st.write("Answer the questions below to get your personalized course recommendations")
        
        # Create a form for all questions
        with st.form("questionnaire_form"):
            for question in QUESTIONS:
                st.write(f"### {question['text']}")
                
                if question.get("multiple", False):
                    # For multiple selection questions
                    selected_options = st.multiselect(
                        "Select options (max 2):",
                        question["options"],
                        key=f"q{question['id']}_multi",
                        default=[],  # Ensure it starts empty
                        max_selections=question.get("max_selections", 2)  # Enforce max selections
                    )
                    
                    # If more than max selections are made, keep only the first two
                    if len(selected_options) > question.get("max_selections", 2):
                        selected_options = selected_options[:question.get("max_selections", 2)]
                        st.warning(f"Only the first {question.get('max_selections', 2)} selections will be considered.")
                    
                    st.session_state.user_responses[question["id"]] = selected_options
                else:
                    # For single selection questions
                    selected_option = st.selectbox(
                        "Select one option:",
                        question["options"],
                        key=f"q{question['id']}_single"
                    )
                    st.session_state.user_responses[question["id"]] = selected_option
                
                st.markdown("---")
            
            # Submit button
            submitted = st.form_submit_button("Get My Recommendations", use_container_width=True)
        
        if submitted:
            # Show loading spinner while processing
            with st.spinner('🔄 Analyzing your responses...'):
                # Process for ML model
                ml_form_data = {}
                
                for question in QUESTIONS:
                    if question.get("llm_only", False):
                        continue  # Skip questions that are only for LLM
                    
                    if "model_field" in question and "mapping" in question:
                        user_answer = st.session_state.user_responses.get(question["id"])
                        
                        if question.get("multiple", False) and isinstance(user_answer, list):
                            # For multiple selection, use the first one for the model (simplified)
                            if user_answer:
                                mapped_value = question["mapping"].get(user_answer[0])
                                if mapped_value:
                                    ml_form_data[question["model_field"]] = mapped_value
                        else:
                            # For single selection
                            if user_answer:
                                mapped_value = question["mapping"].get(user_answer)
                                if mapped_value:
                                    ml_form_data[question["model_field"]] = mapped_value
                
                # Process model prediction
                try:
                    processed_data = preprocess_input(ml_form_data)
                    prediction = model.predict(processed_data)
                    proba = model.predict_proba(processed_data)
                    course = label_encoder.inverse_transform(prediction)[0]
                    confidence = np.max(proba) * 100
                    
                    # Store predictions
                    st.session_state.model_prediction = {
                        "course": course,
                        "confidence": confidence,
                        "probabilities": {
                            label_encoder.classes_[i]: (proba[0][i] * 100).round(1) 
                            for i in range(len(label_encoder.classes_))
                        }
                    }
                    
                    # Generate LLM recommendation
                    llm_recommendation = generate_llm_response(
                        st.session_state.user_responses,
                        course,
                        round(confidence, 1)
                    )
                    
                    st.session_state.llm_recommendation = llm_recommendation
                    
                    # Move to results page
                    st.session_state.current_page = "results"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
    
    # Results Page
    elif st.session_state.current_page == "results":
        st.title("🎓 University Course Recommendation System")
        # Show balloons animation when results are displayed
        st.balloons()
        
        # Main container for recommendations with new header
        st.header("🌟 Your Perfect Course Match!")
        
        if st.session_state.llm_recommendation:
            # Extract top 2 recommendations using simple parsing
            recommendations = st.session_state.llm_recommendation.split("Top 3 Recommended Courses")[1].split("\n")[1:7]
            top_courses = [r.strip().split(":")[0].replace(".", "").strip() for r in recommendations if r.strip() and ":" in r][:2]
            
            # Create two columns for the recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"🥇 **First Choice**\n\n{top_courses[0]}")
                
            with col2:
                st.info(f"🥈 **Second Choice**\n\n{top_courses[1]}")
            
            # Student Profile Summary
            with st.expander("👤 View Your Profile Summary", expanded=True):
                profile_text = st.session_state.llm_recommendation.split("Student Profile Overview")[1].split("Top 3 Recommended Courses")[0]
                st.write(profile_text.strip())
            
            # Detailed Analysis
            with st.expander("📊 ML Model's Prediction", expanded=False):
                # ML Model Prediction
                if st.session_state.model_prediction:
                    st.subheader("🤖 ML Model Suggestion")
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.metric("Course", st.session_state.model_prediction["course"])
                    with mcol2:
                        st.metric("Confidence", f"{st.session_state.model_prediction['confidence']:.1f}%")
                    
                    # Show top 5 probabilities
                    proba_df = pd.DataFrame({
                        'Course': list(st.session_state.model_prediction["probabilities"].keys()),
                        'Probability (%)': list(st.session_state.model_prediction["probabilities"].values())
                    }).sort_values('Probability (%)', ascending=False).head()
                    
                    st.dataframe(proba_df, use_container_width=True)
                
                # LLM Analysis
                st.subheader("🧠 Expert Analysis")
                analysis = st.session_state.llm_recommendation.split("Comments on the ML Model's Recommendation")[1].split("Additional Advice")[0]
                st.write(analysis.strip())
            
            # Additional Advice
            with st.expander("💡 Personal Advice", expanded=False):
                advice = st.session_state.llm_recommendation.split("Additional Advice for the Student")[1]
                st.write(advice.strip())
            
            # Centered Start Over button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Start New Assessment", use_container_width=True):
                    st.session_state.user_responses = {}
                    st.session_state.model_prediction = None
                    st.session_state.llm_recommendation = None
                    st.session_state.current_page = "questionnaire"
                    st.rerun()

if __name__ == "__main__":
    main()
