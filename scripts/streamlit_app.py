import streamlit as st
import pandas as pd

data = pd.read_csv("data/train.csv")

st.markdown('---')
st.title("Big Data Project **2023**")
st.markdown("""<style>body {
    background-color: #eee;
}

.fullScreenFrame > div {
    display: flex;
    justify-content: center;
}
</style>""", unsafe_allow_html=True)

st.image("https://i2.wp.com/hr-gazette.com/wp-content/uploads/2018/10/bigstock-Recruitment-Concept-Idea-Of-C-250362193.jpg", caption = "San Francisco Crime Classification", width=400)

#st.markdown("<p style='text-align: center; color: grey;'>Employees and Departments</p>", unsafe_allow_html=True)


# Sample Data
st.markdown('---')
st.header('Descriptive Data Analysis')
st.subheader('Data Characteristics')
st.write(data)
st.markdown('`data` table')
st.write(data.describe())
st.markdown('`depts` table')
st.write(data.describe())

st.subheader('Some samples from the data')
st.markdown('`emps` table')
st.write(data.head(5))
st.markdown("`depts` table")
st.write(data.head(5))

# st.markdown('---')
# st.header("Exploratory Data Analysis")
# st.subheader('Q1')
# st.text('The distribution of employees in departments')
# st.bar_chart(q1)

# st.subheader('Q2')
# st.text('The average salary in departments')
# st.table(q2)
# st.line_chart(q2['sal_avg'], width=400)

# st.markdown('---')
# st.header('Predictive Data Analytics')
# st.subheader('ML Model')
# st.markdown('1. Linear Regression Model')
# st.markdown('Settings of the model')
# st.table(pd.DataFrame([['setting1', 1.0], ['setting2', 0.01], ['....','....']], columns = ['setting', 'value']))

# st.markdown('2. SVC Regressor')
# st.markdown('Settings of the model')
# st.table(pd.DataFrame([['setting1', 1.0], ['setting2', 'linear'], ['....','....']], columns = ['setting', 'value']))

# st.subheader('Results')
# st.text('Here you can display metrics you are using and values you got')
# st.table(pd.DataFrame([]))
# st.markdown('<center>Results table</center>', unsafe_allow_html = True)
# st.subheader('Training vs. Error chart')
# st.write("matplotlib or altair chart")
# st.subheader('Prediction')
# st.text('Given a sample, predict its value and display results in a table.')
# st.text('Here you can use input elements but it is not mandatory')
