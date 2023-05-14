import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import  Point

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

st.image("https://www.kaggleusercontent.com/kf/123989573/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K2ZBnkAkLRLB8aP28TTrjw.VJpiJ1s3aw_YWMY_r4NtukBEelhsA4BZovjPdgE7yA0SzGIn-H76nATDGlHmuAab6HkjOsowDcQa1i68Ehfqu7OtM1r-Zb2Ey0JBsJhoLzWtscbFi3xljOpC9JAWUXIb8LsDlv8HekaFvBazQq5I1sQu0Lq_L1Zke8r9B0PQ7mXDxEO7fzOjF8YvdwQMNJgBmOVIPcaxwoL6AjJCH0-WxUSwmyAxTwhRWJHa99lVbypOQXTjPE3h5cS06u77V0K5gWUPgFyUqgIEY-h3UTL-EMs6OG9O-uMNlH5cNM7Yl1AOU7Ou17s06LAmiN_Mn85Dm0RfygM5LLq79YR4O6T_MNSNjdcQreq3ctTRDoW11vFAaSXvEv037rBJdZWU4kLw2khxqGBl7d_71uInMKkmhkdxXn2XkJ6eJvAuUBDpZo8YZ1N4JaNHQhSwOC72G8gvsTwH8i96Q1lEnaViyUp8P1e66sdjTClVzp8gEeApRUwVVD5AmBz7koLN7c42ZxqdbggE9TCm1Ip9IV1ownNDA653mp-Jyegbb9YCKaEvWOM3PGJ330pe6QLSYsFBSWv7pbJwzWMCcDlQtp5KTTCaLxCjlvoj3ZdFOtdsWydH2HLKNPDK7b6tkB4IH7cAc8IjqG7guegxbQ6GnS83Bj-dQQ.F86tqCv9v5zeCo2p9c8_qw/__results___files/__results___44_0.png", caption = "San Francisco Crime Classification", width=400)

#st.markdown("<p style='text-align: center; color: grey;'>Employees and Departments</p>", unsafe_allow_html=True)


# Sample Data
st.markdown('---')
st.header('Descriptive Data Analysis')
st.subheader('Data Samples')
st.write(data)

# First query
st.markdown('---')
st.header('First Query')
st.subheader('total number of crimes in each police district and then orders these districts in descending order by this count')
st.write(data.groupby('PdDistrict').size().sort_values(ascending=False))
# make the bar chart
st.bar_chart(data.groupby('PdDistrict').size().sort_values(ascending=False))

# Second query
st.markdown('---')
st.header('Second Query')
st.subheader('Crimes per day of the week')
st.write(data.groupby('DayOfWeek').size().sort_values(ascending=False))
# make the bar chart
st.bar_chart(data.groupby('DayOfWeek').size().sort_values(ascending=False))

# Third query
st.markdown('---')
st.header('Third Query')
# crime category counts
st.subheader('Crime category counts')
st.write(data.groupby('Category').size().sort_values(ascending=False))
# make the bar chart
st.bar_chart(data.groupby('Category').size().sort_values(ascending=False))

# Fourth query
# Districts with highest crime count
st.markdown('---')
st.header('Fourth Query')
st.subheader('Districts with highest crime count')
st.write(data.groupby('PdDistrict').size().sort_values(ascending=False))
# make the bar chart
st.bar_chart(data.groupby('PdDistrict').size().sort_values(ascending=False))

# Fifth query
# resolution counts
st.markdown('---')
st.header('Fifth Query')
st.subheader('Resolution counts')
st.write(data.groupby('Resolution').size().sort_values(ascending=False))
# make the bar chart
st.bar_chart(data.groupby('Resolution').size().sort_values(ascending=False))

# Sixth query
# scatter plot of crime location
st.markdown('---')
st.header('Sixth Query')
st.subheader('Scatter plot of crime location')
# using the x and y coordinates
import matplotlib.pyplot as plt
import seaborn as sns
with plt.style.context('fivethirtyeight'):
    fig, ax = plt.subplots(1, 1, figsize=(19, 19))
    sns.scatterplot(data=data.iloc[:100_000], x='X', y='Y', alpha=0.6, palette='mako', hue='Category', size='Category')
    plt.title("Scatterplot of category crimes", fontsize=22)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')

st.pyplot(fig)


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