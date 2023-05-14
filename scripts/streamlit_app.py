import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

data = pd.read_csv("data/train.csv")

st.markdown("---")
st.title("Big Data Project **2023**")
st.markdown(
    """<style>body {
    background-color: #eee;
}

.fullScreenFrame > div {
    display: flex;
    justify-content: center;
}
</style>""",
    unsafe_allow_html=True,
)

st.image(
    "https://www.kaggleusercontent.com/kf/123989573/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K2ZBnkAkLRLB8aP28TTrjw.VJpiJ1s3aw_YWMY_r4NtukBEelhsA4BZovjPdgE7yA0SzGIn-H76nATDGlHmuAab6HkjOsowDcQa1i68Ehfqu7OtM1r-Zb2Ey0JBsJhoLzWtscbFi3xljOpC9JAWUXIb8LsDlv8HekaFvBazQq5I1sQu0Lq_L1Zke8r9B0PQ7mXDxEO7fzOjF8YvdwQMNJgBmOVIPcaxwoL6AjJCH0-WxUSwmyAxTwhRWJHa99lVbypOQXTjPE3h5cS06u77V0K5gWUPgFyUqgIEY-h3UTL-EMs6OG9O-uMNlH5cNM7Yl1AOU7Ou17s06LAmiN_Mn85Dm0RfygM5LLq79YR4O6T_MNSNjdcQreq3ctTRDoW11vFAaSXvEv037rBJdZWU4kLw2khxqGBl7d_71uInMKkmhkdxXn2XkJ6eJvAuUBDpZo8YZ1N4JaNHQhSwOC72G8gvsTwH8i96Q1lEnaViyUp8P1e66sdjTClVzp8gEeApRUwVVD5AmBz7koLN7c42ZxqdbggE9TCm1Ip9IV1ownNDA653mp-Jyegbb9YCKaEvWOM3PGJ330pe6QLSYsFBSWv7pbJwzWMCcDlQtp5KTTCaLxCjlvoj3ZdFOtdsWydH2HLKNPDK7b6tkB4IH7cAc8IjqG7guegxbQ6GnS83Bj-dQQ.F86tqCv9v5zeCo2p9c8_qw/__results___files/__results___44_0.png",
    caption="San Francisco Crime Classification",
    width=400,
)

# st.markdown("<p style='text-align: center; color: grey;'>Employees and Departments</p>", unsafe_allow_html=True)


# Sample Data
st.markdown("---")
st.header("Descriptive Data Analysis")
st.subheader("Data Samples")
st.write(data)

# First query
st.markdown("---")
st.header("First Query")
st.subheader("Total number of crimes in each police district")

query = pd.read_csv("output/q1.csv")
st.write(query)
st.bar_chart(query, x="pd_district", y="crime_count")


# Second query
st.markdown("---")
st.header("Second Query")
st.subheader("Number of crimes in each category:")

query = pd.read_csv("output/q2.csv")
st.write(query)
st.bar_chart(query, x="Category", y="Count")


# Third query
st.markdown("---")
st.header("Third Query")
st.subheader("Number of crimes per day of the week:")

query = pd.read_csv("output/q3.csv")
st.write(query)
st.bar_chart(query, x="DayOfWeek", y="Count")

# Fourth query
st.markdown("---")
st.header("Fourth Query")
st.subheader("Most common resolution types:")

# Read the header
header = pd.read_csv("output/q4.csv", nrows=0).columns.tolist()

# Read the rest of the data
query = pd.read_csv(
    "output/q4.csv", header=None, skiprows=1, sep=",(?=\d)", engine="python"
)

# Assign the header to the data
query.columns = header

st.write(query)
st.bar_chart(query, x="Resolution", y="Count")

# Fifth query
st.markdown("---")
st.header("Fifth Query")
st.subheader("Most common crime locations:")

query = pd.read_csv("output/q5.csv")
st.write(query)
st.bar_chart(query, x="Address", y="Count")

# Sixth query
# scatter plot of crime location
st.markdown("---")
st.header("Sixth Query")
st.subheader("Scatter plot of crime location")
# using the x and y coordinates
import matplotlib.pyplot as plt
import seaborn as sns

with plt.style.context("fivethirtyeight"):
    fig, ax = plt.subplots(1, 1, figsize=(19, 19))
    sns.scatterplot(
        data=data.iloc[:100_000],
        x="X",
        y="Y",
        alpha=0.6,
        palette="mako",
        hue="Category",
        size="Category",
    )
    plt.title("Scatterplot of category crimes", fontsize=22)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")

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
