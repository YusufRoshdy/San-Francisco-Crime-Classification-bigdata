import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

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
    "output/images/SF_crime_areas.png",
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

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

fig, ax = plt.subplots(1, 1, figsize=(19, 6))
sns.histplot(x='Category', data=data, palette='Paired', ax=ax, color = 'red')
ax.tick_params(axis='x', rotation=90)
plt.title("Category crimes", fontsize=22)

st.pyplot(fig)

# interpolation by day of week
week = data['DayOfWeek'].value_counts()

fig, ax = plt.subplots(figsize=(12,8))
week.plot(kind="bar", table=True,  color='g', ax=ax)
plt.xticks([])
plt.xlabel('days and weeks',fontsize=15,labelpad=30)
plt.ylabel('Number of criminals',fontsize=25)
plt.title('Interpolation by day of the week',fontsize=25)

st.pyplot(fig)

st.markdown('''
### Description of the plot
This plot shows the frequency of crimes for each day of the week in our dataset. The y-axis represents the number of crimes, while the x-axis represents the days of the week. The plot is a bar chart, with each bar corresponding to a day of the week.
''')

st.markdown('''
### Descriptive statistics
''')

st.write(week.describe())


# Estimate the number of crimes by district
dist = data["PdDistrict"].value_counts()

fig, ax = plt.subplots(figsize=(12,8))
dist.plot(kind="bar", table=True, color='g', ax=ax)
plt.xticks([])
plt.xlabel('area',fontsize=15,labelpad=30)
plt.ylabel('Number of criminals',fontsize=25)
plt.title('Precipitation by regions',fontsize=25)

st.pyplot(fig)

st.markdown('''
### Description of the plot
This plot shows the frequency of crimes for each district in our dataset. The y-axis represents the number of crimes, while the x-axis represents the different districts. The plot is a bar chart, with each bar corresponding to a district.
''')

st.markdown('''
### Descriptive statistics
''')

st.write(dist.describe())

# Visual breakdown of categories of crimes by their number TOP-10 categories:
kind = data['Category'].value_counts()

fig, ax = plt.subplots(figsize=(20,12))
kind.plot(kind="barh", color='g', ax=ax)

plt.ylabel('Type',fontsize=15)
plt.xlabel('Number of criminals',fontsize=15)
plt.title('Videos of criminals',fontsize=25)

st.pyplot(fig)

st.markdown('''
### Description of the plot
This horizontal bar plot shows the frequency of different categories of crimes in our dataset. 
The x-axis represents the number of crimes, while the y-axis represents different crime categories.
Each bar corresponds to a different category of crime.
''')


# Let's see how many crimes in different categories occur in different periods of time:
data['Dates'] = pd.to_datetime(data['Dates'])
data['Hour'] = data['Dates'].dt.hour
data_new = data.groupby(['Hour', 'Category'],
                            as_index=False).count().iloc[:, :4]
data_new.rename(columns={'Dates': 'Incidents'}, inplace=True)

unique_cats = list(data_new.groupby(['Category']).sum().sort_values(['Incidents']).index)

for i in range(0, len(unique_cats), 5):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 10))
    ax = sns.lineplot(x='Hour', y='Incidents', data=data_new[data_new['Category'].isin(unique_cats[i: i+5])], hue='Category')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6)
    plt.suptitle('Number of crimes in an interval of time')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    st.pyplot(fig)

    st.markdown(f'''
    ### Description of the plot for crime categories {unique_cats[i]} to {unique_cats[min(i+4, len(unique_cats)-1)]}
    This line plot shows the number of crimes committed at different hours for the crime categories {unique_cats[i]} to {unique_cats[min(i+4, len(unique_cats)-1)]}.
    The x-axis represents the hour of the day, while the y-axis represents the number of incidents.
    Each line corresponds to a different category of crime.
    ''')

# load images from output/images/ the crime categories
image_names = ['ASSAULT', 'LARCENY-THEFT', 'DRUG-NARCOTIC', 'NON-CRIMINAL', 'OTHER OFFENSES']
from PIL import Image

for image_name in image_names:
    image = Image.open(f'output/images/{image_name}.png')
    st.image(image, caption=image_name, use_column_width=True)

st.markdown('''
### Description of the images
These images show the number of crimes committed at different hours for the crime categories ASSAULT, LARCENY-THEFT, DRUG-NARCOTIC, NON-CRIMINAL, and OTHER OFFENSES.
Each image corresponds to a different category of crime.
''')





st.markdown('---')
st.header('Predictive Data Analytics')
st.subheader('ML Model')
st.markdown('##### 1. Linear Regression Model')
st.markdown('Baseline model')
st.table(pd.DataFrame([['setting1', 1.0], ['setting2', 0.01], ['....','....']], columns = ['setting', 'value']))

# fine-tuned model
st.markdown('Fine-tuned model')
st.table(pd.DataFrame([['setting1', 1.0], ['setting2', 0.01], ['....','....']], columns = ['setting', 'value']))

# model evaluation
st.markdown('Model evaluation')

# model interpretation
st.markdown('Model interpretation')



st.markdown('#####2. Decision Tree Model')
st.markdown('Baseline model')
st.table(pd.DataFrame([['setting1', 1.0], ['setting2', 'linear'], ['....','....']], columns = ['setting', 'value']))

# fine-tuned model
st.markdown('Fine-tuned model')
st.table(pd.DataFrame([['setting1', 1.0], ['setting2', 'linear'], ['....','....']], columns = ['setting', 'value']))

# model evaluation
st.markdown('Model evaluation')

# model interpretation
st.markdown('Model interpretation')

st.markdown('#####3. Random Forest Model')
st.markdown('Baseline model')
st.table(pd.DataFrame([['setting1', 1.0], ['setting2', 'linear'], ['....','....']], columns = ['setting', 'value']))

# fine-tuned model
st.markdown('Fine-tuned model')
st.table(pd.DataFrame([['setting1', 1.0], ['setting2', 'linear'], ['....','....']], columns = ['setting', 'value']))

# model evaluation
st.markdown('Model evaluation')


st.subheader('Results')
st.text('Here you can display metrics you are using and values you got')
st.table(pd.DataFrame([]))
st.markdown('<center>Results table</center>', unsafe_allow_html = True)
st.subheader('Training vs. Error chart')
st.write("matplotlib or altair chart")
st.subheader('Prediction')
st.text('Given a sample, predict its value and display results in a table.')
st.text('Here you can use input elements but it is not mandatory')
