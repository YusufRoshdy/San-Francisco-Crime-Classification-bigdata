#!/bin/bash
# Creat hive db and execute some queries
hive -f sql/db.hql


# Writing the query results to csv
# Q1
echo "pd_district,crime_count" > output/q1.csv
cat /root/q1/* >> output/q1.csv

# Q2
echo "Category,Count" > output/q2.csv
cat /root/q2/* >> output/q2.csv

# Q3
echo "DayOfWeek,Count" > output/q3.csv
cat /root/q3/* >> output/q3.csv

# Q4
echo "Resolution,Count" > output/q4.csv
cat /root/q4/* >> output/q4.csv

# Q5
echo "Address,Count" > output/q5.csv
cat /root/q5/* >> output/q5.csv
