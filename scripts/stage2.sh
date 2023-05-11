#!/bin/bash
# Creat hive db and execute some queries
hive -f sql/db.hql


# Writing the query results to csv
# Q1
echo "pd_district,crime_count" > output/q1.csv
cat /root/q1/* >> output/q1.csv

# Q2
echo "Dates,Category,Descript,DayOfWeek,PdDistrict,Resolution,Address,X,Y" > output/q2.csv
for file in /root/q2/*; do
  cut -d',' -f2- "$file" >> output/q2.csv
done
