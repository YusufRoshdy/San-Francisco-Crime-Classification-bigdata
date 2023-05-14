DROP DATABASE IF EXISTS projectdb CASCADE;

CREATE DATABASE projectdb;
USE projectdb;

SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;

-- Create tables

-- crime_data table
CREATE EXTERNAL TABLE crime_data STORED AS AVRO LOCATION '/project/crime_data' TBLPROPERTIES ('avro.schema.url'='/project/avsc/crime_data.avsc');

-- Queries
-- The results are saved to a local directory in a comma-separated format.

-- Query 1:
-- The total number of crimes in each police district ordered in descending order by count.
INSERT OVERWRITE LOCAL DIRECTORY '/root/q1'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT pd_district, crime_count
FROM (
    SELECT pd_district, COUNT(*) AS crime_count
    FROM crime_data
    GROUP BY pd_district
) tmp
ORDER BY crime_count DESC;

-- Query 2:
-- Number of crimes in each category:
INSERT OVERWRITE LOCAL DIRECTORY '/root/q2'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT category, COUNT(*) as Count
FROM crime_data
GROUP BY category
ORDER BY Count DESC;


-- Query 3:
-- Number of crimes per day of the week:
INSERT OVERWRITE LOCAL DIRECTORY '/root/q3'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT day_of_week, COUNT(*) as Count
FROM crime_data
GROUP BY day_of_week
ORDER BY Count DESC;


-- Query 4:
-- Most common resolution types:
INSERT OVERWRITE LOCAL DIRECTORY '/root/q4'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT resolution, COUNT(*) as Count
FROM crime_data
GROUP BY resolution
ORDER BY Count DESC;


-- Query 5:
-- Most common crime locations:
INSERT OVERWRITE LOCAL DIRECTORY '/root/q5'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT address, COUNT(*) as Count
FROM crime_data
GROUP BY address
ORDER BY Count DESC
LIMIT 15;


