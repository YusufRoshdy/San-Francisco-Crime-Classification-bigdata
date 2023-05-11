DROP DATABASE IF EXISTS projectdb CASCADE;

CREATE DATABASE projectdb;
USE projectdb;

SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;

-- Create tables

-- crime_data table
CREATE EXTERNAL TABLE crime_data STORED AS AVRO LOCATION '/project/crime_data' TBLPROPERTIES ('avro.schema.url'='/project/avsc/crime_data.avsc');

-- Query 1:
-- This query calculates the total number of crimes in each police district and then orders these districts in descending order by this count.
-- The results are saved to a local directory in a comma-separated format.
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
-- This query selects all records from the crime_data table where the crime category is 'WARRANTS' and the crime occurred on a 'Monday'.
-- The results are saved to a local directory in a comma-separated format.
INSERT OVERWRITE LOCAL DIRECTORY '/root/q2'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT *
FROM crime_data
WHERE category = 'WARRANTS' AND day_of_week = 'Monday';

