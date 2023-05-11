DROP DATABASE IF EXISTS projectdb CASCADE;

CREATE DATABASE projectdb;
USE projectdb;

SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;

-- Create tables

-- crime_data table
CREATE EXTERNAL TABLE crime_data STORED AS AVRO LOCATION '/project/crime_data' TBLPROPERTIES ('avro.schema.url'='/project/avsc/crime_data.avsc');

-- 
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


INSERT OVERWRITE LOCAL DIRECTORY '/root/q2'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT *
FROM crime_data
WHERE category = 'WARRANTS' AND day_of_week = 'Monday';

