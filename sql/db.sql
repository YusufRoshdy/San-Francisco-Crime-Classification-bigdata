-- Connect to the desired database (Replace 'your_database_name' with the actual database name)
\c project

-- start the transaction block
START TRANSACTION;


-- Drop the table if it exists
DROP TABLE IF EXISTS crime_data;

-- Create the table
CREATE TABLE crime_data (
    id SERIAL PRIMARY KEY,
    dates TIMESTAMP,
    category VARCHAR(50),
    descript VARCHAR(255),
    day_of_week VARCHAR(20),
    pd_district VARCHAR(50),
    resolution VARCHAR(50),
    address VARCHAR(255),
    x DOUBLE PRECISION,
    y DOUBLE PRECISION
);

-- Add any additional constraints if needed

-- Load data from the CSV file (Replace 'path/to/your/file.csv' with the actual file path)
COPY crime_data (dates, category, descript, day_of_week, pd_district, resolution, address, x, y)
FROM '/root/San-Francisco-Crime-Classification-bigdata/data/train.csv' DELIMITER ',' CSV HEADER;

-- Perform any additional CRUD operations if needed

-- Commit the transaction block
COMMIT;
