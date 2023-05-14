-- find the rows with category WARRANTS
SELECT * FROM crime_data WHERE category = 'WARRANTS';

-- find the rows with category WARRANTS and day_of_week = 'Monday'
SELECT * FROM crime_data WHERE category = 'WARRANTS' AND day_of_week = 'Monday';

SELECT pd_district, COUNT(*) FROM crime_data GROUP BY pd_district ORDER BY COUNT(*) DESC;

-- find the number of crimes for each week
SELECT date_trunc('week', dates) AS week, COUNT(*) FROM crime_data GROUP BY week ORDER BY week;
