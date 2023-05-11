#!/bin/bash
psql -U postgres -c 'DROP DATABASE IF EXISTS project;'

psql -U postgres -c 'CREATE DATABASE project;'

psql -U postgres -d project -f sql/db.sql

sqoop import-all-tables \
    --connect jdbc:postgresql://localhost/project \
    --username postgres \
    --warehouse-dir /project \
    --as-avrodatafile