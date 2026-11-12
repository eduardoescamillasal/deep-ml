-- your query
SELECT  MIN(id) as id, email
FROM person
GROUP BY email
order by id;
