/* Welcome to the SQL mini project. You will carry out this project partly in
the PHPMyAdmin interface, and partly in Jupyter via a Python connection.

This is Tier 2 of the case study, which means that there'll be less guidance for you about how to setup
your local SQLite connection in PART 2 of the case study. This will make the case study more challenging for you: 
you might need to do some digging, aand revise the Working with Relational Databases in Python chapter in the previous resource.

Otherwise, the questions in the case study are exactly the same as with Tier 1. 

PART 1: PHPMyAdmin
You will complete questions 1-9 below in the PHPMyAdmin interface. 
Log in by pasting the following URL into your browser, and
using the following Username and Password:

URL: https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

In this case study, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */


/* QUESTIONS 
/* Q1: Some of the facilities charge a fee to members, but some do not.
Write a SQL query to produce a list of the names of the facilities that do. */

SELECT name
FROM Facilities
WHERE membercost != 0;


/* Q2: How many facilities do not charge a fee to members? */

SELECT COUNT(name)
FROM Facilities
WHERE membercost = 0;

So 4 facilities are available to members for free.

/* Q3: Write an SQL query to show a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost.
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

SELECT facid, name, membercost, monthlymaintenance
FROM Facilities
WHERE membercost != 0
AND membercost < (monthlymaintenance * 0.2);


/* Q4: Write an SQL query to retrieve the details of facilities with ID 1 and 5.
Try writing the query without using the OR operator. */

SELECT *
FROM Facilities
WHERE facid IN (1,5)



/* Q5: Produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100. Return the name and monthly maintenance of the facilities
in question. */

SELECT name, monthlymaintenance,
CASE WHEN monthlymaintenance < 100 THEN 'cheap'	
ELSE 'expensive' END AS alias
FROM Facilities


/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Try not to use the LIMIT clause for your solution. */

SELECT M.firstname, M.surname
FROM Members as M
WHERE M.joindate = (SELECT MAX(joindate)
                    FROM Members)


/* Q7: Produce a list of all members who have used a tennis court.
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

SELECT concat (M.firstname,' ', M.surname) AS Member_Name, F.name AS Facility
FROM Facilities as F 
LEFT JOIN Bookings as B
ON F.facid = B.facid
LEFT JOIN Members as M
ON B.memid = M.memid
WHERE F.name LIKE 'Tenn%'
ORDER BY Member_Name


/* Q8: Produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30. Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

SELECT concat (M.firstname,' ',M.surname) as Member_Name, F.name as Facility,
CASE WHEN M.memid = 0 THEN B.slots * F.guestcost
ELSE B.slots * F.membercost END AS COST
FROM Bookings as B
INNER JOIN Facilities as F 
ON F.facid = B.facid
INNER JOIN Members as M 
ON M.memid = B.memid
WHERE B.starttime LIKE '2012-09-14%'
AND CASE WHEN M.memid = 0 THEN B.slots * F.guestcost
ELSE B.slots * F.membercost END > 30
ORDER BY COST DESC


/* Q9: This time, produce the same result as in Q8, but using a subquery. */
SELECT *
FROM (
    SELECT concat (M.firstname,' ',M.surname) as Member_Name, F.name as Facility,
	CASE WHEN M.memid = 0 THEN B.slots * F.guestcost
	ELSE B.slots * F.membercost END AS COST
	FROM Bookings as B
	LEFT JOIN Facilities as F 
	ON F.facid = B.facid
	AND B.starttime LIKE '2012-09-14%'
	LEFT JOIN Members as M 
	ON M.memid = B.memid) AS subquery
WHERE subquery.cost > 30
ORDER BY subquery.cost DESC

/* PART 2: SQLite

Export the country club data from PHPMyAdmin, and connect to a local SQLite instance from Jupyter notebook 
for the following questions.  

QUESTIONS:
/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

print(pd.read_sql_query("select name, \
                                sum(case when memid = 0 then slots * guestcost else slots * membercost end) as revenue \
                                from Bookings \
                                left join Facilities \
                                using(facid) \
                                group by name \
                                having revenue < 1000;", engine))




/* Q11: Produce a report of members and who recommended them in alphabetic surname,firstname order */

SELECT concat (M1.surname,' ',M2.firstname) as members, concat (M2.surname,' ', M2.firstname) as Recommendation
FROM Members AS M1
INNER JOIN Members as M2
ON M1.recommendedby = M2.memid
WHERE M1.memid > 0 AND M2.memid >0
ORDER BY Recommendation

print(pd.read_sql_query("select m1.surname||' '||m1.firstname as members, \
                                m2.surname||' '||m2.firstname as recommendation\
                                from Members as m1 \
                                inner join Members as m2 \
                                on m1.recommendedby = m2.memid \
                                where m1.memid > 0 and m2.memid>0 \
                                order by recommendation;", engine))


/* Q12: Find the facilities with their usage by member, but not guests */
SELECT concat (M.firstname,' ',M.surname) as Member, F.name as Facility, count(concat (M.firstname,' ',M.surname))
FROM Members as M
LEFT JOIN Bookings as B
on M.memid = B.memid
Left JOIN Facilities as F
on B.facid = F.facid
WHERE M.memid > 0 and F.name IS NOT NULL
GROUP BY F.name, member

print(pd.read_sql_query("select m.firstname||' '||m.surname as member, \
                                f.name as Facility, count(m.memid) as uses\
                                from Members as m \
                                inner join Bookings as b \
                                on b.memid = m.memid \
                                inner join Facilities as f \
                                on f.facid = b.facid \
                                where m.memid > 0\
                                and f.name is not null\
                                group by f.name, member;", engine))


/* Q13: Find the facilities usage by month, but not guests */

print(pd.read_sql_query("select strftime('%m', b.starttime) as Month, \
                                f.name as Facility, count(f.facid) as Uses \
                                from Members as m \
                                inner join Bookings as b\
                                on m.memid = b.memid \
                                inner join Facilities as f\
                                on b.facid = f. facid \
                                where m.memid > 0 \
                                AND f.name is not null \
                                group by Month, Facility;", engine)