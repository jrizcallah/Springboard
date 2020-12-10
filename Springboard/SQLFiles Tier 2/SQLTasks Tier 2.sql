/* Welcome to the SQL mini project. You will carry out this project partly in
the PHPMyAdmin interface, and partly in Jupyter via a Python connection.

This is Tier 2 of the case study, which means that there'll be less guidance for you about how to setup
your local SQLite connection in PART 2 of the case study. This will make the case study more challenging for you:
you might need to do some digging, and revise the Working with Relational Databases in Python chapter in the previous resource.

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

SELECT Name
FROM Facilities
WHERE membercost > 0

/* Q2: How many facilities do not charge a fee to members? */

SELECT COUNT(*)
FROM Facilities
WHERE membercost = 0


/* Q3: Write an SQL query to show a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost.
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

SELECT facid, name, membercost, monthlymaintenance
FROM Facilities
WHERE membercost < 0.2 * monthlymaintenance
AND membercost >0


/* Q4: Write an SQL query to retrieve the details of facilities with ID 1 and 5.
Try writing the query without using the OR operator. */

SELECT *
FROM Facilities
WHERE facid
IN ( 1, 5 )


/* Q5: Produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100. Return the name and monthly maintenance of the facilities
in question. */

SELECT name, monthlymaintenance,
CASE WHEN monthlymaintenance >= 100 THEN 'expensive'
WHEN monthlymaintenance < 100 THEN 'cheap' END as cost
FROM Facilities


/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Try not to use the LIMIT clause for your solution. */

SELECT firstname, surname
FROM Members
WHERE joindate = (SELECT MAX(joindate)
		   FROM Members)


/* Q7: Produce a list of all members who have used a tennis court.
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

SELECT
	DISTINCT CONCAT(m.firstname, " ", m.surname) AS fullname,
	t.name AS court_name,
	COUNT(b.starttime) AS times_used
FROM
	Members as M
INNER JOIN
	Bookings as b
ON
	b.memid = m.memid
INNER JOIN
	(SELECT name, faced
	FROM Facilities
	WHERE name LIKE 'Tennis%') AS t
ON
	b.facid = t.facid
GROUP BY
	fullname
ORDER BY
	m.surname


/* Q8: Produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30. Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

/*For some reason, this online thing will not let me use CTE. Online, it says that MySQL prior to version 8.0 does not support WITH statements. I don't know if that is what is causing it, but I could not get this to work without that the subquery. But I got it done. */

SELECT facility_name, member_name, SUM( booking_cost ) AS total_cost, starttime, bookid
FROM 	(
	SELECT
		f.name AS facility_name,
		CONCAT( m.firstname, " ", m.surname ) AS member_name,
		CASE WHEN b.memid =0
			THEN f.guestcost
		WHEN b.memid !=0
			THEN f.membercost END AS booking_cost,
		b.starttime,
		b.bookid
	FROM Facilities AS f
	INNER JOIN Bookings AS b
	ON b.facid = f.facid
	INNER JOIN Members AS m
	ON b.memid = m.memid
	WHERE b.starttime LIKE '2012-09-14%'
	) AS temp_table
GROUP BY bookid
HAVING total_cost >30
ORDER BY total_cost DESC


/* Q9: This time, produce the same result as in Q8, but using a subquery. */

/* I already did that, sorry. */


/* PART 2: SQLite

Export the country club data from PHPMyAdmin, and connect to a local SQLite instance from Jupyter notebook
for the following questions.

QUESTIONS:
/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

WITH subtable AS (
	SELECT name,
	SUM(membercost) AS total_member,
	SUM(guestcost) AS total_guest
	FROM Facilities as f
	INNER JOIN Bookings as b
	ON b.facid = f.facid
	GROUP BY f.facid)
SELECT name, (total_member+total_guest) AS total_revenue
FROM subtable
WHERE total_revenue > 1000
ORDER BY total_revenue DESC

/* Q11: Produce a report of members and who recommended them in alphabetic surname,firstname order */

SELECT 	m1.memid,
	m1.firstname,
	m1.surname,
	m1.recommendedby,
	2.firstname AS rec_first,
	m2.surname AS rec_sur
FROM
	Members AS m1
INNER JOIN
	(
	SELECT 	firstname,
		surname,
		memid
	FROM Members) AS m2
ON m2.memid = m1.recommendedby
ORDER BY m1.surname, m1.firstname


/* Q12: Find the facilities with their usage by member, but not guests */

WITH subtable AS (
	SELECT *
	FROM Facilities as f
	INNER JOIN Bookings as b
	ON f.facid = b.facid
	WHERE b.memid != 0)
SELECT
	COUNT(DISTINCT bookid) AS times_used,
	name
FROM
	subtable
GROUP BY
	facid


/* Q13: Find the facilities usage by month, but not guests */
/* I could not figure out how to do this in SQL alone, so I brought the data into python and used pandas to count the number of times each facility was used each month. */

SELECT f.name,concat(m.firstname,' ',m.surname) as Member,
count(f.name) as bookings,

sum(case when month(starttime) = 1 then 1 else 0 end) as Jan,
sum(case when month(starttime) = 2 then 1 else 0 end) as Feb,
sum(case when month(starttime) = 3 then 1 else 0 end) as Mar,
sum(case when month(starttime) = 4 then 1 else 0 end) as Apr,
sum(case when month(starttime) = 5 then 1 else 0 end) as May,
sum(case when month(starttime) = 6 then 1 else 0 end) as Jun,
sum(case when month(starttime) = 7 then 1 else 0 end) as Jul,
sum(case when month(starttime) = 8 then 1 else 0 end) as Aug,
sum(case when month(starttime) = 9 then 1 else 0 end) as Sep,
sum(case when month(starttime) = 10 then 1 else 0 end) as Oct,
sum(case when month(starttime) = 11 then 1 else 0 end) as Nov,
sum(case when month(starttime) = 12 then 1 else 0 end) as Decm

FROM Members m
inner join Bookings bk on bk.memid = m.memid
inner join Facilities f on f.facid = bk.facid
where m.memid>0
and year(starttime) = 2012

group by f.name,concat(m.firstname,' ',m.surname)
order by f.name,m.surname,m.firstname
